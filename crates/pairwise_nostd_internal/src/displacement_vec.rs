use crate::misc::View3DSpec;
use crate::spatial::{CartesianBlock, CellWidth};
use core::iter::IntoIterator;

/// <div class="warning">
///
/// We probably want to avoid exposing this type publicly. If we decide to
/// publicly expose it, we may want to re-draft the documentation for the
/// type and its methods (we need to disentangle what the type represents
/// and what is an implementation detail)
///
/// The language here could also be improved.
///
/// </div>
///
/// TODO: reframe this as a 3D index offset, that could theoretically apply
/// to any 2 3D arrays (in the context of CartesianBlock, it corresponds to
/// a spatial offset)
///
/// This type is only meaningful when you have 2 `CartesianBlock` references,
/// block_a and block_b; (they may reference the same block or 2 distinct
/// blocks). In this context, the type represents a displacement (mathematical)
/// vector, in units of indices, between a point in block_a and a point in
/// block_b (we sometimes call this a separation vector).
///
/// At a high-level, if you have a pair of points from block_a and block_b
/// separated by a displacement vector, then adding the displacement vector to
/// the position of the point in block_a gives the position of the point in
/// block_b. This description glosses over a few thorny details related to
/// coordinate systems, which we sort out down below.
///
/// # Coordinate Systems
///
/// Coordinate systems become relevant when block_a and block_b describe
/// different regions of space. 3D indices used to access local values or
/// weights normally use coordinate-systems defined locally to the block. We
/// can also consider a "global index space." In other words, imagine that
/// blocks a and b are concatenated with other blocks to make a single global
/// block.
///
/// Let's define the "global displacement vec" as describing the displacement
/// between 2 points in the "global index space." We define the
/// "local_displacement_vec" as the vector you can add to the position of the
/// from block_a, in block_a's local coordinate system, to get the position of
/// the point in block_b, in block_b's local coordinate system.
///
/// These `i`th component of these quantities are related by:
/// ```text
/// total_displacement_vec[i] =
///   local_displacement_vec[i] +
///   (block_b.start_idx_global_offset[i] - block_a.start_idx_global_offset[i])
/// ```
///
///
/// As you can see, when block_a and block_b reference the same block, there is
/// no difference between these quantities.
///
/// # Ideas for Improvement:
/// - can we come up with a better name than local_displacement_vec?
/// - maybe we introduce this idea of local-vs-global coordinate systems
///   elsewhere? In the documentation of [`CartesianBlock`] or maybe we add
///   some higher-level narrative docs (maybe at the module-level) to
///   introduce the concept?
#[derive(Clone)]
pub struct IdxOffset3D {
    local_offset_zyx: [isize; 3],
}

impl IdxOffset3D {
    /// computes the "local index displacement vector." The core documentation
    /// [for the IndexDisplacementVec type](IndexDisplacementVec) provides more
    /// context.
    pub fn local_displacement_vec(&self) -> &[isize; 3] {
        &self.local_offset_zyx
    }

    /// computes the "global index displacement vector." The core documentation
    /// [for the IndexDisplacementVec type](IndexDisplacementVec) provides more
    /// context.
    ///
    /// This method primarily exists for self-documenting purposes
    pub fn global_displacement_vec(
        &self,
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) -> [isize; 3] {
        let mut out = [0_isize; 3];
        for i in 0..3 {
            let off_a = block_a.start_idx_global_offset[i] as isize;
            let off_b = block_b.start_idx_global_offset[i] as isize;
            out[i] = (off_b - off_a) + self.local_offset_zyx[i];
        }
        out
    }

    /// calculates squared distance represented by the offset
    pub fn distance_squared(
        &self,
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
        cell_widths: &CellWidth,
    ) -> f64 {
        let d_index_units = self.global_displacement_vec(block_a, block_b);
        let mut out = 0.0;
        for i in 0..3 {
            let comp = (d_index_units[i] as f64) * cell_widths.widths_zyx[i];
            out += comp * comp;
        }
        out
    }
}

// define the logic for iterating over IdxOffset3D instances
// ---------------------------------------------------------
// the following probably belongs in its own file. We will want to expose
// something like this to help with the tiling pattern when people are using
// MPI (in practice, we may expose this logic with a separate construct and
// just reuse the shared logic)
//
// At the moment, this is implemented like an iterator. In practice, this isn't
// the appropriate abstraction. Instead, we probably want to make this
// "indexable" so we can efficiently partition out the work (to be distributed
// among "teams"). I think we instead want to produce something like a
// range object (i.e. that is indexable)
#[derive(Clone)]
struct Bounds {
    start_offsets_zyx: [isize; 3], // inclusive
    stop_offsets_zyx: [isize; 3],  // inclusive
}

/// this is used in the context of iterating over "'local' index
/// offsets".
///
/// `cur_offset` holds the internals of a IdxOffset3D. This
/// function returns the next IdxOffset3D ("next" in the sense
/// of the sequence or iteration order).
///
/// If `cur_offset` is equal to
/// ```text
/// [
///     bounds.stop_offsets_zyx[0] - 1,
///     bounds.stop_offsets_zyx[1] - 1,
///     bounds.stop_offsets_zyx[2] - 1,
/// ]
/// ```
/// then the returned value won't correspond to a valid output
/// IdxOffset3D.
///
/// # TODO
/// We should consider whether there is a performance benefit to avoiding
/// a copy and just mutating the cur_offset argument directly.
fn get_next_offset(mut offset_zyx: [isize; 3], bounds: &Bounds) -> [isize; 3] {
    offset_zyx[2] += 1;
    if offset_zyx[2] == bounds.stop_offsets_zyx[2] {
        offset_zyx[2] = bounds.start_offsets_zyx[2];
        offset_zyx[1] += 1;
        if offset_zyx[1] == bounds.stop_offsets_zyx[1] {
            offset_zyx[1] = bounds.start_offsets_zyx[1];
            offset_zyx[0] += 1;
        }
    }
    offset_zyx
}

/// for the moment, this won't know anything about the min/max separation
/// bins
///
/// # Note
/// At the moment, the iterator steps through IdxOffset3D instances. But, I
/// think that it would make more sense to attach an iterator to
/// View3DSpec and simply perform the transformation
pub struct Iter {
    bounds: Bounds,
    next_offset_zyx: [isize; 3],
}

impl Iterator for Iter {
    type Item = IdxOffset3D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bounds.stop_offsets_zyx == self.next_offset_zyx {
            None
        } else {
            let out = Some(IdxOffset3D {
                local_offset_zyx: self.next_offset_zyx,
            });
            self.next_offset_zyx = get_next_offset(self.next_offset_zyx, &self.bounds);
            out
        }
    }
}

/// the idea is for this to be an indexable "range" object that contains all
/// relevant [`IdxOffset3D`] instances
///
/// # Important
/// We are using "range" in the Python and Julia sense (i.e. the range
/// object is more like a collection, that is iterable, but isn't itself an
/// iterator). This contrasts with Rust's ranges, which are iterators. The
/// design of rust's ranges is widely regarded to be a mistake.
///
/// # ToDo
/// When documenting this have type, frame the description in terms of
/// "cross" statistics.
/// - we can enumerate all [`IdxOffset3D`] values
/// - It’s convenient to think of this list of [`IdxOffset3D`] instances as
///   a 3D array (there is a straight-forward mapping from index to
///   [`IdxOffset3D`]).
/// - At the corners of this 3D array, you have the largest offset
///   (only a small subset of pairs are separated by this amount)
/// - the smallest [`IdxOffset3D`] describe the offsets between the most
///   pairs of points.
/// - Auto-correlation is just a special case.
///   - For equal sized blocks, I think this is 1 less than half of the
///     blocks
pub struct IdxOffset3DSeq {
    // this is Lazy Sequence
    bounds: Bounds,
    mapping_props: View3DSpec,
    // Can we just get rid of this now that we have internal_offset? I
    // think so, but it also seems plausible that there could be a special
    // case where we need
    // -> this case would come up in a scenario where one of the blocks
    //    holds 0 elements and we are adjusting the bounds based on the
    //    minimum and maximum bin edges.
    // -> In that scenario can internal_offset be 0 for auto correlation?
    //    If so, then I think that means we can't tell the difference from
    //    cross-correlation.
    // -> I guess the followup question will be: does it actually matter?
    //    As I write it out, I don't think it does. I think the **only**
    //    reason we draw a distinction between auto and cross is for
    //    getting the starting index. And I don't think it will matter!
    is_auto: bool,
    // this is 0 for cross-operations and positive for auto-operations
    internal_offset: isize,
}

impl IdxOffset3DSeq {
    pub fn new_auto(block: &CartesianBlock) -> Result<Self, &'static str> {
        let stop_offsets_zyx = *block.idx_spec.shape();
        let bounds = Bounds {
            start_offsets_zyx: [
                0, // <- we never have a negative z in "auto" pairs
                -(stop_offsets_zyx[1] - 1),
                -(stop_offsets_zyx[2] - 1),
            ],
            stop_offsets_zyx,
        };

        let mapping_props = View3DSpec::from_shape_contiguous([
            (bounds.stop_offsets_zyx[0] - bounds.start_offsets_zyx[0]) as usize,
            (bounds.stop_offsets_zyx[1] - bounds.start_offsets_zyx[1]) as usize,
            (bounds.stop_offsets_zyx[2] - bounds.start_offsets_zyx[2]) as usize,
        ])?;

        let first_idx_offset = get_next_offset([0, 0, 0], &bounds);
        // convert the first 3D index offset to a 3D index
        let [iz, iy, ix] = [
            first_idx_offset[0] - bounds.start_offsets_zyx[0],
            first_idx_offset[1] - bounds.start_offsets_zyx[1],
            first_idx_offset[2] - bounds.start_offsets_zyx[2],
        ];
        // convert the 3D index to a 1D offset
        let internal_offset = mapping_props.map_idx3d_to_1d(iz, iy, ix);

        Ok(Self {
            bounds,
            mapping_props,
            is_auto: true,
            internal_offset,
        })
    }

    pub fn new_cross(
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) -> Result<Self, &'static str> {
        let is_bad = (block_b.start_idx_global_offset[0] < block_a.start_idx_global_offset[0])
            || ((block_b.start_idx_global_offset[0] == block_a.start_idx_global_offset[0])
                && (block_b.start_idx_global_offset[1] < block_a.start_idx_global_offset[1]))
            || ((block_b.start_idx_global_offset[0] == block_a.start_idx_global_offset[0])
                && (block_b.start_idx_global_offset[1] == block_a.start_idx_global_offset[1])
                && (block_b.start_idx_global_offset[2] <= block_a.start_idx_global_offset[2]));
        if is_bad {
            //  produce an empty iterator
            return Err("reverse the argument order OR try to use new_auto_iter");
        }
        // I **think** this using block_b's shape for the stop-offsets
        // and block_a's shape for inferring the start-offsets is the
        // correct thing to do...
        // -> this also makes sense on a deeper level
        // -> suppose we were executing this branch where block_a & block_b
        //    referred to the same block
        // -> that means we would end up iterating over 2N+1 entries where
        //    N is the number of entries we would encounter with
        //    new_auto_itere
        let stop_offsets_zyx = *block_b.idx_spec.shape();
        let start_offsets_zyx = [
            -(block_a.idx_spec.shape()[0] - 1),
            -(block_a.idx_spec.shape()[1] - 1),
            -(block_a.idx_spec.shape()[2] - 1),
        ];
        let bounds = Bounds {
            start_offsets_zyx,
            stop_offsets_zyx,
        };

        let mapping_props = View3DSpec::from_shape_contiguous([
            (bounds.stop_offsets_zyx[0] - bounds.start_offsets_zyx[0]) as usize,
            (bounds.stop_offsets_zyx[1] - bounds.start_offsets_zyx[1]) as usize,
            (bounds.stop_offsets_zyx[2] - bounds.start_offsets_zyx[2]) as usize,
        ])?;
        Ok(Self {
            bounds,
            mapping_props,
            is_auto: false,
            internal_offset: 0,
        })
    }

    fn _num_items_per_axis(&self) -> [isize; 3] {
        [
            self.bounds.stop_offsets_zyx[0] - self.bounds.start_offsets_zyx[0],
            self.bounds.stop_offsets_zyx[1] - self.bounds.start_offsets_zyx[1],
            self.bounds.stop_offsets_zyx[2] - self.bounds.start_offsets_zyx[2],
        ]
    }

    /// get the number of elements in the sequence
    pub fn len(&self) -> usize {
        // should we precompute this?
        let [nz, ny, nx] = self._num_items_per_axis();
        (nz * ny * nx - self.internal_offset) as usize
    }

    /// get the ith [`IdxOffset3D`]
    ///
    /// # Note
    /// Unfortunately, we can't get indexing syntax to work (e.g.
    /// `obj[idx]`). The function used for indexing expects must return a
    /// reference. While this makes a lot of sense for a container, it
    /// doesn't make a ton of sense for a range-like object
    pub fn get(&self, index: usize) -> IdxOffset3D {
        let [iz, iy, ix] = self
            .mapping_props
            .map_idx1d_to_3d((index as isize) + self.internal_offset);

        IdxOffset3D {
            local_offset_zyx: [
                iz + self.bounds.start_offsets_zyx[0],
                iy + self.bounds.start_offsets_zyx[1],
                ix + self.bounds.start_offsets_zyx[2],
            ],
        }
    }
}

// I suspect the thing we actually want to do is have the ability to create
// `n` iterators and to access the ith iterator

impl IntoIterator for IdxOffset3DSeq {
    type Item = IdxOffset3D;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        if self.is_auto {
            let mut out = Iter {
                bounds: self.bounds.clone(),
                next_offset_zyx: [0, 0, 0],
            };
            out.next();
            out
        } else {
            Iter {
                bounds: self.bounds.clone(),
                next_offset_zyx: self.bounds.start_offsets_zyx,
            }
        }
    }
}
