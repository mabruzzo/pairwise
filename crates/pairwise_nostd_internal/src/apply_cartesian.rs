use crate::accumulator::Accumulator;
use crate::misc::{View3DProps, get_bin_idx};
use crate::parallel::{ReductionSpec, TeamProps, TeamRank};
use core::cmp;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

// it might make more sense to provide this in a separate format that reduces
// round-off error (e.g, global_domain_width and global_domain_shape)
#[derive(Clone, Copy)]
pub struct CellWidth {
    widths_zyx: [f64; 3],
}

impl CellWidth {
    pub fn new(widths_zyx: [f64; 3]) -> Result<CellWidth, &'static str> {
        if widths_zyx.iter().any(|elem| elem <= &0.0) {
            Err("each cell width must be positive")
        } else {
            Ok(CellWidth { widths_zyx })
        }
    }
}

#[derive(Clone)]
pub struct CartesianBlock<'a> {
    value_components_zyx: [&'a [f64]; 3],
    weights: &'a [f64],
    /// the layout information
    idx_props: View3DProps,
    /// the index offset from the left edge of the domain in cells
    start_idx_global_offset: [usize; 3],
    // if we ever choose to support an AMR context, we may want to hold a
    // multiple for how wide the cells are compared to some either the
    // coarsest cell-widths or the finest cell-widths
}

impl<'a> CartesianBlock<'a> {
    /// create a new instance
    pub fn new(
        value_components_zyx: [&'a [f64]; 3],
        weights: &'a [f64],
        idx_props: View3DProps,
        start_idx_global_offset: [usize; 3],
    ) -> Result<CartesianBlock<'a>, &'static str> {
        if weights.len() < idx_props.contiguous_length() {
            Err("length of weights is inconsistent with strides and shape")
        } else if value_components_zyx
            .iter()
            .any(|x| x.len() != weights.len())
        {
            Err("the length of each value component must be the same as weights")
        } else {
            Ok(Self {
                value_components_zyx,
                weights,
                idx_props,
                start_idx_global_offset,
            })
        }
    }
}

mod displacement_vec {
    // todo: stop importing everything from super
    use super::*;
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
    pub struct IndexDisplacementVec {
        local_displacement_vec_zyx: [isize; 3],
    }

    impl IndexDisplacementVec {
        /// computes the "local index displacement vector." The core documentation
        /// [for the IndexDisplacementVec type](IndexDisplacementVec) provides more
        /// context.
        pub fn local_displacement_vec(&self) -> &[isize; 3] {
            &self.local_displacement_vec_zyx
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
                out[i] = (off_b - off_a) + self.local_displacement_vec_zyx[i];
            }
            out
        }

        /// calculates squared distance represented by the displacement vector
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

    // define the logic for iterating over DisplacementRays
    // ----------------------------------------------------
    // the following probably belongs in its own file. We will want to expose
    // something like this to help with the tiling pattern when people are using
    // MPI (in practice, we may expose this logic with a separate construct and
    // just reuse the shared logic)
    //
    // At the moment, this is implemented like an iterator. In practice, this isn't
    // the appropriate abstraction. Instead, we probably want to make this
    // "indexable" so we can efficiently partition out the work (to be distributed
    // among "threadgroups"). I think we instead want to produce something like a
    // range object (i.e. that is indexable)
    #[derive(Clone)]
    struct Bounds {
        start_offsets_zyx: [isize; 3], // inclusive
        stop_offsets_zyx: [isize; 3],  // inclusive
    }

    /// this is used in the context of iterating over "'local' index
    /// displacement vectors".
    ///
    /// `cur_offset` holds the internals of a displacement vector. This
    /// function returns the next displacement vector ("next" in the sense
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
    /// displacement vector.
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
    /// At the moment, the iterator steps through displacement vectors. But, I
    /// think that it would make more sense to attach an iterator to
    /// View3DProps and simply perform the transformation
    pub struct Iter {
        bounds: Bounds,
        next_offset_zyx: [isize; 3],
    }

    impl Iterator for Iter {
        type Item = IndexDisplacementVec;

        fn next(&mut self) -> Option<Self::Item> {
            if self.bounds.stop_offsets_zyx == self.next_offset_zyx {
                None
            } else {
                let out = Some(IndexDisplacementVec {
                    local_displacement_vec_zyx: self.next_offset_zyx,
                });
                self.next_offset_zyx = get_next_offset(self.next_offset_zyx, &self.bounds);
                out
            }
        }
    }

    /// the idea is for this to be an indexable "range" object that contains all
    /// relevant `IndexDisplacementVec` instances
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
    /// - we can enumerate all `Idx3DOffset` values
    /// - It’s convenient to think of this list of `Idx3DOffset` instances as
    ///   a 3D array (there is a straight-forward mapping from index to
    ///   `Idx3DOffset).
    /// - At the corners of this 3D array, you have the largest displacements
    ///   (only a small subset of pairs are separated by this amount)
    /// - the smallest displacements describe the offsets between the most
    ///   pairs of points.
    /// - Auto-correlation is just a special case.
    ///   - For equal sized blocks, I think this is 1 less than half of the
    ///     blocks
    pub struct DisplacementSeq {
        bounds: Bounds,
        mapping_props: View3DProps,
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

    impl DisplacementSeq {
        pub fn new_auto(block: &CartesianBlock) -> Result<Self, &'static str> {
            let stop_offsets_zyx = *block.idx_props.shape();
            let bounds = Bounds {
                start_offsets_zyx: [
                    0, // <- we never have a negative z in "auto" pairs
                    -(stop_offsets_zyx[1] - 1),
                    -(stop_offsets_zyx[2] - 1),
                ],
                stop_offsets_zyx,
            };

            let mapping_props = View3DProps::from_shape_contiguous([
                (bounds.stop_offsets_zyx[0] - bounds.start_offsets_zyx[0]) as usize,
                (bounds.stop_offsets_zyx[1] - bounds.start_offsets_zyx[1]) as usize,
                (bounds.stop_offsets_zyx[2] - bounds.start_offsets_zyx[2]) as usize,
            ])?;

            let first_displacement_vec = get_next_offset([0, 0, 0], &bounds);
            // convert the first displacement vector to a 3D index
            let [iz, iy, ix] = [
                first_displacement_vec[0] - bounds.start_offsets_zyx[0],
                first_displacement_vec[1] - bounds.start_offsets_zyx[1],
                first_displacement_vec[2] - bounds.start_offsets_zyx[2],
            ];
            // convert the 3D index to a 1D offset
            let internal_offset = mapping_props.map_idx(iz, iy, ix);

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
            let stop_offsets_zyx = *block_b.idx_props.shape();
            let start_offsets_zyx = [
                -(block_a.idx_props.shape()[0] - 1),
                -(block_a.idx_props.shape()[1] - 1),
                -(block_a.idx_props.shape()[2] - 1),
            ];
            let bounds = Bounds {
                start_offsets_zyx,
                stop_offsets_zyx,
            };

            let mapping_props = View3DProps::from_shape_contiguous([
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

        /// get the ith displacement vector
        ///
        /// # Note
        /// Unfortunately, we can't get indexing syntax to work (e.g.
        /// `obj[idx]`). The function used for indexing expects must return a
        /// reference. While this makes a lot of sense for a container, it
        /// doesn't make a ton of sense for a range-like object
        pub fn get(&self, index: usize) -> IndexDisplacementVec {
            let [iz, iy, ix] = self
                .mapping_props
                .reverse_map_idx((index as isize) + self.internal_offset);

            IndexDisplacementVec {
                local_displacement_vec_zyx: [
                    iz + self.bounds.start_offsets_zyx[0],
                    iy + self.bounds.start_offsets_zyx[1],
                    ix + self.bounds.start_offsets_zyx[2],
                ],
            }
        }
    }

    // I suspect the thing we actually want to do is have the ability to create
    // `n` iterators and to access the ith iterator

    impl IntoIterator for DisplacementSeq {
        type Item = IndexDisplacementVec;
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
} // mod Displacement

fn apply_cartesian_fixed_separation(
    statepack: &mut ArrayViewMut1<f64>,
    accum: &impl Accumulator,
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    separation_vec: displacement_vec::IndexDisplacementVec,
    offset: usize,
    stride: usize,
    // todo: accept pairwise_fn (we'll need to change the interface)
) {
    // index_offset holds the offset you add to an index of block_a to get the
    // index of block_b that is separated by the separation_vec.
    let index_offset = separation_vec.local_displacement_vec();
    let mut idx_a_start = [0_isize; 3];
    let mut idx_a_stop = [0_isize; 3];
    for i in 0..3 {
        // start is abs(index_offset[i]) if index_offset[i] < 0. Otherwise, it's 0
        let start = cmp::max(-index_offset[i], 0_isize);
        // compute the number of elements along axis (we can definitely make
        // this more concise -- but we should be very clear what it means)
        let n_elem: isize = if index_offset[i] < 0 {
            cmp::min(
                block_a.idx_props.shape()[i] - start,
                block_b.idx_props.shape()[i],
            )
        } else {
            cmp::min(
                block_a.idx_props.shape()[i],
                block_b.idx_props.shape()[i] - index_offset[i],
            )
        };
        idx_a_start[i] = start;
        idx_a_stop[i] = start + n_elem;
    }

    let [k_start, j_start, i_start] = idx_a_start;
    let [k_stop, j_stop, i_stop] = idx_a_stop;
    let [k_offset, j_offset, i_offset] = index_offset;

    let _step = 1_usize; // this should change if using vectors or are on a GPU
    // we are going to need to do a bunch of work here to
    // any other step-size

    let my_offset = offset as isize;
    let get_i_itr = || ((i_start + my_offset)..i_stop).step_by(stride);

    for k in k_start..k_stop {
        for j in j_start..j_stop {
            for i in get_i_itr() {
                let i_a = block_a.idx_props.map_idx(k, j, i) as usize;
                let va_x = block_a.value_components_zyx[2][i_a];
                let va_y = block_a.value_components_zyx[1][i_a];
                let va_z = block_a.value_components_zyx[0][i_a];
                let wa = block_a.weights[i_a];

                let i_b = block_b
                    .idx_props
                    .map_idx(k + k_offset, j + j_offset, i + i_offset)
                    as usize;

                let vb_x = block_b.value_components_zyx[2][i_b];
                let vb_y = block_b.value_components_zyx[1][i_b];
                let vb_z = block_b.value_components_zyx[0][i_b];
                let wb = block_b.weights[i_b];

                let dv = (va_x * vb_x) + (va_y * vb_y) + (va_z * vb_z);
                accum.consume(statepack, dv, wa * wb);
            }
        }
    }
}

pub struct CartesianCalcContext<'a, T: Accumulator> {
    accum: T,
    block_a: CartesianBlock<'a>,
    block_b: CartesianBlock<'a>,
    displacement_seq: displacement_vec::DisplacementSeq,
    squared_distance_bin_edges: &'a [f64],
    cell_width: CellWidth, // maybe recombine cell_width and CartesianBlock?
                           // todo: accept pairwise_fn (we'll need to change the interface)
}

impl<'a, T: Accumulator> CartesianCalcContext<'a, T> {
    pub fn new(
        accum: T,
        block_a: &CartesianBlock<'a>,
        block_b: Option<&CartesianBlock<'a>>,
        squared_distance_bin_edges: &'a [f64],
        cell_width: &CellWidth, // maybe recombine cell_width and CartesianBlock?
                                // todo: accept pairwise_fn (we'll need to change the interface)
    ) -> Result<Self, &'static str> {
        // Check that bin_edges are monotonically increasing
        if !squared_distance_bin_edges.is_sorted() {
            return Err("squared_distance_bin_edges must monotonically increase");
        }

        // todo: we want to move away from using this iterable. Instead we want to
        //       map an index to a displacement vector (to enable GPU support)
        let (seq, other_block_ref) = match block_b {
            Some(block_b) => (
                displacement_vec::DisplacementSeq::new_cross(block_a, block_b)?,
                block_b,
            ),
            None => (
                displacement_vec::DisplacementSeq::new_auto(block_a)?,
                block_a,
            ),
        };

        Ok(Self {
            accum,
            block_a: block_a.clone(),
            block_b: other_block_ref.clone(),
            displacement_seq: seq,
            squared_distance_bin_edges,
            cell_width: *cell_width,
        })
    }

    fn n_spatial_bins(&self) -> usize {
        self.squared_distance_bin_edges.len() - 1
    }
}

impl<'a, T: Accumulator> ReductionSpec for CartesianCalcContext<'a, T> {
    fn statepacks_shape(&self) -> [usize; 2] {
        [self.accum.statepack_size(), self.n_spatial_bins()]
    }

    fn collect_team_contrib(&self, team_props: &mut impl TeamProps) {
        let team_size = team_props.team_size() as usize;
        let league_rank = team_props.league_rank() as usize;
        let league_size = team_props.league_size() as usize;
        let nominal_chunk_size = self.displacement_seq.len() / league_size;
        // TODO: distribute the remainder of the work more evenly
        // let remainder = self.displacement_seq.len() % league_size;
        let start = nominal_chunk_size * league_rank;
        let displacement_vec_indices = if (league_rank + 1) == league_size {
            start..self.displacement_seq.len()
        } else {
            start..(start + nominal_chunk_size)
        };

        // set up the reduction function
        let reduce_buf_fn = |reduce_buf: &mut ArrayViewMut1<f64>, other_buf: &ArrayView1<f64>| {
            self.accum.merge(reduce_buf, other_buf.view());
        };

        // iterate over the displacement_vec (the displacement-vectors that a
        // thread-team considers is determined by the league_rank)
        for index in displacement_vec_indices {
            let displacement_vec = self.displacement_seq.get(index);

            // compute the squared distance that corresponds to the thread-length
            let distance2 =
                displacement_vec.distance_squared(&self.block_a, &self.block_b, &self.cell_width);
            let Some(distance_bin_idx) = get_bin_idx(distance2, self.squared_distance_bin_edges)
            else {
                continue; /* skip over the rest of this loop */
            };

            // add up contributions for all pais separated by displacement_vec
            // and perform a reduction so that they are all held in a single
            // buffer
            team_props.team_reduce(
                &|reduce_buf: &mut ArrayViewMut1<f64>, team_rank: TeamRank| {
                    self.accum.reset_statepack(reduce_buf);
                    apply_cartesian_fixed_separation(
                        reduce_buf,
                        &self.accum,
                        &self.block_a,
                        &self.block_b,
                        displacement_vec.clone(),
                        team_rank.get() as usize, // offset
                        team_size,
                    );
                },
                &reduce_buf_fn,
            );

            // now at this point, we update the output buffer
            team_props.update_teambuf_if_root(
                &|statepacks: &mut ArrayViewMut2<f64>, bufs: &ArrayView2<f64>| {
                    // since we already did the reduction, the only part of
                    // bufs that we care about is:
                    let contrib = bufs.index_axis(Axis(1), 0);
                    self.accum.merge(
                        &mut statepacks.index_axis_mut(Axis(1), distance_bin_idx),
                        contrib,
                    );
                },
            );
        }
    }

    fn init_team_statepacks(&self, buf: &mut ArrayViewMut2<f64>) {
        // this is going to be very slow!
        for distance_bin_idx in 0..self.n_spatial_bins() {
            self.accum
                .reset_statepack(&mut buf.index_axis_mut(Axis(1), distance_bin_idx));
        }
    }

    fn league_reduce(&self, primary: &mut ArrayViewMut2<f64>, other: ArrayView2<f64>) {
        // this is going to be very slow!
        for distance_bin_idx in 0..self.n_spatial_bins() {
            self.accum.merge(
                &mut primary.index_axis_mut(Axis(1), distance_bin_idx),
                other.index_axis(Axis(1), distance_bin_idx),
            );
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn cartesian_block() {
        let velocity_x = [0.0; 4];
        let velocity_y = [0.0; 4];
        let velocity_z = [4.0, 1.0, 2.0, -3.0];
        let weights = [1.0; 4];

        let block = CartesianBlock::new(
            [&velocity_z, &velocity_y, &velocity_x],
            &weights,
            View3DProps::from_shape_contiguous([1, 1, 4]).unwrap(),
            [0, 0, 0],
        );
        assert!(block.is_ok());
    }
}
