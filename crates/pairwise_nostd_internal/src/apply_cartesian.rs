use crate::accumulator::Accumulator;
use crate::misc::get_bin_idx;
use core::cmp;
use ndarray::{ArrayViewMut1, ArrayViewMut2, Axis};

/// Check if a 3D array shape (for a View3DProps) is valid
fn check_shape(shape_zyx: &[usize; 3]) -> Result<(), &'static str> {
    if shape_zyx.contains(&0) {
        Err("shape_zyx must not hold 0")
    } else {
        Ok(())
    }
}

/// View3DProps specifies how a "3D" array is laid out in memory
/// It _must_ be contiguous along the fast axis, which is axis 2.
///
///
#[derive(Clone)]
pub struct View3DProps {
    // these are signed ints because we do a lot of math with negative offsets
    // and want to avoid excessive casts
    shape_zyx: [isize; 3],
    strides_zyx: [isize; 3],
}

impl View3DProps {
    /// Create a contiguous-in-memory View3DProps from shape_zyx alone
    pub fn from_shape_contiguous(shape_zyx: [usize; 3]) -> Result<View3DProps, &'static str> {
        check_shape(&shape_zyx)?;
        Ok(Self {
            shape_zyx: [
                shape_zyx[0] as isize,
                shape_zyx[1] as isize,
                shape_zyx[2] as isize,
            ],
            strides_zyx: [
                (shape_zyx[1] * shape_zyx[2]) as isize,
                shape_zyx[2] as isize,
                1_isize,
            ],
        })
    }

    /// Create a View3DProps from shape_zyx and strides_zyx
    pub fn from_shape_strides(
        shape_zyx: [usize; 3],
        strides_zyx: [usize; 3],
    ) -> Result<View3DProps, &'static str> {
        check_shape(&shape_zyx)?;

        if strides_zyx[2] != 1 {
            Err("the blocks must be contiguous along the fast axis")
        } else if strides_zyx[1] < shape_zyx[2] * strides_zyx[2] {
            Err("the length of the contiguous axis can't exceed strides_zyx[1]")
        } else if strides_zyx[0] < shape_zyx[1] * strides_zyx[1] {
            Err("the length of axis 1 can't exceed strides_zyx[0]")
        } else if strides_zyx[0] < strides_zyx[1] {
            Err("strides_zyx[1] must not exceed strides_zyx[0]")
        } else {
            Ok(Self {
                shape_zyx: [
                    shape_zyx[0] as isize,
                    shape_zyx[1] as isize,
                    shape_zyx[2] as isize,
                ],
                strides_zyx: [
                    strides_zyx[0] as isize,
                    strides_zyx[1] as isize,
                    strides_zyx[2] as isize,
                ],
            })
        }
    }

    /// returns the number of elements that a slice must have to be described
    /// by self
    pub fn contiguous_length(&self) -> usize {
        (self.shape_zyx[0] * self.strides_zyx[0]) as usize
    }

    pub fn shape(&self) -> &[isize; 3] {
        &self.shape_zyx
    }

    /// map a 3D index to 1D
    pub fn map_idx(&self, iz: isize, iy: isize, ix: isize) -> isize {
        iz * self.strides_zyx[0] + iy * self.strides_zyx[1] + ix
    }
}

// it might make more sense to provide this in a separate format that reduces
// round-off error (e.g, global_domain_width and global_domain_shape)
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
struct IndexDisplacementVec {
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
// among "threadgroups")

// for the moment, this won't know anything about the min/max separation bins
struct IndexDisplacementVecItr {
    start_offsets_zyx: [isize; 3], // inclusive
    stop_offsets_zyx: [isize; 3],  // inclusive
    next_offset_zyx: [isize; 3],
}

impl IndexDisplacementVecItr {
    pub fn new_cross_iter(
        _block_a: &CartesianBlock,
        _block_b: &CartesianBlock,
    ) -> IndexDisplacementVecItr {
        todo!("not implemented yet!");
    }

    fn _update_to_next_offset(&mut self) {
        self.next_offset_zyx[2] += 1;
        if self.next_offset_zyx[2] == self.stop_offsets_zyx[2] {
            self.next_offset_zyx[2] = self.start_offsets_zyx[2];
            self.next_offset_zyx[1] += 1;
            if self.next_offset_zyx[1] == self.stop_offsets_zyx[1] {
                self.next_offset_zyx[1] = self.start_offsets_zyx[1];
                self.next_offset_zyx[0] += 1;
            }
        }
    }

    pub fn new_auto_iter(block: &CartesianBlock) -> IndexDisplacementVecItr {
        let stop_offsets_zyx = block.idx_props.shape_zyx;
        let start_offsets_zyx = [
            0, // <- we never have a negative z
            -(stop_offsets_zyx[1] - 1),
            -(stop_offsets_zyx[2] - 1),
        ];

        let mut out = Self {
            stop_offsets_zyx,
            start_offsets_zyx,
            next_offset_zyx: [0, 0, 0],
        };
        out._update_to_next_offset();
        out
    }
}

impl Iterator for IndexDisplacementVecItr {
    type Item = IndexDisplacementVec;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stop_offsets_zyx == self.next_offset_zyx {
            None
        } else {
            let out = Some(IndexDisplacementVec {
                local_displacement_vec_zyx: self.next_offset_zyx,
            });
            self._update_to_next_offset();
            out
        }
    }
}

fn apply_cartesian_fixed_separation(
    statepack: &mut ArrayViewMut1<f64>,
    accum: &impl Accumulator,
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    separation_vec: IndexDisplacementVec,
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

    for k in k_start..k_stop {
        for j in j_start..j_stop {
            for i in i_start..i_stop {
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

pub fn apply_cartesian(
    statepacks: &mut ArrayViewMut2<f64>,
    accum: &impl Accumulator,
    block_a: &CartesianBlock,
    block_b: Option<&CartesianBlock>,
    squared_distance_bin_edges: &[f64],
    cell_width: &CellWidth, // maybe recombine cell_width and CartesianBlock?
                            // todo: accept pairwise_fn (we'll need to change the interface)
) -> Result<(), &'static str> {
    // TODO: check size of output buffers

    // Check that bin_edges are monotonically increasing
    if !squared_distance_bin_edges.is_sorted() {
        return Err("squared_distance_bin_edges must monotonically increase");
    }

    if let Some(block_b) = block_b {
        IndexDisplacementVecItr::new_cross_iter(block_a, block_b).for_each(
            |displacement_vec: IndexDisplacementVec| {
                let distance2 = displacement_vec.distance_squared(block_a, block_b, cell_width);
                if let Some(distance_bin_idx) = get_bin_idx(distance2, squared_distance_bin_edges) {
                    apply_cartesian_fixed_separation(
                        &mut statepacks.index_axis_mut(Axis(1), distance_bin_idx),
                        accum,
                        block_a,
                        block_b,
                        displacement_vec,
                    );
                }
            },
        );
    } else {
        IndexDisplacementVecItr::new_auto_iter(block_a).for_each(
            |displacement_vec: IndexDisplacementVec| {
                let distance2 = displacement_vec.distance_squared(block_a, block_a, cell_width);
                if let Some(distance_bin_idx) = get_bin_idx(distance2, squared_distance_bin_edges) {
                    apply_cartesian_fixed_separation(
                        &mut statepacks.index_axis_mut(Axis(1), distance_bin_idx),
                        accum,
                        block_a,
                        block_a,
                        displacement_vec,
                    );
                }
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn idx_props_simple() {
        let idx_props = View3DProps::from_shape_strides([2, 3, 4], [18, 6, 1]).unwrap();
        assert_eq!(idx_props.map_idx(0, 0, 0), 0);
        assert_eq!(idx_props.map_idx(0, 0, 3), 3);
        assert_eq!(idx_props.map_idx(0, 1, 0), 6);
        assert_eq!(idx_props.map_idx(1, 1, 0), 24);
    }

    #[test]
    fn idx_props_contig() {
        let idx_props = View3DProps::from_shape_contiguous([2, 3, 4]).unwrap();
        assert_eq!(idx_props.map_idx(0, 0, 0), 0);
        assert_eq!(idx_props.map_idx(0, 0, 3), 3);
        assert_eq!(idx_props.map_idx(0, 1, 0), 4);
        assert_eq!(idx_props.map_idx(1, 1, 0), 16);
    }

    #[test]
    fn idx_props_errs() {
        assert!(View3DProps::from_shape_contiguous([2, 3, 0]).is_err());
        assert!(View3DProps::from_shape_contiguous([2, 0, 4]).is_err());
        assert!(View3DProps::from_shape_contiguous([0, 3, 4]).is_err());

        assert!(View3DProps::from_shape_strides([2, 3, 4], [18, 6, 0]).is_err());
        assert!(View3DProps::from_shape_strides([2, 3, 4], [18, 3, 1]).is_err());
        assert!(View3DProps::from_shape_strides([2, 3, 4], [18, 20, 1]).is_err());
        assert!(View3DProps::from_shape_strides([2, 3, 4], [11, 4, 1]).is_err());
    }

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
