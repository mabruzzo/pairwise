use crate::accumulator::Accumulator;
use core::cmp;
use ndarray::{ArrayViewMut1, Axis};

fn _check_shape(shape_zyx: &[usize; 3]) -> Result<(), &'static str> {
    if shape_zyx.contains(&0) {
        Err("shape_zyx must not hold 0")
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct View3DProps {
    shape_zyx: [isize; 3],
    strides_zyx: [isize; 3],
}

impl View3DProps {
    pub fn from_shape_contiguous(shape_zyx: [usize; 3]) -> Result<View3DProps, &'static str> {
        _check_shape(&shape_zyx)?;
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

    pub fn from_shape_strides(
        shape_zyx: [usize; 3],
        strides_zyx: [usize; 3],
    ) -> Result<View3DProps, &'static str> {
        _check_shape(&shape_zyx)?;

        if strides_zyx[2] != 1 {
            Err("the blocks must be contiguous along the fast axis")
        } else if strides_zyx[1] < shape_zyx[2] * strides_zyx[2] {
            Err("the length of the contiguous axis can't exceed strides_zyx[1]")
        } else if strides_zyx[0] < shape_zyx[1] * strides_zyx[1] {
            Err("the stride along axis 0 is too small")
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

    /// returns the number of elements in a slice described by self
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

pub struct CartesianBlock<'a> {
    value_components_zyx: [&'a [f64]; 3],
    weights: &'a [f64],
    /// the layout information
    idx_props: View3DProps,
    /// the offset from the left edge of the domain in cells
    left_edge_global_offset: [usize; 3],
    /// it might be more accurate to provide block_width than cell_width
    cell_width: [f64; 3],
}

impl<'a> CartesianBlock<'a> {
    /// create a new instance
    pub fn new(
        value_components_zyx: [&'a [f64]; 3],
        weights: &'a [f64],
        idx_props: View3DProps,
        left_edge_global_offset: [usize; 3],
        cell_width: [f64; 3],
    ) -> Result<CartesianBlock<'a>, &'static str> {
        if weights.len() < idx_props.contiguous_length() {
            Err("length of weights is inconsistent with strides and shape")
        } else if value_components_zyx
            .iter()
            .any(|x| x.len() != weights.len())
        {
            Err("the length of each value component must be the same as weights")
        } else if cell_width.iter().any(|elem| elem <= &0.0) {
            Err("the length of each cell width must be positive")
        } else {
            Ok(Self {
                value_components_zyx,
                weights,
                idx_props,
                left_edge_global_offset,
                cell_width,
            })
        }
    }
}

/// encapsulates the displacement ray whose head and tail are located at the
/// center of cells. This vector spans 2 separate Blocks and the bounding
/// points in terms of the local index
///
/// TODO: I'm thinking that this should really just encapsulate the quantity
///       that is returned by `local_displacement_vec` (we don't really care
///       about the ray's start & end ponts, we just care about the
///       displacement vector)
struct IndexDisplacementRay {
    /// head of the ray
    block_a_idx: [usize; 3],
    /// tail of the ray
    block_b_idx: [usize; 3],
}

impl IndexDisplacementRay {
    /// computes the "local-part" of the displacement vector in terms of local
    /// indices. The result is a mathematical vector with 3-components (1 per
    /// spatial dimension)
    ///
    /// To better define what this means, let's consider the total
    /// displacement vector that describes the ray.
    /// - for the sake of discussion, consider a "global index space." In other
    ///   words, imagine we concatenated all of the blocks together into a
    ///   single giant cartesian grid.
    /// - The ith component of the ray's tail index and head index, in this
    ///   "global index space", are defined as:
    ///   - `tail[i] = block_a_idx[i] + block_a.left_edge_global_offset[i]`
    ///   - `head[i] = block_b_idx[i] + block_b.left_edge_global_offset[i]`
    /// - The ith component of the total displacement vector is:
    ///   `total_displacement_vec[i] = head[i] - tail[i]`
    /// - We can rewrite this as:
    ///   ```text
    ///   total_displacement_vec[i] =
    ///      local_displacement[i] + global_displacement_vec[i]
    ///   ```
    ///   where `local_displacement[i] = block_b_idx[i] - block_a_idx[i]`
    pub fn local_displacement_vec(&self) -> [isize; 3] {
        // todo: can we come up with a shorter description
        // todo:
        [
            (self.block_b_idx[0] as isize) - (self.block_a_idx[0] as isize),
            (self.block_b_idx[1] as isize) - (self.block_a_idx[1] as isize),
            (self.block_b_idx[2] as isize) - (self.block_a_idx[2] as isize),
        ]
    }

    /// calculates the (mathematical) displacement vector
    pub fn calc_displacement_vec(
        &self,
        block_a: CartesianBlock,
        block_b: CartesianBlock,
    ) -> [f64; 3] {
        assert!(block_a.cell_width == block_b.cell_width);
        let mut out = [0.0; 3];
        for i in 0..3 {
            let start = block_a.left_edge_global_offset[i] + self.block_a_idx[i];
            let end = block_b.left_edge_global_offset[i] + self.block_b_idx[i];
            out[i] = (end - start) as f64 * block_a.cell_width[i];
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

/*
enum BlockBOriginOffset {
    NoOffset,
    NoOffsetZY(usize),
    NoOffsetZ(usize, usize),
    Offset(usize, usize, usize),
    Exhausted,
}


struct DisplacementRayItr {
    block_shape_a: [usize; 3],
    block_shape_b: [usize; 3],
    global_index_offset: [usize; 3],
    cell_width: [f64; 3],
    min_squared_length: f64,
    max_squared_length: f64,
    next_block_b_idx: BlockBOriginOffset,
}

impl DisplacementRayItr {
    fn new(
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
        squared_bin_edges: &[f64],
    ) -> DisplacementRayItr {
        let max_idx_b = [1_usize; 3]; // <- a placeholder

        let next_block_b_idx = if squared_bin_edges[0] == 0.0
            && block_a.left_edge_global_offset == block_b.left_edge_global_offset
        {
            BlockBOriginOffset::NoOffset
        } else {
            panic!("I don't think this is always correct!");
            Some([0_usize, 0_usize, min_nonzero_idx_b[2]])
        };

        Self {
            min_nonzero_idx_b,
            max_idx_b,
            next_block_b_idx,
        }
    }
}

impl Iterator for DisplacementRayItr {
    type Item = IndexDisplacementRay;

    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement me!");
    }
}
*/

fn apply_cartesian_fixed_separation(
    statepack: &mut ArrayViewMut1<f64>,
    accum: &impl Accumulator,
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    separation_vec: IndexDisplacementRay,
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

    let step = 1_usize; // this should change if using vectors or are on a GPU
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

/*
fn apply_cartesian(
    statepack: &mut ArrayViewMut1<f64>,
    accum: &impl Accumulator,
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    separation_vec: IndexDisplacementRay,
    // todo: accept pairwise_fn (we'll need to change the interface)
) {
}
*/

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
}
