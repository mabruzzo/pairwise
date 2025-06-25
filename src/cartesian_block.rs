use core::cmp;
use vecpack::ScalarVecCompat;

use crate::vecpack::ScalarVecCompat;

pub struct CartesianBlock<'a> {
    value_components: [&'a [f64]; 3],
    weights: &'a [f64],
    /// the layout information
    block_shape: [usize; 3],
    strides: [usize; 3],
    /// the offset from the left edge of the domain in cells
    left_edge_global_offset: [usize; 3],
    /// it might be more accurate to provide block_width than cell_width
    cell_width: [f64; 3],
}

impl<'a> CartesianBlock<'a> {
    /// create a new instance
    pub fn new(
        value_components: [&'a [f64]; 3],
        weights: &'a [f64],
        block_shape: [usize; 3],
        strides: [usize; 3],
        left_edge_global_offset: [usize; 3],
        cell_width: [f64; 3],
    ) -> Result<CartesianBlock<'a>, &'static str> {
        let min_len = strides[0] * block_shape[0];
        if strides[2] != 1 {
            Err("the blocks must be contiguous along the fast axis")
        } else if (strides[0] < block_shape[1] * block_shape[1]) || (strides[1] < block_shape[2]) {
            Err("there is a problem with the strides")
        } else if weights.len() < min_len {
            Err("length of weights is inconsistent with strides and shape")
        } else if value_components.iter().any(|x| x.len() != weights.len()) {
            Err("the length of each value component must be the same as weights")
        } else if cell_width.iter().any(|elem| elem <= &0.0) {
            Err("the length of each cell width must be positive")
        } else {
            Ok(Self {
                value_components,
                weights,
                block_shape,
                strides,
                left_edge_global_offset,
                cell_width,
            })
        }
    }
}

/// encapsulates the displacement ray whose head and tail are located at the
/// center of cells. This vector spans 2 separate Blocks and the bounding
/// points in terms of the local index
struct IndexDisplacementRay {
    block_a_idx: [usize; 3],
    /// tail of the ray
    block_b_idx: [usize; 3],
}

impl IndexDisplacementRay {
    /// calculates the (mathematical) displacement vector
    fn calc_displacement_vec(&self, block_a: CartesianBlock, block_b: CartesianBlock) -> [f64; 3] {
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
    global_idx_offset: [usize; 3],
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

fn fixed_sep_apply_accum_helper_simple(
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    separation_vec: IndexDisplacementRay,
    // todo consolidate with weight_sum into a single object
    weighted_sum: &mut [f64],
    weight_total: &mut [f64],
) {
    // consider modifying separation_vec, to hold more immediately useful info
    let mut niter = [0_usize; 3];
    for i in 0..3 {
        niter[i] = cmp::min(
            block_a.block_shape[i] - separation_vec.block_a_idx[i],
            block_b.block_shape[i] - separation_vec.block_b_idx[i],
        );
    }
    let [nz, ny, nx] = niter;

    let step = 1_usize; // this should change if using vectors or are on a GPU
                        // we are going to need to do a bunch of work here to
                        // any other step-size

    // consider changing this order?
    let [valsA_x, valsA_y, valsA_z] = block_a.value_components;
    let [valsB_x, valsB_y, valsB_z] = block_b.value_components;

    for offset_z in 0..nz {
        let iz_a = separation_vec.block_a_idx[0] + offset_z;
        let iz_b = separation_vec.block_b_idx[0] + offset_z;
        for offset_y in 0..ny {
            let iy_a = separation_vec.block_a_idx[1] + offset_y;
            let iy_b = separation_vec.block_b_idx[1] + offset_y;

            for offset_x in 0..nx {
                let idx_a = (iz_a * block_a.strides[0]
                    + iy_a * block_a.strides[1]
                    + separation_vec.block_a_idx[2]
                    + offset_x);

                let vA_x = valsA_x[idx_a];
                let vA_y = valsA_y[idx_a];
                let vA_z = valsA_z[idx_a];
                let wA = block_a.weights[idx_a];

                let idx_b = (iz_b * block_b.strides[0]
                    + iy_b * block_b.strides[1]
                    + separation_vec.block_b_idx[2]
                    + offset_x);

                let vB_x = valsB_x[idx_b];
                let vB_y = valsB_y[idx_b];
                let vB_z = valsB_z[idx_b];
                let wB = block_b.weights[idx_b];

                // here is the actual kernel
                let dv =
                    ((vA_x - vB_x).powi(2) + (vA_y - vB_y).powi(2) + (vA_z - vB_z).powi(2)).sqrt();
                let weight_product = wA * wB;

                // now update accumulator register

                // the index will depend on stride
                // (this logic should be factored out)
                weighted_sum[0] += dv * weight_product;
                weight_total[0] += weight_product;
            }
        }
    }
}

/*
/// apply the accumulation algorithm for a fixed separation
fn fixed_sep_apply_accum_helper<VecPack: ScalarVecCompat<T = f64>>(
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    separation_vec: IndexDisplacementRay,
    // todo consolidate with weight_sum into a single object
    weighted_sum: &mut [f64],
    weight_sum: &mut [f64],
) {
    // consider modifying separation_vec, to hold more immediately useful info
    let mut niter = [0_usize; 3];
    for i in 0..3 {
        niter[i] = cmp::min(
            block_a.block_shape[i] - separation_vec.block_a_idx[i],
            block_b.block_shape[i] - separation_vec.block_b_idx[i],
        );
    }
    let [nz, ny, nx] = niter;

    let step = 1_usize; // this should change if using vectors or are on a GPU
                        // we are going to need to do a bunch of work here to
                        // any other step-size

    // consider changing this order?
    let [valsA_x, valsA_y, valsA_z] = block_a.value_components;
    let [valsB_x, valsB_y, valsB_z] = block_b.value_components;

    for offset_z in 0..nz {
        let iz_a = separation_vec.block_a_idx[0] + offset_z;
        let iz_b = separation_vec.block_b_idx[0] + offset_z;
        for offset_y in 0..ny {
            let iy_a = separation_vec.block_a_idx[1] + offset_y;
            let iy_b = separation_vec.block_b_idx[1] + offset_y;

            let mut idx_a =
                (iz * block_a.strides[0] + iy * block_a.strides[1] + separation_vec.block_a_idx[2]);
            let vA_x = VecPack::copy_from(valsA_x, idx_a);
            let vA_y = VecPack::copy_from(valsA_y, idx_a);
            let vA_z = VecPack::copy_from(valsA_z, idx_a);
            let wA = VecPack::copy_from(block_a.weights, idx_a);

            let mut idx_b =
                (iz * block_a.strides[0] + iy * block_a.strides[1] + separation_vec.block_a_idx[2]);
            let vB_x = VecPack::copy_from(valsB_x, idx_b);
            let vB_y = VecPack::copy_from(valsB_y, idx_b);
            let vB_z = VecPack::copy_from(valsB_z, idx_b);
            let wB = VecPack::copy_from(block_b.weights, idx_b);

            for offset_x in 1..nx {

                // do work...
            }
        }
    }
}
*/
