//! This module defines datatypes used for representing spatial grid data

use crate::misc::View3DSpec;

// todo: maybe put PointProps into this file?

// it might make more sense to provide this in a separate format that reduces
// round-off error (e.g, global_domain_width and global_domain_shape)
#[derive(Clone, Copy)]
pub struct CellWidth {
    pub(crate) widths_zyx: [f64; 3],
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
    pub(crate) value_components_zyx: [&'a [f64]; 3],
    pub(crate) weights: &'a [f64],
    /// the layout information
    pub(crate) idx_spec: View3DSpec,
    /// the index offset from the left edge of the domain in cells
    /// (this is isize because it is used in the context of signed ints)
    pub(crate) start_idx_global_offset: [isize; 3],
    // if we ever choose to support an AMR context, we may want to hold a
    // multiple for how wide the cells are compared to some either the
    // coarsest cell-widths or the finest cell-widths
}

impl<'a> CartesianBlock<'a> {
    /// create a new instance
    pub fn new(
        value_components_zyx: [&'a [f64]; 3],
        weights: &'a [f64],
        idx_props: View3DSpec,
        start_idx_global_offset: [isize; 3],
    ) -> Result<CartesianBlock<'a>, &'static str> {
        if weights.len() < idx_props.required_length() {
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
                idx_spec: idx_props,
                start_idx_global_offset,
            })
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
            View3DSpec::from_shape_contiguous([1, 1, 4]).unwrap(),
            [0, 0, 0],
        );
        assert!(block.is_ok());
    }

    #[test]
    fn cartesian_block_err() {
        let velocity_x = [0.0; 3];
        let velocity_y = [0.0; 4];
        let velocity_z = [4.0, 1.0, 2.0, -3.0];
        let weights = [1.0; 4];

        let block = CartesianBlock::new(
            [&velocity_z, &velocity_y, &velocity_x],
            &weights,
            View3DSpec::from_shape_contiguous([1, 1, 4]).unwrap(),
            [0, 0, 0],
        );
        assert!(block.is_err());
    }
}
