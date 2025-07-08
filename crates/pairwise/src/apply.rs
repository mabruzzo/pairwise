use std::num::NonZeroU32;

use crate::parallel_serial::SerialExecutor;
use ndarray::ArrayViewMut2;
use pairwise_nostd_internal::{
    Accumulator, CartesianBlock, CartesianCalcContext, CellWidth, Executor,
};

pub fn apply_cartesian<T: Accumulator + Clone>(
    statepacks: &mut ArrayViewMut2<f64>,
    accum: &T,
    block_a: &CartesianBlock,
    block_b: Option<&CartesianBlock>,
    squared_distance_bin_edges: &[f64],
    cell_width: &CellWidth, // maybe recombine cell_width and CartesianBlock?
                            // todo: accept pairwise_fn (we'll need to change the interface)
) -> Result<(), &'static str> {
    let context = CartesianCalcContext::new(
        accum.clone(),
        block_a,
        block_b,
        squared_distance_bin_edges,
        cell_width,
    )?;

    // todo: make this customizable
    let mut executor = SerialExecutor;
    executor.drive_reduce(
        statepacks,
        &context,
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(1).unwrap(),
    )?;
    Ok(())
}
