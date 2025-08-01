use crate::parallel_serial::SerialExecutor;
use pairwise_nostd_internal::{
    BinEdges, CartesianBlock, CellWidth, Executor, PairOperation, Reducer, StatePackViewMut,
    TwoPointCartesian,
};
use std::num::NonZeroU32;

// we may throw this out, but we are only using it right now for the sake of preserving tests...
pub fn apply_cartesian<T: Reducer + Clone, B: BinEdges + Clone>(
    binned_statepacks: &mut StatePackViewMut,
    reducer: &T,
    block_a: &CartesianBlock,
    block_b: Option<&CartesianBlock>,
    cell_width: &CellWidth,
    squared_distance_bin_edges: &B, // should this be passed by value?
    pair_op: PairOperation,
) -> Result<(), &'static str> {
    let context = TwoPointCartesian::new(
        reducer.clone(),
        block_a.clone(),
        block_b.cloned(),
        *cell_width,
        squared_distance_bin_edges.clone(),
        pair_op,
    )?;

    // todo: make this customizable
    let mut executor = SerialExecutor;
    executor.drive_reduce(
        binned_statepacks,
        &context,
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(1).unwrap(),
    )?;

    Ok(())
}
