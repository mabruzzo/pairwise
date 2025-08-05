use crate::{Error, parallel_serial::SerialExecutor};
use pairwise_nostd_internal::{
    BinEdges, CartesianBlock, CellWidth, PairOperation, PointProps, Reducer, StatePackViewMut,
    TwoPointCartesian, apply_accum,
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
) -> Result<(), Error> {
    let context = TwoPointCartesian::new(
        reducer.clone(),
        block_a.clone(),
        block_b.cloned(),
        *cell_width,
        squared_distance_bin_edges.clone(),
        pair_op,
    )
    .map_err(Error::internal_legacy_adhoc)?;

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

/*
// we may throw this out, but we are only using it right now for the sake of preserving tests...
pub fn apply_unstructured<T: Reducer + Clone, B: BinEdges + Clone>(
    binned_statepacks: &mut StatePackViewMut,
    reducer: &T,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    squared_distance_bin_edges: &B, // should this be passed by value?
    pair_op: PairOperation,
) -> Result<(), &'static str> {
    apply_accum(
        binned_statepacks,
        reducer,
        points_a,
        points_b,
        squared_distance_bin_edges,
        pairwise_fn
    )?;

    Ok(())
}
*/
