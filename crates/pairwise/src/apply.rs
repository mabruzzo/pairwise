use crate::{Error, parallel_serial::SerialExecutor};
use pairwise_nostd_internal::{
    BinEdges, CartesianBlock, CellWidth, PairOperation, Reducer, StatePackViewMut,
    TwoPointCartesian, TwoPointUnstructured, UnstructuredPoints,
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

/// Apply a function to each pair of points and accumulate the results.
/// This wraps TwoPoints, and exists to streamline existing tests
/// TODO this should be deleted eventually
pub fn apply_accum<'a, R: Reducer + Clone, B: BinEdges + Clone>(
    statepack: &mut StatePackViewMut,
    reducer: &R,
    points_a: &UnstructuredPoints<'a>,
    points_b: Option<&UnstructuredPoints<'a>>,
    squared_distance_bin_edges: &B,
    pair_op: PairOperation,
) -> Result<(), Error> {
    let twopoint = TwoPointUnstructured::new(
        reducer.clone(),
        points_a.clone(),
        points_b.cloned(),
        squared_distance_bin_edges.clone(),
        pair_op,
    )
    .map_err(Error::internal_legacy_adhoc)?;

    let one = NonZeroU32::new(1).unwrap();
    SerialExecutor {}.drive_reduce(statepack, &twopoint, one, one)?;

    Ok(())
}
