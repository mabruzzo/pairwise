use crate::{Error, TwoPoint, parallel_serial::SerialExecutor};
use pairwise_nostd_internal::{BinEdges, PairOperation, PointProps, Reducer, StatePackViewMut};
use std::num::NonZeroU32;

/// Apply a function to each pair of points and accumulate the results.
/// This wraps TwoPoints, and exists to streamline existing tests
/// TODO this should be deleted eventually
pub fn apply_accum<'a, R: Reducer + Clone, B: BinEdges + Clone>(
    statepack: &mut StatePackViewMut,
    reducer: &R,
    points_a: &PointProps<'a>,
    points_b: Option<&PointProps<'a>>,
    squared_distance_bin_edges: &B,
    pair_op: PairOperation,
) -> Result<(), Error> {
    let twopoint = TwoPoint::new(
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
