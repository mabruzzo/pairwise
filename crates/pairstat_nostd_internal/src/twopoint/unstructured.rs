use crate::bins::BinEdges;
use crate::misc::{segment_idx_bounds, squared_diff_norm};
use crate::parallel::{BinnedDatum, ReductionSpec, StandardTeamParam, Team};
use crate::reducer::{Datum, Reducer};
use crate::state::StatePackViewMut;
use crate::twopoint::common::PairOperation;
use ndarray::ArrayView2;

/// Collection of point properties.
///
/// We place the following constraints on contained arrays:
/// - axis 0 is the slow axis and it corresponds to different components of
///   vector quantities (e.g. position, values).
/// - axis 1 is the fast axis. The length along this axis coincides with
///   the number of points. We require that it is contiguous (i.e. the stride
///   is unity).
/// - In other words the shape of one of these arrays is `(D, n_points)`,
///   where `D` is the number of spatial dimensions and `n_points` is the
///   number of points.
#[derive(Clone)]
pub struct UnstructuredPoints<'a> {
    positions: ArrayView2<'a, f64>,
    // TODO allow values to have a different dimensionality than positions
    values: ArrayView2<'a, f64>,
    weights: Option<&'a [f64]>,
    n_points: usize,
    n_spatial_dims: usize,
}

impl<'a> UnstructuredPoints<'a> {
    /// create a new instance
    pub fn new(
        positions: ArrayView2<'a, f64>,
        values: ArrayView2<'a, f64>,
        weights: Option<&'a [f64]>,
    ) -> Result<UnstructuredPoints<'a>, &'static str> {
        let n_spatial_dims = positions.shape()[0];
        let n_points = positions.shape()[1];
        // TODO: should we place a requirement on the number of spatial_dims?
        if positions.is_empty() {
            Err("positions must hold at least n_spatial_dims")
        } else if positions.strides()[1] != 1 {
            Err("positions must be contiguous along the fast axis")
        } else if values.shape()[0] != n_spatial_dims {
            // TODO: in the future, we will allow values to be 1D (i.e. a scalar)
            Err("values must currently have the same number of spatial \
                dimensions as positions")
        } else if values.shape()[1] != n_points {
            Err("values must have the same number of points as positions")
        } else if weights.is_some_and(|w| w.len() != n_points) {
            Err("weights must have the same number of points as positions")
        } else {
            Ok(Self {
                positions,
                values,
                weights,
                n_points,
                n_spatial_dims,
            })
        }
    }

    /// If no weights are provided, returns 1.0, i.e., weights are just counts.
    pub fn get_weight(&self, idx: usize) -> f64 {
        if let Some(weights) = self.weights {
            weights[idx]
        } else {
            1.0
        }
    }
}

/// TODO
/// Computes contributions to binned statistics from values computed from the
/// specified pairs of points.
///
/// In more detail, `statepack`, acts as a container of accumulator state. It
/// holds an accum_state per bin.
///
/// When `points_b` is `None`, the function considers all unique pairs of
/// points within `points_a`. Otherwise, all pairs of points between `points_a`
/// and `points_b` are considered.
///
/// For each pair of points:
/// - The value contributed by the pair is determined by `pairstat_fn`
/// - The bin the value is contributed to is determined by the distance between
///   the points and the `squared_distance_bin_edges` argument
pub struct TwoPointUnstructured<'a, R: Reducer, B: BinEdges> {
    reducer: R,
    points_a: UnstructuredPoints<'a>,
    points_b: UnstructuredPoints<'a>,
    is_auto: bool, // true when points_a is the same as points_b
    squared_distance_bin_edges: B,
    pair_op: PairOperation,
}

impl<'a, R: Reducer, B: BinEdges> TwoPointUnstructured<'a, R, B> {
    // TODO starting with squared distance bins is a major footgun.
    pub fn new(
        reducer: R,
        points_a: UnstructuredPoints<'a>,
        points_b: Option<UnstructuredPoints<'a>>,
        squared_distance_bin_edges: B,
        pair_op: PairOperation,
    ) -> Result<Self, &'static str> {
        //  if points_b is not None, make sure a and b have the same number of
        // spatial dimensions
        if let Some(points_b) = points_b {
            if points_a.n_spatial_dims != points_b.n_spatial_dims {
                return Err(
                    "points_a and points_b must have the same number of spatial dimensions",
                );
            } else if points_a.weights.is_some() != points_b.weights.is_some() {
                return Err(
                    "points_a and points_b must both provide weights or neither \
                    should provide weights",
                );
            }

            Ok(Self {
                reducer,
                points_a,
                points_b,
                is_auto: false,
                squared_distance_bin_edges,
                pair_op,
            })
        } else {
            Ok(Self {
                reducer,
                points_a: points_a.clone(),
                points_b: points_a,
                is_auto: true,
                squared_distance_bin_edges,
                pair_op,
            })
        }
    }
}

impl<'a, R: Reducer, B: BinEdges> ReductionSpec for TwoPointUnstructured<'a, R, B> {
    type ReducerType = R;
    /// return a reference to the reducer
    fn get_reducer(&self) -> &Self::ReducerType {
        &self.reducer
    }

    /// The number of bins in this reduction.
    fn n_bins(&self) -> usize {
        self.squared_distance_bin_edges.n_bins()
    }

    // we could actually eliminate this method if we really want to
    fn inner_team_loop_bounds(
        &self,
        team_id: usize,
        team_info: &StandardTeamParam,
    ) -> (usize, usize) {
        segment_idx_bounds(self.points_a.n_points, team_id, team_info.n_teams)
    }

    const NESTED_REDUCE: bool = false;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        inner_index: usize,
        team: &mut T,
    ) {
        // the current way we divide up the work is terribly uneven when
        // `self.is_auto` is true. See the partitioning strategy implemented
        // in libvsf, within the pairstat python package for a better approach

        let i_a = inner_index;
        let (i_b_start, i_b_stop) = if self.is_auto {
            (i_a + 1, self.points_a.n_points)
        } else {
            (0, self.points_b.n_points)
        };

        match &self.pair_op {
            PairOperation::ElementwiseMultiply => {
                apply_accum_helper::<T, false>(
                    binned_statepack,
                    &self.reducer,
                    i_a,
                    i_b_start,
                    i_b_stop,
                    &self.points_a,
                    &self.points_b,
                    &self.squared_distance_bin_edges,
                    team,
                );
            }
            PairOperation::ElementwiseSub => {
                apply_accum_helper::<T, true>(
                    binned_statepack,
                    &self.reducer,
                    i_a,
                    i_b_start,
                    i_b_stop,
                    &self.points_a,
                    &self.points_b,
                    &self.squared_distance_bin_edges,
                    team,
                );
            }
        }
    }
}

/// Updates `binned_statepack` with contributions from pairs of values.
///
/// In each pair, one value comes from `points_a` and the other comes from
/// `points_b`.
///
/// ## Current Implementation
/// Under the current implementation, this considers the index-pairs:
/// - `(i_a, i_b_start)`
/// - `(i_a, i_b_start+1)`
/// - `(i_a, ...)`
/// - `(i_a, i_b_stop-2)`
/// - `(i_a, i_b_stop-1)`
#[allow(clippy::too_many_arguments)]
fn apply_accum_helper<T: Team, const SUBTRACT: bool>(
    binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
    reducer: &impl Reducer,
    i_a: usize,
    i_b_start: usize,
    i_b_stop: usize,
    points_a: &UnstructuredPoints,
    points_b: &UnstructuredPoints,
    squared_distance_bin_edges: &impl BinEdges,
    team: &mut T,
) {
    let step = team.standard_team_info().n_members_per_team;
    // TODO confirm that step_by doesn't trip up the GPU (maybe compare to while-loop)
    for nominal_i_b in (i_b_start..i_b_stop).step_by(step) {
        team.collect_pairs_then_apply(
            binned_statepack,
            reducer,
            &|collect_pad: &mut [BinnedDatum], member_id: usize| {
                assert!(!T::IS_VECTOR_PROCESSOR);
                let i_b = nominal_i_b + member_id;

                // calculate the bin-index associated with i_b (if any)
                // - I'm pretty confident we can write a branch-free version
                let maybe_bin_index = if i_b >= i_b_stop {
                    None // can only come up if multiple members per team
                } else {
                    let distance_squared = squared_diff_norm(
                        points_a.positions,
                        points_b.positions,
                        i_a,
                        i_b,
                        points_a.n_spatial_dims,
                    );
                    squared_distance_bin_edges.bin_index(distance_squared)
                };

                // now we write a value into collect_pad
                collect_pad[0] = if let Some(bin_index) = maybe_bin_index {
                    let datum = Datum {
                        value: if SUBTRACT {
                            [
                                points_b.values[[0, i_b]] - points_a.values[[0, i_a]],
                                points_b.values[[1, i_b]] - points_a.values[[1, i_a]],
                                points_b.values[[2, i_b]] - points_a.values[[2, i_a]],
                            ]
                        } else {
                            [
                                points_b.values[[0, i_b]] * points_a.values[[0, i_a]],
                                points_b.values[[1, i_b]] * points_a.values[[1, i_a]],
                                points_b.values[[2, i_b]] * points_a.values[[2, i_a]],
                            ]
                        },
                        weight: points_a.get_weight(i_a) * points_b.get_weight(i_b),
                    };

                    BinnedDatum { bin_index, datum }
                } else {
                    // the fact BinnedDatum::Datum::weight has a value of 0
                    // means that this has no impact on the output result
                    BinnedDatum::zeroed()
                }
            },
        );
    }
}
