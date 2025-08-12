use crate::bins::BinEdges;
use crate::misc::squared_diff_norm;
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
#[cfg_attr(feature = "fmt", derive(Debug))]
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

    #[inline(always)]
    fn outer_team_loop_bounds(
        &self,
        _team_id: usize,
        team_info: &StandardTeamParam,
    ) -> (usize, usize) {
        // require serial implementation TODO
        assert_eq!(team_info.n_members_per_team, 1);
        assert_eq!(team_info.n_teams, 1);

        (0, self.points_a.n_points)
    }

    fn inner_team_loop_bounds(
        &self,
        outer_index: usize,
        _team_id: usize,
        team_info: &StandardTeamParam,
    ) -> (usize, usize) {
        // require serial implementation TODO
        assert_eq!(team_info.n_members_per_team, 1);
        assert_eq!(team_info.n_teams, 1);

        if self.is_auto {
            (outer_index + 1, self.points_a.n_points)
        } else {
            (0, self.points_b.n_points)
        }
    }

    const NESTED_REDUCE: bool = false;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        outer_index: usize,
        inner_index: usize,
        team: &mut T,
    ) {
        match &self.pair_op {
            PairOperation::ElementwiseMultiply => {
                apply_accum_helper::<T, false>(
                    binned_statepack,
                    &self.reducer,
                    outer_index,
                    inner_index,
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
                    outer_index,
                    inner_index,
                    &self.points_a,
                    &self.points_b,
                    &self.squared_distance_bin_edges,
                    team,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_accum_helper<T: Team, const SUBTRACT: bool>(
    binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
    reducer: &impl Reducer,
    outer_index: usize,
    inner_index: usize,
    points_a: &UnstructuredPoints,
    points_b: &UnstructuredPoints,
    squared_distance_bin_edges: &impl BinEdges,
    team: &mut T,
) {
    team.collect_pairs_then_apply(
        binned_statepack,
        reducer,
        &|collect_pad: &mut [BinnedDatum], member_id: usize| {
            assert!(!T::IS_VECTOR_PROCESSOR);

            let i_a = outer_index;
            // this will change when we have more that 1 member per team
            let i_b = inner_index + member_id;

            let distance_squared = squared_diff_norm(
                points_a.positions,
                points_b.positions,
                i_a,
                i_b,
                points_a.n_spatial_dims,
            );

            collect_pad[0] = if let Some(distance_bin_idx) =
                squared_distance_bin_edges.bin_index(distance_squared)
            {
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

                BinnedDatum {
                    bin_index: distance_bin_idx,
                    datum,
                }
            } else {
                BinnedDatum::zeroed()
            }
        },
    );
}
