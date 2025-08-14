use crate::bins::BinEdges;
use crate::misc::{View2DUnsignedSpec, segment_idx_bounds};
use crate::parallel::{BinnedDatum, ReductionSpec, StandardTeamParam, Team};
use crate::reducer::{Datum, Reducer};
use crate::state::StatePackViewMut;
use crate::twopoint::common::PairOperation;
use ndarray::ArrayView2;

/// calculate the squared euclidean norm for two (mathematical) vectors
fn squared_diff_norm(v1_i: f64, v1_j: f64, v1_k: f64, v2_i: f64, v2_j: f64, v2_k: f64) -> f64 {
    let delta_v_i = v2_i - v1_i;
    let delta_v_j = v2_j - v1_j;
    let delta_v_k = v2_k - v1_k;
    delta_v_i * delta_v_i + (delta_v_j * delta_v_j + delta_v_k * delta_v_k)
}

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
    positions: &'a [f64],
    values: &'a [f64],
    weights: &'a [f64],
    /// specifies layout of positions and values
    ///
    /// be aware that `[n_points, n_spatial_dims] = self.idx_2d_spec.clone()`
    idx_2d_spec: View2DUnsignedSpec,
}

impl<'a> UnstructuredPoints<'a> {
    // todo: delete me! This is a temporary stop gap to make the transition
    //       easier
    pub fn from_contiguous(
        positions: &'a ArrayView2<'a, f64>,
        values: &'a ArrayView2<'a, f64>,
        weights: &'a [f64],
    ) -> Result<UnstructuredPoints<'a>, &'static str> {
        let n_spatial_dims = positions.shape()[0];
        let n_points = positions.shape()[1];

        if n_spatial_dims != values.shape()[0] {
            Err("n_spatial_dims is inconsistent")
        } else if n_points != values.shape()[1] {
            Err("n_points is inconsistent")
        } else {
            let Some(pos): Option<&'a [f64]> = positions.as_slice() else {
                return Err("positions must be contiguous & in standard order");
            };
            let Some(vals): Option<&'a [f64]> = values.as_slice() else {
                return Err("values must be contiguous & in standard order");
            };
            Self::new(pos, vals, weights, n_points, None)
        }
    }

    /// create a new instance
    pub fn new(
        positions: &'a [f64],
        values: &'a [f64],
        weights: &'a [f64],
        n_points: usize,
        spatial_dim_stride: Option<usize>,
    ) -> Result<UnstructuredPoints<'a>, &'static str> {
        let spatial_dim_stride = spatial_dim_stride.unwrap_or(n_points);

        if n_points == 0 {
            return Err("n_points must be positive");
        } else if n_points > spatial_dim_stride {
            return Err("n_points exceeds spatial_dim_stride");
        } else if values.len() != positions.len() {
            return Err("values and positions have different lengths");
        }

        let n_spatial_dims = positions.len() / spatial_dim_stride;

        if positions.len() % spatial_dim_stride != 0 {
            Err("positions.len() isn't divisible by spatial_dim_stride")
        } else if n_spatial_dims != 3 {
            // we can probably relax this in the future
            Err("for now, we require 3 spatial dimensions")
        } else if weights.len() != n_points {
            Err("weights.len() must be n_points")
        } else {
            Ok(Self {
                positions,
                values,
                weights,
                idx_2d_spec: View2DUnsignedSpec::from_shape_strides(
                    [n_spatial_dims, n_points],
                    [spatial_dim_stride, 1],
                )?,
            })
        }
    }

    #[inline]
    fn n_spatial_dims(&self) -> usize {
        self.idx_2d_spec.shape()[0]
    }

    #[inline]
    fn n_points(&self) -> usize {
        self.idx_2d_spec.shape()[1]
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
            if points_a.n_spatial_dims() != points_b.n_spatial_dims() {
                return Err(
                    "points_a and points_b must have the same number of spatial dimensions",
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
    fn team_loop_bounds(&self, team_id: usize, team_info: &StandardTeamParam) -> (usize, usize) {
        segment_idx_bounds(self.points_a.n_points(), team_id, team_info.n_teams)
    }

    const NESTED_REDUCE: bool = false;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        team_loop_idx: usize,
        team: &mut T,
    ) {
        // the current way we divide up the work is terribly uneven when
        // `self.is_auto` is true. See the partitioning strategy implemented
        // in libvsf, within the pairstat python package for a better approach

        let i_a = team_loop_idx;
        let (i_b_start, i_b_stop) = if self.is_auto {
            (i_a + 1, self.points_a.n_points())
        } else {
            (0, self.points_b.n_points())
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
    idx_a: usize,
    idx_b_start: usize,
    idx_b_stop: usize,
    points_a: &UnstructuredPoints,
    points_b: &UnstructuredPoints,
    squared_distance_bin_edges: &impl BinEdges,
    team: &mut T,
) {
    let step = team.standard_team_info().n_members_per_team;

    debug_assert_eq!(points_a.n_spatial_dims(), 3);
    let idx_a0 = points_a.idx_2d_spec.map_idx2d_to_1d(0, idx_a);
    let idx_a1 = points_a.idx_2d_spec.map_idx2d_to_1d(1, idx_a);
    let idx_a2 = points_a.idx_2d_spec.map_idx2d_to_1d(2, idx_a);

    // TODO confirm that step_by doesn't trip up the GPU (maybe compare to while-loop)
    for nominal_idx_b in (idx_b_start..idx_b_stop).step_by(step) {
        team.collect_pairs_then_apply(
            binned_statepack,
            reducer,
            &|collect_pad: &mut [BinnedDatum], member_id: usize| {
                assert!(!T::IS_VECTOR_PROCESSOR);
                let idx_b = nominal_idx_b + member_id;
                let idx_b0 = points_b.idx_2d_spec.map_idx2d_to_1d(0, idx_b);
                let idx_b1 = points_b.idx_2d_spec.map_idx2d_to_1d(1, idx_b);
                let idx_b2 = points_b.idx_2d_spec.map_idx2d_to_1d(2, idx_b);

                // calculate the bin-index associated with i_b (if any)
                // - I'm pretty confident we can write a branch-free version
                let maybe_bin_index = if idx_b >= idx_b_stop {
                    None // can only come up if multiple members per team
                } else {
                    let distance_squared = squared_diff_norm(
                        points_a.positions[idx_a0],
                        points_a.positions[idx_a1],
                        points_a.positions[idx_a2],
                        points_b.positions[idx_b0],
                        points_b.positions[idx_b1],
                        points_b.positions[idx_b2],
                    );
                    squared_distance_bin_edges.bin_index(distance_squared)
                };

                // now we write a value into collect_pad
                collect_pad[0] = if let Some(bin_index) = maybe_bin_index {
                    let datum = Datum {
                        value: if SUBTRACT {
                            [
                                points_b.values[idx_b0] - points_a.values[idx_a0],
                                points_b.values[idx_b1] - points_a.values[idx_a1],
                                points_b.values[idx_b2] - points_a.values[idx_a2],
                            ]
                        } else {
                            [
                                points_b.values[idx_b0] * points_a.values[idx_a0],
                                points_b.values[idx_b1] * points_a.values[idx_a1],
                                points_b.values[idx_b2] * points_a.values[idx_a2],
                            ]
                        },
                        weight: points_a.weights[idx_a] * points_b.weights[idx_b],
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
