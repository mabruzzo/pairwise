use crate::bins::BinEdges;
use crate::misc::squared_diff_norm;
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
pub struct PointProps<'a> {
    positions: ArrayView2<'a, f64>,
    // TODO allow values to have a different dimensionality than positions
    values: ArrayView2<'a, f64>,
    weights: Option<&'a [f64]>,
    n_points: usize,
    n_spatial_dims: usize,
}

impl<'a> PointProps<'a> {
    /// create a new instance
    pub fn new(
        positions: ArrayView2<'a, f64>,
        values: ArrayView2<'a, f64>,
        weights: Option<&'a [f64]>,
    ) -> Result<PointProps<'a>, &'static str> {
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
/// - The value contributed by the pair is determined by `pairwise_fn`
/// - The bin the value is contributed to is determined by the distance between
///   the points and the `squared_distance_bin_edges` argument
///
/// Details about the considered statistics are encapsulated by `accum`.
/// Statistical contributions (for all bins) are tracked within `stateprops`.
///
/// TODO: I don't love that we are directly accepting
///       `squared_distance_bin_edges`. Frankly it seems like a recipe for
///       disaster (I myself could imagine forgetting to square things). I
///       think this is Ok while we get everything working, but we definitely
///       should revisit!
pub fn apply_accum(
    statepack: &mut StatePackViewMut,
    reducer: &impl Reducer,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    squared_distance_bin_edges: &impl BinEdges,
    pair_op: PairOperation,
) -> Result<(), &'static str> {
    // maybe we make separate functions for auto-stats vs cross-stats?
    // TODO: check size of output buffers

    //  if points_b is not None, make sure a and b have the same number of
    // spatial dimensions
    if let Some(points_b) = points_b {
        if points_a.n_spatial_dims != points_b.n_spatial_dims {
            return Err("points_a and points_b must have the same number of spatial dimensions");
        } else if points_a.weights.is_some() != points_b.weights.is_some() {
            return Err(
                "points_a and points_b must both provide weights or neither \
                should provide weights",
            );
        }
    }

    if let Some(points_b) = points_b {
        match pair_op {
            PairOperation::ElementwiseMultiply => {
                apply_accum_helper::<false, false>(
                    statepack,
                    reducer,
                    points_a,
                    points_b,
                    squared_distance_bin_edges,
                );
            }
            PairOperation::ElementwiseSub => {
                apply_accum_helper::<false, true>(
                    statepack,
                    reducer,
                    points_a,
                    points_b,
                    squared_distance_bin_edges,
                );
            }
        }
    } else {
        match pair_op {
            PairOperation::ElementwiseMultiply => {
                apply_accum_helper::<true, false>(
                    statepack,
                    reducer,
                    points_a,
                    points_a,
                    squared_distance_bin_edges,
                );
            }
            PairOperation::ElementwiseSub => {
                apply_accum_helper::<true, true>(
                    statepack,
                    reducer,
                    points_a,
                    points_a,
                    squared_distance_bin_edges,
                );
            }
        }
    }
    Ok(())
}

fn apply_accum_helper<const CROSS: bool, const SUBTRACT: bool>(
    statepack: &mut StatePackViewMut,
    reducer: &impl Reducer,
    points_a: &PointProps,
    points_b: &PointProps,
    squared_distance_bin_edges: &impl BinEdges,
) {
    for i_a in 0..points_a.n_points {
        let i_b_start = if CROSS { i_a + 1 } else { 0 };
        for i_b in i_b_start..points_b.n_points {
            // compute the distance between the points, then the distance bin
            let distance_squared = squared_diff_norm(
                points_a.positions,
                points_b.positions,
                i_a,
                i_b,
                points_a.n_spatial_dims,
            );
            if let Some(distance_bin_idx) = squared_distance_bin_edges.bin_index(distance_squared) {
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

                reducer.consume(&mut statepack.get_state_mut(distance_bin_idx), &datum);
            }
        }
    }
}
