use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use pairwise_internal::{Accumulator, get_bin_idx, squared_diff_norm};

/// Collection of point properties.
///
/// We place the following constraints on contained arrays:
/// - axis 0 is the slow axis and it corresponds to different components of
///   vector quantities (e.g. position, values).
/// - axis 1 is the fast axis. The length along this axis coincides with
///   the number of points. We require that it is contiguous (i.e. the stride
///   is unity).
/// - In other words the shape of one of these vectors is `(D, n_points)`,
///   where `D` is the number of spatial dimensions and `n_points` is the
///   number of points.
pub struct PointProps<'a> {
    positions: ArrayView2<'a, f64>,
    // TODO allow values to have a difference dimensionality than positions
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

fn apply_accum_helper<const CROSS: bool>(
    stateprops: &mut ArrayViewMut2<f64>,
    accum: &impl Accumulator,
    points_a: &PointProps,
    points_b: &PointProps,
    squared_bin_edges: &[f64],
    pairwise_fn: impl Fn(ArrayView2<f64>, ArrayView2<f64>, usize, usize) -> f64,
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
            if let Some(distance_bin_idx) = get_bin_idx(distance_squared, squared_bin_edges) {
                // get the value. This is hardcoded to correspond to the velocity structure
                // function when accumulated with Mean
                // TODO switch on pairwise op?
                let val = pairwise_fn(points_a.values, points_b.values, i_a, i_b);

                // get the weight
                let pair_weight = points_a.get_weight(i_a) * points_b.get_weight(i_b);

                accum.consume(
                    &mut stateprops.index_axis_mut(Axis(1), distance_bin_idx),
                    val,
                    pair_weight,
                );
            }
        }
    }
}

// maybe we want to make separate functions for auto-stats vs
// cross-stats
// TODO: generalize to allow faster calculations for regular spatial grids
pub fn apply_accum(
    stateprops: &mut ArrayViewMut2<f64>,
    accum: &impl Accumulator,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    // TODO should distance_bin_edges should be a member of the AccumKernel Struct
    distance_bin_edges: &[f64],
    pairwise_fn: &impl Fn(ArrayView2<f64>, ArrayView2<f64>, usize, usize) -> f64,
) -> Result<(), String> {
    // TODO check size of output buffers

    // Check that bin_edges are monotonically increasing
    if !distance_bin_edges.is_sorted() {
        return Err(String::from(
            "bin_edges must be sorted (monotonically increasing)",
        ));
    }

    //  if points_b is not None, make sure a and b have the same number of
    // spatial dimensions
    if let Some(points_b) = points_b {
        if points_a.n_spatial_dims != points_b.n_spatial_dims {
            return Err(String::from(
                "points_a and points_b must have the same number of spatial dimensions",
            ));
        } else if points_a.weights.is_some() != points_b.weights.is_some() {
            return Err(String::from(
                "points_a and points_b must both provide weights or neither \
                should provide weights",
            ));
        }
    }

    // I think this alloc is worth it? Could use a buffer?
    let squared_bin_edges: Vec<f64> = distance_bin_edges.iter().map(|x| x.powi(2)).collect();

    if let Some(points_b) = points_b {
        apply_accum_helper::<false>(
            stateprops,
            accum,
            points_a,
            points_b,
            &squared_bin_edges,
            pairwise_fn,
        )
    } else {
        apply_accum_helper::<true>(
            stateprops,
            accum,
            points_a,
            points_a,
            &squared_bin_edges,
            pairwise_fn,
        )
    }
    Ok(())
}
