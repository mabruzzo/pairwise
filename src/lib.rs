use ndarray::ArrayView2;

// below we declare that vecpack is part of the package (but we don't
// currently use any features from it)
mod cartesian_block;
mod vecpack; // we need to use `mod` the first time we try to import

pub trait Accumulator {
    fn consume(&mut self, val: f64, weight: f64, bin_idx: usize);
    fn merge(&mut self, other: &Self);
}

pub struct Mean {
    weight: Vec<f64>,
    total: Vec<f64>,
}

impl Mean {
    /// Create a Mean instance with length `n` registers
    pub fn new(n: usize) -> Result<Mean, &'static str> {
        if n == 0 {
            Err("n can't be zero")
        } else {
            Ok(Mean {
                weight: vec![0.0; n],
                total: vec![0.0; n],
            })
        }
    }

    pub fn get_value(&self, out: &mut [f64], weights_out: &mut [f64]) {
        for i in 0..self.weight.len() {
            // TODO need to think about divide by 0
            out[i] = self.total[i] / self.weight[i];

            // is this the most efficient way to do this?
            weights_out[i] = self.weight[i];
        }
    }
}

impl Accumulator for Mean {
    fn consume(&mut self, val: f64, weight: f64, bin_idx: usize) {
        self.weight[bin_idx] += weight;
        self.total[bin_idx] += val * weight;
    }

    fn merge(&mut self, other: &Mean) {
        for i in 0..self.weight.len() {
            self.total[i] += other.total[i];
            self.weight[i] += other.weight[i];
        }
    }
}

pub struct Histogram {
    hist_bin_edges: Vec<f64>,
    n_hist_bins: usize,
    // histogram bins are on the fast axis, distance bins are on the slow axis
    weight: Vec<f64>,
}

impl Histogram {
    pub fn new(n_distance_bins: usize, hist_bin_edges: &[f64]) -> Result<Histogram, &'static str> {
        if n_distance_bins == 0 {
            Err("n_distance_bins can't be zero")
        } else if !hist_bin_edges.is_sorted() {
            return Err("hist_bin_edges must be sorted (monotonically increasing)");
        } else {
            let n_hist_bins = hist_bin_edges.len() - 1;
            Ok(Histogram {
                weight: vec![0.0; n_distance_bins * n_hist_bins],
                hist_bin_edges: hist_bin_edges.to_vec(),
                n_hist_bins,
            })
        }
    }

    pub fn get_value(&self, out: &mut [f64]) {
        out.copy_from_slice(&self.weight);
    }
}

impl Accumulator for Histogram {
    fn consume(&mut self, val: f64, weight: f64, dist_bin_idx: usize) {
        if let Some(hist_bin_idx) = get_bin_idx(val, &self.hist_bin_edges) {
            self.weight[dist_bin_idx * self.n_hist_bins + hist_bin_idx] += weight;
        }
    }

    fn merge(&mut self, other: &Self) {
        for i in 0..self.weight.len() {
            self.weight[i] += other.weight[i];
        }
    }
}

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

// TODO use binary search, and have specialized version for regularly spaced bins?
/// Get the index of the bin that the squared distance falls into.
/// Returns None if its out of bounds.
fn get_bin_idx(distance_squared: f64, squared_bin_edges: &[f64]) -> Option<usize> {
    // index of first element greater than distance_squared
    // (or squared_bin_edges.len() if none are greater)
    let mut first_greater = 0;
    for &edge in squared_bin_edges.iter() {
        if distance_squared < edge {
            break;
        }
        first_greater += 1;
    }
    if (first_greater == squared_bin_edges.len()) || (first_greater == 0) {
        None
    } else {
        Some(first_greater - 1)
    }
}

/// calculate the squared norm of the difference between two (mathematical) vectors
/// which are part of rust vecs that encodes a list of vectors with dimension on
/// the "slow axis"
fn squared_diff_norm(
    v1: ArrayView2<f64>,
    v2: ArrayView2<f64>,
    i1: usize,
    i2: usize,
    n_spatial_dims: usize,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..n_spatial_dims {
        sum += (v1[[k, i1]] - v2[[k, i2]]).powi(2);
    }
    sum
}

// TODO come up with a better name
pub fn diff_norm(points_a: &PointProps, points_b: &PointProps, i_a: usize, i_b: usize) -> f64 {
    squared_diff_norm(
        points_a.values,
        points_b.values,
        i_a,
        i_b,
        points_a.n_spatial_dims,
    )
    .sqrt()
}

pub fn dot_product(points_a: &PointProps, points_b: &PointProps, i_a: usize, i_b: usize) -> f64 {
    let mut sum = 0.0;
    for k in 0..points_a.n_spatial_dims {
        sum += points_a.values[[k, i_a]] * points_b.values[[k, i_b]];
    }
    sum
}

fn apply_accum_helper<const CROSS: bool>(
    accum: &mut impl Accumulator,
    points_a: &PointProps,
    points_b: &PointProps,
    squared_bin_edges: &[f64],
    pairwise_fn: impl Fn(&PointProps, &PointProps, usize, usize) -> f64,
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
                let val = pairwise_fn(points_a, points_b, i_a, i_b);

                // get the weight
                let pair_weight = points_a.get_weight(i_a) * points_b.get_weight(i_b);

                accum.consume(val, pair_weight, distance_bin_idx);
            }
        }
    }
}

// maybe we want to make separate functions for auto-stats vs
// cross-stats
// TODO: generalize to allow faster calculations for regular spatial grids
pub fn apply_accum(
    accum: &mut impl Accumulator,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    // TODO should distance_bin_edges should be a member of the AccumKernel Struct
    distance_bin_edges: &[f64],
    pairwise_fn: &impl Fn(&PointProps, &PointProps, usize, usize) -> f64,
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
        apply_accum_helper::<false>(accum, points_a, points_b, &squared_bin_edges, pairwise_fn)
    } else {
        apply_accum_helper::<true>(accum, points_a, points_a, &squared_bin_edges, pairwise_fn)
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // eventually, we will break up the functionality between files and we
    // will distribute tests appropriately

    // Accumulator Tests
    // -----------------

    #[test]
    fn mean_0len() {
        assert!(Mean::new(0).is_err());
    }

    #[test]
    fn mean_consume_once() {
        let mut accum = Mean::new(1).unwrap();
        accum.consume(4.0, 1.0, 0_usize);

        let mut mean_vec = vec![0.0];
        let mut weight_vec = vec![0.0];
        accum.get_value(&mut mean_vec, &mut weight_vec);

        assert_eq!(mean_vec[0], 4.0);
        assert_eq!(weight_vec[0], 1.0);
    }

    #[test]
    fn mean_consume_twice() {
        let mut accum = Mean::new(1).unwrap();
        accum.consume(4.0, 1.0, 0);
        accum.consume(8.0, 1.0, 0);

        let mut mean_vec = vec![0.0];
        let mut weight_vec = vec![0.0];
        accum.get_value(&mut mean_vec, &mut weight_vec);
        assert_eq!(mean_vec[0], 6.0);
        assert_eq!(weight_vec[0], 2.0);
    }

    #[test]
    fn merge() {
        let mut accum = Mean::new(1).unwrap();
        accum.consume(4.0, 1.0, 0);
        accum.consume(8.0, 1.0, 0);

        let mut accum_other = Mean::new(1).unwrap();
        accum_other.consume(1.0, 1.0, 0);
        accum_other.consume(3.0, 1.0, 0);
        accum.merge(&accum_other);

        let mut mean_vec = vec![0.0];
        let mut weight_vec = vec![0.0];
        accum.get_value(&mut mean_vec, &mut weight_vec);
        assert_eq!(mean_vec[0], 4.0);
        assert_eq!(weight_vec[0], 4.0);
    }

    #[test]
    fn hist_0len() {
        assert!(Histogram::new(0, &[0.0, 1.0]).is_err());
    }

    #[test]
    fn hist_nonmonotonic() {
        assert!(Histogram::new(3, &[1.0, 0.0]).is_err());
    }
}
