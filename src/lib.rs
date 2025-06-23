use ndarray::ArrayView2;

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

    // apply_accum Tests
    // -----------------
    // most of these tests could be considered integration-tests

    // based on numpy!
    // https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    //
    // I suspect we'll use this a lot! If we may want to define
    // a `assert_isclose!` macro to provide a nice error message (or an
    // `assert_allclose!` macro to operate upon arrays)
    fn _isclose(actual: f64, ref_val: f64, rtol: f64, atol: f64) -> bool {
        let actual_nan = actual.is_nan();
        let ref_nan = ref_val.is_nan();
        if actual_nan || ref_nan {
            actual_nan && ref_nan
        } else {
            (actual - ref_val).abs() <= (atol + rtol * ref_val.abs())
        }
    }

    #[test]
    fn apply_accum_errors() {
        #[rustfmt::skip]
        let positions = [
             6.0,  7.0,
            12.0, 13.0,
            18.0, 19.0
        ];

        #[rustfmt::skip]
        let values = [
            -9., -8.,
            -3., -2.,
             3.,  4.
        ];

        let bin_edges = [10.0, 5.0, 3.0];
        let mut accum = Mean::new(bin_edges.len() - 1).unwrap();
        let points = PointProps::new(
            ArrayView2::from_shape((3, 2), &positions).unwrap(),
            ArrayView2::from_shape((3, 2), &values).unwrap(),
            None,
        )
        .unwrap();

        let result = apply_accum(&mut accum, &points, None, &bin_edges, &diff_norm);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("monotonic"));

        let bin_edges = vec![3.0, 5.0, 0.0];
        let result = apply_accum(&mut accum, &points, None, &bin_edges, &diff_norm);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("monotonic"));

        // TODO: this should be an error
        // let bin_edges = vec![0.0, 3.0, 3.0];

        // should fail for mismatched spatial dimensions
        let points_b = PointProps::new(
            ArrayView2::from_shape((2, 3), &positions).unwrap(),
            ArrayView2::from_shape((2, 3), &values).unwrap(),
            None,
        )
        .unwrap();
        let result = apply_accum(
            &mut accum,
            &points,
            Some(&points_b),
            &[0.0, 3.0, 5.0],
            &diff_norm,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("spatial dimensions"));

        // should fail if 1 points object provides weights an the other doesn't
        let weights = [1.0, 0.0];
        let points_b = PointProps::new(
            ArrayView2::from_shape((3, 2), &positions).unwrap(),
            ArrayView2::from_shape((3, 2), &values).unwrap(),
            Some(&weights),
        )
        .unwrap();
        let result = apply_accum(
            &mut accum,
            &points,
            Some(&points_b),
            &[0.0, 3.0, 5.0],
            &diff_norm,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("weights"));
    }

    #[test]
    fn test_apply_accum_auto() {
        // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        let bin_edges = [2., 6., 10., 15.];

        // check the means (using results computed by pyvsf)
        let expected_mean = [8.41281820819169, 15.01110699893027, f64::NAN];
        let expected_weight = [7., 3., 0.];
        let mut mean_accum = Mean::new(bin_edges.len() - 1).unwrap();
        let points = PointProps::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();
        let result = apply_accum(&mut mean_accum, &points, None, &bin_edges, &diff_norm);
        assert_eq!(result, Ok(()));

        // output buffers
        let mut mean = [0.0; 3];
        let mut weight = [0.0; 3];
        mean_accum.get_value(&mut mean, &mut weight);

        for i in 0..3 {
            // we might need to adopt an actual rtol
            assert!(
                _isclose(mean[i], expected_mean[i], 3.0e-16, 0.0),
                "actual mean = {}, expected mean = {}",
                mean[i],
                expected_mean[i]
            );
            assert_eq!(weight[i], expected_weight[i], "unequal weights");
        }

        // check the histograms (using results computed by pyvsf)

        // the hist_bin_edges were picked such that:
        // - the number of bins is unequal to the distance bin count
        // - there would be a value smaller than the leftmost bin-edge
        // - there would be a value larger than the leftmost bin-edge
        let hist_bin_edges = [6.0, 10.0, 14.0];
        #[rustfmt::skip]
        let expected_hist_weights = [
            4., 3.,
            0., 2.,
            0., 0.
        ];
        let mut hist_accum = Histogram::new(bin_edges.len() - 1, &hist_bin_edges).unwrap();
        let result = apply_accum(&mut hist_accum, &points, None, &bin_edges, &diff_norm);
        assert_eq!(result, Ok(()));
        let mut hist_weights = [0.0; 6];
        hist_accum.get_value(&mut hist_weights);
        for i in 0..6 {
            assert_eq!(hist_weights[i], expected_hist_weights[i]);
        }
    }

    #[test]
    fn test_apply_accum_cross() {
        // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions_a: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values_a: Vec<f64> = (-9..9).map(|x| x as f64).collect();

        let points_a = PointProps::new(
            ArrayView2::from_shape((3, 6), &positions_a).unwrap(),
            ArrayView2::from_shape((3, 6), &values_a).unwrap(),
            None,
        )
        .unwrap();

        // we intentionally padded positions_b with a point
        // that is so far away from everything else that it can't
        // fit inside a separation bin
        #[rustfmt::skip]
        let positions_b = [
            0., 1., 1000.,
            2., 3., 1000.,
            4., 5., 1000.,
        ];

        #[rustfmt::skip]
        let values_b = [
            -3., -2., 1000.,
            -1.,  0., 1000.,
             1.,  2., 1000.,
        ];

        let points_b = PointProps::new(
            ArrayView2::from_shape((3, 3), &positions_b).unwrap(),
            ArrayView2::from_shape((3, 3), &values_b).unwrap(),
            None,
        )
        .unwrap();

        let bin_edges = [17., 21., 25.];

        // the expected results were printed by pyvsf
        let expected_mean = [6.274664681905207, 6.068727871100932];
        let expected_weight = [4., 6.];

        // perform the calculation!
        let mut accum = Mean::new(bin_edges.len() - 1).unwrap();
        let result = apply_accum(
            &mut accum,
            &points_a,
            Some(&points_b),
            &bin_edges,
            &diff_norm,
        );
        assert_eq!(result, Ok(()));

        // output buffers
        let mut mean = [0.0; 2];
        let mut weight = [0.0; 2];
        accum.get_value(&mut mean, &mut weight);

        for i in 0..2 {
            assert!(
                _isclose(mean[i], expected_mean[i], 3.0e-16, 0.0),
                "actual mean = {}, expected mean = {}",
                mean[i],
                expected_mean[i]
            );
            assert_eq!(weight[i], expected_weight[i], "unequal weights");
        }
    }

    #[test]
    fn test_apply_accum_auto_corr() {
        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        let bin_edges = [2., 6., 10., 15.];

        // check the means (using results computed by pyvsf)
        let expected_mean = [284.57142857142856, 236.0, f64::NAN];
        let expected_weight = [7., 3., 0.];
        let mut mean_accum = Mean::new(bin_edges.len() - 1).unwrap();
        let points = PointProps::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();
        let result = apply_accum(&mut mean_accum, &points, None, &bin_edges, &dot_product);
        assert_eq!(result, Ok(()));

        // output buffers
        let mut mean = [0.0; 3];
        let mut weight = [0.0; 3];
        mean_accum.get_value(&mut mean, &mut weight);

        for i in 0..3 {
            // we might need to adopt an actual rtol
            assert!(
                _isclose(mean[i], expected_mean[i], 3.0e-16, 0.0),
                "actual mean = {}, expected mean = {}",
                mean[i],
                expected_mean[i]
            );
            assert_eq!(weight[i], expected_weight[i], "unequal weights");
        }
    }
}
