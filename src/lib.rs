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

    pub fn consume(&mut self, val: f64, weight: f64, bin_idx: usize) {
        self.weight[bin_idx] += weight;
        self.total[bin_idx] += val * weight;
    }

    pub fn get_value(&self, out: &mut [f64], weights_out: &mut [f64]) {
        for i in 0..self.weight.len() {
            // TODO need to think about divide by 0
            out[i] = self.total[i] / self.weight[i];

            // is this the most efficient way to do this?
            weights_out[i] = self.weight[i];
        }
    }

    pub fn merge(&mut self, other: &Mean) {
        for i in 0..self.weight.len() {
            self.total[i] += other.total[i];
            self.weight[i] += other.weight[i];
        }
    }
}

/// assumes that different vector-components are indexed along the slow axis.
/// In other words, the kth component of the ith point (for positions &
/// values) is located at an index of `i + k*spatial_dim_stride`
pub struct PointProps<'a> {
    positions: &'a [f64],
    // TODO allow values to have a difference dimensionality than positions
    values: &'a [f64],
    weights: Option<&'a [f64]>,
    n_points: usize,
    // we could potentially handle this separately
    n_spatial_dims: usize,
}

impl<'a> PointProps<'a> {
    pub fn new(
        positions: &'a [f64],
        values: &'a [f64],
        weights: Option<&'a [f64]>,
        n_spatial_dims: usize,
    ) -> Result<PointProps<'a>, &'static str> {
        if positions.is_empty() {
            return Err("positions must hold at least n_spatial_dims");
        } else if positions.len() % n_spatial_dims != 0 {
            return Err("the length of positions must be an integer multiple of n_spatial_dims");
        }
        let n_points = positions.len() / n_spatial_dims;

        if weights.is_some_and(|w| w.len() != n_points) {
            Err("weights must be have the same number of points as positions")
        } else if values.len() != n_points && values.len() != positions.len() {
            // assumes vector or scalar values
            Err("values must be have the same number of points as positions")
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
fn get_distance_bin(distance_squared: f64, squared_bin_edges: &[f64]) -> Option<usize> {
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
    v1: &[f64],
    v2: &[f64],
    i1: usize,
    i2: usize,
    spatial_dim_stride_1: usize,
    spatial_dim_stride_2: usize,
    n_spatial_dims: usize,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..n_spatial_dims {
        let idx1 = i1 + k * spatial_dim_stride_1;
        let idx2 = i2 + k * spatial_dim_stride_2;
        sum += (v1[idx1] - v2[idx2]).powi(2);
    }
    sum
}

fn apply_accum_helper<const CROSS: bool, F>(
    accum: &mut Mean,
    points_a: &PointProps,
    points_b: &PointProps,
    squared_bin_edges: &[f64],
    pairwise_fn: F,
) where
    // using the trait Fn instead of just taking a closure lets the comiler specialize
    F: Fn(&PointProps, &PointProps, usize, usize) -> f64,
{
    for i_a in 0..points_a.n_points {
        let i_b_start = if CROSS { i_a + 1 } else { 0 };
        for i_b in i_b_start..points_b.n_points {
            // compute the distance between the points, then the distance bin
            let distance_squared = squared_diff_norm(
                points_a.positions,
                points_b.positions,
                i_a,
                i_b,
                points_a.n_points,
                points_b.n_points,
                points_a.n_spatial_dims,
            );
            if let Some(distance_bin_idx) = get_distance_bin(distance_squared, squared_bin_edges) {
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
    accum: &mut Mean,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    // TODO should distance_bin_edges should be a member of the AccumKernel Struct
    distance_bin_edges: &[f64],
    /* pairwise_op, */ // probably an Enum
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

    let pairwise_fn =
        |points_a: &PointProps, points_b: &PointProps, i_a: usize, i_b: usize| -> f64 {
            squared_diff_norm(
                points_a.values,
                points_b.values,
                i_a,
                i_b,
                points_a.n_points,
                points_b.n_points,
                points_a.n_spatial_dims,
            )
            .sqrt()
        };
    if let Some(points_b) = points_b {
        apply_accum_helper::<false, _>(accum, points_a, points_b, &squared_bin_edges, &pairwise_fn)
    } else {
        apply_accum_helper::<true, _>(accum, points_a, points_a, &squared_bin_edges, &pairwise_fn)
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
    fn accum_0len() {
        assert!(Mean::new(0).is_err());
    }

    #[test]
    fn consume_once() {
        let mut accum = Mean::new(1).unwrap();
        accum.consume(4.0, 1.0, 0_usize);

        let mut mean_vec = vec![0.0];
        let mut weight_vec = vec![0.0];
        accum.get_value(&mut mean_vec, &mut weight_vec);

        assert_eq!(mean_vec[0], 4.0);
        assert_eq!(weight_vec[0], 1.0);
    }

    #[test]
    fn consume_twice() {
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

    // apply_accum Tests
    // -----------------

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
        let points = PointProps::new(&positions, &values, None, 3_usize).unwrap();

        let result = apply_accum(&mut accum, &points, None, &bin_edges);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("monotonic"));

        let bin_edges = vec![3.0, 5.0, 0.0];
        let result = apply_accum(&mut accum, &points, None, &bin_edges);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("monotonic"));

        // TODO: this should be an error
        // let bin_edges = vec![0.0, 3.0, 3.0];

        // should fail for mismatched spatial dimensions
        let points_b = PointProps::new(&positions, &values, None, 2_usize).unwrap();
        let result = apply_accum(&mut accum, &points, Some(&points_b), &[0.0, 3.0, 5.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("spatial dimensions"));

        // should fail if 1 points object provides weights an the other doesn't
        let weights = [1.0, 0.0];
        let points_b = PointProps::new(&positions, &values, Some(&weights), 3_usize).unwrap();
        let result = apply_accum(&mut accum, &points, Some(&points_b), &[0.0, 3.0, 5.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("weights"));
    }

    #[test]
    fn test_apply_accum_auto() {
        // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values: Vec<f64> = (-9..9).map(|x| x as f64).collect();

        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        let bin_edges = [2., 6., 10., 15.];

        // the expected results were printed by pyvsf
        let expected_mean = [4.206409104095845, 7.505553499465134, f64::NAN];
        let expected_weight = [7., 3., 0.];

        // perform the calculation!
        let mut accum = Mean::new(bin_edges.len() - 1).unwrap();
        let points = PointProps::new(&positions, &values, None, 3_usize).unwrap();
        let result = apply_accum(&mut accum, &points, None, &bin_edges);
        assert_eq!(result, Ok(()));

        // output buffers
        let mut mean = [0.0; 3];
        let mut weight = [0.0; 3];
        accum.get_value(&mut mean, &mut weight);

        for i in 0..3 {
            // we might need to adopt an actual rtol
            assert!(
                _isclose(mean[i], expected_mean[i], 0.0, 0.0),
                "actual mean = {}, expected mean = {}",
                mean[i],
                expected_mean[i]
            );
            assert_eq!(weight[i], expected_weight[i], "unequal weights");
        }
    }

    #[test]
    fn test_apply_accum_cross() {
        // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions_a: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values_a: Vec<f64> = (-9..9).map(|x| x as f64).collect();

        let points_a = PointProps::new(&positions_a, &values_a, None, 3_usize).unwrap();

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

        let points_b = PointProps::new(&positions_b, &values_b, None, 3_usize).unwrap();

        let bin_edges = [17., 21., 25.];

        // the expected results were printed by pyvsf
        let expected_mean = [6.274664681905207, 6.068727871100932];
        let expected_weight = [4., 6.];

        // perform the calculation!
        let mut accum = Mean::new(bin_edges.len() - 1).unwrap();
        let result = apply_accum(&mut accum, &points_a, Some(&points_b), &bin_edges);
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
}
