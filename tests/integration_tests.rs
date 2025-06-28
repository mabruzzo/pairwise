use ndarray::ArrayView2;
use pairwise::{Histogram, Mean, PointProps, apply_accum, diff_norm, dot_product};

#[cfg(test)]
mod tests {
    use super::*;

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
