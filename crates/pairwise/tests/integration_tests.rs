mod common;

use common::prepare_statepack;
use ndarray::ArrayView2;
use pairwise::{
    ComponentSumMean, EuclideanNormHistogram, EuclideanNormMean, PairOperation, PointProps,
    StatePackViewMut, apply_accum, get_output_from_statepack_array,
};

// Things are a little unergonomic!

#[cfg(test)]
mod tests {
    use pairwise_nostd_internal::IrregularBinEdges;

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

        let squared_distance_bin_edges = [0.0, 1.0, 9.0, 16.0];
        let squared_distance_bins = IrregularBinEdges::new(&squared_distance_bin_edges).unwrap();
        let reducer = EuclideanNormMean::new();
        let mut statepack = prepare_statepack(squared_distance_bin_edges.len(), &reducer);
        let points = PointProps::new(
            ArrayView2::from_shape((3, 2), &positions).unwrap(),
            ArrayView2::from_shape((3, 2), &values).unwrap(),
            None,
        )
        .unwrap();

        // should fail for mismatched spatial dimensions
        let points_b = PointProps::new(
            ArrayView2::from_shape((2, 3), &positions).unwrap(),
            ArrayView2::from_shape((2, 3), &values).unwrap(),
            None,
        )
        .unwrap();
        let result = apply_accum(
            &mut StatePackViewMut::from_array_view(statepack.view_mut()),
            &reducer,
            &points,
            Some(&points_b),
            &squared_distance_bins,
            PairOperation::ElementwiseSub,
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
            &mut StatePackViewMut::from_array_view(statepack.view_mut()),
            &reducer,
            &points,
            Some(&points_b),
            &squared_distance_bins,
            PairOperation::ElementwiseSub,
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
        let distance_bin_edges: [f64; 4] = [2.0, 6., 10., 15.];
        let squared_bin_edges = distance_bin_edges.map(|x| x.powi(2));
        let squared_distance_bins = IrregularBinEdges::new(&squared_bin_edges).unwrap();

        // check the means (using results computed by pyvsf)
        let expected_mean = [8.41281820819169, 15.01110699893027, f64::NAN];
        let expected_weight = [7., 3., 0.];
        let mean_reducer = EuclideanNormMean::new();
        let n_spatial_bins = distance_bin_edges.len() - 1;
        let mut mean_statepack = prepare_statepack(n_spatial_bins, &mean_reducer);
        let points = PointProps::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();
        let result = apply_accum(
            &mut StatePackViewMut::from_array_view(mean_statepack.view_mut()),
            &mean_reducer,
            &points,
            None,
            &squared_distance_bins,
            PairOperation::ElementwiseSub,
        );
        assert_eq!(result, Ok(()));

        // output buffers
        let mean_result_map =
            get_output_from_statepack_array(&mean_reducer, &mean_statepack.view());

        for i in 0..n_spatial_bins {
            // we might need to adopt an actual rtol
            assert!(
                _isclose(mean_result_map["mean"][i], expected_mean[i], 3.0e-16, 0.0),
                "actual mean = {}, expected mean = {}",
                mean_result_map["mean"][i],
                expected_mean[i]
            );
            assert_eq!(
                mean_result_map["weight"][i], expected_weight[i],
                "unequal weights"
            );
        }

        // check the histograms (using results computed by pyvsf)

        // the hist_buckets were picked such that:
        // - the number of bins is unequal to the distance bin count
        // - there would be a value smaller than the leftmost bin-edge
        // - there would be a value larger than the leftmost bin-edge
        let hist_buckets = IrregularBinEdges::new(&[6.0, 10.0, 14.0]).unwrap();

        #[rustfmt::skip]
        let expected_hist_weights = [
            4., 0., 0.,
            3., 2., 0.,
        ];
        let hist_reducer = EuclideanNormHistogram::from_bin_edges(hist_buckets);
        let mut hist_statepack = prepare_statepack(distance_bin_edges.len() - 1, &hist_reducer);
        let result = apply_accum(
            &mut StatePackViewMut::from_array_view(hist_statepack.view_mut()),
            &hist_reducer,
            &points,
            None,
            &squared_distance_bins,
            PairOperation::ElementwiseSub,
        );
        assert_eq!(result, Ok(()));
        let hist_result_map =
            get_output_from_statepack_array(&hist_reducer, &hist_statepack.view());
        for (i, expected) in expected_hist_weights.iter().enumerate() {
            assert_eq!(
                hist_result_map["weight"][i], *expected,
                "problem at index {}",
                i
            );
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

        let distance_bin_edges: [f64; 3] = [17., 21., 25.];
        let squared_bin_edges = distance_bin_edges.map(|x| x.powi(2));
        let square_distance_bins = IrregularBinEdges::new(&squared_bin_edges).unwrap();

        // the expected results were printed by pyvsf
        let expected_mean = [6.274664681905207, 6.068727871100932];
        let expected_weight = [4., 6.];

        // perform the calculation!
        let reducer = EuclideanNormMean::new();
        let n_spatial_bins = distance_bin_edges.len() - 1;
        let mut statepack = prepare_statepack(n_spatial_bins, &reducer);

        let result = apply_accum(
            &mut StatePackViewMut::from_array_view(statepack.view_mut()),
            &reducer,
            &points_a,
            Some(&points_b),
            &square_distance_bins,
            PairOperation::ElementwiseSub,
        );
        assert_eq!(result, Ok(()));

        let output = get_output_from_statepack_array(&reducer, &statepack.view());

        for i in 0..n_spatial_bins {
            assert!(
                _isclose(output["mean"][i], expected_mean[i], 3.0e-16, 0.0),
                "actual mean = {}, expected mean = {}",
                output["mean"][i],
                expected_mean[i]
            );
            assert_eq!(output["weight"][i], expected_weight[i], "unequal weights");
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
        let distance_bin_edges: [f64; 4] = [2., 6., 10., 15.];
        let squared_bin_edges = distance_bin_edges.map(|x| x.powi(2));
        let squared_distance_bins = IrregularBinEdges::new(&squared_bin_edges).unwrap();

        // check the means (using results computed by pyvsf)
        let expected_mean = [284.57142857142856, 236.0, f64::NAN];
        let expected_weight = [7., 3., 0.];
        let mean_reducer = ComponentSumMean::new();
        let n_spatial_bins = distance_bin_edges.len() - 1;
        let mut mean_statepack = prepare_statepack(n_spatial_bins, &mean_reducer);

        let points = PointProps::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();
        let result = apply_accum(
            &mut StatePackViewMut::from_array_view(mean_statepack.view_mut()),
            &mean_reducer,
            &points,
            None,
            &squared_distance_bins,
            PairOperation::ElementwiseMultiply,
        );
        assert_eq!(result, Ok(()));

        let output = get_output_from_statepack_array(&mean_reducer, &mean_statepack.view());

        for i in 0..3 {
            // we might need to adopt an actual rtol
            assert!(
                _isclose(output["mean"][i], expected_mean[i], 3.0e-16, 0.0),
                "actual mean = {}, expected mean = {}",
                output["mean"][i],
                expected_mean[i]
            );
            assert_eq!(output["weight"][i], expected_weight[i], "unequal weights");
        }
    }
}
