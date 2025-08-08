mod common;

use common::{assert_consistent_results, prepare_statepack};
use ndarray::ArrayView2;
use pairwise::{
    AccumulatorBuilder, EuclideanNormMean, PairOperation, StatePackViewMut, UnstructuredPoints,
    apply_accum,
};

// Things are a little unergonomic!

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use pairwise_nostd_internal::IrregularBinEdges;

    use super::*;

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
        let points = UnstructuredPoints::new(
            ArrayView2::from_shape((3, 2), &positions).unwrap(),
            ArrayView2::from_shape((3, 2), &values).unwrap(),
            None,
        )
        .unwrap();

        // should fail for mismatched spatial dimensions
        let points_b = UnstructuredPoints::new(
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

        // should fail if 1 points object provides weights an the other doesn't
        let weights = [1.0, 0.0];
        let points_b = UnstructuredPoints::new(
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
    }

    #[test]
    fn test_apply_accum_auto() {
        // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

        let points = UnstructuredPoints::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();

        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        let mut accumulator = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .dist_bin_edges(&[2.0, 6., 10., 15.])
            .build()
            .unwrap();

        // the expected results were produced by pyvsf
        let expected = HashMap::from([
            ("weight", vec![7., 3., 0.]),
            ("mean", vec![8.41281820819169, 15.01110699893027, f64::NAN]),
        ]);
        let rtol_atol_vals = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);

        assert!(
            accumulator
                .collect_unstructured_contributions(points.clone(), None)
                .is_ok()
        );

        let mean_result_map = accumulator.get_output();
        assert_consistent_results(&mean_result_map, &expected, &rtol_atol_vals);
    }

    #[test]
    fn test_apply_accum_auto_hist() {
        // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

        let points = UnstructuredPoints::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();

        let mut accumulator = AccumulatorBuilder::new()
            .calc_kind("hist_astro_sf1")
            // the bin edges are chosen so that some values don't fit
            // inside the bottom bin
            .dist_bin_edges(&[2.0, 6., 10., 15.])
            // the hist_buckets were picked such that:
            // - the number of bins is unequal to the distance bin count
            // - there would be a value smaller than the leftmost bin-edge
            // - there would be a value larger than the leftmost bin-edge
            .hist_bucket_edges(&[6.0, 10.0, 14.0])
            .build()
            .unwrap();

        #[rustfmt::skip]
        let expected_hist_weights = [
            4., 0., 0.,
            3., 2., 0.,
        ];
        assert!(
            accumulator
                .collect_unstructured_contributions(points.clone(), None)
                .is_ok()
        );
        let hist_result_map = accumulator.get_output();
        println!("{hist_result_map:#?}");
        for (i, expected) in expected_hist_weights.iter().enumerate() {
            assert_eq!(
                hist_result_map["weight"][i], *expected,
                "problem at index {i}",
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

        let points_a = UnstructuredPoints::new(
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

        let points_b = UnstructuredPoints::new(
            ArrayView2::from_shape((3, 3), &positions_b).unwrap(),
            ArrayView2::from_shape((3, 3), &values_b).unwrap(),
            None,
        )
        .unwrap();

        let mut accumulator = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .dist_bin_edges(&[17., 21., 25.])
            .build()
            .unwrap();
        // I don't know if I like this... Instead, should we be passing
        // &mut accumulator, and the other args to a standalone function?
        let rslt = accumulator.collect_unstructured_contributions(points_a, Some(points_b));
        assert!(rslt.is_ok());

        let output = accumulator.get_output();

        // the expected results were produced by pyvsf
        let expected = HashMap::from([
            ("weight", vec![4., 6.]),
            ("mean", vec![6.274664681905207, 6.068727871100932]),
        ]);
        let rtol_atol_vals = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);

        assert_consistent_results(&output, &expected, &rtol_atol_vals);
    }

    #[test]
    fn test_apply_accum_auto_corr() {
        // keep in mind that we interpret positions as a (3, ...) array
        // so position 0 is [6,12,18]
        let positions: Vec<f64> = (6..24).map(|x| x as f64).collect();
        let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

        let points = UnstructuredPoints::new(
            ArrayView2::from_shape((3, 6), &positions).unwrap(),
            ArrayView2::from_shape((3, 6), &values).unwrap(),
            None,
        )
        .unwrap();

        let mut accumulator = AccumulatorBuilder::new()
            .calc_kind("2pcf")
            // the bin edges are chosen so that some values don't fit
            // inside the bottom bin
            .dist_bin_edges(&[2., 6., 10., 15.])
            .build()
            .unwrap();
        // I don't know if I like this... Instead, should we be passing
        // &mut accumulator, and the other args to a standalone function?
        let rslt = accumulator.collect_unstructured_contributions(points, None);
        assert!(rslt.is_ok());

        let output = accumulator.get_output();

        let expected = HashMap::from([
            ("weight", vec![7., 3., 0.]),
            ("mean", vec![284.57142857142856, 236.0, f64::NAN]),
        ]);
        let rtol_atol_vals = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);

        assert_consistent_results(&output, &expected, &rtol_atol_vals);
    }
}
