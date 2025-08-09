mod common;

use common::assert_consistent_results;
use ndarray::ArrayView2;
use pairstat::{AccumulatorBuilder, RuntimeSpec, UnstructuredPoints, process_unstructured};
use std::collections::HashMap;

// todo: we can get rid of the test module in integration tests
#[cfg(test)]
mod tests {

    use super::*;

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
        assert!(process_unstructured(&mut accumulator, points, None, &RuntimeSpec).is_ok());

        // the expected results were produced by pyvsf
        let expected = HashMap::from([
            ("weight", vec![7., 3., 0.]),
            ("mean", vec![8.41281820819169, 15.01110699893027, f64::NAN]),
        ]);
        let rtol_atol_vals = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);
        assert_consistent_results(&accumulator.get_output(), &expected, &rtol_atol_vals);
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

        assert!(process_unstructured(&mut accumulator, points.clone(), None, &RuntimeSpec).is_ok());
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
        assert!(
            process_unstructured(&mut accumulator, points_a, Some(points_b), &RuntimeSpec).is_ok()
        );

        // the expected results were produced by pyvsf
        let expected = HashMap::from([
            ("weight", vec![4., 6.]),
            ("mean", vec![6.274664681905207, 6.068727871100932]),
        ]);
        let rtol_atol_vals = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);

        assert_consistent_results(&accumulator.get_output(), &expected, &rtol_atol_vals);
    }
}
