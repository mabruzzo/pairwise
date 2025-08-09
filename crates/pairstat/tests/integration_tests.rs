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
}
