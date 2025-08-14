use ndarray::{Array2, ArrayView2, Axis};
use pairstat::{
    AccumulatorBuilder, CartesianBlock, CellWidth, Error, RuntimeSpec, UnstructuredPoints,
    View3DSpec, process_cartesian, process_unstructured,
};
use std::collections::HashMap;

mod common;

// it may be better to specify things in terms of CartesianBlocks and then
// to generate PointProp instances from CartesianBlocks. (In fact, we may wish
// to consider adding that to the main API -- there are some clear use-cases)

// this is used to compare operations with CartesianBlock and
// UnstructuredPoints
pub struct TestDataWrapper {
    // for use with Cartesian_Grid
    cartesian_values_zyx: [Vec<f64>; 3],
    cartesian_weights: Vec<f64>,
    idx_props: View3DSpec,
    cell_widths: [f64; 3],
    cartesian_start_idx_global_offset: [isize; 3],
    // for use with UnstructuredPoints
    position_list: Array2<f64>,
    value_list: Array2<f64>,
    weight_list: Vec<f64>,
}

impl TestDataWrapper {
    /// constructs an instance using a set of points with integer positions
    fn from_integer_position_points(
        positions: ArrayView2<i32>,
        values: ArrayView2<f64>,
        weights: Option<&[f64]>,
    ) -> TestDataWrapper {
        assert!(positions.shape() == values.shape());
        assert_eq!(positions.len_of(Axis(0)), 3);
        let n_points = positions.len_of(Axis(1));

        let weight_list = if let Some(weight_slice) = weights {
            assert_eq!(weight_slice.len(), n_points);
            Vec::from(weight_slice)
        } else {
            vec![1.0; n_points]
        };

        // determine the minimum and maximum values
        // (there's probably a more efficient way to do this)
        let mut min_pos = [i32::MAX; 3];
        let mut max_pos = [i32::MIN; 3];
        for dim in 0..3 {
            min_pos[dim] = *positions.index_axis(Axis(0), dim).iter().min().unwrap();
            max_pos[dim] = *positions.index_axis(Axis(0), dim).iter().max().unwrap();
        }

        let shape = [
            (max_pos[0] + 1 - min_pos[0]) as usize,
            (max_pos[1] + 1 - min_pos[1]) as usize,
            (max_pos[2] + 1 - min_pos[2]) as usize,
        ];

        let idx_props = View3DSpec::from_shape_contiguous(shape).unwrap();
        let cartesian_size = idx_props.required_length();

        let mut cartesian_values_zyx = [
            vec![0.0; cartesian_size],
            vec![0.0; cartesian_size],
            vec![0.0; cartesian_size],
        ];
        let mut cartesian_weights = vec![0.0; cartesian_size];

        for i in 0..n_points {
            let cartesian_idx = idx_props.map_idx3d_to_1d(
                (positions[[0, i]] - min_pos[0]) as isize,
                (positions[[1, i]] - min_pos[1]) as isize,
                (positions[[2, i]] - min_pos[2]) as isize,
            ) as usize;
            cartesian_values_zyx[0][cartesian_idx] = values[[0, i]];
            cartesian_values_zyx[1][cartesian_idx] = values[[1, i]];
            cartesian_values_zyx[2][cartesian_idx] = values[[2, i]];
            cartesian_weights[cartesian_idx] = weight_list[i];
        }

        TestDataWrapper {
            // for use with Cartesian_Grid
            cartesian_values_zyx,
            cartesian_weights,
            idx_props,
            cell_widths: [1.0, 1.0, 1.0],
            cartesian_start_idx_global_offset: [
                min_pos[0] as isize,
                min_pos[1] as isize,
                min_pos[2] as isize,
            ],
            // for use with UnstructuredPoints
            position_list: positions.mapv(f64::from),
            value_list: values.to_owned(),
            weight_list,
        }
    }

    pub fn point_props<'a>(&'a self) -> UnstructuredPoints<'a> {
        UnstructuredPoints::new(
            self.position_list.as_slice().unwrap(),
            self.value_list.as_slice().unwrap(),
            &self.weight_list,
            self.position_list.shape()[1],
            None,
        )
        .unwrap()
    }

    pub fn cartesian_block<'a>(&'a self) -> CartesianBlock<'a> {
        CartesianBlock::new(
            [
                &self.cartesian_values_zyx[0],
                &self.cartesian_values_zyx[1],
                &self.cartesian_values_zyx[2],
            ],
            &self.cartesian_weights,
            self.idx_props.clone(),
            self.cartesian_start_idx_global_offset,
        )
        .unwrap()
    }

    pub fn cell_width(&self) -> CellWidth {
        CellWidth::new(self.cell_widths).unwrap()
    }

    pub fn has_equal_cell_width(&self, other: &TestDataWrapper) -> bool {
        self.cell_widths == other.cell_widths
    }
}

/// the premise is to try to centralize as much logic as possible, to minimize
/// how much code is duplicated (and how much we need to change when we make
/// changes)
fn exec_calc(
    data: &TestDataWrapper,
    data_other: &Option<TestDataWrapper>,
    accum_builder: &AccumulatorBuilder,
    unstructured: bool,
) -> Result<HashMap<&'static str, Vec<f64>>, Error> {
    let mut accum = accum_builder.build()?;
    let rslt = if unstructured {
        let points = data.point_props();
        let points_other = data_other.as_ref().map(|d| d.point_props());
        process_unstructured(&mut accum, points, points_other, &RuntimeSpec)
    } else {
        let block = data.cartesian_block();
        let block_other = if let Some(d) = data_other {
            if !data.has_equal_cell_width(d) {
                panic!("data_other has different cell_widths from data");
            }
            Some(d.cartesian_block())
        } else {
            None
        };
        let cell_width = data.cell_width();
        process_cartesian(&mut accum, block, block_other, cell_width, &RuntimeSpec)
    };
    rslt.map(|_| accum.get_output())
}

#[test]
fn test_apply_accum_auto() {
    // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

    // keep in mind that we interpret positions as a (3, ...) array
    // so position 0 is [6,12,18]
    let positions: Vec<i32> = (6_i32..24_i32).collect();
    let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

    let data = TestDataWrapper::from_integer_position_points(
        ArrayView2::from_shape((3, 6), &positions).unwrap(),
        ArrayView2::from_shape((3, 6), &values).unwrap(),
        None,
    );

    // the expected results were produced by pyvsf
    let expected = HashMap::from([
        ("weight", vec![7., 3., 0.]),
        ("mean", vec![8.41281820819169, 15.01110699893027, f64::NAN]),
    ]);
    let rtol_atol_sets = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);

    // to store the builder, we can't chain off of the constructor
    let mut accum_builder = AccumulatorBuilder::new();
    accum_builder
        .calc_kind("astro_sf1")
        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        .dist_bin_edges(&[2.0, 6., 10., 15.]);

    for use_unstructured in [true, false] {
        let output = exec_calc(&data, &None, &accum_builder, use_unstructured).unwrap();
        common::assert_consistent_results(&output, &expected, &rtol_atol_sets);
    }
}

#[test]
fn test_apply_accum_auto_hist() {
    // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

    // keep in mind that we interpret positions as a (3, ...) array
    // so position 0 is [6,12,18]
    let positions: Vec<i32> = (6_i32..24_i32).collect();
    let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

    let data = TestDataWrapper::from_integer_position_points(
        ArrayView2::from_shape((3, 6), &positions).unwrap(),
        ArrayView2::from_shape((3, 6), &values).unwrap(),
        None,
    );

    // check the weights (using results computed by pyvsf)
    #[rustfmt::skip]
    let expected_weights = vec![
        4., 0., 0.,
        3., 2., 0.,
    ];
    let expected = HashMap::from([("weight", expected_weights)]);
    let rtol_atol_sets = HashMap::from([("weight", [0.0, 0.0])]);

    // to store the builder, we can't chain off of the constructor
    let mut accum_builder = AccumulatorBuilder::new();
    accum_builder
        .calc_kind("hist_astro_sf1")
        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        .dist_bin_edges(&[2.0, 6., 10., 15.])
        // the hist_buckets were picked such that:
        // - the number of bins is unequal to the distance bin count
        // - there would be a value smaller than the leftmost bin-edge
        // - there would be a value larger than the leftmost bin-edge
        .hist_bucket_edges(&[6.0, 10.0, 14.0]);

    for use_unstructured in [true, false] {
        let output = exec_calc(&data, &None, &accum_builder, use_unstructured).unwrap();
        common::assert_consistent_results(&output, &expected, &rtol_atol_sets);
    }
}

#[test]
fn test_apply_accum_cross() {
    // this is loosely based on some inputs from pyvsf:tests/test_vsf_props

    // we intentionally padded positions_b with a point
    // that is so far away from everything else that it can't
    // fit inside a separation bin
    #[rustfmt::skip]
    let pos_a: [i32; 9] = [
        0_i32, 1_i32, 50_i32,
        2_i32, 3_i32, 0_i32,
        4_i32, 5_i32, 0_i32,
    ];

    #[rustfmt::skip]
    let vals_a = [
        -3., -2., 1000.,
        -1.,  0., 1000.,
         1.,  2., 1000.,
    ];

    let data_a = TestDataWrapper::from_integer_position_points(
        ArrayView2::from_shape((3, 3), &pos_a).unwrap(),
        ArrayView2::from_shape((3, 3), &vals_a).unwrap(),
        None,
    );

    // keep in mind that we interpret positions as a (3, ...) array
    // so position 0 is [6,12,18]
    let pos_b: Vec<i32> = (6_i32..24_i32).collect();
    let vals_b: Vec<f64> = (-9..9).map(|x| x as f64).collect();

    let data_b = Some(TestDataWrapper::from_integer_position_points(
        ArrayView2::from_shape((3, 6), &pos_b).unwrap(),
        ArrayView2::from_shape((3, 6), &vals_b).unwrap(),
        None,
    ));

    // the expected results were produced by pyvsf
    let expected = HashMap::from([
        ("weight", vec![4., 6.]),
        ("mean", vec![6.274664681905207, 6.068727871100932]),
    ]);
    let rtol_atol_sets = HashMap::from([("weight", [0.0, 0.0]), ("mean", [3.0e-16, 0.0])]);

    // to store the builder, we can't chain off of the constructor
    let mut accum_builder = AccumulatorBuilder::new();
    accum_builder
        .calc_kind("astro_sf1")
        .dist_bin_edges(&[17., 21., 25.]);

    for use_unstructured in [true, false] {
        let output = exec_calc(&data_a, &data_b, &accum_builder, use_unstructured).unwrap();
        //eprintln!("{:#?}", output);
        common::assert_consistent_results(&output, &expected, &rtol_atol_sets);
    }
}

#[test]
fn test_apply_accum_auto_corr() {
    // keep in mind that we interpret positions as a (3, ...) array
    // so position 0 is [6,12,18]
    let positions: Vec<i32> = (6_i32..24_i32).collect();
    let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

    let data = TestDataWrapper::from_integer_position_points(
        ArrayView2::from_shape((3, 6), &positions).unwrap(),
        ArrayView2::from_shape((3, 6), &values).unwrap(),
        None,
    );

    // check the means (using results computed by pyvsf)
    let expected = HashMap::from([
        ("mean", vec![284.57142857142856, 236.0, f64::NAN]),
        ("weight", vec![7., 3., 0.]),
    ]);
    let rtol_atol_sets = HashMap::from([("mean", [3.0e-16, 0.0]), ("weight", [0.0, 0.0])]);

    // to store the builder, we can't chain off of the constructor
    let mut accum_builder = AccumulatorBuilder::new();
    accum_builder
        .calc_kind("2pcf")
        // the bin edges are chosen so that some values don't fit
        // inside the bottom bin
        .dist_bin_edges(&[2., 6., 10., 15.]);

    for use_unstructured in [true, false] {
        let output = exec_calc(&data, &None, &accum_builder, use_unstructured).unwrap();
        common::assert_consistent_results(&output, &expected, &rtol_atol_sets);
    }
}
