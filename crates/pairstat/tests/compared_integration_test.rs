use ndarray::{Array2, ArrayView2, Axis};
use pairstat::{
    AccumulatorBuilder, CartesianBlock, CellWidth, RuntimeSpec, UnstructuredPoints, View3DSpec,
    process_cartesian, process_unstructured,
};
use std::collections::HashMap;

// use ndarray::indices
// use rand::distr::{Distribution, Uniform};
// use rand_xoshiro::Xoshiro256PlusPlus;
// use rand_xoshiro::rand_core::SeedableRng;

mod common;

// Things are a little unergonomic!

// it may be better to specify things in terms of CartesianBlocks and then
// to generate PointProp instances from CartesianBlocks. (In fact, we may wish
// to consider adding that to the main API -- there are some clear use-cases)

// this is to be used to compare operations with CartesianBlock and
// UnstructuredPoints
pub struct TestScenario {
    // for use with Cartesian_Grid
    cartesian_values_zyx: [Vec<f64>; 3],
    cartesian_weights: Vec<f64>,
    idx_props: View3DSpec,
    cell_widths: [f64; 3],
    // for use with UnstructuredPoints
    position_list: Array2<f64>,
    value_list: Array2<f64>,
    weight_list: Vec<f64>,
}

impl TestScenario {
    /*
     // in the future, I think we want to be able to pass in grid properties
     // rather than rely upon hardcoded examples
     //
     // The use of rng in this scenario isn't really accomplishing much of
     // anything... At this point, its mostly a proof of concept to help us
     // later if we want to generate gaussian random fields from
     // power-spectra (the power-spectrum itself specifies the magnitude of
     // wavenumbers, and we will need to randomly generate phases)
     pub fn setup(seed: u64, shape_zyx: [usize; 3]) -> TestScenario {
         let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
         let idx_props = View3DSpec::from_shape_contiguous(shape_zyx).unwrap();
         let cell_widths = [1.0, 1.0, 1.0];

         let n_points = shape_zyx[0] * shape_zyx[1] * shape_zyx[2];
         let distribution = Uniform::new_inclusive(0.4, 0.6).unwrap();

         let mut make_array_fn = || -> Vec<f64> {
             let mut vec = vec![0.0; n_points];
             for idx in indices(shape_zyx) {
                 let (iz, iy, ix) = idx;
                 let i = idx_props.map_idx3d_to_1d(iz as isize, iy as isize, ix as isize);
                 vec[i as usize] = distribution.sample(&mut rng) * (iy as f64).sin();
             }
             vec
         };

         let cartesian_values_zyx = [make_array_fn(), make_array_fn(), make_array_fn()];
         let cartesian_weights: Vec<f64> = vec![1.0; n_points];

         let mut position_list = Array2::zeros([3, n_points]);
         let mut value_list = Array2::zeros([3, n_points]);
         let mut weight_list = vec![0.0; n_points];

         let [nz, ny, nx] = *idx_props.shape();

         for iz in 0..nz {
             for iy in 0..ny {
                 for ix in 0..nx {
                     let point_idx = (ix + nx * (iy + ny * iz)) as usize;
                     position_list[[2, point_idx]] = (0.5 + (ix as f64)) * cell_widths[2];
                     position_list[[1, point_idx]] = (0.5 + (iy as f64)) * cell_widths[1];
                     position_list[[0, point_idx]] = (0.5 + (ix as f64)) * cell_widths[0];

                     let cartesian_idx = idx_props.map_idx3d_to_1d(iz, iy, ix) as usize;
                     for dim in 0..3 {
                         // in this loop, we dim = 0 corresponds to the
                         // z-direction and dim = 0 corresponds to the x-axis
                         value_list[[dim, point_idx]] = cartesian_values_zyx[dim][cartesian_idx];
                     }
                     weight_list[point_idx] = cartesian_weights[cartesian_idx];
                 }
             }
         }

         TestScenario {
             // for use with Cartesian_Grid
             cartesian_values_zyx,
             cartesian_weights,
             idx_props,
             cell_widths,
             // for use with UnstructuredPoints
             position_list,
             value_list,
             weight_list,
         }
     }
    */

    /// constructs an instance using a set of points with integer positions
    fn from_integer_position_points(
        positions: ArrayView2<i32>,
        values: ArrayView2<f64>,
        weights: Option<&[f64]>,
    ) -> TestScenario {
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

        TestScenario {
            // for use with Cartesian_Grid
            cartesian_values_zyx,
            cartesian_weights,
            idx_props,
            cell_widths: [1.0, 1.0, 1.0],
            // for use with UnstructuredPoints
            position_list: positions.mapv(f64::from),
            value_list: values.to_owned(),
            weight_list,
        }
    }

    pub fn point_props<'a>(&'a self) -> UnstructuredPoints<'a> {
        UnstructuredPoints::new(
            self.position_list.view(),
            self.value_list.view(),
            Some(&self.weight_list),
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
            /* start_idx_global_offset: */ [0, 0, 0],
        )
        .unwrap()
    }

    pub fn cell_width(&self) -> CellWidth {
        CellWidth::new(self.cell_widths).unwrap()
    }
}

#[test]
fn test_apply_accum_auto_corr() {
    // keep in mind that we interpret positions as a (3, ...) array
    // so position 0 is [6,12,18]
    let positions: Vec<i32> = (6_i32..24_i32).collect();
    let values: Vec<f64> = (-9..9).map(|x| 2.0 * (x as f64)).collect();

    let scenario = TestScenario::from_integer_position_points(
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

    for use_points in [true, false] {
        let mut accum = AccumulatorBuilder::new()
            .calc_kind("2pcf")
            // the bin edges are chosen so that some values don't fit
            // inside the bottom bin
            .dist_bin_edges(&[2., 6., 10., 15.])
            .build()
            .unwrap();
        // actually run the test
        let rslt = if use_points {
            let points = scenario.point_props();
            process_unstructured(&mut accum, points, None, &RuntimeSpec)
        } else {
            let block = scenario.cartesian_block();
            let cell_width = scenario.cell_width();
            process_cartesian(&mut accum, block, None, cell_width, &RuntimeSpec)
        };
        assert!(rslt.is_ok());
        let output = accum.get_output();

        println!("{:#?}", output);

        common::assert_consistent_results(&output, &expected, &rtol_atol_sets);
    }
}

/*
#[test]
#[ignore] // TODO: WE NEED TO COME BACK TO THIS (I think there is a bug in setup)
fn test_random_autocorr_scenario() {
    let seed = 10582441886303702641_u64;
    // let scenario = TestScenario::setup(seed, [4, 3, 2]);
    let scenario = TestScenario::setup(seed, [4, 1, 1]);

    let distance_bin_edges: &[f64] = &[0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25];
    let squared_distance_bin_edges: Vec<f64> =
        distance_bin_edges.iter().map(|x| x.powi(2)).collect();

    let mean_accum = Mean;
    let n_spatial_bins = distance_bin_edges.len() - 1;
    let mut statepacks = common::prepare_statepacks(n_spatial_bins, &mean_accum);

    // we can run things with point_props
    let result = apply_accum(
        &mut statepacks.view_mut(),
        &mean_accum,
        &scenario.point_props(),
        None,
        &squared_distance_bin_edges,
        &dot_product,
    );
    assert_eq!(result, Ok(()));

    let output_points = get_output(&mean_accum, &statepacks.view());

    for mut col in statepacks.columns_mut() {
        mean_accum.reset_statepack(&mut col);
    }

    // we can run things with cartesian_block
    let block = scenario.cartesian_block();
    let result = apply_cartesian(
        &mut statepacks.view_mut(),
        &mean_accum,
        &block,
        None,
        &squared_distance_bin_edges,
        &scenario.cell_width(),
    );

    //let result = apply_accum(
    //    &mut statepacks.view_mut(),
    //    &mean_accum,
    //    &scenario.point_props(),
    //    None,
    //    &squared_distance_bin_edges,
    //    &dot_product,
    //);

    assert_eq!(result, Ok(()));

    let output_cartesian = get_output(&mean_accum, &statepacks.view());
    println!("points: {:?}", output_points);
    println!("cartesian: {:?}", output_cartesian);

    // the fact that the inputs are sorta garbage may make this comparison
    // a little problematic (e.g. it may make make the means unstable)
    let rtol_atol_sets = HashMap::from([("weight", [0.0, 0.0]), ("mean", [0.0, 0.0])]);
    common::assert_consistent_results(&output_cartesian, &output_points, &rtol_atol_sets);
}
*/
