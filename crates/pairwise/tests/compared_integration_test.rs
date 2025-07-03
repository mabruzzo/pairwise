use ndarray::{Array2, indices};
use pairwise::{
    Accumulator, CartesianBlock, CellWidth, Mean, PointProps, View3DProps, apply_accum,
    dot_product, get_output,
};

use rand::distr::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

mod common;

// Things are a little unergonomic!

// this is to be used to compare operations with CartesianBlock and
// PointProps
pub struct TestScenario {
    // for use with Cartesian_Grid
    cartesian_values_zyx: [Vec<f64>; 3],
    cartesian_weights: Vec<f64>,
    idx_props: View3DProps,
    cell_widths: [f64; 3],
    // for use with PointProps
    position_list: Array2<f64>,
    value_list: Array2<f64>,
    weight_list: Vec<f64>,
}

impl TestScenario {
    // in the future, I think we want to be able to pass in grid properties
    // rather than rely upon hardcoded examples
    pub fn setup(seed: u64, shape_zyx: [usize; 3]) -> TestScenario {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let idx_props = View3DProps::from_shape_contiguous(shape_zyx).unwrap();
        let cell_widths = [1.0, 1.0, 1.0];

        let n_points = shape_zyx[0] * shape_zyx[1] * shape_zyx[2];
        let distribution = Uniform::new_inclusive(0.4, 0.6).unwrap();

        let mut make_array_fn = || -> Vec<f64> {
            let mut vec = vec![0.0; n_points];
            for idx in indices(shape_zyx) {
                let (iz, iy, ix) = idx;
                let i = idx_props.map_idx(iz as isize, iy as isize, ix as isize);
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

                    let cartesian_idx = idx_props.map_idx(iz, iy, ix) as usize;
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
            // for use with PointProps
            position_list,
            value_list,
            weight_list,
        }
    }

    pub fn point_props<'a>(&'a self) -> PointProps<'a> {
        PointProps::new(
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

mod tests {
    use super::*;

    #[test]
    fn test_stuff() {
        let seed = 10582441886303702641_u64;
        let scenario = TestScenario::setup(seed, [4, 3, 2]);

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
        let result = apply_accum(
            &mut statepacks.view_mut(),
            &mean_accum,
            &scenario.point_props(),
            None,
            &squared_distance_bin_edges,
            &dot_product,
        );

        assert_eq!(result, Ok(()));

        let output_cartesian = get_output(&mean_accum, &statepacks.view());

        println!("{:?}", output_points);
        println!("{:?}", output_cartesian);
        assert_eq!(
            output_points["weight"], output_cartesian["weight"],
            "\"weight\" isn't consistent!"
        );

        // the fact that the inputs are sorta garbage may make this comparison
        // a little problematic (e.g. it may make make the answers unstable)
        let itr = output_points["mean"]
            .iter()
            .zip(output_cartesian["mean"].iter())
            .enumerate();
        for (i, (pts_mean, grid_mean)) in itr {
            assert!(
                common::isclose(*grid_mean, *pts_mean, 0.0, 0.0),
                "index {} of mean aren't consistent. actual: {}, ref: {}",
                i,
                *grid_mean,
                *pts_mean
            );
        }
    }
}
