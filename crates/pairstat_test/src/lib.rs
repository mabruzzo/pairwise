use ndarray::{Array2, ArrayView2, Axis};
use pairstat_nostd_internal::{CartesianBlock, CellWidth, UnstructuredPoints, View3DSpec};
use rand::distr::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

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
    /// Uses the smallest possible bounding CartesianBlock with weight 0
    /// for missing points.
    pub fn from_integer_position_points(
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

    pub fn from_block(
        value_components: [Vec<f64>; 3],
        cartesian_weights: Vec<f64>,
        idx_props: View3DSpec,
        start_idx_global_offset: [isize; 3],
        block_dimensions: [f64; 3],
    ) -> TestDataWrapper {
        let required_len = idx_props.required_length();
        let n_elements = idx_props.n_elements();

        assert_eq!(value_components[0].len(), required_len);
        assert_eq!(value_components[1].len(), required_len);
        assert_eq!(value_components[2].len(), required_len);
        assert_eq!(cartesian_weights.len(), required_len);
        assert!(
            (block_dimensions[0] > 0.0)
                && (block_dimensions[1] > 0.0)
                && (block_dimensions[2] > 0.0),
        );

        let mut position_list = Array2::<f64>::zeros([3, n_elements]);
        let mut value_list = Array2::<f64>::zeros([3, n_elements]);
        let mut weight_list = vec![0.0; n_elements];

        let cell_widths = [
            block_dimensions[0] / (idx_props.shape()[0] as f64),
            block_dimensions[1] / (idx_props.shape()[1] as f64),
            block_dimensions[2] / (idx_props.shape()[2] as f64),
        ];

        let mut counter = 0;
        for i in 0..idx_props.shape()[0] {
            for j in 0..idx_props.shape()[1] {
                for k in 0..idx_props.shape()[2] {
                    position_list[[0, counter]] = (i as f64) * cell_widths[0];
                    position_list[[1, counter]] = (j as f64) * cell_widths[1];
                    position_list[[2, counter]] = (k as f64) * cell_widths[2];

                    let idx = idx_props.map_idx3d_to_1d(i, j, k) as usize;

                    value_list[[0, counter]] = value_components[0][idx];
                    value_list[[1, counter]] = value_components[1][idx];
                    value_list[[2, counter]] = value_components[2][idx];

                    weight_list[counter] = cartesian_weights[idx];

                    counter += 1;
                }
            }
        }

        TestDataWrapper {
            // for use with Cartesian_Grid
            cartesian_values_zyx: value_components,
            cartesian_weights,
            idx_props,
            cell_widths,
            cartesian_start_idx_global_offset: start_idx_global_offset,
            // for use with UnstructuredPoints
            position_list,
            value_list,
            weight_list,
        }
    }

    pub fn from_random(
        shape: [usize; 3],
        block_dimensions: [f64; 3],
        seed: u64,
    ) -> TestDataWrapper {
        let idx_spec = View3DSpec::from_shape_contiguous(shape).unwrap();
        let length = idx_spec.n_elements();

        let mut my_rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let dist = Uniform::try_from(-1.0..=1.0).unwrap();
        let mut value_components = [vec![0.0; length], vec![0.0; length], vec![0.0; length]];
        for i in 0..length {
            value_components[0][i] = dist.sample(&mut my_rng);
            value_components[1][i] = dist.sample(&mut my_rng);
            value_components[2][i] = dist.sample(&mut my_rng);
        }

        Self::from_block(
            value_components,
            vec![1.0; length],
            idx_spec,
            [0isize, 0isize, 0isize],
            block_dimensions,
        )
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
