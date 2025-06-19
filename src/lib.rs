use std::usize;

struct Mean {
    weight: Vec<f64>,
    total: Vec<f64>,
}

impl Mean {
    /// Create a Mean instance with length `n` registers
    pub fn new(n: usize) -> Result<Mean, String> {
        if n == 0 {
            Err(String::from("n can't be zero"))
        } else {
            Ok(Mean {
                weight: vec![0.0; n],
                total: vec![0.0; n],
            })
        }
    }

    pub fn initialize(&mut self) {
        self.weight.fill(0.0);
        self.total.fill(0.0);
    }

    pub fn consume(&mut self, val: f64, weight: f64, partition_idx: usize) {
        self.weight[partition_idx] += weight;
        self.total[partition_idx] += val * weight;
    }

    pub fn get_value(&self, out: &mut [f64], weights_out: &mut [f64]) {
        for i in 0..self.weight.len() {
            // TODO need to think about divide by 0
            out[i] = self.total[i] / self.weight[i];

            // is this the most efficient way to do this?
            weights_out[i] = self.weight[i];
        }
    }

    // TODO consider default implementation
    // It would only work in cases where the registers are purely additive,
    // which might not be idomatic.
    pub fn merge(&mut self, other: &Mean) {
        for i in 0..self.weight.len() {
            self.total[i] += other.total[i];
            self.weight[i] += other.weight[i];
        }
    }
}

// I would potentially like to see us do better than this, but it seemse like
// ok place to start
//
// TODO: implement PointProps::new (i.e. no attributes should be public)
struct PointProps<'a> {
    // the convention used in pyvsf holds that different vector-components are
    // indexed along the slow axis.
    //
    // In other words, the ith component of the jth point (for positions &
    // values) is located at an index of `j + i*spatial_dim_stride`
    pub positions: &'a [f64],
    // for the moment, let's assume that values always holds vector-values
    // (since that is more general)
    pub values: &'a [f64],
    pub weights: Option<&'a [f64]>,
    pub n_points: usize,
    // we may want to just get rid of spatial_dim_stride and assume contiguous
    // (i.e. spatial_dim_stride == n_points)
    pub spatial_dim_stride: usize,
    // we could potentially handle this separately
    pub n_spatial_dims: usize,
}

// maybe we want to make separate functions for auto-stats vs
// cross-stats
fn apply_accum(
    accum: &mut Mean,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    bin_edges: &[f64],
    /* pairwise_op, */ // probably an Enum
) -> Result<(), String> {
    // TODO: check that bin_edges monotonically increases
    // TODO: if points_b is not None, make sure there is agreement in the
    //       n_spatial_dims attribute
    Err(String::from("Not implemented yet!"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consume_once() {
        let mut accum = Mean::new(1).unwrap();
        accum.initialize();
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
        accum.initialize();
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
        accum.initialize();
        accum.consume(4.0, 1.0, 0);
        accum.consume(8.0, 1.0, 0);

        let mut accum_other = Mean::new(1).unwrap();
        accum_other.initialize();
        accum_other.consume(1.0, 1.0, 0);
        accum_other.consume(3.0, 1.0, 0);
        accum.merge(&accum_other);

        let mut mean_vec = vec![0.0];
        let mut weight_vec = vec![0.0];
        accum.get_value(&mut mean_vec, &mut weight_vec);
        assert_eq!(mean_vec[0], 4.0);
        assert_eq!(weight_vec[0], 4.0);
    }

    #[test]
    fn test_apply_accum() {
        // this is based on some inputs from pyvsf:tests/test_vsf_props

        // set up our separation bins (they must monotonically increase)
        let bin_edges = vec![17.0, 21.0, 25.0];

        // set up our accumulator
        let mut accum = Mean::new(bin_edges.len() - 1).unwrap();
        accum.initialize();

        #[rustfmt::skip]
        let positions = vec![
             6.,  7.,  8.,  9., 10., 11.,
            12., 13., 14., 15., 16., 17.,
            18., 19., 20., 21., 22., 23.,
        ];

        #[rustfmt::skip]
        let velocities = vec![
            -9., -8., -7., -6., -5., -4.,
            -3., -2., -1.,  0.,  1.,  2.,
             3.,  4.,  5.,  6.,  7.,  8.,
        ];

        let points = PointProps {
            positions: &positions,
            values: &velocities,
            weights: None,
            n_points: 6_usize,
            spatial_dim_stride: 6_usize,
            n_spatial_dims: 6_usize,
        };

        let result = apply_accum(&mut accum, &points, None, &bin_edges);
        assert!(result.is_err());
    }
}
