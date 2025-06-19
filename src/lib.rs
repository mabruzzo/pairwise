use std::{f64::DIGITS, process::Output, usize};

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

    // TODO this should potentially be merged with new
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

// TODO: implement PointProps::new (i.e. no attributes should be public)
struct PointProps<'a> {
    // the convention used in pyvsf holds that different vector-components are
    // indexed along the slow axis.
    //
    // In other words, the kth component of the ith point (for positions &
    // values) is located at an index of `i + k*spatial_dim_stride`
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

impl PointProps<'_> {
    pub fn get_weight(&self, idx: usize) -> f64 {
        if let Some(weights) = self.weights {
            weights[idx]
        } else {
            1.0
        }
    }
}

// TODO use binary search, and have specialized version for regularly spaced bins?
fn get_distance_bin(distance_squared: f64, squared_bin_edges: &[f64]) -> Option<usize> {
    // index of first element greater than distance_squared
    // (or squared_bin_edges.len() if none are greater)
    let mut first_greater = 0;
    for &edge in squared_bin_edges.iter() {
        if distance_squared < edge {
            break;
        }
        first_greater += 1;
    }
    if (first_greater == squared_bin_edges.len()) || (first_greater == 0) {
        None
    } else {
        Some(first_greater - 1)
    }
}

// maybe we want to make separate functions for auto-stats vs
// cross-stats
// TODO: generalize to allow faster calculations for regular spatial grids
fn apply_accum(
    out: &mut Vec<f64>,
    weights_out: &mut Vec<f64>,
    accum: &mut Mean,
    points_a: &PointProps,
    points_b: Option<&PointProps>,
    // TODO decide bin, partition, or something else, then unify the naming
    // TODO should bin_edges should be a member of the AccumKernel Struct
    bin_edges: &[f64],
    /* pairwise_op, */ // probably an Enum
) -> Result<(), String> {
    // TODO check size of output buffers

    // Check that bin_edges are monotonically increasing
    if !bin_edges.is_sorted() {
        return Err(String::from(
            "bin_edges must be sorted (monotonically increasing)",
        ));
    }

    //  if points_b is not None, make sure a and b have the same number of
    // spatial dimensions
    if let Some(points_b) = points_b {
        if points_a.n_spatial_dims != points_b.n_spatial_dims {
            return Err(String::from(
                "points_a and points_b must have the same number of spatial dimensions",
            ));
        }
    }

    accum.initialize();
    // I think this alloc is worth it? Could use a buffer?
    let squared_bin_edges: Vec<f64> = bin_edges.iter().map(|x| x.powi(2)).collect();

    // complicated case first: two arrays, we combine points with pairwise_op
    if let Some(points_b) = points_b {
        for i in 0..points_a.n_points {
            for j in 0..points_b.n_points {
                // compute the distance between the points, then the distance bin
                let mut distance_squared = 0.0;
                for k in 0..points_a.n_spatial_dims {
                    distance_squared += (points_a.positions[i + k * points_a.spatial_dim_stride]
                        - points_b.positions[j + k * points_b.spatial_dim_stride])
                        .powi(2);
                }
                let Some(distance_bin) = get_distance_bin(distance_squared, &squared_bin_edges)
                else {
                    continue;
                };

                // get the value
                // TODO switch on pairwise op?
                // TODO I want to talk this design through before we roll further with it
                // I was imagining that there would be separate AccumKernels for, e.g.
                // structure functions and correlations, rather than using Mean with
                // different pairwise_ops. I think it might be possible to write some
                // nice "AccumKernel combinators" to make this clean, but maybe it would
                // require too much magic.
                let val = points_a.values[i] - points_b.values[j];

                // get the weight
                let pair_weight = points_a.get_weight(i) * points_b.get_weight(j);

                accum.consume(val, pair_weight, distance_bin);
            }
            accum.get_value(out, weights_out);
        }
    } else {
        // single array case
        for i in 0..points_a.n_points {
            let mut distance_squared = 0.0;
            for k in 0..points_a.n_spatial_dims {
                let idx = i + k * points_a.spatial_dim_stride;
                distance_squared += points_a.positions[idx].powi(2);
            }
            let Some(distance_bin) = get_distance_bin(distance_squared, &squared_bin_edges) else {
                continue;
            };

            accum.consume(points_a.values[i], points_a.get_weight(i), distance_bin);
        }
        accum.get_value(out, weights_out);
    }
    Ok(())
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

        // output buffers
        let mut mean_vec = vec![0.0; 3];
        let mut weight_vec = vec![0.0; 3];

        let points = PointProps {
            positions: &positions,
            values: &velocities,
            weights: None,
            n_points: 6_usize,
            spatial_dim_stride: 6_usize,
            n_spatial_dims: 3_usize,
        };

        let result = apply_accum(
            &mut mean_vec,
            &mut weight_vec,
            &mut accum,
            &points,
            None,
            &bin_edges,
        );
        assert!(result == Ok(()));
        // TODO check that this is actually right

        // should fail for non-monotonic bins
        let unsorted_bin_edges = vec![25.0, 21.0, 17.0];

        let result = apply_accum(
            &mut mean_vec,
            &mut weight_vec,
            &mut accum,
            &points,
            None,
            &unsorted_bin_edges,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("monotonic"));

        // should fail for mismatched spatial dimensions
        let points_b = PointProps {
            positions: &positions,
            values: &velocities,
            weights: None,
            n_points: 6_usize,
            spatial_dim_stride: 6_usize,
            n_spatial_dims: 3_usize,
        };

        let result = apply_accum(
            &mut mean_vec,
            &mut weight_vec,
            &mut accum,
            &points,
            Some(&points_b),
            &bin_edges,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("spatial dimensions"));
    }
}
