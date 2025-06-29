//! Miscellaneous machinery used to implement the package

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};

use pairwise_internal::{Accumulator, OutputDescr};

// TODO: refactor so that Histogram doesn't hold a vector and move to
//       pairwise_internal/accumulator.rs (to )
pub struct Histogram {
    // maybe call these hist_bucket_edges?
    hist_bin_edges: Vec<f64>,
    n_hist_bins: usize,
}

impl Histogram {
    pub fn new(hist_bin_edges: &[f64]) -> Result<Histogram, &'static str> {
        if hist_bin_edges.len() < 2 {
            Err("hist_bin_edges must include at least 2 elements")
        } else if !hist_bin_edges.is_sorted() {
            Err("hist_bin_edges must be monotonically increasing")
        } else {
            let n_hist_bins = hist_bin_edges.len() - 1;
            Ok(Histogram {
                hist_bin_edges: hist_bin_edges.to_vec(),
                n_hist_bins,
            })
        }
    }
}

impl Accumulator for Histogram {
    fn statepack_size(&self) -> usize {
        self.n_hist_bins
    }

    /// initializes the storage tracking the acumulator's state
    fn reset_statepack(&self, statepack: &mut ArrayViewMut1<f64>) {
        statepack.fill(0.0);
    }

    /// consume the value and weight to update the statepack
    fn consume(&self, statepack: &mut ArrayViewMut1<f64>, val: f64, weight: f64) {
        if let Some(hist_bin_idx) = get_bin_idx(val, &self.hist_bin_edges) {
            statepack[[hist_bin_idx]] += weight;
        }
    }

    /// merge the state-packs tracked by `statepack` and other, and update
    /// `statepack` accordingly
    fn merge(&self, statepack: &mut ArrayViewMut1<f64>, other: ArrayView1<f64>) {
        for i in 0..self.n_hist_bins {
            statepack[[i]] = other[[i]];
        }
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::SingleVecComp {
            size: self.n_hist_bins,
            name: "weight",
        }
    }

    fn value_from_statepack(&self, value: &mut ArrayViewMut1<f64>, statepack: &ArrayView1<f64>) {
        for i in 0..self.n_hist_bins {
            value[[i]] = statepack[[i]];
        }
    }
}

// TODO use binary search, and have specialized version for regularly spaced bins?
/// Get the index of the bin that the squared distance falls into.
/// Returns None if its out of bounds.
///
/// # Note
/// This is only public so that it can be used in other files. It's not
/// intended to be used outside ofpublic
pub fn get_bin_idx(distance_squared: f64, squared_bin_edges: &[f64]) -> Option<usize> {
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

/// calculate the squared norm of the difference between two (mathematical) vectors
/// which are part of rust vecs that encodes a list of vectors with dimension on
/// the "slow axis"
///
/// # Note
/// This is only public so that it can be used in other crates (it isn't meant
/// to be exposed outside of the package)
pub fn squared_diff_norm(
    v1: ArrayView2<f64>,
    v2: ArrayView2<f64>,
    i1: usize,
    i2: usize,
    n_spatial_dims: usize,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..n_spatial_dims {
        sum += (v1[[k, i1]] - v2[[k, i2]]).powi(2);
    }
    sum
}

// TODO c/// computes the euclidean distance between 2 (mathematical) vectors taken from
/// `values_a` and `values_b`.
///
/// # Assumptions
/// This function assumes that spatial dimension varies along axis 0 of
/// `values_a` and `values_b` (and that it is the same value for both arrays)
pub fn diff_norm(
    values_a: ArrayView2<f64>,
    values_b: ArrayView2<f64>,
    i_a: usize,
    i_b: usize,
) -> f64 {
    // TODO come up with a better name for this function

    squared_diff_norm(values_a, values_b, i_a, i_b, values_a.shape()[0]).sqrt()
}

/// computes a dot product between two (mathematical) vectors taken from
/// `values_a` and `values_b`.
///
/// # Assumptions
/// This function assumes that spatial dimension varies along axis 0 of
/// `values_a` and `values_b` (and that it is the same value for both arrays)
pub fn dot_product(
    values_a: ArrayView2<f64>,
    values_b: ArrayView2<f64>,
    i_a: usize,
    i_b: usize,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..values_a.shape()[0] {
        sum += values_a[[k, i_a]] * values_b[[k, i_b]];
    }
    sum
}
