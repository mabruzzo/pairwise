//! Miscellaneous machinery used to implement the package

use ndarray::{ArrayView1, ArrayViewMut1};

use pairwise_internal::{Accumulator, OutputDescr, get_bin_idx};

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
