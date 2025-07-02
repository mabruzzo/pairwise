//! Miscellaneous machinery used to implement the package

use std::collections::HashMap;

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use pairwise_internal::{Accumulator, OutputDescr, get_bin_idx, squared_diff_norm};

/// compute the output quantities from an Accumulator's state properties and
/// return the result in a HashMap.
///
/// # Notes
/// This is primarily used for testing.
///
/// TODO: I'm not sure I really want this to be a part of the standard API.
///       Before the 1.0 release, we should either move this to a private
///       testing_helpers crate OR we should explicitly decide to make this
///       part of the public API.
pub fn get_output(
    accum: &impl Accumulator,
    stateprops: &ArrayView2<f64>,
) -> HashMap<&'static str, Vec<f64>> {
    let description = accum.output_descr();
    let n_bins = stateprops.shape()[1];
    let n_comps = description.n_per_statepack();

    let mut buffer = Vec::new();
    buffer.resize(n_comps * n_bins, 0.0);
    let mut buffer_view = ArrayViewMut2::from_shape([n_comps, n_bins], &mut buffer).unwrap();
    accum.values_from_statepacks(&mut buffer_view, stateprops);

    match description {
        OutputDescr::MultiScalarComp(names) => {
            let _to_vec = |row: ArrayView1<f64>| row.iter().cloned().collect();
            let row_iter = buffer_view.rows().into_iter().map(_to_vec);
            HashMap::from_iter(names.iter().cloned().zip(row_iter))
        }
        OutputDescr::SingleVecComp { name, .. } => HashMap::from([(name, buffer)]),
    }
}

/// computes the euclidean distance between 2 (mathematical) vectors taken from
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
            statepack[[i]] += other[[i]];
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
