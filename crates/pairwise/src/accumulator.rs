use std::collections::HashMap;

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

use pairwise_nostd_internal::{
    AccumStateView, AccumStateViewMut, Accumulator, Datum, OutputDescr, StatePackViewMut,
    get_bin_idx,
};

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
    statepack: &StatePackViewMut,
) -> HashMap<&'static str, Vec<f64>> {
    get_output_from_statepack_array(accum, &statepack.as_array_view())
}

// todo: figure out how to remove me before the first release
//
// we're holding onto this for now since the name of the standard type we use
// to describe a statepack, [`StatePackViewMut`] implies that the statepack
// is mutable. This operates on immutable state data, which makes sense from
// a design perspective.
//
// There are 2 obvious solutions:
// 1. create an object, in the pairwise crate (rather than in the no_std
//     crate), that actually owns the StatePack data, and pass that in as a
//     reference. If we do that, then `StatePackViewMut` would typically
//     view the data inside of the type.
// 2. Alternatively, we might rename `StatePackViewMut` to something a little
//    more generic...
//
// I'm not in a rush to do this since a different solution may present itself
// as we work on developing an ergonomic higher-level API
pub fn get_output_from_statepack_array(
    accum: &impl Accumulator,
    statepack_data: &ArrayView2<f64>,
) -> HashMap<&'static str, Vec<f64>> {
    let description = accum.output_descr();
    let n_bins = statepack_data.shape()[1];
    let n_comps = description.n_per_accum_state();

    // todo: the rest of this function could definitely be optimized!
    let mut buffer = vec![0.0; n_comps * n_bins];
    let mut buffer_view = ArrayViewMut2::from_shape([n_comps, n_bins], &mut buffer).unwrap();
    for i in 0..n_bins {
        accum.value_from_accum_state(
            &mut buffer_view.index_axis_mut(Axis(1), i),
            &AccumStateView::from_array_view(statepack_data.index_axis(Axis(1), i)),
        );
    }

    match description {
        OutputDescr::MultiScalarComp(names) => {
            let _to_vec = |row: ArrayView1<f64>| row.iter().cloned().collect();
            let row_iter = buffer_view.rows().into_iter().map(_to_vec);
            HashMap::from_iter(names.iter().cloned().zip(row_iter))
        }
        OutputDescr::SingleVecComp { name, .. } => HashMap::from([(name, buffer)]),
    }
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
    fn accum_state_size(&self) -> usize {
        self.n_hist_bins
    }

    /// initializes the storage tracking the acumulator's state
    fn init_accum_state(&self, accum_state: &mut AccumStateViewMut) {
        accum_state.fill(0.0);
    }

    /// consume the value and weight to update the accum_state
    fn consume(&self, accum_state: &mut AccumStateViewMut, datum: &Datum) {
        if let Some(hist_bin_idx) = get_bin_idx(datum.value, &self.hist_bin_edges) {
            accum_state[hist_bin_idx] += datum.weight;
        }
    }

    /// merge the state information tracked by `accum_state` and `other`, and
    /// update `accum_state` accordingly
    fn merge(&self, accum_state: &mut AccumStateViewMut, other: &AccumStateView) {
        for i in 0..self.n_hist_bins {
            accum_state[i] += other[i];
        }
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::SingleVecComp {
            size: self.n_hist_bins,
            name: "weight",
        }
    }

    fn value_from_accum_state(&self, value: &mut ArrayViewMut1<f64>, accum_state: &AccumStateView) {
        for i in 0..self.n_hist_bins {
            value[[i]] = accum_state[i];
        }
    }
}
