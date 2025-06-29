//! Miscellaneous machinery used to implement the package
//!
//! # Accumulation Machinery
//!
//! The architecture of this crate is built upon the concept of accumulation.
//!
//! ## Broader Context
//!
//! The idea is that we want to compute binned statistics for a stream of
//! values, where each value is a vector `𝒗 = [𝒙,𝒚,w]`, where `𝒙` and `𝒚`
//! can themselves be vectors and `w` is always a scalar. Let's define the
//! ith element of the stream as `𝒗ᵢ = [𝒙ᵢ, 𝒚ᵢ, wᵢ]`. In more detail,
//! - `𝒙ᵢ` is used for binning
//! - `𝒚ᵢ` is the quantity that contributes to the statistic
//! - `wᵢ` is the weighting applied to `𝒚ᵢ`.
//!
//! For simplicity, let's assume `𝒚ᵢ` is a scalar `yᵢ` (it may be useful to
//! come back to this for longitudinal statistics and tensors).
//!
//! If a statistic just summed the values of `wᵢ` and totally ignored `yᵢ`,
//! that would be equivalent to a normal histogram. Other statistics that we
//! compute can be thought of generalizations of histograms (this idea is also
//! described by scipy's
//! [binned_statistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html)
//! function.)
//!
//! ## Coming back to Accumulation Machinery
//!
//! Now, the actual binning is taken care of separately. The accumulation
//! machinery is responsible for computing the statistic within a single bin.
//! We draw a distinction the current state of the accumulator between the
//! actual accumulation logic.
//! - We refer to the current state of the accumulator as its state-pack.
//! - The accumulation logic is encapsulated by the functions implemented by
//!   the `Accumulator` trait. At the time of writing, an Accumulator
//!   implements logic for modifyin a single state-pack.
//!
//! At a high-level, external code tracks separate state-packs for each bin,
//! and invokes accumulators for separate state-packs. The external code needs
//! flexibility for how it stores the state-packs.
//!
//! We will revisit this in the future once we are done architecting other
//! parts of the design.

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

/// describes the output components from a single Accumulator statepack
pub enum OutputDescr {
    MultiScalarComp(&'static [&'static str]),
    SingleVecComp { size: usize, name: &'static str },
}

impl OutputDescr {
    /// the number of components to allocate per component
    pub fn n_per_statepack(&self) -> usize {
        match self {
            Self::MultiScalarComp(names) => names.len(),
            Self::SingleVecComp { size, .. } => *size,
        }
    }
}

// Accumulators generally operate on individual state-packs (the implementation
// is currently inefficient, but we will refactor and try to come up with
// better abstractions once we are done mapping out all the requirements)
//
// In the context of the larger package, there will be n accumulators
pub trait Accumulator {
    /// the number of f64 elements needed to track the accumulator data
    fn statepack_size(&self) -> usize;

    /// initializes the storage tracking the acumulator's state
    fn reset_statepack(&self, statepack: &mut ArrayViewMut1<f64>);

    /// consume the value and weight to update the statepack
    fn consume(&self, statepack: &mut ArrayViewMut1<f64>, val: f64, weight: f64);

    /// merge the state-packs tracked by `statepack` and other, and update
    /// `statepack` accordingly
    fn merge(&self, statepack: &mut ArrayViewMut1<f64>, other: ArrayView1<f64>);

    /// extract all output-values from a single statepack. Expects `value` to
    /// have the shape given by `[self.n_value_comps()]` and `statepack` to
    /// have the shape provided by `[self.statepack_size()]`
    ///
    /// Use `self.value_prop` to interpret the meaning of each value component
    fn value_from_statepack(&self, value: &mut ArrayViewMut1<f64>, statepack: &ArrayView1<f64>);

    /// Describes the outputs produced from a single statepack
    fn output_descr(&self) -> OutputDescr;

    // the functions down below apply to multiple state-packs at a time (they
    // probably should not be part of this trait

    /// Maybe we should detach this...
    fn values_from_statepacks(
        &self,
        values: &mut ArrayViewMut2<f64>,
        statepacks: &ArrayView2<f64>,
    ) {
        // get the number of derivable quantities per pack (this is slow and dumb)

        // sanity checks:
        assert!(values.shape()[0] == self.output_descr().n_per_statepack());
        assert!(statepacks.shape()[0] == self.statepack_size());
        assert!(values.shape()[1] == statepacks.shape()[1]);

        for i in 0..values.shape()[1] {
            self.value_from_statepack(
                &mut values.index_axis_mut(Axis(1), i),
                &statepacks.index_axis(Axis(1), i),
            );
        }
    }
}

pub struct Mean;

impl Mean {
    const TOTAL: usize = 0;
    const WEIGHT: usize = 1;

    const VALUE_MEAN: usize = 0;
    const VALUE_WEIGHT: usize = 1;
    const OUTPUT_COMPONENTS: &'static [&'static str] = &["mean", "weight"];
}

impl Accumulator for Mean {
    fn statepack_size(&self) -> usize {
        2_usize
    }

    fn reset_statepack(&self, statepack: &mut ArrayViewMut1<f64>) {
        statepack[[Mean::TOTAL]] = 0.0;
        statepack[[Mean::WEIGHT]] = 0.0;
    }

    fn consume(&self, statepack: &mut ArrayViewMut1<f64>, val: f64, weight: f64) {
        statepack[[Mean::WEIGHT]] += weight;
        statepack[[Mean::TOTAL]] += val * weight;
    }

    fn merge(&self, statepack: &mut ArrayViewMut1<f64>, other: ArrayView1<f64>) {
        statepack[[Mean::TOTAL]] += other[[Mean::TOTAL]];
        statepack[[Mean::WEIGHT]] += other[[Mean::WEIGHT]];
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::MultiScalarComp(Mean::OUTPUT_COMPONENTS)
    }

    fn value_from_statepack(&self, value: &mut ArrayViewMut1<f64>, statepack: &ArrayView1<f64>) {
        value[[Mean::VALUE_MEAN]] = statepack[[Mean::TOTAL]] / statepack[[Mean::WEIGHT]];
        value[[Mean::VALUE_WEIGHT]] = statepack[[Mean::WEIGHT]];
    }
}

pub struct Histogram {
    // maybe call these hist_bucket_edges?
    // in the future, we may want to make this into a slice
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{NewAxis, s};
    use std::collections::HashMap;

    // this is inefficient, but it gets the job done for now
    //
    // this is probably an indication that we could improve the Accumulator API
    fn _get_output_single(
        accum: &impl Accumulator,
        stateprop: &ArrayView1<f64>,
    ) -> HashMap<&'static str, Vec<f64>> {
        _get_output(accum, &stateprop.slice(s![.., NewAxis]))
    }

    fn _get_output(
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

    #[test]
    fn mean_consume_once() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);

        accum.consume(&mut statepack, 4.0, 1.0);
        let value_map = _get_output_single(&accum, &statepack.view());

        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 1.0);
    }

    #[test]
    fn mean_consume_twice() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);

        accum.consume(&mut statepack, 4.0, 1.0);
        accum.consume(&mut statepack, 8.0, 1.0);

        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["mean"][0], 6.0);
        assert_eq!(value_map["weight"][0], 2.0);
    }

    #[test]
    fn merge() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);
        accum.consume(&mut statepack, 4.0, 1.0);
        accum.consume(&mut statepack, 8.0, 1.0);

        let mut storage_other = [0.0, 0.0];
        let mut statepack_other = ArrayViewMut1::from_shape([2], &mut storage_other).unwrap();
        accum.reset_statepack(&mut statepack_other);
        accum.consume(&mut statepack_other, 1.0, 1.0);
        accum.consume(&mut statepack_other, 3.0, 1.0);

        accum.merge(&mut statepack, statepack_other.view());

        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 4.0);
    }

    #[test]
    fn hist_invalid_hist_edges() {
        assert!(Histogram::new(&[0.0]).is_err());
    }
    #[test]
    fn hist_nonmonotonic() {
        assert!(Histogram::new(&[1.0, 0.0]).is_err());
    }

    #[test]
    fn hist_consume() {
        let accum = Histogram::new(&[0.0, 1.0, 2.0]).unwrap();

        let mut storage = [0.0, 0.0];
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);

        accum.consume(&mut statepack, 0.5, 1.0);
        accum.consume(&mut statepack, -50.0, 1.0);
        accum.consume(&mut statepack, 1000.0, 1.0);

        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 0.0);

        accum.consume(&mut statepack, 1.1, 5.0);
        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 5.0);
    }
}
