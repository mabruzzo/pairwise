//! Define basic accumulator machinery (that doesn't require the standard lib)
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
