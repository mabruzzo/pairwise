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
//! In practice, we use [`Datum`] to package together `yᵢ` & `wᵢ`
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
//! We draw a distinction between the current state of the accumulator between
//! the actual accumulation logic.
//! - We refer to the current state of a single accumulator as the
//!   `accum_state`.
//! - The accumulation logic is encapsulated by the functions implemented by
//!   the `Reducer` trait. At the time of writing, a Reducer
//!   implements logic for modifying a single `accum_state` at a time.
//! - From the perspective of an accumulator, the `accum_state` is packaged
//!   inside within the [`AccumStateView`] & [`AccumStateViewMut`] types (the
//!   exact type depends on context)
//!
//! At a high-level, external code manages and each bin's `accum_state`. A
//! collection of `accum_state`s is usually managed by a [`StatePackViewMut`]
//! instance. Currently, the accumulators are designed to be agnostic about
//! the precise way a given `accum_state` is organized in memory. This is done
//! to give additional flexibility to the external code driving the
//! calculation.
//!
//! We will revisit this in the future once we are done architecting other
//! parts of the design.

use crate::bins;
use crate::state::{AccumStateView, AccumStateViewMut};
use ndarray::ArrayViewMut1;

/// Instances of this element are consumed by the Reducer
///
/// In the future, we might want to store vector components rather than the
/// scalar (we can convert the vector components to a scalar as necessary
/// within accumulators).
///
/// # Supporting SIMD Instructions
/// I have some doubts about how well we will be able to get autovectorization
/// to work since we are reading data from multiple pointers that we access
/// through abstractions (it may be hard to properly communicate to the
/// compiler without lots of unsafe code). The fact that the operations that
/// we want to vectorize are split across a few layers of abstraction adds
/// further complication.
///
/// To support SIMD, we probably want to make this take a generic type (we'll
/// need to define a trait for essential floating point operations) rather than
/// hardcoding the struct members to floats and have the Reducers operate
/// on these generic types. By doing this, we could pass in
/// [`DVec2`](https://docs.rs/glam/latest/glam/f64/struct.DVec2.html) from the
/// `glam` package or
/// [`std::f64x2`](https://doc.rust-lang.org/std/simd/type.f64x2.html) from
/// nightly. For this to work, we would probably to transmute the input
/// array buffers. If using `glam`'s types, we might want to look into
/// `bytemuck` to safely transmute. If using nightly, `std::simd` [has some
/// support for doing this](https://doc.rust-lang.org/std/simd/struct.Simd.html#layout-1).
///
/// # General Optimization Notes
/// We probably want to ensure that the reducer methods are inlined. If
/// they aren't inlined, then its conceivable that using a struct may
/// introduce a performance penalty, especially if we start encoding
/// (mathematical) vector components. In more detail, passing the values in
/// a struct to a function, means that the values need to be packed in memory
/// to match the struct's internal representation to communicate across the
/// function boundaries (this issue goes away when the function is inlined).
///
/// There's definitely more to this story. As I understand it,
/// platform-specific calling conventions will influence how values are passed
/// to functions even if we passed each struct-member as individual args. Thus,
/// we probably want to ensure inlining to get optimal code either way (i.e.
/// there's no real disadvantage to using the struct, *yet*).
///
/// In practice, my real concern is in the context of SIMD optimizations.
/// - I have some concerns that the packing multiple SIMD-sized data packs into
///   a struct to pass the information into an inlined function, may present
///   more of a challenge to the optimizer than just passing each SIMD-sized
///   data pack directly to the function as arguments
/// - First, while an optimizer should be able to make the same optimizations
///   in both cases, the act of packing values into a struct certainly makes
///   more work for the optimizer. It's unclear whether the "quantity of work"
///   actually matters... (it seems plausible that an optimizer might have a
///   heuristic telling it "give up" if there seems to be "too much" to avoid
///   needlessly increasing compile times)
/// - Second, I *think* the types offered by `std::simd` (e.g. `f64x2`) may
///   receive special treatment from the rust compiler to encourage
///   optimizations with SIMD operations (I may be wrong). If this is indeed
///   the case, I have some concerns that placing such types within a struct
///   *could* interfere with this treatment (due to the struct's memory
///   representation requirements).
///
/// With all that said, I think it's worthwhile use an explicit type for now.
/// I think that there are significant advantages to improving the readability
/// of the code and it would be relatively move away from using the struct
/// representation later (it would be easier to do that before we start
/// introducing more Reducers).
///
/// I don't love that this defines copy, but it's important for examples
#[derive(Clone, Copy)]
pub struct Datum {
    pub value: f64,
    pub weight: f64,
}

impl Datum {
    pub fn zeroed() -> Self {
        Datum {
            value: 0.0,
            weight: 0.0,
        }
    }
}

/// describes the output components from a single Reducer accum_state
pub enum OutputDescr {
    MultiScalarComp(&'static [&'static str]),
    SingleVecComp { size: usize, name: &'static str },
}

impl OutputDescr {
    /// the number of components to allocate per component
    pub fn n_per_accum_state(&self) -> usize {
        match self {
            Self::MultiScalarComp(names) => names.len(),
            Self::SingleVecComp { size, .. } => *size,
        }
    }
}

/// Reducers generally operate on individual `accum_state`s.
///
/// The implementation is currently inefficient, but we will refactor and try
/// to come up with better abstractions once we are done mapping out all the
/// requirements.
pub trait Reducer {
    /// the number of f64 elements needed to track the accumulator data
    fn accum_state_size(&self) -> usize;

    /// initializes the storage tracking the acumulator's state.
    ///
    /// You need to call this function before you start working with the
    /// storage. You can also use this to reset the accumulator's state since
    /// it blindly overwrites any existing values.
    ///
    /// <div class="warning">
    ///
    /// It's unfortunate that the storage needs to be explicitly initialized
    /// and it is indeed a footgun of this low-level API. This decision is a
    /// consequence of the fact that we want to maintain flexibility over the
    /// organization of the storage used for accumulators.
    ///
    /// This design rationale makes slightly more sense in the context that
    /// types implementing the [`Reducer`] trait aren't themselves
    /// accumulators, but are instead objects that provide logic for working
    /// with accumulator-storage. Maybe a better abstraction will present
    /// itself?
    ///
    /// </div>
    fn init_accum_state(&self, accum_state: &mut AccumStateViewMut);

    /// consume the value and weight to update the accum_state
    fn consume(&self, accum_state: &mut AccumStateViewMut, datum: &Datum);

    /// merge the state information tracked by `accum_state` and `other`, and
    /// update `accum_state` accordingly
    fn merge(&self, accum_state: &mut AccumStateViewMut, other: &AccumStateView);

    /// extract all output-values from a single accum_state. Expects `value` to
    /// have the shape given by `[self.n_value_comps()]` and `accum_state` to
    /// have the shape provided by `[self.accum_state_size()]`
    ///
    /// Use `self.value_prop` to interpret the meaning of each value component
    fn value_from_accum_state(&self, value: &mut ArrayViewMut1<f64>, accum_state: &AccumStateView);

    /// Describes the outputs produced from a single accum_state
    fn output_descr(&self) -> OutputDescr;
}

#[derive(Clone, Copy)]
pub struct Mean;

impl Mean {
    const TOTAL: usize = 0;
    const WEIGHT: usize = 1;

    const VALUE_MEAN: usize = 0;
    const VALUE_WEIGHT: usize = 1;
    const OUTPUT_COMPONENTS: &'static [&'static str] = &["mean", "weight"];
}

impl Reducer for Mean {
    fn accum_state_size(&self) -> usize {
        2_usize
    }

    fn init_accum_state(&self, accum_state: &mut AccumStateViewMut) {
        accum_state[Mean::TOTAL] = 0.0;
        accum_state[Mean::WEIGHT] = 0.0;
    }

    fn consume(&self, accum_state: &mut AccumStateViewMut, datum: &Datum) {
        accum_state[Mean::WEIGHT] += datum.weight;
        accum_state[Mean::TOTAL] += datum.value * datum.weight;
    }

    fn merge(&self, accum_state: &mut AccumStateViewMut, other: &AccumStateView) {
        accum_state[Mean::TOTAL] += other[Mean::TOTAL];
        accum_state[Mean::WEIGHT] += other[Mean::WEIGHT];
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::MultiScalarComp(Mean::OUTPUT_COMPONENTS)
    }

    fn value_from_accum_state(&self, value: &mut ArrayViewMut1<f64>, accum_state: &AccumStateView) {
        value[[Mean::VALUE_MEAN]] = accum_state[Mean::TOTAL] / accum_state[Mean::WEIGHT];
        value[[Mean::VALUE_WEIGHT]] = accum_state[Mean::WEIGHT];
    }
}

// TODO: refactor so that Histogram doesn't hold a vector and move to
//       pairwise_internal/accumulator.rs (to )
pub struct Histogram<BinsType: bins::BinEdges> {
    bins: BinsType,
}

// I think it's good form to have this constructor but I'm not sure that we
// really need it?
impl<B: bins::BinEdges> Histogram<B> {
    pub fn from_bin_edges(bins: B) -> Histogram<B> {
        Histogram { bins }
    }
}

impl<B: bins::BinEdges> Reducer for Histogram<B> {
    fn accum_state_size(&self) -> usize {
        self.bins.len()
    }

    /// initializes the storage tracking the acumulator's state
    fn init_accum_state(&self, accum_state: &mut AccumStateViewMut) {
        accum_state.fill(0.0);
    }

    /// consume the value and weight to update the accum_state
    fn consume(&self, accum_state: &mut AccumStateViewMut, datum: &Datum) {
        if let Some(hist_bin_idx) = self.bins.bin_index(datum.value) {
            accum_state[hist_bin_idx] += datum.weight;
        }
    }

    /// merge the state information tracked by `accum_state` and `other`, and
    /// update `accum_state` accordingly
    fn merge(&self, accum_state: &mut AccumStateViewMut, other: &AccumStateView) {
        for i in 0..self.bins.len() {
            accum_state[i] += other[i];
        }
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::SingleVecComp {
            size: self.bins.len(),
            name: "weight",
        }
    }

    fn value_from_accum_state(&self, value: &mut ArrayViewMut1<f64>, accum_state: &AccumStateView) {
        for i in 0..self.bins.len() {
            value[[i]] = accum_state[i];
        }
    }
}
