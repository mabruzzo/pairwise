//! Define basic reducer machinery (that doesn't require the standard lib)
//!
//! # Reducer Machinery
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
//! In practice, we use [`Datum`] to package together `𝒚ᵢ` & `wᵢ`
//!
//! If a statistic just summed the values of `wᵢ` and totally ignored `𝒚ᵢ`,
//! that would be equivalent to a normal histogram. Other statistics that we
//! compute can be thought of generalizations of histograms (this idea is also
//! described by scipy's
//! [binned_statistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html)
//! function.)
//!
//! ## Coming back to Reducer Machinery
//!
//! Now, the actual binning is taken care of separately. The reducer
//! machinery is responsible for computing the statistic within a single bin.
//! We draw a distinction between the current state of the reduction and the
//! actual reducer logic.
//! - We refer to the current state of a single reduction as the `accum_state`
//! - The reducer logic is encapsulated by the functions implemented by
//!   the `Reducer` trait. At the time of writing, a Reducer
//!   implements logic for modifying a single `accum_state` at a time.
//! - From the perspective of a reducer, the `accum_state` is packaged
//!   inside within the [`AccumStateView`] & [`AccumStateViewMut`] types (the
//!   exact type depends on context)
//!
//! At a high-level, external code manages and each bin's `accum_state`. A
//! collection of `accum_state`s is usually managed by a [`StatePackViewMut`]
//! instance. Currently, the reducers are designed to be agnostic about
//! the precise way a given `accum_state` is organized in memory. This is done
//! to give additional flexibility to the external code driving the
//! calculation.
//!
//! We will revisit this in the future once we are done architecting other
//! parts of the design.

use crate::bins;
use crate::state::{AccumStateView, AccumStateViewMut};
use core::marker::PhantomData;
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
    // TODO: should this really be public to all?
    pub value: [f64; 3],
    pub(crate) weight: f64,
}

impl Datum {
    pub fn zeroed() -> Self {
        Datum {
            value: [0.0, 0.0, 0.0],
            weight: 0.0,
        }
    }

    // this is intended primarily for testing and migration
    #[inline]
    pub fn from_scalar_value(value: f64, weight: f64) -> Self {
        Datum {
            value: [value, 0.0, 0.0],
            weight,
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

/// this encodes the logic to convert the vector value in a [`Datum`] from
/// a vector to a scalar.
pub trait ScalarizeOp: Copy + Clone {
    /// returns the value of datum after coercing to a scalar
    fn scalarized_value(datum: &Datum) -> f64;
}

#[derive(Copy, Clone)]
pub struct TakeComp0;

impl ScalarizeOp for TakeComp0 {
    #[inline(always)]
    fn scalarized_value(datum: &Datum) -> f64 {
        datum.value[0]
    }
}

/// To be used with simple correlation
#[derive(Clone, Copy)]
pub struct ComponentSum;

impl ScalarizeOp for ComponentSum {
    #[inline(always)]
    fn scalarized_value(datum: &Datum) -> f64 {
        datum.value[0] + (datum.value[1] + datum.value[2])
    }
}

// the following Reducers all "scalarize" the value in Datum before actually
// the reduction. In other words, they somehow map it from a vector to a
// scalar.

#[derive(Clone, Copy)]
pub struct ScalarMean<T: ScalarizeOp>(PhantomData<T>);

impl<T: ScalarizeOp> ScalarMean<T> {
    const TOTAL: usize = 0;
    const WEIGHT: usize = 1;

    const VALUE_MEAN: usize = 0;
    const VALUE_WEIGHT: usize = 1;
    const OUTPUT_COMPONENTS: &'static [&'static str] = &["mean", "weight"];

    #[inline(always)]
    pub fn new() -> Self {
        Self(PhantomData::<T>)
    }
}

// we are only implementing this to silence clippy::new_without_default
impl<T: ScalarizeOp> Default for ScalarMean<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ScalarizeOp> Reducer for ScalarMean<T> {
    fn accum_state_size(&self) -> usize {
        2_usize
    }

    fn init_accum_state(&self, accum_state: &mut AccumStateViewMut) {
        accum_state[Self::TOTAL] = 0.0;
        accum_state[Self::WEIGHT] = 0.0;
    }

    fn consume(&self, accum_state: &mut AccumStateViewMut, datum: &Datum) {
        accum_state[Self::WEIGHT] += datum.weight;
        accum_state[Self::TOTAL] += T::scalarized_value(datum) * datum.weight;
    }

    fn merge(&self, accum_state: &mut AccumStateViewMut, other: &AccumStateView) {
        accum_state[Self::TOTAL] += other[Self::TOTAL];
        accum_state[Self::WEIGHT] += other[Self::WEIGHT];
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::MultiScalarComp(Self::OUTPUT_COMPONENTS)
    }

    fn value_from_accum_state(&self, value: &mut ArrayViewMut1<f64>, accum_state: &AccumStateView) {
        value[[Self::VALUE_MEAN]] = accum_state[Self::TOTAL] / accum_state[Self::WEIGHT];
        value[[Self::VALUE_WEIGHT]] = accum_state[Self::WEIGHT];
    }
}

pub type Comp0Mean = ScalarMean<TakeComp0>;
pub type ComponentSumMean = ScalarMean<ComponentSum>;

#[derive(Clone)]
pub struct ScalarHistogram<BinsType: bins::BinEdges, T: ScalarizeOp> {
    bins: BinsType,
    _dummy: PhantomData<T>,
}

// I think it's good form to have this constructor but I'm not sure that we
// really need it?
impl<B: bins::BinEdges, T: ScalarizeOp> ScalarHistogram<B, T> {
    pub fn from_bin_edges(bins: B) -> Self {
        ScalarHistogram {
            bins,
            _dummy: PhantomData::<T>,
        }
    }
}

impl<B: bins::BinEdges, T: ScalarizeOp> Reducer for ScalarHistogram<B, T> {
    fn accum_state_size(&self) -> usize {
        self.bins.n_bins()
    }

    /// initializes the storage tracking the acumulator's state
    fn init_accum_state(&self, accum_state: &mut AccumStateViewMut) {
        accum_state.fill(0.0);
    }

    /// consume the value and weight to update the accum_state
    fn consume(&self, accum_state: &mut AccumStateViewMut, datum: &Datum) {
        if let Some(hist_bin_idx) = self.bins.bin_index(T::scalarized_value(datum)) {
            accum_state[hist_bin_idx] += datum.weight;
        }
    }

    /// merge the state information tracked by `accum_state` and `other`, and
    /// update `accum_state` accordingly
    fn merge(&self, accum_state: &mut AccumStateViewMut, other: &AccumStateView) {
        for i in 0..self.bins.n_bins() {
            accum_state[i] += other[i];
        }
    }

    fn output_descr(&self) -> OutputDescr {
        OutputDescr::SingleVecComp {
            size: self.bins.n_bins(),
            name: "weight",
        }
    }

    fn value_from_accum_state(&self, value: &mut ArrayViewMut1<f64>, accum_state: &AccumStateView) {
        for i in 0..self.bins.n_bins() {
            value[[i]] = accum_state[i];
        }
    }
}

pub type Comp0Histogram<B> = ScalarHistogram<B, TakeComp0>;
pub type ComponentSumHistogram<B> = ScalarHistogram<B, ComponentSum>;
