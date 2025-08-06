//! This implements a quick-and-dirty first attempt at the "high-level" API
//!
//! # A bird's eye view (motivation)
//! - we leverage Rust's generics and type system to write a lot of highly
//!   specialized code paths
//! - however, there is a large combinatorical problem with selecting the
//!   appropriate configuration.
//!   - we don't want to demand that users be familiar with all of the
//!     highly tuned implementation paths. A similar philosophy applies to
//!     other crates and libraries (e.g. the regex crate)
//!   - we want a more dynamical interface, where you can treat different
//!     variations of a calculation in a very similar manner
//! - furthermore, we need to make all of the variants easy to dynamically
//!   access to expose them to a python library.
//!
//! # Guiding Principles:
//! - It's clear that we are going to use an accumulator-oriented API
//! - We need to be mindful of 2 important use-cases: (i) the fact that we
//!   want people to be able easily build MPI programs that use this library
//!   as a building block and (ii) the fact that people may be computing
//!   multiple statistics. This means that
//!   1. our accumulator API must be easy to serialize (for communicating
//!      between machines)
//!      - for speed purposes, we probably want to serialize accumulator
//!        config & accumulator state separately. We might do the former
//!        with JSON. The latter holds more information, but can probably
//!        be treated as a stream of bytes
//!   2. we need to think about "equality checking"  and user expectations:
//!      - if we have a single monolithic accumulator object, containing state
//!        & configuration, the users would probably expect that we'll check
//!        equality to prevent the merging of unrelated errors.
//!      - if we store the state and configuration separately, I think it
//!        becomes much more obvious that keeping track of this is the user's
//!        responsibility
//!      - it turns out that equality checking can be tricky!
//!
//! - GPU support consideration. This is a very low priority, but we should
//!   pay some attention to what this may entail... In other words, we don't
//!   want to throw everything away. In more detail"
//!   - full GPU support requires a lot of thought. There are lots of things
//!     that we *could* choose to support.
//!   - in my mind, to ship experimental GPU support (especially if we want
//!     to provide it in a python package on PyPI) we probably want to put the
//!     GPU accelerated functionality in a shared library they we treat as a
//!     Run-Time Loadable Extension/Plugin.
//!     - Basically, we'd try to use `dlopen` to bring in functions (see the
//!       libloading crate). This would let us conditionally provide the
//!       functionality **if the hardware and dependencies are present on the
//!       system**
//!     - this could also act as a mechanism for optionally providing SIMD
//!       accelerated functions
//!   - doing this wouldn't be terribly difficult
//!
//!
//! # Other Thoughts
//! I fully expect that this initial API will be suboptimal. Personal
//! experienc with the Grackle project has clearly established that API design
//! is very hard! To add to this:
//! - I have minimal experience designing APIs for languages with
//!   sophisticated type-systems like C++/Rust
//! - the fact that I'm focusing so much on binding to python, and considering
//!   other constrains probably means that the design may be a little
//!   suboptimal for rust.
//!
//! Finally, there are 2 topics that I'm intentionally ignoring in this first
//! implementation. There are hugely important, but I can't keep track of
//! everything and need to start writing code! I am hoping that choices become
//! more apparent later on:
//! 1. How do we handle data-ownership and minimize unnecessary copies? This
//!    initial implementation copies arrays, particularly in the context of
//!    bin edges, very aggressively!
//!    - for context, the underlying algorithms are designed to operate on
//!      slices.
//!    - this has important implications for serialization
//!    - I'm being aggressive to simplify things, because the actual
//!      calculation can be **VERY** expensive "at scale" (i.e. a little extra
//!      overhead won't affect initial library adoption)
//! 2. relatedly, should we be organizing the API within some overarching
//!    context that manages memory? This is a very C oriented way of thinking
//!    (basically all "objects" a user interacts with would be handles). My
//!    gut instinct is "no" (we can get by with "smart pointers"), but it
//!    would certainly simplify some things...
use crate::{
    Error,
    apply::apply_accum,
    reducers::{EuclideanNormHistogram, EuclideanNormMean, get_output},
};
use std::{collections::HashMap, sync::LazyLock};

use pairwise_nostd_internal::{
    ComponentSumHistogram, ComponentSumMean, IrregularBinEdges, PairOperation, Reducer,
    RegularBinEdges, StatePackView, StatePackViewMut, UnstructuredPoints, merge_full_statepacks,
    reset_full_statepack, validate_bin_edges,
};

// current design:
// -> we have an `Accumulator`, which is composed of an `AccumulatorDescr` &
//    an `AccumulatorData`
// -> `AccumulatorDescr` tracks distance bin edges, the Reducer, and the
//    `PairOperation` instance
//    -> I think this may be a little too "coarse."
//       -> I always liked the idea of associating multiple accumulators with
//          a single set of distance bin edges
//       -> there is more stuff to validate when we call `Accumulator::merge``
//       -> but, we've got to start *somewhere*
//    -> To start, we are going to store a copy of the configuration
//       within `AccumulatorDescr` and use it for equality-checks and for
//       serialization (it's suboptimal, but it simplifies a lot about the
//       initial implementation)

/// Represents an accumulator
///
/// # Note
/// I remain somewhat unconvinced that we really want this type to exist. It
/// may be better to just have people use the constituent parts
///
/// # Implementation Note
/// This should probably not implement [`Clone`]. See [`AccumulatorData`] for
/// more details
pub struct Accumulator {
    data: AccumulatorData,
    descr: AccumulatorDescr,
}

impl Accumulator {
    /// merge the state information tracked by `accum_state` and `other`, and
    /// update `accum_state` accordingly
    pub fn merge(&mut self, other: &Accumulator) -> Result<(), Error> {
        if self.descr == other.descr {
            self.descr
                .reducer
                .merge(&mut self.data.mut_view(), &other.data.view());
            Ok(())
        } else {
            todo!("create and return the appropriate error type");
        }
    }

    /// compute the output quantities from the accumulator's state and return
    /// the result in a HashMap.
    ///
    /// # Note
    /// We should probably let the caller provide the output memory.
    pub fn get_output(&self) -> HashMap<&'static str, Vec<f64>> {
        self.descr.reducer.get_output(&self.data.view())
    }

    /// Reset the tracked accumulator state
    pub fn reset_data(&mut self) {
        self.descr
            .reducer
            .reset_full_statepack(&mut self.data.mut_view());
    }
}

// this is the logic that actually executes the reductions
impl Accumulator {
    // a compelling case could be made that this should be a standalone fn
    pub fn collect_unstructured_contributions<'a>(
        &mut self,
        points_a: UnstructuredPoints<'a>,
        points_b: Option<UnstructuredPoints<'a>>,
    ) -> Result<(), Error> {
        self.launch_reduction_helper(SpatialInfo::Unstructured { points_a, points_b })
    }

    fn launch_reduction_helper<'a>(&mut self, spatial_info: SpatialInfo<'a>) -> Result<(), Error> {
        let reducer = &self.descr.reducer;
        let Some(ref squared_distance_edges) = self.descr.config.squared_distance_bin_edges else {
            panic!("Bug:this should be unreachable!")
        };

        reducer.exec_reduction(
            &mut self.data.mut_view(),
            spatial_info,
            squared_distance_edges,
        )
    }
}

/// Holds the data associated with an accumulator
///
/// # Implementation Note
/// It would be **really** nice for this type to be able to wrap externally
/// allocated data. In my mind there are 3 ways to do this:
/// 1. Hold an enum where one variant holds a [`Vec`] and the other variant
///    holds a mutable slice. In this case we would need to associate a
///    lifetime parameter with the type
/// 2. Hold an enum where one variant holds a [`Vec`] and the other variant
///    tracks `Arc<RefCell<T>>`, where `T` is a custom type that can be
///    converted to a slice
/// 3. Some kind of registry approach (we would need to associate a lifetime
///    parameter with this type)
/// 4. We could have some kind of unsafe api where we pass in a pointer and
///    have this type take ownership (in addition to the normal vector
///    representation)
///
/// The downside of introducing a lifetime parameter is that its "infectious".
/// The [`Accumulator`] will also need to have a lifetime parameter, since it
/// holds this type.
///
/// For now, we explicitly choose not to derive [`Clone`] since some of these
/// choices are somewhat incompatible with cloning.
struct AccumulatorData {
    // it would be nice to hold externally allocated data...
    // -> Maybe the answer is to create an enum with 2 variants?
    //    - the 1st variant is `Vec<f64>`
    //    - the other variant is
    data: Vec<f64>,
    n_state: usize,
    state_size: usize,
}

impl AccumulatorData {
    // I'm hesitant to implement DerefMut because it forces us to publicly
    // expose StatePackViewMut
    fn mut_view<'a>(&'a mut self) -> StatePackViewMut<'a> {
        StatePackViewMut::from_slice(self.n_state, self.state_size, self.data.as_mut_slice())
    }

    fn view<'a>(&'a self) -> StatePackView<'a> {
        StatePackView::from_slice(self.n_state, self.state_size, self.data.as_slice())
    }
}

/// Encapsulates the accumulator properties
struct AccumulatorDescr {
    config: Config,
    reducer: Box<dyn WrappedReducer>,
}

impl Clone for AccumulatorDescr {
    fn clone(&self) -> AccumulatorDescr {
        AccumulatorDescr {
            config: self.config.clone(),
            // since self was already constructed from self.config, we should
            // be able to construct the new instance from config
            reducer: wrapper_reducer_from_config(&self.config).expect("there is a bug"),
        }
    }
}

impl PartialEq for AccumulatorDescr {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
    }
}

impl Eq for AccumulatorDescr {}

// Our current approach leverages dynamic dispatch. My concern with an
// enum-based approach:
// - it won't very well with the dynanamic plugin approach that is described
//   up above..
// - readability and maintainability...
//   - there are "a lot" of variations to support. Combinatorics from the
//     number of available "knobs" make this larger than just the number of
//     available Reducers
//   - I think the enum code would be far more scattered
// I'm happy to be proven wrong!

/// A wrapper type that only exists so we can implement the `Eq` trait.
///
/// Ordinarily, [`f64`], and by extension `Vec<f64>` doesn't implement `Eq`
/// since `NaN` != `NaN`. We can implement it here since
/// [`pairwise_nostd_internal::validate_bin_edges`] ensures there aren't any
/// `NaN` values
#[derive(Clone)]
struct ValidatedBinEdgeVec(Vec<f64>);

impl ValidatedBinEdgeVec {
    fn new(edges: Vec<f64>) -> Result<Self, &'static str> {
        validate_bin_edges(&edges)?;
        Ok(Self(edges))
    }
}

impl PartialEq for ValidatedBinEdgeVec {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ValidatedBinEdgeVec {}

#[derive(Clone, PartialEq, Eq)]
enum BinEdgeSpec {
    Vec(ValidatedBinEdgeVec),
    Regular(RegularBinEdges),
}

/// A configuration object
#[derive(Clone, PartialEq, Eq)]
struct Config {
    reducer_name: String,
    // I'm fairly confident that supporting the Histogram with the
    // [`IrregularBinEdges`] type introduces self-referential struct issues
    // (at a slightly higher level)
    hist_reducer_bucket: Option<RegularBinEdges>,
    // eventually, we should add an option for variance

    // I'm not so sure the following should actually be tracked in this struct
    // TODO: this shouldn't be an option
    squared_distance_bin_edges: Option<BinEdgeSpec>,
}

/// an internal type that will be used to encode the spatial information
#[derive(Clone)]
enum SpatialInfo<'a> {
    Unstructured {
        points_a: UnstructuredPoints<'a>,
        points_b: Option<UnstructuredPoints<'a>>,
    },
    //Cartesian{ // <- (to be uncommented)
    //    block_a: CartesianBlock<'a>,
    //    block_b: Option<CartesianBlock<'a>>,
    //    cell_width: CellWidth
    //}
}

// making this a type-alias wasn't have the appropriate effect (we weren't
// properly coercing closures to function pointers)
struct MkWrappedReducerFn(fn(&Config) -> Result<Box<dyn WrappedReducer>, Error>);

fn build_registry() -> HashMap<String, MkWrappedReducerFn> {
    // - "2pcf", "hist_cf"
    // - "hist_astro_sf1", "astro_sf1", "astro_sf2", "sf2_tensor",
    //   "sf3_tensor"
    // - "longitudinal_sf2", "longitudinal_sf3"
    //let func = ;
    let mut out: HashMap<String, MkWrappedReducerFn> = HashMap::from([
        // correlation function machinery
        (
            "hist_cf".to_owned(),
            MkWrappedReducerFn(|conf: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                let Some(ref edges) = conf.hist_reducer_bucket else {
                    return Err(Error::bucket_edges(conf.reducer_name.clone(), true));
                };
                Ok(Box::new(WrappedReducerImpl {
                    reducer: ComponentSumHistogram::from_bin_edges(edges.clone()),
                    pair_op: PairOperation::ElementwiseMultiply,
                }) as Box<dyn WrappedReducer>)
            }),
        ),
        (
            "2pcf".to_owned(),
            MkWrappedReducerFn(|conf: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                if conf.hist_reducer_bucket.is_some() {
                    return Err(Error::bucket_edges(conf.reducer_name.clone(), false));
                }
                Ok(Box::new(WrappedReducerImpl {
                    reducer: ComponentSumMean::new(),
                    pair_op: PairOperation::ElementwiseMultiply,
                }) as Box<dyn WrappedReducer>)
            }),
        ),
        // shift to structure functions
        (
            "hist_astro_sf1".to_owned(),
            MkWrappedReducerFn(|conf: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                let Some(ref edges) = conf.hist_reducer_bucket else {
                    return Err(Error::bucket_edges(conf.reducer_name.clone(), true));
                };
                Ok(Box::new(WrappedReducerImpl {
                    reducer: EuclideanNormHistogram::from_bin_edges(edges.clone()),
                    pair_op: PairOperation::ElementwiseSub,
                }) as Box<dyn WrappedReducer>)
            }),
        ),
        (
            "astro_sf1".to_owned(),
            MkWrappedReducerFn(|conf: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                if conf.hist_reducer_bucket.is_some() {
                    return Err(Error::bucket_edges(conf.reducer_name.clone(), false));
                }
                Ok(Box::new(WrappedReducerImpl {
                    reducer: EuclideanNormMean::new(),
                    pair_op: PairOperation::ElementwiseSub,
                }) as Box<dyn WrappedReducer>)
            }),
        ),
    ]);
    out
}

static REDUCER_MAKER_REGISTRY: LazyLock<HashMap<String, MkWrappedReducerFn>> =
    LazyLock::new(build_registry);

fn wrapper_reducer_from_config(config: &Config) -> Result<Box<dyn WrappedReducer>, Error> {
    let name = &config.reducer_name;
    if let Some(func) = REDUCER_MAKER_REGISTRY.get(name) {
        func.0(config)
    } else {
        Err(Error::reducer_name(
            name.clone(),
            REDUCER_MAKER_REGISTRY.keys().cloned().collect(),
        ))
    }
}

trait WrappedReducer {
    // (it would be nice if other_binned_statepack could pass in a type that is
    // more clearly immutable than StatePackViewMut)
    fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
    );

    // we should probably let the caller provide the output memory
    fn get_output(&self, binned_statepack: &StatePackView) -> HashMap<&'static str, Vec<f64>>;

    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut);

    fn accum_state_size(&self) -> usize;

    // we probably need to pass other arguments...
    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        squared_distance_bin_edges: &BinEdgeSpec,
    ) -> Result<(), Error>;
}

/// Represents a wrapped [`Reducer`].
///
/// # Note
/// To support [`ScalarHistogram`] with the [`IrregularBinEdges`] type will be
/// annoying. We will have to play a "game," where the [`Reducer`] doesn't
/// have persistent state. Instead, we will have to pass in the config to each
/// method and reconstruct the reducer exactly when we need it...
struct WrappedReducerImpl<R: Reducer + Clone> {
    reducer: R,
    pair_op: PairOperation,
}

impl<R: Reducer + Clone> WrappedReducer for WrappedReducerImpl<R> {
    fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
    ) {
        merge_full_statepacks(&self.reducer, binned_statepack, other_binned_statepack);
    }

    // we should probably let the caller provide the output memory
    // (it would be nice if we could pass in a type that is more clearly
    // immutable than StatePackViewMut)
    fn get_output(&self, binned_statepack: &StatePackView) -> HashMap<&'static str, Vec<f64>> {
        get_output(&self.reducer, binned_statepack)
    }

    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut) {
        reset_full_statepack(&self.reducer, binned_statepack)
    }

    fn accum_state_size(&self) -> usize {
        self.reducer.accum_state_size()
    }

    // we probably need to pass other arguments...
    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        squared_distance_bin_edges: &BinEdgeSpec,
    ) -> Result<(), Error> {
        match spatial_info {
            SpatialInfo::Unstructured {
                ref points_a,
                ref points_b,
            } => {
                match squared_distance_bin_edges {
                    BinEdgeSpec::Vec(v) => {
                        // there shouldn't be any problem unwrapping (since was already validated)
                        let edges = IrregularBinEdges::new(v.0.as_slice()).expect("this is a bug");
                        apply_accum(
                            binned_statepack,
                            &self.reducer,
                            points_a,
                            points_b.as_ref(),
                            &edges,
                            self.pair_op,
                        )
                    }
                    BinEdgeSpec::Regular(edges) => apply_accum(
                        binned_statepack,
                        &self.reducer,
                        points_a,
                        points_b.as_ref(),
                        edges,
                        self.pair_op,
                    ),
                }
            }
        }
    }
}

/*
/// Describes the accumulator for a two-point calculation.
///
/// Holds info, like the Reducer instance, and the distance bin edges.
///
/// # Implementation Note
/// I'm not sure we actually want to include the distance bin edges
*/

// ugh... its inelegant that this is separate from BinEdgeSpec
enum RawBinEdgeSpec {
    Vec(Vec<f64>), // unvalidated
    Regular(RegularBinEdges),
}

#[derive(Default)]
pub struct AccumulatorBuilder {
    // ugh... its very inelegant that we aren't directly modifying a Config
    // object!
    calc_kind: Option<String>,
    bucket_edges: Option<RawBinEdgeSpec>,
    distance_sqr_bin_edges: Option<RawBinEdgeSpec>,
}

impl AccumulatorBuilder {
    // unclear whether we should pass any args to new. It doesn't seem to be
    // the general practice.
    pub fn new() -> AccumulatorBuilder {
        AccumulatorBuilder::default()
    }

    /// Specify the kind of calculation
    ///
    /// All error-handling occurs during [`AccumulatorBuilder::build`]
    ///
    /// # Developer Note
    /// Is this really the best way? There is a tension here. Do we use a
    /// curated set of high-level names? In a more feature complete version we
    /// might accept arguments like:
    /// - "2pcf", "hist_cf"
    /// - "hist_astro_sf1", "astro_sf1", "astro_sf2", "sf2_tensor",
    ///   "sf3_tensor"
    /// - "longitudinal_sf2", "longitudinal_sf3"
    ///
    /// Also need to think about the fact that we probably want to be able to
    /// compute variance in each case. That can probably be a separate argument
    pub fn calc_kind(&mut self, calc_kind: String) -> &mut AccumulatorBuilder {
        self.calc_kind = Some(calc_kind);
        self
    }

    // todo: maybe we force people to specify a BinEdgeSpec instead?
    pub fn hist_irregular_bucket_edges(&mut self, edges: &[f64]) -> &mut AccumulatorBuilder {
        self.bucket_edges = Some(RawBinEdgeSpec::Vec(edges.to_vec()));
        self
    }

    pub fn hist_regular_bucket_edges(&mut self, edges: RegularBinEdges) -> &mut AccumulatorBuilder {
        self.bucket_edges = Some(RawBinEdgeSpec::Regular(edges.clone()));
        self
    }

    pub fn irregular_distance_square_edges(&mut self, edges: &[f64]) -> &mut AccumulatorBuilder {
        self.distance_sqr_bin_edges = Some(RawBinEdgeSpec::Vec(edges.to_vec()));
        self
    }

    pub fn build(&self) -> Result<Accumulator, Error> {
        //todo!("not implemented yet!")

        // construct the Config
        let config = Config {
            reducer_name: self.calc_kind.clone().unwrap(),
            // TODO deal with me!
            hist_reducer_bucket: None,
            // TODO deal with me!
            squared_distance_bin_edges: None,
        };

        let reducer = wrapper_reducer_from_config(&config)?;
        let descr = AccumulatorDescr { config, reducer };

        let state_size = descr.reducer.accum_state_size() as usize;
        let n_state = 1; // FIXME
        let data = AccumulatorData {
            data: vec![0.0; state_size * n_state],
            n_state,
            state_size,
        };
        let mut out = Accumulator { data, descr };
        out.reset_data();
        Ok(out)
    }
}
