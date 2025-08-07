//! This module introduces the machinery to wrap the various types that
//! implement [`Reducer`] in a uniform manner and enable runtime polymorphism.
//! This machinery is primarily used to implement [`crate::AccumulatorDescr`]
//!
//! Our current approach for doing this involves dynamic dispatch.

use crate::{
    Error,
    apply::apply_accum,
    reducers::{EuclideanNormHistogram, EuclideanNormMean, get_output},
};

use pairwise_nostd_internal::{
    BinEdges, ComponentSumHistogram, ComponentSumMean, IrregularBinEdges, PairOperation, Reducer,
    RegularBinEdges, StatePackView, StatePackViewMut, UnstructuredPoints, merge_full_statepacks,
    reset_full_statepack, validate_bin_edges,
};
use std::{collections::HashMap, sync::LazyLock};

/// A wrapper type that only exists so we can implement the `Eq` trait.
///
/// Ordinarily, [`f64`], and by extension `Vec<f64>` doesn't implement `Eq`
/// since `NaN` != `NaN`. We can implement it here since
/// [`pairwise_nostd_internal::validate_bin_edges`] ensures there aren't any
/// `NaN` values
#[derive(Clone)]
pub(crate) struct ValidatedBinEdgeVec(Vec<f64>);

impl ValidatedBinEdgeVec {
    pub(crate) fn new(edges: Vec<f64>) -> Result<Self, Error> {
        validate_bin_edges(&edges).map_err(Error::internal_legacy_adhoc)?;
        Ok(Self(edges))
    }

    fn as_irregular_edge_view<'a>(&'a self) -> IrregularBinEdges<'a> {
        // TODO consider introducing a way to bypass error checks when
        // we construct IrregularBinEdges from ValidatedBinEdgeVec
        IrregularBinEdges::new(self.0.as_slice()).expect(
            "There must be a bug: either in the pre-validation of the bin \
            edges, OR that somehow mutated bin-edges after they were \
            pre-validated!",
        )
    }
}

impl PartialEq for ValidatedBinEdgeVec {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ValidatedBinEdgeVec {}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum BinEdgeSpec {
    Regular(RegularBinEdges),
    Vec(ValidatedBinEdgeVec),
}

impl BinEdgeSpec {
    pub(crate) fn leftmost_edge(&self) -> f64 {
        match self {
            BinEdgeSpec::Regular(edges) => edges.leftmost_edge(),
            BinEdgeSpec::Vec(v) => v.0[0],
        }
    }

    pub(crate) fn n_bins(&self) -> usize {
        match self {
            BinEdgeSpec::Regular(edges) => edges.n_bins(),
            BinEdgeSpec::Vec(v) => v.0.len() - 1,
        }
    }
}

/// A configuration object
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Config {
    pub(crate) reducer_name: String,
    // I'm fairly confident that supporting the Histogram with the
    // [`IrregularBinEdges`] type introduces self-referential struct issues
    // (at a slightly higher level)
    pub(crate) hist_reducer_bucket: Option<BinEdgeSpec>,
    // eventually, we should add an option for variance

    // I'm not so sure the following should actually be tracked in this struct
    pub(crate) squared_distance_bin_edges: BinEdgeSpec,
}

/// an internal type that is used to encode the spatial information
#[derive(Clone)]
pub(crate) enum SpatialInfo<'a> {
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

/// Wraps a function pointer that constructs, a boxed [`WrappedReducer`] trait
/// object.
///
/// # Note
/// Make this a type-alias doesn't appear to have the desired effect. I suspect
/// that it doesn't properly coerce a closure to function pointer
struct MkWrappedReducerFn(fn(&Config) -> Result<Box<dyn WrappedReducer>, Error>);

/// returns a hashmap, where keys map the names of two-point calculations to
/// functions that construct the appropriate boxed [`WrappedReducer`] trait
/// objects.
fn build_registry() -> HashMap<String, MkWrappedReducerFn> {
    let out: HashMap<String, MkWrappedReducerFn> = HashMap::from([
        // -------------------------------------------------
        // define mappings for correlation function reducers
        // -------------------------------------------------
        (
            "hist_cf".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                let Some(ref edges) = c.hist_reducer_bucket else {
                    return Err(Error::bucket_edge_presence(c.reducer_name.clone(), true));
                };
                if let BinEdgeSpec::Regular(edges) = edges {
                    Ok(Box::new(WrappedReducerImpl {
                        reducer: ComponentSumHistogram::from_bin_edges(edges.clone()),
                        pair_op: PairOperation::ElementwiseMultiply,
                    }) as Box<dyn WrappedReducer>)
                } else {
                    Ok(Box::new(WrappedIrregularHist {
                        pair_op: PairOperation::ElementwiseMultiply,
                    }) as Box<dyn WrappedReducer>)
                }
            }),
        ),
        (
            "2pcf".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                if c.hist_reducer_bucket.is_some() {
                    return Err(Error::bucket_edge_presence(c.reducer_name.clone(), false));
                }
                Ok(Box::new(WrappedReducerImpl {
                    reducer: ComponentSumMean::new(),
                    pair_op: PairOperation::ElementwiseMultiply,
                }) as Box<dyn WrappedReducer>)
            }),
        ),
        // -----------------------------------------------
        // define mappings for structure function reducers
        // -----------------------------------------------
        (
            "hist_astro_sf1".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                let Some(ref edges) = c.hist_reducer_bucket else {
                    return Err(Error::bucket_edge_presence(c.reducer_name.clone(), true));
                };
                if let BinEdgeSpec::Regular(edges) = edges {
                    Ok(Box::new(WrappedReducerImpl {
                        reducer: EuclideanNormHistogram::from_bin_edges(edges.clone()),
                        pair_op: PairOperation::ElementwiseSub,
                    }) as Box<dyn WrappedReducer>)
                } else {
                    Ok(Box::new(WrappedIrregularHist {
                        pair_op: PairOperation::ElementwiseMultiply,
                    }) as Box<dyn WrappedReducer>)
                }
            }),
        ),
        (
            "astro_sf1".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                if c.hist_reducer_bucket.is_some() {
                    return Err(Error::bucket_edge_presence(c.reducer_name.clone(), false));
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

/// global variable that holds a registry that describes the various kinds of
/// two-point calculations that the crate supports.
///
/// This variable is lazily initialized, in a threadsafe manner, using the
/// [`build_registry`] function. In more detail, it holds a hashmap, where
/// keys map the names of known two-point calculations to functions that
/// construct the appropriate boxed [`WrappedReducer`] trait objects.
///
/// # Note
/// This may not be the optimal way to encode this information.
static REDUCER_MAKER_REGISTRY: LazyLock<HashMap<String, MkWrappedReducerFn>> =
    LazyLock::new(build_registry);

/// constructs the appropriate [`WrappedReducer`] trait object that
/// corresponds to the specified configuration
pub(crate) fn wrapped_reducer_from_config(
    config: &Config,
) -> Result<Box<dyn WrappedReducer>, Error> {
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

/// A trait that primarily wraps a [`Reducer`] type in order to support
/// dynamic dispatch.
///
/// The main purpose of this function is to help implement the
/// [`crate::AccumulatorDescr`] type.
///
/// # Dynamic Dispatch vs "Enum Dispatch"
///
/// TODO: EDIT ME
/// My concern with an enum-based approach:
/// - it won't very well with the dynanamic plugin approach that is described
///   up in comments within accumulator.rs.
///   - Update: actually, if we continue to treat `Config` in a similar manner,
///     to how its treated now (i.e. its synchronized with the wrapped reducer
///     and the "source-of-truth" for serialization), then I'm not as concerned
/// - My main concern focuses on readability and maintainability...
///   - there are "a lot" of variations to support. Combinatorics from generic
///     types (especially when we want to make different choices for structure
///     functions and correlation functions), significantly increase the
///     number of variations...
///   - Let's consider what it takes to add a new reducer:
///     - under the current system, we just need to update [`build_registry`].
///     - under an enum-system, we would need to update
///       - [`build_registry`] (or whatever equivalent we shift to using).
///       - The enum-system would also have equivalent methods for each method
///         that this trait currently declares (there are 5 at the time of
///         writing), and there would be a match statement over all variants.
///         We would need to update the variant in every method.
///     - while we could potentially use macros to reduce code duplication, I
///       worry about readability
///       - to be fair, we do now use a macro in [`WrappedIrregularHist`].
///         With that said, this covers very special cases (that we may want
///         to consider removing...).
///       - if we don't remove [`WrappedIrregularHist`], we would need to have
///         special handling in the enum-dispatch approach for these special
///         cases (which could significantly complicate the macros)
/// - At a surface-level, it might seem like the enum-system would let us
///   remove all machinery to make [`Config`] implement the [`Eq`] trait
///   (namely, the machinery to implement bin-edge equality checks)
///   - in practice, that isn't really true.
///   - performing equality checks for distance edge bins and Histogram
///     reducers requires us to implement the vast majority of this machinery
///     anyway.
///
/// There are some other considerations:
/// - do we want to support a future where external code can implement new
///   Reducers? (At the moment, we don't)
/// - is the performance benefit of enum-dispatch significant
///   - I'm sketipical, at the moment
///   - But, I could totally see it being more important if we supported
///     multiple reducers per accumulator.
///
/// Importantly, I'm happy to be proven wrong and change our approach. If
/// anybody wants to implement an enum-dispatch approach, I would happily
/// consider it!
pub(crate) trait WrappedReducer {
    /// merge the state information tracked by `binned_statepack` and
    /// `other_binned_statepack`, and update `binned_statepack` accordingly
    ///
    /// # Note
    /// The `config` argument **must** be identical to the argument passed
    /// into [`wrapped_reducer_from_config`]. It is _only_ used to help
    /// implement the [`WrappedIrregularHist`] type
    fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
        config: &Config,
    );

    /// constructs a hashmap holding the calculation outputs
    ///
    fn get_output(
        &self,
        binned_statepack: &StatePackView,
        config: &Config,
    ) -> HashMap<&'static str, Vec<f64>>;

    /// Res
    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut, config: &Config);

    /// Returns the size of individual accumulator states.
    ///
    /// In a binned_statepack, the total number of entries is the product of
    /// the number returned by this method and the number of bins.
    ///
    /// # Note
    /// While the number of outputs per bin is commonly the same as the value
    /// returned by this function, that need not be the case. For example,
    /// imagine we used an algorithm like Kahan summation to attain improved
    /// accuracy.
    fn accum_state_size(&self, config: &Config) -> usize;

    // we probably need to pass other arguments...
    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        config: &Config,
    ) -> Result<(), Error>;
}

/// Represents a wrapped [`Reducer`].
///
/// This type wraps **any** kind of reducer, except for [`ScalarHistogram`]
/// defined using the [`IrregularBinEdges`] type. That case requires special
/// handling (see [`WrappedIrregularHist`] for more details).type for more details.
///  This is a special case among reducers.  will be
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
        _config: &Config,
    ) {
        merge_full_statepacks(&self.reducer, binned_statepack, other_binned_statepack);
    }

    // we should probably let the caller provide the output memory
    // (it would be nice if we could pass in a type that is more clearly
    // immutable than StatePackViewMut)
    fn get_output(
        &self,
        binned_statepack: &StatePackView,
        _config: &Config,
    ) -> HashMap<&'static str, Vec<f64>> {
        get_output(&self.reducer, binned_statepack)
    }

    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut, _config: &Config) {
        reset_full_statepack(&self.reducer, binned_statepack)
    }

    fn accum_state_size(&self, _config: &Config) -> usize {
        self.reducer.accum_state_size()
    }

    // we probably need to pass other arguments...
    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        config: &Config,
    ) -> Result<(), Error> {
        exec_reduction_helper(
            &self.reducer,
            binned_statepack,
            spatial_info,
            &config.squared_distance_bin_edges,
            self.pair_op,
        )
    }
}

/// Represents a [`ScalarHistogram`] with the [`IrregularBinEdges`] type.
///
/// # Note
/// We have to play a "game," where the [`Reducer`] doesn't have persistent
/// state. Instead, we reconstruct the reducer exactly when we need it.
/// This is done to avoid lifetime issues in [`AccumulatorDescr`]
struct WrappedIrregularHist {
    pair_op: PairOperation,
}

/// Calls function with a temporary histogram reducer that has irregular
/// bucket edges.
///
/// In more detail, this macro:
/// - temporarily constructs a [`ScalarHistogram`] with the
///   [`IrregularBinEdges`] type (using details from the provided
///   [`PairOperation`] and [`Config`] arguments)
/// - passes a reference to the temporary reducer to the specified function,
///   as the first argument (replacing the `reducer_ref` placeholder), and
///   return's the function's result
/// - the temporary reducer is dropped as we leave scope.
///
/// In more detail, this is intended to reduce boilerplate code within
/// [`WrappedIrregularHist`]. We could replace this with a function if we
/// converted [`WrappedIrregularHist`] so that it is a generic type with
/// respect to [`pairwise_nostd_internal::ScalarizeOp`]
macro_rules! reconstruct_hist_reducer_and_forward{
    ($pair_op:expr, $config:expr; $func:ident(reducer_ref, $($args:expr),*)) => {
        {
            let edges = match $config.hist_reducer_bucket {
                Some(BinEdgeSpec::Vec(ref v)) => v.as_irregular_edge_view(),
                _ => panic!("Bug: should be unreachable!"),
            };
            match $pair_op {
                PairOperation::ElementwiseMultiply => {
                    let my_reducer = ComponentSumHistogram::from_bin_edges(edges);
                    $func(&my_reducer, $($args),*)
                }
                PairOperation::ElementwiseSub => {
                    let my_reducer = EuclideanNormHistogram::from_bin_edges(edges);
                    $func(&my_reducer, $($args),*)
                }
            }
        }
    }
}

impl WrappedReducer for WrappedIrregularHist {
    fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
        config: &Config,
    ) {
        reconstruct_hist_reducer_and_forward!(
            self.pair_op,
            config;
            merge_full_statepacks(
                reducer_ref,
                binned_statepack,
                other_binned_statepack
            )
        )
    }

    fn get_output(
        &self,
        binned_statepack: &StatePackView,
        config: &Config,
    ) -> HashMap<&'static str, Vec<f64>> {
        reconstruct_hist_reducer_and_forward!(
            self.pair_op, config; get_output(reducer_ref, binned_statepack)
        )
    }

    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut, config: &Config) {
        reconstruct_hist_reducer_and_forward!(
            self.pair_op, config; reset_full_statepack(reducer_ref, binned_statepack)
        )
    }

    fn accum_state_size(&self, config: &Config) -> usize {
        // this can't be a closure and accept a generic arg
        fn f(reducer: &impl Reducer) -> usize {
            reducer.accum_state_size()
        }

        reconstruct_hist_reducer_and_forward!(
            self.pair_op, config; f(reducer_ref,) // the macro needs trailing `,`
        )
    }

    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        config: &Config,
    ) -> Result<(), Error> {
        reconstruct_hist_reducer_and_forward!(
            self.pair_op,
            config;
            exec_reduction_helper(
                reducer_ref,
                binned_statepack,
                spatial_info,
                &config.squared_distance_bin_edges,
                self.pair_op
            )
        )
    }
}

fn exec_reduction_helper<R: Clone + Reducer>(
    reducer: &R,
    binned_statepack: &mut StatePackViewMut,
    spatial_info: SpatialInfo,
    squared_distance_bin_edges: &BinEdgeSpec,
    pair_op: PairOperation,
) -> Result<(), Error> {
    match spatial_info {
        SpatialInfo::Unstructured {
            ref points_a,
            ref points_b,
        } => match squared_distance_bin_edges {
            BinEdgeSpec::Vec(v) => apply_accum(
                binned_statepack,
                reducer,
                points_a,
                points_b.as_ref(),
                &v.as_irregular_edge_view(),
                pair_op,
            ),
            BinEdgeSpec::Regular(edges) => apply_accum(
                binned_statepack,
                reducer,
                points_a,
                points_b.as_ref(),
                edges,
                pair_op,
            ),
        },
    }
}
