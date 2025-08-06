use crate::{
    Error,
    apply::apply_accum,
    reducers::{EuclideanNormHistogram, EuclideanNormMean, get_output},
};

use pairwise_nostd_internal::{
    ComponentSumHistogram, ComponentSumMean, IrregularBinEdges, PairOperation, Reducer,
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
    pub(crate) fn new(edges: Vec<f64>) -> Result<Self, &'static str> {
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
pub(crate) enum BinEdgeSpec {
    Vec(ValidatedBinEdgeVec),
    Regular(RegularBinEdges),
}

/// A configuration object
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Config {
    pub(crate) reducer_name: String,
    // I'm fairly confident that supporting the Histogram with the
    // [`IrregularBinEdges`] type introduces self-referential struct issues
    // (at a slightly higher level)
    pub(crate) hist_reducer_bucket: Option<RegularBinEdges>,
    // eventually, we should add an option for variance

    // I'm not so sure the following should actually be tracked in this struct
    // TODO: this shouldn't be an option
    pub(crate) squared_distance_bin_edges: Option<BinEdgeSpec>,
}

/// an internal type that will be used to encode the spatial information
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

// making this a type-alias wasn't have the appropriate effect (we weren't
// properly coercing closures to function pointers)
struct MkWrappedReducerFn(fn(&Config) -> Result<Box<dyn WrappedReducer>, Error>);

fn build_registry() -> HashMap<String, MkWrappedReducerFn> {
    let out: HashMap<String, MkWrappedReducerFn> = HashMap::from([
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

pub(crate) trait WrappedReducer {
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
