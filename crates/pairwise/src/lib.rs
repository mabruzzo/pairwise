/*!
Provides parallelized routines for directly computing spatial two-point
statistics from multi-dimensional data (e.g. 2-point correlation functions,
structure functions).

<div class="warning">

This crate is still in early development. The initial goal is to serve as a
backend for the [pyvsf python package](https://pyvsf.readthedocs.io/en/latest/).

</div>

# High-Level: 2-Point Statistics.

2-point statistics come up in a number of contexts, including
[astronomy/cosmology](https://en.wikipedia.org/wiki/Correlation_function_(astronomy))
and
[turbulence](https://en.wikipedia.org/wiki/Turbulence#Kolmogorov's_theory_of_1941).

The calculation of these quantities generally consists of computing a value
from each unique pair of spatial points (taken from a single data sets or from
2 separate data sets) and computing a statistic on those values. Often times
each value is partitioned into a "bin" based on the spatial separation of the
points that the value was computed from.

# User Guide

<div class="warning"> ADD ME </div>

# Developer Guide

See the crate-level documentation for [`pairwise_nostd_internal`].

*/

#![deny(rustdoc::broken_intra_doc_links)]

// inform build-system of the crates in this package
mod apply;
mod error;
mod misc;
mod parallel_serial;
mod reducers;

// pull in symbols that visible outside of the package
pub use apply::apply_cartesian;
pub use error::Error;
pub use misc::diff_norm;
pub use pairwise_nostd_internal::{
    CartesianBlock, CellWidth, Comp0Histogram, Comp0Mean, ComponentSumHistogram, ComponentSumMean,
    Executor, IrregularBinEdges, OutputDescr, PairOperation, PointProps, Reducer, RegularBinEdges,
    StatePackViewMut, View3DSpec, apply_accum, dot_product,
};
pub use parallel_serial::SerialExecutor;
pub use reducers::{
    EuclideanNormHistogram, EuclideanNormMean, get_output, get_output_from_statepack_array,
};
