// inform build-system of the crates in this package
mod misc;
mod parallel_serial;
mod reducers;

// pull in symbols that visible outside of the package
pub use misc::diff_norm;
pub use pairwise_nostd_internal::{
    Executor, Mean, OutputDescr, PointProps, Reducer, StatePackViewMut, apply_accum, dot_product,
};
pub use parallel_serial::SerialExecutor;
pub use reducers::{Histogram, get_output, get_output_from_statepack_array};
