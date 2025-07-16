// inform build-system of the crates in this package
mod accumulator;
mod misc;
mod parallel_serial;

// pull in symbols that visible outside of the package
pub use accumulator::{Histogram, get_output, get_output_from_statepack_array};
pub use misc::diff_norm;
pub use pairwise_nostd_internal::{
    Accumulator, Mean, OutputDescr, PointProps, StatePackViewMut, apply_accum, dot_product,
};
