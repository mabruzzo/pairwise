// inform build-system of the crates in this package
mod misc;

// pull in symbols that visible outside of the package
pub use misc::{Histogram, diff_norm, get_output};
pub use pairwise_nostd_internal::{
    Accumulator, Mean, OutputDescr, PointProps, apply_accum, dot_product,
};
