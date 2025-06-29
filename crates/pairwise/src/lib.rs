// inform build-system of the crates in this package
mod apply_points;
mod misc;

// pull in symbols that visible outside of the package
pub use apply_points::{PointProps, apply_accum};
pub use misc::{Histogram, get_output};
pub use pairwise_internal::{Accumulator, Mean, OutputDescr, diff_norm, dot_product};
