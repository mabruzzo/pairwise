// inform build-system of the crates in this package
mod accumulator;
mod apply;
mod misc;
mod parallel_serial;

// pull in symbols that visible outside of the package
pub use accumulator::{Histogram, get_output};
pub use apply::apply_cartesian;
pub use misc::diff_norm;
pub use pairwise_nostd_internal::{
    Accumulator, CartesianBlock, CellWidth, Mean, OutputDescr, PointProps, View3DProps,
    apply_accum, dot_product,
};
