#![no_std]
mod accumulator;
mod apply_cartesian;
mod apply_points;
mod misc;

pub use accumulator::{Accumulator, Mean, OutputDescr};
pub use apply_cartesian::{CartesianBlock, CellWidth, apply_cartesian};
pub use apply_points::{PointProps, apply_accum};
pub use misc::{View3DProps, dot_product, get_bin_idx, squared_diff_norm};
