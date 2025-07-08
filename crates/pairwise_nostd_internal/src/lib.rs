#![no_std]
mod accumulator;
mod apply_cartesian;
mod apply_points;
mod misc;
mod parallel;

pub use accumulator::{Accumulator, Mean, OutputDescr};
pub use apply_cartesian::{CartesianBlock, CartesianCalcContext, CellWidth};
pub use apply_points::{PointProps, apply_accum};
pub use misc::{View3DProps, dot_product, get_bin_idx, squared_diff_norm};
pub use parallel::{ReductionSpec, TeamProps, TeamRank};
