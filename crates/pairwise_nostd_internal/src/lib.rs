#![no_std]
mod accumulator;
mod apply_points;
mod misc;
mod state;

pub use accumulator::{Accumulator, Datum, Mean, OutputDescr};
pub use apply_points::{PointProps, apply_accum};
pub use misc::{dot_product, get_bin_idx, squared_diff_norm};
pub use state::{AccumStateView, AccumStateViewMut, StatePackViewMut};
