#![no_std]
mod accumulator;
mod misc;

pub use accumulator::{Accumulator, Mean, OutputDescr};
pub use misc::{dot_product, get_bin_idx, squared_diff_norm};
