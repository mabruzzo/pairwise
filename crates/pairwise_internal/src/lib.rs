#![no_std]
mod accumulator;
mod misc;

pub use accumulator::{Accumulator, Mean, OutputDescr};
pub use misc::{diff_norm, dot_product, get_bin_idx, squared_diff_norm};
