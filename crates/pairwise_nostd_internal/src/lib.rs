#![no_std]
mod accumulator;
mod apply_points;
mod misc;
mod parallel;
mod state;

// I'm not really sure we want to publicly expose reduce_sample, but for now,
// we expose it to support testing...
pub mod reduce_sample;

pub use accumulator::{Accumulator, Datum, Mean, OutputDescr, reset_full_statepack};
pub use apply_points::{PointProps, apply_accum};
pub use misc::{dot_product, get_bin_idx, squared_diff_norm};
pub use parallel::{
    BinnedDatum, Executor, ReductionSpec, StandardTeamParam, TeamMemberProp, TeamProps,
    ThreadMember, fill_single_team_statepack,
};
pub use state::{AccumStateView, AccumStateViewMut, StatePackViewMut};
