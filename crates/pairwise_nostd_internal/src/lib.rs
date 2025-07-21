#![no_std]
mod apply_points;
mod misc;
mod parallel;
mod reduce_utils;
mod reducer;
mod state;

// I'm not really sure we want to publicly expose reduce_sample, but for now,
// we expose it to support testing...
pub mod reduce_sample;

pub use apply_points::{PointProps, apply_accum};
pub use misc::{dot_product, get_bin_idx, squared_diff_norm};
pub use parallel::{
    BatchedReduction, BinnedDatum, Executor, NestedReduction, ReductionCommon, StandardTeamParam,
    TeamMemberProp, TeamProps, ThreadMember, fill_single_team_statepack_batched,
    fill_single_team_statepack_nested,
};
pub use reduce_utils::reset_full_statepack;
pub use reducer::{Datum, Mean, OutputDescr, Reducer};
pub use state::{AccumStateView, AccumStateViewMut, StatePackViewMut};
