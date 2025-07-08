//! Our parallelism abstractions use the concepts of thread teams & reductions

use core::num::NonZeroU32;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub struct TeamRank(u32);

impl TeamRank {
    pub fn new(rank: u32) -> Self {
        TeamRank(rank)
    }

    pub fn get(&self) -> u32 {
        self.0
    }
}

/// the goal here is to use a parallelism strategy with thread-teams. This is
/// hardly a novel concept. The actual implementation will be tailored to the
/// "backend" that we are using (e.g. serial, multi-threading, GPU)
///
/// To make things work, we borrow the terminology used by Kokkos:
/// we have a league of `league_size` thread-teams. Each team is composed
/// `thread_size` threads.
pub trait TeamProps {
    // for now, we explicitly choose not to provide a method to directly access
    // team_rank. The current approach for accessing team_rank (i.e. only as
    // part of `team_reduce`) generally encourages a design where the serial
    // implementation can support arbirary team sizes (and is capable of
    // returning the exact same result as parallel implementations). If we
    // make team_rank accessible, implementations can generally only support
    // a team_size of 1.

    fn team_size(&self) -> u32;
    fn league_rank(&self) -> u32;
    fn league_size(&self) -> u32;

    /// Triggers a barrier for all threads in the thread team, then has each
    /// team-member invoke `prep_buf_fn` to prepare separate input buffers
    /// used in for a reduction. Afterwards, `reduce_buf_fn` is used to perform
    /// reductions with all of the buffers
    ///
    /// `prep_buf_fn` and `reduce_buf_fn` generally have the signatures:
    /// ```text
    /// prep_buf_fn(reduce_buf: &mut ArrayViewMut1<f64>, team_rank: TeamRank)
    /// reduce_buf_fn(reduce_buf: &mut ArrayViewMut1<f64>, other_buf: ArrayView1<f64>)
    /// ```
    ///
    /// # Note:
    /// I suspect that the memory barrier requires a memory fence
    fn team_reduce(
        &mut self,
        prep_buf_fn: &impl Fn(&mut ArrayViewMut1<f64>, TeamRank),
        reduce_buf_fn: &impl Fn(&mut ArrayViewMut1<f64>, &ArrayView1<f64>),
    );

    /// Triggers a barrier for all threads in the thread team. Then the rank-0
    /// member of the team invokes `f` to update the shared teambuf
    fn update_teambuf_if_root(&mut self, f: &impl Fn(&mut ArrayViewMut2<f64>, &ArrayView2<f64>));
}

/// This describes a reduction operation
///
/// The premise is that you do error-checking during construction. When we
/// invoke the call method, we can be confident that the arguments and buffers
/// are all valid/compatible
///
/// # Note
/// It may be worth considering it would be better not to make this a trait,
/// but instead a generic type, where we register components.
pub trait ReductionSpec {
    // not sure that the next functions should be a method
    fn statepacks_shape(&self) -> [usize; 2];

    fn collect_team_contrib(&self, team_props: &mut impl TeamProps);

    fn init_team_statepacks(&self, buf: &mut ArrayViewMut2<f64>);

    fn league_reduce(&self, primary: &mut ArrayViewMut2<f64>, other: ArrayView2<f64>);
}

pub trait Executor {
    type TeamPropType: TeamProps;

    // I suspect that we may want to set up team_size & league_size elsewhere
    fn drive_reduce(
        out: &mut ArrayViewMut2<f64>,
        reduction_spec: &impl ReductionSpec,
        team_size: NonZeroU32,
        league_size: NonZeroU32,
    ) -> Result<(), &'static str>;
}
