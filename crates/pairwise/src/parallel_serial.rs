//! Implements the "serial" backend for running thread teams

use core::num::NonZeroU32;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, iter::Axes};
use pairwise_nostd_internal::{Executor, ReductionSpec, TeamProps, TeamRank};

pub struct SerialTeam<'a> {
    // we probably want accum to be a reference
    team_size: u32,
    league_rank: u32,
    league_size: u32,
    // used to accumulate local contributions
    scratch_per_rank: &'a mut ArrayViewMut2<'a, f64>,
    // we have 1 statepack per spatial bin
    team_statepacks: &'a mut ArrayViewMut2<'a, f64>,
}

impl<'a> TeamProps for SerialTeam<'a> {
    fn team_size(&self) -> u32 {
        self.team_size
    }

    fn league_rank(&self) -> u32 {
        self.league_rank
    }

    fn league_size(&self) -> u32 {
        self.league_size
    }

    fn team_reduce(
        &mut self,
        prep_buf_fn: &impl Fn(&mut ArrayViewMut1<f64>, TeamRank),
        reduce_buf_fn: &impl Fn(&mut ArrayViewMut1<f64>, &ArrayView1<f64>),
    ) {
        let mut statepack_vec: Vec<ArrayViewMut1<f64>> =
            self.scratch_per_rank.columns_mut().into_iter().collect();
        // in a multi-threaded-backend, all threads in a team would call
        // this function at once and we would need to do some kind of
        // synchronization.
        //
        // In this serial-backend, a single thread calls this function, and
        for (i, reduce_buf) in statepack_vec.iter_mut().enumerate() {
            prep_buf_fn(&mut reduce_buf, TeamRank::new(i as u32));
        }

        // at this point, we would probably have a memory fence on
        // self.scratch_per_rank (or maybe just synchronize the threads)

        // TODO: ensure that the iteration order is consistent with other
        //       backends to make sure we produce consistent results
        if !self.team_size.is_power_of_two() {
            // `1` is considered a power of 2
            todo!("haven't implemented reduction logic in this scenario!")
        }
        let mut remaining_pairs = self.team_size / 2;

        while remaining_pairs > 0 {
            for i in 0..remaining_pairs {
                reduce_buf_fn(statepack_vec[i], statepack_vec[i + remaining_pairs].view());
            }
            remaining_pairs /= 2;
        }
    }

    fn update_teambuf_if_root(&mut self, f: &impl Fn(&mut ArrayViewMut2<f64>, &ArrayView2<f64>)) {
        // in a multi-threaded-backend, this would be slightly more complex...
        let [statepack_size, _] = self.team_statepacks.shape();
        f(&mut self.team_statepacks, self.scratch_per_rank.view())
    }
}

pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn drive_reduce(
        out: &mut ArrayViewMut2<f64>,
        reduction_spec: &impl ReductionSpec,
        team_size: NonZeroU32,
        league_size: NonZeroU32,
    ) -> Result<(), &'static str> {
        let team_size = team_size.get();
        let out_shape = out.shape();
        if out_shape != reduction_spec.statepacks_shape() {
            return Err("the out argument doesn't have the correct shape!");
        }
        let [statepack_size, n_bins] = out_shape;
        let out_size = statepack_size * n_bins;

        let mut team_statepacks = vec![0.0; out_size * (league_size.get() as usize)];
        let mut tmp_slice: &[f64] = &team_statepacks[..];
        let mut vec_of_statepacks: Vec<ArrayViewMut2<f64>> = tmp_slice
            .chunks_exact_mut(out_size)
            .map(|buf: &mut [f64]| ArrayViewMut2::from_shape(*out_shape, buf))
            .collect();

        // fill up the statepacks for each team in the league
        for (cur_team_statepacks, league_rank) in vec_of_statepacks.enumerate() {
            reduction_spec.init_team_statepacks(cur_team_statepacks);

            let scratch_per_rank = vec![0.0; team_size * statepack_size];
            let mut team_props = SerialTeam {
                team_size: team_size,
                league_rank,
                league_size: league_size.get(),
                scratch_per_rank: &mut ArrayViewMut2::from_shape(
                    [statepack_size, team_size],
                    &scratch_per_rank,
                ),
                team_statepacks: &mut cur_team_statepacks,
            };

            // in a multi-threaded environment, we would invoke the following
            // function `team_size` times. Then we would perform some
            // synchronization among the threads in calls to team_reduce and
            // update_teambuf_if_root

            for team_rank in 0..team_size {
                reduction_spec.collect_team_contrib(&mut team_props)
            }
        }

        // consolidate each team's statepack
        // TODO: consider iteration order. All executors should have the
        //       same iteration order! The order within `team_reduce` is
        //       more-correct. Maybe we factor out the logic into a
        //       generic helper function...
        for i in 1..vec_of_statepacks.len() {
            reduction_spec.league_reduce(&mut vec_of_statepacks[0], &vec_of_statepacks[i].view());
        }

        out.assign(vec_of_statepacks[0]);
        Ok(())
    }
}
