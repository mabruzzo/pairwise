//! Implements the "serial" backend for running thread teams

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use pairwise_nostd_internal::{Executor, MemberId, ReductionSpec, TeamProps};
use std::default::Default;
use std::num::NonZeroU32;

pub struct SerialTeam<'a> {
    n_members_per_team: u32,
    team_id: u32,
    n_teams: u32,
    accum_state_size: usize,
    // used to accumulate local contributions
    scratch_per_rank: &'a mut [f64],
    // we have 1 AccumulatorState per spatial bin
    statepack: &'a mut [f64],
}

impl<'a> SerialTeam<'a> {
    fn scratch_per_rank_shape(&self) -> [usize; 2] {
        let team_size = self.n_members_per_team as usize;
        [self.scratch_per_rank.len() / team_size, team_size]
    }

    fn team_statepacks_shape(&self) -> [usize; 2] {
        [
            self.accum_state_size,
            self.statepack.len() / self.accum_state_size,
        ]
    }
}

impl<'a> TeamProps for SerialTeam<'a> {
    fn n_members_per_team(&self) -> u32 {
        self.n_members_per_team
    }

    fn team_id(&self) -> u32 {
        self.team_id
    }

    fn n_teams(&self) -> u32 {
        self.n_teams
    }

    fn team_reduce(
        &mut self,
        prep_buf_fn: &impl Fn(&mut ArrayViewMut1<f64>, MemberId),
        reduce_buf_fn: &impl Fn(&mut ArrayViewMut1<f64>, &ArrayView1<f64>),
    ) {
        let team_size = self.n_members_per_team;
        let mut scratch_per_rank =
            ArrayViewMut2::from_shape(self.scratch_per_rank_shape(), self.scratch_per_rank)
                .unwrap();
        let mut statepack_vec: Vec<ArrayViewMut1<f64>> =
            scratch_per_rank.columns_mut().into_iter().collect();
        // in a multi-threaded-backend, all threads in a team would call
        // this function at once and we would need to do some kind of
        // synchronization.
        //
        // In this serial-backend, a single thread calls this function, and
        for (i, reduce_buf) in statepack_vec.iter_mut().enumerate() {
            prep_buf_fn(reduce_buf, MemberId::new(i as u32));
        }

        // at this point, we would probably have a memory fence on
        // self.scratch_per_rank (or maybe just synchronize the threads)

        // TODO: ensure that the iteration order is consistent with other
        //       backends to make sure we produce consistent results
        if !team_size.is_power_of_two() {
            // `1` is considered a power of 2
            todo!("haven't implemented reduction logic in this scenario!")
        }
        let mut remaining_pairs = (team_size / 2) as usize;

        while remaining_pairs > 0 {
            for i in 0..remaining_pairs {
                let [left_statepack, right_statepack] = statepack_vec
                    .get_disjoint_mut([i, i + remaining_pairs])
                    .unwrap();
                reduce_buf_fn(left_statepack, &right_statepack.view());
            }
            remaining_pairs /= 2;
        }
    }

    fn update_teambuf_if_root(&mut self, f: &impl Fn(&mut ArrayViewMut2<f64>, &ArrayView2<f64>)) {
        // in a multi-threaded-backend, this would be slightly more complex...
        let scratch_per_rank =
            ArrayView2::from_shape(self.scratch_per_rank_shape(), self.scratch_per_rank).unwrap();
        let mut team_statepacks =
            ArrayViewMut2::from_shape(self.team_statepacks_shape(), self.statepack).unwrap();
        f(&mut team_statepacks, &scratch_per_rank)
    }
}

pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn drive_reduce(
        &mut self,
        out: &mut ArrayViewMut2<f64>,
        reduction_spec: &impl ReductionSpec,
        n_members_per_team: NonZeroU32,
        n_teams: NonZeroU32,
    ) -> Result<(), &'static str> {
        let n_members_per_team = n_members_per_team.get() as usize;
        let n_teams = n_teams.get() as usize;
        let mut out_shape: [usize; 2] = [Default::default(); 2];
        out_shape.copy_from_slice(out.shape());
        let [accum_state_size, n_bins] = out_shape;
        if out_shape != reduction_spec.statepacks_shape() {
            return Err("the out argument doesn't have the correct shape!");
        }
        let team_statepack_size = accum_state_size * n_bins;

        let mut team_statepacks: Vec<f64> = vec![0.0; team_statepack_size * n_teams];

        // the reason that the following loop isn't more straight-forward way
        // is to make sure it is bitwise reproducible with the parallel case

        // fill up the statepacks for each team in the league
        for team_id in 0..n_teams {
            let statepack_offset = team_id * team_statepack_size;
            reduction_spec.init_team_statepacks(
                &mut ArrayViewMut2::from_shape(
                    out_shape,
                    &mut team_statepacks
                        [statepack_offset..(statepack_offset + team_statepack_size)],
                )
                .unwrap(),
            );

            let mut accum_state_per_member: Vec<f64> =
                vec![0.0; n_members_per_team * accum_state_size];

            let mut team_props = SerialTeam {
                n_members_per_team: n_members_per_team.try_into().unwrap(),
                team_id: team_id.try_into().unwrap(),
                n_teams: n_teams.try_into().unwrap(),
                accum_state_size,
                scratch_per_rank: &mut accum_state_per_member,
                statepack: &mut team_statepacks
                    [statepack_offset..(statepack_offset + team_statepack_size)],
            };

            // in a multi-threaded environment, each member of the team would invoke
            // collect_team_contrib. Then we would perform some synchronization among
            // the threads in calls to team_reduce and update_teambuf_if_root
            reduction_spec.collect_team_contrib(&mut team_props)
        }

        // after the above loop is done, each team's statepacks need to be combined

        let mut vec_of_statepacks: Vec<ArrayViewMut2<f64>> = team_statepacks
            .as_mut_slice()
            .chunks_exact_mut(team_statepack_size)
            .map(|buf: &mut [f64]| ArrayViewMut2::from_shape(out_shape, buf).unwrap())
            .collect();

        // consolidate each team's statepack
        // TODO: consider iteration order. All executors should have the
        //       same iteration order! The order within `team_reduce` is
        //       more-correct. Maybe we factor out the logic into a
        //       generic helper function...
        for i in 1..vec_of_statepacks.len() {
            let [left_statepacks, right_statepacks] =
                vec_of_statepacks.get_disjoint_mut([0, i]).unwrap();
            reduction_spec.league_reduce(left_statepacks, right_statepacks.view());
        }

        out.assign(&vec_of_statepacks[0].view());
        Ok(())
    }
}
