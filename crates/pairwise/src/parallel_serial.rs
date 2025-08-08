//! Implements the "serial" backend for running thread teams
use crate::Error;
use pairwise_nostd_internal::{
    BinnedDatum, Reducer, ReductionSpec, StandardTeamParam, StatePackViewMut, Team,
    fill_single_team_binned_statepack,
};
use std::num::NonZeroU32;

// this a really silly implementation that only supports a 1 member per team
// and a team size of 1
struct SerialTeam {
    team_id: usize,
    team_param: StandardTeamParam,
}

struct DummyWrapper<T>(T);

impl Team for SerialTeam {
    const IS_VECTOR_PROCESSOR: bool = false;
    type SharedDataHandle<T> = DummyWrapper<T>;

    fn standard_team_info(&self) -> StandardTeamParam {
        self.team_param
    }

    fn team_id(&self) -> usize {
        self.team_id
    }

    fn exec_once(
        &mut self,
        statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        f: &impl Fn(&mut StatePackViewMut),
    ) {
        // the only thread in the team is the root member
        f(&mut statepack.0)
    }

    fn calccontribs_combine_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        reducer: &impl Reducer,
        bin_index: usize,
        get_member_contrib: &impl Fn(&mut StatePackViewMut, usize),
    ) {
        let n_members = 1;
        let accum_state_size = reducer.accum_state_size();
        // we should really pre-allocate the memory in a constructor

        // create a temporary StatePack that holds a temporary accumulator
        // state so that can be filled by each team member (we should really
        // pre-allocate this).
        let mut tmp_state_buffer = vec![0.0; n_members * accum_state_size];
        let mut tmp_statepack =
            StatePackViewMut::from_slice(n_members, accum_state_size, &mut tmp_state_buffer);

        // reset the state in the statepack
        for i in 0..n_members {
            reducer.init_accum_state(&mut tmp_statepack.get_state_mut(i));
        }

        // step 0: apply a barrier
        // (obviously this is a no-op for a serial implementation)

        // step 1: each team member calls the get_member_contrib closure
        get_member_contrib(&mut tmp_statepack, 0_usize);

        // step 2: perform a local reduction among all the team members
        if n_members == 1 {
            // no-op since tmp_statepack.get_state(0) already holds all
            // contributions
        } else {
            panic!("not currently supported in a serial implementation")
        }

        // step 3: have the member holding the total contribution to update the
        //         `accum_state` stored at `bin_index` in `statpack`
        if binned_statepack.0.n_states() > bin_index {
            reducer.merge(
                &mut binned_statepack.0.get_state_mut(bin_index),
                &tmp_statepack.get_state(0),
            );
        }
    }

    fn collect_pairs_then_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        reducer: &impl Reducer,
        get_datum_bin_pair: &impl Fn(&mut [BinnedDatum], usize),
    ) {
        // step 0: apply a barrier
        // (obviously this is a no-op for a serial implementation)

        // we should obviously pre-allocate this memory
        let mut collect_pad = [BinnedDatum::zeroed()];

        // step 1: each team member calls the get_datum_bin_pair closure
        get_datum_bin_pair(&mut collect_pad, 0_usize);

        // step 2: gather the datum-bin pairs into the memory of a single
        //         team member
        // (obviously this is a no-op for a serial implementation)

        // step 3: the team member holding every datum-bin pair now uses
        //         them to update statepack
        let bin_index = collect_pad[0].bin_index;
        if binned_statepack.0.n_states() > bin_index {
            reducer.consume(
                &mut binned_statepack.0.get_state_mut(bin_index),
                &collect_pad[0].datum,
            );
        }
    }
}

pub struct SerialExecutor;

// I'm increasingly thinking that we don't want the Executor trait. If
// nothing else, it shouldn't be declared in pairwise_nostd_internal

impl SerialExecutor {
    pub fn drive_reduce(
        &mut self,
        out: &mut StatePackViewMut,
        reduce_spec: &impl ReductionSpec,
        n_members_per_team: NonZeroU32,
        n_teams: NonZeroU32,
    ) -> Result<(), Error> {
        let reducer = reduce_spec.get_reducer();
        let n_bins = reduce_spec.n_bins();

        let team_param = StandardTeamParam {
            n_members_per_team: n_members_per_team.get() as usize,
            n_teams: n_teams.get() as usize,
        };

        if team_param.n_members_per_team != 1 {
            return Err(Error::integer_range(
                "team_param.n_members_per_team",
                team_param.n_members_per_team as i64,
                1,
                1,
            ));
        } else if team_param.n_teams != 1 {
            // todo: we should add support for n_teams (that's straight-forward)
            return Err(Error::integer_range(
                "team_param.n_teams",
                team_param.n_teams as i64,
                1,
                1,
            ));
        } else if (out.n_states() != n_bins) || (out.state_size() != reducer.accum_state_size()) {
            return Err(Error::binned_statepack_shape(
                n_bins as u64,
                reducer.accum_state_size() as u64,
                out.n_states() as u64,
                out.state_size() as u64,
            ));
        }

        // allocate a temporary statepack
        let mut tmp = vec![0.0; out.total_size()];
        let mut tmp_binned_statepack =
            StatePackViewMut::from_slice(out.n_states(), out.state_size(), &mut tmp);

        // now we initialize statepack (this is inefficient!)
        for i in 0..tmp_binned_statepack.n_states() {
            reducer.init_accum_state(&mut tmp_binned_statepack.get_state_mut(i));
        }

        // if we had multiple teams this would theoretically be a for loop
        {
            // move tmp_binned_statepack into the shared_binned_statepack variable
            let mut shared_binned_statepack = DummyWrapper(tmp_binned_statepack);

            // either fill_single_team_nested or fill_single_team_batched
            fill_single_team_binned_statepack(
                &mut shared_binned_statepack,
                &mut SerialTeam {
                    team_id: 0_usize,
                    team_param,
                },
                reduce_spec,
            );

            // move the contents of shared_statepack back to tmp_statepack
            // (if something goes wrong, it's probably due to an error
            // right here)
            tmp_binned_statepack = shared_binned_statepack.0;
        }

        // if we supported multiple teams, we'd perform a reduction here
        // to consolidate all of the temporary statepacks

        // finally, update out
        for i in 0..out.n_states() {
            reducer.merge(
                &mut out.get_state_mut(i),
                &tmp_binned_statepack.get_state(i),
            );
        }
        Ok(())
    }
}
