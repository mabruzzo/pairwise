//! Implements the "serial" backend for running thread teams

use pairwise_nostd_internal::{
    Accumulator, BinnedDatum, Executor, ReductionSpec, StandardTeamParam, StatePackViewMut,
    TeamProps, ThreadMember, fill_single_team_statepack, reset_full_statepack,
    serial_merge_accum_states,
};
use std::num::NonZeroU32;

// this a really silly implementation that only supports a 1 member per team
// and a team size of 1
struct SerialTeam;

struct DummyWrapper<T>(T);

impl TeamProps for SerialTeam {
    type SharedDataHandle<T> = DummyWrapper<T>;
    type MemberPropType = ThreadMember;

    fn exec_if_root_member(
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
        accum: &impl Accumulator,
        bin_index: usize,
        get_member_contrib: &impl Fn(&mut StatePackViewMut, ThreadMember),
    ) {
        let n_members = 1;
        let accum_state_size = accum.accum_state_size();

        // create a StatePack that holds a separate, temporary accumulator
        // state for each team member (should really pre-allocate this).
        let mut tmp_state_buffer = vec![0.0; n_members * accum_state_size];
        let mut tmp_statepack =
            StatePackViewMut::from_slice(n_members, accum_state_size, &mut tmp_state_buffer);
        // reset the state in the statepack
        reset_full_statepack(accum, &mut tmp_statepack);

        // step 0: apply a barrier
        // (obviously this is a no-op for a serial implementation)

        // step 1: each team member calls the get_member_contrib closure
        get_member_contrib(&mut tmp_statepack, ThreadMember::new(0_u32));

        // step 2: perform a local reduction among all the team members
        // (ensure that tmp_statepack.get_state(0) holds all contributions)
        if n_members > 1 {
            serial_merge_accum_states(accum, &mut tmp_statepack);
        }

        // step 3: have the member holding the total contribution to update the
        //         `accum_state` stored at `bin_index` in `statpack`
        if binned_statepack.0.n_states() > bin_index {
            accum.merge(
                &mut binned_statepack.0.get_state_mut(bin_index),
                &tmp_statepack.get_state(0),
            );
        }
    }

    fn collect_pairs_then_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        accum: &impl Accumulator,
        get_datum_bin_pair: &impl Fn(&mut [BinnedDatum], ThreadMember),
    ) {
        // step 0: apply a barrier
        // (obviously this is a no-op for a serial implementation)

        // we should obviously pre-allocate this memory
        let mut collect_pad = [BinnedDatum::zeroed()];

        // step 1: each team member calls the get_datum_bin_pair closure
        get_datum_bin_pair(&mut collect_pad, ThreadMember::new(0_u32));

        // step 2: gather the datum-bin pairs into the memory of a single
        //         team member
        // (obviously this is a no-op for a serial implementation)

        // step 3: the team member holding every datum-bin pair now uses
        //         them to update statepack
        let bin_index = collect_pad[0].bin_index;
        if binned_statepack.0.n_states() > bin_index {
            accum.consume(
                &mut binned_statepack.0.get_state_mut(bin_index),
                &collect_pad[0].datum,
            );
        }
    }
}

pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn drive_reduce(
        &mut self,
        out: &mut StatePackViewMut,
        reduction_spec: &impl ReductionSpec,
        n_members_per_team: NonZeroU32,
        n_teams: NonZeroU32,
    ) -> Result<(), &'static str> {
        let team_param = StandardTeamParam {
            n_members_per_team: n_members_per_team.get() as usize,
            n_teams: n_teams.get() as usize,
        };

        let accum = reduction_spec.get_accum();

        if team_param.n_members_per_team != 1 {
            Err("only supports 1 member per team")
        } else if team_param.n_teams != 1 {
            // todo: we should add support for n_teams (that's straight-forward)
            Err("only supports 1 team")
        } else if (out.n_states() != reduction_spec.n_bins())
            || (out.state_size() != accum.accum_state_size())
        {
            Err("out has the wrong shape")
        } else {
            // allocate a temporary statepack
            let mut tmp = vec![0.0; out.total_size()];
            let mut tmp_statepack =
                StatePackViewMut::from_slice(out.n_states(), out.state_size(), &mut tmp);

            // now we initialize statepack (this is inefficient!)
            for i in 0..tmp_statepack.n_states() {
                accum.init_accum_state(&mut tmp_statepack.get_state_mut(i));
            }

            // if we had multiple teams this would theoretically be a for loop
            {
                // move tmp_statepack into the shared_statepack variable
                let mut shared_statepack = DummyWrapper(tmp_statepack);

                fill_single_team_statepack(
                    &mut shared_statepack,
                    &mut SerialTeam,
                    0, // <- the team_id,
                    &team_param,
                    reduction_spec,
                );

                // move the contents of shared_statepack back to tmp_statepack
                // (if something goes wrong, it's probably due to an error
                // right here)
                tmp_statepack = shared_statepack.0;
            }

            // if we supported multiple teams, we'd perform a reduction here
            // to consolidate all of the temporary statepacks

            // finally, update out
            for i in 0..out.n_states() {
                accum.merge(&mut out.get_state_mut(i), &tmp_statepack.get_state(i));
            }
            Ok(())
        }
    }
}

// we may want to look back at older commits for inspiration for implementing
// a serial executor that can simulate teams with arbitrary sizes and arbitrary
// numbers of threads
