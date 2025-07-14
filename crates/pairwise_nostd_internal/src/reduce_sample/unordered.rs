use crate::accumulator::{Accumulator, DataElement, Mean};
use crate::parallel::{BinnedDataElement, MemberId, ReductionSpec, StandardTeamParam};
use crate::reduce_sample::chunked::SampleDataStreamView;
use crate::state::StatePackViewMut;

pub fn naive_mean_unordered(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    tot_weight_bins: &mut [f64],
    mean_times_weight_bins: &mut [f64],
) {
    let n_bins = tot_weight_bins.len();
    assert_eq!(n_bins, mean_times_weight_bins.len());

    for i in 0..stream.len() {
        let bin_index = stream.bin_indices[i];
        let x = stream.x_array[i];
        let w = stream.weights[i];
        if bin_index < n_bins {
            tot_weight_bins[bin_index] += w;
            mean_times_weight_bins[bin_index] += w * f(x);
        }
    }
}

pub fn accumulator_mean_unordered(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
) {
    assert_eq!(accum.accum_state_size(), statepack.state_size());
    let n_bins = statepack.n_states();

    for i in 0..stream.len() {
        let bin_index = stream.bin_indices[i];
        let datum = DataElement {
            value: f(stream.x_array[i]),
            weight: stream.weights[i],
        };
        if bin_index < n_bins {
            accum.consume(&mut statepack.get_state_mut(bin_index), &datum);
        }
    }
}

// we skipped over a few steps to get to this version
pub struct MeanUnorderedReduction<'a, F: Fn(f64) -> f64> {
    stream: SampleDataStreamView<'a>,
    f: &'a F, // todo: stop storing f (we should hard-code the function)
    accum: Mean,
    n_bins: usize,
}

impl<'a, F: Fn(f64) -> f64> MeanUnorderedReduction<'a, F> {
    pub fn new(stream: SampleDataStreamView<'a>, f: &'a F, accum: Mean, n_bins: usize) -> Self {
        Self {
            stream,
            f,
            accum,
            n_bins,
        }
    }
}

impl<'a, F: Fn(f64) -> f64> ReductionSpec for MeanUnorderedReduction<'a, F> {
    type AccumulatorType = Mean;

    fn get_accum(&self) -> &Self::AccumulatorType {
        &self.accum
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    fn inner_team_loop_bounds(
        &self,
        _outer_index: usize,
        team_id: usize,
        team_info: &StandardTeamParam,
    ) -> (usize, usize) {
        panic!("this isn't correct yet for multiple threads per team");
        let len = self.stream.len();
        if len <= team_info.n_teams {
            return if team_id < len {
                (team_id, team_id + 1)
            } else {
                (0, 0)
            };
        }

        let chunk_size = len / team_info.n_teams;
        let start = chunk_size * team_id;
        let stop = if (team_id + 1) < team_info.n_teams {
            start + chunk_size
        } else {
            len
        };
        (start, stop)
    }

    const NESTED_REDUCE: bool = false;

    fn get_datum_index_pair(
        &self,
        outer_index: usize,
        inner_index: usize,
        member_id: MemberId,
        team_param: &StandardTeamParam,
    ) -> BinnedDataElement {
        todo!("implement me!")
    }
}
