// this file is intended for illustrative purposes
// see module-level documentation for chunked.rs for more detail

use core::usize;

use crate::accumulator::{
    Accumulator, DataElement, Mean, merge_full_statepacks, reset_full_statepack,
};
use crate::parallel::{BinnedDataElement, MemberId, ReductionSpec, StandardTeamParam};
use crate::reduce_sample::chunked::SampleDataStreamView;
use crate::state::StatePackViewMut;

// let's write a naive implementation
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

// now let's write an implementation that uses chunks
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

// introduce some tools to help with the implementation
// -> the idea is to break a single index range into segments
mod segment {
    pub fn get_index_bounds(
        n_indices: usize,
        seg_index: usize,
        n_segments: usize,
    ) -> (usize, usize) {
        let nominal_seglen = n_indices / n_segments;
        // we can be smarter about how we divide the extra work
        // -> right now, we stick it all with the last seg_index
        let start = nominal_seglen * seg_index;
        if (seg_index + 1) == n_segments {
            (start, n_indices)
        } else {
            (start, start + nominal_seglen)
        }
    }
}

// consolidates the statepacks in such a way that scratch_statepacks[0]
// contains the results of every other statepack
//
// this function makes no guarantees about the final state of other
// entries within scratch_statepack
fn consolidate_scratch_statepacks(accum: &Mean, scratch_statepacks: &mut [StatePackViewMut]) {
    // if we wanted to simulate parallelism, then the way this loop works is
    // pretty inefficient
    for i in 1..scratch_statepacks.len() {
        let [main, other] = scratch_statepacks.get_disjoint_mut([0, i]).unwrap();
        merge_full_statepacks(accum, main, other);
    }
}

// now we write an implementation that cuts up the problem in a way that is
// well-suited for simple "flat" parallelism. The choice of the word "flat"
// will make more sense later.
//
// Essentially, the strategy is to cut up the index-range into chunks. This
// would be the easiest way to divide work among threads.
//
// scratch_statepack_storage is provided to simulate the fact that each thread
pub fn mockflatparallel_mean_unordered(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
    scratch_statepacks: &mut [StatePackViewMut],
) {
    // a scratch_statepack has been provided for every segment
    let n_segments = scratch_statepacks.len();
    assert!(n_segments > 0);

    // this for-loop simulates parallelism. Essentially the work done in each
    // pass through the loop would be done by a separate thread
    for seg_index in 0..n_segments {
        let local_statepack = &mut scratch_statepacks[seg_index];
        reset_full_statepack(accum, local_statepack);

        // determine DataStream indices to be processed in the current segment
        let idx_bounds = segment::get_index_bounds(stream.len(), seg_index, n_segments);

        // now do the work!
        _mockflatparallel_mean_unordered(stream, f, accum, local_statepack, idx_bounds);
    }

    // after all the above loop is done, let's merge together our results such
    // that scratch_statepacks[0] holds all the contributions
    consolidate_scratch_statepacks(accum, scratch_statepacks);

    // finally add the contribution to statepack
    merge_full_statepacks(accum, statepack, scratch_statepacks.get(0).unwrap());
}

// this helper function does the heavy lifting for
// mockflatparallel_mean_unordered
fn _mockflatparallel_mean_unordered(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
    i_bounds: (usize, usize),
) {
    let n_bins = statepack.n_states();
    for i in i_bounds.0..i_bounds.1 {
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

// TODO: come back and clean up this function and description (make it so that
//       the partitioning within a segment isn't hardcoded)

// now we write an implementation that cuts up the problem in a way that is
// well-suited for simple "nested" parallelism.
// - Like the flat case, we break indices of stream into segments
// - But now, when we process the segments, we break up segment processing
//
// This strategy lends itself well to teams of threads. The idea is that the
// team members can parallelize the or threading with
// vectorization. The idea is that:
// - members of  each subchunk can b
// It also works well when
// you want to use threading and SIMD
pub fn mocknestparallel_mean_unordered(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
    scratch_statepacks: &mut [StatePackViewMut],
) {
    // a scratch_statepack has been provided for every segment
    let n_segments = scratch_statepacks.len();
    assert!(n_segments > 0);

    // this for-loop simulates parallelism. Essentially the work done in each
    // pass through the loop would be done by a separate thread
    for seg_index in 0..n_segments {
        let local_statepack = &mut scratch_statepacks[seg_index];
        reset_full_statepack(accum, local_statepack);

        // determine DataStream indices to be processed in the current segment
        let idx_bounds = segment::get_index_bounds(stream.len(), seg_index, n_segments);

        // now do the work!
        _mockflatparallel_mean_unordered(stream, f, accum, local_statepack, idx_bounds);
    }

    // after all the above loop is done, let's merge together our results such
    // that scratch_statepacks[0] holds all the contributions
    consolidate_scratch_statepacks(accum, scratch_statepacks);

    // finally add the contribution to statepack
    merge_full_statepacks(accum, statepack, scratch_statepacks.get(0).unwrap());
}

// this helper function does the heavy lifting for
// mockflatparallel_mean_unordered
fn _mocknestedparallel_mean_unordered<const N_PARTIAL_RESULTS: usize>(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
    i_bounds: (usize, usize),
    subchunk: usize,
) {
    let n_bins = statepack.n_states();
    assert!(N_PARTIAL_RESULTS > 0);

    let n_steps = (i_bounds.1 - i_bounds.0).div_ceil(N_PARTIAL_RESULTS);

    let mut buf = [BinnedDataElement::zeroed(); N_PARTIAL_RESULTS];

    for main_step in 0..n_steps {
        // this inner loop simulates different threads in a team
        // - basically, we would be parallelizing the process of collecting
        //   data elements and their associated bin indices.
        // - in real world problems, gathering bin indices can be expensive
        for offset in 0..N_PARTIAL_RESULTS {
            let i = main_step * N_PARTIAL_RESULTS + offset;
            buf[offset] = if i < i_bounds.1 {
                BinnedDataElement {
                    bin_index: stream.bin_indices[i],
                    datum: DataElement {
                        value: f(stream.x_array[i]),
                        weight: stream.weights[i],
                    },
                }
            } else {
                // since the weight of this data element is 0, it won't affect
                // the accumulation
                BinnedDataElement::zeroed()
            };
        }

        // now we step through and all relevant entries to the accumulator:
        for e in buf.iter() {
            let bin_index = e.bin_index;
            let datum = &e.datum;

            if bin_index < n_bins {
                accum.consume(&mut statepack.get_state_mut(bin_index), datum);
            }
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
        todo!("implement me!");
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
