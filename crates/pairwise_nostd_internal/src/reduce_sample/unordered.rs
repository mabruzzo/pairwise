// this file is intended for illustrative purposes
// see module-level documentation for chunked.rs for more detail

use crate::accumulator::{
    Accumulator, DataElement, Mean, merge_full_statepacks, reset_full_statepack,
};
use crate::parallel::{BinnedDataElement, ReductionSpec, StandardTeamParam, ThreadMember};
use crate::reduce_sample::chunked::{QuadraticPolynomial, SampleDataStreamView};
use crate::state::StatePackViewMut;

// Version 0: our simple naive implementation
pub fn naive_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
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
            mean_times_weight_bins[bin_index] += w * f.call(x);
        }
    }
}

// Version 1: an implementation that uses accumulator machinery
pub fn accumulator_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
) {
    assert_eq!(accum.accum_state_size(), statepack.state_size());
    let n_bins = statepack.n_states();

    for i in 0..stream.len() {
        let bin_index = stream.bin_indices[i];
        let datum = DataElement {
            value: f.call(stream.x_array[i]),
            weight: stream.weights[i],
        };
        if bin_index < n_bins {
            accum.consume(&mut statepack.get_state_mut(bin_index), &datum);
        }
    }
}

// At this point, we will introduce a new version of the function that
// rearranges the work in a manner that would be suitable for parallelization.
//
// For our purposes, we define 2 levels of parallelized processing:
// 1. a fine-grained synchronous/collaborative level where the instructions
//    are applied to multiple values that are packed together in memory in a
//    synchronized manner
//    - on a CPU, this parallelism is achieved with SIMD instructions
//    - on a GPUs, these operations are performed generally performed by
//      threads grouped in a Warp (AMD calls them Wave Fronts)
//    - technically GPU threads in a Warp have a lot more flexibility, but
//      they perform extremely well in these contexts. But we target this sort
//      of parallelism for maximum portability.
// 2. a coarser-grained parallelism where separate threads operate much more
//    independently from each other.
//
// We will focus on rewriting the function in a manner conducive to the first
// type and afterwards, we'll write another version conducive to both types

// Version 2: In this version we organize the work in a manner conducive to
// fine-grained synchronous parallel processing.
// - we won't actually achieve this kind of processing in this function, but
//   we'll explain how we might do it
// - in a later version of this functionality, we will worry about parallelism
//
// Motivating our design.
// - ideally, we would love to somehow parallelize the process of generating
//   the data element and bin-index from the data source, and using these
//   values to update the appropriate accum_state bins.
// - Unfortunately, the unpredictable order is a large obstacle to
//   parallelizing the update of accumulator state (in a general way)
// - So, we do the best we possibly can. And treat the task of generating the
//   (data-element, bin-index) pair separately from updating the appropriate
//   accumulator state. This is a batching strategy:
//   - pre-generate a batch of pairs and store them in a collection-pad
//     (this operation can be parallelized)
//   - then we update the accumulator from the batch of pairs (this must be
//     sequential)
//
// Because this function is defined in a no_std, the collection-pad is
// pre-allocated.
//
// It also accepts an optional tuple holding the starting & stopping indices to
// consider from the stream (more on that later)
pub fn restructured1_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
    collect_pad: &mut [BinnedDataElement],
    start_stop_idx: Option<(usize, usize)>,
) {
    let batch_size = collect_pad.len();
    assert!(batch_size > 0);
    assert_eq!(accum.accum_state_size(), statepack.state_size());
    let n_bins = statepack.n_states();

    let (i_start, i_stop) = start_stop_idx.unwrap_or((0, stream.len()));
    assert!(i_stop >= i_start); // when they are equal, we do no work

    let n_batches = (i_stop - i_start).div_ceil(batch_size);
    for batch_idx in 0..n_batches {
        // While we execute the next loop sequentially, we can parallelize it:
        // - on CPUs, it can be vectorized
        // - on GPUs, threads will work together
        #[allow(clippy::needless_range_loop)]
        for pad_idx in 0..batch_size {
            let i = i_start + batch_idx * batch_size + pad_idx;

            collect_pad[pad_idx] = if i == i_stop {
                // NOTE: we can process an arbitrary number of zeroed data
                //       elements without affecting the output since it holds
                //       a weight of 0
                BinnedDataElement::zeroed()
            } else {
                BinnedDataElement {
                    bin_index: stream.bin_indices[i],
                    datum: DataElement {
                        value: f.call(stream.x_array[i]),
                        weight: stream.weights[i],
                    },
                }
            };
        }

        // now we sequentially process the (data-element, bin-index) pairs
        // -> there isn't a general, portable way to do this
        for e in collect_pad.iter() {
            let bin_index = e.bin_index;
            let datum = &e.datum;

            if bin_index < n_bins {
                accum.consume(&mut statepack.get_state_mut(bin_index), datum);
            }
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

// now we finally write a function that carves up the task in a way conducive
// to achieving the coarser grained parallelism.
//
// Essentially, the strategy is to cut up the index-range into segment.
// - we accumulate the contributions of each segment in a separate temporary
//   statepack (we use `restructured1_mean_unordered` to actually gather these
//   contributions)
// - at the very end, we gather up all the contributions and use them to update
//   the output statepack
//
// scratch_statepack_storage is provided to simulate the fact that each thread
pub fn restructured2_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
    scratch_statepacks: &mut [StatePackViewMut],
    collect_pad: &mut [BinnedDataElement],
) {
    // a scratch_statepack has been provided for every segment
    let n_segments = scratch_statepacks.len();
    assert!(n_segments > 0);

    // each pass through this loop considers a separate index-segment.
    // -> this work could be distributed among different threads (or thread
    //    groups)
    // -> if we did parallelize this loop, each thread (or thread group) would
    //    need to allocate its own collect_pad
    #[allow(clippy::needless_range_loop)]
    for seg_index in 0..n_segments {
        let local_statepack = &mut scratch_statepacks[seg_index];
        reset_full_statepack(accum, local_statepack);

        // determine DataStream indices to be processed in the current segment
        let idx_bounds = segment::get_index_bounds(stream.len(), seg_index, n_segments);

        // now do the work!
        restructured1_mean_unordered(
            stream,
            f,
            accum,
            local_statepack,
            collect_pad,
            Some(idx_bounds),
        );
    }

    // after all the above loop is done, let's merge together our results such
    // that scratch_statepacks[0] holds all the contributions
    consolidate_scratch_statepacks(accum, scratch_statepacks);

    // finally add the contribution to statepack
    merge_full_statepacks(accum, statepack, scratch_statepacks.first().unwrap());
}

/* THIS ISN'T READY YET!!
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
*/
