use pairwise::{Mean, StatePackViewMut, get_output};

use pairwise_nostd_internal::{
    BinnedDataElement,
    reduce_sample::{
        chunked::{SampleDataStreamView, accumulator_mean_chunked, naive_mean_chunked},
        unordered::{
            accumulator_mean_unordered, naive_mean_unordered, restructured1_mean_unordered,
            restructured2_mean_unordered,
        },
    },
    reset_full_statepack,
};

use rand::distr::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;
use std::{
    collections::HashMap,
    num::{NonZeroU8, NonZeroU32},
};

mod common;

struct OwnedSampleDataStream {
    bin_indices: Vec<usize>,
    x_array: Vec<f64>,
    weights: Vec<f64>,
    chunk_lens: Option<Vec<usize>>,
}

impl OwnedSampleDataStream {
    fn as_view<'a>(&'a self) -> SampleDataStreamView<'a> {
        let chunk_lens: Option<&[usize]> = match &self.chunk_lens {
            Some(vec) => Some(vec),
            None => None,
        };
        SampleDataStreamView::new(
            &self.bin_indices,
            &self.x_array,
            &self.weights,
            chunk_lens, //self.chunk_lens.as_slice(),
        )
        .unwrap()
    }
}

/// setup an OwnedSampleDataStream
fn setup_sample_data_stream(
    seed: u64,
    num_chunks: NonZeroU32,
    max_chunk_size: NonZeroU8,
    max_bin_index: usize,
) -> OwnedSampleDataStream {
    let mut my_rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let value_sampler = {
        // we intentionally use integers so that operations are associative
        let value_dist = Uniform::try_from(-5..10).unwrap();
        move |rng: &mut Xoshiro256PlusPlus| value_dist.sample(rng) as f64
    };
    let weight_sampler = |_: &mut Xoshiro256PlusPlus| 1.0;
    let bin_index_dist = Uniform::try_from(0..=max_bin_index).unwrap();
    let chunk_size_dist = Uniform::try_from(1..=(max_chunk_size.get() as usize)).unwrap();

    let mut bin_indices: Vec<usize> = Vec::new();
    let mut x_array: Vec<f64> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut chunk_lens: Vec<usize> = Vec::new();

    for _ in 0_u32..num_chunks.get() {
        let chunk_len = chunk_size_dist.sample(&mut my_rng);
        let bin_index = bin_index_dist.sample(&mut my_rng);
        for _ in 0..chunk_len {
            x_array.push(value_sampler(&mut my_rng));
            weights.push(weight_sampler(&mut my_rng));
            bin_indices.push(bin_index);
        }
        chunk_lens.push(chunk_len)
    }

    OwnedSampleDataStream {
        bin_indices,
        x_array,
        weights,
        chunk_lens: Some(chunk_lens),
    }
}

#[derive(Debug, Clone, Copy)]
enum StreamKind {
    Unordered,
    Chunked,
}

#[derive(Debug)]
enum WrapperError {
    Unimplemented,
    Failure,
}

// in this case, we largely accept statepack just for the sake of maintaining
// a consistent interface with other flavors of this function
fn wrapped_naive(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    version: StreamKind,
    statepack: &mut StatePackViewMut,
) -> Result<common::BinnedStatMap, WrapperError> {
    let n_bins = statepack.n_states();
    let mut tot_weight_bins = vec![0.0; n_bins];
    let mut mean_times_weight_bins = vec![0.0; n_bins];
    match version {
        StreamKind::Unordered => {
            naive_mean_unordered(stream, f, &mut tot_weight_bins, &mut mean_times_weight_bins)
        }
        StreamKind::Chunked => {
            naive_mean_chunked(stream, f, &mut tot_weight_bins, &mut mean_times_weight_bins)
        }
    }
    for i in 0..n_bins {
        mean_times_weight_bins[i] /= tot_weight_bins[i];
    }
    Ok(HashMap::from([
        ("mean", mean_times_weight_bins),
        ("weight", tot_weight_bins),
    ]))
}

type BoxedFunc = Box<
    dyn Fn(
        &SampleDataStreamView,
        StreamKind,
        &mut StatePackViewMut,
    ) -> Result<common::BinnedStatMap, WrapperError>,
>;

// we picked something that will return an integer when passed an integer
#[inline(always)]
fn my_func(x: f64) -> f64 {
    x.powi(3) + x.powi(2) - 1.0
}

fn build_registry() -> HashMap<&'static str, BoxedFunc> {
    let mut out: HashMap<&'static str, BoxedFunc> = HashMap::new();

    //let naive_fn: BoxedFunc = );
    out.insert(
        "naive",
        Box::new(
            |stream: &SampleDataStreamView, version, statepack: &mut StatePackViewMut| {
                wrapped_naive(stream, &my_func, version, statepack)
            },
        ),
    );

    out.insert(
        "accumulator",
        Box::new(
            |stream: &SampleDataStreamView, version, statepack: &mut StatePackViewMut| {
                let accum = Mean;
                reset_full_statepack(&accum, statepack);
                match version {
                    StreamKind::Chunked => {
                        accumulator_mean_chunked(stream, &my_func, &accum, statepack)
                    }
                    StreamKind::Unordered => {
                        accumulator_mean_unordered(stream, &my_func, &accum, statepack)
                    }
                }
                let out = get_output(&accum, statepack);
                Ok(out)
            },
        ),
    );

    out.insert(
        "restructured1",
        Box::new(
            |stream: &SampleDataStreamView, version, statepack: &mut StatePackViewMut| {
                let accum = Mean;
                reset_full_statepack(&accum, statepack);
                match version {
                    StreamKind::Chunked => return Err(WrapperError::Unimplemented),
                    StreamKind::Unordered => {
                        let mut collect_pad = vec![BinnedDataElement::zeroed(); 4];
                        restructured1_mean_unordered(
                            stream,
                            &my_func,
                            &accum,
                            statepack,
                            &mut collect_pad,
                            None,
                        );
                    }
                }
                let out = get_output(&accum, statepack);
                Ok(out)
            },
        ),
    );

    out.insert(
        "restructured2",
        Box::new(
            |stream: &SampleDataStreamView, version, statepack: &mut StatePackViewMut| {
                let accum = Mean;
                reset_full_statepack(&accum, statepack);
                // we are going to break up the work in a way that is
                // equivalent to having 3 thread teams with 8 members per team
                let n_teams = 3;
                let n_members_per_team = 8;

                // allocate the scratch statepacks:
                let n_bins = statepack.n_states();
                let accum_size = statepack.state_size();
                let mut scratch_statepacks_buf = vec![0.0; statepack.total_size() * n_teams];
                let mut scratch_statepacks: Vec<StatePackViewMut> = scratch_statepacks_buf
                    .chunks_exact_mut(statepack.total_size())
                    .map(|chunk: &mut [f64]| {
                        StatePackViewMut::from_slice(n_bins, accum_size, chunk)
                    })
                    .collect();

                match version {
                    StreamKind::Chunked => return Err(WrapperError::Unimplemented),
                    StreamKind::Unordered => {
                        let mut collect_pad = vec![BinnedDataElement::zeroed(); n_members_per_team];
                        restructured2_mean_unordered(
                            stream,
                            &my_func,
                            &accum,
                            statepack,
                            &mut scratch_statepacks,
                            &mut collect_pad,
                        );
                    }
                }
                let out = get_output(&accum, statepack);
                Ok(out)
            },
        ),
    );

    //return Err(WrapperError::Unimplemented)

    out
}

mod tests {
    use super::*;

    #[test]
    fn test_reduce_sample() {
        // this tests that every version of the sample algorithm returns
        // exactly the same value.
        // -> We **ONLY** expect this to work if we operate on integer values
        // -> Different versions will generally produce slightly different
        //    results since they reorder operations and floating point addition
        //    is not strictly associative

        // generate the sample date stream
        let seed = 10582441886303702641_u64;
        let num_chunks = NonZeroU32::new(26_u32).unwrap();
        let max_chunk_size = NonZeroU8::new(35_u8).unwrap();
        let max_bin_index = 12_usize;
        let stream = setup_sample_data_stream(seed, num_chunks, max_chunk_size, max_bin_index);

        let n_bins = 9; // this is intentionally smaller than max_bin_index
        let mut statepack_buf = vec![0.0; 2 * n_bins];
        let mut statepack = StatePackViewMut::from_slice(n_bins, 2, &mut statepack_buf);

        // go through and generate the registry of all implementations for
        // the sample algorithm
        let registry = build_registry();

        // let's come up with the reference answers. The results subsequent
        // executions will be compared against this case
        let ref_map =
            registry["naive"](&stream.as_view(), StreamKind::Unordered, &mut statepack).unwrap();

        // specify tolerances (we currently require bitwise identical results!)
        let rtol_atol_sets = HashMap::from([("weight", [0.0, 0.0]), ("mean", [0.0, 0.0])]);

        // now, let's iterate over all of the implementations
        for (key, func) in &registry {
            println!("{}", key);
            for version in [StreamKind::Chunked, StreamKind::Unordered] {
                match func(&stream.as_view(), version, &mut statepack) {
                    Ok(calculated_map) => {
                        //println!("{:#?}", calculated_map);
                        common::assert_consistent_results(
                            &calculated_map,
                            &ref_map,
                            &rtol_atol_sets,
                        );
                    }
                    Err(WrapperError::Unimplemented) => {
                        eprintln!("(\"{}\", {:?}): isn't implemented", key, version)
                    }
                    _ => panic!(
                        "(\"{}\", {:?}): something went wrong implemented",
                        key, version
                    ),
                }
            }
        }
    }
}
