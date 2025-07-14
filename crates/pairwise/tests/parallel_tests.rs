use pairwise_nostd_internal::reduce_sample::{
    chunked::{SampleDataStreamView, naive_mean_chunked},
    unordered::naive_mean_unordered,
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

enum StreamKind {
    Unordered,
    Chunked,
}

fn wrapped_naive(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    version: StreamKind,
    n_bins: usize,
) -> common::BinnedStatMap {
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
    HashMap::from([
        ("mean", mean_times_weight_bins),
        ("weight", tot_weight_bins),
    ])
}

mod tests {
    use super::*;

    #[test]
    fn test_reduce_sample() {
        let seed = 10582441886303702641_u64;
        let num_chunks = NonZeroU32::new(26_u32).unwrap();
        let max_chunk_size = NonZeroU8::new(35_u8).unwrap();
        let max_bin_index = 12_usize;
        let stream = setup_sample_data_stream(seed, num_chunks, max_chunk_size, max_bin_index);

        let n_bins = 9; // this is intentionally smaller than max_bin_index

        let f = |x: f64| x.powi(3) + x.powi(2) - 1.0;

        let ref_map = wrapped_naive(&stream.as_view(), &f, StreamKind::Unordered, n_bins);
        let other_map = wrapped_naive(&stream.as_view(), &f, StreamKind::Chunked, n_bins);

        let rtol_atol_sets = HashMap::from([("weight", [0.0, 0.0]), ("mean", [0.0, 0.0])]);
        common::assert_consistent_results(&other_map, &ref_map, &rtol_atol_sets);
    }
}
