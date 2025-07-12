//! This crate is primarily intended to define a simple example of how the
//! reduction-parallelism functionality works. We don't currently intend to
//! expose anything defined here as part of the public API.
//!
//! This serves 2 purposes:
//! 1. a pedagogical resource for learning/teaching the underlying machinery
//! 2. simplified logic we can use for easy unit-testing
//!
//! # Overview
//! This crate effectively solves a simple problem with increasingly
//! sophisticated solutions that are representative of how the reduction
//! machinery is intended to be used.
//!
//! # The Problem
//! Suppose we have a stream of elements, where each element is composed of a
//! precomputed bin-index, `i`, a value, `x`, and a weight `w`. For some
//! function `f`, we want to compute the weighted average value of `f(x)`.

use crate::accumulator::{Accumulator, DataElement, Mean};
use crate::state::{AccumStateViewMut, StatePackViewMut};

pub struct SampleDataStreamView<'a> {
    bin_indices: &'a [usize],
    x_array: &'a [f64],
    weights: &'a [f64],
    chunk_lens: Option<&'a [usize]>,
}

impl<'a> SampleDataStreamView<'a> {
    // ignore the chunk_lens argument
    pub fn new(
        bin_indices: &'a [usize],
        x_array: &'a [f64],
        weights: &'a [f64],
        chunk_lens: Option<&'a [usize]>,
    ) -> Result<Self, &'static str> {
        let len = bin_indices.len();

        if chunk_lens.is_some_and(|arr| len != arr.iter().sum()) {
            Err("the values in chunk_lens don't add up the bin_indices.len()")
        } else if (len != x_array.len()) || (len != weights.len()) {
            Err("the lengths of the slices are NOT consistent")
        } else {
            Ok(SampleDataStreamView {
                bin_indices,
                x_array,
                weights,
                chunk_lens,
            })
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bin_indices.len()
    }
}

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

pub fn naive_mean_chunked(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    tot_weight_bins: &mut [f64],
    mean_times_weight_bins: &mut [f64],
) {
    let n_bins = tot_weight_bins.len();
    assert_eq!(n_bins, mean_times_weight_bins.len());

    let Some(chunk_lens) = stream.chunk_lens else {
        panic!("the data stream wasn't chunked!")
    };

    let mut i_global = 0;
    for chunk_len in chunk_lens.iter().cloned() {
        let bin_index = stream.bin_indices[i_global];
        if bin_index < n_bins {
            let mut local_tot_weight = 0.0;
            let mut tmp_mean_times_weight = 0.0;
            for i in i_global..(i_global + chunk_len) {
                local_tot_weight += stream.weights[i];
                tmp_mean_times_weight += stream.weights[i] * f(stream.x_array[i]);
            }
            tot_weight_bins[bin_index] += local_tot_weight;
            mean_times_weight_bins[bin_index] += tmp_mean_times_weight;
        }
        i_global += chunk_len
    }
}

pub fn accumulator_mean_chunked(
    stream: &SampleDataStreamView,
    f: &impl Fn(f64) -> f64,
    accum: &Mean,
    statepack: &mut StatePackViewMut,
) {
    let Some(chunk_lens) = stream.chunk_lens else {
        panic!("the data stream wasn't chunked!")
    };

    let n_bins = statepack.n_states();
    assert_eq!(accum.accum_state_size(), statepack.state_size());
    assert_eq!(accum.accum_state_size(), 2);
    let mut tmp_buffer = [0.0; 2];
    let mut tmp_accum_state = AccumStateViewMut::from_contiguous_slice(&mut tmp_buffer);

    let mut i_global = 0;
    for chunk_len in chunk_lens.iter().cloned() {
        let bin_index = stream.bin_indices[i_global];
        if bin_index < n_bins {
            accum.reset_accum_state(&mut tmp_accum_state);
            for i in i_global..(i_global + chunk_len) {
                accum.consume(
                    &mut tmp_accum_state,
                    &DataElement {
                        value: f(stream.x_array[i]),
                        weight: stream.weights[i],
                    },
                );
            }
            accum.merge(
                &mut statepack.get_state_mut(bin_index),
                &tmp_accum_state.as_view(),
            );
        }
        i_global += chunk_len
    }
}
