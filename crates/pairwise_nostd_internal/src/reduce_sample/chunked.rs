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
//!
//! We consider 2 versions of this problem.
//! -> in this file, we consider the scenario where we have chunks of data
//!    with a known bin-index
//! -> in unordered.rs, we consider a variation where we assume that the
//!    data is in a totally random order
//! The logic has been separated between files to facilitate side-by-side
//! comparisons

use crate::accumulator::{Accumulator, DataElement, Mean};
//use crate::parallel::{BinnedDataElement, MemberId, ReductionSpec, StandardTeamParam};
use crate::state::{AccumStateViewMut, StatePackViewMut};

// Defining some basic functionality for implementing this example:
// ================================================================

// for simplicity, we assume that the function that operates on the data stream
// can always be modelled as a quadratic polynomial (this assumption is
// primarily made to simplify the code)

/// Models the quadratic polynomial: `a*x^2 + b*x + c`
#[derive(Clone, Copy)]
pub struct QuadraticPolynomial {
    a: f64,
    b: f64,
    c: f64,
}

impl QuadraticPolynomial {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    pub fn call(&self, x: f64) -> f64 {
        self.c + x * (self.b + x * self.a)
    }
}

#[derive(Clone)]
pub struct SampleDataStreamView<'a> {
    // we only make the members public so that they can get reused in unordered.rs
    pub bin_indices: &'a [usize],
    pub x_array: &'a [f64],
    pub weights: &'a [f64],
    pub chunk_lens: Option<&'a [usize]>,
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

pub fn naive_mean_chunked(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
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
                tmp_mean_times_weight += stream.weights[i] * f.call(stream.x_array[i]);
            }
            tot_weight_bins[bin_index] += local_tot_weight;
            mean_times_weight_bins[bin_index] += tmp_mean_times_weight;
        }
        i_global += chunk_len
    }
}

pub fn accumulator_mean_chunked(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
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
                        value: f.call(stream.x_array[i]),
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
