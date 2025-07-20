#[cfg(feature = "cuda")]
use cust_core::DeviceCopy;

// Defining some basic functionality for implementing this example:
// ================================================================

// for simplicity, we assume that the function that operates on the data stream
// can always be modelled as a quadratic polynomial (this assumption is
// primarily made to simplify the code)

/// Models the quadratic polynomial: `a*x^2 + b*x + c`

#[derive(Clone, Copy)]
#[cfg_attr(feature = "cuda", derive(DeviceCopy))]
#[repr(C)]
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

#[derive(Clone, Copy)]
#[repr(C)]
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

    pub fn first_index_of_chunk(&self, i: usize) -> Result<usize, &'static str> {
        if let Some(chunk_lens) = &self.chunk_lens {
            if i >= chunk_lens.len() {
                Err("i exceeds the number of chunks")
            } else {
                Ok(chunk_lens.iter().take(i).sum())
            }
        } else {
            Err("the stream doesn't have chunks")
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bin_indices.len()
    }
}
