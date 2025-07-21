use crate::device_replicate_view::{DeviceReplicateView, RemoteMemoryRegistry};
use core::{convert::From, marker::PhantomData, ptr, slice};

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

// All of the following is intended to support GPU programming. This is all
// implemented manually, but ideally we would use a derive trait (or something)

#[derive(Clone, Copy)]
#[cfg_attr(feature = "cuda", derive(DeviceCopy))]
#[repr(C)]
pub struct SampleDataStreamViewMsgPacket<'a> {
    bin_indices: *const usize,
    bin_indices_len: usize,

    x_array: *const f64,
    x_array_len: usize, // this is unnecessary

    weights: *const f64,
    weights_len: usize, // this is unnecessary

    chunk_lens: *const usize,
    chunk_lens_len: usize, // this is very necessary

    // I think I'm using PhantomData properly
    _marker: PhantomData<&'a ()>,
}

#[cfg(feature = "cuda")]
unsafe impl<'a> DeviceReplicateView for SampleDataStreamView<'a> {
    // the lifetime on the preceeding line indicates that this implementation
    // block applies to a SampleDataStreamView with ANY lifetime

    // the lifetime parameters here account for the fact that the associated
    // type is parameterized in terms of a lifetime.
    type MessagePacket<'x> = SampleDataStreamViewMsgPacket<'x>;

    fn create_message_packet<'b, 'c, 'r>(
        &'b self,
        registry: &'c mut impl RemoteMemoryRegistry<'r>,
    ) -> Self::MessagePacket<'r>
    where
        // there is an implied bound that 'c outlives 'r
        'r: 'b, // 'r outlives 'b
    {
        let (chunk_lens_ptr, chunk_lens_len) = match self.chunk_lens {
            Some(chunk_lens) => (
                registry.reserve_dev_ptr_from_slice(chunk_lens),
                chunk_lens.len(),
            ),
            None => (ptr::null(), 0),
        };
        SampleDataStreamViewMsgPacket {
            bin_indices: registry.reserve_dev_ptr_from_slice(self.bin_indices),
            bin_indices_len: self.bin_indices.len(),
            x_array: registry.reserve_dev_ptr_from_slice(self.x_array),
            x_array_len: self.x_array.len(),
            weights: registry.reserve_dev_ptr_from_slice(self.weights),
            weights_len: self.weights.len(),
            chunk_lens: chunk_lens_ptr,
            chunk_lens_len,
            _marker: PhantomData,
        }
    }
}

//TODO: rename feature = "cuda" to feature = "cuda-support" to avoid confusion
//      with target_os = "cuda" (the latter is means that we are explicitly
//      compiling a GPU kernel)
#[cfg(target_os = "cuda")]
impl<'a> From<SampleDataStreamViewMsgPacket<'a>> for SampleDataStreamView<'a> {
    #[inline]
    fn from(msg_packet: SampleDataStreamViewMsgPacket<'a>) -> SampleDataStreamView<'a> {
        // we may want to stop wrapping chunk_lens
        let chunk_lens: Option<&'a [usize]> = if msg_packet.chunk_lens_len == 0 {
            None
        } else {
            unsafe {
                Some(slice::from_raw_parts(
                    msg_packet.chunk_lens,
                    msg_packet.chunk_lens_len,
                ))
            }
        };

        unsafe {
            SampleDataStreamView {
                bin_indices: slice::from_raw_parts(
                    msg_packet.bin_indices,
                    msg_packet.bin_indices_len,
                ),
                x_array: slice::from_raw_parts(msg_packet.x_array, msg_packet.x_array_len),
                weights: slice::from_raw_parts(msg_packet.weights, msg_packet.weights_len),
                chunk_lens,
            }
        }
    }
}
