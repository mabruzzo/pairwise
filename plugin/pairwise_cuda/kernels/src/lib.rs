use cuda_std::prelude::*;
use pairwise_nostd_internal::reduce_sample::common::{QuadraticPolynomial, SampleDataStreamView};

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn naive_mean_chunked(
    bin_indices: &[usize],
    x_array: &[f64],
    weights: &[f64],
    chunk_lens: &[usize],
    f: QuadraticPolynomial,
    out: *mut f64,
) {
    if chunk_lens.len() == 0 {
        return ();
    }
    let data_stream = if let Ok(data_stream) =
        SampleDataStreamView::new(bin_indices, x_array, weights, Some(chunk_lens))
    {
        data_stream
    } else {
        return ();
    };
    let idx = thread::index_1d() as usize;
    if (idx == 0) {
        let elem = unsafe { &mut *out.add(idx) };
        *elem = f.call(data_stream.x_array[0]);
    }
}
