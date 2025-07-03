// the reason this is named mod.rs has to do with some complexities of how
// testing is handled
//
// we are following the advice of the rust book
// https://doc.rust-lang.org/book/ch11-03-test-organization.html#submodules-in-integration-tests

use ndarray::Array2;
use pairwise::Accumulator;

// based on numpy!
// https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
//
// I suspect we'll use this a lot! If we may want to define
// a `assert_isclose!` macro to provide a nice error message (or an
// `assert_allclose!` macro to operate upon arrays)
pub fn isclose(actual: f64, ref_val: f64, rtol: f64, atol: f64) -> bool {
    let actual_nan = actual.is_nan();
    let ref_nan = ref_val.is_nan();
    if actual_nan || ref_nan {
        actual_nan && ref_nan
    } else {
        (actual - ref_val).abs() <= (atol + rtol * ref_val.abs())
    }
}

pub fn prepare_statepacks(n_spatial_bins: usize, accum: &impl Accumulator) -> Array2<f64> {
    assert!(n_spatial_bins > 0);
    let mut statepacks = Array2::<f64>::zeros((accum.statepack_size(), n_spatial_bins));
    for mut col in statepacks.columns_mut() {
        accum.reset_statepack(&mut col);
    }
    statepacks
}
