// the reason this is named mod.rs has to do with some complexities of how
// testing is handled
//
// we are following the advice of the rust book
// https://doc.rust-lang.org/book/ch11-03-test-organization.html#submodules-in-integration-tests

use ndarray::Array2;
use pairwise::{Reducer, StatePackViewMut};
use std::collections::HashMap;

pub type BinnedStatMap = HashMap<&'static str, Vec<f64>>;

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

enum ArgDesignation {
    Actual,
    Expected,
}

// I think this should really be a macro
pub fn assert_consistent_results(
    actual: &HashMap<&'static str, Vec<f64>>,
    expected: &HashMap<&'static str, Vec<f64>>,
    rtol_atol_vals: &HashMap<&'static str, [f64; 2]>,
) {
    // it would be nice to handle the following more gracefully
    // (and provide a better error message!)
    let mut key_problem: Option<ArgDesignation> = None;
    if rtol_atol_vals.len() != actual.len() {
        key_problem = Some(ArgDesignation::Actual);
    } else if rtol_atol_vals.len() != expected.len() {
        key_problem = Some(ArgDesignation::Expected);
    } else {
        for k in rtol_atol_vals.keys() {
            if !actual.contains_key(k) {
                key_problem = Some(ArgDesignation::Actual);
                break;
            } else if !expected.contains_key(k) {
                key_problem = Some(ArgDesignation::Expected);
            }
        }
    }

    match key_problem {
        Some(ArgDesignation::Actual) => {
            panic!("`actual` & `rtol_atol_vals` don't have matching keys")
        }
        Some(ArgDesignation::Expected) => {
            panic!("`expected` & `rtol_atol_vals` don't have matching keys")
        }
        None => (),
    }

    for (key, [rtol, atol]) in rtol_atol_vals {
        let len = expected[key].len();
        assert_eq!(
            actual[key].len(),
            len,
            "the lengths of the '{}' entry in actual and ref are unequal",
            key,
        );

        for i in 0..len {
            let actual_val = actual[key][i];
            let ref_val = expected[key][i];
            assert!(
                isclose(actual_val, ref_val, *rtol, *atol),
                "map[\"{}\"][{}] values aren't to within rtol={}, atol={}\
            \n  actual   = {}\
            \n  expected = {}",
                key,
                i,
                rtol,
                atol,
                actual_val,
                ref_val
            );
        }
    }
}

// helper function that sets up the statepack, which holds one accum_state
// per spatial bin
pub fn prepare_statepack(n_spatial_bins: usize, reducer: &impl Reducer) -> Array2<f64> {
    assert!(n_spatial_bins > 0);
    let mut statepack = Array2::<f64>::zeros((reducer.accum_state_size(), n_spatial_bins));
    let mut statepack_view = StatePackViewMut::from_array_view(statepack.view_mut());
    for i in 0..n_spatial_bins {
        reducer.init_accum_state(&mut statepack_view.get_state_mut(i));
    }
    statepack
}
