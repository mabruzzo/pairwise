//! Miscellaneous machinery used to implement the package

use pairwise_nostd_internal::squared_diff_norm;

use ndarray::ArrayView2;

/// computes the euclidean distance between 2 (mathematical) vectors taken from
/// `values_a` and `values_b`.
///
/// # Assumptions
/// This function assumes that spatial dimension varies along axis 0 of
/// `values_a` and `values_b` (and that it is the same value for both arrays)
pub fn diff_norm(
    values_a: ArrayView2<f64>,
    values_b: ArrayView2<f64>,
    i_a: usize,
    i_b: usize,
) -> f64 {
    // TODO come up with a better name for this function

    squared_diff_norm(values_a, values_b, i_a, i_b, values_a.shape()[0]).sqrt()
}
