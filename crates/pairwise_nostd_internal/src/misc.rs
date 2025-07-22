use ndarray::ArrayView2;

/// calculate the squared norm of the difference between two (mathematical) vectors
/// which are part of rust vecs that encodes a list of vectors with dimension on
/// the "slow axis"
///
/// # Note
/// This is only public so that it can be used in other crates (it isn't meant
/// to be exposed outside of the package)
pub fn squared_diff_norm(
    v1: ArrayView2<f64>,
    v2: ArrayView2<f64>,
    i1: usize,
    i2: usize,
    n_spatial_dims: usize,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..n_spatial_dims {
        let diff = v1[[k, i1]] - v2[[k, i2]];
        sum += diff * diff; // NOTE: .powi can't be used in no_std crates
    }
    sum
}

/// computes a dot product between the (mathematical) vectors taken from
/// `values_a` and `values_b`.
///
/// # Assumptions
/// This function assumes that spatial dimension varies along axis 0 of
/// `values_a` and `values_b` (and that it is the same value for both arrays)
pub fn dot_product(
    values_a: ArrayView2<f64>,
    values_b: ArrayView2<f64>,
    i_a: usize,
    i_b: usize,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..values_a.shape()[0] {
        sum += values_a[[k, i_a]] * values_b[[k, i_b]];
    }
    sum
}

/// computes the bounds of an index-segment
///
/// The premise is that we can break up `n_indices` up into `n_segments`
/// composed of 1 or more indices. If we had an array holding the bounds for
/// each segment (in increasing order), then the value at `seg_index` is
/// equivalent to the value returned by this function.
///
/// When `n_indices` is not an integer multiple of `n_segments`, this function
/// tries to make the segments as similar in size as possible
// TODO: refactor this function so that it doesn't contain a for-loop. When we
//       do refactor, we should test the new version EXTENSIVELY (with a lot
//       more tests than we currently have (because it is so easy to mess up)
// TODO: consider making n_segments have the type NonZeroUsize
pub fn segment_idx_bounds(n_indices: usize, seg_index: usize, n_segments: usize) -> (usize, usize) {
    let quotient = n_indices / n_segments;
    let remainder = n_indices % n_segments;
    let calc_seg_size = |seg_idx| quotient + ((seg_idx < remainder) as usize);

    // this is a stupid approach, but I've written this exact function many times!
    // Each time I do it cleverly, I **always** get it slightly wrong
    let mut start = 0;
    let mut stop = calc_seg_size(0);
    for cur_seg_idx in 1..=seg_index {
        start = stop;
        stop = start + calc_seg_size(cur_seg_idx);
    }
    (start, stop)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn segment_evenly_split() {
        assert_eq!(segment_idx_bounds(2, 0, 2), (0, 1));
        assert_eq!(segment_idx_bounds(2, 1, 2), (1, 2));

        assert_eq!(segment_idx_bounds(6, 0, 3), (0, 2));
        assert_eq!(segment_idx_bounds(6, 1, 3), (2, 4));
        assert_eq!(segment_idx_bounds(6, 2, 3), (4, 6));

        assert_eq!(segment_idx_bounds(15, 0, 5), (0, 3));
        assert_eq!(segment_idx_bounds(15, 1, 5), (3, 6));
        assert_eq!(segment_idx_bounds(15, 2, 5), (6, 9));
        assert_eq!(segment_idx_bounds(15, 3, 5), (9, 12));
        assert_eq!(segment_idx_bounds(15, 4, 5), (12, 15));
    }

    #[test]
    fn segment_unevenly_split() {
        // if we try a more clever implementation, we should expand coverage
        // in this case
        assert_eq!(segment_idx_bounds(22, 0, 5), (0, 5));
        assert_eq!(segment_idx_bounds(22, 1, 5), (5, 10));
        assert_eq!(segment_idx_bounds(22, 2, 5), (10, 14));
        assert_eq!(segment_idx_bounds(22, 3, 5), (14, 18));
        assert_eq!(segment_idx_bounds(22, 4, 5), (18, 22));
    }

    #[test]
    fn segment_more_segments_than_indices() {
        assert_eq!(segment_idx_bounds(2, 0, 5), (0, 1));
        assert_eq!(segment_idx_bounds(2, 1, 5), (1, 2));
        for seg_index in 2..5 {
            let (start, stop) = segment_idx_bounds(2, seg_index, 5);
            assert_eq!(
                start, stop,
                "expect the bounds to be equal when there are more segments\
                than bounds and we have already considered all indices"
            );
        }
    }

    #[test]
    fn segment_edge_cases() {
        let nidx_segidx_nseg_triples = [
            // consider: n_segments == 0 && seg_index < n_segments
            (0_usize, 0_usize, 2_usize),
            (0_usize, 1_usize, 2_usize),
            // consider: seg_index >= n_segments && n_segments == 0
            (0_usize, 2_usize, 2_usize),
            (0_usize, 3_usize, 2_usize),
        ];
        for (n_indices, seg_index, n_segments) in nidx_segidx_nseg_triples {
            let (start, stop) = segment_idx_bounds(n_indices, seg_index, n_segments);
            assert_eq!(
                start, stop,
                "start & stop are supposed to be the same value for \
                (n_indices = {} seg_index = {}, n_segments = {}",
                n_indices, seg_index, n_segments
            )
        }
    }
}
