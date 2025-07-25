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

/// Check if a 3D array shape (for a View3DSpec) is valid
fn check_shape(shape_zyx: &[usize; 3]) -> Result<(), &'static str> {
    if shape_zyx.contains(&0) {
        Err("shape_zyx must not hold 0")
    } else {
        Ok(())
    }
}

/// View3DSpec specifies how a "3D" array is laid out in memory.
///
/// The array _must_ be contiguous along the fast axis, which is axis 2. For
/// concreteness, an array with shape `[a, b, c]`, has `a` elements along axis
/// 0 and `c` elements along axis 2.
///
/// # How it's Intended to be Used
/// The basic premise is that an instance of this type is used alongside a
/// slice and maps 3D indices to the slice's 1d index. This type is most
/// useful, compared to say using a full blow multidimensional view type
/// (e.g. ndarray::ArrayView3), when working with multiple separate slices
/// that are all interpreted as multidimensional arrays that all share a
/// common shape and data layout.
#[allow(dead_code)] // this is a temporary stopgap solution
#[derive(Clone)]
pub struct View3DSpec {
    // Developer Notes
    // ---------------
    // - I suspect that we'll probably want to reuse this for 2D arrays (at
    //   least initially - if performance is a problem, we can always add a
    //   type specialized for 2D at a later date)
    //   - you would represent a 2d array by setting the length along axis 0
    //     to always be 1
    //   - writing a specialized 2d version would probably only be marginally
    //     faster for most methods (the exception is `reverse_map_idx`, which
    //     could be significantly faster, but we don't need it in most cases)
    // - we use signed ints because we do a lot of math with negative offsets
    //   and want to avoid excessive casts
    // - we should consider moving away from using z, y, and x since those can
    //   be mapped arbitrarily. Instead I think we should talk about axis-0,
    //   axis-1, and axis-2 (the way numpy does it)
    shape_zyx: [isize; 3],
    strides_zyx: [isize; 3],
}

#[allow(dead_code)] // <- this is a temporary stopgap solution
impl View3DSpec {
    /// Create a contiguous-in-memory View3DSpec from shape_zyx alone
    pub fn from_shape_contiguous(shape_zyx: [usize; 3]) -> Result<View3DSpec, &'static str> {
        check_shape(&shape_zyx)?;
        Ok(Self {
            shape_zyx: [
                shape_zyx[0] as isize,
                shape_zyx[1] as isize,
                shape_zyx[2] as isize,
            ],
            strides_zyx: [
                (shape_zyx[1] * shape_zyx[2]) as isize,
                shape_zyx[2] as isize,
                1_isize,
            ],
        })
    }

    /// Create a View3DSpec from shape_zyx and strides_zyx
    pub fn from_shape_strides(
        shape_zyx: [usize; 3],
        strides_zyx: [usize; 3],
    ) -> Result<View3DSpec, &'static str> {
        check_shape(&shape_zyx)?;

        if strides_zyx[2] != 1 {
            Err("the blocks must be contiguous along the fast axis")
        } else if strides_zyx[1] < shape_zyx[2] * strides_zyx[2] {
            Err("the length of the contiguous axis can't exceed strides_zyx[1]")
        } else if strides_zyx[0] < shape_zyx[1] * strides_zyx[1] {
            Err("the length of axis 1 can't exceed strides_zyx[0]")
        } else if strides_zyx[0] < strides_zyx[1] {
            Err("strides_zyx[1] must not exceed strides_zyx[0]")
        } else {
            Ok(Self {
                shape_zyx: [
                    shape_zyx[0] as isize,
                    shape_zyx[1] as isize,
                    shape_zyx[2] as isize,
                ],
                strides_zyx: [
                    strides_zyx[0] as isize,
                    strides_zyx[1] as isize,
                    strides_zyx[2] as isize,
                ],
            })
        }
    }

    /// the number of elements a slice must have to be described by `&self`
    ///
    /// This method is primarily intended for error checks.
    ///
    /// In more detail, the number of elements that a given instance,
    /// `idx_spec`, supports accessing with 3D indices is given by
    /// `idx_spec.shape().iter().product()`. If `idx_spec` describes a fully
    /// contiguous 3D array, that value is also returned by this method. While
    /// this type _does_ require the fast axis (i.e. axis-2) to be contiguous,
    /// it provides flexibility for the other axes. When `idx_spec` doesn't
    /// describe a fully contiguous array, the value that this method returns
    /// will exceed `idx_spec.shape().iter().product()`.
    pub fn required_length(&self) -> usize {
        let max_idx_1d = self.map_idx3d_to_1d(
            self.shape_zyx[0] - 1,
            self.shape_zyx[1] - 1,
            self.shape_zyx[2] - 1,
        );
        // based on the invariants enforced by the constructor, tmp is always
        // always be positive. Thus, it can be losslessly converted to usize
        (max_idx_1d as usize) + 1_usize
    }

    pub fn shape(&self) -> &[isize; 3] {
        &self.shape_zyx
    }

    /// map a 3D index to 1D
    pub fn map_idx3d_to_1d(&self, iz: isize, iy: isize, ix: isize) -> isize {
        iz * self.strides_zyx[0] + iy * self.strides_zyx[1] + ix
    }

    /// map a 1D index to a 3D index
    pub fn map_idx1d_to_3d(&self, idx: isize) -> [isize; 3] {
        let iz = idx / self.strides_zyx[0];
        let iy = (idx - iz * self.strides_zyx[0]) / self.strides_zyx[1];
        let ix = idx - iz * self.strides_zyx[0] - iy * self.strides_zyx[1];
        [iz, iy, ix]
    }
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

    // maps between a 3D & 1D index
    struct IdxMappingPair([isize; 3], isize);

    fn check_index_mapping(idx_spec: &View3DSpec, pairs: &[IdxMappingPair]) {
        for IdxMappingPair(idx_3d, idx_1d) in pairs {
            // try 3D to 1D:
            assert_eq!(
                idx_spec.map_idx3d_to_1d(idx_3d[0], idx_3d[1], idx_3d[2]),
                *idx_1d,
                "3D index, {:?}, was mapped to the wrong value",
                idx_3d,
            );
            // try 1D to 3D:
            assert_eq!(
                idx_spec.map_idx1d_to_3d(*idx_1d),
                idx_3d.as_slice(),
                "1D index, {}, was mapped to the wrong 3D index",
                idx_1d
            );
        }
    }

    #[test]
    fn idx_spec_simple() {
        let idx_spec = View3DSpec::from_shape_strides([2, 3, 4], [18, 6, 1]).unwrap();
        assert_eq!(idx_spec.shape(), [2, 3, 4].as_slice());
        assert_eq!(idx_spec.required_length(), 34);
        check_index_mapping(
            &idx_spec,
            &[
                IdxMappingPair([0, 0, 0], 0),
                IdxMappingPair([0, 0, 3], 3),
                IdxMappingPair([0, 1, 0], 6),
                IdxMappingPair([0, 2, 0], 12),
                IdxMappingPair([0, 2, 3], 15),
                IdxMappingPair([1, 0, 0], 18),
                IdxMappingPair([1, 1, 0], 24),
                IdxMappingPair([1, 2, 3], 33),
            ],
        );
    }

    #[test]
    fn idx_spec_contig() {
        let idx_spec = View3DSpec::from_shape_contiguous([2, 3, 4]).unwrap();
        assert_eq!(idx_spec.shape(), [2, 3, 4].as_slice());
        assert_eq!(idx_spec.required_length(), 24);
        check_index_mapping(
            &idx_spec,
            &[
                IdxMappingPair([0, 0, 0], 0),
                IdxMappingPair([0, 0, 3], 3),
                IdxMappingPair([0, 1, 0], 4),
                IdxMappingPair([0, 2, 0], 8),
                IdxMappingPair([0, 2, 3], 11),
                IdxMappingPair([1, 0, 0], 12),
                IdxMappingPair([1, 1, 0], 16),
                IdxMappingPair([1, 2, 3], 23),
            ],
        );
    }

    #[test]
    fn idx_props_errs() {
        assert!(View3DSpec::from_shape_contiguous([2, 3, 0]).is_err());
        assert!(View3DSpec::from_shape_contiguous([2, 0, 4]).is_err());
        assert!(View3DSpec::from_shape_contiguous([0, 3, 4]).is_err());

        assert!(View3DSpec::from_shape_strides([2, 3, 4], [18, 6, 0]).is_err());
        assert!(View3DSpec::from_shape_strides([2, 3, 4], [18, 3, 1]).is_err());
        assert!(View3DSpec::from_shape_strides([2, 3, 4], [18, 20, 1]).is_err());
        assert!(View3DSpec::from_shape_strides([2, 3, 4], [11, 4, 1]).is_err());
    }
}
