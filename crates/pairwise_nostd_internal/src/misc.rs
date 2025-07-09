use ndarray::ArrayView2;

// TODO use binary search, and have specialized version for regularly spaced bins?
/// Get the index of the bin that the squared distance falls into.
/// Returns None if its out of bounds.
///
/// # Note
/// This is only public so that it can be used in other files. It's not
/// intended to be used outside ofpublic
pub fn get_bin_idx(distance_squared: f64, squared_bin_edges: &[f64]) -> Option<usize> {
    // index of first element greater than distance_squared
    // (or squared_bin_edges.len() if none are greater)
    let mut first_greater = 0;
    for &edge in squared_bin_edges.iter() {
        if distance_squared < edge {
            break;
        }
        first_greater += 1;
    }
    if (first_greater == squared_bin_edges.len()) || (first_greater == 0) {
        None
    } else {
        Some(first_greater - 1)
    }
}

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

/// Check if a 3D array shape (for a View3DProps) is valid
fn check_shape(shape_zyx: &[usize; 3]) -> Result<(), &'static str> {
    if shape_zyx.contains(&0) {
        Err("shape_zyx must not hold 0")
    } else {
        Ok(())
    }
}

/// View3DProps specifies how a "3D" array is laid out in memory. It _must_ be
/// contiguous along the fast axis, which is axis 2.
///
/// For concreteness, an array with shape `[a, b, c]`, has `a` elements along
/// axis 0 and `c` elements along axis 2.
#[derive(Clone)]
pub struct View3DProps {
    // these are signed ints because we do a lot of math with negative offsets
    // and want to avoid excessive casts
    shape_zyx: [isize; 3],
    strides_zyx: [isize; 3],
}

impl View3DProps {
    /// Create a contiguous-in-memory View3DProps from shape_zyx alone
    pub fn from_shape_contiguous(shape_zyx: [usize; 3]) -> Result<View3DProps, &'static str> {
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

    /// Create a View3DProps from shape_zyx and strides_zyx
    pub fn from_shape_strides(
        shape_zyx: [usize; 3],
        strides_zyx: [usize; 3],
    ) -> Result<View3DProps, &'static str> {
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

    /// returns the number of elements that a slice must have to be described
    /// by self
    pub fn contiguous_length(&self) -> usize {
        (self.shape_zyx[0] * self.strides_zyx[0]) as usize
    }

    pub fn shape(&self) -> &[isize; 3] {
        &self.shape_zyx
    }

    /// map a 3D index to 1D
    pub fn map_idx(&self, iz: isize, iy: isize, ix: isize) -> isize {
        iz * self.strides_zyx[0] + iy * self.strides_zyx[1] + ix
    }

    pub fn reverse_map_idx(&self, idx: isize) -> [isize; 3] {
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
    fn idx_props_simple() {
        let idx_props = View3DProps::from_shape_strides([2, 3, 4], [18, 6, 1]).unwrap();
        assert_eq!(idx_props.map_idx(0, 0, 0), 0);
        assert_eq!(idx_props.map_idx(0, 0, 3), 3);
        assert_eq!(idx_props.map_idx(0, 1, 0), 6);
        assert_eq!(idx_props.map_idx(1, 1, 0), 24);
    }

    #[test]
    fn idx_props_contig() {
        let idx_props = View3DProps::from_shape_contiguous([2, 3, 4]).unwrap();
        assert_eq!(idx_props.map_idx(0, 0, 0), 0);
        assert_eq!(idx_props.map_idx(0, 0, 3), 3);
        assert_eq!(idx_props.map_idx(0, 1, 0), 4);
        assert_eq!(idx_props.map_idx(1, 1, 0), 16);
    }

    #[test]
    fn idx_props_errs() {
        assert!(View3DProps::from_shape_contiguous([2, 3, 0]).is_err());
        assert!(View3DProps::from_shape_contiguous([2, 0, 4]).is_err());
        assert!(View3DProps::from_shape_contiguous([0, 3, 4]).is_err());

        assert!(View3DProps::from_shape_strides([2, 3, 4], [18, 6, 0]).is_err());
        assert!(View3DProps::from_shape_strides([2, 3, 4], [18, 3, 1]).is_err());
        assert!(View3DProps::from_shape_strides([2, 3, 4], [18, 20, 1]).is_err());
        assert!(View3DProps::from_shape_strides([2, 3, 4], [11, 4, 1]).is_err());
    }
}
