//! This module is intended to introduce a placeholder type to help us
//! "block-out" logic to properly support SIMD in the future
//!
//! At some point in the future, we should start playing with the std::simd or
//! the glam crate
//!
//! The basic premise is that we create a struct, VecPack that mimics the role
//! of a vector-register.
//! - background: a vector-register holds `N` lanes of values. Many CPU
//!   architectures support 2 f64 values or 4 f32 values (or integer types).
//! - this struct acts like an array of `N` f64 values stored in
//!   regular memory. We will implement a subset of operations binary
//!   operations (e.g. addition, subtraction, multiplication) that operate on
//!   pairs of VecPack instances.
//! - if the struct has an alignment consistent with the corresponding vector
//!   register (& we provide enough hints about the input data's alignment),
//!   this can be really helpful for auto-vectorization
//!   -> this is especially if you have lots of inlined operations within a
//!      single function
//!   -> autovectorization may not happen if you try to pass the struct
//!      between (non-inlined functions). This has to do with the fact that
//!      function calling conventions make certain assumptions about the
//!      location of memory
//!   -> we can help with vectorization by specializing certain operation to
//!      use vector-intrinsics on a platform-by-platform basis.
//! - this sort of technique is used in many languages. AFAICT, rust's
//!   experimental std::simd feature also works this way. Thus, we should make
//!   efforts to use the same interface as std::simd (so we can swap out)
//!
//! At the moment, vectorization is NOT presently the goal of these types.
//! Instead, the goal is to "block-out" the requirements for achieving
//! vectorization (and we can worry about vectorization later).

use core::ops::{Add, AddAssign, Mul, Sub};
use std::usize;

// I don't love this
pub trait ScalarVecCompat<Other = Self, Output = Self>:
    Add<Other, Output = Output>
    + AddAssign<Other>
    + Sub<Other, Output = Output>
    + Mul<Other, Output = Output>
{
    type T;
    const NLANES: usize;
    fn sqrt(self) -> Self;
    fn copy_from(slice: &[Self::T], offset: usize) -> Self;
    fn zero() -> Self;
    fn shift_elements_left<const OFFSET: usize>(self, padding: Self::T) -> Self;
}

// make sure we don't let external code directly access the underlying array
// (in the future, we may consider enforcing alignment)
#[derive(Copy, Clone)]
struct VecPack<T, const N: usize>([T; N]);

// for Add, Sub, and Mul we may want to use a procedural macro
impl<T: Add<Output = T> + Copy, const N: usize> Add for VecPack<T, N> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut tmp = self.0;
        for i in 0..N {
            tmp[i] = self.0[i] + other.0[i];
        }
        Self(tmp)
    }
}

impl<T: AddAssign + Copy, const N: usize> AddAssign for VecPack<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub for VecPack<T, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut tmp = self.0;
        for i in 0..N {
            tmp[i] = self.0[i] - other.0[i];
        }
        Self(tmp)
    }
}

impl<T: Mul<Output = T> + Copy, const N: usize> Mul for VecPack<T, N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut tmp = self.0;
        for i in 0..N {
            tmp[i] = self.0[i] * other.0[i];
        }
        Self(tmp)
    }
}

impl<T: Copy, const N: usize> VecPack<T, N> {
    const LEN: usize = N;

    pub const fn len(&self) -> usize {
        return N;
    }

    #[inline]
    pub fn from_array(array: [T; N]) -> Self {
        Self(array.clone())
    }

    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() >= Self::LEN);
        Self(<[T; N]>::try_from(slice).unwrap())
    }

    pub fn to_array(&self) -> [T; N] {
        self.0
    }

    /// initializes all elements to a single value
    pub fn splat(value: T) -> Self {
        Self([value; N])
    }
}

impl<const N: usize> VecPack<f32, N> {
    // this currently requires that we use std
    pub fn sqrt(self) -> Self {
        let mut out = self;
        for i in 0..N {
            out.0[i] = self.0[i].sqrt();
        }
        out
    }
}

impl<const N: usize> VecPack<f64, N> {
    // this currently requires that we use std
    pub fn sqrt(self) -> Self {
        let mut out = self;
        for i in 0..N {
            out.0[i] = self.0[i].sqrt();
        }
        out
    }
}

/*
fn has_alignment(slc: &[f64], align: usize) -> bool {
    // there are special rules about manually tracking pointer lifetimes
    // (I think we will be okay since the slice definitely outlives the ptr)
    let ptr = slc.as_ptr();
    ptr.addr() % align == 0
}
*/

impl ScalarVecCompat for f64 {
    type T = Self;
    const NLANES: usize = 1;

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn copy_from(slice: &[f64], offset: usize) -> Self {
        slice[offset]
    }

    fn zero() -> Self {
        0.0
    }
    fn shift_elements_left<const OFFSET: usize>(self, padding: f64) -> f64 {
        if OFFSET > 0 {
            self
        } else {
            padding
        }
    }
}

impl<const N: usize> ScalarVecCompat for VecPack<f64, N> {
    type T = f64;
    const NLANES: usize = N;

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn copy_from(slice: &[f64], offset: usize) -> Self {
        Self::from_slice(&slice[offset..])
    }

    fn zero() -> Self {
        Self::splat(0.0)
    }

    // should be exactly like std::simd shift_elements_left
    fn shift_elements_left<const OFFSET: usize>(self, padding: f64) -> Self {
        if OFFSET >= N {
            Self::splat(padding)
        } else {
            let mut out = self;
            let n_copied = N - OFFSET;
            out.0[..n_copied].clone_from_slice(&self.0[OFFSET..]);
            out.0[n_copied..].fill(padding);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let v0 = VecPack::from_array([0., 1., 2., 3.]);
        let v1 = VecPack::from_array([4., 5., 6., 7.]);
        let sum = v0 + v1;
        assert_eq!(sum.to_array(), [4., 6., 8., 10.]);
    }

    #[test]
    fn sub() {
        let v0 = VecPack::from_array([0., 1., 2., 3.]);
        let v1 = VecPack::from_array([4., 5., 6., 7.]);
        let diff = v1 - v0;
        assert_eq!(diff.to_array(), [4., 4., 4., 4.]);
    }

    #[test]
    fn mul() {
        let v0 = VecPack::from_array([0., 1., 2., 3.]);
        let v1 = VecPack::from_array([4., 5., 6., 7.]);
        let product = v0 * v1;
        assert_eq!(product.to_array(), [0., 5., 12., 21.]);
    }

    const fn _get_nlanes<T: ScalarVecCompat>(_: &T) -> usize {
        T::NLANES
    }

    #[test]
    fn misc() {
        let v0 = VecPack::from_array([0., 1., 2., 3.]);
        assert_eq!(_get_nlanes(&v0), 4);
    }

    #[test]
    fn dummy() {
        let val = 4.0;
        assert_eq!(<f64 as ScalarVecCompat>::sqrt(val), 2.);
    }
}
