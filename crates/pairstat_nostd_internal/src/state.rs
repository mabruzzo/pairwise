//! Introduces the [`AccumStateView`] and [`AccumStateViewMut`] types
//!
//! Currently these simply wrap `ndarray::ArrayView1<f64>` and
//! `ndarray::ArrayViewMut1<f64>` types, respectively. We hope to move away
//! from this in the future.
//!
//! # Why do we need separate types to represent immutable & mutable views?
//!
//! To be more clear: `ndarray::ArrayView1<f64>` and
//! `ndarray::ArrayViewMut1<f64>` seem like direct analogues to the
//! immutable and mutable slice types, `&[f64]` and `&mut [f64]`.
//! **Is there anything fundamentally special about rust's slice types?**
//!
//! It turns out that the answer is "yes"
//! - suppose that you had a type called `StatePackViewMut` that holds a block
//!   of memory and that represents a list of  accumulator states. To use a
//!   type called `AccumStateView` for both mutable and immutable views, you
//!   would have to be write methods with the following signatures:
//!     ```text
//!     impl StatePackViewMut {
//!         fn get_state_view(&self, usize i) -> &AccumStateView;
//!         fn get_state_view_mut(&mut self, usize i) -> &mut AccumStateView;
//!     }
//!     ```
//!   The fact that we return an immutable reference in one case and a mutable
//!   reference in the other case is the key takeaway here. Doing anything else
//!   (with a single view type) won't properly model lifetimes.
//! - This is very allowed if the `StatePackViewMut` already internally tracks
//!   a list of `AccumStateView` instances. But, we would like to avoid doing
//!   that for performance purposes (it would also be messy from a bookkeeping
//!   perspective).
//! - Instead, we would prefer to construct an instance of the `AccumStateView`
//!   return and return a mutable/immutable reference to that instance.
//! - This is forbidden in just about all cases because you would be returning
//!   a dangling reference to a cleaned up local variable.
//! - However, there is an exception for special kinds of types known as
//!   [Dynamically Sized Types](https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts)
//!   (DSTs). Rust's slices (e.g. `[f64]`) and string slices, `str` are
//!   prominent examples of DSTs. These are effectively implemented as
//!   "fat pointers" (i.e. they contain the pointer-address and metadata about
//!   the size of the referenced data).
//! - My immediate thought was: "why can't we do that?" It turns out that DSTs
//!   aren't a fully fledged concept. They are *extremely* restricted outside
//!   the standard library. We can *only* define a new DST by having it wrap an
//!   existing DST (namely the slice type, `[f64]`).
//! - We can't do that unless we make 1 of the following 2 restrictions:
//!     1. We force the contents of a `StatePackView` to always be contiguous
//!        (this is bad for GPU memory coalescing)
//!     2. We forbid holding a mutable `StatePackView` and an immutable
//!        `StatePackView` for a single `StatePackList` at a given time.
//!
//!   We should definitely re-evaluate this in the future, but I'm not willing
//!   to make these restrictions at this time.
//! - Any other approach for trying to implement `StatePackView` as a DST,
//!   would either:
//!     - be stupidly slow, OR
//!     - trigger undefined behavior when we create mutable & immutable
//!       `StatePackView` instances at the same time from the same
//!       `StatePackList`: you wouldn having mutable & immutable slices that
//!       include overlapping memory regions (this isn't allowed in safe or
//!       unsafe Rust)
use crate::misc::View2DUnsignedSpec;
use core::{
    num::NonZeroUsize,
    ops::{Index, IndexMut},
};

pub struct AccumStateView<'a> {
    // when we refactor this to stop wrapping ArrayView1, we really *need* to
    // wrap a pointer rather than a slice.
    // - this will introduce unsafe blocks of logic, but it's essential for
    //   avoiding a scenario with undefined behavior
    // - the concern is that we could end up in a scenario where we have a
    //   pair of `AccumStateView` and `AccumStateViewMut` instances both
    //   provide views to different regions of `StatePackViewMut`
    // - if we use slices then we will have a mutable slice and an immutable
    //   slice that reference an overlapping region of memory at the same time.
    //   That isn't allowed! It will trigger undefined behavior
    //
    // We probably need to use `core::marker::PhantomData` to probably track
    // lifetimes (see https://doc.rust-lang.org/nomicon/phantom-data.html)
    len: NonZeroUsize,
    stride: usize,
    data: &'a [f64],
}

impl<'a> AccumStateView<'a> {
    /// Private constructor used by other types in this module
    fn internal_new(len: NonZeroUsize, stride: usize, data: &'a [f64]) -> Self {
        debug_assert!(((len.get() - 1) * stride) < data.len());
        Self { len, stride, data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len.get()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<'a> Index<usize> for AccumStateView<'a> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index * self.stride)
    }
}

pub struct AccumStateViewMut<'a> {
    // when we refactor this to stop wrapping ArrayViewMut1, we really *need*
    // to wrap a pointer rather than a slice.
    // - this will introduce unsafe blocks of logic, but it's essential for
    //   avoiding a scenario with undefined behavior
    // - the concern is that we could end up in a scenario where we have a
    //   pair of `AccumStateView` and `AccumStateViewMut` instances both
    //   provide views to different regions of `StatePackViewMut`
    // - if we use slices then we will have a mutable slice and an immutable
    //   slice that reference an overlapping region of memory at the same time.
    //   That isn't allowed! It will trigger undefined behavior
    //
    // We probably need to use `core::marker::PhantomData` to probably track
    // lifetimes (see https://doc.rust-lang.org/nomicon/phantom-data.html)
    len: NonZeroUsize,
    stride: usize,
    data: &'a mut [f64],
}

impl<'a> AccumStateViewMut<'a> {
    /// Private constructor used by other types in this module
    fn internal_new(len: NonZeroUsize, stride: usize, data: &'a mut [f64]) -> Self {
        debug_assert!(((len.get() - 1) * stride) < data.len());
        Self { len, stride, data }
    }

    // consider returning an option rather than panicing
    pub fn from_contiguous_slice(data: &'a mut [f64]) -> Self {
        let Some(len) = NonZeroUsize::new(data.len()) else {
            panic!("can't construct an empty AccumStateViewMut");
        };
        let stride = 1;
        Self { len, stride, data }
    }

    pub fn as_view<'b>(&'b self) -> AccumStateView<'b> {
        AccumStateView {
            len: self.len,
            stride: self.stride,
            data: self.data,
        }
    }

    // todo: consider whether this should actually be a public method shipped
    // in a release
    pub fn fill(&mut self, val: f64) {
        self.data.fill(val);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len.get()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<'a> Index<usize> for AccumStateViewMut<'a> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index * self.stride)
    }
}

impl<'a> IndexMut<usize> for AccumStateViewMut<'a> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index * self.stride)
    }
}

/// Represents a read-only Collection of accumulator states
///
/// # Note
/// Ideally, we would be able to totally dispose of this type... But for now
/// it serves a purpose...
pub struct StatePackView<'a> {
    // see StatePackViewMut<'a> for refactoring notes
    data: &'a [f64],
    idx_spec: View2DUnsignedSpec,
}

impl<'a> StatePackView<'a> {
    pub fn from_slice(n_state: usize, state_size: usize, data: &'a [f64]) -> Self {
        assert!(n_state > 0);
        assert!(state_size > 0);
        let idx_spec = View2DUnsignedSpec::from_shape_contiguous([state_size, n_state]).unwrap();
        if idx_spec.required_length() > data.len() {
            panic!("data doesn't hold the appropriate number of elements");
        }
        Self { data, idx_spec }
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn get_state(&self, i: usize) -> AccumStateView<'_> {
        let start = self.idx_spec.map_idx2d_to_1d(0, i);
        let len = unsafe { NonZeroUsize::new_unchecked(self.state_size()) };
        AccumStateView::internal_new(len, self.idx_spec.strides()[0], &self.data[start..])
    }

    #[inline]
    pub fn state_size(&self) -> usize {
        self.idx_spec.shape()[0]
    }

    #[inline]
    pub fn n_states(&self) -> usize {
        self.idx_spec.shape()[1]
    }
}

/// Represents a collection of accumulator States
///
/// This type is often used to store a binned statepack.
///
/// # Data Representation
/// This type stores accumulator states in an interleaved manner. In other
/// words, the data associated with an [`StatePackView`] instance
/// returned by [`Self::get_state`] cannot be contiguous unless the
/// [`Self::n_states`] method returns `1`.[^single_elem]
///
/// ## Benefits
/// This choice has some theoretical benefits:
/// - when updating a binned statepack with values from another binned
///   statepack, this representation would facillitate SIMD optimizations
///   on a CPU. (It could also facillitate memory-colasence if the operation
///   were parallelized on GPUs)
/// - If you implemented [`crate::Team::calccontribs_combine_apply``] for a
///   team where each member corresponds to a vector lane, you would
///   theoretically want to hold the temporary accum_states in an interleaved
///   manner (as is done by this type)
///
/// ## Disadvantages
/// You **need** to implement [`AccumStateView`] and [`AccumStateViewMut`] in
/// terms of pointers if you want to access 2 [`AccumStateViewMut`] instances
/// (or a [`AccumStateViewMut`] instance and a [`AccumStateView`]) instance
/// from `self` at the same time. It is **IMPOSSIBLE** to do that if
/// [`AccumStateView`] and [`AccumStateViewMut`] are implemented in terms of
/// slices.
///
/// To be clear, if the types were implemented in terms of slices, the
/// fundamental issue is the slices would refer to overlapping regions of
/// memory (as I understand it, that will trigger undefined behavior)
///
/// # Note
/// There is some benefit to defining this even though it wraps ArrayViewMut2
/// since it helps contain all references to the ndarray package to a single
/// file.
///
/// [^single_elem]: Technically, a [`StatePackView`] instance returned by
///     [`Self::get_state`] could be considered contiguous if
///     [`Self::state_size`] returns `1`. But that's only because a 1-element
///     array is always contiguous.
pub struct StatePackViewMut<'a> {
    // when we refactor this to stop wrapping ArrayViewMut2, I *think*, we
    // probably want to wrap a pointer rather than a slice.
    //
    // If we use a pointer, we should consider whether we need to make use of
    // core::marker::PhantomData. I don't *think* it would necessary for this
    // scenario, but we should review
    // https://doc.rust-lang.org/nomicon/phantom-data.html
    data: &'a mut [f64],
    idx_spec: View2DUnsignedSpec,
}

impl<'a> StatePackViewMut<'a> {
    pub fn from_slice(n_state: usize, state_size: usize, data: &'a mut [f64]) -> Self {
        assert!(n_state > 0);
        assert!(state_size > 0);
        let idx_spec = View2DUnsignedSpec::from_shape_contiguous([state_size, n_state]).unwrap();
        if idx_spec.required_length() > data.len() {
            panic!("data doesn't hold the appropriate number of elements");
        }
        Self { data, idx_spec }
    }

    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn as_view<'b>(&'b self) -> StatePackView<'b> {
        StatePackView {
            data: self.data,
            idx_spec: self.idx_spec.clone(),
        }
    }

    #[inline]
    pub fn get_state(&self, i: usize) -> AccumStateView<'_> {
        let start = self.idx_spec.map_idx2d_to_1d(0, i);
        let len = unsafe { NonZeroUsize::new_unchecked(self.state_size()) };
        AccumStateView::internal_new(len, self.idx_spec.strides()[0], &self.data[start..])
    }

    #[inline]
    pub fn get_state_mut(&mut self, i: usize) -> AccumStateViewMut<'_> {
        let start = self.idx_spec.map_idx2d_to_1d(0, i);
        let len = unsafe { NonZeroUsize::new_unchecked(self.state_size()) };
        AccumStateViewMut::internal_new(len, self.idx_spec.strides()[0], &mut self.data[start..])
    }

    #[inline]
    pub fn state_size(&self) -> usize {
        self.idx_spec.shape()[0]
    }

    #[inline]
    pub fn n_states(&self) -> usize {
        self.idx_spec.shape()[1]
    }

    pub fn total_size(&self) -> usize {
        self.state_size() * self.n_states()
    }
}

/*
// keep in mind: we have explicitly reversed the axes order compared to [`StatePackViewMut`]
pub struct CollatedStatePackViewMut<'a> {
    data: &'a mut [f64],
    n_states: usize,
    state_size: usize,
}

impl<'a> CollatedStatePackViewMut<'a> {
    pub fn from_slice(
        n_states: usize,
        state_size: usize,
        data: &'a mut [f64],
    ) -> Result<Self, &'static str> {
        if (n_states * state_size) != data.len() {
            Err("slice has wrong length")
        } else if (n_states == 0) || (state_size == 0) {
            Err("can't represent an empty statepack")
        } else {
            Ok(Self {
                data,
                n_states,
                state_size,
            })
        }
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    pub fn n_states(&self) -> usize {
        self.n_states
    }

    #[inline]
    pub fn get_state(&self, i: usize) -> AccumStateView {
        assert!(i < self.n_states);
        let start = i * self.state_size;
        let stop = start + self.state_size;
        AccumStateView::from_contiguous_slice(&self.data[start..stop])
    }

    #[inline]
    pub fn get_state_mut(&mut self, i: usize) -> AccumStateViewMut {
        assert!(i < self.n_states);
        let start = i * self.state_size;
        let stop = start + self.state_size;
        AccumStateViewMut::from_contiguous_slice(&mut self.data[start..stop])
    }

    fn disjoint_state_mut_pair(&mut self, i: usize, j: usize) -> [AccumStateViewMut; 2] {
        assert!(i < self.n_states);
        assert!(j < self.n_states);
        let i_start = i * self.state_size;
        let i_stop = i_start + self.state_size;
        let j_start = j * self.state_size;
        let j_stop = j_start + self.state_size;

        let [slc_left, slc_right] = self
            .data
            .get_disjoint_mut([i_start..i_stop, j_start..j_stop])
            .unwrap();
        [
            AccumStateViewMut::from_contiguous_slice(slc_left),
            AccumStateViewMut::from_contiguous_slice(slc_right),
        ]
    }
}
*/
