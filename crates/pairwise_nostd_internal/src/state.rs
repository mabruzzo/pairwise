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

use core::{
    f64,
    ops::{Index, IndexMut},
};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

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
    data: ArrayView1<'a, f64>,
}

impl<'a> AccumStateView<'a> {
    // todo: remove this method before a release!
    pub fn from_array_view(array_view: ArrayView1<'a, f64>) -> Self {
        Self { data: array_view }
    }

    pub fn from_contiguous_slice(state: &'a [f64]) -> Self {
        Self {
            data: ArrayView1::from_shape([state.len()], state).unwrap(),
        }
    }

    // todo: remove this method before a release!
    pub fn as_array_view(&self) -> ArrayView1<f64> {
        self.data.view()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> Index<usize> for AccumStateView<'a> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
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
    data: ArrayViewMut1<'a, f64>,
}

impl<'a> AccumStateViewMut<'a> {
    // todo: remove this method before a release!
    pub fn from_array_view(array_view: ArrayViewMut1<'a, f64>) -> Self {
        Self { data: array_view }
    }

    pub fn from_contiguous_slice(state: &'a mut [f64]) -> Self {
        Self {
            data: ArrayViewMut1::from_shape([state.len()], state).unwrap(),
        }
    }

    pub fn as_view<'b>(&'b self) -> AccumStateView<'b> {
        AccumStateView {
            data: self.data.view(),
        }
    }

    // todo: consider whether this should actually be a public method shipped
    // in a release
    pub fn fill(&mut self, val: f64) {
        self.data.fill(val);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> Index<usize> for AccumStateViewMut<'a> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl<'a> IndexMut<usize> for AccumStateViewMut<'a> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

/// Represents a collection of Accumulator States
///
/// # Note
/// There is some benefit to defining this even though it wraps ArrayViewMut2
/// since it helps contain all references to the ndarray package to a single
/// file.
pub struct StatePackViewMut<'a> {
    // when we refactor this to stop wrapping ArrayViewMut2, I *think*, we
    // probably want to wrap a pointer rather than a slice.
    //
    // If we use a pointer, we should consider whether we need to make use of
    // core::marker::PhantomData. I don't *think* it would necessary for this
    // scenario, but we should review
    // https://doc.rust-lang.org/nomicon/phantom-data.html
    data: ArrayViewMut2<'a, f64>,
}

impl<'a> StatePackViewMut<'a> {
    // todo: remove this method before our release
    pub fn from_array_view(array_view: ArrayViewMut2<'a, f64>) -> Self {
        Self { data: array_view }
    }

    pub fn from_slice(n_state: usize, state_size: usize, xs: &'a mut [f64]) -> Self {
        Self {
            data: ArrayViewMut2::from_shape([state_size, n_state], xs).unwrap(),
        }
    }

    // todo: remove this method before our release
    pub fn as_array_view(&self) -> ArrayView2<f64> {
        self.data.view()
    }

    pub fn as_array_view_mut(&mut self) -> ArrayViewMut2<f64> {
        self.data.view_mut()
    }

    #[inline]
    pub fn get_state(&self, i: usize) -> AccumStateView {
        AccumStateView::from_array_view(self.data.index_axis(Axis(1), i))
    }

    #[inline]
    pub fn get_state_mut(&mut self, i: usize) -> AccumStateViewMut {
        AccumStateViewMut::from_array_view(self.data.index_axis_mut(Axis(1), i))
    }

    pub fn state_size(&self) -> usize {
        self.data.len_of(Axis(0))
    }

    pub fn n_states(&self) -> usize {
        self.data.len_of(Axis(1))
    }

    pub fn total_size(&self) -> usize {
        self.state_size() * self.n_states()
    }

    // we probably want to add something like get_pair_disjoint_mut, which would
    // would be inspired by the slice type's more general get_disjoint_mut
    // method
}
