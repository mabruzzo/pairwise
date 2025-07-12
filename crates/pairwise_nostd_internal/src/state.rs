use ndarray::{ArrayView1, ArrayViewMut1, ArrayViewMut2, Axis};

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
    pub fn from_array_view(array_view: ArrayViewMut2<'a, f64>) -> Self {
        Self { data: array_view }
    }

    pub fn get_state(&self, i: usize) -> ArrayView1<f64> {
        self.data.index_axis(Axis(1), i)
    }

    pub fn get_state_mut(&mut self, i: usize) -> ArrayViewMut1<f64> {
        self.data.index_axis_mut(Axis(1), i)
    }

    pub fn state_size(&self) -> usize {
        self.data.len_of(Axis(0))
    }

    pub fn n_states(&self) -> usize {
        self.data.len_of(Axis(1))
    }

    // we probably want to add something like get_pair_disjoint_mut, which would
    // would be inspired by the slice type's more general get_disjoint_mut
    // method
}
