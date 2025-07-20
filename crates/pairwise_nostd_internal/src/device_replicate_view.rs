//! The intent for this system of crate to be used to support analysis code in
//! a scripting language (namely Python), and the nature of the calculations
//! places some constraints on us:
//! 1. We want to avoid lots of heap allocations, so we design the crate to
//!    to work with view types (i.e. they are types with "pointer semantics"
//!    that represent a view into external allocations). This alone isn't
//!    significant (lots of rust code operates on slices!)
//! 2. Where possible we want to avoid placing onerous requirements on
//!    massaging data into a proper form. Accordingly, we design our interface
//!    around common data representations. For example, it is common for
//!    components of a vector field to be represented as distinct pointers.
//!
//! These constraints lead to scenarios where functions need to work with up
//! to 9 different buffers. To make the code easier to reason about, we define
//! structs, that act as "composite views" by bundling together the pointers.
//!
//! # Implications for GPUs
//!
//! This use of these "composite views" have important implications for GPU
//! calculations. It largely comes up with transferring data back and forth,
//! since you can't simply copy over an arbitrary pointer (actually, you can
//! with Unified Memory, but with significant overhead)
//!
//! Essentially, what we need to do is make remote clones of each pointer/slice
//! in a composite view and reconstruct the composite view in the kernel.
//!
//! While we could do this manually, there are a number of MAJOR disadvantages.
//! If this were just one-off situation, we would just deal with it. But, since
//! we do this a bunch of times, we need to come up with some abstractions
//!
//! <div class="warning">
//!
//! We are coloring slightly outside the lines by having
//! [`RemoteMemoryRegistry::reserve_dev_ptr_from_slice`] return a rust
//! [`pointer`] than somthing like `cust::memory::DevicePointer`.
//!
//! </div>
//!
//! # Other Thoughts
//! The abstractions in this optimal are probably suboptimal, but they are
//! probably "good enough," since GPU support is very experimental, and we are
//! taking a very CPU-first perspective.

#[cfg(feature = "cuda")]
pub use cust_core::DeviceCopy;

#[cfg(not(feature = "cuda"))]
pub trait DeviceCopy {}

/// Describes a type that is a "scoped memory registry"
///
/// The premise is that instances of types that implement this trait live for
/// some fixed lifetime `'a`. During the instance's lifetime, the instance can
/// reserve multiple read-only GPU buffers that are all constrained to have
/// the lifetime `'a`.
///
/// The pointer returned by [`reserve_dev_ptr_from_slice`] type needs to be
/// used with significant care. It would be exceptionally easy to trigger
/// undefined behavior. It only exists so that it can be within the context of
/// [`DeviceReplicateView::create_message_packet`].
pub trait RemoteMemoryRegistry<'a> {
    /// Allocates a buffer on the device with the size of `slice`, clones the
    /// contents of `slice` into that buffer, and returns a rust [`pointer`]
    /// to the type.
    ///
    /// The pointer this function returns must not be dereferenced on the CPU.
    ///
    /// # Safety
    /// The caller must ensure that `self` outlives the pointer this function
    /// returns. Otherwise, the pointer will end up dangling.
    ///
    /// The caller must also ensure that the memory that the pointer points to
    /// is **never** written to using the returned pointer or any pointer
    /// that is defined from it. This applies to all operations on the GPU.
    ///
    /// this is intended to **only** be used by
    /// [`DeviceReplicateView::create_message_packet`]
    fn reserve_dev_ptr_from_slice<T: DeviceCopy>(&mut self, slice: &[T]) -> *const T;
}

/// Types that implement this trait are generally read-only view types. This
/// trait describes the ability to replicate the underlying data on a GPU.
///
/// The basic idea is that this trait produces make a message packet type,
/// that holds the references to the duplicated data and is easy to copy. On
/// the GPU, the message packet type will be easily converted back to the type
/// that implements this trait
///
/// # Safety
/// This type generally requires a bunch of unsafe code. It would be nice to
/// make it possible to derive this type.
pub unsafe trait DeviceReplicateView {
    type MessagePacket<'x>: Copy + DeviceCopy;

    /// Allocates buffers on the GPU (managed by the `registry` argument) for
    /// all of the data viewed by `self`, copies over the data, and returns an
    /// instance of the lightweight message packet type that can be copied to
    /// the GPU and coerced back to an instance of `Self`.
    ///
    /// The lifetime of the returned packet type is linked to the lifetime of
    /// `registry`, since the packet type references memory managed by
    /// `registry`.
    ///
    /// # Lifetimes
    /// This is a little confusing since it makes use of a
    /// [generic associated type](https://rust-lang.github.io/rfcs/1598-generic_associated_types.html).
    ///
    /// The lifetime parameters are saying that:
    /// - this function is called by passing a shared reference with some
    ///   lifetime 'a to self.
    /// - it also takes a mutable reference, called `registry`, where the
    ///   reference has a lifetime 'b to a type that implements the
    ///   [`RemoteMemoryRegistry`] trait
    ///   - the `RemoteMemoryRegistry` trait itself has a lifetime 'r, which
    ///     describes how long the memory registry lives
    ///   - there is an implicit requirement that 'r outlives 'b
    ///   - we explicitly add a bound that 'r outlives 'a
    /// - the returned message packet has the lifetime 'r (the same lifetime
    ///   as `registry`)
    fn create_message_packet<'a, 'b, 'r>(
        &'a self,
        registry: &'b mut impl RemoteMemoryRegistry<'r>,
    ) -> Self::MessagePacket<'r>
    where
        // there is an implied bound that 'b outlives 'r
        'r: 'a; // 'registry outlives 'a
}
