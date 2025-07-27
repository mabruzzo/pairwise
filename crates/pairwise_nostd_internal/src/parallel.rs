//! This module defines parallelism abstractions. If you aren't familiar with
//! our high-level concepts, please go see the [developer guide](`super#parallelism-overview`)

use crate::reduce_utils::reset_full_statepack;
use crate::reducer::{Datum, Reducer};
use crate::state::StatePackViewMut;
use core::num::NonZeroU32;

pub struct MemberID(pub usize);

/// This struct holds standardized parameters that describe Team parallelism.
/// Different backends will obviously require extra parameters.
///
/// (This was implemented without a whole lot of thought: we may want to
/// reevaluate things, like exposing public members)
#[derive(Clone, Copy)]
pub struct StandardTeamParam {
    pub n_members_per_team: usize,
    pub n_teams: usize,
}

/// Used to hold a datum and its associated bin_index
///
/// This is only used when we need to pack this information into memory. In
/// most cases, track these in separate variables
///
/// I don't love that this defines copy, but it's important for examples
#[derive(Clone, Copy)]
pub struct BinnedDatum {
    pub bin_index: usize,
    pub datum: Datum,
}

impl BinnedDatum {
    pub fn zeroed() -> Self {
        BinnedDatum {
            bin_index: 0,
            datum: Datum::zeroed(),
        }
    }
}

/// A team (implementer of TeamProps) is composed of 1 or more members, who
/// work together in a tightly-coupled, synchronous manner to collaboratively
/// complete a single unit of work at a time. This trait can describe:
/// - a team of threads (fast on GPUs, used on CPUs for testing)
/// - a team of 1 thread that "simulates" the role of multiple members, for
///   testing purposes. (basically it goes through and does the work of 1
///   member at a time)
/// - a team where each member corresponds to a SIMD vector lane (invoked by a
///   single thread at a time)
///
/// The methods are all designed to be entered by all members of a team the
/// same time. Calls to these methods should be written *as if* there is a
/// barrier at the start of the method that will hang until all members catch
/// up. (Whether there is a barrier or not is an implementation detail)
///
/// # Design Consideration: why not expose member ids?
/// For now, we have explicitly chosen **NOT** to provide methods that directly
/// expose a member id (they are inevitably exposed through closures). This
/// encourages a design where we can write a serial implementation supporting
/// arbitrary team sizes (and is capable of returning bitwise identical results
/// to parallel implementations). If this trait provided a method that directly
/// exposes member_id, then the majority of non-trivial code using that method
/// would be only be compatible with a serial implementation when the number of
/// threads per team is 1.
pub trait TeamProps {
    const IS_VECTOR_PROCESSOR: bool;
    /// This is a TeamProp-specific type for protecting shared data (i.e. to
    /// prevent multiple members of a thread-team from accessing the data at
    /// the same time).
    ///
    /// `std::sync::Mutex<T>` might be a good choice for an initial
    /// implementation of TeamProps that uses `std::thread`.
    ///
    /// # Optimization Considerations
    /// In practice, use of a type like `std::sync::Mutex<T>` is suboptimal
    /// since it performs locking every time we want to access the mutex. We
    /// can do better since its tailored to the Thread Team implementation.
    /// - For example, for a serial implementation of a thread-team, you could
    ///   implement this as: `struct MyWrapper<T> {wrapped_data: T}`
    /// - You could do the same thing if the thread-team is designed so that
    ///   it **only** has a single thread per team.
    /// - Since (at the time of writing) all thread-team members only ever
    ///   access wrapped data using the team-member with an id of 0, you can
    ///   make a similar kind of optimizations in all implementations
    type SharedDataHandle<T>;

    fn standard_team_info(&self) -> StandardTeamParam;

    /// team_id satisfies `0 <= team_id < self.standard_team_info().n_teams`
    fn team_id(&self) -> usize;

    /// Has the root member of the team execute `f`, which modifies `statepack`
    ///
    /// The other team members do nothing during this function call.
    fn exec_once(
        &mut self,
        statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        f: &impl Fn(&mut StatePackViewMut),
    );

    /// Ensures all team members are synchronized, then does 3 things:
    /// 1. Team members collectively call the `get_member_contrib` closure.
    ///    Each member records the contributions from the call in a distinct
    ///    `accum_state` (NOT binned_statepack, which is untouched).
    ///    `accum_state` is passed into `get_member_contrib` within a
    ///    [`StatePackViewMut`] type.
    /// 2. Combines the contributions from each member (in a "nested
    ///    reduction") so a single member holds the total contribution.
    /// 3. This member updates `accum_state` stored at `bin_index` in
    ///    `binned_statpack`
    fn calccontribs_combine_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        reducer: &impl Reducer,
        bin_index: usize,
        get_member_contrib: &impl Fn(&mut StatePackViewMut, MemberID),
    );

    /// Ensures all team members are synchronized, then does 3 things:
    /// 1. each team member calls the `get_datum_bin_pair` closure. Each member
    ///    computes and records a [`BinnedDatum`] instance to memory provided by
    ///    `&mut self`.
    /// 2. gathers the recorded [`BinnedDatum`] instances into memory accessible
    ///    by one of the team members
    /// 3. that team member uses the batch of [`BinnedDatum`] instances to
    ///    sequentially update `binned_statepack`
    fn collect_pairs_then_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        reducer: &impl Reducer,
        get_datum_bin_pair: &impl Fn(&mut [BinnedDatum], MemberID),
    );
}

/// Used for specifying the details of a binned reduction, providing an
/// interface to for external code to carry it out (e.g.
/// [`fill_single_team_binned_statepack`]), potentially in parallel.
///
/// At a high-level, types that implement this trait generally:
///
/// 1. encode details about a _binned reduction_, wherein data elements are
///    partitioned into bins, with separate `accum_state`s for each bin. The
///    collection of `accum_state`s for each bin is called a `statepack`.
///    Executing a binned reduction consists of generating (datum, bin-index)
///    pairs from the data source and using each to update the appropriate
///    `accum_state`.
///
/// 2. have access to the data-source used in the binned reduction, from which
///    (datum, bin-index) pairs from the are drawn.
///
/// 3. know how to best decompose the overall binned reduction into
///    _units of work_ to be distriubted across the available teams. A unit of
///    work consists of generating and processing a unique subset of pairs,
///
/// # `outer_team_loop_bounds` and `inner_team_loop_bounds`
/// The trait exposes information about the units of work that a given team is
/// responsible for completing with the [`Self::outer_team_loop_bounds`] and
/// [`Self::inner_team_loop_bounds`]. The following snippet illustrates how
/// they are intended to be used.
///
/// In the context of this snippet, `team_param` holds a [`StandardTeamParam`]
/// instance. You should imagine that all of the members of the team with the
/// id of `team_id` are executing this snippet in lockstep with each other (and
/// the function calls produce exactly the same result for each member).
///
/// ```ignore
/// let (outer_start, outer_stop) = reduce_spec.outer_team_loop_bounds(team_id, &team_param);
/// for outer_idx in outer_start..outer_stop {
///     let (inner_start, inner_stop) =
///         reduce_spec.inner_team_loop_bounds(outer_idx, team_id, &team_param);
///     for inner_idx in inner_start..innser_stop {
///
///         // do work
///     }
/// }
/// ```
///
/// In the above snippet, each `(outer_idx, inner_idx)` pair corresponds to a
/// _unit of work_. (**DON'T FORGET:** the members of a given team work together to
/// complete a single unit of work at a time). The trait also provides methods
/// to actually complete the work associated with this pair.
/// those methods shortly.
///
/// The number of teams and the number of members per team are commonly encoded
/// within instances of [`StandardTeamParam`].
pub trait ReductionSpec {
    // An important premise is that you do error-checking while constructing
    // structs, and you design the logic such that you don't have to do **any**
    // error-handling in this trait's methods.

    type ReducerType: Reducer;

    /// return a reference to the reducer
    fn get_reducer(&self) -> &Self::ReducerType;

    /// The number of bins in this reduction.
    fn n_bins(&self) -> usize;

    // I would really like us to replace `outer_team_loop_bounds` and
    // `inner_team_loop_bounds` with a function that returns a single iterator
    // (See the `inner_team_loop_bounds` docstring for more details)

    /// Provides the bounds of the outer loop that all members of team share in
    /// a given call to the [`fill_single_team_binned_statepack`] function
    ///
    /// For more details about how this is used, see the docstring of
    /// [`Self::inner_team_loop_bounds`] or the implementation of
    /// [`fill_single_team_binned_statepack`]
    ///
    /// # Note
    /// we will eventually be able to remove this method. It is totally
    /// unnecessary in the vast majority of cases. It **only** exists to make
    /// it easier to port over the [`crate::apply_accum`] function for
    /// [`crate::PointProps`]
    #[inline(always)]
    fn outer_team_loop_bounds(
        &self,
        _team_id: usize,
        _team_info: &StandardTeamParam,
    ) -> (usize, usize) {
        (0, 1)
    }

    /// Provides the bounds of the inner loop that all members of a team share
    /// in a given call to the [`fill_single_team_binned_statepack`] function
    ///
    /// # Preference for iterators
    /// move to using iterators rather than this
    /// nested outer and inner loop. But, we may want to stick with the outer
    /// and inner loops until we have a minimum viable GPU implementation. This
    /// is mostly to avoid any codegen surprises.
    ///
    /// # Other Thoughts
    /// When the [`Self::outer_team_loop_bounds`] function is removed, we
    /// should rename this function (maybe `team_loop_bounds`?)
    fn inner_team_loop_bounds(
        &self,
        outer_index: usize,
        team_id: usize,
        team_info: &StandardTeamParam,
    ) -> (usize, usize);

    /// The associated constant indicates whether this is a "Nested Reduction"
    /// or "Batched Reduction." These concepts are defined in
    /// [`super#kinds-of-parallel-binned-reductions`].
    ///
    /// More concretely, the value of this program is a promise about the
    /// [`TeamProps`] method used by [`Self::add_contributions`]:
    /// - when `true`, the type implementing trait promises that
    ///   [`Self::add_contributions`] will call
    ///   [`TeamProps::calccontribs_combine_apply`] (importantly, it won't
    ///   call [`TeamProps::collect_pairs_then_apply`]).
    /// - when `false`, the type implementing trait promises that
    ///   [`Self::add_contributions`] will call
    ///   [`TeamProps::collect_pairs_then_apply`] (importantly, it won't call
    ///   [`TeamProps::calccontribs_combine_apply`])
    ///
    /// This promise gives us the ability to control how we allocate memory.
    /// Violating this promise produces undefined behavior.
    ///
    /// # Note
    /// I have a plan involving some minimal refactoring that will encode this
    /// promise as a compile-time invariant. But, it's unclear that it's worth
    /// the effort so we defer it, for now.
    const NESTED_REDUCE: bool;

    /// Intended to be called collectively by the members of a team can
    /// collaboratively update the `binned_statepack` with the contributions
    /// associated with the `(outer_index, inner_index)` pair.
    ///
    /// See [`Self::NESTED_REDUCE`] for limitations on the implementation.
    ///
    /// # Note
    /// This function should display slightly specialized behavior based on the
    /// value of [`TeamProps::IS_VECTOR_PROCESSOR`]
    /// - when `false`: team members essentially correspond to threads. Each
    ///   thread in a given team will execute this function at the same time.
    ///   In this case, you can assume that `accum_state_buf` holds just a
    ///   single `accum_state`
    /// - when `true`: there is only a single thread in the team and the
    ///   number of members in the team correspond to the number of vector
    ///   lanes that the should be used in CPU SIMD instructions. In this
    ///   case, you can assume that the `accum_state_buf` argument has an entry
    ///   for each vector lane. Each entry is expected to be filled with
    ///   contributions during a single call to this function.
    fn add_contributions<T: TeamProps>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        outer_index: usize,
        inner_index: usize,
        team: &mut T,
    );
}

/// Initialize and fill a single Thread Team's `statepack`.
///
/// # How to use this function
/// This function should generally[^serial_exception] be used as follows:
/// - Calling the functions:
///   - all team members of execute this function at the same time:
///   - arguments:
///     - `binned_statepack`: specifies the `StatePackViewMut` that the team
///       will initialize and update
///     - `team`: will be tailored to the identity of each team member.
///     - `team_param` & `reduce_spec` are **exactly** the same for each member
/// - During execution, the members of the team proceed through the loop in
///   lock-step. The work done by each member **only** differs during calls to
///   `team`'s methods.
///
/// # How this fits into the broader picture?
/// This only worries about a single team's work. (TODO: ADD MORE)
///
/// [^serial_exception]: There is a minor exception when using a special Serial
///     implementation of a thread team that can work with with an arbitrary
///     number of teams and members per team
pub fn fill_single_team_binned_statepack<T>(
    binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
    team: &mut T,
    reduce_spec: &impl ReductionSpec,
) where
    T: TeamProps,
{
    let reducer = reduce_spec.get_reducer();
    let team_param = team.standard_team_info();
    let team_id = team.team_id();

    // TODO: consider distributing work among team members
    team.exec_once(binned_statepack, &|statepack: &mut StatePackViewMut| {
        reset_full_statepack(reducer, statepack);
    });

    let (outer_start, outer_stop) = reduce_spec.outer_team_loop_bounds(team_id, &team_param);
    for outer_idx in outer_start..outer_stop {
        let (inner_start, inner_stop) =
            reduce_spec.inner_team_loop_bounds(outer_idx, team_id, &team_param);
        for inner_idx in inner_start..inner_stop {
            reduce_spec.add_contributions(binned_statepack, outer_idx, inner_idx, team);
        }
    }
}

/// a trait for expressing how to launch a reduction
///
/// This is mostly intended as a placeholder. The idea is that we would
/// implement something like this for each parallelism "backend," so we have a
/// uniform interface for easily switching between backends.
///
/// **NOTE:** Types that implement this trait are intended to live entirely on
/// the CPU. A GPU backend would provide a type that implements this crate in
/// order to execute the CPU calls that are needed for managing memory and
/// launching GPU calculations
///
/// This isn't a method of ReduceCommon, because it may be helpful to separate
/// out some of the GPU interop stuff.
pub trait Executor {
    // I suspect that we may want to set up team_size & league_size elsewhere
    fn drive_reduce(
        &mut self,
        out: &mut StatePackViewMut,
        reduction_spec: &impl ReductionSpec,
        n_members_per_team: NonZeroU32,
        n_teams: NonZeroU32,
    ) -> Result<(), &'static str>;
}
