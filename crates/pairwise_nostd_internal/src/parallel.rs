//! Our parallelism abstractions use the concepts of teams & reductions
//!
//! In general, a parallel reduction is a way of characterizing a certain kind
//! of calculation that can be broken into parts, where each part can be
//! computed simultaneously, and each of the partial results can be combined
//! together into a single result.
//!
//! The idea of a "team" is an abstraction that we define in the context of
//! describing parallelism. When we decompose a parallel reduction into pieces
//! or "units of work":
//! - we can distributed the units of work among 1 or more teams
//! - a team is composed of 1 or more members. The members of a team work
//!   together in a tightly-coupled, synchronous manner to collaboratively
//!   complete a single task at a time.
//!
//! This abstraction nicely maps to hardware:
//! - on GPUs, you can imagine that each team-member corresponds to a separate
//!   thread in a single Warp (aka a WaveFront)
//! - on CPUs, it is optimal for a team to be represented by a single thread,
//!   and for each member to correspond to a lane use by SIMD instructions
//!
//! Of course we also have flexibility to adjust the definition of this
//! hardware mapping

use crate::accumulator::{Accumulator, Datum};
use crate::reduce_utils::reset_full_statepack;
use crate::state::StatePackViewMut;
use core::num::NonZeroU32;

/// Used to describes the properties of a team member.
///
/// The main reason purpose for this trait's existence is so that methods of
/// [`ReductionSpec`] can accept a type that implements this trait as an
/// argument and provide a specialization for:
/// - the case when the members of a team are represented by a full thread
/// - the case when the members of a team are vector-lanes driven by a single
///   thread.
///
/// # Note
/// If we are clever enough, I think we can probably get rid of this trait.
pub trait TeamMemberProp {
    const IS_VECTOR_PROCESSOR: bool;

    /// when SELF::IS_VECTOR_PROCESSOR is `true`, this will return 0
    fn get_id(&self) -> u32;
}

pub struct ThreadMember(u32);

impl ThreadMember {
    pub fn new(rank: u32) -> Self {
        ThreadMember(rank)
    }
}

impl TeamMemberProp for ThreadMember {
    const IS_VECTOR_PROCESSOR: bool = false;

    fn get_id(&self) -> u32 {
        self.0
    }
}

/// This struct holds standardized parameters that describe Team parallelism.
/// Different backends will obviously require extra parameters.
///
/// (This was implemented without a whole lot of thought: we may want to
/// reevaluate things, like exposing public members)
///
/// # Note
/// When we want to use SIMD vectorization on CPU, n_members_per_team holds the
/// number of vector lanes that are to be used in each operation
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

/// The basic idea here is to define an abstraction where a team is composed of
/// 1 or more members, and the members work together in a tightly-coupled,
/// synchronous manner to collaboratively complete a single task at a time.
/// (This is hardly a novel concept!)
///
/// The methods are all designed to be entered by all members of a team the
/// same time. Calls to these methods should be written *as if* there is a
/// barrier at the start of the method that will hang until all members catch
/// up. (Whether there is a barrier or not is an implementation detail)
///
/// # Design Consideration
/// For now, we have explicitly chosen **NOT** to provide methods that directly
/// expose a member id. The current approach for accessing the id (i.e. only
/// when closures are called from within methods), generally encourages a
/// design where we can write a serial implementation can support arbitrary
/// team sizes (and is capable of returning bitwise identical results to
/// parallel implementations). If this trait provided a method that directly
/// exposes member_id, then the majority of non-trivial code using that method
/// would be only be compatible with a serial implementation when the number of
/// threads per team is 1.
///
/// # Naming Considerations
/// We may to come up with better terminology. At this point, I'm comfortable
/// saying pretty definitively that we this trait can describe:
/// - a team of threads (fast on GPUs, used on CPUs for testing)
/// - a team of 1 thread that "simulates" the role of multiple members, for
///   testing purposes. (basically it goes through and does the work of 1
///   member at a time)
/// - a where each member corresponds to a SIMD vector lane (this is invoked
///   by a single thread at a time)
///
/// It might be beneficial to use terms like Lanes (in the context of vector
/// processors)
pub trait TeamProps {
    /// This is a TeamProp-specific type for protecting shared data (i.e. to
    /// prevent multiple members of a thread-team from accessing the data at
    /// the same time).
    ///
    /// To 0th order, you can just imagine that this is `std::sync::Mutex<T>`
    /// (or some equivalent on GPUs). In fact, that might be a good choice for
    /// an initial implementation of TeamProps that uses `std::thread`
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

    /// The choice of type indicates whether the members of a team correspond
    /// to individual threads OR the lanes driven by a single thread
    type MemberPropType: TeamMemberProp;

    /// Has the root member of the team execute `f`, which modifies `statepack`
    ///
    /// The other team members do nothing during this function call.
    fn exec_if_root_member(
        &mut self,
        statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        f: &impl Fn(&mut StatePackViewMut),
    );

    /// Triggers a barrier for all threads in the thread team, then this does
    /// 3 things:
    /// 1. team members collectively call the `get_member_contrib` closure.
    ///    - each member records the contributions from the call in a distinct
    ///      `accum_state` that is provided by `&mut self`.
    ///    - The `accum_state` variable is passed into the closure within a
    ///      [`StatePackViewMut`] type (since that is the type used to refer
    ///      to a collection of 1 or more accum_state).
    ///    - **IMPORTANTLY:** the memory associated with binned_statepacks is
    ///      left completely untouched.
    /// 2. Then, we combine the contributions from each member (in a "nested
    ///    reduction") so a single member holds the total contribution.
    /// 3. Finally, this single member use this total contribution to update
    ///    the `accum_state` stored at `bin_index` in `binned_statpack`
    ///
    /// # Note:
    /// I suspect that the memory barrier requires a memory fence
    fn calccontribs_combine_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        accum: &impl Accumulator,
        bin_index: usize,
        get_member_contrib: &impl Fn(&mut StatePackViewMut, Self::MemberPropType),
    );

    /// Triggers a barrier for all threads in the thread team. Then this does
    /// 3 things:
    /// 1. each team member calls the `get_datum_bin_pair` closure
    ///    - each member computes and records a [`BinnedDatum`] instance
    ///      to memory provided by `&mut self`.
    /// 2. Then, we gather the recorded [`BinnedDatum`] instances into
    ///    memory accessibly by one of the team members
    /// 3. finally, that team member sequentially uses the batch of
    ///    [`BinnedDatum`] instances to update `binned_statepack`
    fn collect_pairs_then_apply(
        &mut self,
        binned_statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        accum: &impl Accumulator,
        get_datum_bin_pair: &impl Fn(&mut [BinnedDatum], Self::MemberPropType),
    );
}

/// Used for specifying the details of a binned reduction.
///
/// <div class="warning">
///
/// **IMPORTANT**: I think this trait does too much and should probably be
/// broken up. I go into more detail at the very end of the docstring
///
/// </div>
///
/// At a high-level, types that implement this trait generally:
/// 1. encode details about a binned reduction
/// 2. have access to the data-source used in the binned reduction.
/// 3. know how to best decompose the overall binned reduction into 1 or more
///    smaller "units of work." We provide more detail momentarily (including a
///    definition for a "unit of work").
///
/// This trait essentially provides a standardized interface for external code
/// to actually carry out the reduction. (e.g. [`fill_single_team_statepack`])
///
/// # More detailed explanation
///
/// As we talk through the properties of this trait, it may be useful to look
/// at the implementation of [`fill_single_team_statepack`].
///
/// ## Broader Context
/// Before we proceed, it's important to make sure we are on the same page
/// about a few things.
///
/// ### What a is a binned reduction
/// Let's ensure that we are using consistent terminology to describe what
/// happens in a binned reduction:
/// - Types that implement this trait are implemented in terms of an
///   [`Accumulator`] instances.
/// - A simple reduction only involves 1 bin. In this scenario:
///   - we track a single accumulator state (a single accumulator state is
///     commonly called an `accum_state`)
///   - to carry out the reduction, we simply generate every relevant datum
///     (represented by [`Datum`]) from a data-source and use them to update
///     the `accum_state`
/// - This trait supports binned reductions. Consequently:
///   - a separate `accum_state` is tracked for each bin. We refer to this
///     collection of `accum_state`s as a `statepack`.
///   - when we generate each datum from the data source, we also generate a
///     bin index that specifies which `accum_state` will be updated that
///     datum. We sometimes represent this pair of information with the
///     [`BinnedDatum`] type.
///
/// ### How to we define a "unit of work?"
/// As we've just said the overall work of a binned reduction consists of
/// generating (datum, bin-index) pairs from the data source and using this
/// information to update the appropriate `accum_state`.
///
/// When we decompose this process, each "unit of work" consists of generating
/// a unique subsets of the (datum, bin-index) pairs and using them to update
/// the appropriate `accum_state`s.
///
/// ### Parallelism Considerations
/// It's also important to understand that this trait is usually used to
/// parallelize operations using our "team." abstraction. The number of teams
/// and the number of members per team are commonly encoded within instances
/// of [`StandardTeamParam`].
///
/// Let's provide some basic intuition:
/// - the simplest serial implementation of the algorithm is equivalent to
///   having 1 team composed of a single member
/// - if we totally ignore SIMD instructions for the moment, the most
///   efficient CPU threading implementation will generally have `n` threads
///   act operate as `n` single-member teams.
/// - members of a multi-member teams may map to separate threads (fast on
///   GPUs) or (on CPUs) separate lanes involved in SIMD operations of a single
///   thread
/// - we provide significantly more detail in other parts of this module
///
/// In more detail:
/// - each team is assigned multiple independent "units of work"
/// - For simplicity, it's convenient to think of separate teams as separate,
///   completely independent entities. We can combine the results of each
///   team at the very end of a reduction after every team is done.
///
/// **IMPORTANTLY:** all members of a given team work together on a
/// single "unit of work" at a time. They do this work in a synchronized,
/// lock-step manner. (So if a team is assigned 2 "units of work," the work
/// together until the first unit is completely done before they all move onto
/// the next unit). This is an important point that we will repeat.
///
/// ## Expressing the binned reduction details
///
/// The trait exposes information about the units of work that a given team is
/// responsible for completing with the [`outer_team_loop_bounds`] and
/// [`inner_team_loop_bounds`]. The following snippet illustrates how they are
/// intended to be used.
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
/// In the above snippet, the nested for-loop iterates over every "unit of
/// work" that the team is responsible for completing. Specifically, each
/// `(outer_idx, inner_idx)` pair maps to a distinct "unit of work."
/// (**DON'T FORGET:** the members of a given team work together to
/// complete a single unit of work at a time). The trait also provides methods
/// to actually complete the work associated with this pair. We'll discuss
/// those methods shortly.
///
/// Developer Note: I really want to eliminate the outer loop and delete
/// [`outer_team_loop_bounds`]. I also want to replace
/// [`inner_team_loop_bounds`] with a function that returns an iterator. The
/// docstrings for both functions highlight considerations for doing those
/// things.
///
/// ## Completing a "Unit of Work"
///
/// We now discuss the methods that can be used to complete a "unit of work."
/// Recall from earlier that we defined a "unit of work" as the task of
/// generating a series of (datum, bin-index) pairs from the data source and
/// using this information to update the appropriate `accum_state`s.
///
/// Types that implement this trait fall into 2 categories. This category is
/// specified by the [`ReductionSpec::NESTED_REDUCE`] associated constant and
/// is constrained by the nature of the underlying data source. As we note at
/// the very end of this docstring, this trait should probably be split off
/// into separate traits to reflect these categories
///
/// when [`reductionspec::nested_reduce`] is `true`, then:
/// - we know ahead of time that all datum generated in the "unit-of-work"
///   share a common bin index.
/// - the shared bin index is computed by [`infer_bin_index`].
/// - the type provides the [`collect_team_contrib`] method to collect
///   contributions for the team members.
/// - the intention is to:
///   1. Have the team collectively call [`collect_team_contrib`]. Each member
///      will record contributions to a separate `accum_state`
///   2. Then, these separate `accum_state` should be combined (in a "nested
///      reduction") so that a single member knows the total contribution
///      from the current unit of work.
///   3. Finally, that single member will use the total contribution to update
///      the accum_state from the appropriate statepack bin.
///
/// when [`reductionspec::nested_reduce`] is `false`, then:
/// - we don't know of any relationship between the data elements and the
///   associated bin indices in a "unit-of-work." This really restricts the
///   level of parallelism, we can achieve.
/// - The best we can do is adopt a "batching strategy," that has the
///   has the team members use [`get_datum_index_pair`] to pre-generate a
///   batch of (datum, bin-index) pairs from the data source. (the batch of
///   entries is to be sequentially processed)
/// - the intention is to:
///    1. have the team members collectively call [`get_datum_index_pair`].
///       Each member pre-generates a (datum, bin-index) pair from the data
///       source, represented as a [`BinnedDatum`] instance
///    2. Then the [`BinnedDatum`] instances should be gathered into a
///       collection-pad accessible in the memory of a single team member
///    3. have that team member process the batch of values in
///       [`BinnedDatum`] values to sequentially update the binned
///       statepack
///
/// # Development Note
/// An important premise is that you do error-checking while constructing
/// structs, and you design the logic such that you don't have to do **any**
/// error-handling in this trait's methods.
///
/// # Ideas for Improvement:
/// The use of `ReductionSpec::NESTED_REDUCE` to determine which methods should
/// be used is suboptimal. While I'm not in a huge rush to do this, this is
/// sign that we should break up this trait.
///
/// If we have a module `reduction`, I could imagine that we might define
/// traits with names like
/// - `reduction::Nested`
/// - `reduction::BinnedDataBatching`
///
/// where each trait is a superset of a third trait (maybe `ReductionInfo`?)
/// that declares common methods. (If we do this, then we would need to split
/// apart [`fill_single_team_statepack`] into separate functions)
pub trait ReductionSpec {
    // this trait's associated items are split into 2 parts.
    //   1. the associated supported by all trait implementers
    //   2. associated items conditionally supported by trait implementers
    // TODO: Figure out how to better express the conditionally implemented
    //       associated items (maybe use multiple traits?)

    // Associated items that are always used
    // -------------------------------------
    type AccumulatorType: Accumulator;

    /// return a reference to the accumulator
    fn get_accum(&self) -> &Self::AccumulatorType;

    /// The number of bins in this reduction.
    fn n_bins(&self) -> usize;

    // I would really like us to replace `outer_team_loop_bounds` and
    // `inner_team_loop_bounds` with a function that returns a single iterator
    // (See the `inner_team_loop_bounds` docstring for more details)

    /// Provides the bounds of the outer loop that all members of team share in
    /// a given call to the [`fill_single_team_statepack`] function
    ///
    /// For more details about how this is used, see the docstring of
    /// [`inner_team_loop_bounds`] or the implementation of
    /// [`fill_single_team_statepack`]
    ///
    /// # Note
    /// we will eventually be able to remove this method. It is totally
    /// unnecessary in the vast majority of cases. It **only** exists to make
    /// it easier to port over the [`apply_accum`] function for [`PointProps`]
    #[inline(always)]
    fn outer_team_loop_bounds(
        &self,
        _team_id: usize,
        _team_info: &StandardTeamParam,
    ) -> (usize, usize) {
        (0, 1)
    }

    /// Provides the bounds of the inner loop that all members of a team share
    /// in a given call to the [`fill_single_team_statepack`] function
    ///
    ///
    /// # Preference for iterators
    /// move to using iterators rather than this
    /// nested outer and inner loop. But, we may want to stick with the outer
    /// and inner loops until we have a minimum viable GPU implementation. This
    /// is mostly to avoid any codegen surprises.
    ///
    /// # Other Thoughts
    /// When the [`outer_team_loop_bounds`] function is removed, we should
    /// rename this function (maybe `team_loop_bounds`?)
    fn inner_team_loop_bounds(
        &self,
        outer_index: usize,
        team_id: usize,
        team_info: &StandardTeamParam,
    ) -> (usize, usize);

    // Associated items that are conditionally used
    // --------------------------------------------
    // this is bad practice and is something we should fix (maybe we define
    // separate traits). But I think we should wait until after we have
    // implemented a real parallel backend (e.g. multithreading) and
    // implemented the core algorithms

    const NESTED_REDUCE: bool;

    /// return the bin index shared by each datum that is generated from the
    /// data-source for the specified `(outer_index, inner_index)`
    ///
    /// **IMPORTANT:** DO NOT USE unless Self::NESTED_REDUCE holds `true`
    #[allow(unused)] // <- suppresses unused variable warnings
    fn infer_bin_index(
        &self,
        outer_index: usize,
        inner_index: usize,
        team_info: &StandardTeamParam,
    ) -> Option<usize> {
        // this **MUST** be overwritten when NESTED_REDUCE is true
        None
    }

    /// Intended to be called collectively by the members of a team so that
    /// each team member can compute a separate part of the accumulator
    /// state contribution with the `(outer_index, inner_index)` pair.
    ///
    /// Each datum considered within this calculation is associated with the
    /// bin index returned by [`infer_bin_index`].
    ///
    /// Each member stores its contribution in the `accum_state_buf` variable.
    /// Below, we discuss the precise meaning of this variable.
    ///
    /// **IMPORTANT:** DO NOT USE unless Self::NESTED_REDUCE holds `true`
    ///
    /// <div class="warning">
    ///
    /// `accum_states` has the type of `StatePackViewMut` purely because its
    /// the appropriate type for tracking 1 or more accumulator states. This is
    /// scratch space intended to be regularly overwritten. The memory used for
    /// tracking binned accumulator states is totally distinct
    ///
    /// </div>
    ///
    /// # Note
    /// This function should display slightly specialized behavior based on the
    /// value of [`TeamMemberProp::IS_VECTOR_PROCESSOR`]
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
    #[allow(unused)] // <- suppresses unused variable warnings
    fn compute_team_contrib<T: TeamMemberProp>(
        &self,
        accum_state_buf: &mut StatePackViewMut,
        outer_index: usize,
        inner_index: usize,
        member_prop: &T,
        team_param: &StandardTeamParam,
    ) {
        // this **MUST** be overwritten when NESTED_REDUCE is true
    }

    /// Intended to be called collectively by the members of a team so that
    /// each team member can determine the (datum, bin-index) pair that was
    /// reserved for it by the specified `(outer_index, inner_index)` pair.
    ///
    /// Each member records its (datum, bin-index) pair to `collect_pad`.
    /// Below, we discuss the precise meaning of this variable. If a member
    /// doesn't have an associated datum, it records a datum with a finite
    /// value and a weight of 0 (since it won't influence the reduction).
    ///
    /// **IMPORTANT:** DO NOT USE unless Self::NESTED_REDUCE holds `false`
    ///
    /// # Note
    /// This function should display slightly specialized behavior based on the
    /// value of [`TeamMemberProp::IS_VECTOR_PROCESSOR`]
    /// - when `false`: team members essentially correspond to threads. Each
    ///   thread in a given team will execute this function at the same time.
    ///   In this case, you can assume that `collect_pad` just holds space for
    ///   a single [`BinnedDatum`] instance.
    /// - when `true`: there is only a single thread in the team and the
    ///   number of members in the team correspond to the number of vector
    ///   lanes that the should be used in CPU SIMD instructions. In this case,
    ///   you can assume that the `collect_pad` argument has an entry for each
    ///   vector lane. The expectation is for each entry to be filled in a
    ///   single call to this function.
    ///
    /// # Implementation Note
    /// Currently, in a single collective call to this function, each team
    /// member only records a single [`BinnedDatum`] instance. In the
    /// future, we might consider recording multiple instances per member (it
    /// may be important for getting the most out of SIMD instructions).
    #[allow(unused)] // <- suppresses unused variable warnings
    fn get_datum_index_pair<T: TeamMemberProp>(
        &self,
        collect_pad: &mut [BinnedDatum],
        outer_index: usize,
        inner_index: usize,
        member_prop: &T,
        team_id: usize, // <- primarily used in the sample problem
        team_param: &StandardTeamParam,
    ) {
        // this **MUST** be overwritten when NESTED_REDUCE is true
    }
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
/// - During execution the members of the team proceed through the loop in
///   lock-step. The work done by each member **only** differs during calls to
///   `team`'s methods.
///
/// # How this fits into the broader picture?
/// This only worries about a single thread team's work. (TODO: ADD MORE)
///
/// [^serial_exception]: There is a minor exception when using a special Serial
///     implementation of a thread team that can work with with an arbitrary
///     number of teams and members per team
pub fn fill_single_team_statepack<T, R>(
    binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
    team: &mut T,
    team_id: usize,
    team_param: &StandardTeamParam,
    reduce_spec: &R,
) where
    T: TeamProps,
    R: ReductionSpec,
{
    // your mental picture should be that all threads in a given thread team
    // advance thru this loop together in a synchronized manner
    // - we will make it clear in the comment when the team-members behave
    //   differently from each other
    // - there is a bit of an exception with the way a "serial team" will

    let accum = reduce_spec.get_accum();

    // To begin, let's ensure that each `accum_state` within `statepack` is
    // consistent with having no contributions

    // TODO: consider distributing work among team members
    team.exec_if_root_member(binned_statepack, &|statepack: &mut StatePackViewMut| {
        reset_full_statepack(accum, statepack);
    });

    // now let's move to the crux of the work that is done by the thread team
    // - As we know, the overall task of performing a reduction can be broken
    //   into smaller units of work
    //
    // in the reduction. A ReductionSpec organizes
    let (outer_start, outer_stop) = reduce_spec.outer_team_loop_bounds(team_id, team_param);
    for outer_idx in outer_start..outer_stop {
        let (inner_start, inner_stop) =
            reduce_spec.inner_team_loop_bounds(outer_idx, team_id, team_param);
        for inner_idx in inner_start..inner_stop {
            // the if-statement depends on the properties of the underlying data source

            if R::NESTED_REDUCE {
                // in this case, each considered datum in the current unit of work
                // shares a common bin_index. That bin_index is specified by
                let bin_index = reduce_spec.infer_bin_index(outer_idx, inner_idx, team_param);

                // bin_index will be None if the data isn't relevat for the current task
                if let Some(bin_index) = bin_index {
                    // the next function call does 3 things
                    // 1. the team members call [`collect_team_contrib`]. They
                    //    each store the contributions from this call in a
                    //    distinct region of memory
                    //    - that memory is provided by `team` via the
                    //      `tmp_accum_states` argument
                    //    - to be completely clear, the memory referenced in
                    //      `tmp_accum_states` is COMPLETELY distinct from
                    //      `binned_statepack`
                    // 2. `team` performs a nested reduction to combine
                    //    together all the contributions from each of the
                    //    temporary variables into a total contribution
                    // 3. Finally, `team` has a single member use this total
                    //    contribution to update the `accum_state` stored at
                    //    `bin_index` in `binned_statpack`
                    team.calccontribs_combine_apply(
                        binned_statepack,
                        reduce_spec.get_accum(),
                        bin_index,
                        &|tmp_accum_states: &mut StatePackViewMut,
                          member_prop: T::MemberPropType| {
                            // we should make sure that reset_accum_state is called!!!!
                            reduce_spec.compute_team_contrib(
                                tmp_accum_states,
                                outer_idx,
                                inner_idx,
                                &member_prop,
                                team_param,
                            )
                        },
                    );
                }
            } else {
                // in this case, the bin-index order is unpredictable.
                // The best we can do is adopt a "batching strategy." We split
                // the "unit of work" into 2 parts. The first part is
                // parallelized by team-members & the second part is sequential
                // In the next function call:
                // 1. the team collectively calls `get_datum_index_pair`, where
                //    each member pre-generates a (datum, bin-index) pair from
                //    the data-source. The pair is represented as a
                //    `BinnedDatum` instance
                // 2. then, these `BinnedDatum` instances are gathered into a
                //    collection-pad that can be accessed by one of the team
                //    members
                // 3. finally, that team member sequentially uses the batch
                //    of `BinnedDatum` instances to update
                //    `binned_statepack`
                team.collect_pairs_then_apply(
                    binned_statepack,
                    reduce_spec.get_accum(),
                    &|collect_pad: &mut [BinnedDatum], member_prop: T::MemberPropType| {
                        // we should make sure that reset_accum_state is called!!!!
                        reduce_spec.get_datum_index_pair(
                            collect_pad,
                            outer_idx,
                            inner_idx,
                            &member_prop,
                            team_id,
                            team_param,
                        );
                    },
                );
            }
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
pub trait Executor {
    // I suspect that we may want to set up team_size & league_size elsewhere
    fn drive_reduce(
        &mut self,
        out: &mut StatePackViewMut,
        reduction_spec: &impl ReductionSpec,
        team_size: NonZeroU32,
        league_size: NonZeroU32,
    ) -> Result<(), &'static str>;
}
