//! Our parallelism abstractions use the concepts of thread teams & reductions

use crate::accumulator::{Accumulator, DataElement};
use crate::state::{AccumStateViewMut, StatePackViewMut};
use core::num::NonZeroU32;

pub struct MemberId(u32);

impl MemberId {
    pub fn new(rank: u32) -> Self {
        MemberId(rank)
    }

    pub fn get(&self) -> u32 {
        self.0
    }
}

/// This struct holds standardized parameters that describe Thread Team
/// parallelism. Different backends will obviously require extra parameters.
///
/// (This was implemented without a whole lot of thought: we may want to
/// reevaluate things, like exposing public members)
pub struct StandardTeamParam {
    pub n_members_per_team: usize,
    pub n_teams: usize,
}

/// Used to hold a data element and its assoicated bin_index
///
/// This is only used when we need to pack this information into memory. In
/// most cases, track these in separate variables
pub struct BinnedDataElement {
    pub bin_index: usize,
    pub datum: DataElement,
}

impl BinnedDataElement {
    pub fn zeroed() -> Self {
        BinnedDataElement {
            bin_index: 0,
            datum: DataElement::zeroed(),
        }
    }
}

/// the goal here is to use a parallelism strategy with thread-teams. This is
/// hardly a novel concept. The actual implementation will be tailored to the
/// "backend" that we are using (e.g. serial, multi-threading, GPU)
///
/// The methods are all designed to be entered by all members of a team the
/// same time. Calls to these methods should be written *as if* there is a
/// barrier at the start of the method that will hang until all members catch
/// up. (Whether there is a barrier or not is an implementation detail)
///
/// # Design Consideration
/// For now, we have explicitly chosen **NOT** to provide methods that directly
/// expose a member id. The current approach for accessing team_rank (i.e. only
/// when closures are called from within methods), generally encourages a
/// design where we can write a serial implementation can support arbitrary
/// team sizes (and is capable of returning bitwise identical results to
/// parallel implementations). If this trait provided a method that directly
/// exposes member_id, then the majority of non-trivial code using that method
/// would be only be compatible with a serial implementation when the number of
/// threads per team is 1.
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
    /// 1. each team member calls the `get_member_contrib` closure. Each call
    ///    store the contributions from this call in a separate temporary
    ///    variable (provided by `&mut self`).
    /// 2. then all team members participate in a local reduction to combine
    ///    together all the contributions from each of the temporary variables
    /// 3. Finally, `team` has a single member use this total contribution to
    ///    update the `accum_state` stored at `bin_index` in `statpack`
    ///
    /// # Note:
    /// I suspect that the memory barrier requires a memory fence
    fn calccontribs_combine_apply(
        &mut self,
        statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        accum: &impl Accumulator,
        bin_index: usize,
        get_member_contrib: &impl Fn(&mut AccumStateViewMut, MemberId),
    );

    /// Triggers a barrier for all threads in the thread team. Then this does
    /// 3 things:
    /// 1. each team member calls the `get_element_bin_pair` closure, which
    ///    gets a data-element and its associated bin index, returned in a
    ///    [`BinnedDataElement`]
    /// 2. we gather [`BinnedDataElement`] instance into the memory of a single
    ///    team member
    /// 3. we have that team member sequentially use the information in
    ///    each `BinnedDataElement` to update the statepack
    fn getelements_gather_apply(
        &mut self,
        statepack: &mut Self::SharedDataHandle<StatePackViewMut>,
        accum: &impl Accumulator,
        get_element_bin_pair: &impl Fn(MemberId) -> BinnedDataElement,
    );
}

/// Used for specifying the details of a binned reduction.
///
/// At a high-level, types that implement this trait generally:
/// 1. encode details about a binned reduction
/// 2. have access to the data-source used in the binned reduction.
/// 3. know how to best decompose the overall binned reduction into 1 or more
///    smaller "units of work." We provide more detail momentary (including a
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
///   - to carry out the reduction, we simply generate every relevant data
///     element (represented by [`DataElement`]) from a data-source and use
///     them to update the `accum_state`
/// - This trait supports binned reductions. Consequently:
///   - a separate `accum_state` is tracked for each bin. We refer to this
///     collection of `accum_state`s as a `statepack`.
///   - when we generate each data element from the data source, we also
///     generate a bin index that specifies which `accum_state` will be updated
///     by the `accum_state`. We sometimes represent this pair of information
///     with the [`BinnedDataElement`] type.
///
/// ### How to we define a "unit of work?"
/// As we've just said the overall work of a binned reduction consists of
/// generating data-elements and their associated bin index from the data
/// source and using this information to update the appropriate `accum_state`.
///
/// When we decompose this process, each "unit of work" consists of generating
/// a unique subsets of the data-elements (and their associated bin indices)
/// and using them to update the appropriate `accum_state`s.
///
/// ### Parallelism Considerations
/// It's also important to understand that this trait is used in the context of
/// thread teams. The number of teams and the number of members per team are
/// commonly encoded within instances of [`StandardTeamParam`].
/// - the simplest serial implementation of the algorithm is equivalent to
///   having 1 team of 1 member
/// - most CPU threading implementations will be fastest with 1-member thread
///   teams.
///
/// In more detail:
/// - each thread team is assigned multiple independent "units of work"
/// - For simplicity, it's convenient to think of each thread team as a
///   separate, completely independent entity. We can combine the results of
///   each team at the very end of a reduction after every team is done.
///
/// **IMPORTANTLY:** all members of a given thread team work together on a
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
/// id of `team_id` are executing in lockstep with each other.
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
/// work" that the team, is responsible for completing. Specifically, each
/// `(outer_idx, inner_idx)` pair maps to a distinct "unit of work."
/// (**DON'T FORGET:** the members of a given team work together to
/// complete a single unit of work at a time). The trait provides methods to
/// actually complete the work associated with this pair. We'll discuss those
/// shortly.
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
/// generating a series of data-elements and their associated bin indices from
/// the data source and using this information to update the appropriate
/// `accum_state`s.
///
/// Types that implement this trait fall into 2 categories. This category is
/// specified by the [`ReductionSpec::NESTED_REDUCE`] associated constant and
/// is constrained by the nature of the underlying data source.
///
/// when [`reductionspec::nested_reduce`] is `true`, then:
/// - we know ahead of time that all data elements in a "unit-of-work" share
///   a common bin index.
/// - that shared bin index is computed by [`infer_distance_bin`].
/// - use the [`collect_team_contrib`] method to collect contributes for a
///   given team member in this scenario
/// - the intention is to:
///   1. have each team member call [`collect_team_contrib`]. Afterward, each
///      member will have store contributions from the current unit of work in
///      a local `accum_state` value.
///   2. Then, we combine the contributions from each member (in a "nested
///      reduction") so that the member with an id of 0 will have the total
///      contributions from the current unit in their own local variable.
///   3. Finally, the member with a member_id of 0, will use the total
///      contribution to update the appropriate accum_state from statepack.
///
/// when [`reductionspec::nested_reduce`] is `false`, then:
/// - we don't know of any relationship between the data-elements and the
///   associated bin indices in a "unit-of-work." This really restricts the
///   level of parallelism, so we simply do the best we possibly can do
/// - thus we use the [`get_datum_index_pair`] method to simply have each
///   team member compute a data-element from the data source and pre-compute
///   the associated bin-index
/// - the intention is to:
///    1. have each team member call [`get_datum_index_pair`], which
///       gets a data-element and its associated bin index, returned in a
///       a [`BinnedDataElement`] instance
///    2. gather [`BinnedDataElement`] instance into the memory of a single
///       team member
///    3. have that team member sequentially use the information in each
///       `BinnedDataElement` to update the statepack
///
/// # Development Note
/// An important premise is that you do error-checking while constructing
/// structs, and you design the logic such that you don't have to do **any**
/// error-handling in this trait's methods.
pub trait ReductionSpec {
    // this trait's associated items are split into 2 parts.
    //   1. the associated supported by all trait implementors
    //   2. associated items conditionally supported by trait implementors
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

    /// return the bin index that all data elements share for
    /// `(outer_index, inner_index)`.
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

    /// Collects all contributions to `accum_state` from the data elements
    /// reserved for the team member with id given by `member_id` and the
    /// `(outer_index, inner_index)` pair. This assumes that all data
    /// elements are associated with the index returned by [`infer_bin_index`]
    ///
    /// **IMPORTANT:** DO NOT USE unless Self::NESTED_REDUCE holds `true`
    #[allow(unused)] // <- suppresses unused variable warnings
    fn collect_team_contrib(
        &self,
        accum_state: &mut AccumStateViewMut,
        outer_index: usize,
        inner_index: usize,
        member_id: &MemberId,
        team_param: &StandardTeamParam,
    ) {
        // this **MUST** be overwritten when NESTED_REDUCE is true
    }

    /// Computes data element and its associated bin index reserved for the
    /// team member with id given by `member_id` and the
    /// `(outer_index, inner_index)` pair
    ///
    /// **IMPORTANT:** DO NOT USE unless Self::NESTED_REDUCE holds `false`
    ///
    /// # Note
    /// To support SIMD, we might want to consider making it possible to
    /// compute multiple `BinnedDataElement` instances at once (in this case,
    /// we should probably pass in a buffer that gets modified)
    #[allow(unused)] // <- suppresses unused variable warnings
    fn get_datum_index_pair(
        &self,
        outer_index: usize,
        inner_index: usize,
        member_id: MemberId,
        team_param: &StandardTeamParam,
    ) -> BinnedDataElement {
        // this **MUST** be overwritten when NESTED_REDUCE is false
        BinnedDataElement::zeroed()
    }
}

/// Initialize and fill a single Thread Team's `statepack`.
///
/// # How to use this function
/// This function should generally[^serial_exception] be used as follows:
/// - Calling the functions:
///   - all team members of execute this function at the same time:
///   - arguments:
///     - `statepack`: specifies the `StatePackViewMut` that the team will
///       initialize and update
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
    statepack: &mut T::SharedDataHandle<StatePackViewMut>,
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
    team.exec_if_root_member(statepack, &|statepack: &mut StatePackViewMut| {
        for i in 0..statepack.n_states() {
            accum.reset_accum_state(&mut statepack.get_state_mut(i));
        }
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
                // in this case, each considered data-element in the current unit of work
                // shares a common bin_index. That bin_index is specified by
                let bin_index = reduce_spec.infer_bin_index(outer_idx, inner_idx, team_param);

                // bin_index will be None if the data isn't relevat for the current task
                if let Some(bin_index) = bin_index {
                    // the next function call does 3 things
                    // 1. each team member calls [`collect_team_contrib`]. They
                    //    will each store the contributions from this call in a
                    //    separate temporary variable (provided by `team`).
                    // 2. `team` performs a nested reduction to combine
                    //    together all the contributions from each of the
                    //    temporary variables into a total contribution
                    // 3. Finally, `team` has a single member use this total
                    //    contribution to update the `accum_state` stored at
                    //    `bin_index` in `statpack`
                    team.calccontribs_combine_apply(
                        statepack,
                        reduce_spec.get_accum(),
                        bin_index,
                        &|accum_state: &mut AccumStateViewMut, member_id: MemberId| {
                            // we should make sure that reset_accum_state is called!!!!
                            reduce_spec.collect_team_contrib(
                                accum_state,
                                outer_idx,
                                inner_idx,
                                &member_id,
                                team_param,
                            )
                        },
                    );
                }
            } else {
                // in this case, the bin-index order is unpredictable.
                // the next function effectively:
                // 1. has each team member call `get_datum_index_pair`, which
                //    gets a data-element and its associated bin index,
                //    returned in a a `BinnedDataElement` instance
                // 2. gathers `BinnedDataElement` instances into the memory of
                //    a single team member
                // 3. has that team member sequentially use the information in
                //    each `BinnedDataElement` to update the statepack
                team.getelements_gather_apply(
                    statepack,
                    reduce_spec.get_accum(),
                    &|member_id: MemberId| -> BinnedDataElement {
                        // we should make sure that reset_accum_state is called!!!!
                        reduce_spec
                            .get_datum_index_pair(outer_idx, inner_idx, member_id, team_param)
                    },
                );
            }
        }
    }
}

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
