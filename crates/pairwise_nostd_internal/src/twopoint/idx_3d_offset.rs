//! This module defines machinery that primarily is used internally for
//! calculating two-point statistics when using [`CartesianBlock`] instances.
//!
//! We probably want to expose a narrow subset of this functionality to help
//! with "tiling patterns" when people are using MPI and/or are considering
//! periodic boundaries (in practice, we may expose this logic through a
//! separate construct and just reuse some of the logic).

use crate::misc::View3DSpec;
use crate::twopoint::spatial::{CartesianBlock, CellWidth};
use core::cmp;

#[cfg(test)] // disable outside of testing, for now
use std::iter::IntoIterator;

/// Describes the offset between 3D indices.
///
/// This type is primarily used for describing the relationship between a pair
/// of 3D indices. Consider a pair of 3D indices, `idx_a` and `idx_b`. At a
/// conceptual level, an [`Idx3DOffset`] instance describes the pair when:
/// * the instance can be "added" to `idx_a` to get `idx_b`
/// * the "negated" instance can be "added" to `idx_b` to get `idx_a`
///
/// This idea of "adding" and "negating" (i.e. multiply by -1), is mostly
/// conceptual. In practice, we don't directly implement code operations.
///
/// To elaborate further, we are free to use an [`Idx3DOffset`] instance to
/// describe either:
/// - the offset between 2 indices that correspond to separate arrays, (e.g.
///   `idx_A` comes `array A` while `idx_B` comes from ` array B`)
/// - OR the offset between 2 indices that correspond to the same arrays
///
/// # Motivation (Why does it exist?)
/// This type comes in the context of enumerating all unique pairs in a single
/// array or between separate 3D arrays. There are many ways to do that, but
/// in some cases, it can be computationally convenient to subdivide the pairs
/// where all pairs in a given group are described by a common [`Idx3DOffset`]
/// instance.
///
/// # Significance when used with [`CartesianBlock`]s
/// A 3D index in the context of a [`CartesianBlock`], corresponds to a
/// spatial location. In this context, an [`Idx3DOffset`] corresponds to the
/// spatial separation (or displacement) between the 2 points. This can be
/// important when you only want to consider pairs of points where the
/// displacement satisfies some special conditions
#[derive(Clone)]
#[cfg_attr(test, derive(Debug, Hash, Eq, PartialEq))] // only derived for testing
pub(crate) struct Idx3DOffset([isize; 3]);

impl Idx3DOffset {
    /// returns the underlying value (the signed index offset along each axis)
    #[inline]
    pub(crate) fn value(&self) -> &[isize; 3] {
        &self.0
    }

    /// computes the "displacement vector" in units of indices for a pair
    /// where one member comes from `block_a` and the other comes from
    /// `block_b`.
    ///
    /// This method primarily exists for self-documenting purposes
    pub(crate) fn displacement_idx_units(
        &self,
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) -> [isize; 3] {
        let mut out = [0_isize; 3];
        #[allow(clippy::needless_range_loop)] // <- todo get rid of me
        for i in 0..3 {
            let off_a = block_a.start_idx_global_offset[i];
            let off_b = block_b.start_idx_global_offset[i];
            out[i] = (off_b - off_a) + self.0[i];
        }
        out
    }

    /// calculates squared distance represented by the offset
    pub(crate) fn distance_squared(
        &self,
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
        cell_widths: &CellWidth,
    ) -> f64 {
        let d_index_units = self.displacement_idx_units(block_a, block_b);
        let mut out = 0.0;
        #[allow(clippy::needless_range_loop)] // <- todo get rid of me
        for i in 0..3 {
            let comp = (d_index_units[i] as f64) * cell_widths.widths_zyx[i];
            out += comp * comp;
        }
        out
    }
}

/// computes the triple for-loop bounds for enumerating all indices in
/// `block_a` that are part of measurement pairs described by `index_offset`
pub(crate) fn get_block_a_start_stop_indices(
    index_offset: &Idx3DOffset,
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
) -> ([isize; 3], [isize; 3]) {
    let mut idx_a_start = [0_isize; 3];
    let mut idx_a_stop = [0_isize; 3];
    for i in 0..3 {
        // start is abs(index_offset[i]) if index_offset[i] < 0. Otherwise, it's 0
        let start = cmp::max(-index_offset.0[i], 0_isize);
        // compute the number of elements along axis (we can definitely make
        // this more concise -- but we should be very clear what it means)
        let n_elem: isize = if index_offset.0[i] < 0 {
            cmp::min(
                block_a.idx_spec.shape()[i] - start,
                block_b.idx_spec.shape()[i],
            )
        } else {
            cmp::min(
                block_a.idx_spec.shape()[i],
                block_b.idx_spec.shape()[i] - index_offset.0[i],
            )
        };
        idx_a_start[i] = start;
        idx_a_stop[i] = start + n_elem;
    }

    (idx_a_start, idx_a_stop)
}

/// a helper type that encodes constraints on the family of [`Idx3DOffset`]
/// instances that describe pairs of measurements, where separate measurements
/// are drawn from distinct [`CartesianBlocks`].
///
/// An [`Idx3DOffset`] instance, `idxoff`, satisfies bounds if, for each axis
/// `i`, the value `idxoff.value()[i]` lies in the interval
/// `[start_offsets_zyx[i], stop_offsets_zyx[i])`.
///
/// When considering 2 [`CartesianBlock`]s, block_a and block_b, the family
/// then, for each `i`:
/// - `stop_offsets_zyx[i] = block_b.idx_spec.shape()[i]` and
/// - `start_offsets_zyx[i] = -(block_a.idx_spec.shape()[i] - 1)`.
#[derive(Clone)]
struct Bounds {
    start_offsets_zyx: [isize; 3], // inclusive
    stop_offsets_zyx: [isize; 3],  // exclusive
}

/// store the internal representation of the next [`Idx3DOffset`] instance
///
/// This is used in the context of iterating over [`Idx3DOffset`] instances.
/// `offset_zyx` holds the internals of the current [`Idx3DOffset`] instance.
/// This function mutates the variable's contents, in place, so that it holds
/// the internals of the next [`Idx3DOffset`] instance that the iterator
/// visits.
///
/// If `offset_zyx` is equal to
/// ```text
/// [
///     bounds.stop_offsets_zyx[0] - 1,
///     bounds.stop_offsets_zyx[1] - 1,
///     bounds.stop_offsets_zyx[2] - 1,
/// ]
/// ```
/// then the resulting value of `offset_zyx` won't correspond to a valid
/// [`Idx3DOffset`] instance.
fn update_to_next_offset(offset_zyx: &mut [isize; 3], bounds: &Bounds) {
    offset_zyx[2] += 1;
    if offset_zyx[2] == bounds.stop_offsets_zyx[2] {
        offset_zyx[2] = bounds.start_offsets_zyx[2];
        offset_zyx[1] += 1;
        if offset_zyx[1] == bounds.stop_offsets_zyx[1] {
            offset_zyx[1] = bounds.start_offsets_zyx[1];
            offset_zyx[0] += 1;
        }
    }
}

/// iterates through a sequence [`Idx3DOffset`] instances
///
/// # Larger Significance
/// In practice this is somewhat redundant with [`Idx3DOffsetSeq`]. At this
/// time, most code prefers to use [`Idx3DOffsetSeq`], which provides "random
/// access" to the sequence of [`Idx3DOffset`] instances. In more detail,
/// * external code currently uses [`Idx3DOffsetSeq`] because
///   1. conceptually, a sequence is somewhat simpler than an iterator.
///   2. partitioning elements in a sequence (for the sake of parallelism) is
///      a simpler than partitioning elements in an iterator.
///   3. I have some concerns about how optimal GPU code generation would be
///      when using iterators (this probably doesn't matter much if we aren't
///      using the iterator in a very tight loop)
///
/// * this type exists for historical reasons (early prototypes used logic
///   that resembled an iterator, not a sequence).
///
/// * we are holding onto this logic, for the moment, since using an iterator
///   to visit a sequence of [`Idx3DOffset`] instances is probably faster (at
///   least on CPUs) than passing a sequence of indices to [`Idx3DOffsetSeq`].
///   At this time, it's unclear whether the performance difference is
///   significant enough to merit the extra complexity.
///
/// # Implementation Detail
/// At the moment, the iterator tracks the internal representation of
/// [`Idx3DOffset`] instances. But, it may be conceptually simpler to create
/// an iterator over 3D indices in a [`View3DSpec`] instance and map each
/// index to a [`Idx3DOffset`] instance
///
/// # Optimization Opportunity
/// As in [`Idx3DOffsetSeq`], this type currently ignores the spatial
/// displacement vector associated with each [`Idx3DOffset`]. Accounting for
/// instance. This is an opportunity for enhanced performance beacuase most real-world code only cares about a
/// certain range of displacement magnitudes
#[cfg(test)] // disable outside of testing, for now
pub(crate) struct Iter {
    bounds: Bounds,
    /// the internal representation of the next [`Idx3DOffset`]
    next_offset_zyx: [isize; 3],
}

#[cfg(test)] // disable outside of testing, for now
impl Iterator for Iter {
    type Item = Idx3DOffset;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bounds.stop_offsets_zyx[0] == self.next_offset_zyx[0] {
            None
        } else {
            let out = Some(Idx3DOffset(self.next_offset_zyx));
            update_to_next_offset(&mut self.next_offset_zyx, &self.bounds);
            out
        }
    }
}

/// a (1d) lazy sequence of [`Idx3DOffset`] instances
///
/// This type represents a readonly finite sequence of [`Idx3DOffset`]
/// instances. Importantly, no [`Idx3DOffset`] instance appears more than once
/// in the sequences. In more detail:
/// - instances are initialized to hold every [`Idx3DOffset`] that can
///   describe every pair of meausrements in either:
///   1. a single [`CartesianBlock`] instance (see [`Self::new_auto`])
///   2. from a pair of [`CartesianBlock`]s (see [`Self::new_cross`])
///
/// - this type implements the methods of a sequence. The number of contained
///   [`Idx3DOffset`] instances in a sequence is provided by [`Self::len`],
///   and you can access the `i`th entry with [`Self::get`]
///
/// - this is a lazy sequence in the sense that the [`Idx3DOffset`] instances
///   are computed when they are accessed (i.e. there isn't an underlying
///   array containing every [`Idx3DOffset`] instance). This is similar to
///   "range" types in the Python and Julia programming languages (**note:
///   Rust's "range" types are very different**)
///
/// # Developer Docs: Internal representation
///
/// <div class="warning"> These details can/will change @ any time. </div>
///
/// This is a little complicated. So we are going to go into gory detail. We
/// will primarily frame this discussion in terms of the case where we want
/// pairs drawn from 2 separate [`CartesianBlock`]s (we explain how we handle
/// generalize to the case with a single [`CartesianBlock`] at the very end).
///
/// ## Pairs from separate [`CartesianBlock`]s
/// In order to implement a lazy sequence of the members of the family of
/// [`Idx3DOffset`] instances that describes every unique measurement pair
/// drawn from drawn from 2 separate [`CartesianBlock`] instances, block_a
/// and block_b, we basically need 2 things:
/// 1. the number of "family members," `n_family_members`
/// 2. logic to map every non-negative integer `i` that is less than
///    `n_family_members` to a distinct "family member"
///
/// If block_a and block_b have shapes `shape_a` and `shape_b`, then the ith
/// component of `offset_arr3D`'s shape is `shape_a[i] + shape_b[i] - 1`.
///
/// We can also arrange the [`Idx3DOffset`] instances within ``offset_arr3D``
/// such that  ``offset_arr3D[a,b,c]`` such that ``Idx3DOffset`` has the
/// value of
/// ```text
/// Idx3DOffset([a - shape_a[0] + 1, b - shape_a[1] + 1, c - shape_a[2] + 1)
/// ```
/// Consequently, we can map every 3D index of ``i3off_arr`` to a unique
/// [`Idx3DOffset`] instance.
///
/// Finally, we can imagine "flattening" `i3off_arr` into a 1D array (of
/// length, `n_family_members`). This flattened array, `flat_i3off_arr` is
/// the "thing" represented by this type.
///
/// In practice, we use a [`View3DSpec`] instance to map the 1D index of the
/// `flat_i3off_arr` into the equivalent 3D index (for `i3off_arr`)
/// and then we calculate the appropriate [`Idx3DOffset`] instance from the
/// 3D index.
///
/// ## Pairs from a single [`CartesianBlock`]
///
/// The idea is to treat this scenario exactly like the 2 [`CartesianBlock`]s
/// case, with a small tweak. In more detail, we can imagine building up a
/// `flat_i3off_arr`, where block_a and block_b refer to the same
/// [`CartesianBlock`] instance. To avoid both (i) double-counting measurement
/// pairs, and (ii) every pair composed of the same measurement, this type only
/// reveals considers `flat_i3off_arr[internal_arr_offset..]`, where
/// `flat_i3off_arr[internal_arr_offset]` corresponds to the first
/// [`Idx3DOffset`] instance after `Idx3DOffset([0,0,0])`
pub(crate) struct Idx3DOffsetSeq {
    bounds: Bounds,
    mapping_props: View3DSpec,
    internal_arr_offset: usize, // 0 for cross-region & positive for single-region
}

impl Idx3DOffsetSeq {
    fn setup_bounds_and_viewspec(
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) -> Result<(Bounds, View3DSpec), &'static str> {
        let bounds = Bounds {
            start_offsets_zyx: [
                -(block_a.idx_spec.shape()[0] - 1),
                -(block_a.idx_spec.shape()[1] - 1),
                -(block_a.idx_spec.shape()[2] - 1),
            ],
            stop_offsets_zyx: *block_b.idx_spec.shape(),
        };

        let viewspec = View3DSpec::from_shape_contiguous([
            (bounds.stop_offsets_zyx[0] - bounds.start_offsets_zyx[0]) as usize,
            (bounds.stop_offsets_zyx[1] - bounds.start_offsets_zyx[1]) as usize,
            (bounds.stop_offsets_zyx[2] - bounds.start_offsets_zyx[2]) as usize,
        ])?;

        Ok((bounds, viewspec))
    }

    /// construct the sequence of all [`Idx3DOffset`] instances that describe
    /// every unique pair of measurements from a single [`CartesianBlock`]
    pub fn new_auto(block: &CartesianBlock) -> Result<Self, &'static str> {
        // construct bounds and mapping_props as if we had 2 cartesian blocks
        // with the same shape
        let (bounds, mapping_props) = Self::setup_bounds_and_viewspec(block, block)?;

        let mut first_idx_offset = [0, 0, 0];
        update_to_next_offset(&mut first_idx_offset, &bounds);
        let internal_offset = mapping_props.map_idx3d_to_1d(
            first_idx_offset[0] - bounds.start_offsets_zyx[0],
            first_idx_offset[1] - bounds.start_offsets_zyx[1],
            first_idx_offset[2] - bounds.start_offsets_zyx[2],
        );

        if let Ok(internal_arr_offset) = usize::try_from(internal_offset) {
            Ok(Self {
                bounds,
                mapping_props,
                internal_arr_offset,
            })
        } else {
            // this should be unreachable!
            Err("something went wrong in mapping_props.map_idx3d_to_1d!")
        }
    }

    /// construct the sequence of all [`Idx3DOffset`] instances that describe
    /// every unique pair of measurements from 2 [`CartesianBlock`]s
    pub fn new_cross(
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) -> Result<Self, &'static str> {
        let is_bad = (block_b.start_idx_global_offset[0] < block_a.start_idx_global_offset[0])
            || ((block_b.start_idx_global_offset[0] == block_a.start_idx_global_offset[0])
                && (block_b.start_idx_global_offset[1] < block_a.start_idx_global_offset[1]))
            || ((block_b.start_idx_global_offset[0] == block_a.start_idx_global_offset[0])
                && (block_b.start_idx_global_offset[1] == block_a.start_idx_global_offset[1])
                && (block_b.start_idx_global_offset[2] <= block_a.start_idx_global_offset[2]));
        if is_bad {
            Err("reverse the argument order OR try to use new_auto_iter")
        } else {
            let (bounds, mapping_props) = Self::setup_bounds_and_viewspec(block_a, block_b)?;
            Ok(Self {
                bounds,
                mapping_props,
                internal_arr_offset: 0,
            })
        }
    }

    /// get the number of elements in the sequence
    pub fn len(&self) -> usize {
        self.mapping_props.n_elements() - self.internal_arr_offset
    }

    /// get the ith [`Idx3DOffset`]
    ///
    /// # Note
    /// Unfortunately, we can't get indexing syntax to work (e.g.
    /// `obj[idx]`). The function used for indexing expects must return a
    /// reference. While this makes a lot of sense for a container, it
    /// doesn't make a ton of sense for a range-like object
    pub fn get(&self, index: usize) -> Idx3DOffset {
        let [iz, iy, ix] = self
            .mapping_props
            .map_idx1d_to_3d((index + self.internal_arr_offset) as isize);

        Idx3DOffset([
            iz + self.bounds.start_offsets_zyx[0],
            iy + self.bounds.start_offsets_zyx[1],
            ix + self.bounds.start_offsets_zyx[2],
        ])
    }
}

#[cfg(test)] // disable outside of testing, for now
impl IntoIterator for Idx3DOffsetSeq {
    type Item = Idx3DOffset;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            bounds: self.bounds.clone(),
            next_offset_zyx: self.get(0).0,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::collections::HashMap;

    /// the premise here is that we count up, in a brute force manner, the
    /// number of occurrences of [`Idx3DOffset`] that occur for 2 viewspecs
    ///
    /// currently, we don't use the counts, but the counts will be important
    /// if we ever want to partition work in a nice manner
    fn collect_cross_counts(a: &View3DSpec, b: &View3DSpec) -> HashMap<Idx3DOffset, usize> {
        let mut out = HashMap::new();
        let [nz_a, ny_a, nx_a] = *a.shape();
        let [nz_b, ny_b, nx_b] = *b.shape();

        for iz_a in 0..nz_a {
            for iy_a in 0..ny_a {
                for ix_a in 0..nx_a {
                    // find all offsets
                    for iz_b in 0..nz_b {
                        for iy_b in 0..ny_b {
                            for ix_b in 0..nx_b {
                                let tmp =
                                    Idx3DOffset([(iz_b - iz_a), (iy_b - iy_a), (ix_b - ix_a)]);
                                out.entry(tmp)
                                    .and_modify(|counter| *counter += 1)
                                    .or_insert(1);
                            }
                        }
                    }
                    // done iterating over points from b
                }
            }
        }

        out
    }

    /// the premise here is that we count up, in a brute force manner, the
    /// number of occurrences of [`Idx3DOffset`] that occur within 1 viewspec
    ///
    /// currently, we don't use the counts, but the counts will be important
    /// if we ever want to partition work in a nice manner
    fn collect_single_counts(viewspec: &View3DSpec) -> HashMap<Idx3DOffset, usize> {
        // we do this in a very stupid way, to make sure we do it correctly
        let vec = {
            let mut tmp = Vec::<[isize; 3]>::new();
            let [nz, ny, nx] = *viewspec.shape();
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        tmp.push([iz, iy, ix]);
                    }
                }
            }
            tmp
        };

        // now build up the counts
        let mut out = HashMap::new();
        for (i, [iz_a, iy_a, ix_a]) in vec.iter().enumerate() {
            for [iz_b, iy_b, ix_b] in vec.iter().skip(i + 1) {
                let tmp = Idx3DOffset([(iz_b - iz_a), (iy_b - iy_a), (ix_b - ix_a)]);
                out.entry(tmp)
                    .and_modify(|counter| *counter += 1)
                    .or_insert(1);
            }
        }
        out
    }

    // this is used for holding a CartesianBlock with garbage values
    // (its necessary since the Cartesian is a "view type")
    #[derive(Clone)]
    struct DummyCartesianBlockSrc {
        // it doesn't matter what data actually holds, we should never see it
        data: Vec<f64>,
        idx_spec: View3DSpec,
        start_idx_global_offset: [isize; 3],
    }

    impl<'a> DummyCartesianBlockSrc {
        fn new(idx_spec: View3DSpec, start_idx_global_offset: [isize; 3]) -> Self {
            DummyCartesianBlockSrc {
                data: vec![0.0; idx_spec.required_length()],
                idx_spec,
                start_idx_global_offset,
            }
        }

        fn get_block(&'a self) -> CartesianBlock<'a> {
            let slc: &'a [f64] = self.data.as_slice();
            CartesianBlock {
                value_components_zyx: [slc, slc, slc],
                weights: slc,
                idx_spec: self.idx_spec.clone(),
                start_idx_global_offset: self.start_idx_global_offset,
            }
        }
    }

    fn get_num_occurrences(
        idx_offset: &Idx3DOffset,
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) -> usize {
        let (start, stop) = get_block_a_start_stop_indices(idx_offset, block_a, block_b);
        ((stop[0] - start[0]) as usize)
            * ((stop[1] - start[1]) as usize)
            * ((stop[2] - start[2]) as usize)
    }

    fn check_idxspec_seq(
        idxoffset_seq: &Idx3DOffsetSeq,
        reference_counts: &HashMap<Idx3DOffset, usize>,
        descr: &str,
        block_a: &CartesianBlock,
        block_b: &CartesianBlock,
    ) {
        let mut seen = HashMap::<Idx3DOffset, usize>::new();
        for i in 0..idxoffset_seq.len() {
            let cur_offset = idxoffset_seq.get(i);
            let n_occurrences = get_num_occurrences(&cur_offset, block_a, block_b);
            assert!(
                seen.insert(cur_offset.clone(), n_occurrences).is_none(),
                "idxoffset_seq.get({i}) provides {cur_offset:?} more than once for {descr}"
            );

            let expected_occurrence_count = reference_counts.get(&cur_offset);
            if let Some(expected_occurrence_count) = expected_occurrence_count {
                assert_eq!(
                    n_occurrences, *expected_occurrence_count,
                    "idxoffset_seq.get({i}) provides {cur_offset:?} with the wrong number of occurrences, which is invalid for {descr}"
                )
            } else {
                assert!(
                    expected_occurrence_count.is_some(),
                    "idxoffset_seq.get({i}) provides {cur_offset:?}, which is invalid for {descr}"
                );
            }
        }

        if seen.len() < reference_counts.len() {
            let entry = reference_counts
                .keys()
                .find(|k| !seen.contains_key(*k))
                .unwrap();
            panic!("idxoffset_seq is missing {entry:?} for {descr}")
        }
    }

    fn check_iter(
        idxoffset_itr: Iter,
        reference_counts: &HashMap<Idx3DOffset, usize>,
        descr: &str,
    ) {
        let mut seen = HashMap::<Idx3DOffset, usize>::new();
        for (i, cur_offset) in idxoffset_itr.enumerate() {
            assert!(
                reference_counts.contains_key(&cur_offset),
                "element {i} from the iterator provides {cur_offset:?}, which is invalid for {descr}"
            );
            let n_occurrences = 1; // todo: fixme
            assert!(
                seen.insert(cur_offset.clone(), n_occurrences).is_none(),
                "idxoffset_seq.get({i}) provides {cur_offset:?} more than once for {descr}"
            );
        }

        if seen.len() < reference_counts.len() {
            let entry = reference_counts
                .keys()
                .find(|k| !seen.contains_key(*k))
                .unwrap();
            panic!("iterator is missing {entry:?} for {descr}")
        }
    }

    macro_rules! check2block {
        ($name:ident, $shape_a:expr, $shape_b:expr) => {
            #[test]
            fn $name() {
                let shape_a = $shape_a;
                let shape_b = $shape_b;

                let dummy_block_a = DummyCartesianBlockSrc::new(
                    View3DSpec::from_shape_contiguous(shape_a).unwrap(),
                    [0, 0, 0],
                );
                let dummy_block_b = DummyCartesianBlockSrc::new(
                    View3DSpec::from_shape_contiguous(shape_b).unwrap(),
                    [
                        shape_a[0] as isize,
                        shape_a[1] as isize,
                        shape_a[2] as isize,
                    ],
                );
                let reference_counts =
                    collect_cross_counts(&dummy_block_a.idx_spec, &dummy_block_b.idx_spec);
                //println!("{reference_counts:?}");
                let idxoffset_seq = Idx3DOffsetSeq::new_cross(
                    &dummy_block_a.get_block(),
                    &dummy_block_b.get_block(),
                )
                .unwrap();

                let descr = format!("blocks with the shapes {shape_a:?} & {shape_b:?}");

                check_idxspec_seq(
                    &idxoffset_seq,
                    &reference_counts,
                    &descr,
                    &dummy_block_a.get_block(),
                    &dummy_block_b.get_block(),
                );
                check_iter(idxoffset_seq.into_iter(), &reference_counts, &descr)
            }
        };
    }

    macro_rules! check1block {
        ($name:ident, $shape:expr) => {
            #[test]
            fn $name() {
                let shape = $shape;

                let dummy_block = DummyCartesianBlockSrc::new(
                    View3DSpec::from_shape_contiguous(shape).unwrap(),
                    [0, 0, 0],
                );
                let reference_counts = collect_single_counts(&dummy_block.idx_spec);
                let idxoffset_seq = Idx3DOffsetSeq::new_auto(&dummy_block.get_block()).unwrap();

                let descr = format!("single block with the shape {shape:?}");

                check_idxspec_seq(
                    &idxoffset_seq,
                    &reference_counts,
                    &descr,
                    &dummy_block.get_block(),
                    &dummy_block.get_block(),
                );
                check_iter(idxoffset_seq.into_iter(), &reference_counts, &descr)
            }
        };
    }

    #[test]
    fn check_auto_single_element() {
        let dummy_block = DummyCartesianBlockSrc::new(
            View3DSpec::from_shape_contiguous([1, 1, 1]).unwrap(),
            [0, 0, 0],
        );
        let rslt = Idx3DOffsetSeq::new_auto(&dummy_block.get_block());

        let Ok(idxoffset_seq) = rslt else {
            panic!(
                "we currently expect Idx3DOffsetSeq::new_auto to succeed when handed a 1-element block"
            )
        };

        assert_eq!(
            idxoffset_seq.len(),
            0,
            "a standalone 1-element block contains no pairs"
        );
        let mut iter = idxoffset_seq.into_iter();
        assert!(
            iter.next().is_none(),
            "a standalone 1-element block contains no pairs"
        )
    }

    check2block!(block_pair_same_shape, [2, 3, 4], [2, 3, 4]);
    check2block!(block_pair_single_pair, [1, 1, 1], [1, 1, 1]);
    check2block!(block_pair_diff_shapes, [3, 4, 2], [2, 3, 4]);
    check2block!(block_pair_one_elem_lopsided, [2, 3, 4], [1, 1, 1]);
    check2block!(block_pair_one_elem_lopsided_reverse, [1, 1, 1], [2, 3, 4]);
    check2block!(block_pair_case1a, [2, 3, 4], [5, 1, 1]);
    check2block!(block_pair_case1b, [5, 1, 1], [2, 3, 4]);
    check2block!(block_pair_case2a, [2, 3, 4], [1, 5, 1]);
    check2block!(block_pair_case2b, [1, 5, 1], [2, 3, 4]);
    check2block!(block_pair_case3a, [2, 3, 4], [1, 1, 5]);
    check2block!(block_pair_case3b, [1, 1, 5], [2, 3, 4]);

    check1block!(block_single_1pair_a, [2, 1, 1]);
    check1block!(block_single_1pair_b, [1, 2, 1]);
    check1block!(block_single_1pair_c, [1, 1, 2]);
    check1block!(block_single_arbitrary_a, [2, 3, 4]);
    check1block!(block_single_arbitrary_b, [4, 2, 3]);
    check1block!(block_single_arbitrary_c, [3, 4, 2]);
}
