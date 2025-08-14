//! Define API functions to actually drive the reduction
//!
//! In practice, these functions all call into a private method of
//! [`Accumulator`]. In other words, these methods could all be methods of
//! [`Accumulator`].
//!
//! I'm choosing to make these act as standalone functions, because I'm
//! worried about [`Accumulator`]s becoming a "god" object. I'll talk through
//! my current thinking below, but **I'm totally open to adopting a different
//! design**
//!
//! ## Current Rationale
//! It may be useful to enumerate the inputs into the calculation
//!
//! 1. primary spatial info (e.g. `CartesianBlock`,
//!    [`pairstat_nostd_internal::UnstructuredPoints`])
//! 2. optional secondary spatial measurements (for cross or inter-tile ops)
//! 3. distance edge bins
//! 4. reducer config: the specific reducer-config, PairwiseOperation
//! 5. runtime-parameters that don't have a meaningful impact on the output
//!    (i.e. results mathematically consistent, if not bitwise identical),
//!    but impact performance (see [`RuntimeSpec`])
//! 6. reducer state
//! 7. maybe someday: extra options tied to spatial representation. Examples
//!    might include:
//!    - for [`pairstat_nostd_internal::UnstructuredPoints`]: we might want to
//!      use a distance metric other than Euclidean (e.g. for images)
//!    - for `CartesianBlock`: we might want to support consideration of a
//!      subset of displacement vectors, rather than always forcing an
//!      isotropic calculation (e.g. only consider displacements parallel to
//!      axis2 or in the "axis1-axis2 plane")
//!
//! Currently, [`Accumulator`] is responsible for inputs 3,4, and 6 (it may
//! also make sense for it to take responsibility for input 7).
//! - Because, it tracks reducer state, I primarily think of it as a
//!   data-class, that just happens to track relevant additional metadata.
//! - this perspective is informed by the fact that reducer-state is the
//!   primary output of a calculation. To elaborate:
//!   - distance edge bins and the reducer config are linked to the reducer
//!     state because they describe what's in the reducer... I think of them
//!     as relatively lightweight, easily copied metadata.[^lightweight]
//!   - I can easily rationalize all of the public methods because they all
//!     relate to updating/resetting/post-processing "the data"
//!   - all other non-public methods are just implementation details that we
//!     are free to change (in fact, we could radically change a lot if we
//!     decided to use enum-dispatch)
//!   - if the reducer state were just a temporary buffer (that doesn't carry
//!     any state between calculations), and we spit out a totally different
//!     result, I would feel **VERY** differently.
//! - I feel pretty strongly that we don't want to include input 5 inside
//!   of [`Accumulator`] because that feels like we are heading into
//!   "god-object" territory...
//!   - to elaborate, there's no causal link between runtime-parameters that
//!     have no meaningful impact on the result and the reducer state)
//!   - I have less of an opinion about input #7
//!
//! I guess its worth discussing an alternative design. It's very common for
//! fft libraries (or C libraries in general), to create a "plan" object that
//! includes all of the configuration data. I would be receptive to a design
//! like this where the reducer state is stored separately.
//! - then we could attach the functionality that drives the reduction
//!   directly as a method to the command or plan object... (the method would
//!   take inputs 1, 2, and 6 as arguments)
//! - I guess the key reason why I'm more ok with this is that the
//!   plan/command object is largely immutable and relatively lightweight.
//!
//!
//! [^lightweight]: strictly speaking, we don't want to copy these things
//!     around unnecessarily, since they involve a heap allocations for distance
//!     bin edges. But, if this ever becomes a problem, we can consider using
//!     `Arc` (since the type is immutable).

use pairstat_nostd_internal::{CartesianBlock, CellWidth, UnstructuredPoints};

use crate::{Accumulator, Error, wrapped_reducer::SpatialInfo};

/// This is mostly a placeholder.
///
/// In the future this might track information like:
/// - standard parallelism parameters (n_teams, n_members_per_team)
/// - the backend to use (e.g. GPUs or Threads or Serial)
/// - which GPU to use if there are multiple GPUs
pub struct RuntimeSpec;

/// Update `accum` with contributions from the specified measurements
pub fn process_cartesian<'a>(
    accum: &mut Accumulator,
    block_a: CartesianBlock<'a>,
    block_b: Option<CartesianBlock<'a>>,
    cell_width: CellWidth,
    _: &RuntimeSpec,
) -> Result<(), Error> {
    accum.exec_reduction(SpatialInfo::Cartesian {
        block_a,
        block_b,
        cell_width,
    })
}

/// Update `accum` with contributions from the specified measurements
pub fn process_unstructured<'a>(
    accum: &mut Accumulator,
    points_a: UnstructuredPoints<'a>,
    points_b: Option<UnstructuredPoints<'a>>,
    _: &RuntimeSpec,
) -> Result<(), Error> {
    accum.exec_reduction(SpatialInfo::Unstructured { points_a, points_b })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AccumulatorBuilder;
    use ndarray::ArrayView2;

    #[test]
    fn check_mismatched_unstructured_points() {
        let mut accum = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .dist2_bin_edges(&[0.0, 1.0, 9.0, 16.0])
            .build()
            .unwrap();

        let positions = [6.0, 7.0, 12.0, 13.0, 18.0, 19.0];

        let values = [-9., -8., -3., -2., 3., 4.];
        let points_a = UnstructuredPoints::new(
            ArrayView2::from_shape((3, 2), &positions).unwrap(),
            ArrayView2::from_shape((3, 2), &values).unwrap(),
            &[1.0; 2],
        )
        .unwrap();

        // should fail for mismatched spatial dimensions
        let points_b = UnstructuredPoints::new(
            ArrayView2::from_shape((2, 3), &positions).unwrap(),
            ArrayView2::from_shape((2, 3), &values).unwrap(),
            &[1.0; 3],
        )
        .unwrap();
        assert!(
            process_unstructured(&mut accum, points_a.clone(), Some(points_b), &RuntimeSpec)
                .is_err()
        );
    }
}
