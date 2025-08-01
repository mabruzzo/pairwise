use std::collections::HashMap;

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2, Axis};

use pairwise_nostd_internal::{AccumStateView, OutputDescr, Reducer, StatePackViewMut};

/// compute the output quantities from an accumulator's state properties and
/// return the result in a HashMap.
///
/// # Notes
/// This is primarily used for testing.
///
/// TODO: I'm not sure I really want this to be a part of the standard API.
///       Before the 1.0 release, we should either move this to a private
///       testing_helpers crate OR we should explicitly decide to make this
///       part of the public API.
/// compute the output quantities from an accumulator's state properties and
/// return the result in a HashMap.
///
/// # Notes
/// This is primarily used for testing.
///
/// TODO: I'm not sure I really want this to be a part of the standard API.
///       Before the 1.0 release, we should either move this to a private
///       testing_helpers crate OR we should explicitly decide to make this
///       part of the public API.
pub fn get_output(
    reducer: &impl Reducer,
    statepack: &StatePackViewMut,
) -> HashMap<&'static str, Vec<f64>> {
    get_output_from_statepack_array(reducer, &statepack.as_array_view())
}

// todo: figure out how to remove me before the first release
//
// we're holding onto this for now since the name of the standard type we use
// to describe a statepack, [`StatePackViewMut`] implies that the statepack
// is mutable. This operates on immutable state data, which makes sense from
// a design perspective.
//
// There are 2 obvious solutions:
// 1. create an object, in the pairwise crate (rather than in the no_std
//     crate), that actually owns the StatePack data, and pass that in as a
//     reference. If we do that, then `StatePackViewMut` would typically
//     view the data inside of the type.
// 2. Alternatively, we might rename `StatePackViewMut` to something a little
//    more generic...
//
// I'm not in a rush to do this since a different solution may present itself
// as we work on developing an ergonomic higher-level API
pub fn get_output_from_statepack_array(
    reducer: &impl Reducer,
    statepack_data: &ArrayView2<f64>,
) -> HashMap<&'static str, Vec<f64>> {
    let description = reducer.output_descr();
    let n_bins = statepack_data.shape()[1];
    let n_comps = description.n_per_accum_state();

    // todo: the rest of this function could definitely be optimized!
    let mut buffer = vec![0.0; n_comps * n_bins];
    let mut buffer_view = ArrayViewMut2::from_shape([n_comps, n_bins], &mut buffer).unwrap();
    for i in 0..n_bins {
        reducer.value_from_accum_state(
            &mut buffer_view.index_axis_mut(Axis(1), i),
            &AccumStateView::from_array_view(statepack_data.index_axis(Axis(1), i)),
        );
    }

    match description {
        OutputDescr::MultiScalarComp(names) => {
            let _to_vec = |row: ArrayView1<f64>| row.iter().cloned().collect();
            let row_iter = buffer_view.rows().into_iter().map(_to_vec);
            HashMap::from_iter(names.iter().cloned().zip(row_iter))
        }
        OutputDescr::SingleVecComp { name, .. } => HashMap::from([(name, buffer)]),
    }
}
