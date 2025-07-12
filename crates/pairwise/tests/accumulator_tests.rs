use ndarray::{NewAxis, s};
use pairwise::{Accumulator, Histogram, Mean, get_output};
use pairwise_nostd_internal::{AccumStateView, AccumStateViewMut, DataElement};
use std::collections::HashMap;

// this is inefficient, but it gets the job done for now
//
// this is probably an indication that we could improve the Accumulator API
fn _get_output_single(
    accum: &impl Accumulator,
    stateprop: &AccumStateView,
) -> HashMap<&'static str, Vec<f64>> {
    get_output(accum, &stateprop.as_array_view().slice(s![.., NewAxis]))
}

// TODO: factor out this function and the get_output function from
//       integration_tests.rs (they are identical)

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn mean_consume_once() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.reset_accum_state(&mut statepack);

        accum.consume(
            &mut statepack,
            &DataElement {
                value: 4.0,
                weight: 1.0,
            },
        );
        let value_map = _get_output_single(&accum, &statepack.as_view());

        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 1.0);
    }

    #[test]
    fn mean_consume_twice() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.reset_accum_state(&mut statepack);

        accum.consume(
            &mut statepack,
            &DataElement {
                value: 4.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut statepack,
            &DataElement {
                value: 8.0,
                weight: 1.0,
            },
        );

        let value_map = _get_output_single(&accum, &statepack.as_view());
        assert_eq!(value_map["mean"][0], 6.0);
        assert_eq!(value_map["weight"][0], 2.0);
    }

    #[test]
    fn merge() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.reset_accum_state(&mut statepack);
        accum.consume(
            &mut statepack,
            &DataElement {
                value: 4.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut statepack,
            &DataElement {
                value: 8.0,
                weight: 1.0,
            },
        );

        let mut storage_other = [0.0, 0.0];
        let mut statepack_other = AccumStateViewMut::from_contiguous_slice(&mut storage_other);
        accum.reset_accum_state(&mut statepack_other);
        accum.consume(
            &mut statepack_other,
            &DataElement {
                value: 1.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut statepack_other,
            &DataElement {
                value: 3.0,
                weight: 1.0,
            },
        );

        accum.merge(&mut statepack, &statepack_other.as_view());

        let value_map = _get_output_single(&accum, &statepack.as_view());
        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 4.0);
    }

    #[test]
    fn hist_invalid_hist_edges() {
        assert!(Histogram::new(&[0.0]).is_err());
    }
    #[test]
    fn hist_nonmonotonic() {
        assert!(Histogram::new(&[1.0, 0.0]).is_err());
    }

    #[test]
    fn hist_consume() {
        let accum = Histogram::new(&[0.0, 1.0, 2.0]).unwrap();

        let mut storage = [0.0, 0.0];
        let mut statepack = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.reset_accum_state(&mut statepack);

        accum.consume(
            &mut statepack,
            &DataElement {
                value: 0.5,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut statepack,
            &DataElement {
                value: -50.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut statepack,
            &DataElement {
                value: 1000.0,
                weight: 1.0,
            },
        );

        let value_map = _get_output_single(&accum, &statepack.as_view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 0.0);

        accum.consume(
            &mut statepack,
            &DataElement {
                value: 1.1,
                weight: 5.0,
            },
        );
        let value_map = _get_output_single(&accum, &statepack.as_view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 5.0);
    }
}
