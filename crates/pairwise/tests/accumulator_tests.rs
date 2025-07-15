use ndarray::{NewAxis, s};
use pairwise::{Accumulator, Histogram, Mean, get_output_from_statepack_array};
use pairwise_nostd_internal::{AccumStateView, AccumStateViewMut, Datum};
use std::collections::HashMap;

// this is inefficient, but it gets the job done for now
//
// this is probably an indication that we could improve the Accumulator API
fn _get_output_single(
    accum: &impl Accumulator,
    stateprop: &AccumStateView,
) -> HashMap<&'static str, Vec<f64>> {
    get_output_from_statepack_array(accum, &stateprop.as_array_view().slice(s![.., NewAxis]))
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
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.init_accum_state(&mut accum_state);

        accum.consume(
            &mut accum_state,
            &Datum {
                value: 4.0,
                weight: 1.0,
            },
        );
        let value_map = _get_output_single(&accum, &accum_state.as_view());

        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 1.0);
    }

    #[test]
    fn mean_consume_twice() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.init_accum_state(&mut accum_state);

        accum.consume(
            &mut accum_state,
            &Datum {
                value: 4.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut accum_state,
            &Datum {
                value: 8.0,
                weight: 1.0,
            },
        );

        let value_map = _get_output_single(&accum, &accum_state.as_view());
        assert_eq!(value_map["mean"][0], 6.0);
        assert_eq!(value_map["weight"][0], 2.0);
    }

    #[test]
    fn merge() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.init_accum_state(&mut accum_state);
        accum.consume(
            &mut accum_state,
            &Datum {
                value: 4.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut accum_state,
            &Datum {
                value: 8.0,
                weight: 1.0,
            },
        );

        let mut storage_other = [0.0, 0.0];
        let mut accum_state_other = AccumStateViewMut::from_contiguous_slice(&mut storage_other);
        accum.init_accum_state(&mut accum_state_other);
        accum.consume(
            &mut accum_state_other,
            &Datum {
                value: 1.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut accum_state_other,
            &Datum {
                value: 3.0,
                weight: 1.0,
            },
        );

        accum.merge(&mut accum_state, &accum_state_other.as_view());

        let value_map = _get_output_single(&accum, &accum_state.as_view());
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
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        accum.init_accum_state(&mut accum_state);

        accum.consume(
            &mut accum_state,
            &Datum {
                value: 0.5,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut accum_state,
            &Datum {
                value: -50.0,
                weight: 1.0,
            },
        );
        accum.consume(
            &mut accum_state,
            &Datum {
                value: 1000.0,
                weight: 1.0,
            },
        );

        let value_map = _get_output_single(&accum, &accum_state.as_view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 0.0);

        accum.consume(
            &mut accum_state,
            &Datum {
                value: 1.1,
                weight: 5.0,
            },
        );
        let value_map = _get_output_single(&accum, &accum_state.as_view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 5.0);
    }
}
