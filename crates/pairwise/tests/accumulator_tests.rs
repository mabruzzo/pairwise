use ndarray::{ArrayView1, ArrayViewMut1, NewAxis, s};
use pairwise::{Accumulator, Histogram, Mean, get_output};
use std::collections::HashMap;

// this is inefficient, but it gets the job done for now
//
// this is probably an indication that we could improve the Accumulator API
fn _get_output_single(
    accum: &impl Accumulator,
    stateprop: &ArrayView1<f64>,
) -> HashMap<&'static str, Vec<f64>> {
    get_output(accum, &stateprop.slice(s![.., NewAxis]))
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
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);

        accum.consume(&mut statepack, 4.0, 1.0);
        let value_map = _get_output_single(&accum, &statepack.view());

        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 1.0);
    }

    #[test]
    fn mean_consume_twice() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);

        accum.consume(&mut statepack, 4.0, 1.0);
        accum.consume(&mut statepack, 8.0, 1.0);

        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["mean"][0], 6.0);
        assert_eq!(value_map["weight"][0], 2.0);
    }

    #[test]
    fn merge() {
        let accum = Mean;

        let mut storage = [0.0, 0.0];
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);
        accum.consume(&mut statepack, 4.0, 1.0);
        accum.consume(&mut statepack, 8.0, 1.0);

        let mut storage_other = [0.0, 0.0];
        let mut statepack_other = ArrayViewMut1::from_shape([2], &mut storage_other).unwrap();
        accum.reset_statepack(&mut statepack_other);
        accum.consume(&mut statepack_other, 1.0, 1.0);
        accum.consume(&mut statepack_other, 3.0, 1.0);

        accum.merge(&mut statepack, statepack_other.view());

        let value_map = _get_output_single(&accum, &statepack.view());
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
        let mut statepack = ArrayViewMut1::from_shape([2], &mut storage).unwrap();
        accum.reset_statepack(&mut statepack);

        accum.consume(&mut statepack, 0.5, 1.0);
        accum.consume(&mut statepack, -50.0, 1.0);
        accum.consume(&mut statepack, 1000.0, 1.0);

        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 0.0);

        accum.consume(&mut statepack, 1.1, 5.0);
        let value_map = _get_output_single(&accum, &statepack.view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 5.0);
    }
}
