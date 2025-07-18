// this is based on example code

use cust::prelude::*;
use pairwise_cuda::{exec_vecadd, ExecContext};

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 100_000;

#[test]
fn test_vecadd() {
    let exec_context = ExecContext::new().unwrap();

    // generate our random vectors.
    let lhs = vec![2.0f32; NUMBERS_LEN];
    let rhs = vec![3.0f32; NUMBERS_LEN];
    let out = exec_vecadd(&exec_context, &lhs, &rhs).unwrap();

    for v in out {
        assert_eq!(v, 5.0f32)
    }
}
