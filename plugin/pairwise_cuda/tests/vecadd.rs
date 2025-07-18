// this is based on example code

use cust::prelude::*;
use pairwise_cuda::exec_vecadd;

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 100_000;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

#[test]
fn test_vecadd() {
    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init().unwrap();

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[]).unwrap();

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    // generate our random vectors.
    let lhs = vec![2.0f32; NUMBERS_LEN];
    let rhs = vec![3.0f32; NUMBERS_LEN];
    let out = exec_vecadd(&stream, &module, &lhs, &rhs).unwrap();

    for v in out {
        assert_eq!(v, 5.0f32)
    }
}
