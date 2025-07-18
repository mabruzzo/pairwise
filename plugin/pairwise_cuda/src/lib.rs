use cust::{module, prelude::*, stream, util::SliceExt};

use std::error::Error;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// This caches all of the information needed to execute a cuda kernel
/// (this may not be an optimal abstraction)
pub struct ExecContext {
    stream: stream::Stream,
    module: module::Module,
    // We don't need the context for anything but it must be kept alive.
    _context: Context,
}

impl ExecContext {
    pub fn new() -> Result<ExecContext, Box<dyn Error>> {
        let context = cust::quick_init()?;

        Ok(ExecContext {
            // make a CUDA stream to issue calls to
            stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
            // Make the CUDA module (a module holds the GPU code for the
            // kernels) they can be made from PTX code, cubins, or fatbins.
            module: Module::from_ptx(PTX, &[])?,
            _context: context,
        })
    }
}

pub fn exec_vecadd(
    exec_context: &ExecContext,
    lhs: &[f32],
    rhs: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    // allocate the GPU memory needed to house our numbers and copy them over.
    let lhs_gpu = lhs.as_dbuf()?;
    let rhs_gpu = rhs.as_dbuf()?;
    let len = lhs_gpu.len();
    if len != rhs_gpu.len() {
        panic!("unequal sizes");
    }

    // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
    // cost of the copy, but you need to be careful not to read from the buffer.
    let mut out = vec![0.0f32; lhs_gpu.len()];
    let out_buf = out.as_slice().as_dbuf()?;

    // retrieve the `vecadd` kernel from the module so we can calculate the right launch config.
    let vecadd = exec_context.module.get_function("vecadd")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (len as u32).div_ceil(block_size);

    println!("using {grid_size} blocks and {block_size} threads per block");

    let stream = &exec_context.stream;
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            vecadd<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                len,
                rhs_gpu.as_device_ptr(),
                len,
                out_buf.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;

    // copy back the data from the GPU.
    out_buf.copy_to(&mut out)?;
    Ok(out)
}
