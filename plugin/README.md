At the moment the stuff in this directory is compiled totally separately from
other things in the repository. I need to add installation instructions here.

Long-term, I'm thinking that maybe we should be compiling these crates as
dynamic libraries and loading in the functionality with the libloading crate,
but I'm open to suggestions.

## Installation
See rust-CUDA for details. The writeup there is a little out of date since we
are currently using a version of rust-CUDA pinned to a git hash

Be aware that in addition to requiring LLVM-7, installing this stuff also
requires a clang-compiler. These are used for totally distinct versions
- LLVM-7 is used in the codegeneration process
- the clang-compiler is used to generate bindings (via bindgen) for the CPU
  functions for launching kernels, managing memory, etc
