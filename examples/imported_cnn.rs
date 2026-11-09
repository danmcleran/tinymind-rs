//! Runs two convolutional models written by the PyTorch importer
//! (`tools/import_pytorch`), the importer's worked examples
//! (`tools/import_pytorch/example_cnn.py`): a 12x12x1 network with a padded
//! convolution, max pooling, depthwise and pointwise convolutions, global average
//! pooling and a softmax head (`examples/common/imported_cnn_model.rs`), and a
//! 9x11x2 network with a strided convolution, average pooling, flatten, two dense
//! layers and tanh (`examples/common/imported_cnn_flat_model.rs`). The output is
//! byte-identical to the same models imported with tinymind (C++)'s own importer
//! and run on its C++ runtime (`scripts/compare-with-cpp.sh`).
//!
//! Run: `cargo run --release --features std --example imported_cnn`. The
//! generated modules are `no_std`; `std` is only for quantizing the float test
//! inputs here.

use std::io::{self, BufWriter, Write};

#[path = "common/imported_cnn_driver.rs"]
mod imported_cnn_driver;

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    imported_cnn_driver::run(&mut out)?;
    out.flush()
}
