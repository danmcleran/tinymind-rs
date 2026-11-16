//! Runs three int8 models written by the ONNX importer (`tools/import_onnx`),
//! the importer's worked examples (`tools/import_onnx/example_onnx.py`): small
//! PyTorch networks exported to ONNX and quantized by ONNX Runtime's
//! `quantize_static`, the way a trained model is. `onnx_cnn` (QDQ format,
//! per-channel weights) has a strided convolution, a pointwise and a padded
//! one, three dense layers with tanh and sigmoid between them and a softmax;
//! `onnx_cnn_qop` (QOperator format, per-tensor weights) a padded convolution,
//! a pointwise one, two dense layers and a sigmoid; `onnx_mlp` a MatMul with a
//! separate bias Add, tanh, a dense layer and a softmax. The generated modules
//! are `examples/common/imported_onnx_{cnn,cnn_qop,mlp}_model.rs`.
//!
//! tinymind (C++)'s own ONNX importer reads the same models into its headers,
//! and `scripts/cpp/imported_onnx.cpp` runs them on tinymind (C++)'s int8
//! layers: the output is byte-identical (`scripts/compare-with-cpp.sh`).
//!
//! Run: `cargo run --release --example imported_onnx`.

use std::io::{self, BufWriter, Write};

#[path = "common/imported_onnx_driver.rs"]
mod imported_onnx_driver;

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    imported_onnx_driver::run(&mut out)?;
    out.flush()
}
