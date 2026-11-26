//! Runs five int8 models written by the ONNX importer (`tools/import_onnx`),
//! the importer's worked examples (`tools/import_onnx/example_onnx.py`): small
//! PyTorch networks exported to ONNX and quantized by ONNX Runtime's
//! `quantize_static`, the way a trained model is. `onnx_cnn` (QDQ format,
//! per-channel weights) has a strided convolution, a pointwise and a padded
//! one, three dense layers with tanh and sigmoid between them and a softmax;
//! `onnx_cnn_qop` (QOperator format, per-tensor weights) a padded convolution,
//! a pointwise one, two dense layers and a sigmoid; `onnx_mlp` a MatMul with a
//! separate bias Add, tanh, a dense layer and a softmax. `onnx_resnet` (QDQ,
//! per-channel) and `onnx_resnet_qop` (QOperator, per-tensor, ending in a
//! softmax) are a residual network: a convolution, max pooling, a depthwise
//! and a pointwise convolution added to the tanh of the block's input, a
//! sigmoid, average and global average pooling and a dense layer. The
//! generated modules are
//! `examples/common/imported_onnx_{cnn,cnn_qop,mlp,resnet,resnet_qop}_model.rs`.
//! Four variants run two of them with the importer's int4 and 16x8 options:
//! `onnx_cnn_int4` and `onnx_resnet_int4` (int4 weights in every dense and
//! convolution layer), `onnx_cnn_16x8` and `onnx_resnet_qop_16x8` (int16
//! activations with the int8 weights).
//!
//! tinymind (C++)'s own ONNX importer reads the same models into its headers,
//! and `scripts/cpp/imported_onnx.cpp` runs them on tinymind (C++)'s int8
//! layers, and the variants through tinymind (C++)'s own code generator with
//! the same options: the output is byte-identical
//! (`scripts/compare-with-cpp.sh`).
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
