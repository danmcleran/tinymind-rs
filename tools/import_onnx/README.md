# ONNX importer

Turns an int8 ONNX model, quantized by ONNX Runtime, into a Rust module for
the `tinymind` crate: the model's int8 weights and int32 biases, integer
requantizers, activation tables, softmax parameters and a `forward`
function. The module is integer-only and `no_std`.

It is a port of tinymind (C++)'s
[`apps/import_onnx`](https://github.com/danmcleran/tinymind/tree/master/apps/import_onnx)
with a Rust back end. The parser is tinymind (C++)'s, unchanged (CI checks
that it still is): the same layers, scales, zero points, int8 weights, int32
biases and requantizer multipliers and shifts. tinymind (C++)'s importer writes
per-layer constants for a network you write by hand; this one also writes the
network, so the module runs as generated.

## Use

Needs Python 3.10+, numpy and `onnx`:

```bash
pip install numpy onnx
python3 tools/import_onnx/tinymind_import_onnx.py model_int8.onnx model.rs model
```

```rust
mod model;

let y = model::forward(&input); // &model::Input -> model::Output
```

Quantize the model with ONNX Runtime first, int8 weights and activations,
QDQ (the default) or QOperator format, per tensor or per channel:

```python
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

quantize_static("model.onnx", "model_int8.onnx", calibration_data_reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                per_channel=False)
```

TensorFlow/Keras and JAX models get there through ONNX
(`python -m tf2onnx.convert --saved-model ... --opset 13`, `jax2onnx`), a
PyTorch model through `torch.onnx.export`; tinymind (C++)'s
[importer README](https://github.com/danmcleran/tinymind/tree/master/apps/import_onnx)
has the whole recipe.

`Input` is the model's input without the batch axis: a feature map
`[[[i8; C]; W]; H]` or a vector. Feature maps are NHWC, the transpose of the
NCHW tensor the ONNX model takes (the importer reorders the weights to
match). The input's grid is `INPUT_SCALE` and `INPUT_ZERO_POINT`, the output's
`OUTPUT_SCALE` and `OUTPUT_ZERO_POINT`: real value = `SCALE * (q - ZERO_POINT)`.

## What it supports

A chain of layers from the model's one input to its one output:

| ONNX (QDQ / QOperator) | Rust |
|---|---|
| `Conv` / `QLinearConv` (group 1, no dilation, explicit padding) | `QConv2D` or `QConv2DPerChannel`; `QPointwiseConv2D` / `QPointwiseConv2DPerChannel` for a 1x1, stride 1, unpadded convolution |
| `Gemm` / `QGemm`, `MatMul` / `QLinearMatMul`, with a bias `Add` / `QLinearAdd` after a `MatMul` folded in | `QDense` or `QDensePerChannel` |
| `Relu` fused into the layer before it (ONNX Runtime does this) | the requantizer's lower bound |
| a standalone `Relu` (same grid in and out) | `qrelu` |
| `Sigmoid` / `QLinearSigmoid`, `Tanh` | a 256-entry table (`apply_lut`), built as tinymind (C++)'s `buildQSigmoidLUT` / `buildQTanhLUT` build it |
| `Softmax` over a vector, as the last layer (QDQ form) | `QSoftmax` (TFLite Micro's int8 softmax) |
| `Flatten`, `Reshape` to 2D, `Identity` | no layer: a dense layer reads the feature map in place |

Anything else is refused with a `ValueError` naming the node, never skipped:
pooling, residual additions and other operators, grouped (depthwise) or
dilated convolutions, `auto_pad` `SAME_*`, uint8 tensors, weight zero points
other than 0, QOperator's `QLinearSoftmax`, a model whose layers branch, and
a tensor requantized between two layers (a layer must read its input in the
grid the layer before it wrote it in).

Two results differ from ONNX Runtime's by design:

- A softmax's output is on TFLite's probability grid, scale 1/256 and zero
  point -128, which is what `QSoftmax` computes; ONNX Runtime quantizes it to
  1/255. The module's `OUTPUT_SCALE` says 1/256.
- ONNX Runtime requantizes in float and folds a `MatMul` + `Add` in two
  roundings, the module in integers and one, so a layer's output can differ
  from ONNX Runtime's by one step.

## Files

- `tinymind_import_onnx.py`: the importer (`import_onnx_model(model, out, name)`,
  or the command line above).
- `models/*.onnx`: three small example models (`cnn`: QDQ, per-channel
  weights, strided, pointwise and padded convolutions, tanh, sigmoid, softmax;
  `cnn_qop`: QOperator, per-tensor weights, sigmoid; `mlp`: a MatMul with a
  separate bias Add, tanh, softmax), built by `make_models.py` (needs torch and
  onnxruntime).
- `example_onnx.py`: writes `examples/common/imported_onnx_*_model.rs`
  (`--rust`), or tinymind (C++)'s header for a model plus a C++ forward pass
  over tinymind (C++)'s layers (`--cpp`), which `scripts/cpp/imported_onnx.cpp`
  runs.
- `test_onnx.py`: checks the example modules against ONNX Runtime and the
  importer's refusals (needs onnxruntime and cargo).

## How it is checked

- `scripts/compare-with-cpp.sh` runs the three example models through
  tinymind (C++)'s own importer and its int8 layers and compares the output with
  `examples/imported_onnx.rs` byte for byte (16 inputs per model), regenerates
  the three committed modules and compares them, and checks that the parser
  here is tinymind (C++)'s, line for line. `tests/imported_onnx_golden.rs`
  checks the example against the C++ output in `cargo test`.
- `test_onnx.py`: on the same inputs, every output of `onnx_cnn_qop` is within
  one step of ONNX Runtime's (46 of 48 equal), and the softmax models'
  probabilities are within 1.5/256 of ONNX Runtime's. It also builds a model
  that requantizes a tensor between two layers and one with a second output,
  and checks that the importer refuses both.
- Planted defects in the back end (per-channel requantizers in reverse order,
  the activation table's zero points swapped, the softmax built from the
  output scale) and in the parser (no NHWC reordering of the dense weights
  after a Flatten) each fail both checks. `ci/panic-never` links the three
  modules' `forward` with no reachable panic.
