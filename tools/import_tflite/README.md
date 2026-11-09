# TFLite translator

Turns an int8 TensorFlow Lite model into a Rust module for the `tinymind`
crate: the model's own int8 weights and int32 biases, integer requantizers,
softmax parameters and a `forward` function. The module is integer-only and
`no_std`.

It is a port of tinymind (C++)'s
[`benchmarks/framework_comparison_f411/gen/make_mlperf.py`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411),
which translates the MLPerf Tiny reference models into tinymind (C++)'s layers,
with a Rust back end. The arithmetic is tinymind (C++)'s, unchanged: requantizer
multipliers are derived from the model's float scales as TFLite Micro derives
them, so the module computes the same integers as tinymind (C++) and TFLM.

## Use

Needs Python 3.10+, numpy and the `tflite` package (flatbuffer bindings of the
TFLite schema; no TensorFlow):

```bash
pip install numpy tflite==2.18.0 flatbuffers==25.12.19
python3 tools/import_tflite/tinymind_import_tflite.py model.tflite model.rs model
```

```rust
mod model;

let y = model::forward(&input); // &model::Input -> model::Output
```

`Input` is the model's input without the batch axis: a feature map
`[[[i8; C]; W]; H]` (NHWC, TFLite's layout) or a vector. Its grid is
`INPUT_SCALE` and `INPUT_ZERO_POINT`, the output's `OUTPUT_SCALE` and
`OUTPUT_ZERO_POINT`.

Feature maps live in one byte arena inside `forward`, planned the way the
translator of tinymind (C++) plans its own: each map goes first-fit at the lowest offset
that overlaps no map still to be read, so a chain ping-pongs between two slots,
a residual block keeps its skip tensor until the `ADD` that reads it, and maps
of every shape share the same bytes (the MLPerf ResNet-8: 49,152 B, as in
tinymind (C++)). Each layer gets typed NHWC views of its slots through safe
`as_chunks`/`first_chunk` and `split_at_mut`, so the generated module needs
Rust 1.88 or later (the crate itself needs 1.85).

## What it supports

A graph, branches included (residual connections), of:

| TFLite operator | Becomes |
|---|---|
| `CONV_2D` (no dilation; SAME or VALID padding) | `QConv2DPerChannel`, or `QPointwiseConv2DPerChannel` for a 1x1 stride-1 one |
| `DEPTHWISE_CONV_2D` (depth multiplier 1) | `QDepthwiseConv2D` |
| `AVERAGE_POOL_2D` (over the whole map) | `QGlobalAvgPool2D` |
| `FULLY_CONNECTED` (per-tensor weights) | `QDense` |
| `SOFTMAX` (one row) | `QSoftmax` |
| `RESHAPE` | nothing, or a copy of a feature map into a vector |
| `ADD` (both inputs and the output of one shape) | `QAdd` |

with fused `NONE`, `RELU` or `RELU6` activations and int8 tensors. Anything else
is rejected with an error naming the operator, as is an `ADD` that broadcasts.

## How it is checked

- `mlperf.py kws` and `mlperf.py ic` fetch the MLPerf Tiny keyword-spotting
  (DS-CNN) and image-classification (ResNet-8, three residual blocks) reference
  models at the commit tinymind (C++)'s benchmark pins, check their SHA-256,
  translate each, run the module with cargo on the benchmark's 8 seeded random
  inputs, and compare the hash of every output byte with tinymind (C++)'s
  expected hash, which tinymind (C++) and TFLite's reference kernels produce.
  `scripts/compare-with-cpp.sh` runs both (the models are downloaded, not
  committed: they are Apache-2.0, MLCommons). Planted mistakes (SAME padding on
  the wrong side, a transposed depthwise kernel, the requantizer's shift sign,
  per-channel scales collapsed; in `ADD`, one input's multiplier or zero point
  given to the other, a left shift of 19; a buffer plan that overwrites the
  skip tensor) each change the hash.
- `test_translator.py` checks what those models cannot show: the softmax
  parameters equal tinymind (C++)'s on 504 calibrations, the `ADD` parameters
  equal tinymind (C++)'s `buildQAddParams` on 48 calibrations, and fused ReLU
  and ReLU6 ranges, on `ADD` too (the models' activations all have zero point
  -128, where a fused ReLU changes nothing).
- On the NUCLEO-F411RE both translated models give the same hashes
  (`bench/`, workloads `mlperf-kws` and `mlperf-ic`).
