---
title: Int8 inference
layout: default
nav_order: 7
---

# Int8 inference

Alongside the Q-format networks, the crate runs **int8 affine-quantized** models, the format TensorFlow Lite, Arm CMSIS-NN and most microcontroller NPUs use. A real value is `scale * (q - zero_point)`: activations are int8 with any zero point, weights are int8 with zero point 0, and biases and accumulators are int32. Inference is integer-only, so it needs no FPU, and the module is `no_std` like the rest of the library.

## The pieces

| Rust (`tinymind::int8`) | What it does | tinymind (C++) |
|---|---|---|
| `Requantizer` | Rescales an int32 accumulator to the output tensor's int8 grid with a Q0.31 multiplier and a shift, adds the zero point and saturates | `Requantizer` (`cpp/qaffine.hpp`) |
| `QDense<'a, IN, OUT>` | Fully connected layer: `acc[o] = bias[o] + sum_i w[o][i] * (x[i] - zp)`, then requantize | `QDense` (`cpp/qdense.hpp`) |
| `QDensePerChannel<'a, IN, OUT>` | The same with one requantizer per output, as TFLite quantizes dense layers by default | `QPointwiseConv2DPerChannel` on a 1x1 input (`cpp/qpointwiseconv2d.hpp`) |
| `qrelu`, `qrelu6` | ReLU and ReLU6 in the quantized grid | `cpp/qactivations.hpp` |
| `QConv2D<'a, C, KH, KW, F, SH, SW, PT, PB, PL, PR>` | 2D convolution over an NHWC tensor: `F` filters of `KH` x `KW` x `C`, strides `SH`, `SW` (default 1) and zero-point padding on each side (default 0); one requantizer | `QConv2D` (`cpp/qconv2d.hpp`) |
| `QConv2DPerChannel<...>` | The same with one requantizer per filter, as TFLite quantizes convolutions | `QConv2DPerChannel` |
| `QDepthwiseConv2D<'a, C, KH, KW, SH, SW, PT, PB, PL, PR>` | One `KH` x `KW` filter per channel, channels not mixed; one requantizer per channel | `QDepthwiseConv2D` (`cpp/qdepthwiseconv2d.hpp`) |
| `QPointwiseConv2D<'a, C, F>`, `QPointwiseConv2DPerChannel<'a, C, F>` | 1x1 convolution: the same `C`-to-`F` dense layer at every pixel | `QPointwiseConv2D`, `QPointwiseConv2DPerChannel` (`cpp/qpointwiseconv2d.hpp`) |
| `QMaxPool2D<PH, PW, SH, SW>`, `QAvgPool2D<...>`, `QGlobalAvgPool2D` | Max, average (rounded half away from zero) and global average pooling, in the input's scale | `cpp/qpool2d.hpp` |
| `apply_lut` | Any activation through a 256-entry int8 table, such as sigmoid or tanh | `qApplyLUT` |
| `QSoftmax` | Softmax over a row of int8 logits onto the 1/256 probability grid, bit-exact with TFLite's int8 softmax (and so with CMSIS-NN's `arm_softmax_s8`) | `QSoftmaxFixedPoint1D` (`cpp/qsoftmax_fixedpoint.hpp`) |
| `calibration` (feature `std`) | Host side: build requantizers, sigmoid/tanh tables and softmax parameters from float scales; affine parameters from a float range; quantize and dequantize | `cpp/include/qcalibration.hpp` |

A layer borrows its weights (`&[[i8; IN]; OUT]`, row-major as PyTorch and TFLite store them) and biases, so they can live in flash as `static` data. On a device, run the calibration on the host and embed the resulting integers; the five fields of a `Requantizer` are plain constants.

## Convolution and pooling

Tensors are NHWC without the batch axis, as nested arrays: an input is `[[[i8; C]; W]; H]`, and convolution weights are `[[[[i8; C]; KW]; KH]; F]` (OHWI, as TFLite stores them). Kernel, strides and padding are const parameters of the layer type, and `forward` takes the input and a caller-owned output, so layers can share buffers. The output's size must be the one the layer produces, `(H + PT + PB - KH) / SH + 1` rows and likewise columns: a wrong size, a zero stride or a kernel larger than the padded input is a compile error, not a run-time check. Stable Rust cannot compute that size from the input's (C++ does it with template arithmetic), so the output array spells it out, and the compiler checks it. Padding holds the input zero point, as TFLite pads, so `PT = PB = PL = PR = 1` around a 3x3 kernel is TFLite's SAME padding at stride 1.

The [`int8` API reference]({{ site.baseurl }}/api/tinymind/int8/) has a compiled example: a convolution followed by max pooling.

The C++ layers gather patches, tile output rows and pair filters for speed; those add the same integer products, so the results are the same. The Rust layers are plain loops for now.

## Example: a PyTorch-trained XOR network

The `int8_xor` example is a port of tinymind (C++)'s [PyTorch int8 XOR example](https://danmcleran.github.io/tinymind/examples/pytorch_quant_xor.html). A 2-4-1 network trained in PyTorch and quantized to int8 runs as dense layer, quantized ReLU, dense layer and sigmoid table:

```bash
cargo run --release --features std --example int8_xor -- output
```

```
  in=(0.0000,0.0000)  q_input=(-128,-128)  q_logit=-128  prob=0.1211  pred=0  label=0
  in=(0.0000,1.0000)  q_input=(-128,127)  q_logit=127  prob=0.8828  pred=1  label=1
  in=(1.0000,0.0000)  q_input=(127,-128)  q_logit=127  prob=0.8828  pred=1  label=1
  in=(1.0000,1.0000)  q_input=(127,127)  q_logit=-126  prob=0.1250  pred=0  label=0
```

Its stdout and decision surface are byte-identical to the C++ example's, including the sigmoid table, which both build from the same float math.

## Example: a keyword-spotting shaped CNN

The `kws_cortex_m_int8` example is a port of tinymind (C++)'s example of the same name: a network shaped like a small keyword spotter, with synthetic weights, run once with a per-layer report.

```
input [20 x 20 x 1] -> QConv2D 3x3, 8 filters    -> [18 x 18 x 8]
                    -> QMaxPool2D 2x2            -> [9 x 9 x 8]
                    -> QDepthwiseConv2D 3x3      -> [7 x 7 x 8]   (one requantizer per channel)
                    -> QPointwiseConv2D 8 -> 16  -> [7 x 7 x 16]
                    -> QGlobalAvgPool2D          -> [16]
                    -> QPointwiseConv2D 16 -> 10 -> 10 logits
```

```bash
cargo run --release --features std --example kws_cortex_m_int8
```

The weights and the input are drawn from C++'s `std::mt19937` through libstdc++'s `uniform_real_distribution<float>`, reproduced in the example, and calibrated with `calibration::per_channel_symmetric_scales`, `build_requantizer` and `quantize`. Every tensor (the quantized input, every weight table and requantizer, and each layer's output) is byte-identical to the C++ example's, which CI compiles unchanged and dumps. The report's sizes and timings are each build's own (the Rust max-pool layer has no state, so it reports 0 bytes where the C++ empty struct reports 1). With these synthetic weights the signal shrinks at every layer and the ten logits come out 0 in both versions; the comparison of the intermediate tensors is what checks the layers.

## Importing a PyTorch model

`tools/import_pytorch` in the repository turns a trained `torch.nn.Sequential` of `Linear`, `ReLU`, `Sigmoid`, `Tanh` and `Softmax` layers into a Rust module: int8 weights and biases, integer requantizers, baked activation tables and a `forward` function. The generated module is integer-only and `no_std`, so nothing is computed on the device at startup.

```python
import tinymind_import as ti

layers = ti.from_torch(net)
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration_inputs,
                        "my_model.rs", "my_model")
```

```rust
mod my_model;
let y = my_model::forward(&input);
```

It is a port of tinymind (C++)'s PyTorch importer with a Rust back end: calibration (min/max, percentile or KL-divergence ranges per layer) and quantization are tinymind (C++)'s code, so the int8 numbers are the same. The `imported_mlp` example runs two models written by the importer (a sigmoid MLP and a softmax classifier); CI imports the same two models with tinymind (C++)'s importer, runs them on the tinymind (C++) runtime, and checks the outputs are byte-identical. See the importer's [README](https://github.com/danmcleran/tinymind-rs/tree/master/tools/import_pytorch).

## How it is checked

- **Golden output** from the C++ examples (for `kws_cortex_m_int8`, every tensor), and the live comparison in `scripts/compare-with-cpp.sh`, as for the other examples.
- **tinymind (C++)'s own known answers** for the requantizer and dense layer, including its check of the requantizer's saturating path against an exact 64-bit computation over more than ten thousand cases.
- **Kani proofs** for every input: the requantizer's fast single-rounding path equals the standard two-step rounding (gemmlowp's, which TFLite uses) wherever it is used, and the requantizer, the dense layer, the convolutions, the pooling layers, the table lookup and the softmax never panic, whatever their parameters; the pooling divide rounds half away from zero.
- **Convolution and pooling against tinymind (C++)**: 57 layer configurations over random tensors, among them every path the C++ takes in its portable build (windows gathered into patches and read row by row, several tiles per output row, odd filter counts and output widths, strides above the kernel size, asymmetric padding and windows wholly in the padding), with requantizers on both paths and pooling ranges with `qmin > qmax`. Ten planted defects (padding with 0 instead of the zero point, swapped strides, one requantizer for every filter, rounding toward zero, ...) each fail it. (tinymind (C++)'s Arm DSP kernels add the same products in another order, so they give the same integers; the comparison runs the portable build.)
- **Softmax against TFLite**: the golden rows and sweep hashes that tinymind (C++)'s unit test takes from the TFLite Micro reference kernel, and a sweep against the tinymind (C++) softmax and its calibration over 504 input scales and betas and row lengths 1 to 511.
- **No reachable panics**: the int8 path is part of the Cortex-M firmware image that CI links with an undefined panic handler.

Where the C++ would hit undefined behaviour (an int32 accumulator overflowing, a shift outside its documented range), the Rust port wraps or clamps instead, so every function returns a value; inside the documented ranges the results are identical.

## Not yet

The rest of tinymind (C++)'s int8 family (residual add, explicit padding, recurrent cells, attention) is not ported yet, and the importer does not emit convolutions yet: it accepts dense layers with ReLU, sigmoid, tanh or softmax only. tinymind (C++)'s ONNX importer is not ported. Decision trees and gradient-boosted ensembles, with their own importer, are on the [Decision trees]({{ site.baseurl }}/trees) page.

The [API reference]({{ site.baseurl }}/api/tinymind/int8/) documents every function.
