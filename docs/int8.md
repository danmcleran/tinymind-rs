---
title: Int8 inference
layout: default
nav_order: 5
---

# Int8 inference

Alongside the Q-format networks, the crate runs **int8 affine-quantized** models, the format TensorFlow Lite, Arm CMSIS-NN and most microcontroller NPUs use. A real value is `scale * (q - zero_point)`: activations are int8 with any zero point, weights are int8 with zero point 0, and biases and accumulators are int32. Inference is integer-only, so it needs no FPU, and the module is `no_std` like the rest of the library.

## The pieces

| Rust (`tinymind::int8`) | What it does | C++ (tinymind) |
|---|---|---|
| `Requantizer` | Rescales an int32 accumulator to the output tensor's int8 grid with a Q0.31 multiplier and a shift, adds the zero point and saturates | `Requantizer` (`cpp/qaffine.hpp`) |
| `QDense<'a, IN, OUT>` | Fully connected layer: `acc[o] = bias[o] + sum_i w[o][i] * (x[i] - zp)`, then requantize | `QDense` (`cpp/qdense.hpp`) |
| `QDensePerChannel<'a, IN, OUT>` | The same with one requantizer per output, as TFLite quantizes dense layers by default | `QPointwiseConv2DPerChannel` on a 1x1 input (`cpp/qpointwiseconv2d.hpp`) |
| `qrelu`, `qrelu6` | ReLU and ReLU6 in the quantized grid | `cpp/qactivations.hpp` |
| `apply_lut` | Any activation through a 256-entry int8 table, such as sigmoid or tanh | `qApplyLUT` |
| `QSoftmax` | Softmax over a row of int8 logits onto the 1/256 probability grid, bit-exact with TFLite's int8 softmax (and so with CMSIS-NN's `arm_softmax_s8`) | `QSoftmaxFixedPoint1D` (`cpp/qsoftmax_fixedpoint.hpp`) |
| `calibration` (feature `std`) | Host side: build requantizers, sigmoid/tanh tables and softmax parameters from float scales; quantize and dequantize | `cpp/include/qcalibration.hpp` |

A layer borrows its weights (`&[[i8; IN]; OUT]`, row-major as PyTorch and TFLite store them) and biases, so they can live in flash as `static` data. On a device, run the calibration on the host and embed the resulting integers; the five fields of a `Requantizer` are plain constants.

## Example: a PyTorch-trained XOR network

The `int8_xor` example is a port of tinymind's [PyTorch int8 XOR example](https://danmcleran.github.io/tinymind/examples/pytorch_quant_xor.html). A 2-4-1 network trained in PyTorch and quantized to int8 runs as dense layer, quantized ReLU, dense layer and sigmoid table:

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

It is a port of tinymind's PyTorch importer with a Rust back end: calibration (min/max, percentile or KL-divergence ranges per layer) and quantization are tinymind's code, so the int8 numbers are the same. CI imports one model with both importers and runs it on both runtimes; the outputs are byte-identical. See the importer's [README](https://github.com/danmcleran/tinymind-rs/tree/master/tools/import_pytorch).

## How it is checked

- **Golden output** from the C++ example, and the live comparison in `scripts/compare-with-cpp.sh`, as for the other examples.
- **tinymind's own known answers** for the requantizer and dense layer, including its check of the requantizer's saturating path against an exact 64-bit computation over more than ten thousand cases.
- **Kani proofs** for every input: the requantizer's fast single-rounding path equals the standard two-step rounding (gemmlowp's, which TFLite uses) wherever it is used, and the requantizer, the dense layer, the table lookup and the softmax never panic, whatever their parameters.
- **Softmax against TFLite**: the golden rows and sweep hashes that tinymind's unit test takes from the TFLite Micro reference kernel, and a sweep against tinymind's C++ softmax and its calibration over 504 input scales and betas and row lengths 1 to 511.
- **No reachable panics**: the int8 path is part of the Cortex-M firmware image that CI links with an undefined panic handler.

Where the C++ would hit undefined behaviour (an int32 accumulator overflowing, a shift outside its documented range), the Rust port wraps or clamps instead, so every function returns a value; inside the documented ranges the results are identical.

## Not yet

Convolution, depthwise convolution, pooling and the rest of tinymind's int8 family (recurrent cells, attention, trees) are not ported yet, so the importer accepts dense layers with ReLU, sigmoid, tanh or softmax only. tinymind's ONNX importer is not ported.

The [API reference]({{ site.baseurl }}/api/tinymind/int8/) documents every function.
