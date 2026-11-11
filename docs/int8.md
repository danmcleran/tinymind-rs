---
title: Int8 inference
layout: default
nav_order: 9
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
| `QAdd` | Elementwise addition of two tensors with their own scales and zero points (TFLite ADD), for residual connections; any 1-, 2- or 3-dimensional int8 array, the same for all three | `QAdd` (`cpp/qadd.hpp`) |
| `QMul` | Elementwise multiplication of two tensors with their own zero points (TFLite MUL), for squeeze-and-excite gating; any 1-, 2- or 3-dimensional int8 array, the same for all three | `QMul` (`cpp/qmul.hpp`) |
| `QPad2D<PT, PB, PL, PR>` | Explicit padding of an NHWC tensor with a constant (TFLite PAD); the convolutions pad by themselves | `QPad2D` (`cpp/qpad.hpp`) |
| `QConcat` | Two NHWC tensors joined along the channel axis, each rescaled onto the output grid (TFLite CONCATENATION) | `QConcat2_2D` (`cpp/qconcat.hpp`) |
| `apply_lut` | Any activation through a 256-entry int8 table, such as sigmoid or tanh | `qApplyLUT` |
| `QLstmCell<'a, IN, HID, C>` | One time step of an LSTM cell: gates i, f, g, o from sigmoid and tanh tables, cell state `C` = `i8` (default) or `i16`, both states updated in place | `QLSTMCell` (`cpp/qlstm.hpp`) |
| `QGruCell<'a, IN, HID>` | One time step of a GRU cell (reset gate applied before the recurrent multiply), hidden state updated in place | `QGRUCell` (`cpp/qgru.hpp`) |
| `QSoftmax` | Softmax over a row of int8 logits onto the 1/256 probability grid, bit-exact with TFLite's int8 softmax (and so with CMSIS-NN's `arm_softmax_s8`) | `QSoftmaxFixedPoint1D` (`cpp/qsoftmax_fixedpoint.hpp`) |
| `calibration` (feature `std`) | Host side: build requantizers, rescalers, additions, multiplications, sigmoid/tanh tables, softmax and recurrent-cell parameters from float scales; affine parameters from a float range; quantize and dequantize | `cpp/include/qcalibration.hpp` |

A layer borrows its weights (`&[[i8; IN]; OUT]`, row-major as PyTorch and TFLite store them) and biases, so they can live in flash as `static` data. On a device, run the calibration on the host and embed the resulting integers; the five fields of a `Requantizer` are plain constants.

## Convolution and pooling

Tensors are NHWC without the batch axis, as nested arrays: an input is `[[[i8; C]; W]; H]`, and convolution weights are `[[[[i8; C]; KW]; KH]; F]` (OHWI, as TFLite stores them). Kernel, strides and padding are const parameters of the layer type, and `forward` takes the input and a caller-owned output, so layers can share buffers. The output's size must be the one the layer produces, `(H + PT + PB - KH) / SH + 1` rows and likewise columns: a wrong size, a zero stride or a kernel larger than the padded input is a compile error, not a run-time check. Stable Rust cannot compute that size from the input's (C++ does it with template arithmetic), so the output array spells it out, and the compiler checks it. Padding holds the input zero point, as TFLite pads, so `PT = PB = PL = PR = 1` around a 3x3 kernel is TFLite's SAME padding at stride 1.

The [`int8` API reference]({{ site.baseurl }}/api/tinymind/int8/) has a compiled example: a convolution followed by max pooling.

The C++ layers gather patches, tile output rows and pair filters for speed; the Rust layers fold the zero point into each filter's bias, hold its weights and a prepared requantizer across the pixels, and read windows inside the input as slices. A convolution over two or more input channels widens its input rows and each filter's weights to 16 bits first and adds the products in pairs; one over a single channel gathers each window once and runs its filters in pairs; pointwise layers run two pixels at a time, and with 32 or more input channels two filters as well. Both add the same integer products, so the results are the same. On the boards the Rust layers run a keyword-spotting shaped network in 13-26% fewer cycles than tinymind (C++)'s portable build, the MLPerf DS-CNN and ResNet-8 each in 31% fewer ([On-board results]({{ site.baseurl }}/benchmarks)).

On Cortex-M4, M7 and M33 those pairs of 16-bit products can become `SMLAD` instructions, from plain Rust, if LLVM is allowed to analyse long sums: add `-C llvm-args=-arm-parallel-dsp-load-limit=512` to the target's `rustflags` in `.cargo/config.toml`. It is an LLVM internal option, not a stable Rust one; on the F411 it took the ResNet-8 from 58.8M to 38.3M cycles.

## Recurrent cells

`QLstmCell` and `QGruCell` run one time step each: `forward` reads an int8 input and the previous state and writes the new state into the caller's buffers, and the caller loops over the sequence. That keeps the cells free of buffers and lets the state live wherever the application wants it, for streaming audio or sensor windows that arrive one sample at a time. Weights are gate-major, `[gate][unit][input]`, in TFLite's and PyTorch's gate order for the LSTM (input, forget, cell, output) and reset, update, new for the GRU. PyTorch's `nn.LSTM` keeps two bias vectors (`bias_ih` and `bias_hh`); add them before quantizing. The GRU applies its reset gate before the recurrent multiply (Keras' `reset_after=False`, whose kernels store the gates as update, reset, new, so reorder them); PyTorch's `nn.GRU` and Keras' default apply it after, so their trained weights do not carry over. Each gate's input and recurrent sums are rescaled to a shared lookup-table grid and the sigmoid and tanh tables have fixed output grids (1/256 and 1/128), so a cell is set up from a handful of float scales: `calibration::build_lstm_params` and `build_gru_params` turn them into multipliers and shifts, `quantize_gate_biases` puts the biases on the table grid, and `build_sigmoid_lut` and `build_tanh_lut` make the tables. An `i16` LSTM cell state keeps 256 times the resolution for long sequences. The [API reference]({{ site.baseurl }}/api/tinymind/int8/calibration/fn.build_lstm_params.html) has a compiled example that calibrates a float LSTM and runs it.

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

The weights and the input are drawn from C++'s `std::mt19937` through libstdc++'s `uniform_real_distribution<float>`, reproduced in the example. The weights are quantized with one symmetric scale per tensor (per channel for the depthwise layer, `calibration::per_channel_symmetric_scales`). Each layer's activation scale comes from calibration: the example runs the same network in float over 32 random inputs, records each layer's output range with `calibration::RangeObserver`, and gives the layer the symmetric scale that covers it (`affine_params_symmetric`), from which `build_requantizer` makes the integer requantizers. It then runs the int8 network and compares its logits with the float model's on the same input:

```
Logits (int8 -> dequantized, float reference):
  class 0:  115 ->  0.001697   float  0.001709
  class 1:  -30 -> -0.000443   float -0.000460
  ...
  class 9:  -15 -> -0.000221   float -0.000214
max |int8 - float| = 0.000020 (0.6% of the float logits' range)
argmax (untrained weights): int8 0, float 0
```

Every tensor (the quantized input, every weight table and requantizer, and each layer's output) and this report are byte-identical to the C++ example's, which CI compiles unchanged and dumps. The per-layer report before it shows each build's own sizes and timings (the Rust max-pool layer has no state, so it reports 0 bytes where the C++ empty struct reports 1).

## Importing a PyTorch model

`tools/import_pytorch` in the repository turns a trained `torch.nn.Sequential` into a Rust module: int8 weights and biases, integer requantizers, baked activation tables and a `forward` function. It takes `Linear`, `ReLU`, `Sigmoid`, `Tanh` and `Softmax` layers, and convolutional networks: `Conv2d` (with stride and padding, including depthwise), `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d(1)` and `Flatten`. The generated module is integer-only and `no_std`, so nothing is computed on the device at startup.

```python
import tinymind_import as ti

layers = ti.from_torch(net)
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration_inputs,
                        "my_model.rs", "my_model")

# A convolutional network: its input shape, and NHWC calibration inputs.
layers = ti.from_torch(cnn, input_shape=(1, 28, 28))     # (channels, height, width)
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(),
                        [x.permute(1, 2, 0).numpy() for x in images],
                        "my_cnn.rs", "my_cnn", input_shape=(28, 28, 1))
```

```rust
mod my_model;
let y = my_model::forward(&input);
```

It is a port of tinymind (C++)'s PyTorch importer with a Rust back end: calibration (min/max, percentile or KL-divergence ranges per layer) and quantization are tinymind (C++)'s code, so the int8 numbers are the same. The `imported_mlp` example runs two models written by the importer (a sigmoid MLP and a softmax classifier), and `imported_cnn` two convolutional ones (padded, depthwise and pointwise convolutions with max and global pooling and softmax; a strided convolution with average pooling, flatten and dense layers). CI imports the same models with tinymind (C++)'s importer, runs them on the tinymind (C++) runtime, and checks the outputs, and for the convolutional ones a hash of every intermediate tensor, are byte-identical. A further check runs the convolutional import path against PyTorch itself, so the NCHW-to-NHWC conversion is tested too. A `BatchNorm2d` after a convolution is folded into it, as tinymind (C++)'s importer does. See the importer's [README](https://github.com/danmcleran/tinymind-rs/tree/master/tools/import_pytorch).

## How it is checked

- **Golden output** from the C++ examples (for `kws_cortex_m_int8`, every tensor), and the live comparison in `scripts/compare-with-cpp.sh`, as for the other examples.
- **tinymind (C++)'s own known answers** for the requantizer and dense layer, including its check of the requantizer's saturating path against an exact 64-bit computation over more than ten thousand cases.
- **Kani proofs** for every input: the requantizer's fast single-rounding path equals the standard two-step rounding (gemmlowp's, which TFLite uses) wherever it is used, and the requantizer, the dense layer, the convolutions, the pooling layers, addition, multiplication, padding, concatenation, the table lookup, the softmax and the LSTM and GRU cells never panic, whatever their parameters; the pooling divide rounds half away from zero.
- **Convolution and pooling against tinymind (C++)**: 57 layer configurations over random tensors, among them every path the C++ takes in its portable build (windows gathered into patches and read row by row, several tiles per output row, odd filter counts and output widths, strides above the kernel size, asymmetric padding and windows wholly in the padding), with requantizers on both paths and pooling ranges with `qmin > qmax`. Ten planted defects (padding with 0 instead of the zero point, swapped strides, one requantizer for every filter, rounding toward zero, ...) each fail it. (tinymind (C++)'s Arm DSP kernels add the same products in another order, so they give the same integers; the comparison runs the portable build.)
- **Addition, multiplication, padding and concatenation against tinymind (C++)**: `QAdd` and `QMul` with their calibration on every pair of int8 inputs for 48 and 32 calibrated layers, plus random integer parameters; `QConcat` on five shapes and `QPad2D` on six padding patterns.
- **Recurrent cells against tinymind (C++)**: seven LSTM shapes (int8 and int16 cell state) and six GRU shapes, each with weights, biases and tables calibrated from random float models and with random integer parameters (tables, zero points and bounds, `qmin > qmax` included), hashing the state after every step of 32-step sequences; the calibration's multipliers and shifts are compared too. Eight planted defects (the cell's tanh table for the g gate, the GRU's `1 - z` off by one, the new gate reading `h` instead of `r * h`, biases truncated instead of rounded, ...) each fail it.
- **The TFLite translator**: the MLPerf Tiny DS-CNN and ResNet-8 give tinymind (C++)'s output hashes, and its padding, multiplication and concatenation match TFLite's own reference kernels on generated models.
- **Softmax against TFLite**: the golden rows and sweep hashes that tinymind (C++)'s unit test takes from the TFLite Micro reference kernel, and a sweep against the tinymind (C++) softmax and its calibration over 504 input scales and betas and row lengths 1 to 511.
- **No reachable panics**: the int8 path is part of the Cortex-M firmware image that CI links with an undefined panic handler.

Where the C++ would hit undefined behaviour (an int32 accumulator overflowing, a shift outside its documented range), the Rust port wraps or clamps instead, so every function returns a value; inside the documented ranges the results are identical.

## Not yet

The rest of tinymind (C++)'s int8 family (attention, positional encoding, state-space and liquid-time cells, mixture of experts) is not ported yet, the importers do not emit the recurrent cells, and the PyTorch importer does not take branches. tinymind (C++)'s ONNX importer is not ported. For int8 TensorFlow Lite models, `tools/import_tflite` translates convolutional networks into a Rust module the same way tinymind (C++)'s MLPerf benchmark translates them into C++; it runs the MLPerf Tiny keyword-spotting model and the residual ResNet-8 with tinymind (C++)'s output hashes ([On-board results]({{ site.baseurl }}/benchmarks)). Decision trees and gradient-boosted ensembles, with their own importer, are on the [Decision trees]({{ site.baseurl }}/trees) page.

The [API reference]({{ site.baseurl }}/api/tinymind/int8/) documents every function.
