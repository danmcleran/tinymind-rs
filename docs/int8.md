---
title: Int8 inference
layout: default
nav_order: 15
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
| `QFft1D<'a, N, HALF>` | Radix-2 FFT of length `N` (a power of two) on int16 real and imaginary buffers, in place, with Q1.15 twiddle tables of `HALF = N / 2` entries borrowed from flash: forward scaled by 1/N, inverse unscaled, squared bin magnitudes in int32 | `QFFT1D` (`cpp/qfft1d.hpp`) |
| `QRFft1D<'a, N, HALF>` | Real-input FFT: `N` real int16 samples in, the `N / 2 + 1` bins that are not redundant out (the rest are their conjugates), scaled by 1/N, from the same twiddle tables as `QFft1D<N>`. It runs an `N / 2`-point complex transform and a split step: at `N = 256`, 448 butterflies instead of 1,024. Its largest error against an exact DFT is about 1.3 to 1.7 times `QFft1D`'s on the same input (about 9 LSB at `N = 256`, where `QFft1D`'s is about 6) | `QRFFT1D` (`cpp/qfft1d.hpp`) |
| `QDenseInt4<'a, IN, OUT, PACKED>`, `QConv2DInt4<'a, C, KH, KW, F, PACKED, ...>`, `QDepthwiseConv2DInt4<'a, C, KH, KW, PACKED, ...>` | The dense, convolution and depthwise layers over int4 weights packed two per byte (element `k` in the low nibble of byte `k / 2` when `k` is even, as TFLite and ONNX pack them), one requantizer per output channel. Each computes exactly what `QDensePerChannel`, `QConv2DPerChannel` or `QDepthwiseConv2D` computes on the same weights as `i8`, for half the weight flash. `PACKED = (weights + 1) / 2` is spelled out and checked at compile time. Layers of up to 1,024 weights unpack them onto the stack and run the int8 kernels; larger ones compute from the packed bytes. `pack_int4`, `unpack_int4` and `int4_packed_bytes` handle the packing | `QDenseInt4`, `QConv2DInt4`, `QDepthwiseConv2DInt4` (`cpp/qint4.hpp`) |
| `QLogMelSpectrogram<'a, N, HALF, BANDS, LOG_FRAC_BITS>` | Log-mel spectrogram of one frame of `N` int16 samples: pre-emphasis, a Q1.15 window, a per-frame block scale (so a quiet frame reaches the FFT at full scale), `QRFft1D`, Q0.16 mel filters and a natural log, as `i16` in Q(`LOG_FRAC_BITS`, default 8); torchaudio's `MelSpectrogram` (power 2, HTK mel scale) followed by a log. Caller-owned scratch: `QMelWorkspace<N, BINS>` | `QLogMelSpectrogram`, `QMelWorkspace` (`cpp/qmel.hpp`) |
| `QMfcc<'a, BANDS, COEFFS>` | MFCCs from log-mel bands: an orthonormal DCT-II in Q1.15, `i32` out | `QMfcc` (`cpp/qmel.hpp`) |
| `QKMeansAnomaly<'a, F, K>` | Anomaly score of an int8 feature vector: the (optionally weighted) squared distance to the nearest of `K` centroids over that cluster's mean squared radius, in Q8 (a typical member of a cluster scores about 256) | `QKMeansAnomaly` (`cpp/qanomaly.hpp`) |
| `QKMeansAnomalyTrainer<F, K>` | Learns a `QKMeansAnomaly` on the device from normal samples, one at a time: online k-means on Q8 centroids (with a cap that turns each centroid into a moving average), feature weights from each feature's spread within its cluster, then the radii from calibration samples. RAM `3 * K * F` bytes plus 18 per cluster and 10 per feature, plus a few fixed bytes (16 on a 64-bit host) | `QKMeansAnomalyTrainer` (`cpp/qanomaly.hpp`) |
| `QDiagGaussianAnomaly<'a, F, K>` | Negative log-likelihood of an int8 feature vector under its most likely component of a diagonal Gaussian mixture, in Q8 nats | `QDiagGaussianAnomaly` (`cpp/qanomaly.hpp`) |
| `calibration` (feature `std`) | Host side: build requantizers, rescalers, additions, multiplications, sigmoid/tanh tables, softmax and recurrent-cell parameters from float scales, FFT twiddle tables, the log-mel window, filters and DCT, Gaussian-mixture anomaly tables from a float fit, and per-channel int4 weights; affine parameters from a float range; quantize and dequantize | `cpp/include/qcalibration.hpp` |

A layer borrows its weights (`&[[i8; IN]; OUT]`, row-major as PyTorch and TFLite store them) and biases, so they can live in flash as `static` data. On a device, run the calibration on the host and embed the resulting integers; the five fields of a `Requantizer` are plain constants.

Define the layers as `const` or `static` items, as the importers generate them: the compiler then knows every weight and requantizer and builds the loops around them. A layer filled in at run time, for example from a model read out of a flash partition, gives the same results but runs slower. On the F411 (2026-11-12) the keyword-spotting shaped network's convolution over one input channel took 2.1 times as long, its pointwise and depthwise layers 10% and 5% longer, and the int8 MLP 1% longer.

## 16x8: int16 activations

The `int16x8` module runs TFLite's 16x8 scheme: int16 activations with twice int8's resolution, for twice the activation RAM, with the weights still int8 and the biases and accumulators int64. Its dense, convolution, depthwise, max-pooling and addition layers have the int8 layers' names, shapes and conventions (a pointwise convolution is a 1x1 `QConv2D` or `QConv2DPerChannel`; there is no average pooling), so such an int8 model becomes a 16x8 one by changing the activation type, the bias type and the requantizers, with no recalibration of the weights. TFLite requires int16 tensors to have zero point 0; these layers accept any.

| Rust | What it does | tinymind (C++) |
|------|--------------|----------------|
| `int16x8::Requantizer` | Rescales an `i64` accumulator to int16 exactly as TFLite's 64-bit `MultiplyByQuantizedMultiplier` does: the Q0.31 multiplier reduced to Q0.15 and one rounding. Defined outside TFLite's contract too (the accumulator and shift are clamped, an out-of-range result saturates) | `Requantizer<int64_t, int16_t>` (`cpp/qaffine.hpp`) |
| `int16x8::QDense`, `QDensePerChannel`, `QConv2D`, `QConv2DPerChannel`, `QDepthwiseConv2D`, `QMaxPool2D` | The int8 layers' shapes and parameters over `i16` tensors, `i8` weights and `i64` biases | `QDense`, `QConv2D`, `QConv2DPerChannel`, `QDepthwiseConv2D`, `QMaxPool2D` with int16 storage |
| `int16x8::QAdd` | Element-wise addition of two int16 tensors (TFLite's int16 ADD, left shift 15), with an `i32` requantizer (`Requantizer32`) onto the int16 output | `QAdd<int16_t, int16_t, int16_t, N>` (`cpp/qadd.hpp`) |
| `int16x8::apply_lut` | TFLite's 513-entry interpolated int16 table (`LUTLookup`): the top 9 bits pick an interval, the low 7 interpolate along it | `qApplyLUT16` (`cpp/qactivations.hpp`) |
| `int16x8::QSoftmax` | TFLite's int16 softmax, bit for bit: an exp table over [-10, 0] and a reciprocal table, both TFLite's, kept as constants | `QSoftmaxInt16` (`cpp/qsoftmax_fixedpoint.hpp`) |
| `int16x8::calibration` (feature `std`) | Host side: requantizers and the addition from float scales, sigmoid and tanh tables, the softmax's parameters, and the softmax's two tables (built as TFLite builds them, to check the constants) | `buildQLUT16`, `buildQSigmoidLUT16`, `buildQTanhLUT16`, `buildQSoftmaxInt16LUTs`, `buildQSoftmaxInt16Params`, `buildQAddParams` |

The 16x8 layers sum products in `i32` where that is exact, gather each convolution window once per pixel, and on the Cortex-M0+ run the 64-bit requantizer in 32-bit arithmetic: the digits model in 16x8 takes 10% fewer cycles than tinymind (C++)'s scalar build on the Cortex-M4F and the same on the M0+ ([On-board results]({{ site.baseurl }}/benchmarks)). tinymind (C++)'s SMLAD and SMLAL forms need `unsafe` here.

## Convolution and pooling

Tensors are NHWC without the batch axis, as nested arrays: an input is `[[[i8; C]; W]; H]`, and convolution weights are `[[[[i8; C]; KW]; KH]; F]` (OHWI, as TFLite stores them). Kernel, strides and padding are const parameters of the layer type, and `forward` takes the input and a caller-owned output, so layers can share buffers. The output's size must be the one the layer produces, `(H + PT + PB - KH) / SH + 1` rows and likewise columns: a wrong size, a zero stride or a kernel larger than the padded input is a compile error, not a run-time check. Stable Rust cannot compute that size from the input's (C++ does it with template arithmetic), so the output array spells it out, and the compiler checks it. Padding holds the input zero point, as TFLite pads, so `PT = PB = PL = PR = 1` around a 3x3 kernel is TFLite's SAME padding at stride 1.

The [`int8` API reference]({{ site.baseurl }}/api/tinymind/int8/) has a compiled example: a convolution followed by max pooling.

The C++ layers gather patches, tile output rows and pair filters for speed; the Rust layers fold the zero point into each filter's bias, hold its weights and a prepared requantizer across the pixels, and read windows inside the input as slices. A convolution over two or more input channels widens its input rows and each filter's weights to 16 bits first and adds the products in pairs; one over a single channel gathers each window once and runs its filters in pairs; pointwise layers run two pixels at a time, and with 32 or more input channels two filters as well. Both add the same integer products, so the results are the same. On the boards the Rust layers run a keyword-spotting shaped network in 12-26% fewer cycles than tinymind (C++)'s portable build, the MLPerf DS-CNN and ResNet-8 each in 31% fewer ([On-board results]({{ site.baseurl }}/benchmarks)).

On Cortex-M4, M7 and M33 those pairs of 16-bit products can become `SMLAD` instructions, from plain Rust, if LLVM is allowed to analyse long sums: add `-C llvm-args=-arm-parallel-dsp-load-limit=512` to the target's `rustflags` in `.cargo/config.toml`. It is an LLVM internal option, not a stable Rust one; on the F411 it took the ResNet-8 from 58.8M to 38.5M cycles.

## Recurrent cells

The PyTorch importer emits a `QLstmCell` for a model that starts with an `nn.LSTM` ([below](#importing-a-pytorch-model)).

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

`tools/import_pytorch` in the repository turns a trained `torch.nn.Sequential` into a Rust module: int8 weights and biases, integer requantizers, baked activation tables and a `forward` function. It takes `Linear`, `ReLU`, `Sigmoid`, `Tanh` and `Softmax` layers, convolutional networks (`Conv2d` with stride and padding, including depthwise, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d(1)` and `Flatten`), and sequence models that start with one `nn.LSTM` layer, whose last hidden state feeds the rest of the network. The generated module is integer-only and `no_std`, so nothing is computed on the device at startup.

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

# A sequence model: nn.Sequential(lstm, nn.Linear(...), ...), only its
# parameters are read. Calibrate with [step][feature] arrays.
layers = ti.from_torch(seq_model, sequence_length=32)
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), sequences,
                        "my_lstm.rs", "my_lstm")
```

```rust
mod my_model;
let y = my_model::forward(&input);
```

It is a port of tinymind (C++)'s PyTorch importer with a Rust back end: calibration (min/max, percentile or KL-divergence ranges per layer) and quantization are tinymind (C++)'s code, so the int8 numbers are the same. The `imported_mlp` example runs two models written by the importer (a sigmoid MLP and a softmax classifier), and `imported_cnn` two convolutional ones (padded, depthwise and pointwise convolutions with max and global pooling and softmax; a strided convolution with average pooling, flatten and dense layers). CI imports the same models with tinymind (C++)'s importer, runs them on the tinymind (C++) runtime, and checks the outputs, and for the convolutional ones a hash of every intermediate tensor, are byte-identical. A further check runs the convolutional import path against PyTorch itself, so the NCHW-to-NHWC conversion is tested too. A `BatchNorm2d` after a convolution is folded into it, as tinymind (C++)'s importer does.

An LSTM model becomes a `QLstmCell` (int16 cell state by default, int8 on request) plus the rest of the network: `forward` runs a whole sequence from zero state, and `head` runs the rest of the network on a hidden state, for code that steps the cell itself as samples arrive. The hidden and cell states get symmetric scales from the calibration sequences, and the gates a fixed grid of 8/127 (they saturate beyond ±8), as tinymind (C++)'s LSTM tests and TFLite's integer LSTM use. tinymind (C++)'s importer computes the same parameters for the `imported_lstm` example's two models, and they are checked against tinymind (C++)'s runtime: a C++ program builds every integer parameter from the same float scales with tinymind (C++)'s calibration functions, runs `QLSTMCell` and `QDense`, and prints the same parameters, a hash of the hidden and cell state after every step and the outputs, byte for byte. A further check runs the import against PyTorch's own `nn.LSTM`. On the example classifier's 16 test sequences the int8 probabilities are within 0.057 of the float model's and pick the same class every time. `nn.GRU` is rejected: PyTorch applies its reset gate after the recurrent product and `QGruCell` before it, so the trained weights would compute something else. See the importer's [README](https://github.com/danmcleran/tinymind-rs/tree/master/tools/import_pytorch).

## Importing an ONNX model

`tools/import_onnx` reads an int8 ONNX model quantized by ONNX Runtime (`quantize_static`, QDQ or QOperator format, weights per tensor or per channel) and writes the same kind of Rust module. ONNX is how TensorFlow/Keras (through `tf2onnx`), JAX and other frameworks reach it; a PyTorch model can come this way too, already quantized, instead of through the PyTorch importer's own calibration.

```bash
python3 tools/import_onnx/tinymind_import_onnx.py model_int8.onnx my_model.rs my_model
```

Its parser is tinymind (C++)'s ONNX importer, unchanged, so the int8 weights, int32 biases and requantizers are the ones tinymind (C++) computes. tinymind (C++)'s importer writes constants for a network you write by hand; this one also writes the network: convolutions (pointwise and depthwise ones on their own layers), max and average pooling, residual additions, dense layers, ReLU folded into the layer before it, sigmoid and tanh tables, and a softmax as the last layer. Feature maps live in buffers inside `forward`, each reused once no later layer reads the map it holds, so a residual block keeps its skip tensor only until the addition. A model with anything else (padded pooling, broadcasting, a layer that requantizes its input) is refused with the node's name. Feature maps are NHWC, the transpose of the model's NCHW input, and a softmax's output is on TFLite's 1/256 grid rather than ONNX Runtime's 1/255. The `imported_onnx` example runs five models quantized by ONNX Runtime, two of them residual networks; CI runs them through tinymind (C++)'s own ONNX importer and C++ layers too, and the outputs are byte-identical. Against ONNX Runtime itself the outputs are within one step and softmax probabilities within 2/256 (ONNX Runtime requantizes in float, and quantizes a softmax to 1/255). See the importer's [README](https://github.com/danmcleran/tinymind-rs/tree/master/tools/import_onnx).

With `int4_dense=True, int4_conv=True` the importer writes the same model with every dense and convolution layer on per-channel int4 weights (`QDenseInt4`, `QConv2DInt4`, `QDepthwiseConv2DInt4`), and with `activation_bits=16` with int16 activations and the int8 weights (the `int16x8` layers), as tinymind (C++)'s code generator does with the same options; CI checks four such models against that generator's C++ output. `tinymind_qat.py` (tinymind (C++)'s, needs PyTorch) fine-tunes a model so its weights sit on the int4 grid before export.

With `arena=True` the module keeps every intermediate activation in one arena that the caller passes, `forward(&input, &mut arena)`, planned by tinymind (C++)'s planner (TFLite Micro's greedy-by-size placement; an addition or activation writes over an input that dies there, through `QAdd::forward_in_place`). The arena is the size tinymind (C++)'s generated model uses (`ARENA_BYTES`, with the plan in `BUFFERS`), so a residual network needs about half the activation memory of a buffer per layer. Such modules view the arena with `as_chunks`, which needs Rust 1.88; the crate builds on 1.85. The [digits example]({{ site.baseurl }}/onnx-digits) runs three of them.

## How it is checked

- **Golden output** from the C++ examples (for `kws_cortex_m_int8`, every tensor), and the live comparison in `scripts/compare-with-cpp.sh`, as for the other examples.
- **tinymind (C++)'s own known answers** for the requantizer and dense layer, including its check of the requantizer's saturating path against an exact 64-bit computation over more than ten thousand cases.
- **Kani proofs** for every input: the requantizer's fast single-rounding path equals the standard two-step rounding (gemmlowp's, which TFLite uses) wherever it is used, and the requantizer, the dense layer, the convolutions, the pooling layers, addition, multiplication, padding, concatenation, the table lookup, the softmax, the LSTM and GRU cells, both FFTs, the int4 and 16x8 layers, the in-place addition, the log-mel spectrogram, the anomaly scores and the k-means trainer never panic, whatever their parameters; the pooling divide rounds half away from zero; the FFT's Q1.15 multiply floors and its magnitudes saturate only where both parts are -32768, the one sum that overflows `i32`; and the FFT's saturation skip on cores without a saturating instruction changes no output.
- **Convolution and pooling against tinymind (C++)**: 57 layer configurations over random tensors, among them every path the C++ takes in its portable build (windows gathered into patches and read row by row, several tiles per output row, odd filter counts and output widths, strides above the kernel size, asymmetric padding and windows wholly in the padding), with requantizers on both paths and pooling ranges with `qmin > qmax`. Ten planted defects (padding with 0 instead of the zero point, swapped strides, one requantizer for every filter, rounding toward zero, ...) each fail it. (tinymind (C++)'s Arm DSP kernels add the same products in another order, so they give the same integers; the comparison runs the portable build.)
- **Addition, multiplication, padding and concatenation against tinymind (C++)**: `QAdd` and `QMul` with their calibration on every pair of int8 inputs for 48 and 32 calibrated layers, plus random integer parameters; `QConcat` on five shapes and `QPad2D` on six padding patterns.
- **Recurrent cells against tinymind (C++)**: seven LSTM shapes (int8 and int16 cell state) and six GRU shapes, each with weights, biases and tables calibrated from random float models and with random integer parameters (tables, zero points and bounds, `qmin > qmax` included), hashing the state after every step of 32-step sequences; the calibration's multipliers and shifts are compared too. Eight planted defects (the cell's tanh table for the g gate, the GRU's `1 - z` off by one, the new gate reading `h` instead of `r * h`, biases truncated instead of rounded, ...) each fail it.
- **The ONNX importer**: five models quantized by ONNX Runtime (QDQ and QOperator, per-channel and per-tensor weights, two residual networks with pooling) give byte-identical outputs here and on tinymind (C++)'s importer and layers, and the parser is checked to be tinymind (C++)'s, unchanged.
- **The TFLite translator**: the MLPerf Tiny DS-CNN and ResNet-8 give tinymind (C++)'s output hashes, and its padding, multiplication and concatenation match TFLite's own reference kernels on generated models.
- **Softmax against TFLite**: the golden rows and sweep hashes that tinymind (C++)'s unit test takes from the TFLite Micro reference kernel, and a sweep against the tinymind (C++) softmax and its calibration over 504 input scales and betas and row lengths 1 to 511.
- **FFT against tinymind (C++)**: `QFFT1D`, `QRFFT1D` and `buildQFFTTwiddles` at every power-of-two length from 2 to 1024: the tables, then impulses, DC, tones, random samples and full-scale values that saturate, through the forward and inverse transforms, the round trip and the bin magnitudes, and the real FFT of each signal's real part, on the built tables and on random ones. Twelve planted defects in `QFft1D` (one stage without its halving, saturating before the shift, an off-by-one bit reversal, truncating instead of rounding the tables, ...) and nine in `QRFft1D` (a flipped sign in the odd spectrum, the wrong twiddle for the mirrored bin, the first stage without its headroom, ...) each fail it. It also runs on Cortex-M under QEMU, and the boards time a 256-point complex and real-input transform ([On-board results]({{ site.baseurl }}/benchmarks)).
- **Int4 layers against tinymind (C++)**: `QDenseInt4`, `QConv2DInt4` and `QDepthwiseConv2DInt4` on random tensors with the requantizers and biases of the convolution sweep: dense layers with odd and even row lengths, convolutions small enough to unpack their weights and large enough to compute from the packed bytes (through a gathered window, row by row and in place for a one-row kernel, the C++'s three paths), odd windows that start every second filter mid-byte, padding and strides; also `packInt4`, `unpackInt4` and `quantizeInt4PerChannel`, and every layer against the int8 layer on the unpacked weights. Planted defects (the nibbles swapped, no sign extension, a padded tap that does not advance the weights, the depthwise taps transposed, a trailing nibble of 15, ties to even in the quantizer, the unpack threshold moved) each fail it.
- **16x8 against tinymind (C++)**: the 64-bit requantizer on 20,000 accumulators and parameters inside and outside TFLite's contract; dense, convolution, depthwise (windows above and below 256 taps) and max-pooling layers on random tensors, with a trial of small inputs and fine requantizers in each so that an error of one in a product shows; the int16 addition calibrated as TFLite's and with random parameters; the int16 table over all 65,536 inputs on random tables and on built sigmoid and tanh tables; and the int16 softmax on rows of 1, 2, 10 and 300 values. A unit test checks the softmax's constant tables against the builder. Planted defects (the accumulator or shift not clamped, the multiplier truncated instead of rounded to Q0.15, a rounding term off by one, the slope not wrapped, the lookup not rounded, the reciprocal recentred off by one, a depthwise tap off by one, a product off by one, the padding sides swapped, the addition's left shift 16, a table without its midpoint correction) each fail it.
- **Log-mel front end against tinymind (C++)**: `QLogMelSpectrogram`, `QMfcc` and the window, filter and DCT builders for frame lengths 4, 16, 64, 128, 256, 512 and 1024, band counts that leave some bands empty, output formats Q0, Q8 and Q14 and log floors from -32768 to 32767: ten integer test signals, from silence and a one-LSB impulse to full-scale rails, with pre-emphasis off and on, through the built tables and through random ones. Planted defects (a block scale one bit off, the pre-emphasis reading the current sample, no interpolation in the log table, the log truncated instead of rounded, the band weights not advancing, the floor not applied, a symmetric window built as periodic, the DCT's first row scaled like the others, ...) each fail it.
- **Anomaly scores against tinymind (C++)**: `QKMeansAnomaly`, `QKMeansAnomalyTrainer`, `QDiagGaussianAnomaly` and `buildQDiagGaussianAnomaly` on random tables (distances past 2^32 included) and on clustered sample streams with different step caps, with feature weighting on and off, a stream where every distance ties and 600 features. Planted defects (division rounding toward zero, the large-distance form dropped, feature variance counted from the seventh sample instead of the eighth, the radius floored instead of ceiled, ties going to the later cluster, the weight floor at 0, the cap off by one, ...) each fail it.
- **No reachable panics**: the int8 path is part of the Cortex-M firmware image that CI links with an undefined panic handler.

Where the C++ would hit undefined behaviour (an int32 accumulator overflowing, a shift outside its documented range), the Rust port wraps or clamps instead, so every function returns a value; inside the documented ranges the results are identical.

## Not yet

The rest of tinymind (C++)'s int8 family (attention, positional encoding, state-space and liquid-time cells, mixture of experts) is not ported yet, the importers do not emit the GRU cell (PyTorch's `nn.GRU` does not map onto it) or stacked or bidirectional LSTMs, and the PyTorch importer does not take branches. For int8 TensorFlow Lite models, `tools/import_tflite` translates convolutional networks into a Rust module the same way tinymind (C++)'s MLPerf benchmark translates them into C++; it runs the MLPerf Tiny keyword-spotting model and the residual ResNet-8 with tinymind (C++)'s output hashes ([On-board results]({{ site.baseurl }}/benchmarks)). Decision trees and gradient-boosted ensembles, with their own importer, are on the [Decision trees]({{ site.baseurl }}/trees) page.

The [API reference]({{ site.baseurl }}/api/tinymind/int8/) documents every function.
