# Changelog

All notable changes to the `tinymind` crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version bump may
break the API).

## [Unreleased]

## [0.1.3] - 2026-11-07

### Added

- `NeuralNetwork` and `Dense`: trainable networks with any number of hidden
  layers (tinymind's `NeuralNetwork<..., HiddenLayers<...>, ...>`), written as a
  tuple of 1 to 8 `Dense` layers whose sizes the compiler checks. Each layer
  has its own activation. Training is SGD with momentum, bit-identical to
  tinymind's network with the same initial weights, and to
  `MultilayerPerceptron` for one hidden layer.
- `Linear`: the identity activation, for a regression output layer.
- Gradient clipping by value (`set_gradient_clip` on `MultilayerPerceptron` and
  `NeuralNetwork`, tinymind's `GradientClipByValue`). As in tinymind, bias
  weights are never clipped. It adds 12 bytes to a trainable Q16.16 network
  (6 in Q8.8) whether or not it is used.
- `set_acceleration_rate` on both trainable networks.
- Example `energy_efficiency`: a regression on the UCI Energy Efficiency
  dataset (ReLU hidden layer, linear output, clipped gradients), byte-identical
  to tinymind's example. The dataset is in `data/` under CC BY 4.0.

## [0.1.2] - 2026-11-07

### Added

- `InferencePerceptron`: an inference-only multilayer perceptron (tinymind's
  `IsTrainable = false`) that holds only its weights, so a trained network can be
  a `static` in flash. Its outputs are bit-identical to `MultilayerPerceptron`'s
  with the same weights and to tinymind's untrainable network. A Q16.16 2-4-1
  network is 68 bytes instead of 372.
- `MultilayerPerceptron::to_inference`, `set_weights` and per-matrix weight
  setters (`set_input_weights`, `set_input_bias_weights`, `set_hidden_weights`,
  `set_hidden_bias_weights`), to ship trained weights and load them back.

## [0.1.1] - 2026-11-07

### Added

- `int8::QSoftmax`: int8 softmax over a row of logits, bit-exact with TFLite's
  int8 softmax (the TFLite Micro reference kernel, which CMSIS-NN's
  `arm_softmax_s8` also reproduces), in gemmlowp fixed point; integer only, no
  table. `int8::calibration::build_softmax` (feature `std`) computes its
  parameters from the input scale and beta as TFLite Micro does.
- PyTorch importer (`tools/import_pytorch` in the repository, not part of the
  crate): turns a trained `torch.nn.Sequential` of Linear / ReLU / Sigmoid / Tanh /
  Softmax into a `no_std` Rust module with int8 weights, integer requantizers, baked
  activation tables and a `forward` function. Its quantization is identical to
  tinymind's importer, and an imported model runs bit-exact with the same model
  on tinymind's C++ runtime.
- Example `imported_mlp` (feature `std`): two models written by the importer,
  an MLP and a softmax classifier.

## [0.1.0] - 2026-11-07

First release, a port of the parts of [tinymind](https://github.com/danmcleran/tinymind)
its XOR, iris and int8 XOR examples use, bit-exact with the C++ build.

### Added

- `QValue<S, FRAC>`: signed fixed-point values with `i16` or `i32` storage, with
  tinymind's default Truncate and Wrap policies; aliases `Q16_16` and `Q8_8`.
- Activations: `Relu` (every value type), and `Tanh` and `Sigmoid` via
  tinymind's lookup tables with linear interpolation (`Q16_16`, `Q8_8`).
- `MultilayerPerceptron`: a trainable network with one hidden layer, trained by
  SGD with momentum, stored in fixed-size arrays.
- `int8`: int8 affine-quantized inference in the TFLite / CMSIS-NN model, pure
  integer at run time: `Requantizer` (gemmlowp rounding), the dense layer
  `QDense`, `QDensePerChannel` (one requantizer per output, TFLite's default),
  `qrelu`, `qrelu6` and 256-entry activation tables.
- `int8::calibration` (feature `std`): host-side helpers that turn float scales
  into requantizers and sigmoid/tanh tables, and quantize/dequantize values.
- `rand::GlibcRand`: glibc's `srand`/`rand`, bit for bit, for reproducing the
  C++ examples.
- Examples `xor` (Q16.16 and Q8.8), `iris` and `int8_xor` (a PyTorch-trained
  int8 network), whose output is byte-identical to tinymind's C++ examples.

The library is `no_std`, allocation-free, `#![forbid(unsafe_code)]`, and has no
reachable panics; the minimum supported Rust version is 1.85. It has been run on
an STM32F411 (Cortex-M4F) and an STM32L031 (Cortex-M0+), where int8 inference and
on-device training match tinymind's C++ bit for bit (see `bench/README.md` in the
repository).

[Unreleased]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/danmcleran/tinymind-rs/releases/tag/v0.1.0
