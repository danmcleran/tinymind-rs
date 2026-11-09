# Changelog

All notable changes to the `tinymind` crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version bump may
break the API).

## [Unreleased]

### Added

- Int8 convolution and pooling (tinymind (C++)'s `cpp/qconv2d.hpp`,
  `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp` and `cpp/qpool2d.hpp`),
  in `tinymind::int8`: `QConv2D` and `QConv2DPerChannel`, `QDepthwiseConv2D`,
  `QPointwiseConv2D` and `QPointwiseConv2DPerChannel`, `QMaxPool2D`,
  `QAvgPool2D` and `QGlobalAvgPool2D`. NHWC tensors are nested arrays; strides
  and zero-point padding (SAME or any asymmetric amount) are const parameters,
  and an output size that does not match the input, kernel, strides and padding
  is a compile error. Bit-identical to tinymind (C++), checked over 57 random
  layer configurations.
- `int8::calibration::per_channel_symmetric_scales` (tinymind (C++)'s
  `computePerChannelSymmetricScales`), `affine_params_symmetric`
  (`computeAffineParamsSymmetric`) and `RangeObserver`, for fitting activation
  scales from calibration inputs.
- The PyTorch importer (`tools/import_pytorch`, not part of the crate) writes
  convolutional networks: `Conv2D` (stride, padding, depthwise), max, average
  and global average pooling, and flatten, from hand-written descriptors or from
  `nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d(1)` and
  `nn.Flatten` through `from_torch(model, input_shape=...)`. Example
  `imported_cnn` (feature `std`) runs two such models, byte-identical to
  tinymind (C++)'s importer and runtime on the same models.
- Example `kws_cortex_m_int8` (feature `std`), a keyword-spotting shaped int8
  network calibrated from a float reference pass: every tensor and its logits
  report byte-identical to tinymind (C++)'s example.

## [0.1.5] - 2026-11-08

### Added

- Decision trees (`tinymind::tree`, tinymind (C++)'s `cpp/qtree.hpp`): `QTreeNode`,
  `walk_tree`, a single tree `QDecisionTree` and the gradient-boosted ensemble
  `QGbdt` (tinymind (C++)'s `QGBDT`), over a borrowed node pool that can live in
  flash, with `i8`, `u8` or `i16` thresholds and inputs. Bit-identical to
  tinymind (C++) for well-formed models; a malformed one (an index outside the model)
  gives a defined result instead of an out-of-bounds read, and `is_valid` is a
  `const fn` for checking a model at compile time.
- Example `gbdt_tabular_int8` (feature `std`), byte-identical to tinymind (C++)'s.
- `int8::calibration::affine_params_asymmetric` (tinymind (C++)'s
  `computeAffineParamsAsymmetric`).
- A tree importer in the repository (`tools/import_trees`, not part of the
  crate): scikit-learn, XGBoost and LightGBM tree models to a `no_std` Rust
  module with a pool-walk form and a nested-`if` form. It produces the same
  integer trees as tinymind (C++)'s importer. Example `imported_trees` runs two
  scikit-learn classifiers for the UCI Wine data, byte-identical to the tinymind (C++)
  runtime running the same models.

## [0.1.4] - 2026-11-07

### Added

- Optimizers for both trainable networks (`tinymind::optimizer`): `Adam` and
  `RmsProp` (tinymind (C++)'s `AdamOptimizer` and `RmsPropOptimizer`), alongside the
  default `Sgd`. The optimizer is a type parameter that defaults to `Sgd`, so
  existing code and network sizes are unchanged; pick another with
  `with_optimizer`. As in tinymind (C++), the step count for Adam's bias correction
  advances once per layer update, and the adaptive optimizers leave the bias
  weights unchanged. The sealed `Optimizer` trait names them; `optimizer()`
  returns a network's optimizer, and `Adam::timestep()` its step count. Neither
  `Adam` nor `RmsProp` implements `Default`: tinymind (C++)'s default rates are raw
  values tuned for Q8.8, so pass rates suited to your format to `new`.
- Example `anomaly`: an 8-4-8 autoencoder trained on the device on synthetic
  normal vibration windows flags faults by reconstruction error (0 of 256 false
  alarms; frequency, harmonic and impulse faults all caught; a stalled sensor
  half the time). The same code runs on an STM32F411 and an STM32L031, where it
  trains and decides bit for bit as on the host.
- `Sqrt`: fixed-point square root (tinymind (C++)'s `SquareRootApproximation`), the
  floor of the exact root.

### Fixed

- `MultilayerPerceptron`'s forward pass could keep an unreachable panic path (a
  bounds check) when the compiler did not inline enough to see it was
  unreachable, as at opt-level `s` and `z` for some network shapes. It is now
  total; results are unchanged.
- `int8::calibration::quantize_multiplier` keeps the shift in `-31..=31`, the
  requantizer's range, as tinymind (C++) now does: a ratio below 2^-32 gives
  `(0, 0)` and a ratio of 2^31 or more gets shift -31. Before, such a tiny ratio
  could requantize an accumulator near `i32::MIN` to one below the zero point
  instead of to the zero point. Ratios from real models are unaffected. The
  PyTorch importer's copy does the same.

## [0.1.3] - 2026-11-07

### Added

- `NeuralNetwork` and `Dense`: trainable networks with any number of hidden
  layers (tinymind (C++)'s `NeuralNetwork<..., HiddenLayers<...>, ...>`), written as a
  tuple of 1 to 8 `Dense` layers whose sizes the compiler checks. Each layer
  has its own activation. Training is SGD with momentum, bit-identical to
  tinymind (C++)'s network with the same initial weights, and to
  `MultilayerPerceptron` for one hidden layer.
- `Linear`: the identity activation, for a regression output layer.
- Gradient clipping by value (`set_gradient_clip` on `MultilayerPerceptron` and
  `NeuralNetwork`, tinymind (C++)'s `GradientClipByValue`). As in tinymind (C++), bias
  weights are never clipped. It adds 12 bytes to a trainable Q16.16 network
  (6 in Q8.8) whether or not it is used.
- `set_acceleration_rate` on both trainable networks.
- Example `energy_efficiency`: a regression on the UCI Energy Efficiency
  dataset (ReLU hidden layer, linear output, clipped gradients), byte-identical
  to tinymind (C++)'s example. The dataset is in `data/` under CC BY 4.0.

## [0.1.2] - 2026-11-07

### Added

- `InferencePerceptron`: an inference-only multilayer perceptron (tinymind (C++)'s
  `IsTrainable = false`) that holds only its weights, so a trained network can be
  a `static` in flash. Its outputs are bit-identical to `MultilayerPerceptron`'s
  with the same weights and to tinymind (C++)'s untrainable network. A Q16.16 2-4-1
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
  tinymind (C++)'s importer, and an imported model runs bit-exact with the same model
  on the tinymind (C++) runtime.
- Example `imported_mlp` (feature `std`): two models written by the importer,
  an MLP and a softmax classifier.

## [0.1.0] - 2026-11-07

First release, a port of the parts of [tinymind (C++)](https://github.com/danmcleran/tinymind)
its XOR, iris and int8 XOR examples use, bit-exact with the C++ build.

### Added

- `QValue<S, FRAC>`: signed fixed-point values with `i16` or `i32` storage, with
  tinymind (C++)'s default Truncate and Wrap policies; aliases `Q16_16` and `Q8_8`.
- Activations: `Relu` (every value type), and `Tanh` and `Sigmoid` via
  tinymind (C++)'s lookup tables with linear interpolation (`Q16_16`, `Q8_8`).
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
  int8 network), whose output is byte-identical to the tinymind (C++) examples.

The library is `no_std`, allocation-free, `#![forbid(unsafe_code)]`, and has no
reachable panics; the minimum supported Rust version is 1.85. It has been run on
an STM32F411 (Cortex-M4F) and an STM32L031 (Cortex-M0+), where int8 inference and
on-device training match tinymind (C++) bit for bit (see `bench/README.md` in the
repository).

[Unreleased]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/danmcleran/tinymind-rs/releases/tag/v0.1.0
