# Changelog

All notable changes to the `tinymind` crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version bump may
break the API).

## [Unreleased]

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
reachable panics; the minimum supported Rust version is 1.85.

[Unreleased]: https://github.com/danmcleran/tinymind-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/danmcleran/tinymind-rs/releases/tag/v0.1.0
