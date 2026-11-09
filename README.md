# tinymind-rs

[![crates.io](https://img.shields.io/crates/v/tinymind.svg)](https://crates.io/crates/tinymind)
[![docs.rs](https://docs.rs/tinymind/badge.svg)](https://docs.rs/tinymind)
[![CI](https://github.com/danmcleran/tinymind-rs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/danmcleran/tinymind-rs/actions/workflows/ci.yml)

tinymind-rs (crate `tinymind`) brings [tinymind (C++)](https://github.com/danmcleran/tinymind),
a header-only C++ library for neural networks on embedded systems, to Rust. Like
tinymind (C++), it is free, open source software under the MIT license. The library is
`no_std` and allocation-free, and every ported path is verified bit-for-bit
against the C++ build: the XOR example's weights, training log and decision
surface are byte-identical to the C++ output.

Documentation, including the API reference: <https://danmcleran.github.io/tinymind-rs/>

## Status

Early. So far the port covers Q16.16 and Q8.8 arithmetic, ReLU, linear and
lookup-table tanh and sigmoid activations, trainable networks with any number of
hidden layers (SGD with momentum, Adam or RMSprop, optional gradient clipping), an inference-only
one-hidden-layer network that is just its weights, int8
affine-quantized inference (the TFLite/CMSIS-NN model: requantizer, dense layer,
2D convolution (standard, depthwise and pointwise), pooling, quantized ReLU,
activation tables, and a softmax that is bit-exact with TFLite's), and quantized decision trees and gradient-boosted ensembles. The API is not stable yet: before 1.0, a minor version may break it.

The library is `no_std` by default. The `std` feature adds host-side calibration
helpers (`int8::calibration`) that turn float scales into the integer parameters
the int8 layers use.

The minimum supported Rust version is 1.85. CI builds the library for Cortex-M4F
(`thumbv7em-none-eabihf`) and Cortex-M0+ (`thumbv6m-none-eabi`).

It has also run on two boards, an STM32F411 (Cortex-M4F, 100 MHz) and an
STM32L031 (Cortex-M0+, 32 MHz, 32 KB flash, 8 KB RAM). On both, tinymind (C++)'s int8
digits classifier gives the same output hash as the C++ (214 µs per inference on
the M4F, 1.33 ms on the M0+), so does its KWS-shaped convolutional network, in
13-26% fewer cycles than the C++'s portable build (1.90 ms and 20.4 ms), XOR trained on the board reproduces the desktop
weight trajectory bit for bit, and the anomaly example trains on the chip in
0.14 s and 4.2 s respectively. Results and method:
[On-board results](https://danmcleran.github.io/tinymind-rs/benchmarks) and
`bench/`.

## Use it

```toml
[dependencies]
tinymind = "0.1.5"
# For host-side int8 calibration (float scales to requantizers and tables):
# tinymind = { version = "0.1.5", features = ["std"] }
```

The API reference is on [docs.rs](https://docs.rs/tinymind).

To run a model trained in PyTorch, `tools/import_pytorch` in this repository
turns it into a Rust module (int8 weights, integer requantizers and a `forward`
function; `no_std`). See its README. `tools/import_trees` does the same for
scikit-learn, XGBoost and LightGBM tree models.

## Quick start

```bash
cargo run --release --example xor -- output                  # Q16.16, output files in ./output
cargo run --release --example xor -- --format q8.8 output    # the same network in Q8.8
cargo run --release --example iris -- output                 # iris species classifier (ReLU hidden layer)
cargo run --release --example energy_efficiency -- output    # energy-load regression (linear output, clipped gradients)
cargo run --release --example anomaly -- output              # anomaly detection: an autoencoder trained on the device
cargo run --release --features std --example int8_xor -- output   # PyTorch-trained int8 network
cargo run --release --features std --example imported_mlp    # two models written by the PyTorch importer
cargo run --release --features std --example gbdt_tabular_int8 -- output   # int8 gradient-boosted trees
cargo run --release --features std --example kws_cortex_m_int8   # keyword-spotting shaped int8 CNN, per-layer report
cargo run --release --example imported_trees                 # scikit-learn tree models via the tree importer
cargo test --all-features                                # unit, doc and golden tests against the C++ output
PYTHON=<python with numpy and scikit-learn> scripts/compare-with-cpp.sh   # rebuild the C++ side and diff everything
```

The `xor` example takes an optional `--format q16.16|q8.8` (default Q16.16), an
optional output directory (default `output`), then an optional learning rate and
momentum. It writes `nn_fixed_xor.txt`, `xor_training.csv` and
`xor_decision_surface.csv` there and prints a summary on stdout. The `iris` example
takes an optional output directory and data file (default `data/iris.data`, the
UCI Iris dataset under CC BY 4.0; see `NOTICE`) and writes `iris_loss.csv`,
`iris_scatter.csv` and `iris_confusion.csv`. The `energy_efficiency` example
reads `data/ENB2012_data.csv` (the UCI Energy Efficiency dataset, CC BY 4.0) the
same way and writes `energy_loss.csv` and `energy_pred.csv`. The `anomaly`
example trains an 8-4-8 autoencoder on synthetic normal vibration windows and
flags faults by reconstruction error; the same code runs on the boards in
`bench/`. It takes an optional output directory and writes `anomaly_scores.csv`
there. The `int8_xor` example (feature
`std`) runs a PyTorch-trained, int8-quantized 2-4-1 network and writes
`xor_decision_surface.csv`. The `gbdt_tabular_int8` example (feature `std`)
quantizes a five-tree ensemble onto an int8 feature grid, compares it with the
float trees over a 48x48 grid and writes `gbdt_tabular_int8.csv` (the predicted
class at each point); `--golden` before the directory prints tinymind (C++)'s compact
golden report instead. The `kws_cortex_m_int8` example (feature `std`) runs a
keyword-spotting shaped int8 network (convolution, max pooling, depthwise and
pointwise convolution, global average pooling, a dense head) with synthetic
weights and prints each layer's size and time. The `imported_trees` example runs two scikit-learn
classifiers for the UCI Wine data (CC BY 4.0), written by the tree importer, on
the held-out samples and on random inputs.

`scripts/compare-with-cpp.sh` needs a tinymind (C++) checkout, `g++`, and a Python with
numpy and scikit-learn for the importer checks (plus XGBoost and LightGBM for
all of the tree importer's) (set `PYTHON`, for example to a virtual
environment's interpreter). It looks in `../tinymind` unless `TINYMIND_DIR` is
set.

## What is ported

| Rust | tinymind (C++) |
| --- | --- |
| `src/qformat.rs` | `cpp/qformat.hpp`: `QValue`, Truncate and Wrap policies, `i32` and `i16` storage (`Q16_16`, `Q8_8`); `Sqrt` (`SquareRootApproximation` from `cpp/adam.hpp`) |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp`: ReLU, linear, tanh, sigmoid |
| `src/lut.rs` | `cpp/lookupTables.cpp`: Q16.16 and Q8.8 tanh and sigmoid tables, generated by `scripts/extract-luts.py` |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp`, `cpp/gradientClipping.hpp`: one-hidden-layer MLP (`MultilayerPerceptron`), SGD with momentum, Adam or RMSprop, optional gradient clipping, batch size 1; and its inference-only form (`InferencePerceptron`, `IsTrainable = false`) |
| `src/network.rs` | `cpp/neuralnet.hpp` (`NeuralNetwork<..., HiddenLayers<...>>`): any number of hidden layers as a tuple of `Dense` layers, same training |
| `src/optimizer.rs` | `cpp/adam.hpp`, `cpp/rmsprop.hpp`: the `Sgd` (default), `Adam` and `RmsProp` optimizers for both trainable networks |
| `src/rand.rs` | a re-implementation of glibc `rand()`, which the C++ examples use for weight init and sampling |
| `examples/xor.rs` | `examples/xor/xor.cpp` |
| `examples/iris.rs` | `examples/iris/iris.cpp` |
| `examples/energy_efficiency.rs` | `examples/energy_efficiency/energy_efficiency.cpp` |
| `examples/anomaly.rs` | no counterpart: an on-device anomaly-detection demo built from the ported parts |
| `src/int8.rs` | `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qactivations.hpp`, `cpp/qsoftmax_fixedpoint.hpp`: requantizer, int8 dense layer (per-tensor and per-channel), quantized ReLU, activation tables, softmax |
| `src/int8/conv.rs` | `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`: int8 2D convolution (per-tensor and per-channel), depthwise and pointwise convolution, with zero-point padding |
| `src/int8/pool.rs` | `cpp/qpool2d.hpp`: int8 max, average and global average pooling |
| `src/int8/calibration.rs` | `cpp/include/qcalibration.hpp`, `cpp/qactivations.hpp`: host-side calibration (feature `std`) |
| `src/tree.rs` | `cpp/qtree.hpp`: quantized decision tree (`QDecisionTree`) and gradient-boosted ensemble (`QGbdt`) over a shared node pool; not the compile-time `QStatic*` forms |
| `examples/gbdt_tabular_int8.rs` | `examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp` |
| `examples/int8_xor.rs` | `examples/pytorch_quant/xor/xor_quant.cpp` |
| `examples/kws_cortex_m_int8.rs` | `examples/kws_cortex_m_int8/kws_cortex_m_int8.cpp` |
| `tools/import_pytorch/` | `apps/import_pytorch/tinymind_import.py`: the PyTorch importer, emitting Rust |
| `tools/import_trees/` | `apps/import_trees/tinymind_import_trees.py`: the scikit-learn / XGBoost / LightGBM tree importer, emitting Rust |
| `examples/imported_trees.rs` | two models written by that importer, checked against tinymind (C++)'s importer + C++ runtime |
| `examples/imported_mlp.rs` | two models written by the PyTorch importer, checked against tinymind (C++)'s importer + C++ runtime |

Everything else in tinymind (C++) is not ported: floating-point networks, other Q
formats, unsigned and 8/64-bit storage, the other activations, recurrent layers,
the int8 quantization tiers, SIMD backends, and so on.

## How equivalence is proven

- `tests/xor_golden.rs` runs the `xor` example in each format and compares its
  stdout and the two CSV files byte-for-byte with the goldens in
  `tests/golden/<format>/`. The 2.4-2.6 MB `nn_fixed_xor.txt` is checked by
  length, line count and a 64-bit FNV-1a hash instead of being stored. The goldens
  come from the C++ build: Q16.16 from the example as shipped, Q8.8 from the same
  example with both bit counts in `xornet.h` set to 8. `tests/iris_golden.rs`
  does the same for iris against `tests/golden/iris/`, `tests/energy_golden.rs`
  for energy_efficiency against `tests/golden/energy/`, and
  `tests/int8_xor_golden.rs` for the int8 example against `tests/golden/int8_xor/`,
  and `tests/imported_mlp_golden.rs` for the PyTorch importer's models. `TINYMIND_REF`
  records the tinymind (C++) commit they were last verified against. The int8, GBDT and
  PyTorch-importer tests need the `std` feature: run `cargo test --all-features`.
- `tests/deep_mlp.rs` checks networks with one, two and three hidden layers
  against tinymind (C++)'s `NeuralNetwork` (every weight after every training step,
  both formats), trained with SGD, Adam and RMSprop, with and without gradient
  clipping.
- `tests/anomaly_golden.rs` pins the anomaly example's output and hash. The
  demo has no tinymind (C++) counterpart, so this golden is the crate's own output, a
  regression check; its parts are each checked against tinymind (C++) above.
- `tests/inference_mlp.rs` checks the inference-only perceptron against
  tinymind (C++)'s untrainable network (`IsTrainable = false`) with the same weights,
  in both formats and two shapes.
- `tests/softmax_tflite.rs` checks the int8 softmax against the TFLite Micro
  reference kernel's outputs (tinymind (C++)'s unit-test goldens and sweep hashes), and
  `tests/softmax_sweep.rs` against the tinymind (C++) softmax and its calibration
  over 504 calibrations and row lengths 1 to 511.
- `tests/kws_int8_golden.rs` checks every tensor of the `kws_cortex_m_int8`
  example (quantized input, weights, requantizers and each layer's output)
  against `tests/golden/kws_int8/`, printed by tinymind (C++)'s example compiled
  unchanged (`scripts/cpp/kws_int8_dump.cpp`). The example's own stdout is not
  compared: it reports host timings and each build's sizes.
- `tests/conv_sweep.rs` checks the int8 convolution and pooling layers against
  tinymind (C++)'s over random tensors: 13 convolution shapes (per tensor and per
  channel), 7 depthwise, 5 pointwise (both), 5 pooling windows (max and average)
  and 4 global pools, with padding on every side, strides above the kernel size
  and requantizers on both the fused and the two-step path.
- `tests/gbdt_golden.rs` checks the GBDT example's three outputs against
  `tests/golden/gbdt/`, and `tests/tree_sweep.rs` checks the trees against
  tinymind (C++)'s `QDecisionTree` and `QGBDT` over 72 random ensembles (int8, uint8
  and int16 thresholds, ties, inputs on the thresholds).
  `tests/imported_trees_golden.rs` checks the imported-trees example against
  the tinymind (C++) runtime running the same models, imported with the
  tinymind (C++) importer.
- `scripts/compare-with-cpp.sh` builds the C++ examples from a tinymind (C++) checkout
  (xor in both formats, iris, energy efficiency, int8 xor, GBDT) with the flags their
  Makefiles use, plus the C++ programs in `scripts/cpp/` that exercise what no
  example does, runs both sides, and compares every output file (44 checks today, including
  that `data/iris.data` and `data/ENB2012_data.csv` are still tinymind (C++)'s files). It exits nonzero if anything differs.
- `tests/glibc_rand.rs` checks the crate's glibc `rand()` re-implementation
  against the real `srand`/`rand` (through FFI, Linux only) for 212 seeds; the
  examples draw their weights and samples from it, as tinymind (C++)'s do.
- `tests/qformat_oracle.rs`, ported from tinymind (C++)'s qformat oracle, checks every
  `QValue` operation against an exact reference: edge values and a fixed-seed
  sample for nine formats, and every operand pair for Q8.8 (about 26 billion
  checks, run in CI with overflow checks on). A nightly job reruns the sample with
  a fresh seed.
- Kani proofs (`src/kani_proofs.rs`) check, for every input, that the
  `QValue` operations match their definitions (division: that it never panics),
  that tanh and sigmoid stay within their tables' range, that the int8
  requantizer's fast path equals gemmlowp's two-step rounding, and that the
  requantizer, the int8 dense layers, convolutions and pooling, the tables, the
  softmax and the trees never panic, and that pooling's divide rounds half away
  from zero.
- `ci/panic-never` is a Cortex-M firmware image that calls every public function
  and whose panic handler is an undefined symbol, so CI fails to link it if any
  panic path survives optimisation (M0+ and M4F, three optimisation levels).
- `scripts/extract-luts.py` generates `src/lut.rs` from tinymind (C++)'s
  `cpp/lookupTables.cpp`; CI regenerates it and fails if the tables differ.
- CI runs all of these on every pull request (the `cpp-equivalence` job checks out
  tinymind (C++) at `TINYMIND_REF` for the compare script and the table check), and
  branch protection blocks merging unless they pass.

## License

MIT. See `LICENSE` and `NOTICE`.
