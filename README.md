# tinymind-rs

[![crates.io](https://img.shields.io/crates/v/tinymind.svg)](https://crates.io/crates/tinymind)
[![docs.rs](https://docs.rs/tinymind/badge.svg)](https://docs.rs/tinymind)
[![CI](https://github.com/danmcleran/tinymind-rs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/danmcleran/tinymind-rs/actions/workflows/ci.yml)

tinymind-rs (crate `tinymind`) brings [tinymind (C++)](https://github.com/danmcleran/tinymind),
a header-only C++ library for neural networks on embedded systems, to Rust. Like
tinymind (C++), it is free, open source software under the MIT license. The library is
`no_std` and allocation-free, and every ported path is verified bit-for-bit
against the C++ build (floating point: with `std` on a glibc host): the XOR example's weights, training log and decision
surface are byte-identical to the C++ output.

Documentation, including the API reference: <https://danmcleran.github.io/tinymind-rs/>

## Status

The port covers Q16.16 and Q8.8 arithmetic, ReLU, linear and
lookup-table tanh and sigmoid activations, floating-point (`f32`, `f64`) networks
trained with SGD or tinymind (C++)'s floating-point Adam and RMSprop (tanh,
sigmoid, `pow` and `sqrt` from the math library: feature `std`, or `libm`
without std), trainable networks with up to seven hidden layers (SGD with
momentum, Adam or RMSprop, optional gradient clipping) and inference-only
networks of the same depths that are just their weights, int8
affine-quantized inference (the TFLite/CMSIS-NN model: requantizer, dense layer,
2D convolution (standard, depthwise and pointwise), pooling, elementwise addition
(for residual connections) and multiplication, padding, concatenation, quantized
ReLU, activation tables, and a softmax that is bit-exact with TFLite's), quantized decision trees and gradient-boosted ensembles, tabular Q-learning, and LSTM recurrent networks (in Q16.16, Q8.8, `f32` or `f64`). It is a partial port of tinymind (C++), and every ported path is checked bit for bit against it. The API is not stable yet: before 1.0, a minor version may break it.

The library is `no_std` by default. The `std` feature adds host-side calibration
helpers (`int8::calibration`) that turn float scales into the integer parameters
the int8 layers use, and the math (`tanh`, `exp`, `pow`, `sqrt`) of the
floating-point activations and optimizers; the optional `libm` feature provides
that math without `std`, from the `libm` crate (not bit-identical to glibc).

The minimum supported Rust version is 1.85. CI builds the library for Cortex-M4F
(`thumbv7em-none-eabihf`) and Cortex-M0+ (`thumbv6m-none-eabi`).

It has also run on two boards, an STM32F411 (Cortex-M4F, 100 MHz) and an
STM32L031 (Cortex-M0+, 32 MHz, 32 KB flash, 8 KB RAM). On both, tinymind (C++)'s int8
digits classifier gives the same output hash as the C++ (215 µs per inference on
the M4F, 1.33 ms on the M0+), so does its KWS-shaped convolutional network, in
13-27% fewer cycles than the C++'s portable build (1.86 ms and 20.3 ms), and on
the F411 the MLPerf Tiny keyword-spotting and image-classification (ResNet-8)
models, translated from TFLite, give the C++'s output hashes in 132 ms and 644
ms (16% and 25% fewer cycles than the C++'s portable build), XOR trained on the board reproduces the desktop
weight trajectory bit for bit, and the anomaly example trains on the chip in
0.21 s and 4.6 s respectively. Results and method:
[On-board results](https://danmcleran.github.io/tinymind-rs/benchmarks) and
`bench/`.

## Use it

```toml
[dependencies]
tinymind = "0.3.0"
# Host-side int8 calibration, and f32/f64 tanh, sigmoid, AdamFloat, RmsPropFloat:
# tinymind = { version = "0.3.0", features = ["std"] }
# The same float math without std (for example on a Cortex-M4F):
# tinymind = { version = "0.3.0", features = ["libm"] }
```

The API reference is on [docs.rs](https://docs.rs/tinymind).

To run a model trained in PyTorch, `tools/import_pytorch` in this repository
turns it into a Rust module (int8 weights, integer requantizers and a `forward`
function; `no_std`). See its README. `tools/import_trees` does the same for
scikit-learn, XGBoost and LightGBM tree models, and `tools/import_tflite` for
int8 TensorFlow Lite models (convolutional and residual networks such as the
MLPerf Tiny keyword spotter and ResNet-8).

## Quick start

```bash
cargo run --release --example xor -- output                  # Q16.16, output files in ./output
cargo run --release --example xor -- --format q8.8 output    # the same network in Q8.8
cargo run --release --example iris -- output                 # iris species classifier (ReLU hidden layer)
cargo run --release --example energy_efficiency -- output    # energy-load regression (linear output, clipped gradients)
cargo run --release --example anomaly -- output              # anomaly detection: an autoencoder trained on the device
cargo run --release --features std --example int8_xor -- output   # PyTorch-trained int8 network
cargo run --release --features std --example imported_mlp    # two models written by the PyTorch importer
cargo run --release --features std --example imported_cnn    # two convolutional models written by the PyTorch importer
cargo run --release --features std --example gbdt_tabular_int8 -- output   # int8 gradient-boosted trees
cargo run --release --features std --example kws_cortex_m_int8   # keyword-spotting shaped int8 CNN, per-layer report
cargo run --release --example imported_trees                 # scikit-learn tree models via the tree importer
cargo run --release --example maze -- output                 # Q-learning: find the way out of a six-room maze
cargo run --release --example har_activity -- output         # LSTM: classify accelerometer windows into four activities
cargo test --all-features                                # unit, doc and golden tests against the C++ output
PYTHON=<python with the packages below> scripts/compare-with-cpp.sh   # rebuild the C++ side and diff everything
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
there. The `maze` example trains a Q-learner to leave a six-room maze over 5000
episodes, compares its Q table with the optimal one (`maze_qtable.csv`), then
runs 100 greedy test episodes, and writes each episode's path to
`maze_training.csv` and `maze_test.csv`. The `har_activity` example trains a
3-16-4 LSTM network on synthesized accelerometer windows (32 samples of three
axes each) to tell walking, walking upstairs, walking downstairs and sitting apart,
reading each window one sample at a time (upstairs and downstairs differ only
in the order of their samples); it writes `har_samples.csv`,
`har_loss.csv` and `har_confusion.csv`, reads `har.csv` instead of
synthesizing when the second argument names one in the C++ example's format,
and with `--readout-only` first trains only the output weights. The `int8_xor` example (feature
`std`) runs a PyTorch-trained, int8-quantized 2-4-1 network and writes
`xor_decision_surface.csv`. The `gbdt_tabular_int8` example (feature `std`)
quantizes a five-tree ensemble onto an int8 feature grid, compares it with the
float trees over a 48x48 grid and writes `gbdt_tabular_int8.csv` (the predicted
class at each point); `--golden` before the directory prints tinymind (C++)'s compact
golden report instead. The `kws_cortex_m_int8` example (feature `std`) runs a
keyword-spotting shaped int8 network (convolution, max pooling, depthwise and
pointwise convolution, global average pooling, a dense head) with synthetic
weights, fits each layer's activation scale from a float reference pass over
calibration inputs, prints each layer's size and time, and compares its logits
with the float model's. The `imported_trees` example runs two scikit-learn
classifiers for the UCI Wine data (CC BY 4.0), written by the tree importer, on
the held-out samples and on random inputs.

`scripts/compare-with-cpp.sh` needs a tinymind (C++) checkout, `g++`, and a Python with
numpy and scikit-learn for the importer checks (plus XGBoost and LightGBM for
all of the tree importer's, and `tflite==2.18.0`, `flatbuffers==25.12.19` and
`ai-edge-litert==2.3.0` for the TFLite translator's; those download the
MLPerf Tiny models, so they need network access once) (set `PYTHON`, for
example to a virtual environment's interpreter). It looks in `../tinymind` unless `TINYMIND_DIR` is
set.

## What is ported

| Rust | tinymind (C++) |
| --- | --- |
| `src/qformat.rs` | `cpp/qformat.hpp`: `QValue`, Truncate and Wrap policies, `i32` and `i16` storage (`Q16_16`, `Q8_8`); `Sqrt` (`SquareRootApproximation` from `cpp/adam.hpp`); `mul_round_toward_zero` (`multiplyRoundTowardZero`, SGD's momentum terms) and `mul_round_nearest` (`multiplyRoundNearest`, SGD's step); `Scalar`, the network value type, for the Q formats, `f32` and `f64` (formerly `Fixed`, kept as a deprecated alias) |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp`: ReLU, linear, tanh, sigmoid |
| `src/lut.rs` | `cpp/lookupTables.cpp`: Q16.16 and Q8.8 tanh and sigmoid tables, generated by `scripts/extract-luts.py` |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp`, `cpp/gradientClipping.hpp`: one-hidden-layer MLP (`MultilayerPerceptron`), SGD with momentum, Adam or RMSprop, optional gradient clipping, batch size 1; and its inference-only form (`InferencePerceptron`, `IsTrainable = false`) |
| `src/network.rs` | `cpp/neuralnet.hpp` (`NeuralNetwork<..., HiddenLayers<...>>`): up to seven hidden layers as a tuple of 1 to 8 `Dense` layers, same training; and its inference-only form (`InferenceNetwork` over `InferenceDense` layers, `IsTrainable = false`) |
| `src/optimizer.rs` | `cpp/adam.hpp`, `cpp/rmsprop.hpp`: the `Sgd` (default), `Adam` and `RmsProp` optimizers for both trainable networks; for `f32`/`f64` networks `AdamFloat` and `RmsPropFloat` (`AdamOptimizerFloat`, `RmsPropOptimizerFloat`) |
| `src/rand.rs` | a re-implementation of glibc `rand()`, which the C++ examples use for weight init and sampling |
| `examples/xor.rs` | `examples/xor/xor.cpp` |
| `examples/iris.rs` | `examples/iris/iris.cpp` |
| `examples/energy_efficiency.rs` | `examples/energy_efficiency/energy_efficiency.cpp` |
| `src/qlearn.rs` | `cpp/qlearn.hpp`: tabular Q-learning (`QLearner` with a `QTable`, tinymind (C++)'s `QValueTablePolicy`, `ArgMaxPolicy` and `DefaultLearningPolicy`); not the deep Q-network policy |
| `examples/maze.rs` | `examples/maze/maze.cpp`, `mazelearner.h` |
| `src/recurrent.rs` | `cpp/neuralnet.hpp` (`LstmNeuralNetwork`): an LSTM network with one hidden layer (`LstmNetwork`), trained one step back in time with SGD with momentum, Adam or RMSprop and optional gradient clipping |
| `examples/har_activity.rs` | `examples/har_activity/har_activity.cpp` |
| `examples/anomaly.rs` | no counterpart: an on-device anomaly-detection demo built from the ported parts |
| `src/int8.rs` | `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qactivations.hpp`, `cpp/qsoftmax_fixedpoint.hpp`: requantizer, int8 dense layer (per-tensor and per-channel), quantized ReLU, activation tables, softmax |
| `src/int8/conv.rs` | `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`: int8 2D convolution (per-tensor and per-channel), depthwise and pointwise convolution, with zero-point padding |
| `src/int8/pool.rs` | `cpp/qpool2d.hpp`: int8 max, average and global average pooling |
| `src/int8/add.rs` | `cpp/qadd.hpp`: int8 elementwise addition (TFLite ADD), for residual connections |
| `src/int8/compose.rs` | `cpp/qmul.hpp`, `cpp/qpad.hpp`, `cpp/qconcat.hpp`: int8 elementwise multiplication (TFLite MUL), explicit padding (PAD) and channel concatenation (CONCATENATION) |
| `src/int8/calibration.rs` | `cpp/include/qcalibration.hpp`, `cpp/qactivations.hpp`: host-side calibration (feature `std`) |
| `src/tree.rs` | `cpp/qtree.hpp`: quantized decision tree (`QDecisionTree`) and gradient-boosted ensemble (`QGbdt`) over a shared node pool; not the compile-time `QStatic*` forms |
| `examples/gbdt_tabular_int8.rs` | `examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp` |
| `examples/int8_xor.rs` | `examples/pytorch_quant/xor/xor_quant.cpp` |
| `examples/kws_cortex_m_int8.rs` | `examples/kws_cortex_m_int8/kws_cortex_m_int8.cpp` |
| `tools/import_pytorch/` | `apps/import_pytorch/tinymind_import.py`: the PyTorch importer, emitting Rust |
| `tools/import_tflite/` | `benchmarks/framework_comparison_f411/gen/make_mlperf.py`: the int8 TFLite translator, emitting Rust |
| `tools/import_trees/` | `apps/import_trees/tinymind_import_trees.py`: the scikit-learn / XGBoost / LightGBM tree importer, emitting Rust |
| `examples/imported_trees.rs` | two models written by that importer, checked against tinymind (C++)'s importer + C++ runtime |
| `examples/imported_cnn.rs` | two convolutional models written by the PyTorch importer, checked against tinymind (C++)'s importer + C++ runtime |
| `examples/imported_mlp.rs` | two models written by the PyTorch importer, checked against tinymind (C++)'s importer + C++ runtime |

Everything else in tinymind (C++) is not ported: other Q
formats, unsigned and 8/64-bit storage, the other activations, the GRU and Elman
recurrent networks and recurrent networks with more than one hidden layer,
the deep Q-network (`QValueNeuralNetworkPolicy`),
the int8 quantization tiers, SIMD backends, and so on.

## How equivalence is proven

- `tests/xor_golden.rs` runs the `xor` example in each format and compares its
  stdout and the two CSV files byte-for-byte with the goldens in
  `tests/golden/<format>/`. The 1.7-2.6 MB `nn_fixed_xor.txt` is checked by
  length, line count and a 64-bit FNV-1a hash instead of being stored. The goldens
  come from the C++ build: Q16.16 from the example as shipped, Q8.8 from the same
  example with both bit counts in `xornet.h` set to 8. `tests/iris_golden.rs`
  does the same for iris against `tests/golden/iris/`, `tests/energy_golden.rs`
  for energy_efficiency against `tests/golden/energy/`, `tests/maze_golden.rs`
  for the maze against `tests/golden/maze/`, `tests/har_golden.rs` for
  har_activity against `tests/golden/har/`, and
  `tests/int8_xor_golden.rs` for the int8 example against `tests/golden/int8_xor/`,
  and `tests/imported_mlp_golden.rs` and `tests/imported_cnn_golden.rs` for the
  PyTorch importer's models. `TINYMIND_REF`
  records the tinymind (C++) commit they were last verified against. The int8, GBDT and
  PyTorch-importer tests need the `std` feature: run `cargo test --all-features`.
- `tests/deep_mlp.rs` checks networks with one, two and three hidden layers
  against tinymind (C++)'s `NeuralNetwork` (every weight after every training step,
  both formats), trained with SGD, Adam and RMSprop, with and without gradient
  clipping.
- `tests/anomaly_golden.rs` pins the anomaly example's output and hash. The
  demo has no tinymind (C++) counterpart, so this golden is the crate's own output, a
  regression check; its parts are each checked against tinymind (C++) above.
- `tests/float_mlp.rs` (feature `std`) trains `f32` and `f64` networks beside
  tinymind (C++)'s `NeuralNetwork` instantiated with `float` and `double` (its
  floating-point activations and transfer-function bundle), in nine
  configurations (SGD, and its floating-point Adam and RMSprop), and compares
  every weight after every step: bit-identical on
  a glibc host. The committed golden holds the `f64` runs; the `f32` runs depend
  on the C library's `tanhf`/`expf` (correctly rounded since glibc 2.41), so
  `scripts/compare-with-cpp.sh` compares them live, C++ and Rust on one host.
- `tests/inference_mlp.rs` checks the inference-only perceptron against
  tinymind (C++)'s untrainable network (`IsTrainable = false`) with the same weights,
  in both formats and two shapes, and `tests/inference_deep.rs` checks
  `InferenceNetwork` against tinymind (C++)'s untrainable `NeuralNetwork` with one
  to three hidden layers, in five configurations (both formats; tanh and ReLU
  hidden layers; sigmoid and linear outputs).
- `tests/softmax_tflite.rs` checks the int8 softmax against the TFLite Micro
  reference kernel's outputs (tinymind (C++)'s unit-test goldens and sweep hashes), and
  `tests/softmax_sweep.rs` against the tinymind (C++) softmax and its calibration
  over 504 calibrations and row lengths 1 to 511.
- `tests/kws_int8_golden.rs` checks every tensor of the `kws_cortex_m_int8`
  example (quantized input, weights, requantizers and each layer's output) and
  its logits report against `tests/golden/kws_int8/`, printed by tinymind (C++)'s
  example compiled unchanged (`scripts/cpp/kws_int8_dump.cpp`). The rest of its
  stdout, host timings and each build's sizes, is not compared.
- `tests/conv_sweep.rs` checks the int8 convolution and pooling layers against
  tinymind (C++)'s over random tensors: 13 convolution shapes (per tensor and per
  channel), 7 depthwise, 5 pointwise (both), 5 pooling windows (max and average)
  and 4 global pools, with padding on every side, strides above the kernel size
  and requantizers on both the fused and the two-step path.
- `tests/add_sweep.rs` checks the int8 elementwise addition and its
  calibration (`calibration::build_add`) against tinymind (C++)'s `QAdd` and
  `buildQAddParams`: 48 calibrated layers on all 65,536 input pairs, and 80
  layers with random integer parameters, some saturating. `tests/compose_sweep.rs`
  checks multiplication, concatenation and padding (`QMul`, `QConcat`, `QPad2D`)
  and their calibration against tinymind (C++)'s `QMul`, `QConcat2_2D`,
  `QPad2D`, `buildQMulRequantizer` and `buildRescaler`.
- `tests/gbdt_golden.rs` checks the GBDT example's three outputs against
  `tests/golden/gbdt/`, and `tests/tree_sweep.rs` checks the trees against
  tinymind (C++)'s `QDecisionTree` and `QGBDT` over 72 random ensembles (int8, uint8
  and int16 thresholds, ties, inputs on the thresholds).
  `tests/imported_trees_golden.rs` checks the imported-trees example against
  the tinymind (C++) runtime running the same models, imported with the
  tinymind (C++) importer.
- `tests/qlearn_sweep.rs` checks the Q-learner against tinymind (C++)'s
  `QLearner`, hashing the whole Q table after every update: the maze as the
  C++ example runs it (in unsigned Q16.16; the port's signed Q16.16 gives the
  same bits there) and 60 random environments in Q16.16 and Q8.8 with negative
  rewards, states with zero, one, two or more valid actions, and learning
  rates, discount factors and exploration rates the setters must reject.
- `tests/lstm_sweep.rs` checks the LSTM network against tinymind (C++)'s
  `LstmNeuralNetwork` in seven configurations (Q16.16 and Q8.8; tanh or ReLU
  cells; sigmoid or linear outputs; gradient clipping; Adam), hashing the
  outputs and every weight after every step of sequences with state resets;
  `tests/float_mlp.rs` does the same for four `f32` and `f64` LSTM networks.
- The TFLite translator (`tools/import_tflite`, run by
  `scripts/compare-with-cpp.sh`): `mlperf.py kws` and `mlperf.py ic` translate
  the MLPerf Tiny DS-CNN and ResNet-8 and compare the output hash over 8 seeded
  inputs with tinymind (C++)'s (which TFLite's reference kernels also give);
  `test_translator.py` checks its softmax and ADD parameters against
  tinymind (C++)'s on 504 and 48 calibrations; `test_ops.py` checks PAD, MUL and
  CONCATENATION on generated models against TFLite's reference kernels (LiteRT).
- `scripts/compare-with-cpp.sh` builds the C++ examples from a tinymind (C++) checkout
  (xor in both formats, iris, energy efficiency, maze, har_activity, int8 xor, GBDT) with the flags their
  Makefiles use, plus the C++ programs in `scripts/cpp/` that exercise what no
  example does, runs both sides, and compares every output file (74 checks today, including
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
  requantizer, the int8 dense layers, convolutions, pooling, addition,
  multiplication, padding and concatenation, the
  tables, the softmax, the trees, the Q-learner and the LSTM network never panic, and that pooling's divide rounds half away
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
