---
title: Home
layout: default
nav_order: 1
---

<img src="{{ site.baseurl }}/assets/tinymind_logo.png" alt="tinymind logo" width="160" style="display:block;margin:0 auto 1rem;">

# Neural networks for embedded Rust
{: .fs-9 }

**tinymind-rs** (crate `tinymind`) is a Rust port of [tinymind (C++)](https://danmcleran.github.io/tinymind/), a C++ library for neural networks on microcontrollers. Every ported path produces the same bits as the C++ build (floating point: with `std` on a glibc host).
{: .fs-6 .fw-300 }

[XOR example]({{ site.baseurl }}/xor){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Iris example]({{ site.baseurl }}/iris){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Energy example]({{ site.baseurl }}/energy){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Anomaly detection]({{ site.baseurl }}/anomaly){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Decision trees]({{ site.baseurl }}/trees){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Int8 inference]({{ site.baseurl }}/int8){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[API reference]({{ site.baseurl }}/api/tinymind/){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/danmcleran/tinymind-rs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What it is

- **`no_std` and allocation-free.** Without features the library links only `core`. Networks are plain fixed-size arrays sized by const generics: no heap, no `alloc`, no global state.
- **Fixed-point first, floating point too.** The value types are Q-format integers (Q16.16 and Q8.8), int8 in the TFLite/CMSIS-NN affine model, and the int8, uint8 or int16 feature grids of the decision trees, so training and inference need no FPU, and activations come from lookup tables. Like tinymind (C++)'s, the networks also run on `f32` and `f64`, with tanh and sigmoid from the math library (feature `std`, or `libm` without std); with `std` on a glibc host they train bit for bit like tinymind (C++)'s `float` and `double` networks.
- **No `unsafe`, no reachable panics.** The library is `#![forbid(unsafe_code)]`, and CI links a Cortex-M firmware image that calls every public function and whose panic handler does not exist: any panic path left after optimisation would fail the build.
- **Bit-exact with C++.** The port is checked byte for byte against tinymind (C++)'s own C++ build on every pull request. See [Equivalence with tinymind (C++)]({{ site.baseurl }}/equivalence).
- **Built for microcontrollers, checked on them.** CI builds the library for Cortex-M4F (`thumbv7em-none-eabihf`) and Cortex-M0+ (`thumbv6m-none-eabi`), and it has been run on an STM32F411 and an STM32L031: int8 inference and on-device training match the C++ bit for bit there, at about tinymind (C++)'s code size. See [On-board results]({{ site.baseurl }}/benchmarks). Minimum Rust version: 1.85.

How the port is checked, and what Rust's guarantees did and did not buy: [Bit for bit]({{ site.baseurl }}/bit-for-bit).

## Status

The port covers what eight of tinymind (C++)'s examples use (XOR, iris, energy efficiency, the Q-learning maze, LSTM activity recognition, an int8 XOR, int8 gradient-boosted trees and a keyword-spotting shaped int8 CNN), plus networks with several hidden layers, inference-only networks, floating-point (`f32`, `f64`) networks, the Adam and RMSprop optimizers, the int8 softmax, convolution and pooling layers, residual addition, multiplication, padding and concatenation, and a TFLite translator. It is a partial port of tinymind (C++), and every ported path is checked bit for bit against it:

| Ported | Not yet ported |
|---|---|
| Q16.16 (`i32`) and Q8.8 (`i16`) arithmetic, Truncate and Wrap policies | Other Q formats, saturating and rounding policies |
| ReLU, linear, and tanh and sigmoid via lookup tables; int8 inference (requantizer, dense layers, 2D convolution (standard, depthwise and pointwise), pooling, elementwise addition and multiplication, padding, concatenation, quantized ReLU, int8 activation tables, softmax bit-exact with TFLite's); quantized decision trees and gradient-boosted ensembles, with an importer for scikit-learn, XGBoost and LightGBM models | The other activations (capped and leaky ReLU, ...) |
| Trainable networks with up to seven hidden layers (SGD with momentum, Adam or RMSprop; optional gradient clipping); inference-only networks of the same depths (weights only, can live in flash) | Weight decay and learning-rate schedules, the rest of the int8 family (recurrent cells, attention), SIMD |
| Floating-point (`f32`, `f64`) networks, trained with SGD or tinymind (C++)'s floating-point Adam and RMSprop | |
| Tabular Q-learning (`QLearner` over a `QTable`), with tinymind (C++)'s maze example | The deep Q-network (`QValueNeuralNetworkPolicy`) |
| LSTM networks with one hidden layer (`LstmNetwork`, in fixed or floating point), with tinymind (C++)'s activity-recognition example | GRU and Elman networks, recurrent networks with more than one hidden layer |

The crate is on [crates.io](https://crates.io/crates/tinymind); the API is not stable yet (before 1.0, a minor version may break it). tinymind (C++) itself is the full library; this port grows by porting slices of it.

## Quick start

```bash
git clone https://github.com/danmcleran/tinymind-rs
cd tinymind-rs
cargo run --release --example xor -- output                  # Q16.16; output files in ./output
cargo run --release --example xor -- --format q8.8 output    # the same network in Q8.8
cargo run --release --example iris -- output                 # iris species classifier
cargo run --release --example energy_efficiency -- output    # building energy-load regression
cargo run --release --example anomaly -- output              # autoencoder trained on the device flags faults
cargo run --release --features std --example int8_xor -- output   # PyTorch-trained int8 network
cargo run --release --features std --example imported_mlp    # models written by the PyTorch importer
cargo run --release --features std --example imported_cnn    # convolutional models written by the PyTorch importer
cargo run --release --features std --example gbdt_tabular_int8 -- output   # int8 gradient-boosted trees
cargo run --release --features std --example kws_cortex_m_int8   # keyword-spotting shaped int8 CNN
cargo run --release --example imported_trees                 # scikit-learn trees via the tree importer
cargo run --release --example maze -- output                 # Q-learning: find the way out of a six-room maze
cargo run --release --example har_activity -- output         # LSTM: classify accelerometer windows into four activities
cargo test --all-features                                # unit, doc and golden tests
```

To use the library from another project:

```toml
[dependencies]
tinymind = "0.2.0"
# Host-side int8 calibration, and f32/f64 tanh, sigmoid, AdamFloat, RmsPropFloat:
# tinymind = { version = "0.2.0", features = ["std"] }
# The same float math without std (for example on a Cortex-M4F):
# tinymind = { version = "0.2.0", features = ["libm"] }
```

The [API reference]({{ site.baseurl }}/api/tinymind/) starts with a complete example that trains XOR; it runs as a test on every build.
