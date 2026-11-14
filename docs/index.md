---
title: "tinymind for Rust: neural networks for microcontrollers"
layout: default
nav_order: 1
# The title is what link previews (og:title) show; the site name in the
# header links here, so the page needs no sidebar entry of its own.
nav_exclude: true
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
[Q-learning maze]({{ site.baseurl }}/maze){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Activity recognition]({{ site.baseurl }}/har){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Fine-tuning on the device]({{ site.baseurl }}/fine-tuning){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Int8 inference]({{ site.baseurl }}/int8){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[API reference]({{ site.baseurl }}/api/tinymind/){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/danmcleran/tinymind-rs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What it is

- **`no_std` and allocation-free.** The library needs only Rust's `core` library, not `std` or `alloc`, so it builds for bare-metal targets with no operating system. Two optional features add more: `std` for the host-side int8 calibration helpers, and `libm` for floating-point tanh, sigmoid, the float optimizers and the heads' softmax without `std`. Networks are plain fixed-size arrays sized by const generics: no heap, no global state.
- **Fixed point and floating point, as in tinymind (C++).** Both libraries run networks in fixed point and in floating point. The fixed-point value types are Q-format integers (Q16.16 and Q8.8), int8 in the TFLite/CMSIS-NN affine model, and the int8, uint8 or int16 feature grids of the decision trees, so training and inference need no FPU, and activations come from lookup tables. The same networks run on `f32` and `f64`, with tanh and sigmoid from the math library (feature `std`, or `libm` without std); with `std` on a glibc host they train bit for bit like tinymind (C++)'s `float` and `double` networks.
- **No `unsafe`, no reachable panics.** The library is `#![forbid(unsafe_code)]`, and CI links a Cortex-M firmware image that calls every public function and whose panic handler does not exist: any panic path left after optimisation would fail the build.
- **Bit-exact with C++.** The port is checked byte for byte against tinymind (C++)'s own C++ build on every pull request. See [Equivalence with tinymind (C++)]({{ site.baseurl }}/equivalence).
- **Built for microcontrollers, checked on them.** CI builds the library for Cortex-M4F (`thumbv7em-none-eabihf`) and Cortex-M0+ (`thumbv6m-none-eabi`), and it has been run on an STM32F411 and an STM32L031: int8 inference and on-device training match the C++ bit for bit there, at about tinymind (C++)'s code size. See [On-board results]({{ site.baseurl }}/benchmarks). Minimum Rust version: 1.85.

How the port is checked, and what Rust's guarantees did and did not buy: [Bit for bit]({{ site.baseurl }}/bit-for-bit).

## Status

The port covers what ten of tinymind (C++)'s examples use (XOR, iris, energy efficiency, the Q-learning maze and its deep Q-network version, LSTM activity recognition, gas sensor drift with on-device fine-tuning, an int8 XOR, int8 gradient-boosted trees and a keyword-spotting shaped int8 CNN), plus networks with several hidden layers, inference-only networks, floating-point (`f32`, `f64`) networks, the Adam and RMSprop optimizers, the int8 softmax, convolution and pooling layers, residual addition, multiplication, padding and concatenation, int8 LSTM and GRU cells, a 16-bit FFT, and a TFLite translator. It is a partial port of tinymind (C++), and every ported path is checked bit for bit against it:

| Ported | Not yet ported |
|---|---|
| Q16.16 (`i32`) and Q8.8 (`i16`) arithmetic, Truncate and Wrap policies | Other Q formats, saturating and rounding policies |
| ReLU, linear, and tanh and sigmoid via lookup tables; int8 inference (requantizer, dense layers, 2D convolution (standard, depthwise and pointwise), pooling, elementwise addition and multiplication, padding, concatenation, quantized ReLU, int8 activation tables, softmax bit-exact with TFLite's, LSTM and GRU cells, a radix-2 FFT on 16-bit samples); quantized decision trees and gradient-boosted ensembles, with an importer for scikit-learn, XGBoost and LightGBM models | The other activations (capped and leaky ReLU, ...) |
| Trainable networks with up to seven hidden layers (SGD with momentum, Adam or RMSprop; optional gradient clipping); inference-only networks of the same depths (weights only, can live in flash) | Weight decay and learning-rate schedules, the rest of the int8 family (attention, state-space cells), SIMD |
| Floating-point (`f32`, `f64`) networks, trained with SGD or tinymind (C++)'s floating-point Adam and RMSprop | |
| Q-learning (`QLearner`) over a Q table or a deep Q-network (`QNetwork`), with tinymind (C++)'s maze and deep Q-network maze examples | |
| LSTM networks with one hidden layer (`LstmNetwork`, in fixed or floating point), with tinymind (C++)'s activity-recognition example | Trainable GRU and Elman networks, recurrent networks with more than one hidden layer |
| Head-only fine-tuning on the device: a trainable dense head (`DenseHead`) and a nearest-class-mean head (`PrototypeHead`) on top of a frozen model, with tinymind (C++)'s gas sensor drift example | |

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
cargo run --release --example dqn_maze -- output             # the same maze, learned by a deep Q-network
cargo run --release --example har_activity -- output         # LSTM: classify accelerometer windows into four activities
cargo run --release --example gas_sensor_drift -- output     # sensor drift, then on-device fine-tuning from a few samples
cargo test --all-features                                # unit, doc and golden tests
```

To use the library from another project:

```toml
[dependencies]
tinymind = "0.3.5"
# Host-side int8 calibration, and f32/f64 tanh, sigmoid, AdamFloat, RmsPropFloat, DenseHead softmax:
# tinymind = { version = "0.3.5", features = ["std"] }
# The same float math without std (for example on a Cortex-M4F):
# tinymind = { version = "0.3.5", features = ["libm"] }
```

The [API reference]({{ site.baseurl }}/api/tinymind/) starts with a complete example that trains XOR; it runs as a test on every build.
