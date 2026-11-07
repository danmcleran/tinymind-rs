---
title: Home
layout: default
nav_order: 1
---

<img src="{{ site.baseurl }}/assets/tinymind_logo.png" alt="tinymind logo" width="160" style="display:block;margin:0 auto 1rem;">

# Neural networks for embedded Rust
{: .fs-9 }

**tinymind-rs** (crate `tinymind`) is a Rust port of [tinymind](https://danmcleran.github.io/tinymind/), a C++ library for neural networks on microcontrollers. Every ported path produces the same bits as the C++ build.
{: .fs-6 .fw-300 }

[XOR example]({{ site.baseurl }}/xor){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Iris example]({{ site.baseurl }}/iris){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Int8 inference]({{ site.baseurl }}/int8){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[API reference]({{ site.baseurl }}/api/tinymind/){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/danmcleran/tinymind-rs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What it is

- **`no_std` and allocation-free.** The library links only `core`. Networks are plain fixed-size arrays sized by const generics: no heap, no `alloc`, no global state.
- **Fixed-point so far.** The ported value types are Q-format integers (Q16.16 and Q8.8) and int8 in the TFLite/CMSIS-NN affine model, so training and inference need no FPU, and activations come from lookup tables. tinymind's networks are generic over the value type and are also tested with `float` and `double`; floating-point networks are not ported yet.
- **No `unsafe`, no reachable panics.** The library is `#![forbid(unsafe_code)]`, and CI links a Cortex-M firmware image that calls every public function and whose panic handler does not exist: any panic path left after optimisation would fail the build.
- **Bit-exact with C++.** The port is checked byte for byte against tinymind's own C++ build on every pull request. See [Equivalence with tinymind]({{ site.baseurl }}/equivalence).
- **Built for microcontrollers, checked on them.** CI builds the library for Cortex-M4F (`thumbv7em-none-eabihf`) and Cortex-M0+ (`thumbv6m-none-eabi`), and it has been run on an STM32F411 and an STM32L031: int8 inference and on-device training match the C++ bit for bit there, at about tinymind's code size. See [On-board results]({{ site.baseurl }}/benchmarks). Minimum Rust version: 1.85.

## Status

Early. The port covers what three of tinymind's examples use (XOR, iris and an int8 XOR), plus the int8 softmax:

| Ported | Not yet ported |
|---|---|
| Q16.16 (`i32`) and Q8.8 (`i16`) arithmetic, Truncate and Wrap policies | Other Q formats, saturating and rounding policies |
| ReLU, and tanh and sigmoid via lookup tables; int8 dense inference (requantizer, quantized ReLU, int8 activation tables, softmax bit-exact with TFLite's) | The other activations (capped and leaky ReLU, linear, ...) |
| MLP with one hidden layer: trainable (SGD with momentum) or inference-only (weights only, can live in flash) | Multiple hidden layers, recurrent layers, int8 convolution and the rest of the int8 family, SIMD |
| | Floating-point (`float`, `double`) networks |

The crate is on [crates.io](https://crates.io/crates/tinymind); the API is not stable yet (before 1.0, a minor version may break it). tinymind itself is the full library; this port grows by porting slices of it.

## Quick start

```bash
git clone https://github.com/danmcleran/tinymind-rs
cd tinymind-rs
cargo run --release --example xor -- output                  # Q16.16; output files in ./output
cargo run --release --example xor -- --format q8.8 output    # the same network in Q8.8
cargo run --release --example iris -- output                 # iris species classifier
cargo run --release --features std --example int8_xor -- output   # PyTorch-trained int8 network
cargo test                                               # unit, doc and golden tests
```

To use the library from another project:

```toml
[dependencies]
tinymind = "0.1"
# Host-side int8 calibration (float scales to requantizers and tables):
# tinymind = { version = "0.1", features = ["std"] }
```

The [API reference]({{ site.baseurl }}/api/tinymind/) starts with a complete example that trains XOR; it runs as a test on every build.
