---
title: Home
layout: default
nav_order: 1
---

<img src="{{ site.baseurl }}/assets/tinymind_logo.png" alt="tinymind logo" width="160" style="display:block;margin:0 auto 1rem;">

# Fixed-point neural networks for embedded Rust
{: .fs-9 }

**tinymind-rs** (crate `tinymind`) is a Rust port of [tinymind](https://danmcleran.github.io/tinymind/), a C++ library for neural networks on microcontrollers. Every ported path produces the same bits as the C++ build.
{: .fs-6 .fw-300 }

[XOR example]({{ site.baseurl }}/xor){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[API reference]({{ site.baseurl }}/api/tinymind/){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/danmcleran/tinymind-rs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What it is

- **`no_std` and allocation-free.** The library links only `core`. Networks are plain fixed-size arrays sized by const generics: no heap, no `alloc`, no global state.
- **Fixed-point only.** Weights, activations and gradients are Q-format integers, so training and inference need no FPU. Activations come from lookup tables with linear interpolation.
- **No `unsafe`.** The library is `#![forbid(unsafe_code)]`.
- **Bit-exact with C++.** The port is checked byte for byte against tinymind's own C++ build on every pull request. See [Equivalence with tinymind]({{ site.baseurl }}/equivalence).
- **Built for microcontrollers.** CI builds the library for Cortex-M4F (`thumbv7em-none-eabihf`) and Cortex-M0+ (`thumbv6m-none-eabi`). Minimum Rust version: 1.85.

## Status

Early. The port covers what tinymind's XOR example uses, in two formats:

| Ported | Not yet ported |
|---|---|
| Q16.16 (`i32`) and Q8.8 (`i16`) arithmetic, Truncate and Wrap policies | Other Q formats, saturating and rounding policies |
| Tanh and sigmoid lookup-table activations | ReLU and the other activations |
| Trainable MLP with one hidden layer, SGD with momentum | Multiple hidden layers, recurrent layers, int8 inference, SIMD |

The API is not stable yet, and the crate is not on crates.io yet. tinymind itself is the full library; this port grows by porting slices of it.

## Quick start

```bash
git clone https://github.com/danmcleran/tinymind-rs
cd tinymind-rs
cargo run --release --bin xor -- output                  # Q16.16; output files in ./output
cargo run --release --bin xor -- --format q8.8 output    # the same network in Q8.8
cargo test                                               # unit, doc and golden tests
```

To use the library from another project before it is on crates.io:

```toml
[dependencies]
tinymind = { git = "https://github.com/danmcleran/tinymind-rs" }
```

The [API reference]({{ site.baseurl }}/api/tinymind/) starts with a complete example that trains XOR; it runs as a test on every build.
