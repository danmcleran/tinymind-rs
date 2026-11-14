---
title: Bit for bit
layout: default
nav_order: 15
---

# Porting a C++ neural-network library to embedded Rust, bit for bit

[tinymind (C++)](https://danmcleran.github.io/tinymind/) is my C++ template library for neural networks on microcontrollers: fixed-point, int8 and floating-point arithmetic, no heap, small enough that a trainable XOR network fits in a few kilobytes. [tinymind-rs](https://github.com/danmcleran/tinymind-rs) is its Rust port, published on crates.io as [`tinymind`](https://crates.io/crates/tinymind).

The port has one rule that shapes everything else: **every ported path produces the same bits as the C++.** Not "close", not "within a tolerance": the same bytes in every output file. This post is about what that rule demands, how it is checked, and what Rust's guarantees did and did not buy along the way.

## What is in the crate

As of 0.3.4:

- Q16.16 and Q8.8 fixed-point arithmetic (`QValue<i32, 16>`, `QValue<i16, 8>`), with ReLU, linear, and lookup-table tanh and sigmoid activations; the same networks also run on `f32` and `f64`.
- Networks that train on the device with SGD and momentum, Adam or RMSprop (optionally with gradient clipping; for `f32`/`f64` also tinymind (C++)'s floating-point Adam and RMSprop, `AdamFloat` and `RmsPropFloat`): up to seven hidden layers (`NeuralNetwork` over `Dense` layers), and a one-hidden-layer perceptron (`MultilayerPerceptron`), each with an inference-only version that holds nothing but its weights (`InferenceNetwork`, `InferencePerceptron`).
- int8 inference in the TFLite / CMSIS-NN quantization model: a requantizer with gemmlowp rounding, dense layers (per-tensor and per-channel), 2D convolutions (standard, depthwise and pointwise), pooling, residual addition, multiplication, padding, concatenation, quantized ReLU, activation tables, a softmax that is bit-exact with TFLite's, and LSTM and GRU cells; and a radix-2 FFT on 16-bit samples (`QFft1D`, with a host-side twiddle-table builder).
- Q-learning (`QLearner`, module `qlearn`) over a `QTable` or a deep Q-network (`QNetwork`), with tinymind (C++)'s maze and deep Q-network maze examples.
- LSTM networks with one hidden layer (`LstmNetwork`, module `recurrent`), trained one step back in time, in fixed or floating point, with tinymind (C++)'s activity-recognition example.
- Head-only fine-tuning (module `head`): a trainable dense head (`DenseHead`) and a nearest-class-mean head (`PrototypeHead`) that adapt a frozen model on the device, with tinymind (C++)'s gas sensor drift example.
- Quantized decision trees and gradient-boosted ensembles (`QDecisionTree`, `QGbdt`) over a node pool that can live in flash, with int8, uint8 or int16 thresholds.
- Three tools (in the repository, not the crate) that write `no_std` Rust modules: an importer for a trained PyTorch `nn.Sequential` (dense and convolutional), one for scikit-learn, XGBoost and LightGBM tree models, and a translator for int8 TensorFlow Lite models.

The library is `no_std`, allocation-free, has no required dependencies (`libm` is optional), and is `#![forbid(unsafe_code)]`.

## Why bit-exact

Fixed-point arithmetic is deterministic, so two implementations of the same network either agree exactly or one of them has a bug. Allowing a tolerance would hide exactly the bugs a port introduces: an expression evaluated in a different order, a rounding mode that differs in one case in a thousand, a random number drawn one call too early. Bit-exactness turns the C++ library into a complete oracle for the Rust one.

It is also more demanding than it sounds. A few of the things it took:

**Operation order.** A fixed-point multiply truncates, so `(a * b) * c` and `a * (b * c)` can differ in the last bit. Every expression in the port keeps the C++ shape: the weight update is `(lr * g + momentum * dw) + accel * prev_dw`, with `lr * g` rounded to nearest and the two momentum products rounded toward zero as tinymind (C++) rounds them, not any algebraically equal rearrangement. Additions, on the other hand, wrap at the storage width, and wrapping addition is associative, so the order of a sum does not matter. Knowing which operations are order-sensitive tells you where to be careful.

**The random number generator.** The C++ examples initialise weights and draw training samples with `rand()`. The port carries a bit-exact reimplementation of glibc's `srand`/`rand`, checked against the real one over FFI for 212 seeds. It also has to make the same number of draws in the same order: the C++ output neurons carry one outgoing connection that is never used, and initialising it consumes a draw, so the Rust output neurons keep that connection too.

**Printing.** The C++ examples write their results with iostreams and `printf("%g")`. To compare output files byte for byte, the port reimplements those formats rather than using Rust's `{}`.

## How it is checked

(The details are on [Equivalence with tinymind (C++)]({{ site.baseurl }}/equivalence).)

On every pull request, CI checks out tinymind (C++) at a pinned commit, builds its examples (and a few small C++ programs on its library) from source, runs the Rust ports beside them, and compares every output file: 90 checks today. For the XOR example that includes every weight after each of 20,000 training steps, in both Q16.16 and Q8.8. The goldens that stand for tinymind (C++) are copied out of the C++ run, never generated by the Rust code, so a golden test cannot pass by agreeing with itself. The one exception, the anomaly-detection example, has no C++ counterpart: its golden only pins the output against regressions.

The examples only exercise the paths they use, so the arithmetic gets more:

- **A differential oracle**, ported from tinymind (C++)'s, checks every Q-format operation against an exact 64-bit reference. For Q8.8 it is exhaustive: every pair of values, about four billion, in roughly fifteen seconds.
- **Overflow checks on, in release.** Rust overflow is not undefined behaviour, but release builds wrap silently, so a plain `+` that should have been `wrapping_add` gives the right answer and passes. The exhaustive run is built with `overflow-checks = true`, which turns that mistake into a panic. I planted one to make sure: it passed with checks off and failed with them on.
- **Kani proofs** (99 harnesses) for every input rather than a sample: add and subtract wrap, multiply floors then wraps, and the round-toward-zero and round-to-nearest multiplies SGD uses, for six Q formats; the int8 requantizer's single-rounding fast path equals gemmlowp's two-step rounding wherever it is used; and the requantizer, dense layers, convolutions, pooling, addition, multiplication, padding, concatenation, tables, softmax, the LSTM and GRU cells and the FFT never panic, whatever their parameters; the FFT's Q1.15 multiply floors and its magnitudes saturate only where both parts are -32768; the decision trees never panic for any field values of small node pools, the Q-learner, the deep Q-network, the LSTM network and small Q8.8 fine-tuning heads never panic for any parameters, and the heads keep every class count within its cap; and the FFT's saturation skip on cores without a saturating instruction changes no output.

One thing Kani could not do: prove that 64-bit fixed-point division truncates toward zero. The symbolic division timed out after five minutes with two different SAT solvers. That property is covered by the oracle instead, exhaustively for Q8.8.

Every new test or gate in the repository has been shown to fail on a planted defect before it was trusted: a rounding multiply instead of a flooring one, a table lookup without its index clamp, two Newton-Raphson steps instead of three in the softmax.

## What Rust bought, and what it did not

The C++ library is checked with sanitizers, fuzzing and CBMC proofs. The Rust port needs none of the sanitizers: with `#![forbid(unsafe_code)]` and no threads, safe Rust cannot have the undefined behaviour they look for. Where the C++ int8 code would hit undefined behaviour on inputs outside its documented contract (an `int32` accumulator overflowing, a shift out of range), the Rust code wraps or clamps instead, and gives identical results inside the contract.

But "no undefined behaviour" is not "no crash". On a microcontroller, a panic is a crash, and safe Rust panics readily: an out-of-bounds index, an `unwrap`, `clamp` with `min > max`, arithmetic overflow in a debug build.

So the crate has a second guarantee, checked at link time. CI builds a `no_std` Cortex-M firmware image that calls every public function with opaque inputs, and whose panic handler calls a function that does not exist. If any panic path survives optimisation, the image fails to link. It is built for Cortex-M0+ and Cortex-M4F at three optimisation levels.

Keeping that gate green changed the API in small ways:

- The weight getters and setters take and return whole matrices instead of an index, so there is no index to get wrong.
- The requantizer saturates with two `if` statements instead of `clamp`, because its bounds are runtime fields and `clamp` panics when `qmin > qmax`. (Where the bounds are constants, `clamp` is fine, and clippy says so.)
- The activation table lookup clamps its index, which lets the compiler prove the access in bounds. Removing the clamp makes the gate fail.
- Panic-freedom must not depend on inlining. At opt-level `s` and `z`, a helper that took an index from its caller kept its bounds check, and so did the square root's Newton division; both now use total operations (`get(..)`, `checked_div`).

## On real hardware

The port runs on an STM32F411 (Cortex-M4F at 100 MHz) and an STM32L031 (Cortex-M0+ at 32 MHz, 32 KB of flash). On both, the Rust port of tinymind (C++)'s int8 digits classifier (64-32-16-10) produces the same output hash as the C++ build, and XOR trained on the board reproduces the desktop weight trajectory bit for bit, in both formats. An autoencoder trained on the board for anomaly detection (8-4-8, Q16.16) makes every decision bit for bit as on the desktop, even on the 32 KB L031 ([Anomaly detection]({{ site.baseurl }}/anomaly)). A 256-point int8 FFT gives tinymind (C++)'s spectrum hash in 41,991 cycles on the F411 and 91,430 on the L031: about half and 10% fewer than the best gcc `-O3` build of the C++.

int8 MLP, one inference:

| Board | tinymind-rs | tinymind (C++), scalar | Code + weights (Rust) |
|---|---|---|---|
| STM32F411 | 19,149 cycles | 21,002 cycles | 6,236 B |
| STM32L031 | 35,710 cycles | 32,644 cycles | 5,948 B |

TensorFlow Lite Micro does not fit in the L031's 32 KB at all. The method and the full tables are on [On-board results]({{ site.baseurl }}/benchmarks). On the F411, one Q16.16 XOR training step takes 1,233 cycles, about 12 microseconds. The C++ build is faster on the F411 when it uses the Cortex-M4's SMLAD instruction. Arm DSP intrinsics are not available in stable Rust, so the port has no SMLAD path of its own; only its multi-channel convolutions reach SMLAD, through LLVM, when built with an opt-in LLVM option. Convolutional networks are faster than the C++'s portable build even without it: the KWS-shaped CNN by 12-26%, the MLPerf Tiny DS-CNN and ResNet-8 by 31% each (full tables on [On-board results]({{ site.baseurl }}/benchmarks)).

## Shipping a trained network

tinymind (C++) can compile a network without its training machinery. In Rust that is a separate type, `InferencePerceptron`, which holds only the weights. Its constructor is a `const fn`, so a trained network can be a `static`, which puts it in flash:

```rust
use tinymind::{InferencePerceptron, Q8_8, Sigmoid, Tanh};

const fn q(x: i16) -> Q8_8 {
    Q8_8::from_int(x as i32)
}

// XOR by hand: hidden neuron 0 is OR, hidden neuron 1 is AND.
static XOR: InferencePerceptron<Q8_8, Tanh, Sigmoid, 2, 2, 1> = InferencePerceptron::new(
    [[q(4), q(4)], [q(4), q(4)]], // input -> hidden
    [q(-2), q(-6)],               // input bias -> hidden
    [[q(6)], [q(-6)]],            // hidden -> output
    [q(-3)],                      // hidden bias -> output
);

fn main() {
    let [y] = XOR.feed_forward(&[q(1), q(0)]);
    assert!(y > Q8_8::from_raw(128)); // 1 XOR 0 is above 0.5
}
```

It is checked against tinymind (C++)'s untrainable network loaded with the same weights. A Q16.16 2-4-1 network is 68 bytes, against 384 for the trainable one (and 156 for tinymind (C++)'s untrainable network, which also keeps per-neuron state).

## From PyTorch

For int8 models, the importer runs tinymind (C++)'s own calibration and quantization code and writes Rust instead of a C++ header: int8 weights, integer requantizers, baked activation tables and a `forward` function, with no float math left for the device.

```python
import tinymind_import as ti

layers = ti.from_torch(net, input_shape=(1, 12, 12))   # Linear, Conv2d, BatchNorm2d, pools, ReLU, ... (input_shape for conv nets)
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration_inputs,
                        "my_model.rs", "my_model")
```

```rust
mod my_model;

let y = my_model::forward(&input); // [i8; N] -> [i8; M]
```

CI imports the same example models with both importers, runs them on the C++ and the Rust runtimes, and checks that the outputs match byte for byte.

## What it is not

It is a port of slices of tinymind (C++), not all of it. Not yet ported: trainable GRU and Elman networks and recurrent networks with more than one hidden layer, the rest of the int8 layers (attention, state-space cells), and the SIMD backends. A full port in stable Rust is harder than it looks: tinymind (C++) picks storage types and layer shapes with compile-time arithmetic on template parameters, which stable Rust's const generics cannot express yet (`generic_const_exprs`).

The C++ library remains the full product. The Rust port grows one checked slice at a time, and I would like to hear which slice would make it useful to you.

- Crate: <https://crates.io/crates/tinymind>
- API docs: <https://docs.rs/tinymind>
- Source: <https://github.com/danmcleran/tinymind-rs>
- tinymind (C++): <https://danmcleran.github.io/tinymind/>
