---
title: Equivalence with tinymind
layout: default
nav_order: 8
---

# Equivalence with tinymind

The acceptance test for every ported path is that it produces the same bits as tinymind's C++ build. Not close results: identical files.

## What is compared

Four of tinymind's examples are ported, and every file they write must be byte-identical to the C++ example's output:

- **XOR**, in Q16.16 and Q8.8: stdout, the per-step weight trajectory (`nn_fixed_xor.txt`, about 20,000 lines), the learning curve and the decision surface.
- **Iris**: stdout, the loss curve, the per-sample test predictions and the confusion matrix. Its input is the dataset file tinymind ships with the example, checked byte for byte against tinymind's copy.
- **Energy efficiency**: stdout, the loss curve and the per-sample test predictions of a regression (linear output, clipped gradients) on the UCI Energy Efficiency data, again read from tinymind's own copy of the file.
- **Int8 XOR**: stdout and the decision surface of a PyTorch-trained network running in int8. tinymind's int8 path matches TensorFlow Lite's reference kernels, so this also ties the Rust port to TFLite's arithmetic on these cases.

Parts that no tinymind example exercises are compared through small C++ programs built on tinymind's library (in `scripts/cpp/`), whose output must match the Rust port's in the same way:

- **Networks with several hidden layers**: tinymind's `NeuralNetwork` with two and three hidden layers, in both formats, trained for thousands of steps, with a hash of every weight after every step; and two regressions with a linear output and gradient clipping, sized so that the clipping and tinymind's unclipped bias weights both change the result.
- **Inference-only networks**: tinymind's network built with `IsTrainable = false`, loaded with the same weights as the Rust `InferencePerceptron`.
- **Int8 softmax**: tinymind's TFLite-exact softmax and its calibration over 504 calibrations and row lengths from 1 to 511, plus the golden rows tinymind's tests take from the TensorFlow Lite Micro reference kernel.
- **PyTorch importer**: the same models imported with tinymind's importer and with this repository's, run on both runtimes.

Fixed-point arithmetic makes this demanding. A multiply truncates, so `(a * b) * c` and `a * (b * c)` can differ in the last bit, and one bit early in training becomes a different weight trajectory. The port keeps the C++ operation order in every expression, and repeats details that do not affect the result's quality but do affect its bits, such as a random draw the C++ makes for a connection it never uses.

## How it is checked

- **Golden tests.** `cargo test --all-features` runs the examples (xor in each format, iris, energy efficiency, int8 xor and the importer's models) and the comparisons above, and checks their output against goldens produced by the C++ build. The goldens come only from C++ runs, never from the Rust output, or the test would prove nothing.
- **Live comparison.** `scripts/compare-with-cpp.sh` builds tinymind's C++ examples from a tinymind checkout with the flags their Makefiles use, and the C++ programs in `scripts/cpp/`, runs both sides and compares every output file: 29 comparisons today. The Q8.8 xor build is tinymind's example with its two bit-count constants set to 8; nothing else changes.
- **Pinned tinymind commit.** The file `TINYMIND_REF` records the tinymind commit the port was last verified against. CI checks out exactly that commit.
- **Arithmetic oracle.** Ported from tinymind's own oracle: every `QValue` operation is checked against an exact integer reference, on edge values and a random sample for nine formats, and on every pair of operands for Q8.8, about 26 billion checks per pull request. That run keeps Rust's integer-overflow checks on, so an overflow that happens to give the right answer still fails. A nightly job repeats the sample with a fresh seed.
- **Proofs.** [Kani](https://model-checking.github.io/kani/), a model checker built on the same CBMC engine tinymind uses for its own proofs, checks properties for every possible input rather than a sample: that addition, subtraction and multiplication wrap and round exactly as specified, that division never panics, that tanh and sigmoid never index outside their tables or leave their range, that the int8 requantizer's fast path equals gemmlowp's two-step rounding, and that the requantizer, the int8 dense layers, the tables and the softmax never panic, whatever their parameters.
- **No reachable panics.** CI links a Cortex-M firmware image that calls every public function and has no panic handler, for Cortex-M0+ and M4F at three optimisation levels; a panic path that survives optimisation fails the build.
- **Tables.** CI regenerates the activation lookup tables from tinymind's source and fails if they differ from the committed ones.
- **CI and branch protection.** Every pull request runs all of this, and a pull request cannot merge unless it passes. Each check was shown to fail on a deliberately planted defect before it was trusted.
