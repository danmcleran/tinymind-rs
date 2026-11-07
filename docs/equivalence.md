---
title: Equivalence with tinymind
layout: default
nav_order: 4
---

# Equivalence with tinymind

The acceptance test for every ported path is that it produces the same bits as tinymind's C++ build. Not close results: identical files.

## What is compared

The XOR example writes four outputs: stdout, the per-step weight trajectory (`nn_fixed_xor.txt`, about 20,000 lines), the learning curve and the decision surface. For both Q16.16 and Q8.8, all four must be byte-identical to the C++ example's output.

Fixed-point arithmetic makes this demanding. A multiply truncates, so `(a * b) * c` and `a * (b * c)` can differ in the last bit, and one bit early in training becomes a different weight trajectory. The port keeps the C++ operation order in every expression, and repeats details that do not affect the result's quality but do affect its bits, such as a random draw the C++ makes for a connection it never uses.

## How it is checked

- **Golden tests.** `cargo test` runs the `xor` binary in each format and compares its output with goldens produced by the C++ build. The goldens come only from C++ runs, never from the Rust output, or the test would prove nothing.
- **Live comparison.** `scripts/compare-with-cpp.sh` builds tinymind's C++ example from a tinymind checkout, runs both programs and compares every output file. The Q8.8 build is tinymind's example with its two bit-count constants set to 8; nothing else changes.
- **Pinned tinymind commit.** The file `TINYMIND_REF` records the tinymind commit the port was last verified against. CI checks out exactly that commit.
- **Tables.** CI regenerates the activation lookup tables from tinymind's source and fails if they differ from the committed ones.
- **CI and branch protection.** Every pull request runs all of this, and a pull request cannot merge unless it passes. Each check was shown to fail on a deliberately planted defect before it was trusted.

## Findings in tinymind

Matching the C++ bit for bit means matching its bugs too. When the port finds one, it is reported to tinymind and fixed there, and the port follows the fix:

- [tinymind #236](https://github.com/danmcleran/tinymind/pull/236): the XOR network's weights came from glibc's default random seed rather than the seed the example set, because the network was constructed before `main()` seeded the generator.
- [tinymind #239](https://github.com/danmcleran/tinymind/pull/239): with 16-bit storage, the examples' weight initialisation truncates `rand()` before reducing it, so Q8.8 weights fall outside their intended range of [-0.5, 0.5]. Every example tinymind ships in that configuration uses Q16.16 or overwrites the weights, so only the port's Q8.8 run was affected.
