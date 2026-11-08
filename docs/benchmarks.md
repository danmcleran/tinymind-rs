---
title: On-board results
layout: default
nav_order: 9
---

# On-board results

The crate has been run on two microcontroller boards, with the same boards, clocks, timing method, model weights and test samples as tinymind (C++)'s own [framework comparison](https://danmcleran.github.io/tinymind/benchmarks.html), so the numbers sit next to the tinymind (C++) ones.

| Board | Core | Clock |
|---|---|---|
| NUCLEO-F411RE | Cortex-M4F | 100 MHz |
| NUCLEO-L031K6 | Cortex-M0+ (no DSP extension, no hardware divide; 32 KB flash, 8 KB RAM) | 32 MHz |

## Exactness

| Check | F411 | L031 |
|---|---|---|
| int8 digits MLP: output hash over every test sample equals tinymind (C++)'s | match | match |
| xor training, Q16.16: all 20,000 steps equal the golden trajectory | match | match |
| xor training, Q8.8: all 20,000 steps equal the golden trajectory | match | match |
| anomaly detection: training and every decision equal the host run | match | match |

The int8 MLP is tinymind (C++)'s 64-32-16-10 digits classifier with per-channel requantizers (`QDensePerChannel`). tinymind (C++) and TensorFlow Lite for Microcontrollers produce the same output hash on these boards, so the Rust port matches both. For training, the board hashes the exact bytes of the weight trajectory the xor example writes, every weight after every step.

## int8 MLP

Cycles are the median of 33 warm inferences; code + weights excludes the harness and the test data; RAM is static data plus peak stack above the harness's own.

**Cortex-M4F (F411, 360 test samples, 96.11% accuracy)**

| | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **21,386** | **214 µs** | **5,436 B** | **44 B** |
| tinymind (C++), scalar | 20,486 | 205 µs | 5,464 B | 152 B |
| tinymind (C++), +SMLAD | 10,241 | 102 µs | 5,548 B | 124 B |
| TFLM int8 (CMSIS-NN) | 14,030 | 140 µs | 40,624 B | 2,600 B |
| TFLM int8 (reference kernels) | 31,411 | 314 µs | 29,552 B | 2,152 B |

**Cortex-M0+ (L031, first 32 test samples, 96.88% accuracy)**

| | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **42,514** | **1.33 ms** | **5,412 B** | **472 B** |
| tinymind (C++) | 35,963 | 1.12 ms | 5.5–5.8 KB | 200 B |
| TFLM (any build) | does not fit in 32 KB | – | 33.9–50.0 KB | – |
| emlearn float32 (software float) | 707,113 | 22.1 ms | 18.7–19.0 KB | 972 B |

The same Rust code measured 38,456 cycles in a second image whose extra code moves it in flash; with one flash wait state the M0+ is sensitive to code placement, which tinymind (C++) also observed.

## On-device training

The xor network (2-4-1, tanh hidden, sigmoid output, SGD with momentum), trained on the board:

| | F411 Q16.16 | F411 Q8.8 | L031 Q16.16 | L031 Q8.8 |
|---|---:|---:|---:|---:|
| Cycles per training step | 1,207 | 343 | 5,929 | 2,238 |
| Time per step | 12.1 µs | 3.4 µs | 185 µs | 70 µs |
| Inference cycles | 1,118 | 193 | 4,119 | 619 |

## Anomaly detection, trained on the device

The [anomaly-detection example]({{ site.baseurl }}/anomaly) on the boards: an 8-4-8 Q16.16 autoencoder trains for 3,000 steps on the device, then screens 512 windows. Its report and hash (every trained weight and every score) match the host run on both boards.

| | F411 | L031 |
|---|---:|---:|
| Training, 3,000 steps (signal generation included) | 0.14 s | 4.2 s |
| One training step | 4,249 cycles | 35,336 cycles |
| One detection | 1,332 cycles | 8,104 cycles |

## What these show

- Exactness holds on both cores: int8 inference and every training step match the C++ (and, for the anomaly demo, the host run) bit for bit on the chips themselves, not only on a desktop.
- Against tinymind (C++)'s portable C++ the Rust port is 4% slower on the Cortex-M4F and 7–18% slower on the Cortex-M0+, at about the same code size. It has no equivalent of tinymind (C++)'s SMLAD backend: the Arm DSP intrinsics are not available in stable Rust, and inline assembly would give up the crate's no-`unsafe` guarantee.
- These are two boards, one model per task and one compiler version.

The harness, method and raw details are in [`bench/`](https://github.com/danmcleran/tinymind-rs/tree/master/bench) in the repository.
