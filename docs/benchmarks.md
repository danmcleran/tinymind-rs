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
| imported decision trees: every score and class on the 54 Wine samples equals the host run | match | match |
| int8 KWS-shaped CNN (convolutions): output hash equals tinymind (C++)'s | match | match |

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

## int8 convolutions: a KWS-shaped CNN

tinymind (C++)'s keyword-spotting shaped benchmark network: a 20x20x1 input, a 3x3 convolution with 8 filters, 2x2 max pooling, a 3x3 depthwise and an 8-to-16 pointwise convolution, global average pooling and a 16-to-10 dense head, with per-channel requantizers and random weights. Same weights and inputs as the C++ benchmark; the output hash over its 8 test inputs equals tinymind (C++)'s and TFLM's on both boards.

| | F411 (100 MHz) | L031 (32 MHz) |
|---|---:|---:|
| **tinymind-rs**, cycles | **255,516** (2.56 ms) | **729,806** (22.8 ms) |
| tinymind (C++) scalar, `-O3` | 254,900 | 746,112 |
| tinymind (C++) +SMLAD, `-O3` | 181,568 | – (no SMLAD on the M0+) |
| TFLM int8 (CMSIS-NN), `-O3` | 576,368 | does not fit in 32 KB |
| **tinymind-rs**, code + weights / RAM | **5,168 B / 4,660 B** | **5,292 B / 4,752 B** |
| tinymind (C++) scalar `-O3`, flash / RAM | 6,848 B / 5,332 B | 8,452 B / 5,488 B |

Per layer (cycles, each layer timed separately; C++ scalar `-O3` in brackets):

| Layer | F411 | L031 |
|---|---:|---:|
| conv 3x3, 8 filters | 112,051 (173,164) | 414,052 (516,155) |
| max pool 2x2 | 10,714 (10,156) | 20,192 (24,921) |
| depthwise 3x3 | 46,049 (21,054) | 102,429 (73,079) |
| pointwise 8 -> 16 | 78,614 (44,258) | 184,693 (137,483) |
| global average pool | 2,505 (2,282) | 4,703 (5,756) |
| dense 16 -> 10 | 1,479 (1,281) | 3,360 (2,830) |

Overall the Rust port runs level with tinymind (C++)'s portable build (0.2% slower on the M4F, 2.2% faster on the M0+) in less flash and RAM. Per layer it differs: the stem convolution is 20-35% faster, while depthwise and pointwise are 1.3-2.2x slower, because the C++ hoists each filter's weights and requantizer out of the pixel loop. tinymind (C++)'s SMLAD build takes 29% fewer cycles on the M4F.

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

## Decision trees, imported from scikit-learn

The [`imported_trees` example]({{ site.baseurl }}/trees) on the boards: a 90-tree gradient-boosted model (int8 inputs) and a 15-tree voting forest (int16), both 53 of 54 correct on the held-out Wine samples, as on the host. Cycles for one prediction:

| | F411 | L031 |
|---|---:|---:|
| Gradient boosting, walking the node pool | 9,095 (91 µs) | 17,780 (556 µs) |
| Gradient boosting, nested `if`s | 1,841 (18 µs) | 2,627 (82 µs) |
| Random forest, pool / nested `if`s | 2,024 / 462 | 3,936 / 820 |
| 36 trees (3 KB of nodes) read from flash / from an SRAM copy | 3,480 / 3,129 | 6,586 / 6,170 |

The model is read-only data, so the linker places it in flash, and the CPU reads it from there as it reads its instructions; nothing is copied to SRAM. Reading it from flash costs 11-13% on the F411 and 7% on the L031 compared with an SRAM copy, while the nested-`if` form is 4 to 7 times faster than the pool walk. The 7.5 KB gradient-boosting pool would not fit beside the stack in the L031's 8 KB of SRAM.

## What these show

- Exactness holds on both cores: int8 inference and every training step match the C++ (and, for the anomaly demo, the host run) bit for bit on the chips themselves, not only on a desktop.
- Against tinymind (C++)'s portable C++ the Rust MLP is 4% slower on the Cortex-M4F and 7–18% slower on the Cortex-M0+, at about the same code size; the convolutional network runs level with it (0.2% slower on the M4F, 2.2% faster on the M0+) in less flash and RAM. It has no equivalent of tinymind (C++)'s SMLAD backend: the Arm DSP intrinsics are not available in stable Rust, and inline assembly would give up the crate's no-`unsafe` guarantee.
- These are two boards, one model per task and one compiler version.

The harness, method and raw details are in [`bench/`](https://github.com/danmcleran/tinymind-rs/tree/master/bench) in the repository.
