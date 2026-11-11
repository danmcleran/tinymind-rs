---
title: On-board results
layout: default
nav_order: 11
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
| MLPerf Tiny keyword spotting (DS-CNN), translated from TFLite: output hash equals tinymind (C++)'s and TFLite's | match | does not fit |
| MLPerf Tiny image classification (ResNet-8), translated from TFLite: output hash equals tinymind (C++)'s and TFLite's | match | does not fit |

The int8 MLP is tinymind (C++)'s 64-32-16-10 digits classifier with per-channel requantizers (`QDensePerChannel`). tinymind (C++) and TensorFlow Lite for Microcontrollers produce the same output hash on these boards, so the Rust port matches both. For training, the board hashes the exact bytes of the weight trajectory the xor example writes, every weight after every step.

## int8 MLP

Cycles are the median of 33 warm inferences; code + weights excludes the harness and the test data; RAM is static data plus peak stack above the harness's own. Every tinymind (C++) and TFLM figure on this page is from tinymind (C++)'s `-O3` builds; its summary reports list RAM as the larger of its `-O2` and `-O3` builds, so the RAM figures here are recomputed from the `-O3` raw results.

**Cortex-M4F (F411, 360 test samples, 96.11% accuracy)**

| | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **21,502** | **215 µs** | **5,548 B** | **44 B** |
| tinymind (C++), scalar | 20,486 | 205 µs | 5,464 B | 152 B |
| tinymind (C++), +SMLAD | 10,241 | 102 µs | 5,548 B | 124 B |
| TFLM int8 (CMSIS-NN) | 14,030 | 140 µs | 40,624 B | 2,600 B |
| TFLM int8 (reference kernels) | 31,411 | 314 µs | 29,552 B | 2,132 B |

**Cortex-M0+ (L031, first 32 test samples, 96.88% accuracy)**

| | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **42,514** | **1.33 ms** | **5,412 B** | **472 B** |
| tinymind (C++) | 35,963 | 1.12 ms | 5,784 B | 200 B |
| TFLM (any build) | does not fit in 32 KB | – | 38.3–50.0 KB | – |
| emlearn float32 (software float) | 707,113 | 22.1 ms | 18,988 B | 972 B |

The same Rust code measured 38,456 cycles in a second image whose extra code moves it in flash; with one flash wait state the M0+ is sensitive to code placement, which tinymind (C++) also observed.

## int8 convolutions: a KWS-shaped CNN

tinymind (C++)'s keyword-spotting shaped benchmark network: a 20x20x1 input, a 3x3 convolution with 8 filters, 2x2 max pooling, a 3x3 depthwise and an 8-to-16 pointwise convolution, global average pooling and a 16-to-10 dense head, with per-channel requantizers and random weights. Same weights and inputs as the C++ benchmark; the output hash over its 8 test inputs equals tinymind (C++)'s and TFLM's on both boards.

| | F411 (100 MHz) | L031 (32 MHz) |
|---|---:|---:|
| **tinymind-rs**, cycles | **186,257** (1.86 ms) | **649,286** (20.3 ms) |
| tinymind (C++) scalar, `-O3` | 254,900 | 746,112 |
| tinymind (C++) +SMLAD, `-O3` | 181,568 | – (no SMLAD on the M0+) |
| TFLM int8 (CMSIS-NN), `-O3` | 576,368 | does not fit in 32 KB |
| **tinymind-rs**, code + weights / RAM | **8,152 B / 4,668 B** | **8,320 B / 4,848 B** |
| tinymind (C++) scalar `-O3`, flash / RAM | 6,848 B / 5,292 B | 8,452 B / 5,476 B |
| tinymind-rs at `opt-level = "s"`, cycles / code + weights | 546,503 / 4,308 B | 1,171,259 / 4,120 B |

Per layer (cycles, each layer timed separately; C++ scalar `-O3` in brackets):

| Layer | F411 | L031 |
|---|---:|---:|
| conv 3x3, 8 filters | 101,984 (173,164) | 400,956 (516,155) |
| max pool 2x2 | 10,578 (10,156) | 20,192 (24,921) |
| depthwise 3x3 | 22,807 (21,054) | 69,218 (73,079) |
| pointwise 8 -> 16 | 37,788 (44,258) | 138,649 (137,483) |
| global average pool | 2,507 (2,282) | 4,704 (5,756) |
| dense 16 -> 10 | 1,604 (1,281) | 3,347 (2,830) |

The Rust port takes 27% fewer cycles than tinymind (C++)'s portable build on the M4F and 13% fewer on the M0+, and is within 3% of its SMLAD build on the M4F without SIMD. It uses less RAM; its flash is 19% more than the C++'s at `-O3` on the M4F and 2% less on the M0+; at `opt-level = "s"` the network fits in 4.1-4.3 KB but runs 1.8-2.9x slower. The stem convolution gathers each window once and runs its filters in pairs; pointwise runs filter by filter, two pixels at a time, and depthwise channel by channel, with the weights, folded bias and prepared requantizer held across the pixels.

## MLPerf Tiny keyword spotting (DS-CNN)

The MLPerf Tiny keyword-spotting reference model (49x10 MFCC input, a 10x4 convolution, four depthwise-separable blocks, global average pooling, a 12-way softmax), translated from its int8 `.tflite` by `tools/import_tflite`. The output hash over 8 seeded random inputs equals tinymind (C++)'s, which TFLite's reference kernels also produce. C++ and TFLM figures are tinymind (C++)'s, measured on the same board. These are not MLPerf results: there is no accuracy run or energy measurement, and MLCommons has not verified them.

| F411, 100 MHz | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **13,225,965** | **132.3 ms** | **46,416 B** | **16,680 B** |
| tinymind (C++), scalar | 15,736,026 | 157.4 ms | 52,748 B | 17,264 B |
| tinymind (C++), +SMLAD | 8,193,064 | 81.9 ms | 53,156 B | 17,248 B |
| TFLM int8 (CMSIS-NN) | 9,583,099 | 95.8 ms | 148,864 B | 24,288 B |
| TFLM int8 (reference kernels) | 67,484,238 | 674.8 ms | 101,944 B | 23,764 B |

The Rust model takes 16% fewer cycles than tinymind (C++)'s portable build, in less flash and RAM; tinymind (C++)'s SMLAD build and TFLM with CMSIS-NN are faster. Most of its time is the four 64-channel pointwise layers (2.17M cycles each), which run in 8-bit and do not reach `SMLAD`.

## MLPerf Tiny image classification (ResNet-8)

The MLPerf Tiny image-classification reference model (32x32x3 CIFAR-10 input, a 3x3 convolution, three residual blocks of 3x3 convolutions with 16, 32 and 64 channels, the last two with a strided 1x1 shortcut, global average pooling, a 10-way softmax), translated the same way; its three residual additions run on `QAdd`. Same method and caveats as above.

| F411, 100 MHz | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **64,407,536** | **644.1 ms** | **130,184 B** | **57,692 B** |
| tinymind-rs, `-C llvm-args=-arm-parallel-dsp-load-limit=512` | 41,401,469 | 414.0 ms | 125,384 B | 57,700 B |
| tinymind (C++), scalar | 85,467,754 | 854.7 ms | 120,836 B | 54,380 B |
| tinymind (C++), +SMLAD | 26,753,191 | 267.5 ms | 108,324 B | 54,416 B |
| TFLM int8 (CMSIS-NN) | 37,372,049 | 373.7 ms | 191,212 B | 56,464 B |
| TFLM int8 (reference kernels) | 191,555,451 | 1,915.6 ms | 155,412 B | 55,372 B |

The Rust model takes 25% fewer cycles than tinymind (C++)'s portable build, in 8% more flash and 6% more RAM. Its convolutions over several input channels widen the input rows and each filter's weights to 16 bits and add products in pairs, and its residual additions look each input's rescaling up in a 256-entry table. The translator plans one byte arena for the feature maps, as tinymind (C++) does (49,152 B), and views it as typed tensors without `unsafe`.

On cores with the DSP extension (Cortex-M4, M7, M33), LLVM can turn those pairs of 16-bit products into `SMLAD`, the instruction behind tinymind (C++)'s +SMLAD build, from plain Rust: no intrinsics, no nightly, no `unsafe`. By default it gives up on long unrolled sums; building with `-C llvm-args=-arm-parallel-dsp-load-limit=512` (in `.cargo/config.toml`) lets it pair them, and the model then takes 41.4M cycles, 52% fewer than the portable C++ but still 1.5 times tinymind (C++)'s +SMLAD build, which also reads four int8 values per load. That option is an LLVM internal setting, not a stable Rust one, and only this model's convolutions use it so far.


## On-device training

The xor network (2-4-1, tanh hidden, sigmoid output, SGD with momentum), trained on the board:

| | F411 Q16.16 | F411 Q8.8 | L031 Q16.16 | L031 Q8.8 |
|---|---:|---:|---:|---:|
| Cycles per training step | 1,221 | 483 | 5,974 | 3,054 |
| Time per step | 12.2 µs | 4.8 µs | 187 µs | 95 µs |
| Inference cycles | 1,110 | 371 | 4,149 | 1,572 |

Cycles per step are averages over all 20,000 steps. The example skips training while its error is within the fixed-point zero tolerance, and both networks solve XOR within about 2,000 steps (see the [XOR example]({{ site.baseurl }}/xor)). Inference time depends on where the activations fall in the tanh and sigmoid tables, so it moves with the trained weights.

## Anomaly detection, trained on the device

The [anomaly-detection example]({{ site.baseurl }}/anomaly) on the boards: an 8-4-8 Q16.16 autoencoder trains for 3,000 steps on the device, then screens 512 windows. Its report and hash (every trained weight and every score) match the host run on both boards.

| | F411 | L031 |
|---|---:|---:|
| Training, 3,000 steps (signal generation included) | 0.21 s | 4.6 s |
| One training step | 6,437 cycles | 39,763 cycles |
| One detection | 1,332 cycles | 8,096 cycles |

## Decision trees, imported from scikit-learn

The [`imported_trees` example]({{ site.baseurl }}/trees) on the boards: a 90-tree gradient-boosted model (int8 inputs) and a 15-tree voting forest (int16), both 53 of 54 correct on the held-out Wine samples, as on the host. Cycles for one prediction:

| | F411 | L031 |
|---|---:|---:|
| Gradient boosting, walking the node pool | 9,276 (93 µs) | 17,780 (556 µs) |
| Gradient boosting, nested `if`s | 1,841 (18 µs) | 2,627 (82 µs) |
| Random forest, pool / nested `if`s | 2,050 / 450 | 3,936 / 820 |
| 36 trees (3 KB of nodes) read from flash / from an SRAM copy | 3,549 / 3,094 | 6,586 / 6,170 |

The model is read-only data, so the linker places it in flash, and the CPU reads it from there as it reads its instructions; nothing is copied to SRAM. Reading it from flash costs 7-15% on the F411 and 7% on the L031 compared with an SRAM copy, while the nested-`if` form is 4 to 7 times faster than the pool walk. The 7.5 KB gradient-boosting pool would not fit beside the stack in the L031's 8 KB of SRAM.

## What these show

- Exactness holds on both cores: int8 inference and every training step match the C++ (and, for the anomaly demo, the host run) bit for bit on the chips themselves, not only on a desktop.
- Against tinymind (C++)'s portable C++ the Rust MLP is 5% slower on the Cortex-M4F and 7–18% slower on the Cortex-M0+, at about the same code size; the KWS-shaped network takes 13-27% fewer cycles, the MLPerf DS-CNN 16% and ResNet-8 25% fewer on the M4F. It has no SMLAD backend of its own: the Arm DSP intrinsics are not available in stable Rust, and inline assembly would give up the crate's no-`unsafe` guarantee; only the 16-bit pairs of multi-channel convolutions reach `SMLAD`, through LLVM, with the opt-in option above.
- These are two boards, one model per task and one compiler version.
- F411 timings are taken with every loop aligned to 16 bytes, so they do not move when unrelated code changes shift the hot loops in flash (without it, a 10-byte shift once made the MLPerf DS-CNN 11% slower). The alignment padding costs 0-2% on most workloads, up to 7.8% on the tree walks and 12% at `opt-level = "s"`. The L031 figures, without it, stay sensitive to placement.

The harness, method and raw details are in [`bench/`](https://github.com/danmcleran/tinymind-rs/tree/master/bench) in the repository.
