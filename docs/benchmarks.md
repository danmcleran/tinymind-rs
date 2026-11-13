---
title: On-board results
layout: default
nav_order: 12
---

# On-board results

The crate has been run on two microcontroller boards, with the same boards, clocks, timing method, model weights and test samples as tinymind (C++)'s own [framework comparison](https://danmcleran.github.io/tinymind/benchmarks.html), and with compiler settings matched to its build (see [Build settings](#build-settings)), so the numbers sit next to the tinymind (C++) ones.

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
| 256-point FFT, forward transform: spectrum hash equals tinymind (C++)'s | match | match |
| fine-tuning heads: a dense training step and prediction, a prototype update and prediction, and a whole adaptation: hashes equal tinymind (C++)'s | match | match |
| MLPerf Tiny keyword spotting (DS-CNN), translated from TFLite: output hash equals tinymind (C++)'s and TFLite's | match | does not fit |
| MLPerf Tiny image classification (ResNet-8), translated from TFLite: output hash equals tinymind (C++)'s and TFLite's | match | does not fit |

The int8 MLP is tinymind (C++)'s 64-32-16-10 digits classifier with per-channel requantizers (`QDensePerChannel`). tinymind (C++) and TensorFlow Lite for Microcontrollers produce the same output hash on these boards, so the Rust port matches both. For training, the board hashes the exact bytes of the weight trajectory the xor example writes, every weight after every step.

## int8 MLP

Cycles are the median of 33 warm inferences; code + weights excludes the harness and the test data; RAM is static data plus peak stack above the harness's own. Every tinymind (C++) and TFLM figure on this page is from tinymind (C++)'s `-O3` builds, as re-measured for its v18.2 release; its summary reports list RAM as the larger of its `-O2` and `-O3` builds, so the RAM figures here are recomputed from the `-O3` raw results.

**Cortex-M4F (F411, 360 test samples, 96.11% accuracy)**

| | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **19,149** | **191 µs** | **6,236 B** | **76 B** |
| tinymind (C++), scalar | 21,002 | 210 µs | 5,848 B | 152 B |
| tinymind (C++), +SMLAD | 10,227 | 102 µs | 5,896 B | 124 B |
| TFLM int8 (CMSIS-NN) | 14,030 | 140 µs | 40,624 B | 2,600 B |
| TFLM int8 (reference kernels) | 31,411 | 314 µs | 29,552 B | 2,132 B |

**Cortex-M0+ (L031, first 32 test samples, 96.88% accuracy)**

| | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **35,710** | **1.12 ms** | **5,948 B** | **232 B** |
| tinymind (C++) | 32,644 | 1.02 ms | 6,136 B | 224 B |
| TFLM (any build) | does not fit in 32 KB | – | 38.3–50.0 KB | – |
| emlearn float32 (software float) | 707,113 | 22.1 ms | 18,988 B | 972 B |

The same Rust code measured 36,811 cycles in a second image whose extra code moves it in flash; with one flash wait state the M0+ is sensitive to code placement, which tinymind (C++) also observed.

## int8 convolutions: a KWS-shaped CNN

tinymind (C++)'s keyword-spotting shaped benchmark network: a 20x20x1 input, a 3x3 convolution with 8 filters, 2x2 max pooling, a 3x3 depthwise and an 8-to-16 pointwise convolution, global average pooling and a 16-to-10 dense head, with per-channel requantizers and random weights. Same weights and inputs as the C++ benchmark; the output hash over its 8 test inputs equals tinymind (C++)'s and TFLM's on both boards.

| | F411 (100 MHz) | L031 (32 MHz) |
|---|---:|---:|
| **tinymind-rs**, cycles | **189,097** (1.89 ms) | **649,286** (20.3 ms) |
| tinymind (C++) scalar, `-O3` | 254,766 | 740,279 |
| tinymind (C++) +SMLAD, `-O3` | 181,853 | – (no SMLAD on the M0+) |
| TFLM int8 (CMSIS-NN), `-O3` | 576,368 | does not fit in 32 KB |
| **tinymind-rs**, code + weights / RAM | **8,056 B / 4,420 B** | **8,352 B / 4,848 B** |
| tinymind (C++) scalar `-O3`, flash / RAM | 6,608 B / 5,308 B | 8,220 B / 5,596 B |
| tinymind-rs at `opt-level = "s"`, cycles / code + weights | 537,118 / 4,372 B | 1,171,205 / 4,120 B |

Per layer (cycles, each layer timed separately; C++ scalar `-O3` in brackets):

| Layer | F411 | L031 |
|---|---:|---:|
| conv 3x3, 8 filters | 98,406 (183,748) | 400,956 (510,808) |
| max pool 2x2 | 9,527 (10,158) | 20,192 (24,921) |
| depthwise 3x3 | 22,061 (20,668) | 69,218 (72,544) |
| pointwise 8 -> 16 | 43,652 (41,835) | 138,649 (135,798) |
| global average pool | 2,583 (2,282) | 4,704 (5,756) |
| dense 16 -> 10 | 1,601 (1,302) | 3,347 (2,696) |

The Rust port takes 26% fewer cycles than tinymind (C++)'s portable build on the M4F and 12% fewer on the M0+, and is 4% behind its SMLAD build on the M4F without SIMD. It uses less RAM; its flash is 22% more than the C++'s at `-O3` on the M4F and 2% more on the M0+; at `opt-level = "s"` the network fits in 4.1-4.4 KB but runs 1.8-2.8x slower. The stem convolution gathers each window once and runs its filters in pairs; pointwise runs filter by filter, two pixels at a time, and depthwise channel by channel, with the weights, folded bias and prepared requantizer held across the pixels.

## MLPerf Tiny keyword spotting (DS-CNN)

The MLPerf Tiny keyword-spotting reference model (49x10 MFCC input, a 10x4 convolution, four depthwise-separable blocks, global average pooling, a 12-way softmax), translated from its int8 `.tflite` by `tools/import_tflite`. The output hash over 8 seeded random inputs equals tinymind (C++)'s, which TFLite's reference kernels also produce. C++ and TFLM figures are tinymind (C++)'s, measured on the same board. These are not MLPerf results: there is no accuracy run or energy measurement, and MLCommons has not verified them.

| F411, 100 MHz | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **10,790,589** | **107.9 ms** | **49,712 B** | **16,812 B** |
| tinymind (C++), scalar | 15,677,563 | 156.8 ms | 53,072 B | 17,288 B |
| tinymind (C++), +SMLAD | 8,011,750 | 80.1 ms | 53,452 B | 17,276 B |
| TFLM int8 (CMSIS-NN) | 9,583,099 | 95.8 ms | 148,864 B | 24,288 B |
| TFLM int8 (reference kernels) | 67,484,238 | 674.8 ms | 101,944 B | 23,764 B |

The Rust model takes 31% fewer cycles than tinymind (C++)'s portable build, in less flash and RAM; tinymind (C++)'s SMLAD build and TFLM with CMSIS-NN are faster. Most of its time is the four 64-channel pointwise layers, which run two filters at two pixels at a time, so each loaded value serves two products; they run in 8-bit and do not reach `SMLAD`, so the parallel-DSP option below does not change this model.

## MLPerf Tiny image classification (ResNet-8)

The MLPerf Tiny image-classification reference model (32x32x3 CIFAR-10 input, a 3x3 convolution, three residual blocks of 3x3 convolutions with 16, 32 and 64 channels, the last two with a strided 1x1 shortcut, global average pooling, a 10-way softmax), translated the same way; its three residual additions run on `QAdd`. Same method and caveats as above.

| F411, 100 MHz | Cycles | Time | Code + weights | RAM |
|---|---:|---:|---:|---:|
| **tinymind-rs** | **58,807,476** | **588.1 ms** | **130,504 B** | **57,428 B** |
| tinymind-rs, `-C llvm-args=-arm-parallel-dsp-load-limit=512` | 38,273,811 | 382.7 ms | 125,496 B | 57,428 B |
| tinymind (C++), scalar | 85,326,662 | 853.3 ms | 124,140 B | 54,352 B |
| tinymind (C++), +SMLAD | 26,368,775 | 263.7 ms | 112,352 B | 54,400 B |
| TFLM int8 (CMSIS-NN) | 37,372,049 | 373.7 ms | 191,212 B | 56,464 B |
| TFLM int8 (reference kernels) | 191,555,451 | 1,915.6 ms | 155,412 B | 55,372 B |

The Rust model takes 31% fewer cycles than tinymind (C++)'s portable build, in 5% more flash and 6% more RAM. Its convolutions over several input channels widen the input rows and each filter's weights to 16 bits and add products in pairs, and its residual additions look each input's rescaling up in a 256-entry table. The translator plans one byte arena for the feature maps, as tinymind (C++) does (49,152 B), and views it as typed tensors without `unsafe`.

On cores with the DSP extension (Cortex-M4, M7, M33), LLVM can turn those pairs of 16-bit products into `SMLAD`, the instruction behind tinymind (C++)'s +SMLAD build, from plain Rust: no intrinsics, no nightly, no `unsafe`. By default it gives up on long unrolled sums; building with `-C llvm-args=-arm-parallel-dsp-load-limit=512` (in `.cargo/config.toml`) lets it pair them, and the model then takes 38.3M cycles, 55% fewer than the portable C++ but still 1.45 times tinymind (C++)'s +SMLAD build, which also reads four int8 values per load. That option is an LLVM internal setting, not a stable Rust one, and only this model's convolutions use it so far: it leaves the KWS-shaped network and the DS-CNN unchanged.


## On-device training

The xor network (2-4-1, tanh hidden, sigmoid output, SGD with momentum), trained on the board:

| | F411 Q16.16 | F411 Q8.8 | L031 Q16.16 | L031 Q8.8 |
|---|---:|---:|---:|---:|
| Cycles per training step | 1,226 | 475 | 5,974 | 3,054 |
| Time per step | 12.3 µs | 4.8 µs | 187 µs | 95 µs |
| Inference cycles | 1,104 | 368 | 4,149 | 1,572 |

Cycles per step are averages over all 20,000 steps. The example skips training while its error is within the fixed-point zero tolerance, and both networks solve XOR within about 2,000 steps (see the [XOR example]({{ site.baseurl }}/xor)). Inference time depends on where the activations fall in the tanh and sigmoid tables, so it moves with the trained weights.

## Anomaly detection, trained on the device

The [anomaly-detection example]({{ site.baseurl }}/anomaly) on the boards: an 8-4-8 Q16.16 autoencoder trains for 3,000 steps on the device, then screens 512 windows. Its report and hash (every trained weight and every score) match the host run on both boards.

| | F411 | L031 |
|---|---:|---:|
| Training, 3,000 steps (signal generation included) | 0.28 s | 4.6 s |
| One training step | 8,811 cycles | 39,763 cycles |
| One detection | 1,611 cycles | 8,096 cycles |

## Fine-tuning heads

The [fine-tuning heads]({{ site.baseurl }}/fine-tuning) of the gas sensor example, `DenseHead<Q16_16, 32, 6>` and `PrototypeHead<Q16_16, 32, 6>`, on 30 labelled 32-value embeddings (6 classes, 5 each). tinymind (C++)'s figures come from its own benchmark harness and board code running its heads on the same data, gcc `-O3`; every hash matches. Measured 2026-11-13.

| Median cycles | F411 Rust | F411 C++ | L031 Rust | L031 C++ |
|---|---:|---:|---:|---:|
| `DenseHead` training step | **10,129** (101 µs) | 7,391 | **52,389** (1.64 ms) | 52,223 |
| the same, with the embedding's gradient | **13,339** | 9,190 | **68,785** | 68,591 |
| `DenseHead` prediction | **2,866** | 2,247 | **15,580** | 15,864 |
| `PrototypeHead` update | **501** | 3,894 | **7,760** | 20,074 |
| `PrototypeHead` prediction | **3,245** | 2,281 | **15,693** | 16,063 |
| Whole adaptation, 30 epochs × 30 samples | **89 ms** | 65 ms | **1.44 s** | 1.45 s |

A whole adaptation, 900 training steps, takes 89 ms on the M4F and 1.44 s on the M0+; a `PrototypeHead` learns a sample in 5 µs and 243 µs. On the M0+ the Rust and the C++ are level, except the prototype update, where the Rust divides in 32 bits and the C++ in 64 (61% fewer cycles). On the M4F the Rust dense step takes 37% more cycles than the C++: most of the prediction gap is LLVM's machine scheduler for the Cortex-M4 (the same issue as the anomaly demo's), and the details are in [`bench/README.md`](https://github.com/danmcleran/tinymind-rs/tree/master/bench).

## Quantized FFT

One 256-point forward transform of `int8::QFft1D` on random 16-bit samples, with tinymind (C++)'s twiddle tables; the spectrum's hash equals the one tinymind (C++) computes. tinymind (C++)'s figures come from its own benchmark harness and board code, running `QFFT1D<256>::forward` on the same signal and tables. Measured 2026-11-12, prefetch on.

| Forward transform, cycles | F411 | L031 |
|---|---:|---:|
| **tinymind-rs**, `opt-level = 3` | **41,991 (420 µs)** | **91,430 (2.86 ms)** |
| **tinymind-rs**, `opt-level = "s"` | **44,255** | **96,784** |
| tinymind (C++), gcc `-O3`, best measured | about 82,900 | about 101,600 |
| tinymind (C++), gcc `-Os`, best measured | about 85,200 | 142,482 |

On the Cortex-M0+ the Rust transform takes about 10% fewer cycles than the best gcc `-O3` build of tinymind (C++), and 32% fewer than its best `-Os` build. On the Cortex-M4F it takes about half the cycles of either. Its code is 2,056 B at `opt-level = "s"` and 3,808 B at 3 on the M0+, where the C++'s is 392 B and 584 B plus a 1 KB bit-reversal table that the Rust transform does without. The squared magnitudes of the 256 bins take 2,731 and 4,863 cycles. The transform works in the caller's buffers with no memory of its own.

gcc's code for tinymind (C++)'s FFT loop depends on the function it is inlined into: on the M0+ at `-O3` the same source measured from 101,600 to 193,277 cycles in different callers, from a loop that keeps its state in registers to one that spills it and reloads constants every butterfly. The table gives the best caller found; the details are in [`bench/README.md`](https://github.com/danmcleran/tinymind-rs/tree/master/bench).

It runs each stage twiddle by twiddle, loading each twiddle pair once per stage where the C++ loads it for every butterfly, and with the stage's size as a constant, so that the M0+'s eight low registers hold the whole butterfly. On cores without a saturating instruction (the M0+) it also checks once per butterfly that both twiddle products fit in 16 bits, which they do except near full scale, and then skips the four saturations, which could not change anything; a Kani proof shows the two forms agree for every input. The butterflies of a stage are independent, so the results are the same. A layer built at run time rather than as a `static` takes 97,221 cycles on the M0+ at `opt-level = 3`, as the table addresses then need registers too.

## Decision trees, imported from scikit-learn

The [`imported_trees` example]({{ site.baseurl }}/trees) on the boards: a 90-tree gradient-boosted model (int8 inputs) and a 15-tree voting forest (int16), both 53 of 54 correct on the held-out Wine samples, as on the host. Cycles for one prediction:

| | F411 | L031 |
|---|---:|---:|
| Gradient boosting, walking the node pool | 8,828 (88 µs) | 17,780 (556 µs) |
| Gradient boosting, nested `if`s | 1,680 (17 µs) | 2,627 (82 µs) |
| Random forest, pool / nested `if`s | 2,054 / 461 | 3,936 / 820 |
| 36 trees (3 KB of nodes) read from flash / from an SRAM copy | 3,793 / 3,345 | 6,586 / 6,170 |

The model is read-only data, so the linker places it in flash, and the CPU reads it from there as it reads its instructions; nothing is copied to SRAM. Reading it from flash costs 13-14% on the F411 and 7% on the L031 compared with an SRAM copy, while the nested-`if` form is 4 to 7 times faster than the pool walk. The 7.5 KB gradient-boosting pool would not fit beside the stack in the L031's 8 KB of SRAM.

## Build settings

The Rust images are built with the same settings as tinymind (C++)'s benchmark images, so a difference in cycles comes from the code, not from the compiler options:

| | tinymind (C++), gcc | tinymind-rs, rustc |
|---|---|---|
| Optimisation | `-O3` | `opt-level = 3` |
| Core | `-mcpu=cortex-m4` / `-mcpu=cortex-m0plus` | `-C target-cpu=cortex-m4` / `cortex-m0plus` |
| Floating point | `-mfloat-abi=hard -mfpu=fpv4-sp-d16` / soft | the target (`thumbv7em-none-eabihf` / `thumbv6m-none-eabi`) |
| Whole program | header-only templates: the model is one translation unit | LTO, one codegen unit |
| Unused code | `-ffunction-sections -fdata-sections -Wl,--gc-sections` | the linker's default |
| Loop alignment (F411) | none | 16 bytes, see below |

The SMLAD rows are the exception, on both sides: tinymind (C++)'s +SMLAD build adds `-DTINYMIND_ENABLE_SIMD_ARM_DSP=1`, and the Rust parallel-DSP row adds `-C llvm-args=-arm-parallel-dsp-load-limit=512`.

## Settings for your own build

The crate cannot choose compiler settings for you: `-C target-cpu` and `-C llvm-args` come from your application's build (its `.cargo/config.toml` or `RUSTFLAGS`), and they apply to the whole program. On a Cortex-M4 they move these workloads both ways. F411 cycles, 2026-11-12, every output hash matching, each column against the first:

| | `target-cpu=cortex-m4` (the figures above) | no `target-cpu` (the `thumbv7em` target's generic tuning) | `cortex-m4`, scheduler off | `cortex-m4`, top-down scheduling |
|---|---:|---:|---:|---:|
| int8 MLP | 19,149 | +4.8% | +6.3% | −1.1% |
| KWS-shaped CNN | 189,097 | −1.5% | +4.0% | −7.3% |
| MLPerf DS-CNN | 10,790,589 | +32.3% | +1.0% | +10.1% |
| MLPerf ResNet-8 | 58,807,476 | +9.5% | +4.6% | +1.4% |
| Anomaly demo: training step | 8,811 | −26.9% | −27.2% | −26.0% |
| Anomaly demo: detection | 1,611 | −17.3% | −16.2% | −17.7% |

"Scheduler off" adds `-C llvm-args=-enable-misched=false`, "top-down" adds `-C llvm-args=-misched-prera-direction=topdown`.

- **Set `target-cpu` for your core.** No other column beats it on every int8 network, and leaving it out costs the DS-CNN a third. The KWS-shaped CNN is the exception: it runs faster without it or with top-down scheduling (the int8 MLP is 1% faster top-down too).
- **Part of that third is this crate's own code shape.** Pointwise layers with 32 or more input channels run two filters at two pixels at a time. Their inner loop keeps four running sums, and it fits in the registers only when LLVM schedules instructions before allocating registers, which `target-cpu=cortex-m4` turns on and the generic tuning does not. So that kernel makes the DS-CNN 9% faster with `target-cpu` and 8% slower without it (14,277,919 cycles, against 13,225,965 for the one-filter loop it replaced; with the generic tuning and `-C llvm-args=-enable-misched=true`, 13,379,028). Smaller layers, such as the KWS-shaped CNN's 8-channel pointwise layer, keep the one-filter loop.
- **The exception is small fixed-point (Q16.16) networks.** With `cortex-m4` (and `cortex-m3`, `cortex-m33`, `cortex-m35p`, which share its scheduling model) LLVM's instruction scheduler starts all of a small, fully unrolled layer's 64-bit products before narrowing any of them, and the products spill to the stack; this is [llvm/llvm-project#241851](https://github.com/llvm/llvm-project/issues/241851). If a network like the anomaly demo dominates your program, measure the top-down setting: it recovers the anomaly demo's loss and even speeds up the int8 MLP by 1%, but costs the DS-CNN 10%.
- **`-C llvm-args` options are LLVM internals**, with no stability promise: the top-down option was named `-misched-topdown` in older LLVM versions, and rustc rejects an option its LLVM does not know, so a Rust update can break a build that uses one. Measure on your own model before relying on one, and recheck when you update Rust.
- On the Cortex-M0+ none of this applies: `cortex-m0plus` does not run that scheduler, and the L031 images were identical with and without the top-down option.

## What these show

- Exactness holds on both cores: int8 inference and every training step match the C++ (and, for the anomaly demo, the host run) bit for bit on the chips themselves, not only on a desktop.
- Against tinymind (C++)'s portable C++ the Rust MLP takes 9% fewer cycles on the Cortex-M4F and is 9–13% slower on the Cortex-M0+; the KWS-shaped network takes 12-26% fewer cycles, the MLPerf DS-CNN 31% and ResNet-8 31% fewer on the M4F. It has no SMLAD backend of its own: the Arm DSP intrinsics are not available in stable Rust, and inline assembly would give up the crate's no-`unsafe` guarantee; only the 16-bit pairs of multi-channel convolutions reach `SMLAD`, through LLVM, with the opt-in option above.
- These are two boards, one model per task and one compiler version. Tuning for the core moves workloads both ways: on the F411, `target-cpu=cortex-m4` makes the int8 networks faster and the anomaly demo's training step 37% slower; see [Settings for your own build](#settings-for-your-own-build).
- F411 timings are taken with every loop aligned to 16 bytes, so they do not move when unrelated code changes shift the hot loops in flash (without it, a 10-byte shift once made the MLPerf DS-CNN 11% slower). The alignment padding costs 0-2% on most workloads, up to 7.8% on the tree walks and 12% at `opt-level = "s"`. The L031 figures, without it, stay sensitive to placement.

The harness, method and raw details are in [`bench/`](https://github.com/danmcleran/tinymind-rs/tree/master/bench) in the repository.
