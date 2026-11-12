---
paths:
  - "bench/**"
---

# On-board checks (`bench/`)

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.

`bench/` is a separate firmware crate (not published; CI only builds it) that runs
on Dan's NUCLEO-F411RE (Cortex-M4F, ST-LINK serial `0667FF3833584B3043112826`) and
NUCLEO-L031K6 (Cortex-M0+, `066DFF485652867187112939`), both on USB. `bench/run.sh`
builds, flashes (OpenOCD, probe by serial) and reads the VCP. It mirrors tinymind's
`benchmarks/framework_comparison_f411` (same clocks, timing, weights, samples):
the int8 digits MLP must hash-match tinymind's expected hash, and xor training on
the board must reproduce the golden trajectory hash. The `anomaly` workload (2D)
runs `examples/common/anomaly_demo.rs`, shared with `examples/anomaly.rs`: its
hash must equal `EXPECT_HASH` in `bench/src/anomaly.rs`, which
`tests/anomaly_golden.rs` pins to the host run. That golden is the one exception
to "goldens come from the C++": the demo has no tinymind counterpart, so it is a
regression pin, said so in the test. The L031 anomaly image is tight on its 32 KB
flash (2026-11-10: 30,000 B, ~2.7 KB free): library growth can overflow it, and CI's
bench build will say so (`train_step` and `score` are `#[inline(never)]` for this
reason, and so is `optimizer::update_all`: the #264 sync's separate bias path
overflowed it by 952 B; inlined weight + bias loops by 928 B; a single chained
iterator fit but cost the F411 32% per step; the never-inlined slice helper costs
10%). #266's round-toward-zero momentum costs the F411 anomaly step +37%
(4,730 -> 6,460 cycles); its hash changed to 0xE57781D0 (stall 39/64, was 30)
and the xor trajectories too, all reproduced on both boards 2026-11-10.
#267's round-to-nearest SGD step: anomaly hash 0x7C074DAD (threshold 128,
stall 46/64), F411 step 6,437, L031 39,763 cycles; Q8.8 xor solves XOR again;
all reproduced on both boards 2026-11-10.
MLP after the dense one-rounding loop (2026-11-12): F411 19,149 (generic
20,073, misched off 20,358, top-down 18,930), L031 35,710 (profile image
36,811), RAM 232 B; raw in the session scratchpad only.
C++ figures refreshed to tinymind v18.2's re-measured `raw_*_O3.txt` (#274,
2026-11-12; RAM recomputed as above). Changes that moved a conclusion: L031
MLP C++ 35,963 -> 32,644 (#275 restored its -O2/-O3 speed), so Rust's 42,514 is
30% slower (was 18%); F411 MLP C++ 20,486 -> 21,002 (Rust 2.7% faster, was
level); KWS L031 Rust 12% fewer cycles (was 13%), flash +2% vs C++ (was -1%);
ResNet-8 flash +5% (was +8%), parallel-DSP 1.45x SMLAD (was 1.43x). The dated
history lines in these notes keep the figures of their day.
The `fft` workload (2026-11-12): `int8::QFft1D` forward at N=256 on the FFT
sweep's random signal, tinymind's twiddle tables and the expected hash
(0xEFABBA1C) generated into OUT_DIR by build.rs from tests/golden/fft (an
`include_str!` of the whole golden overflowed the L031 by 23 KB). F411 71,627
cycles forward, 2,731 magnitudes; L031 172,683 and 4,863; flash over baseline
+2,420 / +2,236 B; stack is the workload's buffers (~4 KB), the transform
uses none. `magnitude_squared` returning `[i32; N]` instead of filling the
caller's array took 4,539 / 7,267 (the copy out). Raw in bench/results/fft-2026-11-12. C++ FFT not timed on boards.
The `kws` workload (4C) is tinymind's framework-comparison KWS-shaped CNN
(`gen/out/kws_tinymind.hpp`, generated into `bench/src/gen/kws.rs` by `gen.py`;
hash `TM_KWS_HASH`), with per-layer cycles in `kws-profile`. After tuning, Rust
takes 13-27% fewer cycles than the C++ scalar build (F411 186,257 vs 254,900,
2.6% behind +SMLAD's 181,568; L031 649,286 vs 746,112; these are generic-core
figures, with target-cpu since 2026-11-11 F411 189,097: 26% fewer, 4.1% behind), in less RAM, 19% more
flash on the F411 and 2% less on the L031 (at `OPT=s bench/run.sh` 4.1-4.3 KB but
1.8-2.9x slower). 2026-11-09: one-channel stems gather each window once
(`conv2d_gather`: zp-filled padding, folded biases, filter pairs, `Requantizer::apply`
per output: an array of 64 prepared FastRequantizers cost the DS-CNN 3.1 KB of
stack for 0.4% speed) and pointwise runs two pixels per pass (`dot2_pixels`;
four spill on the M4; on the M0+ pointwise is 1% slower). DS-CNN 16.15M ->
13.23M; per layer stem 3.09M -> 1.77M, pointwise 2.60M -> 2.17M, depthwise 0.65M
(82 cycles/output: LLVM unrolls the 25x5 pixel loop and keeps the 9 weights on
the stack; a range-based edge path was slower, 0.70M). Tried and dropped: band
path for C=1 (KWS 286K), word loads + sign-extended lanes (LLVM narrows back to
ldrsb). Also measured worse (2026-11-09), so not adopted: depthwise two
adjacent pixels per pass (DS-CNN 13.23M -> 13.94M; KWS dw 22,807 -> 23,009);
depthwise pixel-outer with every channel's window gathered once (14.68M: the
channel-outer loop with weights and a prepared requantizer per channel wins
despite its per-tap edge checks); dense with the input widened to i16 once and
outputs in pairs (F411 MLP 21,517 vs 21,502, fc1/fc2 slower; L031 43,650 vs
42,514; +244 B stack); 16-bit blocked pointwise (only pays with the opt-in
parallel-DSP flag; by estimate 5-10% slower by default from re-widening the
weights per pixel block). Typed tensors cannot
share a ping-pong buffer, so `infer` scopes them (LLVM overlaps the slots); the
profile runs in its own `#[inline(never)]` frame, or the L031 overflows 8 KB.
The `mlperf-kws` and `mlperf-ic` workloads (F411 only) run the MLPerf DS-CNN
and ResNet-8 translated by `tools/import_tflite`: generate
`bench/src/gen/mlperf_{kws,ic}*.rs` with `tools/import_tflite/mlperf.py kws|ic
--out ... --data ...` first (needs a Python with `tflite`), so they are not in
run.sh's default workloads; CI generates them.
**F411 loop alignment** (2026-11-09): F411 timings depended on where hot loops
fall against the ART accelerator's 16-byte flash lines: a 10-byte code shift
moved the DS-CNN by 11%, and shifting all code by 4/8/12 bytes (cortex-m-rt's
`_stext` in memory.x) moved workloads by up to 1.9%. `bench/.cargo/config.toml`
now builds thumbv7em with `-C llvm-args=-align-loops=16` (Cargo joins it with
the cfg table's `-Tlink.x`; an env `CARGO_TARGET_..._RUSTFLAGS` also joins, so
repeating `-Tlink.x` there fails with "region 'FLASH' already defined"). With
it every timing was identical at offsets 0/4/8/12. Cost: 0-2% on most
workloads at opt 3, trees 2% (flash) / 7.8% (SRAM pool: short loops entered
per tree run the padding), 12% for KWS at opt s (744,585 vs 664,196), up to
1.7 KB flash. All F411 figures were re-measured with it. Not on the L031: 32-bit
flash fetches and 4-byte function alignment mean `_stext` cannot shift code by
2, so there is no sweep to show a gain, and its flash is nearly full.
**Build settings match the C++ Makefile** (Dan, 2026-11-11): `bench/.cargo/config.toml`
sets `-C target-cpu=cortex-m4` / `cortex-m0plus` (the C++ `-mcpu`); before that the
triples tuned for a generic core. Same-day A/B on the F411 (hashes all matched):
DS-CNN 13,225,965 -> 11,850,835, ResNet-8 64,407,536 -> 58,807,476 (parallel-DSP
41,401,469 -> 38,273,811), MLP 21,502 -> 20,437, trees inline 1,841 -> 1,680, pool
9,276 -> 8,828 (flash +5.4 KB); worse: KWS 186,257 -> 189,097 (pointwise 37,789 ->
43,652), 36-tree subset 3,549 -> 3,793, anomaly step 6,437 -> 8,811 (+37%), detect
1,332 -> 1,611. **Anomaly cause (2026-11-11): LLVM's pre-RA machine scheduler**,
which `cortex-m4` enables (generic thumbv7em does not). Same unrolling (76
`smull` in `score` both ways), but it issues a neuron's 8 i64 products before
narrowing any: `score` frame 36 -> 260 B, sp loads 32 -> 131; `train_step` frame
164 -> 580 B, sp loads 47 -> 226. `-enable-post-misched=false` changes nothing;
`-enable-misched=false` (target-cpu kept) on the board: step 6,415, detect 1,350,
train/step 6,994, but MLP 20,437 -> 22,428 (+9.7%), KWS 189,097 -> 196,624 (+4.0%),
xor/trees within 1.3%, all hashes matched. So the bench keeps the scheduler on;
documented as a trade-off. Per-function or per-crate `target-cpu`/scheduler
settings do not exist on stable Rust (Cargo profiles vary opt-level per package,
not rustflags). Code shapes tried to survive the scheduler (2026-11-11), all
dropped: `Dense::forward`'s per-neuron sum `#[inline(never)]` (bit-exact) made
the F411 anomaly worse under both settings (cortex-m4 step 9,042, detect 1,784;
generic 6,887 / 1,695: call overhead and inputs reloaded per neuron outweigh
the spills saved); on a standalone 8x4 Q16.16 reproducer, input-outer order
(acc[o] += x[i]*w[i][o]) and an opaque neuron-loop bound (`black_box`) left
cortex-m4 at 88-90 stack accesses (generic 15). Root cause is in LLVM:
CortexM4Model (cortex-m3/m4/m33/m35p) with FeatureUseMISched; `generic` uses
CortexA8Model without it. Reproduced on clang trunk and llc 18; clang's
default CPU for `-march=armv7e-m` is cortex-m4. Forcing top-down scheduling
(`-misched-topdown` on llc 18, `-misched-prera-direction=topdown` on newer
LLVM) removes nearly all spills on the reproducer (llc 18: 9 stack accesses vs
88; clang trunk: none); bottom-up equals the default, so the bottom-up side
hoists the products. F411 with `-C llvm-args=-misched-prera-direction=topdown`
(2026-11-12, target-cpu kept, hashes matched): anomaly step 6,518, detect
1,326; MLP 20,437 -> 20,835 (+1.9%); KWS
189,097 -> 175,319 (-7.3%, below the C++ +SMLAD 181,568); text anomaly 30,848,
mlp 42,216, kws 24,668. Full A/B the same day (bench/results/topdown-2026-11-12/,
local): DS-CNN 10,790,589 -> 11,875,413 (+10.1%), ResNet-8 58,807,476 ->
59,614,538 (+1.4%), MLP profile +1.7%, KWS profile 182,002 -> 180,069, trees
-2.4%..+0.5% (subset flash 3,793 -> 3,807), xor within 0.3%; anomaly stack
4,304 -> 3,680 B, xor-q16 flash -960 B. L031: every ELF byte-identical: cortex-m0plus
lacks `use-misched`, so the pre-RA machine scheduler skips every function and the
direction flag has nothing to steer (llc 18 on repro.ll: default output ==
`-enable-misched=false`, byte for byte). Forced on (`-enable-misched=true`), it
reorders code but spills the same: ARMv6-M has no SMULL, so every 64-bit
product is a `bl __aeabi_lmul` call, and calls end scheduling regions. Not adopted: the DS-CNN loss outweighs the KWS gain, and the
C++ Makefile has no counterpart. Filed by Dan as
llvm/llvm-project#241851 (2026-11-12; LLVM's AI Tool Use Policy: agents draft,
the human reviews and submits, with a disclosure line). Reproducer and
run.sh: ~/code/llvm-m4-misched/ (local, outside the repo). Re-measured
2026-11-12 before filing, all to the cycle: anomaly step/detect generic
6,437/1,332, cortex-m4 8,811/1,611, misched off 6,415/1,350; MLP 20,437 ->
22,428 and KWS 189,097 -> 196,624 with misched off.
Results (local, gitignored): bench/results/misched-off-2026-11-11/. L031: code differs (KWS +416 B, 224 of
9,406 instructions) but every timing identical to the cycle; anomaly image 30,392 B
(2.3 KB free). `CARGO_TARGET_<TRIPLE>_RUSTFLAGS` joins the config's flags, so to
measure without `target-cpu` edit the config, do not override by env. The
parallel-DSP flag changes only ResNet-8 (KWS and DS-CNN identical with it).
**C++ figures are `-O3` only** (Dan, 2026-11-09): every tinymind (C++) or TFLM number compared with this crate must come from an `-O3` build (host timings too: the examples' Makefile release flags, never `-O2`). tinymind's `results/REPORT*.md` print RAM as max(O2, O3) (`report.py`), so take RAM from `results/raw_*_O3.txt` as (static_ram - baseline) + (stack_bytes - baseline); cycles and flash there are per level. An audit on 2026-11-09 found every cycle figure at `-O3` and fixed eight RAM cells and three L031 flash ranges that had come from the summary. `WORKLOADS="anomaly" bench/run.sh f411 l031` runs one workload. Results and method are in
`bench/README.md`; rerun after changing arithmetic and update the tables.
`bench/gen.py` generates the MLP from tinymind's headers (never hand-copied).
probe-rs is not installed; OpenOCD and arm-none-eabi binutils are.
