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
The `kws` workload (4C) is tinymind's framework-comparison KWS-shaped CNN
(`gen/out/kws_tinymind.hpp`, generated into `bench/src/gen/kws.rs` by `gen.py`;
hash `TM_KWS_HASH`), with per-layer cycles in `kws-profile`. After tuning, Rust
takes 13-27% fewer cycles than the C++ scalar build (F411 186,257 vs 254,900,
2.6% behind +SMLAD's 181,568; L031 649,286 vs 746,112), in less RAM, 19% more
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
**C++ figures are `-O3` only** (Dan, 2026-11-09): every tinymind (C++) or TFLM number compared with this crate must come from an `-O3` build (host timings too: the examples' Makefile release flags, never `-O2`). tinymind's `results/REPORT*.md` print RAM as max(O2, O3) (`report.py`), so take RAM from `results/raw_*_O3.txt` as (static_ram - baseline) + (stack_bytes - baseline); cycles and flash there are per level. An audit on 2026-11-09 found every cycle figure at `-O3` and fixed eight RAM cells and three L031 flash ranges that had come from the summary. `WORKLOADS="anomaly" bench/run.sh f411 l031` runs one workload. Results and method are in
`bench/README.md`; rerun after changing arithmetic and update the tables.
`bench/gen.py` generates the MLP from tinymind's headers (never hand-copied).
probe-rs is not installed; OpenOCD and arm-none-eabi binutils are.
