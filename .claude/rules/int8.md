---
paths:
  - "src/int8.rs"
  - "src/int16x8.rs"
  - "src/int16x8/**"
  - "tests/int16x8_sweep.rs"
  - "scripts/cpp/int16x8_sweep.cpp"
  - "src/int8/**"
  - "examples/kws_cortex_m_int8.rs"
  - "examples/int8_xor.rs"
  - "examples/common/kws_int8_driver.rs"
  - "examples/common/std_random.rs"
  - "examples/common/int8_xor_driver.rs"
  - "tests/conv_sweep.rs"
  - "tests/add_sweep.rs"
  - "tests/compose_sweep.rs"
  - "tests/recurrent_int8_sweep.rs"
  - "tests/softmax_sweep.rs"
  - "tests/mel_sweep.rs"
  - "tests/int4_sweep.rs"
  - "scripts/cpp/int4_sweep.cpp"
  - "tests/logmel_golden.rs"
  - "tests/vibration_golden.rs"
  - "examples/logmel_vibration.rs"
  - "examples/vibration_anomaly.rs"
  - "examples/common/logmel_driver.rs"
  - "examples/common/vibration_driver.rs"
  - "tests/qanomaly_sweep.rs"
  - "scripts/cpp/mel_sweep.cpp"
  - "scripts/cpp/anomaly_sweep.cpp"
  - "tests/softmax_tflite.rs"
  - "tests/kws_int8_golden.rs"
  - "tests/int8_xor_golden.rs"
  - "scripts/cpp/conv_sweep.cpp"
  - "scripts/cpp/add_sweep.cpp"
  - "scripts/cpp/compose_sweep.cpp"
  - "scripts/cpp/recurrent_sweep.cpp"
  - "scripts/cpp/softmax_sweep.cpp"
  - "scripts/cpp/kws_int8_dump.cpp"
  - "tests/fft_sweep.rs"
  - "tests/sweeps/fft.rs"
  - "scripts/cpp/fft_sweep.cpp"
---

# Int8 inference (`src/int8`)

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.

## `src/int8.rs`

C++ source: `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qactivations.hpp`, `cpp/include/simd/simd_dispatch.hpp` (scalar dot)

`TINYMIND_REF` 4cccbcd (v18.2, 2026-11-12): #272 (loop invariants to locals
in QAdd/QPad2D/QEmbedding), #273 (QAdd one-rounding input rescaling +
FastRequantizer output) and #275 (Thumb-1 split of QDense/pointwise loops,
`applyOneRounding`) change C++ code shape only; all 78 comparisons identical,
nothing to port (the Rust kernels have their own loop structure).

Int8 affine inference: gemmlowp primitives, `Requantizer` (i8 output only; both C++ paths, fused and saturating), `QDense<'a, IN, OUT>` borrowing `&[[i8; IN]; OUT]` weights (flash-friendly), `qrelu`/`qrelu6`, 256-entry LUT. Integer only, no_std

## (same, `QSoftmax`)

C++ source: `cpp/qsoftmax_fixedpoint.hpp`

TFLite-exact int8 softmax (`QSoftmaxFixedPoint1D`), one row per `forward::<N>` (the C++ `Rows` loop is the caller's); `N` in `1..512` checked at compile time. tinymind's other softmax, `QSoftmax1D` (exp LUT + int64 divide, TFLite to within rounding), is deliberately not ported. Checked by `tests/softmax_tflite.rs` (TFLite Micro goldens and sweep hashes, extracted by script from tinymind's unit test) and `tests/softmax_sweep.rs` (golden from `scripts/cpp/softmax_sweep.cpp`: tinymind's kernel + `buildQSoftmaxFixedPointParams`, 504 calibrations, rows of 1 to 511)

## `src/int8/calibration.rs`

C++ source: `cpp/include/qcalibration.hpp`, `cpp/qactivations.hpp`

Feature `std`: `quantize_multiplier` (with a ported `frexp`), `build_requantizer`, `quantize`/`dequantize`, sigmoid/tanh LUT builders, `build_softmax` (`buildQSoftmaxFixedPointParams`)

## (same, `QDensePerChannel`)

C++ source: `cpp/qpointwiseconv2d.hpp` (1x1, per channel)

Dense with one requantizer per output, as TFLite quantizes dense layers; verified on the boards against tinymind's benchmark hash

**One-rounding loop** (2026-11-12, mirrors tinymind #275): `QDense` and
`QDensePerChannel` check `Requantizer::one_rounding()` (all requantizers) once
per call and run `forward_with(apply_one_rounding)`, else `forward_with(apply)`
(generic closure, `#[inline(always)]`: two loop copies). With `apply`'s general
path inlined in the output loop, LLVM on the M0+ kept the unrolled dot
product's index constants (1..15, Thumb-1 `ldrsb` takes only a register
offset) in spilled stack slots: L031 MLP 42,514 -> 35,710 cycles, RAM 472 ->
232 B (stack 1,076 -> 836), +536 B flash; F411 20,437 -> 19,149, +784 B; KWS
unchanged; same hashes. The M0+ MAC is now 6 instructions like gcc's (C++
32,644: 9% gap left, per-output work and placement; profile image 36,811).
`apply_one_rounding` masks the shift into 2..=31 so it is total; Kani
`apply_one_rounding_matches_apply` (17 s; a planted `shift.min(30)` fails);
unit test `dense_requantizer_forms` (all fast, mixed, all general; planted
`any` for `all` fails). The `all` check is not folded for const layers (LLVM
reads the requantizers at run time, ~8 instructions each per call).

## `src/int8/conv.rs`, `src/int8/pool.rs` (re-exported from `int8`)

C++ source: `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qpool2d.hpp`

`QConv2D`/`QConv2DPerChannel`, `QDepthwiseConv2D`, `QPointwiseConv2D`/`...PerChannel`, `QMaxPool2D`, `QAvgPool2D`, `QGlobalAvgPool2D`. NHWC nested arrays (`[[[i8; C]; W]; H]`, conv weights OHWI); kernel, strides and padding are layer const params (defaults: stride 1 / pool stride = window, no padding); `forward(&self, input, &mut output)` takes H, W, OH, OW from the arrays and checks `OH == (H+PT+PB-KH)/SH+1` etc. in an inline `const { assert!() }` (stable since 1.79), so a wrong size or invalid layer is a compile error: this is how stable Rust gets around `OutputHeight` arithmetic. Loops tuned on the boards (4C, then 2026-11-09): conv is pixel-outer (window read once for all filters; filter-outer measured 20% slower for the Cin=1 stem), now as `conv2d_gather` (C=1) or `conv2d_band` (C>=2); the original per-tap `conv2d` was removed; depthwise and pointwise run filter-outer with the zero point folded into the bias (`bias - zp*sum w`, equal mod 2^32), weights held across pixels, interior windows read as slices (padded ones take the general per-tap path), and the requantizer prepared once per filter: `int8::FastRequantizer` (crate-private) is the C++ FastRequantizer extended to any qmin/qmax (fused ReLU), proved equal to `Requantizer::apply` by Kani (`fast_requantizer_matches_apply`); filters without a fast form run a `#[cold]` out-of-line loop (one loop branching per output cost depthwise 27%). Pointwise on one pixel keeps the per-output form. **Pointwise 2x2** (2026-11-11): with `C >= PAIR_MIN_CHANNELS` (32) filters run in pairs, two pixels per pass (`dot2x2`, four accumulators; a pair needs both requantizers' fast form, else one at a time); `C` is checked first so short-row layers compile only the old loop (a runtime-gated version still cost KWS +1.9% F411 and ~1 KB flash). F411 with target-cpu: DS-CNN 11,850,835 -> 10,790,589 (-9%, +3,840 B flash); with C=8 pairs made KWS pointwise 43,652 -> 55,682 (+28%; L031 138,649 -> 150,631). C=16-31 and C>=32 on the M0+ not measured. **Without target-cpu (generic
thumbv7em tuning) pairs cost the DS-CNN 8%** (2026-11-12, F411, same day:
13,225,965 at 93cc218 -> 14,277,919): the generic tuning runs no pre-RA machine
scheduler, and the 2x2 inner loop (8x unrolled, 4 sums) spills 28 stack accesses
per 32 MACs (0 with cortex-m4); generic + `-C llvm-args=-enable-misched=true`
gives 13,379,028. Not the CortexM4Model bug (llvm#241851); the opposite case.
No cfg can tell the tunings apart (`rustc --print cfg` identical), so kept,
documented on docs/benchmarks.md. **Layers built at run time** (2026-11-12, F411,
cortex-m4; each KWS layer behind `black_box(&LAYER)`, per-layer cycle stamps
inside the fused `infer`): stem (`conv2d_gather`) 106,987 -> 222,117 (2.08x),
pointwise 43,809 -> 48,428 (+10.5%), depthwise 22,901 -> 24,156 (+5.5%), pools
and fc ~0; whole MLP 20,437 -> 20,685 (+1.2%). `static` layers are as fast as
`const`. Cause in the stem: LICM hoists the 72 weights out of the pixel loop as
sign-extended i32 stack slots (274 stack loads per pixel, ~1,000 instructions)
and `Requantizer::apply`'s general path is inlined 8 times. Tried and dropped:
prepared `[Option<FastRequantizer>; F]` in the gather (runtime 254K, worse; const
-3% stem, +2 KB flash); `apply`'s general path `#[cold] #[inline(never)]`
(runtime -1.7%, const unchanged, +2.3 KB). Documented (docs/int8.md, int8 module
docs), not fixed. Same session: inside the fused KWS `infer` the stem takes
106,987 vs 98,406 timed alone; top-down scheduling gives 99,591 (that is where
the KWS gain under top-down comes from). Tried: filter pairs one pixel at a time
(`dot2` per pixel): DS-CNN cortex-m4 13,627,172, generic 15,645,174, worse both. conv_sweep's `wide` (C=40) and `odd` (C=33, 5 filters, 9 pixels) reach pair/odd-filter/odd-pixel; plants (a01 from w1*x0, odd pixel against w0 twice) fail it; panic-never has a C=32 instance (planted `o1[f1]` fails the link); Kani `pointwise_pairs_never_panic` (C=32, 2 filters, 3 pixels, unwind 34; 286 s locally, 355 s with 3 filters; a planted out-of-range `o1[f1 + 1]` fails it). Same `i32` sums as the C++ modulo 2^32. Checked by `tests/conv_sweep.rs` (golden from `scripts/cpp/conv_sweep.cpp`: 57 configurations x 3 trials, every C++ path incl. patch/row/tiles/odd filters/all-padding windows, both requantizer paths and the C++ FastRequantizer vs general form, pool ranges with qmin > qmax); ten planted defects each fail it. **16-bit band path** (2026-11-09, for ResNet-8): `QConv2D`/`QConv2DPerChannel` with `C >= BAND_MIN_CHANNELS` (2) run `conv2d_band`: per output row the KH input rows widen to an i16 band (`x - zp`), per (row, filter) the weights to i16, interior windows are KH `dot16` calls over contiguous KW*C, edges tap by tap, cold general-requantizer rows tap by tap (small code). Scratch on stack ~2*KH*(W+KW)*C B; not size-limited. LLVM's ARMParallelDSP pass makes `SMLAD` from `acc + x[i]*w[i] + x[i+1]*w[i+1]` over i16 loads (plain safe Rust, opt 3), but it analyses at most 16 loads (`-arm-parallel-dsp-load-limit`, default 16), and fully unrolled constant-length sums exceed it, so by default the code is ldrsh/ldrsh/smlabb (~4.7 c/MAC); `-C llvm-args=-arm-parallel-dsp-load-limit=512` gives pure ldr/ldr/smlad (opt-in, documented; not in bench config). Tried and dropped: packed u32 words (needs `[u32; W*C/2]`, inexpressible; assembling words from i16 folds back to ldrsh), `black_box` length + multi-exit loop (SMLAD but loop overhead ~2.75 c/MAC), unsigned byte lanes (UXTB16 trick: LLVM splits to per-byte uxtb, worse), 2 filters x 2 pixels (68.0M default, 48.1M with flag vs 45.9M, +68 KB flash). Stable Rust exposes no Arm target features in cfg, so no DSP gate: measured on the L031 a 10x10x8 3x3 conv 675,064 vs 835,734 cycles on the old path (F411 303,574 vs 451,012), so the band path is used everywhere; C=3 stem also faster (2.75M vs 3.88M). Kani `conv_band_never_panics` (C=2, unwind 6, 69 s); conv_sweep plants (odd tail, zp not subtracted, band rows off by one) fail. These layers pad internally, so they never need `QPad2D` (tinymind's translator inserts it only when a padded band exceeds its buffer limit); `QPad2D` itself is ported in `src/int8/compose.rs`

## `src/int8/add.rs` (re-exported from `int8`)

C++ source: `cpp/qadd.hpp`, `buildQAddParams` in `cpp/include/qcalibration.hpp`

`QAdd` (TFLite ADD): not size-parameterized; `forward<T: Tensor>(&a, &b, &mut out)` over the sealed `int8::Tensor` (`[i8; N]`, `[[i8; C]; W]`, `[[[i8; C]; W]; H]`), one `T` so shapes must match at compile time (compile_fail doctest). The C++ `(a - zp) * (1 << left_shift)` is UB outside `0..=23`; here a negative left shift is 0 and a larger one saturates (`saturating_shl`, the same shift as in `multiply_by_quantized_multiplier`, Kani `saturating_shl_saturates` vs an i128 spec); the sum wraps. `multiply_by_quantized_multiplier` keeps its own copy: calling `saturating_shl` from it moved code by 10 bytes and the F411 DS-CNN took 11% more cycles (17.88M vs 16.12M; with `-C llvm-args=-align-loops=16` both builds take 16.197M: F411 timings depended on hot-loop alignment against flash lines; the bench now aligns loops, see On-board checks). A Kani proof that the two shifts agree took 566 s (symbolic 32x32 multiply), too slow for CI. From 256 elements `forward` tabulates `scaled_a`/`scaled_b` for all 256 int8 values (2 KB stack) and uses `FastRequantizer`: ResNet-8 adds 185 -> 22 cycles/element; `add_sweep` checks the element-wise path (chunks of 8) equals the table path; plants (table b from a, table off by one) fail; panic-never covers both paths. `calibration::build_add((scale, zp), (scale, zp), (scale, zp), qmin, qmax)` = `buildQAddParams` (f64 from f32 scales, left shift 20). Note make_mlperf.py computes the ADD parameters differently (TFLM's float32 `twice_max` and denominator), so the translator must follow it, not `build_add`. Checked by `tests/add_sweep.rs` (golden from `scripts/cpp/add_sweep.cpp`, built with UBSan clean: 48 calibrated layers x all 65,536 input pairs, 64 direct random-parameter layers, 16 saturating the requantizer's left shift); plants (b with a's multiplier or zero point, left shift 0 as 1, min for max scale, real_out off by 1e-7, real_a in f32, wrapping instead of saturating left shift) each fail it. Before this sweep nothing exercised the left-shift saturation of `multiply_by_quantized_multiplier` (a wrapping plant passed every suite)

## `src/int8/compose.rs` (re-exported from `int8`)

C++ source: `cpp/qmul.hpp`, `cpp/qpad.hpp`, `cpp/qconcat.hpp`, `buildQMulRequantizer`/`buildRescaler`

`QMul` (`forward<T: Tensor>`, FastRequantizer prepared once; product fits i32), `QPad2D<PT, PB, PL, PR> { pad_value }` (const assert OH = H+PT+PB, OW = W+PL+PR with checked adds), `QConcat` (C++ `QConcat2_2D`; const assert CA, CB > 0 and C = CA + CB). `calibration::build_mul_requantizer` and `build_rescaler` are thin wrappers over `build_requantizer` (same f64 ratio as the C++). Checked by `tests/compose_sweep.rs` (golden from `scripts/cpp/compose_sweep.cpp`, UBSan clean: 32 calibrated QMul on all 65,536 pairs, 48 direct, 5 concat shapes x 4, 6 pad shapes x 3); plants (b with a's zp, pad rows offset by PL, pad with 0, concat b with a's requantizer, rescaler scale nudged, both compile_fail doctests made valid) fail; Kani `compose_never_panics` (22 s); panic-never covers all three. Translator support: PAD (H/W only, pad value = output zp, input/output quantization equal) -> QPad2D, MUL (same shapes; ratio via `rq`) -> QMul, CONCATENATION (2 maps on channels, all quantized as the output: TFLite int8 copies then, else rescales in float) -> QConcat with identity requantizers; checked end to end by `tools/import_tflite/test_ops.py` against LiteRT's reference kernels (ai-edge-litert 2.3.0, `OpResolverType.BUILTIN_REF`) on three models built by `tflite_build.py` (flatbuffers via the `tflite` package's builders, no TensorFlow); plants (MUL ratio +0.1%, MUL ReLU dropped (visible only because fused-ReLU outputs are calibrated before the ReLU), pad 0, pad L/R swapped, concat order swapped) fail it

## `src/int8/recurrent.rs` (re-exported from `int8`)

C++ source: `cpp/qlstm.hpp`, `cpp/qgru.hpp`, `buildQLSTMParams`/`buildQGRUParams`/`quantizeQLSTMBiases`/`quantizeQGRUBiases` in `cpp/include/qcalibration.hpp`

2026-11-10, against `TINYMIND_REF` 7e83cb4. `QLstmCell<'a, IN, HID, C = i8>` (C++ `QLSTMCell`; cell state `C` through the sealed `CellStorage`, `i8` or `i16`) and `QGruCell<'a, IN, HID>` (`QGRUCell`, reset before the R_n multiply): one step, state updated in place, caller runs the time loop. Gate-major weights as nested arrays (`[[[i8; IN]; HID]; 4]`, gates i f g o / r z n), so no `4 * HID` const arithmetic. Field names = the C++ ones (as `QAdd`); C++ storage template params other than the cell state are fixed (int8 in/weights/hidden, i32 acc), the only instantiations tinymind uses. LUT output grids fixed (sigmoid 1/256 zp -128, tanh 1/128 zp 0). Both share `gate_preactivations` (`dot_with_zero_point` + two `multiply_by_quantized_multiplier` + bias, saturated to i8); all gates computed before any unit is written (as the C++). Sums the C++ leaves to UB wrap; clamps are two `if`s (qmin > qmax gives qmax). Calibration: `LstmScales`/`LstmParams`/`build_lstm_params`, `GruScales`/`GruParams`/`build_gru_params`, `quantize_gate_biases` (generic over G; f64 divide, round half away, `as i64 as i32` = lround + int32 cast), `SIGMOID_LUT_OUTPUT_SCALE`/`TANH_LUT_OUTPUT_SCALE`. Checked by `tests/recurrent_int8_sweep.rs` (golden from `scripts/cpp/recurrent_sweep.cpp`, UBSan+ASan clean: LSTM 3x4 i8/i16, 1x1 i8, 2x3 i16, 8x16 i8, 16x8 i16, 5x7 i8; GRU 3x4, 1x1, 2x3, 8x16, 16x8, 5x7; 2 calibrated + 2 direct trials each, 32 steps, state hashed per step; shifts >= -3 so no C++ sum overflows); matched the C++ on the first run. Plants that fail it: g gate on the cell tanh LUT, cell zp not added, LSTM clamp order swapped, GRU 1-z off by one, n gate reading h not r*h, r*h without zp, LSTM output rescale with cell scale, biases truncated. Kani `lstm_int8_cell_never_panics`, `lstm_int16_cell_never_panics`, `gru_cell_never_panics` (1 input, 2 units, every field symbolic incl. LUTs; also state within [qmin, qmax] when qmin <= qmax; ~70-105 s each); panic-never covers i8/i16 LSTM and GRU at 3x16 and 5x7 (planted reachable index panics in each fail the link). The PyTorch importer emits the LSTM cell since 2026-11-14 (`importers.md`); not the GRU. PyTorch `nn.LSTM` gate order (i f g o) matches; `nn.GRU` is reset-after (`r * (R_n h + b_hn)`), tinymind's is reset-before, so a PyTorch GRU does not map. TFLite's integer LSTM kernel was not compared with QLSTMCell here; translating one needs a check against TFLite

## `src/int8/fft.rs` (re-exported from `int8`)

C++ source: `cpp/qfft1d.hpp`, `buildQFFTTwiddles` in `cpp/include/qcalibration.hpp`

2026-11-12, against `TINYMIND_REF` 7e83cb4 (qfft1d.hpp unchanged since #222's
C++14 guard; #127 saturates magnitudeSquared). `QFft1D<'a,
N, HALF> { twiddle_cos: &[i16; HALF], twiddle_sin }`: public fields, usable in a
`static`; `HALF` is a second const param because `[i16; N / 2]` needs
`generic_const_exprs`; `STAGES` associated const asserts N >= 2, power of two,
HALF == N/2 (post-mono error; compile_fail doctest). `forward` (scaled,
`saturate((top ± t) >> 1)`), `inverse` (wrapping_neg imag, unscaled, negate
again), `magnitude_squared(&self, ..)` (C++ static; a method here
because an associated fn would need HALF spelled out). q15 = `(a*b) >> 15` in
i32: |a|,|b| <= 32768 so no overflow anywhere (Kani). Bit reversal without the
C++ `size_t[N]` table: reverse-carry increment of j (no RBIT on v6-M),
swapped by masked indices; `slice::swap` kept a panic path at opt-level z (not
inlined). Butterfly indices masked (`& (N-1)`, twiddle `& (HALF-1)`) so no
bounds checks: identity masks, no panic path at 3/s/z. Tried: groups via
`chunks_exact_mut` + `split_at_mut_checked` + zipped iterators: slower (F411
73,938 vs 71,628; L031 174,112 vs 172,685) and kept a panic at z; dropped.
`magnitude_squared(&self, re, im, &mut [i32; N])` fills the caller's array as
the C++ does (returning it cost the copy: L031 7,267 vs 4,863 cycles).
**Twiddle-outer loop** (2026-11-12): per stage `for pair` then `top in
(pair..N).step(full)` (each twiddle loaded once per stage; `butterfly()`
helper, `#[inline(always)]`): F411 71,627 -> 59,965, L031 172,683 -> 153,696,
flash -80/+12 B, same hashes; sweep plants (top start `pair & !1`, twiddle
index +1) fail it; panic-never 3/s/z ok. **C++ timed on the boards**: copy
of tinymind's benchmark harness in a scratch dir (board/, fw/harness.cpp,
Makefile with TM=../tinymind read-only) plus a `be_fft.cpp` backend
(`QFFT1D<256>`, tables and input from tests/golden/fft, hash 0xEFABBA1C
matched): F411 107,265, L031 166,833 at -O3 (image +3.3 KB over baseline incl.
1 KB input array and the 1 KB `size_t[256]` bit-reversal table). Rust 44% /
8% fewer. The static `ldr` count (Rust 116 vs gcc 42 on M0+) was a poor
predictor: the C++ was the slower one. **Constant stage sizes** (2026-11-12):
`transform` dispatches `stage::<SCALE, H>` for H = 1, 2, 4, ...,
2^30 (`if H < N`, folded; a macro over literals, since stable Rust cannot
compute const params), so offsets/strides/trip counts are immediates. L031
static 153,698 -> 117,640 (opt 3), 157,939 -> 135,233 (s); F411 59,965 ->
41,991, 53,565 -> 44,255. Code (L031): 380 -> 1,852 B (s), 1,040 -> 4,676 B
(3). Layer built at run time (bench feature `fft-runtime`, black_box tables):
L031 138,456 (3) / 141,600 (s), F411 45,619 / 43,864. C++ matrix in the
scratch harness (gcc 13.2.1, prefetch on): const object -O3 F411 107,292 L031
166,833; -Os 113,125 / 143,245; run-time object -O3 104,509 / 152,747; -Os
115,202 / 142,482; the sweep's two-tone signal changed no C++ timing by more
than 27 cycles. gcc -O3's code for the const object is generic (no unrolled
stages, no constant twiddles; per-stage __udivsi3 for the stride): the
"gcc specialises a constant FFT object" explanation does not hold here.
**Correction (same day): those C++ -O3 figures were an unlucky caller.** gcc's
register allocation of the same stock loop depends on the function `forward`
is inlined into: in the harness's `be_invoke` (global arrays) a 184 B loop with
10 [sp] loads, 3 stores, 2 literal loads per butterfly (158,603-166,833); a
noinline `(int16_t*, int16_t*)` function 193,277; noinline `Scratch&` 158,224;
a processing function that copies the input into a `Scratch&` then calls
forward (`fw/be_fft_app.cpp` in the scratch harness) a 138 B loop with 4 [sp]
loads, 0 stores, 0 literal loads: 103,492 incl. a 1,861-cycle copy (F411
84,200 incl. 1,278). With a prep pass in that function 114,315 at -O3 and
176,393 at -Os. Not alignment (-falign-loops=8/16: 3%), not -std=c++14
-ffreestanding, not data. Best C++ now quoted: -O3 L031 ~101,600, F411
~82,900; -Os L031 142,482 (first harness, run-time object; ~152,200 in the
processing function), F411 ~85,200 (processing function). Rust vs best:
L031 +16% at 3, -5% at s; F411 about half. Staged C++ in the processing
function: -O3 L031 ~97,700 (only -4%), F411 ~63,200 (-24%); -Os L031 ~166,000
(+9%), F411 ~85,300. Rust's L031 loops still reload the clamp bounds from the
literal pool (13-34 per loop) and spill: that is the remaining gap.
**Saturation skip** (same day): in the scaled butterfly, if both products fit
in i16 (`fits_i16`: `((v >> 15) + 1) as u32 <= 1`, immediates only), all four
halved sums fit (top+t in [-65536, 65534], top-t in [-65535, 65535]) and are
stored with `as i16`; otherwise the saturating path. `const SKIP_SATURATION =
cfg!(not(target_has_atomic = "32"))` (the only cfg differing between thumbv6m
and thumbv7em; ARMv6-M has no ldrex/strex): L031 static 117,640 -> 91,430
(3), 135,233 -> 96,784 (s); run-time layer 97,221 / 98,389; F411 unchanged
(ungated it cost 41,991 -> 48,363 at 3 and 44,255 -> 59,420 at s: ssat is one
cycle). M0+ code 1,852 -> 2,056 B (s), 4,676 -> 3,808 B (3). vs best gcc on the
L031: -10% at -O3 (101,600), -32% at -Os (142,482). `transform::<SCALE, SKIP>`
is pub(crate) so the host checks both: unit test
`saturation_skip_changes_nothing` (lengths 2-1024, random samples/twiddles at
three scales; plants: skip formula `+ 1`, check `>> 16` fail it) and Kani
`fft_saturation_skip_is_exact` (one butterfly, every input, 21 s; compare
elementwise, array `==` is a memcmp loop that needs more unwinding; the `>> 16`
plant gives a counterexample). The host sweep runs SKIP = false; QEMU's M0
sweep runs SKIP = true against the C++ golden (247 lines match). Inverse
(unscaled) keeps the saturating path: top +- t can overflow even with t in
range.
Dropped: `(v >> 15) + 1 <= 1` saturation (LLVM folds `as i16` round trips
back into the clamp; the shift form kept constants out of the M0+ loop but
was no faster there and lost `ssat` on the F411: 85,251), real outputs before
imaginary (F411 -4%, L031 +4%), constant stages for H <= 8 only (141,374 at s
for 764 B less). Bit reversal (reverse-carry, no table) ~9,300 cycles on the
L031; a u16 table would cost 512 B at N=256. Plants (start `pair & !1`, step
2H + (H >> 7), stride + (H >> 6)) fail the sweep. Kani FFT proofs, panic-never
3/s/z, QEMU all pass. Calibration: `build_fft_twiddles::<N, HALF>() ->
(cos, sin)`: `-TAU * k / N` (TAU == the C++ literal as f64), `round()` = lround,
clamp to i16; matched glibc's cos/sin output on the first run. Checked by
`tests/fft_sweep.rs` (golden from `scripts/cpp/fft_sweep.cpp`, UBSan+ASan
clean, identical at -O1 and -O3; N = 2..1024: tables in lines of 16, 9 signals
(impulse, offset impulse, dc, tone, two tones, random, rails, all -32768,
32767/-32768) x {input, mag of input, forward, mag, inverse(forward), inverse}
on built and on random twiddles (-32768 included)); matched on the first run.
Twelve plants each fail it (no >>1 in one stage, sin sign, saturate before
shift, bit reversal off by one, builder trunc, magnitude wraps, inverse
missing final negate, rounding q15, unscaled path wraps, scaled path no
saturation (only from line 55), stride off, no clamp at 32767). Device: bench
`sweeps` reads the tables from the golden (`twiddles ` lines); bench `fft`
(N=256 forward on the random signal; tables and expected hash written by
bench/build.rs from the golden). panic-never: N=2 and N=256
**QRFft1D** (2026-11-15, tinymind #281 at `7336296`): `QRFft1D<'a, N, HALF>`
(same fields and tables as QFft1D; `BINS = HALF + 1` assoc const; methods
generic over `BINS` with `const { check::<BINS>() }`, a wrong length is a
post-mono error, compile_fail doctest). Output arrays double as the inner
work space: `split_first_chunk_mut::<HALF>()` + `first_mut()` for the
Nyquist bin (Option, never None). Load fused with the first stage `(a +- b) >>
2` (plain `as i16`, exact: Kani `rfft_first_stage_fits`); bit reversal of the
packed index by reverse-carry over HALF/2 (rev(2m) over log2(HALF) bits =
rev(m) over one fewer; rev(2m+1) = that + HALF/2). Remaining inner stages are
QFft1D's: `stage`/`run_stages` became free fns over buffer length L and table
length T (stride T/H; QFft1D L=N,T=HALF; QRFft L=T=HALF, FIRST=2), with the
same SKIP_SATURATION. Split = C++ `qrfftSplitBin` (`split_bin`), no halving;
N=2 halves DC/Nyquist (edge shift). **Inlining trap:** after the refactor LLVM
inlined `QFft1D::transform` into the bench's main: L031 91,430 -> 103,209,
F411 41,991 -> 43,604, +4 KB; `#[inline(never)]` on `transform` restores
master's bytes and cycles exactly (run_stages never-inlined instead: 91,807 /
42,015). `forward_impl` is `#[inline(never)]` too. Boards (`fft` workload now
also runs QRFft on the signal's real part, hash 0x316EE142 from the golden's
`rfft` line): F411 23,083, L031 55,823 at opt 3 (QFft1D 41,989 / 91,426 in
the same image); opt s 27,124 / 60,428 (QFft1D 44,241 / 95,758; master's
image 44,255 / 96,784); code L031 3,524 B (3), 2,208 B (s); fft workload
text L031 21,692 (3, baseline 12,436), 12,884 (s, baseline 7,276). C++ x86
said 2.3-2.4x; on the boards 1.8x / 1.6x. C++ QRFFT1D not timed on the
boards. Sweep: each signal's real part through QRFft (forward + magnitude
hashes, `n=N <label> rfft signal=...` lines, no new rng draws so the 247 old
lines are unchanged; 427 lines, UBSan+ASan clean, identical at -O1); matched
first run. Plants failing the sweep: odd sign, mirrored-bin twiddle, Nyquist
as a sum, N=2 halving dropped, first stage >> 1, inner stride (FIRST=4),
even/odd swapped, j1 = j ^ 1, split loop short; split without saturation
passed it (unreachable with built tables), so unit test
`rfft_split_saturates` (values from C++ qrfftSplitBin). Also
`tests/fft_sweep.rs::rfft_is_close_to_the_exact_dft` (C++ test bounds: 2
log2 N + 1 LSB vs f64 DFT/N, log2 N vs QFft1D, N=4..256; DC/Nyquist/impulse);
the end-halving layout fails it (N=64, 70 LSB). Unit tests: N=2 exact,
skip/no-skip identical. Kani: `rfft_forward_never_panics_4`, `_8`,
`rfft_first_stage_fits` (an i16 first-stage plant fails). panic-never N=2, 8,
256 at 3/s/z.

## `src/int8/int4.rs` (re-exported from `int8`)

C++ source: `cpp/qint4.hpp` (tinymind #299, #306, #307; v19.0)

2026-11-25 at `cf3e9ef`. `QDenseInt4<'a, IN, OUT, PACKED>` (forward returns
`[i8; OUT]` like `QDensePerChannel`), `QConv2DInt4<'a, C, KH, KW, F, PACKED,
SH, SW, PT, PB, PL, PR>` and `QDepthwiseConv2DInt4<'a, C, KH, KW, PACKED, ...>`:
`weights: &[u8; PACKED]`, `PACKED == (n + 1) / 2` in the inline const assert
(the one const param stable Rust cannot derive). `pack_int4::<N, PACKED>`,
`unpack_int4(&[u8], index) -> Option<i8>`, `int4_packed_bytes`,
`INT4_UNPACK_MAX_WEIGHTS` (1024 = C++ `TINYMIND_INT4_CONV_UNPACK_MAX_BYTES`).
Unpacking into the nested int8 weight array is expressible on stable (no
`F*KH*KW*C` arithmetic: the nesting is the size), so layers up to the
threshold unpack onto the stack in a `#[inline(never)]` fn and delegate to
`QConv2DPerChannel` / `QDepthwiseConv2D`; larger ones run a tap-by-tap
packed path (padding taps skipped: zp - zp contributes 0). C++ depthwise
always unpacks per channel; ours uses the same threshold (results identical).
Sums wrap in i32 as the int8 layers'. `calibration::quantize_int4_per_channel
<C, N>` (f32 max |w| / 7, lround on the double quotient, clamp -7..7). No SMLAD
speed-ups (no `unsafe`); not timed yet. Checked by `tests/int4_sweep.rs`
(golden from `scripts/cpp/int4_sweep.cpp`, UBSan+ASan clean, -O1 = -O3): 6
dense shapes (odd rows), 7 convs (3 unpack, packed via the C++'s patch,
patch-odd, rows-odd (window 135 > 128) and one-row paths), 2 depthwise (unpack;
C=40 5x6 = 1200 weights packed on our side), pack/unpack all 16 values both
nibbles + odd length + out of range, quantize incl. a zero channel and ties;
every layer also == the int8 layer on unpacked weights (`same=1`, both
sides). Matched first run. Plants that fail it: nibbles swapped, no sign
extension, padded tap not advancing k, depthwise taps transposed, trailing
nibble 15, ties to even, threshold 1200, dense zp dropped. panic-never: dense
64x16 and 3x3 (odd), conv 288 weights (unpack) and 1296 (packed, strided,
padded), depthwise 144 and 1296; planted `packed[k / 2]` fails the link. Kani
`mod int4`: nibbles sign-extend (spec via `<< 4 >> 4`), dense 3x2, conv
(conv_never_panics' shape), depthwise 2 channels (unwind 5); planted OOB
`packed[(k + 1) / 2]` fails dense.
## `src/int16x8.rs`, `src/int16x8/{calibration,softmax_tables}.rs`

C++ source: the int16 paths of `cpp/qaffine.hpp` (`Requantizer<int64_t, int16_t>`), `qdense.hpp`, `qconv2d.hpp`, `qdepthwiseconv2d.hpp`, `qpool2d.hpp`, `qadd.hpp`, `qactivations.hpp` (`qApplyLUT16`, `buildQLUT16`, sigmoid/tanh 16), `qsoftmax_fixedpoint.hpp` (`QSoftmaxInt16`, `populateTFLiteLUT16`, `buildQSoftmaxInt16Params`), `qsoftmax16_tables.hpp` (tinymind #300, #303-#305, #308; v19.0)

2026-11-25 at `cf3e9ef`. Dan's call (2026-11-25): a new top-level module
`int16x8` with the C++/int8 names inside it, not generic int8 layers (struct
literal inference break) and not suffixed types. `Requantizer` (i64 -> i16):
the C++'s 32-bit-halves algorithm is `floor((x * m16 + 2^(t-1)) / 2^t)`
saturated to i32, written directly in i64 (product < 2^62); clamps acc to
+-2^47 and shift to -14..=47 as the C++; then rail +-2^30, zp, two-`if`
clamp. `Requantizer32` = the int8 requantizer's arithmetic with i16 bounds,
for `QAdd` (C++ `Requantizer<int32_t, int16_t>`). Layers: `QDense`,
`QDensePerChannel`, `QConv2D`, `QConv2DPerChannel` (one `conv_layer!` macro
over a shared tap-by-tap `conv2d`), `QDepthwiseConv2D`, `QMaxPool2D`, `QAdd`
(+ sealed `int16x8::Tensor` for i16 arrays; saturating_shl like int8),
`apply_lut` (`LUT_SIZE` 513, slope wrapped to i16 as TFLite), `QSoftmax`
(`forward::<N>`, N 1..65536, tables `softmax_tables::{EXP, RECIPROCAL}`
extracted by `scripts/extract-luts.py` from qsoftmax16_tables.hpp; CI's LUT
step diffs both files; `calibration::build_softmax_luts` = TFLite's
LUTPopulateInt16 and a unit test checks it equals the constants: glibc exp
matched). Straightforward loops, i64 sums wrapping (Kani found bias + dot
overflowing for any i64 bias; the C++ is UB there); no SMLAD.
**Tuned 2026-11-25** on the digits 16x8 model (F411 517,458 -> 408,880,
C++ -O3 scalar 452,807; L031 1,122,048 -> 854,511, C++ 853,668), each step
result-identical: `SUM32_TERMS` = 255 products (|(x - zp) * w| <= 65,535 *
128 < 2^23) summed in i32 (`dot32`, `dot` in i32 chunks); `Requantizer::scale`
uses `i32::try_from(acc)` for a 32x32 -> 64 product (smull); conv windows
<= 255 taps gathered once per pixel into `[i16; 255]` (padding = zp) + one
`dot32`/`dot32_zp0` per filter (larger windows tap by tap); depthwise
windows <= 255 look each tap up once per pixel into `[i32; C]` sums;
`QAdd` (int16): `FastAddInputWide` ported (one nudged 32x32 -> 64 per input;
`half` = shift for TFLite's exactly halved input; three loops; reference
loop otherwise; `add_elements` `#[inline(never)]`); ARMv6-M only
(`SPLIT_MULTIPLY = cfg!(not(target_has_atomic = "32"))`):
`Requantizer::scale_split` (r = 16-bit multiplier < 2^15, acc = ah 2^16 + al,
(A + 2^(t-17) + (B >> 16)) >> (t - 16), t = 15 + shift in 16..=47). F411
per layer (profiled copy): stem 112,077 -> 86,903, c2 284,660 -> 223,954,
add 80,034 -> 34,912, depthwise 25,970 -> 6,221, pointwise 42,068 -> 40,026,
fc 18,891 -> 14,582. The zp-0 dot alone: L031 -3k; the split requantizer:
L031 1,016,150 -> 854,511. Unit tests: `fast_add_input_equals_reference`
(43 parameter sets x every int16 input; plant: no negative nudge),
`split_scale_equals_scale` (shifts -16..50 x 40 multipliers x 200 accs;
plant: high round 2^(t-16)), `conv_paths_equal_direct_sum` (gathered and
tap-by-tap vs an i64 sum; plants: padding 0, a skipped kernel row); the M0
QEMU sweep (`sweeps-b`) runs the split path (the same plant fails it). L031
16x8 image 32,648 B: 120 B under 32 KB. An i8 `forward` before/after
alignment-only change of the plain image moved 470k -> 489k once (the
FastAddInput step alone, profile said faster): F411 layout effects.
Pointwise not a separate type (a 1x1 `QConv2DPerChannel`). `calibration`:
`build_requantizer`, `build_add` (left shift 15, as the C++ code generator's
16x8 ADD), `build_lut`/`build_sigmoid_lut`/`build_tanh_lut` (buildQLUT16, on
the knots), `build_softmax_luts`, `build_softmax`. Checked by
`tests/int16x8_sweep.rs` (golden from `scripts/cpp/int16x8_sweep.cpp`, 78
lines, UBSan+ASan clean; -O1 first differed from -O3: two draws in one C++
expression (unspecified order), now one per statement): rq64 corners x20000,
dense/conv/depthwise (trial 1 = inputs >> 9 with fine requantizers: random
16x8 accumulators mostly saturate or round an error of one away, so a
depthwise tap off by one first passed), 289-tap depthwise (C++ int64 window),
maxpool, add (16 calibrated + 8 random), lut over all 65536 inputs (2 random
tables, 4 built), softmax N 1/2/10/300 (t=7 alternating rails). Matched on
the first run. Plants that fail: acc/shift not clamped, m16 truncated,
rounding term off by one, slope unwrapped, lookup unrounded, reciprocal
recentred +1, depthwise tap +1, dot term +1, padding sides swapped, add left
shift 16, table without midpoint correction; extra i16 clamp before qmin/qmax
is equivalent. panic-never: every layer (planted `lut[x >> 6]` fails the
link). Kani `mod int16x8` (7).

## `src/int8/mel.rs` (re-exported from `int8`)

C++ source: `cpp/qmel.hpp` (tinymind #296, v19.0)

2026-11-25 at `cf3e9ef`. `QLogMelSpectrogram<'a, N, HALF, BANDS, LOG_FRAC_BITS: u32 = 8>`
(HALF as QRFft1D's; `forward::<BINS>` with `const { check::<BINS>() }`: N >= 4
power of two, HALF = N/2, BINS = HALF+1 <= 65535, BANDS > 0, FRAC <= 14),
`QMelWorkspace<N, BINS>` (pub arrays, `const fn new`), `QMfcc<'a, BANDS,
COEFFS> { dct: &[[i16; BANDS]; COEFFS] }` (const assert 1..=BANDS coeffs;
panic-never's first try with BANDS 1 / 2 coeffs did not compile, as meant).
`weights: &'a [u16]` (variable total); band tables that reach past the bins or
the weights (UB in the C++) sum only what exists (`get(..).unwrap_or(&[])` +
zip/take). Arithmetic as the C++: z = (x*2^15 - a*prev)*w exact in i64, shift
in -14..=31, `z << -shift` for quiet frames, offset `(2 log2 N - 106 + 2r) <<
30`, log2 via the 33-entry Q2.30 table (pub(crate) `log2_q30`,
`natural_log_from_log2_q30`), clamps `max(floor).min(32767)` (no `clamp`,
which panics if min > max). Builders in `calibration` (std):
`build_hann_window::<N>(periodic)`, `mel_filter_weight` (pub, C++
`qmelFilterWeight`), `build_mel_filterbank::<N, BANDS, CAP>(sr, fmin, fmax) ->
Option<MelFilterbank>` (C++ returns 0 used when out of room; here None; tables
+ `weight_storage: [u16; CAP]` + `len`, `weights()` slice),
`build_dct_matrix::<BANDS, COEFFS>()`. `10f64.powf`/`log10`/`cos` reach glibc
like the C++: matched on the first run. Checked by `tests/mel_sweep.rs`
(golden from `scripts/cpp/mel_sweep.cpp`, 269 lines, UBSan+ASan clean,
identical at -O1/-O3): N 4/64/128/256/512/1024, bands leaving 7 empty
(n128-narrow), FRAC 0/8/14, floors -32768..32767, symmetric window once;
10 integer signals (no libm) x pre-emphasis 0 / 31785 (/ -32768 with prev
32767, / 1 with prev 1 = tiny silent frame for the negative block shift); two
random-table configs (rand16 has w[0] = 100 so pre 1 gives a frame below 2^14
in Q8; in Q14 it hit the floor and hid the shift). Plants that fail it: block
shift -13, pre-emphasis from x not prev, no interpolation, log truncated,
weights not advancing, floor dropped, MFCC truncated, offset -105, symmetric =
periodic, DCT row 0 scaled as the rest, `z << (-shift - 1)`. Equivalent or
unreached (pass): mel_to_hz via `exp(x ln 10)` (ulps never flip a rounded
weight), band edge `> 1` instead of `> 0`, weight clamp 65534 (no weight
rounds to exactly 1/65536 or 65536 in these configs). panic-never N=4 (2
bands) and N=256 (32 bands), a short weights slice; planted `LOG2_TABLE[x >>
58]` fails the M0+ link. **Tuned 2026-11-25:** y = x * 2^15 - a * prev formed in i32 (range
[-2^31, 2^31 - 2^15], exact; `wrapping_sub` only matters at -2^31) so z = y
* w is one 32x32 -> 64 multiply: features log-mel F411 49,183 -> 43,645
(C++ 46,878), L031 197,936 -> 156,410 (185,172); same hashes; plant `<< 14`
fails the mel sweep. Kani `mod mel`: `log2_q30_brackets`,
`natural_log_never_overflows` (|arg| < 200 << 30), `log_mel_never_panics` (N=4,
every field symbolic, unwind 6).

## `src/int8/anomaly.rs` (re-exported from `int8`)

C++ source: `cpp/qanomaly.hpp` (tinymind #297, v19.0)

2026-11-25 at `cf3e9ef`. `QKMeansAnomaly<'a, F, K> { centroids: &[[i8; F]; K],
inv_radius_sq: &[u32; K], feature_weight: Option<&[u16; F]> }`,
`QDiagGaussianAnomaly<'a, F, K> { means, precision: &[[u32; F]; K], bias: &[i32;
K] }`, `QKMeansAnomalyTrainer<F, K>` (`const fn new`, `Default`; C++ index
getters became whole-array getters `learn_counts`, `calibration_counts`,
`centroids`, `feature_weights`; `reset` keeps cap and weighting as the C++).
`check::<F, K>()` = C++ static_asserts (F, K > 0, F <= 65536). Sums the C++
does in unsigned wrap (`wrapping_add`); UB made total: `rounded_divide` b <= 0
-> 0 and saturating on i32::MIN; a calibration count wrapped to 0 leaves the
radius (checked_div); freeze's `numerator*256 + d/2` wraps as the C++
unsigned. Host builder `calibration::build_diag_gaussian_anomaly(&[[f64; F];
K], &[[f64; F]; K], &[f64; K], scale: f32, zp, floor) -> (means, precision,
bias)` (lround -> `round() as i64`, saturating where the C++ is UB). Checked by
`tests/qanomaly_sweep.rs` (golden from `scripts/cpp/anomaly_sweep.cpp`, 138
lines, UBSan+ASan clean, -O1 = -O3): direct k-means (inv 0 / MAX / small <
1024 so d2 > 2^32 still scores below MAX / any; weights null or random or
65535), trainers 1x1, 4x3, 7x2, 32x8, 600x2 x 9 runs (caps 65535/0/5/40,
weighting on/off, K/2 samples only, calibrate-before-learn, learn after
freeze, run 8 a constant stream: every distance ties, residuals 0 -> weight
65535), feature 1 never moves (weight 65535 path), with F >= 512 feature 0
spread 127 and the rest 1 (weight < 0.5 -> clamp 1); gaussians direct (incl.
precision MAX, bias MIN) and built from random fits (variances under the
floor, floor 0.0001). Matched on the first run. Plants that fail it: division
toward zero, large-distance form dropped (needed the small inv), settled >= 9,
radius floored, Gaussian not rounded, weight clamp at 0 (needed F = 600),
`<=` ties in learn and calibrate (needed the constant stream), cap `<=`, zero
residual -> 1, bias without scale^2, means truncated. **Plant script trap:**
restoring a file with `mv` from a `cp` backup gives it an mtime older than the
last build, so cargo keeps the planted binary; `touch` after restoring.
panic-never F/K 1/1 and 32/8 (planted unchecked `count[seeded]` fails the
link). Kani `mod anomaly`: `normalize_distance_is_exact` (u128 spec, d2 <
2^48), score and Gaussian never panic, `trainer_learn_never_panics` (2x2) and
`trainer_calibrate_never_panics` (1x1, 177 s: the freeze's u64 divisions;
2x2 learn-or-calibrate took 339 s), both from `any_state()` (a cfg(kani)
constructor with every field symbolic), `trainer_step_stays_between`. A
functional `rounded_divide` proof was dropped (395 s even bounded).

## `examples/logmel_vibration.rs`, `examples/vibration_anomaly.rs` (+ `examples/common/{logmel,vibration}_driver.rs`)

C++ source: `examples/logmel_vibration`, `examples/vibration_anomaly` (tinymind #296, #297)

2026-11-25 at `cf3e9ef`. Feature `std`. Both draw from `std::mt19937` +
`std::normal_distribution<double>`: `std_random.rs` gained `canonical_f64`
(libstdc++ `generate_canonical<double, 53>`: two draws, `(lo + hi*2^32) /
2^64`, 1.0 -> nextafter) and `NormalF64` (Marsaglia polar, the second value
saved for the next call, `ret * stddev + mean`); checked against g++ 13
libstdc++ on 35 values incl. 20,001 draws (bit patterns), matched first try.
Signals use glibc sin/cos/pow/fmod (`%` on f64 = fmod) in the C++ expression
order; compare-with-cpp builds both with `-ffp-contract=off` (as har/airq/gas).
All outputs byte-identical on the first run, except by design the
`vibration_anomaly` stdout line "k-means trainer N bytes": C++ 1264 (field
order + padding), Rust 1248 (rustc reorders); the golden test substitutes it,
compare-with-cpp masks it with sed. Large CSVs pinned by length/lines/FNV-1a
in the golden tests (logmel 183 KB + 69 KB, vibration 390 KB); committed:
stdout + small CSVs. Plants that fail: feature rounding +15, AUC ties not
halved, normal's saved value dropped, error bucket rounded; `previous_sample`
off by one passes (pre-emphasis is 0 in that example: equivalent). Plots:
`scripts/plot-examples.py` `logmel`/`vibration` = tinymind's two plot.py.
Site pages `docs/logmel.md`, `docs/vibration.md` (nav 12, 13).

## `examples/kws_cortex_m_int8.rs`, `examples/common/kws_int8_driver.rs`, `examples/common/std_random.rs`, `scripts/cpp/kws_int8_dump.cpp`

C++ source: `examples/kws_cortex_m_int8/kws_cortex_m_int8.cpp`

Feature `std`. Draws from `std::mt19937(42)` + libstdc++'s `uniform_real_distribution<float>` (`std_random.rs`: GCC 13's `generate_canonical<float,24>` is one draw, `float(u) / 2^32`, then `u * (b - a) + a` in f32; x86-64 has no FMA by default, but g++ 15 on CI's Ubuntu 26.04 targets FMA and contracts it, moving three depthwise scales by an ulp: the dump is built with `-ffp-contract=off`). Since tinymind #255 (ported, `TINYMIND_REF` 1389e8f) each layer's activation scale is fitted from a float reference forward pass over 32 calibration inputs (`RangeObserver`, `affine_params_symmetric`); the float sums follow the C++ loop order exactly. The C++ stdout starts with host-ns timings, so the check is a dump: `kws_int8_dump.cpp` `#define main`s and `#include`s the example source unchanged, captures its cout, prints input, weights, requantizers and every layer output, then the stdout from "Logits (int8" on; `tests/kws_int8_golden.rs` compares the driver's `dump()` plus that report. Two plants (31 calibration inputs; observer starting at 0) do not change this model's output, so `RangeObserver` has unit tests. Rust weight_bytes differ only for max pool (ZST 0 vs C++ 1). Planted defects (f64 distribution, per-channel scale off, transposed dw kernel, per-tensor dw requantizer) each fail it

## `examples/int8_xor.rs`, `examples/common/int8_xor_driver.rs`

C++ source: `examples/pytorch_quant/xor/xor_quant.cpp`, `weights.hpp`

Weights and scales copied from `weights.hpp` (the goldens catch any mistake)
