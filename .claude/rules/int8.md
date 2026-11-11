---
paths:
  - "src/int8.rs"
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
  - "tests/softmax_tflite.rs"
  - "tests/kws_int8_golden.rs"
  - "tests/int8_xor_golden.rs"
  - "scripts/cpp/conv_sweep.cpp"
  - "scripts/cpp/add_sweep.cpp"
  - "scripts/cpp/compose_sweep.cpp"
  - "scripts/cpp/recurrent_sweep.cpp"
  - "scripts/cpp/softmax_sweep.cpp"
  - "scripts/cpp/kws_int8_dump.cpp"
---

# Int8 inference (`src/int8`)

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.

## `src/int8.rs`

C++ source: `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qactivations.hpp`, `cpp/include/simd/simd_dispatch.hpp` (scalar dot)

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

## `src/int8/conv.rs`, `src/int8/pool.rs` (re-exported from `int8`)

C++ source: `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qpool2d.hpp`

`QConv2D`/`QConv2DPerChannel`, `QDepthwiseConv2D`, `QPointwiseConv2D`/`...PerChannel`, `QMaxPool2D`, `QAvgPool2D`, `QGlobalAvgPool2D`. NHWC nested arrays (`[[[i8; C]; W]; H]`, conv weights OHWI); kernel, strides and padding are layer const params (defaults: stride 1 / pool stride = window, no padding); `forward(&self, input, &mut output)` takes H, W, OH, OW from the arrays and checks `OH == (H+PT+PB-KH)/SH+1` etc. in an inline `const { assert!() }` (stable since 1.79), so a wrong size or invalid layer is a compile error: this is how stable Rust gets around `OutputHeight` arithmetic. Loops tuned on the boards (4C, then 2026-11-09): conv is pixel-outer (window read once for all filters; filter-outer measured 20% slower for the Cin=1 stem), now as `conv2d_gather` (C=1) or `conv2d_band` (C>=2); the original per-tap `conv2d` was removed; depthwise and pointwise run filter-outer with the zero point folded into the bias (`bias - zp*sum w`, equal mod 2^32), weights held across pixels, interior windows read as slices (padded ones take the general per-tap path), and the requantizer prepared once per filter: `int8::FastRequantizer` (crate-private) is the C++ FastRequantizer extended to any qmin/qmax (fused ReLU), proved equal to `Requantizer::apply` by Kani (`fast_requantizer_matches_apply`); filters without a fast form run a `#[cold]` out-of-line loop (one loop branching per output cost depthwise 27%). Pointwise on one pixel keeps the per-output form. Same `i32` sums as the C++ modulo 2^32. Checked by `tests/conv_sweep.rs` (golden from `scripts/cpp/conv_sweep.cpp`: 57 configurations x 3 trials, every C++ path incl. patch/row/tiles/odd filters/all-padding windows, both requantizer paths and the C++ FastRequantizer vs general form, pool ranges with qmin > qmax); ten planted defects each fail it. **16-bit band path** (2026-11-09, for ResNet-8): `QConv2D`/`QConv2DPerChannel` with `C >= BAND_MIN_CHANNELS` (2) run `conv2d_band`: per output row the KH input rows widen to an i16 band (`x - zp`), per (row, filter) the weights to i16, interior windows are KH `dot16` calls over contiguous KW*C, edges tap by tap, cold general-requantizer rows tap by tap (small code). Scratch on stack ~2*KH*(W+KW)*C B; not size-limited. LLVM's ARMParallelDSP pass makes `SMLAD` from `acc + x[i]*w[i] + x[i+1]*w[i+1]` over i16 loads (plain safe Rust, opt 3), but it analyses at most 16 loads (`-arm-parallel-dsp-load-limit`, default 16), and fully unrolled constant-length sums exceed it, so by default the code is ldrsh/ldrsh/smlabb (~4.7 c/MAC); `-C llvm-args=-arm-parallel-dsp-load-limit=512` gives pure ldr/ldr/smlad (opt-in, documented; not in bench config). Tried and dropped: packed u32 words (needs `[u32; W*C/2]`, inexpressible; assembling words from i16 folds back to ldrsh), `black_box` length + multi-exit loop (SMLAD but loop overhead ~2.75 c/MAC), unsigned byte lanes (UXTB16 trick: LLVM splits to per-byte uxtb, worse), 2 filters x 2 pixels (68.0M default, 48.1M with flag vs 45.9M, +68 KB flash). Stable Rust exposes no Arm target features in cfg, so no DSP gate: measured on the L031 a 10x10x8 3x3 conv 675,064 vs 835,734 cycles on the old path (F411 303,574 vs 451,012), so the band path is used everywhere; C=3 stem also faster (2.75M vs 3.88M). Kani `conv_band_never_panics` (C=2, unwind 6, 69 s); conv_sweep plants (odd tail, zp not subtracted, band rows off by one) fail. These layers pad internally, so they never need `QPad2D` (tinymind's translator inserts it only when a padded band exceeds its buffer limit); `QPad2D` itself is ported in `src/int8/compose.rs`

## `src/int8/add.rs` (re-exported from `int8`)

C++ source: `cpp/qadd.hpp`, `buildQAddParams` in `cpp/include/qcalibration.hpp`

`QAdd` (TFLite ADD): not size-parameterized; `forward<T: Tensor>(&a, &b, &mut out)` over the sealed `int8::Tensor` (`[i8; N]`, `[[i8; C]; W]`, `[[[i8; C]; W]; H]`), one `T` so shapes must match at compile time (compile_fail doctest). The C++ `(a - zp) * (1 << left_shift)` is UB outside `0..=23`; here a negative left shift is 0 and a larger one saturates (`saturating_shl`, the same shift as in `multiply_by_quantized_multiplier`, Kani `saturating_shl_saturates` vs an i128 spec); the sum wraps. `multiply_by_quantized_multiplier` keeps its own copy: calling `saturating_shl` from it moved code by 10 bytes and the F411 DS-CNN took 11% more cycles (17.88M vs 16.12M; with `-C llvm-args=-align-loops=16` both builds take 16.197M: F411 timings depended on hot-loop alignment against flash lines; the bench now aligns loops, see On-board checks). A Kani proof that the two shifts agree took 566 s (symbolic 32x32 multiply), too slow for CI. From 256 elements `forward` tabulates `scaled_a`/`scaled_b` for all 256 int8 values (2 KB stack) and uses `FastRequantizer`: ResNet-8 adds 185 -> 22 cycles/element; `add_sweep` checks the element-wise path (chunks of 8) equals the table path; plants (table b from a, table off by one) fail; panic-never covers both paths. `calibration::build_add((scale, zp), (scale, zp), (scale, zp), qmin, qmax)` = `buildQAddParams` (f64 from f32 scales, left shift 20). Note make_mlperf.py computes the ADD parameters differently (TFLM's float32 `twice_max` and denominator), so the translator must follow it, not `build_add`. Checked by `tests/add_sweep.rs` (golden from `scripts/cpp/add_sweep.cpp`, built with UBSan clean: 48 calibrated layers x all 65,536 input pairs, 64 direct random-parameter layers, 16 saturating the requantizer's left shift); plants (b with a's multiplier or zero point, left shift 0 as 1, min for max scale, real_out off by 1e-7, real_a in f32, wrapping instead of saturating left shift) each fail it. Before this sweep nothing exercised the left-shift saturation of `multiply_by_quantized_multiplier` (a wrapping plant passed every suite)

## `src/int8/compose.rs` (re-exported from `int8`)

C++ source: `cpp/qmul.hpp`, `cpp/qpad.hpp`, `cpp/qconcat.hpp`, `buildQMulRequantizer`/`buildRescaler`

`QMul` (`forward<T: Tensor>`, FastRequantizer prepared once; product fits i32), `QPad2D<PT, PB, PL, PR> { pad_value }` (const assert OH = H+PT+PB, OW = W+PL+PR with checked adds), `QConcat` (C++ `QConcat2_2D`; const assert CA, CB > 0 and C = CA + CB). `calibration::build_mul_requantizer` and `build_rescaler` are thin wrappers over `build_requantizer` (same f64 ratio as the C++). Checked by `tests/compose_sweep.rs` (golden from `scripts/cpp/compose_sweep.cpp`, UBSan clean: 32 calibrated QMul on all 65,536 pairs, 48 direct, 5 concat shapes x 4, 6 pad shapes x 3); plants (b with a's zp, pad rows offset by PL, pad with 0, concat b with a's requantizer, rescaler scale nudged, both compile_fail doctests made valid) fail; Kani `compose_never_panics` (22 s); panic-never covers all three. Translator support: PAD (H/W only, pad value = output zp, input/output quantization equal) -> QPad2D, MUL (same shapes; ratio via `rq`) -> QMul, CONCATENATION (2 maps on channels, all quantized as the output: TFLite int8 copies then, else rescales in float) -> QConcat with identity requantizers; checked end to end by `tools/import_tflite/test_ops.py` against LiteRT's reference kernels (ai-edge-litert 2.3.0, `OpResolverType.BUILTIN_REF`) on three models built by `tflite_build.py` (flatbuffers via the `tflite` package's builders, no TensorFlow); plants (MUL ratio +0.1%, MUL ReLU dropped (visible only because fused-ReLU outputs are calibrated before the ReLU), pad 0, pad L/R swapped, concat order swapped) fail it

## `src/int8/recurrent.rs` (re-exported from `int8`)

C++ source: `cpp/qlstm.hpp`, `cpp/qgru.hpp`, `buildQLSTMParams`/`buildQGRUParams`/`quantizeQLSTMBiases`/`quantizeQGRUBiases` in `cpp/include/qcalibration.hpp`

2026-11-10, against `TINYMIND_REF` 7e83cb4. `QLstmCell<'a, IN, HID, C = i8>` (C++ `QLSTMCell`; cell state `C` through the sealed `CellStorage`, `i8` or `i16`) and `QGruCell<'a, IN, HID>` (`QGRUCell`, reset before the R_n multiply): one step, state updated in place, caller runs the time loop. Gate-major weights as nested arrays (`[[[i8; IN]; HID]; 4]`, gates i f g o / r z n), so no `4 * HID` const arithmetic. Field names = the C++ ones (as `QAdd`); C++ storage template params other than the cell state are fixed (int8 in/weights/hidden, i32 acc), the only instantiations tinymind uses. LUT output grids fixed (sigmoid 1/256 zp -128, tanh 1/128 zp 0). Both share `gate_preactivations` (`dot_with_zero_point` + two `multiply_by_quantized_multiplier` + bias, saturated to i8); all gates computed before any unit is written (as the C++). Sums the C++ leaves to UB wrap; clamps are two `if`s (qmin > qmax gives qmax). Calibration: `LstmScales`/`LstmParams`/`build_lstm_params`, `GruScales`/`GruParams`/`build_gru_params`, `quantize_gate_biases` (generic over G; f64 divide, round half away, `as i64 as i32` = lround + int32 cast), `SIGMOID_LUT_OUTPUT_SCALE`/`TANH_LUT_OUTPUT_SCALE`. Checked by `tests/recurrent_int8_sweep.rs` (golden from `scripts/cpp/recurrent_sweep.cpp`, UBSan+ASan clean: LSTM 3x4 i8/i16, 1x1 i8, 2x3 i16, 8x16 i8, 16x8 i16, 5x7 i8; GRU 3x4, 1x1, 2x3, 8x16, 16x8, 5x7; 2 calibrated + 2 direct trials each, 32 steps, state hashed per step; shifts >= -3 so no C++ sum overflows); matched the C++ on the first run. Plants that fail it: g gate on the cell tanh LUT, cell zp not added, LSTM clamp order swapped, GRU 1-z off by one, n gate reading h not r*h, r*h without zp, LSTM output rescale with cell scale, biases truncated. Kani `lstm_int8_cell_never_panics`, `lstm_int16_cell_never_panics`, `gru_cell_never_panics` (1 input, 2 units, every field symbolic incl. LUTs; also state within [qmin, qmax] when qmin <= qmax; ~70-105 s each); panic-never covers i8/i16 LSTM and GRU at 3x16 and 5x7 (planted reachable index panics in each fail the link). Not yet: the importers do not emit these cells. PyTorch `nn.LSTM` gate order (i f g o) matches; `nn.GRU` is reset-after (`r * (R_n h + b_hn)`), tinymind's is reset-before, so a PyTorch GRU does not map. TFLite's integer LSTM kernel was not compared with QLSTMCell here; translating one needs a check against TFLite

## `examples/kws_cortex_m_int8.rs`, `examples/common/kws_int8_driver.rs`, `examples/common/std_random.rs`, `scripts/cpp/kws_int8_dump.cpp`

C++ source: `examples/kws_cortex_m_int8/kws_cortex_m_int8.cpp`

Feature `std`. Draws from `std::mt19937(42)` + libstdc++'s `uniform_real_distribution<float>` (`std_random.rs`: GCC 13's `generate_canonical<float,24>` is one draw, `float(u) / 2^32`, then `u * (b - a) + a` in f32; x86-64 has no FMA by default, but g++ 15 on CI's Ubuntu 26.04 targets FMA and contracts it, moving three depthwise scales by an ulp: the dump is built with `-ffp-contract=off`). Since tinymind #255 (ported, `TINYMIND_REF` 1389e8f) each layer's activation scale is fitted from a float reference forward pass over 32 calibration inputs (`RangeObserver`, `affine_params_symmetric`); the float sums follow the C++ loop order exactly. The C++ stdout starts with host-ns timings, so the check is a dump: `kws_int8_dump.cpp` `#define main`s and `#include`s the example source unchanged, captures its cout, prints input, weights, requantizers and every layer output, then the stdout from "Logits (int8" on; `tests/kws_int8_golden.rs` compares the driver's `dump()` plus that report. Two plants (31 calibration inputs; observer starting at 0) do not change this model's output, so `RangeObserver` has unit tests. Rust weight_bytes differ only for max pool (ZST 0 vs C++ 1). Planted defects (f64 distribution, per-channel scale off, transposed dw kernel, per-tensor dw requantizer) each fail it

## `examples/int8_xor.rs`, `examples/common/int8_xor_driver.rs`

C++ source: `examples/pytorch_quant/xor/xor_quant.cpp`, `weights.hpp`

Weights and scales copied from `weights.hpp` (the goldens catch any mistake)
