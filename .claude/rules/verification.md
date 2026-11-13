---
paths:
  - "src/kani_proofs.rs"
  - "ci/**"
  - "tests/qformat_oracle.rs"
  - "tests/sweeps/**"
  - "bench/src/sweeps.rs"
  - ".github/workflows/*.yml"
  - "scripts/check-counts.sh"
---

# Verification beyond the goldens

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.

The goldens only prove the paths the examples execute. For the arithmetic:

- **Oracle** (`tests/qformat_oracle.rs`): a port of tinymind's qformat oracle. Its
  known-defect list is empty at `TINYMIND_REF`, so any mismatch is a port bug. If
  tinymind adds a known defect affecting a ported format, port its model too (the
  C++ oracle is in `PORTED`). CI runs the sampled oracle in `cargo test`, the
  exhaustive Q8.8 sweep on every PR, and a nightly soak with a fresh seed
  (`oracle-soak.yml`); a soak failure prints the seed and scale to reproduce with.
- **Overflow checks on in release** for the exhaustive run. Rust overflow is not
  UB, but release builds wrap silently, so a plain `+` that should have been
  `wrapping_add` gives the right answer and passes; with checks on it panics. (A
  planted plain `+` passed with checks off and failed with them on.) Debug-built
  tests (`cargo test`) always have them on.
- **No reachable panics.** `ci/panic-never` is a `no_std` Cortex-M image that calls
  every public function with `black_box` inputs and whose panic handler is an
  undefined symbol: if any panic path survives optimisation, linking fails. CI
  builds it for M0+ and M4F at opt-level 3, s and z. It covers the instantiated
  configurations (both formats, the xor and iris shapes, the int8 path); add new public API and
  new configurations to it. Planted panics (an assert, a dropped index clamp in
  the table lookup) fail it. To keep it passing, total functions over panicking
  ones: `pow2_neg(n > FRAC)` returns `ZERO` (the truncated value) and the weight
  getters return whole matrices instead of taking indices. Only the caller's
  weight-init closure can panic. Panic-freedom must not depend on inlining: a
  helper that takes an index or divisor from its caller keeps its check when it
  is not inlined. 2C found two (the MLP's `weighted_sum` index at opt-level s/z
  once a new network shape was instantiated, and `Sqrt`'s Newton division at every
  level); both now use total lookups (`get(..).map_or`, `checked_div`). Find the
  site by emitting the object (`cargo rustc --release --target .. -- --emit=obj=..`)
  and listing which functions relocate to `core::panicking::*`. An invalid format (`FRAC` 0 or >= the storage
  width) is a compile error wherever its arithmetic is used. Debug builds still
  have overflow-check panics; the oracle and the debug-built golden runs have
  never triggered one.
- **Sanitizers / Miri / MSan / TSan: not needed** while the library is
  `#![forbid(unsafe_code)]` and single-threaded; safe Rust cannot have the UB they
  detect. `unsafe` is ruled out for good (Dan, 2026-11-11; CLAUDE.md), SIMD
  intrinsics and `asm!` included.
- **Kani proofs** (`src/kani_proofs.rs`, `cfg(kani)` only; CI job `kani`, 17 to
  21 minutes on CI, 16 before the recurrent cells): 99 harnesses (4 for the fine-tuning heads, `mod head`, 2026-11-13), for every input. For Q8.8, Q15.1, Q1.15, Q16.16, Q31.1
  and Q1.31: add/sub wrap, multiply floors then wraps, the SGD products
  (`mul_round_toward_zero`, `mul_round_nearest`) round then wrap, division never panics,
  `from_int`, `pow2_neg`, zero tolerance and comparisons match their definitions.
  Tanh and sigmoid (both formats) stay within their tables' range and never index
  out of bounds; ReLU is max(0, x). FFT (2026-11-12, `mod fft`, 19 s for all
  five): `q15_mul` floors (remainder inequality in i64), `saturate_i16`
  clamps, forward and inverse at N=4 never panic or overflow for any samples
  and twiddles (-32768 included; unwind 5), magnitudes = min(exact, i32::MAX)
  (i128 spec). Plants: rounding q15, i16 butterfly sum (overflow), an
  out-of-range bot index, a magnitude threshold 100,000 low; `>=` for `>` is an
  equivalent mutant (2^31 - 1 is not a sum of two squares). Int8: the requantizer's fused path equals
  gemmlowp's two-step rounding (written as tinymind's unit test writes it) for
  every input in its domain; the requantizer, the dense layer, the LUT and the
  softmax (its exp and reciprocal on every `i32`, and the layer for any
  parameters and logits, about 90 s) never panic for any field values; so do
  every convolution and pooling layer on small shapes with taps in the padding
  and the elementwise addition (any parameters),
  and the pooling divide rounds half away from zero (spec in i64; a planted
  round-toward-zero fails it). The conv harnesses need `#[kani::unwind(4)]`:
  without a bound, CBMC unwound the pointwise layer's `flatten`/`zip` loops for
  30+ minutes; with it each harness takes under 90 s. Trees:
  the walk, the ensemble and `is_valid` never panic for any 3-4 node pool
  (indices in or out of range, cycles), root, class, base score and input.
  The Q-learner never panics for any Q values, rates, exploration percent,
  states, actions, valid-action count (above `ACTIONS` too) or random integer. Specs
  avoid the operations they check
  (congruences, remainder inequalities, i128 where the spec could overflow).
  Planted defects (rounding multiply, `pow2_neg` off by one, no table-index
  clamp, interpolation from the upper entry, an unchecked `<<` in the softmax)
  each fail their proof.
  **Not proved: functional division** (truncation towards zero). A symbolic
  64-bit division timed out (> 5 min, MiniSat and Kissat); it is covered by the
  oracle instead, exhaustively for Q8.8. Kani also checks the specs: two of mine
  overflowed (`p - scaled`, `q.abs()` on `i64::MIN`) before they were right.
  Local setup: `cargo install --locked kani-verifier --version 0.68.0`, then
  `cargo kani setup`, which needs `curl` (not on Dan's machine: download the
  release bundle with `gh release download`, check its sha256 against the asset
  `digest` from the GitHub API, then `cargo kani setup --use-local-bundle <file>`).
  Bump `KANI_VERSION` in `ci.yml` and here together.
- **QEMU** (2026-11-11): `scripts/qemu-bench.sh` runs the bench/ workloads on
  QEMU `mps2-an386` (Cortex-M4) and `microbit` (Cortex-M0 = the M0+'s ARMv6-M),
  opt 3/s/z, boards `qemu-m4`/`qemu-m0` in bench/src/board.rs (semihosting
  output and exit; `cycles()` is 0), in CI's `cpp-equivalence` job after the
  bench build. Workload `sweeps` (bench/src/sweeps.rs, 2026-11-11) runs the
  conv, lstm, tree and qlearn sweeps and the oracle on the device: their bodies
  moved to `tests/sweeps/*.rs` (no_std, no alloc; `#[path]`-included by the
  host tests and bench), output streamed into an `Expect` sink compared with
  `include_str!` goldens (first differing golden line reported). conv's big
  tensors are `Box` on the host, an `OnStack` newtype under `target_os =
  "none"`. qlearn's maze line stays host-only (maze_driver is std); the device
  skips the golden's first line. Oracle: `run_format(seed, scale)`; Vecs became
  fixed `Values` (<= 31) and a 20-failure `Tally`; the random single values
  are replayed from a cloned splitmix64 instead of buffered, so the sample is
  unchanged (fingerprint of every checked (inputs, expected) identical to the
  old code, fcb3714dd9b56b5f on Q8.8/Q16.16/Q31.1; counts equal at seed
  12345 scale 3 too). Default sample pinned: `CHECKS_16BIT` 105,792,
  `CHECKS_32BIT` 134,485 (host asserts at seed 0 scale 1; device requires them).
  Sizes: M0 O3 image > 256K (conv alone overflowed by 4.3 KB) and LSTM stack
  73 KB, so QEMU's nRF51 gets flash-size 1M, sram-size 256K (2M flash since
  2026-11-12: the FFT sweep took the M0 O3 sweeps image 1.4 KB past 1M; QEMU
  8.2 accepts 2M). Real F411: all
  match, 679,290,763 cycles, 477 KB. qemu-bench.sh's match regex was
  `^[a-z_]*match`, blind to `oracle_16bit_match=0`; now `[a-z0-9_]`, shown by
  an oracle-only plant (positive / 0 -> MAX-1). Other plants: tree `<`,
  rounding multiply. 60 QEMU runs ~2 min locally.
  Then (same day) add, compose, softmax and recurrent_int8 too: each shared
  module has a `Calibrate` trait (host = `int8::calibration`, device = bench).
  Device reads the C++'s params from the golden (`GoldenLines` by prefix,
  `ints::<N>(line, "key=", occurrence)`; add `calibrated `, compose `mul
  calibrated ` + 2 rq per `concat ` line, softmax every `e=` line) and asserts
  the drawn zero points/range equal the line's (out-of-step guard). recurrent's
  calibrated weights/biases/LUTs are only hashes in the golden (and need
  round/exp/tanh): device `Calibrate` returns None, trial runs zeroed for RNG
  sync, `Expect::keeping(!contains(" calibrated "))` compares the 26 direct
  lines. `powi` (std) replaced by exact `pow2_neg`/`pow2` via from_bits; goldens
  unchanged. `Expect` is line-based now (1 KB line buffer, keep-filter on both
  sides; reports `<sweep>_lines=`). Plants (each fails only its sweep): add b
  minus a's zp (line 2), pad 0 (line 101), softmax `>` for `>=` diff_min (line
  402), GRU 1-z +1 (line 39). Full image: F411 767 KB at O3 (> 512 KB), 321 KB
  at s; `OPT=s WORKLOADS=sweeps bench/run.sh f411` all match, 293,084,567
  cycles, 72 KB stack. With the FFT sweep (2026-11-12): 411 KB at s, all match,
  355,889,556 cycles, 73,280 B stack. QEMU 6 sweep runs ~2 min.
- **Not done yet:** a coverage report. The sweeps on Cortex-M were first meant
  as the prerequisite for an opt-in `unsafe` DSP feature; Dan ruled `unsafe`
  out in any form (2026-11-11, CLAUDE.md), so they guard Arm code generation
  of safe code instead (target-cpu, opt-level, LLVM changes).
