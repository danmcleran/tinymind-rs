---
paths:
  - "src/kani_proofs.rs"
  - "ci/**"
  - "tests/qformat_oracle.rs"
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
  detect. If `unsafe` ever enters (SIMD intrinsics), add Miri and the matching
  sanitizer for that code.
- **Kani proofs** (`src/kani_proofs.rs`, `cfg(kani)` only; CI job `kani`, 17 to
  21 minutes on CI, 16 before the recurrent cells): 86 harnesses, for every input. For Q8.8, Q15.1, Q1.15, Q16.16, Q31.1
  and Q1.31: add/sub wrap, multiply floors then wraps, the SGD products
  (`mul_round_toward_zero`, `mul_round_nearest`) round then wrap, division never panics,
  `from_int`, `pow2_neg`, zero tolerance and comparisons match their definitions.
  Tanh and sigmoid (both formats) stay within their tables' range and never index
  out of bounds; ReLU is max(0, x). Int8: the requantizer's fused path equals
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
- **Not done yet:** a coverage report, and running the goldens on Cortex-M under
  QEMU.
