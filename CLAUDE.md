# CLAUDE.md

Guidance for Claude Code (and any fresh agent) working in this repository.

## What this is

tinymind-rs (repo `tinymind-rs`, crate `tinymind`) is a Rust port of Dan's C++ library
[tinymind](https://github.com/danmcleran/tinymind) (checked out as a sibling at
`../tinymind`). On 2026-11-07 Dan decided to publish it widely as free open
source software, as tinymind is. Treat it as a
public library: documented public API, honest scope and status in the README,
semver once released, and no shortcuts that would embarrass a published crate.

It is still not a full port. A full port was assessed and rejected (2026-10-22):
stable Rust lacks `generic_const_exprs`, so tinymind's compile-time storage-type
selection (`QTypeChooser`) and arithmetic layer shapes (`[T; IN - K + 1]`) do not
translate, Arm DSP intrinsics (SMLAD) are nightly-only, and a C ABI erases generics.
tinymind (C++) stays the primary product; tinymind-rs grows by porting slices of it.

**Naming:** repo `tinymind-rs`, crate `tinymind` (users write `use tinymind::...`).
Commits before 2026-11-07 use the project's old name, rustymind; it was dropped
because that name is taken on crates.io.

**Publishing:** `tinymind` is on crates.io (owner `danmcleran`; docs on docs.rs).
Releases, each with a tag and a GitHub release: 0.1.0 (2026-11-07, `601098f`),
0.1.1 (2026-11-07, `ab43794`: int8 softmax), 0.1.2 (2026-11-07, `7dd0423`:
`InferencePerceptron`), 0.1.3 (2026-11-07, tag `v0.1.3`: `NeuralNetwork` with any
number of hidden layers, `Linear`, gradient clipping, `energy_efficiency`) and 0.1.4
(2026-11-07, tag `v0.1.4`: Adam/RMSprop, `Sqrt`, the `anomaly` example, panic fixes). Agents run the
whole release in a session, including `cargo publish` (Dan's preference,
2026-11-07), but only after Dan's explicit go-ahead for that release: a version
cannot be deleted, only yanked. Semver applies: before 1.0 a minor bump may break
the API, a patch bump may not (additions only is a patch). CI's `cargo package`
step checks what would be published. To release a new version:
1. A release PR: bump `version` in `Cargo.toml`, move the `CHANGELOG.md`
   `[Unreleased]` entries under the new version with its date, update the
   comparison links at the bottom, and refresh the `tinymind` entry in all three
   lockfiles (`cargo update -p tinymind` in `.`, `ci/panic-never` and `bench`).
2. Dan's crates.io token (scoped to `tinymind`) must be in
   `~/.cargo/credentials.toml`; he runs `cargo login` himself in a separate terminal,
   so the token never passes through an agent. Never read or print that file;
   checking that it exists is fine. If `cargo publish` fails on authentication,
   stop and ask Dan to log in again.
3. After the PR merges and CI is green on `master`: from a clean `master`,
   `cargo publish --dry-run`, `cargo publish`, check crates.io and docs.rs (Python
   `urllib` with a generic User-Agent; no curl here), then tag `vX.Y.Z` (annotated)
   and create the GitHub release with the CHANGELOG entry as notes. Crates.io
   trusted publishing from CI (publish on a pushed tag, no token on disk) is the
   alternative if Dan wants one.

Code comments should serve future users and maintainers.

## Current state (2026-11-07)

- tinymind's `examples/xor` (2-4-1 MLP, Q16.16, tanh hidden + sigmoid output, SGD with
  momentum, 20000 steps) is ported and runs **bit-for-bit identically** to the C++
  build: the per-step weight trajectory (`nn_fixed_xor.txt`), learning curve,
  decision surface and stdout are byte-identical.
- tinymind's `examples/pytorch_quant/xor` (a PyTorch-trained 2-4-1 network
  quantized to int8: QDense, qrelu, QDense, sigmoid LUT) is ported as
  `examples/int8_xor.rs` (feature `std`) and bit-exact: stdout and the decision
  surface. Its requantizers and sigmoid table are built from float scales on the
  host; Rust's `f32::exp` reaches the same glibc `expf` as the C++.
- tinymind's `examples/iris` (4-8-3, ReLU hidden, sigmoid outputs, 30000 steps) is
  ported and bit-exact too: stdout and all three CSVs.
- The same xor example in **Q8.8** (`xor --format q8.8`, `i16` storage) is also
  bit-exact. tinymind ships xor only in Q16.16, so the Q8.8 reference is the C++
  example built with both bit counts in `xornet.h` set to 8; the compare script does
  that edit in its temp copy.
- Ported against the tinymind commit in `TINYMIND_REF` (verified byte-identical there).

## Pull model: how work arrives here

Porting is **pulled** from this repo: a session started at the tinymind-rs root reads
tinymind and brings changes over. Nothing is pushed into tinymind-rs from tinymind
sessions.

- **Where tinymind is:** `../tinymind` (override with `TINYMIND_DIR`).
  `.claude/settings.json` adds it as a working directory and denies edits under
  `~/code/tinymind` (`Edit(...)` rules cover every file-writing tool). Tested
  2026-11-07: a relative `Edit(../tinymind/**)` deny rule does *not* match, so the deny
  uses the home-relative path; a checkout elsewhere via `TINYMIND_DIR` is not covered
  by it, only by the rule below. Project settings apply only once the folder-trust
  dialog has been accepted for this repo in an interactive session.
- **tinymind is read-only from here.** Do not edit, commit, switch branches, pull,
  stash or run `make` inside it (`make` writes build outputs into its tree). The
  compare script builds the C++ in a temp dir; that is the only way to build it
  from here. Read-only git is fine: `git -C ../tinymind log/show/diff/blame`.
- **If you find a tinymind bug** (as porting xor found the seed bug fixed in
  tinymind #236), do not work around it silently and do not fix it from here.
  Write it up for Dan (file:line, C++ behaviour, minimal repro, proposed fix); he
  fixes it in a tinymind session. Mirror the C++ as it is until the fix lands.
- **Start of a session:** run `scripts/upstream-changes.sh` (`-p` for patches). It
  lists tinymind commits since `TINYMIND_REF` that touch the sources this crate ports.
  For each one, decide whether the Rust needs the change, port it, regenerate
  goldens if outputs changed, pass every gate, then set `TINYMIND_REF` to the
  tinymind commit you verified against (`git -C ../tinymind rev-parse HEAD`) in the
  same commit.
- **When porting a new tinymind file,** add it to `PORTED` in
  `scripts/upstream-changes.sh` and to the layout table below.
- **Goldens always come from the C++ build** (copy them out of the compare script's
  C++ run), never from the Rust output, or the golden test proves nothing.

## Commands

```bash
cargo build --release --examples
cargo run --release --example xor       # writes output/ (gitignored); args: [--format q16.16|q8.8] [out_dir] [lr] [momentum]
cargo run --release --example iris      # args: [out_dir] [data file, default data/iris.data]
cargo run --release --features std --example int8_xor   # args: [out_dir]
cargo test --all-features           # unit tests + golden tests against the C++ outputs (int8 needs std)
cargo clippy --all-targets -- -D warnings
cargo fmt --check
PYTHON=<python with numpy> scripts/compare-with-cpp.sh   # all C++ comparisons; the importer checks need numpy (no system numpy here: use a venv)
python3 -I scripts/extract-luts.py  # regenerates src/lut.rs from ../tinymind's lookupTables.cpp
CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true cargo test --release --test qformat_oracle -- --ignored  # every Q8.8 pair (~15 s)
scripts/upstream-changes.sh         # tinymind commits since TINYMIND_REF touching ported sources
```

`cargo` comes from a user-level rustup install that `~/.profile`/`~/.bashrc` put on
PATH. A shell older than that install lacks it; use `~/.cargo/bin/cargo` then rather
than `source ~/.cargo/env && ...` (compound commands trip permission checks).

Gate before committing: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-features`, and `PYTHON=<python with numpy> scripts/compare-with-cpp.sh`
(needs `g++`, `../tinymind` and numpy).
CI runs the same gates; run them locally first anyway.

Formatting is rustfmt default (K&R braces). Dan prefers Allman braces in C++, but Rust
code follows rustfmt; do not fight the formatter.

## Layout (Rust module <- C++ source)

| Rust | C++ (tinymind) | Notes |
|------|----------------|-------|
| `src/qformat.rs` | `cpp/qformat.hpp`, `cpp/typeChooser.hpp` | `QValue<S, const FRAC: u32>` over `S` = `i32` (wide `i64`) or `i16` (wide `i32`); methods generated per storage type by `impl_qvalue!`, because stable Rust cannot pick a type from const-generic arithmetic (`QTypeChooser`); Truncate + Wrap policies only; `Fixed` trait = the implicit C++ "ValueType concept" |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp`, `cpp/tanh.hpp`, `cpp/sigmoid.hpp` | `Activation<V>` trait; `Relu` for every `Fixed` type; `Tanh`/`Sigmoid` for `Q16_16` and `Q8_8` |
| `src/lut.rs` | `cpp/lookupTables.cpp` | Q16.16 and Q8.8 tanh + sigmoid tables (96 entries each), generated by `scripts/extract-luts.py`, never hand-typed; CI checks they match `TINYMIND_REF` |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp` | One-hidden-layer MLP, batch 1, default policies; plain arrays, no placement-new buffers. `InferencePerceptron` = C++ `IsTrainable = false`: a separate type holding only weights (no per-neuron state, `const fn new`, `feed_forward(&self)`), same sum order as the trainable one; checked by `tests/inference_mlp.rs` (golden from `scripts/cpp/inference_mlp.cpp`: tinymind's untrainable net loaded through its setters, 4 configs) and unit tests vs `MultilayerPerceptron`. Whole-matrix weight setters, not per-index (no panics) |
| `src/network.rs` | `cpp/neuralnet.hpp` (`NeuralNetwork`, chain path: `InnerLayerChainBuilder`, `ChainNetworkDeltasCalculator`, `ChainNetworkGradientsCalculator`, `ChainBackPropConnectionWeightUpdater`) | Any number of hidden layers: `NeuralNetwork<L>` over a flat tuple of 1-8 `Dense<V, A, IN, OUT>` (impls generated by `impl_layers!`; sealed `Layers` trait; sizes chained by the impl's const params). Rust has no variadic generics, so no `HiddenLayers<...>`. Per-layer activation (C++ uses one for all hidden layers). Init order = C++ by source layer (inputs then bias), then one draw per output neuron. Deltas all computed before any update; update order irrelevant (each connection reads only itself). Checked by `tests/deep_mlp.rs` (golden from `scripts/cpp/deep_mlp.cpp`: tinymind's `NeuralNetwork` with `HiddenLayers<4,4>`, `<8,6,5>`, `<8,6>`, both formats, every weight hashed after every step) and unit tests vs `MultilayerPerceptron` |
| `src/optimizer.rs` | `cpp/adam.hpp`, `cpp/rmsprop.hpp` (`SquareRootApproximation` -> `qformat::Sqrt`) | Optimizer = type parameter (`Opt = Sgd`, zero-sized, so no size cost; `new` exists for `Sgd` only, `with_optimizer` for any). Sealed `OptimizerImpl` (allow `private_interfaces`: it takes the crate-private `Connection`). Faithful quirks, reported to Dan, not on the site: C++ calls `step()` at the start of each *layer* update (output layer first), so Adam's `t` advances per layer; the adaptive path has no bias branch, so bias weights never change. Moments live in the delta-weight slots. `(1 - beta2) * g * g` is left-associative. No `Default` for `Adam`/`RmsProp`: tinymind's defaults are raw values tuned for Q8.8. `NeuralNetwork` now updates layers output-first (order was irrelevant for SGD, matters for Adam's `t`). Checked by the adaptive runs in `tests/deep_mlp.rs`; planted defects (per-step `t`, bias updates, input-first order, `g*g` grouped) each fail |
| `src/rand.rs` | (glibc) | Bit-exact glibc `srand`/`rand` (TYPE_3 additive feedback) |
| `examples/xor.rs`, `examples/common/xor_driver.rs` | `examples/xor/xor.cpp`, `xornet.h` | Example `main` (arguments) + driver (the program, generic over the format, `--format q16.16\|q8.8`); the golden test includes the driver and runs it in-process |
| `src/int8.rs` | `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qactivations.hpp`, `cpp/include/simd/simd_dispatch.hpp` (scalar dot) | Int8 affine inference: gemmlowp primitives, `Requantizer` (i8 output only; both C++ paths, fused and saturating), `QDense<'a, IN, OUT>` borrowing `&[[i8; IN]; OUT]` weights (flash-friendly), `qrelu`/`qrelu6`, 256-entry LUT. Integer only, no_std |
| (same, `QSoftmax`) | `cpp/qsoftmax_fixedpoint.hpp` | TFLite-exact int8 softmax (`QSoftmaxFixedPoint1D`), one row per `forward::<N>` (the C++ `Rows` loop is the caller's); `N` in `1..512` checked at compile time. tinymind's other softmax, `QSoftmax1D` (exp LUT + int64 divide, TFLite to within rounding), is deliberately not ported. Checked by `tests/softmax_tflite.rs` (TFLite Micro goldens and sweep hashes, extracted by script from tinymind's unit test) and `tests/softmax_sweep.rs` (golden from `scripts/cpp/softmax_sweep.cpp`: tinymind's kernel + `buildQSoftmaxFixedPointParams`, 504 calibrations, rows of 1 to 511) |
| `src/int8/calibration.rs` | `cpp/include/qcalibration.hpp`, `cpp/qactivations.hpp` | Feature `std`: `quantize_multiplier` (with a ported `frexp`), `build_requantizer`, `quantize`/`dequantize`, sigmoid/tanh LUT builders, `build_softmax` (`buildQSoftmaxFixedPointParams`) |
| (same, `QDensePerChannel`) | `cpp/qpointwiseconv2d.hpp` (1x1, per channel) | Dense with one requantizer per output, as TFLite quantizes dense layers; verified on the boards against tinymind's benchmark hash |
| `examples/int8_xor.rs`, `examples/common/int8_xor_driver.rs` | `examples/pytorch_quant/xor/xor_quant.cpp`, `weights.hpp` | Weights and scales copied from `weights.hpp` (the goldens catch any mistake) |
| `tools/import_pytorch/tinymind_import.py` | `apps/import_pytorch/tinymind_import.py` | PyTorch importer emitting Rust. Calibration/quantization/observers copied unchanged (Dense/ReLU/Sigmoid/Tanh/Softmax subset); new: integer requantizers (`buildRequantizer` on float32 scales), LUTs via libm `expf`/`tanhf` over ctypes, softmax params (`buildQSoftmaxFixedPointParams` on the float32 scale; tinymind's importer leaves softmax to the hand-written C++), `forward`, `from_torch`, rustfmt'd output. `test_parity.py` vs tinymind's module; `example_mlp.py` (no torch) and `torch_xor.py` (torch) |
| `examples/imported_mlp.rs`, `examples/common/imported_{mlp,classifier}_model.rs`, `examples/common/imported_mlp_driver.rs`, `scripts/cpp/imported_mlp.cpp` | (tinymind importer + C++ runtime) | The importer's two example models (sigmoid MLP, softmax classifier); the C++ program runs tinymind's headers for the same models, with `QSoftmaxFixedPoint1D`. The `*_model.rs` files are generated: regenerate with `example_mlp.py [--model classifier] --rust`, never edit |
| `examples/energy_efficiency.rs`, `examples/common/energy_driver.rs`, `data/ENB2012_data.csv` | `examples/energy_efficiency/energy_efficiency.cpp`, `ENB2012_data.csv` | 8-16-2 regression: ReLU hidden, `Linear` output, `set_gradient_clip((-1, 1))`, lr 0.02, momentum and acceleration 0. Data is CRLF in tinymind's working tree (`eol=crlf`), so `.gitattributes` here pins the same; CC BY 4.0 (`NOTICE`). Gradient clipping (`cpp/gradientClipping.hpp`) applies to weights only: tinymind's `BiasNeuronConnectionWeightUpdater` uses the raw gradient (and no weight decay). This example never pushes a bias gradient past 1, so `scripts/cpp/deep_mlp.cpp` adds two clipped regressions that do (a planted bias clip fails them) |
| `examples/anomaly.rs`, `examples/common/anomaly_demo.rs`, `bench/src/anomaly.rs` | (none) | On-device anomaly detection: 8-4-8 tanh/linear Q16.16 autoencoder, SGD + clip, trained on synthetic normal windows (integer sine table + glibc rand), threshold = 2x max normal error. `no_std`, shared by host and board; hash pinned by `tests/anomaly_golden.rs` (Rust golden, documented exception). Results on `docs/anomaly.md` and `bench/README.md` |
| `examples/iris.rs`, `examples/common/iris_driver.rs`, `data/iris.data` | `examples/iris/iris.cpp`, `iris.data` | 4-8-3 MLP, ReLU hidden + sigmoid output, 3 outputs (exercises the MSE `/ OUT` path). Data is the UCI file, CC BY 4.0 (attribution in `NOTICE`), checked against tinymind's copy |
| `examples/common/cppfmt.rs` | (C++ iostreams) | `fmt_g` (`ostream << double`) and `fmt_fixed` (`std::fixed`), `#[path]`-included by each driver |
| `tests/qformat_oracle.rs` | `unit_test/qformat_oracle/qformat_oracle_unit_test.cpp` | Differential oracle: every `QValue` op vs an exact i64 reference; C++ edge values, splitmix64 sample, chains; ported formats only (signed, full width, Truncate + Wrap). Plus an exhaustive Q8.8 sweep (`#[ignore]`, CI runs it) |
| `tests/glibc_rand.rs` | (glibc) | `GlibcRand` vs the real `srand`/`rand` via FFI, 212 seeds incl. >= 2^31 (Linux/glibc only) |
| `tests/xor_golden.rs`, `tests/iris_golden.rs`, `tests/golden/` | | Goldens from the C++ runs (`q16_16/`, `q8_8/`, `iris/`); xor trajectory checked by FNV-1a hash; shared helper in `tests/common/mod.rs` |

The library (`src/lib.rs`) is `#![no_std]` and allocation-free, mirroring tinymind's
freestanding build. Only the examples, the tests and the opt-in `std` feature
(host-side `int8::calibration`, which needs `f32::exp`/`tanh`) use `std`. Keep it that
way: new library code must not need `alloc`, and anything needing `std` goes behind
the feature. CI runs tests with `--all-features` and also clippy on the plain
no_std library. It is also `#![forbid(unsafe_code)]`, and
SECURITY.md's scope depends on that: do not remove it.

## Equivalence rules (hard-won; read before changing arithmetic)

Bit-exactness against the C++ is the acceptance test. These details all matter:

1. **Operation order.** Fixed-point multiply truncates, so `(a*b)*c != a*(b*c)` in
   general. Keep the C++ expression shape exactly, e.g. the weighted sum is
   `bias_term + (0 + t0 + t1 + ...)` and the weight update is
   `(lr*g + momentum*dw) + accel*prev_dw`.
2. **Multiply** = product in the wide type (`i64` for `i32` storage, `i32` for `i16`),
   arithmetic `>> FRAC` (floors toward -inf), wrap to the storage type.
   **Divide** = `(a as wide * 2^FRAC) / b` truncating toward zero; divide-by-zero gives
   the storage type's MAX/MIN/0 by dividend sign. `QValue / int` (`div_int`, always
   via `i64`, like C++ `applyInteger`) is a different operation from
   `QValue / QValue::from_int(n)`.
3. **Add/sub wrap** (`wrapping_add`) at the storage width; comparisons are on the raw value.
4. **Two seeds.** The C++ example seeds the initial weights with `WEIGHT_INIT_SEED = 1`,
   builds the network, then re-seeds with 7 for the training samples. The Rust driver
   uses two generators. History: the C++ network used to be a global, built before
   `main()` called `srand(7)`, so its weights silently came from glibc's default seed 1.
   Porting found it; tinymind PR #236 made the init seed explicit in 11 examples with
   byte-identical output, so the goldens did not change.
5. **Random-draw count.** C++ output neurons own one unused connection whose
   initialisation consumes a `rand()` draw. `Neuron<V, 1>` keeps it so later draws line up.
6. **Printing.** `QValue` `Display` prints raw LSBs (like C++ `operator<<`). Doubles use
   `fmt_g` (printf `%g`, precision 6), not Rust's `{}`; `std::fixed` +
   `setprecision(n)` is `fmt_fixed` (Rust `{:.n}` rounds the same). Both live in
   `examples/common/cppfmt.rs`. Get expected strings from a compiled C/C++ program,
   never from bash's `printf`, which parses its arguments as `long double`.
7. The C++ computes input-layer node deltas that nothing reads; the port skips them.
8. **Weight init.** The example's `generateRandomWeight` is
   `static_cast<FullWidthValueType>((rand() % (2 * half + 1)) - half)`: computed in
   `int`, then narrowed (`random_weight` in `examples/common/xor_driver.rs`). Before tinymind #239
   the cast came first, which truncated `rand()` to `int16_t` for Q8.8 and put the
   weights in [-1.5, 0.5]. Porting Q8.8 found it; #239 fixed it in 13 examples with
   Q16.16 output unchanged, and the Q8.8 goldens were regenerated at `e78dd7a`.
9. **Int8: undefined behaviour becomes defined.** Where the C++ int8 code would hit
   UB (an `int32` accumulator overflowing, a requantizer shift outside its
   documented range, `shift - 2` overflowing), the port wraps or clamps instead,
   so every function is total; results inside the C++ contract are identical.
   `Requantizer::apply` takes the fused path only for `2 <= shift <= 31` (the C++
   tests `shift >= 2`), and clamps with two `if`s, not `clamp` (which panics when
   `qmin > qmax`).

Never trust output files lying around in the tinymind tree; regenerate the C++
reference (the compare script does). Stale ones were found there once (tinymind #237
fixed the docs that produced them).

## On-board checks (`bench/`)

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
regression pin, said so in the test. The L031 anomaly image leaves only ~200 B of
its 32 KB flash: library growth can overflow it, and CI's bench build will say so
(`train_step` and `score` are `#[inline(never)]` for this reason).
`WORKLOADS="anomaly" bench/run.sh f411 l031` runs one workload. Results and method are in
`bench/README.md`; rerun after changing arithmetic and update the tables.
`bench/gen.py` generates the MLP from tinymind's headers (never hand-copied).
probe-rs is not installed; OpenOCD and arm-none-eabi binutils are.

## Verification beyond the goldens

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
- **Kani proofs** (`src/kani_proofs.rs`, `cfg(kani)` only; CI job `kani`, about
  three minutes): 54 harnesses, for every input. For Q8.8, Q15.1, Q1.15, Q16.16, Q31.1
  and Q1.31: add/sub wrap, multiply floors then wraps, division never panics,
  `from_int`, `pow2_neg`, zero tolerance and comparisons match their definitions.
  Tanh and sigmoid (both formats) stay within their tables' range and never index
  out of bounds; ReLU is max(0, x). Int8: the requantizer's fused path equals
  gemmlowp's two-step rounding (written as tinymind's unit test writes it) for
  every input in its domain; the requantizer, the dense layer, the LUT and the
  softmax (its exp and reciprocal on every `i32`, and the layer for any
  parameters and logits, about 90 s) never panic for any field values. Specs
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

## Porting the next example

1. Read the C++ example and every policy it instantiates, including defaults hidden in
   template parameter lists (`FixedPointTransferFunctions` has ~8 defaulted policies).
2. Port only the code paths that configuration executes. Add Rust tests per module.
3. Extract lookup tables with `scripts/extract-luts.py` (add a row to its `TABLES`),
   never by hand. It asserts counts and widths and must regenerate the existing
   tables unchanged.
4. Build the C++ reference into a temp dir and byte-diff. Divergence on the first data
   line usually means RNG / init order; later divergence usually means operation order
   or a rounding/saturation policy.
5. Commit small goldens to `tests/golden/<format>/`; hash large ones. Refresh them
   with `KEEP_CPP_OUTPUT=<dir> scripts/compare-with-cpp.sh`, which copies the C++
   outputs out.

Next steps follow `reports/Embedded Rust ML priorities.md` (local, not committed):
Dan chose to port what matters most to embedded Rust developers, not the smallest
piece. Done: int8 dense + softmax, the PyTorch importer, multiple hidden layers
(2A), linear output + gradient clipping (2B, `examples/energy_efficiency`),
Adam/RMSprop (2C), the on-device anomaly-detection demo on both boards (2D).
Priority 2 is complete; head-only fine-tuning (train only the last layer) is the
one idea from the report not done. Next by the report: trees/GBDT (cheap, with
an sklearn/XGBoost importer), then int8 conv (output shapes need const
arithmetic; importer-generated code can spell them out). An inference-only
version of the multi-layer `NeuralNetwork` is not done yet (`InferencePerceptron`
covers one hidden layer).

Floating point is also unported: tinymind's networks are generic over the value
type and its unit tests and `examples/lstm_sinusoid_float` run them with `float`
and `double`. Two catches for a port: `core` has no `tanh`/`exp`, so `no_std` needs
the `libm` crate or its own implementation, and bit-exactness with glibc's libm is
not guaranteed (compare results before promising it). The `Fixed` trait would also
need a name that fits floats (e.g. `ValueType`); rename it before the first release.

## Docs site (GitHub Pages)

`docs/` is a Jekyll site with the just-the-docs theme, the same setup as tinymind's
site, published at <https://danmcleran.github.io/tinymind-rs/> by
`.github/workflows/pages.yml`. That workflow builds on every PR (deploys only from
`master`) and adds the crate's rustdoc under `/api/`. Its `build site` job is not a
required check.

- Pages: Home, XOR example, Iris example, Energy example, Int8 inference, Q-format in Rust, On-board
  results, Equivalence with tinymind, API reference, and "Bit for bit" (the launch write-up:
  how the port is checked and what Rust bought; its numbers must be kept current or the
  page dated). Keep them about what is ported; link to tinymind's site for concepts
  rather than copying its pages.
- Do not document tinymind bugs on the site (no "findings" sections, no links to
  tinymind fix PRs, no "before #N fixed it" asides). Dan's call, 2026-11-07. They
  belong in CLAUDE.md, commit messages and write-ups to Dan.
- Code examples live in rustdoc (the crate-level XOR doctest runs in `cargo test`),
  not in the site's Markdown, where nothing would compile them.
- Numbers on the pages (settling step, predictions, surface shares) come from the
  golden outputs. When outputs change, recheck them and rerun
  `scripts/plot-examples.py` (needs matplotlib in a venv; see its docstring).
- `docs/assets/tinymind_logo.png` is copied from tinymind; the OG banner
  `tinymind-rs_og.png` was drawn with Pillow in tinymind's OG layout.
- Not yet: a Releases page (tinymind's `render_releases_page.py` exits when a repo
  has no releases; add it, and `pages-on-release.yml`, with the first release) and
  the knowledge graph (not worth it at this size).
- No local Ruby is needed: the PR build checks the site. To preview locally:
  `cd docs && bundle install && bundle exec jekyll serve`.

## Keeping the docs current

Dan asked (2026-11-07) that the crate docs and the site stay up to date. The
rendered output updates itself: docs.rs builds each crates.io release, and
`pages.yml` redeploys the site, including the API reference built from `master`,
on every merge. The hand-written prose does not, so every PR that adds or changes
public API, an example or a check updates, in the same PR:

- `README.md` (what crates.io shows): Status, Quick start, the "What is ported"
  table, "How equivalence is proven".
- `src/lib.rs` crate docs (the docs.rs front page) and the module docs of any
  module it touches.
- The site: `docs/index.md` status table and quick start, `docs/equivalence.md`
  ("What is compared", "How it is checked"), the example's page (each example
  has one, with a plot from `scripts/plot-examples.py`), `docs/int8.md` for int8
  API, and `docs/bit-for-bit.md` when its "As of" version or its counts change.
- `CHANGELOG.md` `[Unreleased]`.

`scripts/check-docs.sh` (CI, `rust` job) catches the mechanical part: examples
missing from either quick start, broken site links, duplicate `nav_order`, no
CHANGELOG entry for the version, unattributed data files. For the prose, a
fresh-eyes audit (a subagent given the current facts, read-only) found four real
errors and a dozen omissions on 2026-11-07 after three quick releases; repeat it
before each release.

## Conventions

- Do not describe tinymind, or this crate as a design, as fixed-point only: tinymind
  also runs networks in `float`/`double`. "Fixed-point so far" is the accurate scope
  of the port.
- Project names are lowercase in prose: "tinymind" (the C++ library), "tinymind-rs"
  (this port). The crate is `tinymind`; write it in backticks when you mean the crate.
- Copyright: `Copyright (c) 2026 Dan McLeran` (LICENSE). Code translated from tinymind
  also carries tinymind's MIT notice (Intel 2020 + Dan McLeran 2023) via `NOTICE`.
  Do not add Intel copyright lines to new files.
- No dependencies unless there is a real reason: the library targets `no_std`
  embedded users, and a small dependency tree is part of its value.
- Python, if needed, goes in a venv/pyenv, never `pip install --break-system-packages`.
- Repo is public at github.com/danmcleran/tinymind-rs. Default branch `master`.
- **"Commit and push" means the PR flow, the same as tinymind:** branch -> PR -> green
  CI -> squash-merge -> delete the branch (GitHub deletes the remote branch on merge;
  delete the local one). Never push straight to `master`.
- CI (`.github/workflows/ci.yml`) runs two jobs on every PR and on pushes to `master`:
  `rust` (fmt, clippy, test, rustdoc with `-D warnings`, library builds for
  `thumbv7em-none-eabihf` and `thumbv6m-none-eabi`, and the tests again on the MSRV) and
  `cpp-equivalence` (checks out tinymind at
  `TINYMIND_REF` inside the workspace and runs `scripts/compare-with-cpp.sh`). Branch
  protection is meant to require both by job name, so do not rename them, and do not
  skip them with `paths-ignore` or a job-level `if:`: a required check that never
  reports blocks the PR forever. Both jobs were shown to fail on planted defects
  (a fmt error; multiply rounding instead of flooring) before the workflow landed.
- CI uses floating `stable` Rust, so a new clippy lint can turn `master` red without a
  code change. Fix the lint in its own PR.
- MSRV is `rust-version = "1.85"` in `Cargo.toml` (edition 2024's floor). CI tests on it,
  so a newer std API or language feature fails there. Raising it is a deliberate
  decision for Dan, noted in the changelog once there is one, not a side effect.
- `#![warn(missing_docs)]` plus clippy's `-D warnings` means every public item needs a
  doc comment. Prefer making internals `pub(crate)` over documenting them: the
  public API is a semver promise once released. `lut`, `Connection` and `Neuron`
  are crate-private for that reason.

## Working with Dan

These live in tinymind's session memory, which sessions started here do not load:

- Be critical, not agreeable: when an idea has a concrete failure mode, say so and
  propose something better.
- A zero exit status or green check is not evidence. Confirm the result directly, and
  show that any new test or gate fails on a planted defect before trusting it.
- Ask before outward-facing actions (pushing to a new remote, publishing, deleting).
- Use the lowercase project names in all prose.
