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
Releases, each with a tag `vX.Y.Z` and a GitHub release (contents in `CHANGELOG.md`):
0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.1.4, 0.1.5, 0.2.0, 0.3.0, 0.3.1 (2026-11-11), 0.3.2, 0.3.3 (2026-11-12). Agents run the
whole release in a session, including `cargo publish` (Dan's preference,
2026-11-07), but only after Dan's explicit go-ahead for that release: a version
cannot be deleted, only yanked. Semver applies: before 1.0 a minor bump may break
the API, a patch bump may not (additions only is a patch). CI's `cargo package`
step checks what would be published. **To release a new version, follow the
project skill `.claude/skills/crate-release/SKILL.md`** (Dan asked for a
repeatable process, 2026-11-08; it is the single source for the steps): docs
audit as a hard gate (`scripts/check-docs.sh`, a judgment checklist and a
fresh-eyes subagent), the release PR (`scripts/check-docs.sh --release X.Y.Z`
must pass), `cargo publish`, an annotated tag (`--cleanup=verbatim`), the GitHub
release ending with the Pages link, verification on crates.io, docs.rs and
Pages, and the LinkedIn post. Standing rules it repeats: Dan's crates.io token
(scoped to `tinymind`) is in `~/.cargo/credentials.toml`; he runs `cargo login`
himself in a separate terminal, so the token never passes through an agent.
Never read or print that file (checking it exists is fine); if `cargo publish`
fails on authentication, stop and ask Dan to log in again. Crates.io trusted
publishing from CI (publish on a pushed tag, no token on disk) is the
alternative if Dan wants one.

Code comments should serve future users and maintainers.

## How this file is organised

This file is loaded in full every session, so it holds only what every session
needs: the rules, the gates and the map. The rest loads on demand:

- `.claude/rules/*.md`: per-module porting notes (what was ported from where, how
  it is checked, which planted defects fail, what was measured and dropped).
  Each has a `paths:` list and loads when a matching file is read. **Before
  changing a module, make sure its notes file is in context; read it if not.**
  Files: `core.md` (Q formats, MLPs, optimizers, floats, their examples),
  `qlearn.md`, `recurrent.md`, `int8.md`, `trees.md`, `importers.md`,
  `bench.md`, `verification.md`, `docs-site.md`.
- `.claude/notes/roadmap.md`: what is done, what is next and why, the history
  of the C++ recurrent fixes, the release history. Read it when choosing work.

New detail goes in those files, not here. Keep this file short.

## Status

Every ported example and layer is checked bit for bit against the C++ build at
the tinymind commit in `TINYMIND_REF`. What is ported: the layout table below
(README.md's "What is ported" is the public statement). Latest release: see
Publishing above.

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
  `scripts/upstream-changes.sh`, to the layout table below, and its notes to the
  matching `.claude/rules/` file (with its path in that file's `paths:`).
- **Goldens always come from the C++ build** (copy them out of the compare script's
  C++ run), never from the Rust output, or the golden test proves nothing.

## Commands

```bash
cargo build --release --examples
cargo run --release --example xor       # writes output/ (gitignored); args: [--format q16.16|q8.8] [out_dir] [lr] [momentum]
cargo run --release --example iris      # args: [out_dir] [data file, default data/iris.data]
cargo run --release --features std --example int8_xor   # args: [out_dir]
cargo test --all-features           # unit tests + golden tests against the C++ outputs (int8 needs std)
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
PYTHON=<python with numpy> scripts/compare-with-cpp.sh   # all C++ comparisons; the importer checks need numpy, scikit-learn, xgboost, lightgbm (CI pins them; here: a venv)
python3 -I scripts/extract-luts.py  # regenerates src/lut.rs from ../tinymind's lookupTables.cpp
CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true cargo test --release --test qformat_oracle -- --ignored  # every Q8.8 pair (~15 s)
scripts/upstream-changes.sh         # tinymind commits since TINYMIND_REF touching ported sources
scripts/emit-intermediates.sh [host|arm|all]  # MIR, LLVM IR, asm (+ Arm .lst) into target/intermediates/
scripts/qemu-bench.sh               # bench/ workloads + test sweeps under QEMU (Cortex-M4 + M0, opt 3/s/z): correctness only
```

`cargo` comes from a user-level rustup install that `~/.profile`/`~/.bashrc` put on
PATH. A shell older than that install lacks it; use `~/.cargo/bin/cargo` then rather
than `source ~/.cargo/env && ...` (compound commands trip permission checks).

Gate before committing: `cargo fmt --check` (also in `ci/panic-never`, a separate crate
the root `cargo fmt` does not reach), `cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-features`, and `PYTHON=<python with numpy> scripts/compare-with-cpp.sh`
(needs `g++`, `../tinymind`, and numpy plus scikit-learn, XGBoost and LightGBM).
CI runs the same gates; run them locally first anyway.

Formatting is rustfmt default (K&R braces). Dan prefers Allman braces in C++, but Rust
code follows rustfmt; do not fight the formatter.

## Layout (Rust module <- C++ source)

The third column is the notes file in `.claude/rules/`.

| Rust | C++ (tinymind) | Notes |
|------|----------------|-------|
| `src/qformat.rs` | `cpp/qformat.hpp`, `cpp/typeChooser.hpp`, `cpp/include/tinymind_rounding.hpp` | `core.md` |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp`, `cpp/tanh.hpp`, `cpp/sigmoid.hpp` | `core.md` |
| `src/lut.rs` | `cpp/lookupTables.cpp` | `core.md` |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp` | `core.md` |
| `src/network.rs` | `cpp/neuralnet.hpp` (`NeuralNetwork`, chain path: `InnerLayerChainBuilder`, `ChainNetworkDeltasCalculator`, `ChainNetworkGradientsCalculator`, `ChainBackPropConnectionWeightUpdater`) | `core.md` |
| `src/optimizer.rs` | `cpp/adam.hpp`, `cpp/rmsprop.hpp` (`SquareRootApproximation` -> `qformat::Sqrt`; `AdamOptimizerFloat`/`RmsPropOptimizerFloat` -> `AdamFloat`/`RmsPropFloat`, see Floating point below) | `core.md` |
| `src/qlearn.rs`, `examples/maze.rs`, `examples/common/maze_driver.rs` | `cpp/qlearn.hpp` (`QLearner`, `QValueTablePolicy`, `ArgMaxPolicy`, `DefaultLearningPolicy`, `QLearningEnvironment`), `examples/maze` | `qlearn.md` |
| (same, `QNetwork`), `examples/dqn_maze.rs`, `examples/common/dqn_maze_driver.rs` | `cpp/qlearn.hpp` (`QValueNeuralNetworkPolicy`), `examples/dqn_maze` | `qlearn.md` |
| `src/recurrent.rs`, `examples/har_activity.rs`, `examples/common/har_driver.rs` | `cpp/neuralnet.hpp` (`LstmNeuralNetwork`, `LstmHiddenLayer`, `TrainableLstmHiddenLayerNeuron`, `ChainBackPropagationThruTimePolicy` and the recurrent chain calculators), `examples/har_activity` | `recurrent.md` |
| `src/head.rs`, `examples/gas_sensor_drift.rs`, `examples/common/gas_driver.rs`, `tests/head_sweep.rs`, `scripts/cpp/head_sweep.cpp` | `cpp/densehead.hpp`, `cpp/prototypehead.hpp`, `examples/gas_sensor_drift` | `core.md` |
| `src/rand.rs` | (glibc) | `core.md` |
| `examples/xor.rs`, `examples/common/xor_driver.rs` | `examples/xor/xor.cpp`, `xornet.h` | `core.md` |
| `src/int8.rs` | `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qactivations.hpp`, `cpp/include/simd/simd_dispatch.hpp` (scalar dot) | `int8.md` |
| (same, `QSoftmax`) | `cpp/qsoftmax_fixedpoint.hpp` | `int8.md` |
| `src/int8/calibration.rs` | `cpp/include/qcalibration.hpp`, `cpp/qactivations.hpp` | `int8.md` |
| (same, `QDensePerChannel`) | `cpp/qpointwiseconv2d.hpp` (1x1, per channel) | `int8.md` |
| `src/int8/conv.rs`, `src/int8/pool.rs` (re-exported from `int8`) | `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qpool2d.hpp` | `int8.md` |
| `src/int8/add.rs` (re-exported from `int8`) | `cpp/qadd.hpp`, `buildQAddParams` in `cpp/include/qcalibration.hpp` | `int8.md` |
| `src/int8/fft.rs` (re-exported from `int8`), `tests/sweeps/fft.rs`, `scripts/cpp/fft_sweep.cpp` | `cpp/qfft1d.hpp`, `buildQFFTTwiddles` in `cpp/include/qcalibration.hpp` | `int8.md` |
| `src/int8/compose.rs` (re-exported from `int8`) | `cpp/qmul.hpp`, `cpp/qpad.hpp`, `cpp/qconcat.hpp`, `buildQMulRequantizer`/`buildRescaler` | `int8.md` |
| `src/int8/recurrent.rs` (re-exported from `int8`) | `cpp/qlstm.hpp`, `cpp/qgru.hpp`, `buildQLSTMParams`/`buildQGRUParams`/`quantizeQLSTMBiases`/`quantizeQGRUBiases` in `cpp/include/qcalibration.hpp` | `int8.md` |
| `examples/kws_cortex_m_int8.rs`, `examples/common/kws_int8_driver.rs`, `examples/common/std_random.rs`, `scripts/cpp/kws_int8_dump.cpp` | `examples/kws_cortex_m_int8/kws_cortex_m_int8.cpp` | `int8.md` |
| `examples/int8_xor.rs`, `examples/common/int8_xor_driver.rs` | `examples/pytorch_quant/xor/xor_quant.cpp`, `weights.hpp` | `int8.md` |
| `tools/import_pytorch/tinymind_import.py` | `apps/import_pytorch/tinymind_import.py` | `importers.md` |
| `tools/import_tflite/tinymind_import_tflite.py`, `mlperf.py`, `test_translator.py`, `bench/src/mlperf.rs` | `benchmarks/framework_comparison_f411/gen/make_mlperf.py` | `importers.md` |
| `tools/import_pytorch` conv support (4D), `examples/imported_cnn.rs`, `examples/common/imported_cnn{,_flat}_model.rs`, `examples/common/imported_cnn_driver.rs`, `scripts/cpp/imported_cnn.cpp`, `tools/import_pytorch/{example_cnn,torch_cnn}.py` | (tinymind importer's Conv2D descriptor + C++ runtime) | `importers.md` |
| `examples/imported_mlp.rs`, `examples/common/imported_{mlp,classifier}_model.rs`, `examples/common/imported_mlp_driver.rs`, `scripts/cpp/imported_mlp.cpp` | (tinymind importer + C++ runtime) | `importers.md` |
| `tools/import_trees/tinymind_import_trees.py` | `apps/import_trees/tinymind_import_trees.py` | `importers.md` |
| `examples/imported_trees.rs`, `examples/common/imported_trees_{driver,gbdt_model,forest_model,data}.rs`, `scripts/cpp/imported_trees.cpp` | (tinymind tree importer + C++ runtime) | `importers.md` |
| `examples/energy_efficiency.rs`, `examples/common/energy_driver.rs`, `data/ENB2012_data.csv` | `examples/energy_efficiency/energy_efficiency.cpp`, `ENB2012_data.csv` | `core.md` |
| `examples/anomaly.rs`, `examples/common/anomaly_demo.rs`, `bench/src/anomaly.rs` | (none) | `core.md` |
| `src/tree.rs` | `cpp/qtree.hpp` (`QTreeNode`, `walkQTree`, `QDecisionTree`, `QGBDT`) | `trees.md` |
| `examples/gbdt_tabular_int8.rs`, `examples/common/gbdt_driver.rs` | `examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp` | `trees.md` |
| `examples/iris.rs`, `examples/common/iris_driver.rs`, `data/iris.data` | `examples/iris/iris.cpp`, `iris.data` | `core.md` |
| `examples/common/cppfmt.rs` | (C++ iostreams) | `core.md` |
| `tests/qformat_oracle.rs` | `unit_test/qformat_oracle/qformat_oracle_unit_test.cpp` | `core.md` |
| `tests/glibc_rand.rs` | (glibc) | `core.md` |
| `tests/xor_golden.rs`, `tests/iris_golden.rs`, `tests/golden/` |  | `core.md` |

The library (`src/lib.rs`) is `#![no_std]` and allocation-free, mirroring tinymind's
freestanding build. Only the examples, the tests and the opt-in `std` feature
(host-side `int8::calibration`, which needs `f32::exp`/`tanh`) use `std`. Keep it that
way: new library code must not need `alloc`, and anything needing `std` goes behind
the feature. CI runs tests with `--all-features` and also clippy on the plain
no_std library. It is also `#![forbid(unsafe_code)]`, and
SECURITY.md's scope depends on that: do not remove it. No `unsafe`, not even
behind an opt-in feature (Dan, 2026-11-11: it would defeat the purpose of the
crate). So no `asm!` and no Arm intrinsics (nightly's are `unsafe` too); speed
comes from safe code shapes LLVM compiles well.

## Equivalence rules (hard-won; read before changing arithmetic)

Bit-exactness against the C++ is the acceptance test. These details all matter:

1. **Operation order.** Fixed-point multiply truncates, so `(a*b)*c != a*(b*c)` in
   general. Keep the C++ expression shape exactly, e.g. the weighted sum is
   `bias_term + (0 + t0 + t1 + ...)` and the weight update is
   `(lr*g + momentum*dw) + accel*prev_dw`, where since tinymind #266 the two
   momentum products round toward zero (`mul_round_toward_zero`, C++
   `multiplyRoundTowardZero`; plain `*` for floats), and since #267 the step
   `lr*g` rounds to nearest, ties away from zero (`mul_round_nearest`, C++
   `multiplyRoundNearest`). #266 alone left the Q8.8 xor build at 3/4 (the
   floored step drifted weights down half an LSB per step); #267 fixed that
   (4/4, and har_activity's learning rate went 0.05 -> 0.1).
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

`bench/` is a separate firmware crate (not published; CI only builds it) for Dan's
NUCLEO-F411RE (Cortex-M4F) and NUCLEO-L031K6 (Cortex-M0+), both on USB;
`bench/run.sh` builds, flashes and reads the results, which must hash-match
tinymind's. Rerun it after changing arithmetic and update the tables in
`bench/README.md`. Two things bite without touching `bench/`: the L031 anomaly
image is tight on its 32 KB flash, so library growth can overflow it (CI's bench
build will say so), and every C++ figure compared with this crate must come from
an `-O3` build. The Rust bench build settings mirror the C++ benchmark
Makefile's (`-O3` = opt-level 3, `-mcpu` = `-C target-cpu`; table in
`bench/README.md`, "Build settings"): when one side changes a setting, match it
and re-measure both boards. CI runs the same images under QEMU
(`scripts/qemu-bench.sh`, Cortex-M4 and M0): hashes only, so the boards are
still needed for timings. Probe serials, workloads, loop alignment and
everything measured: `.claude/rules/bench.md`.

## Verification beyond the goldens

The goldens only prove the paths the examples execute. Details, setup and the
planted defects each check was shown to catch: `.claude/rules/verification.md`.
What every change must respect:

- **Oracle** (`tests/qformat_oracle.rs`): every `QValue` op against an exact
  reference; the exhaustive Q8.8 sweep runs in release with overflow checks on.
- **No reachable panics** (`ci/panic-never`): add new public API and new
  configurations to it. Prefer total functions (`get(..).map_or`, `checked_div`)
  over panicking ones; panic-freedom must not depend on inlining.
- **Kani proofs** (`src/kani_proofs.rs`, CI job `kani`): new arithmetic and new
  int8 layers get a harness; loops usually need `#[kani::unwind(n)]`.
- Sanitizers and Miri are not needed while the library is `#![forbid(unsafe_code)]`.
- A new test or gate is trusted only after it fails on a planted, reachable defect.

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

What to port next, and why: `.claude/notes/roadmap.md`.

## Docs site (GitHub Pages)

`docs/` is a Jekyll site (just-the-docs) published at
<https://danmcleran.github.io/tinymind-rs/> by `.github/workflows/pages.yml`, with
the crate's rustdoc under `/api/`. Do not document tinymind bugs on the site (Dan's
call, 2026-11-07); code examples live in rustdoc, not in the site's Markdown;
numbers on the pages come from the golden outputs. More: `.claude/rules/docs-site.md`.

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

`scripts/check-docs.sh` (CI, `rust` job) catches the mechanical part: every
example in both quick starts, the README table and on a site page; every public
module in the README table and the crate docs; every test suite and tool named
in the README; site links; distinct `nav_order`; a CHANGELOG entry for the
version and for every top-level public item added since the last tag; data
attribution; lowercase project names; the `tinymind = "X.Y.Z"` lines in README.md
and docs/index.md equal to the Cargo.toml version. `--release X.Y.Z` adds the release PR's
checks (version, CHANGELOG heading and links, lockfiles, `As of`, tag free).
The counts the docs quote are checked where they are computed:
`scripts/check-counts.sh`, called by `compare-with-cpp.sh` ("N checks today")
and by the `kani` CI job ("N harnesses"). Each check was shown to fail on a
planted gap. Prose cannot be checked mechanically: the release skill's judgment
checklist and fresh-eyes subagent audit cover it (that audit found four real
errors and a dozen omissions on 2026-11-07, six and twelve before 0.1.4).

crates.io and docs.rs show the README and crate docs as published, frozen with
the version: a gap at `cargo publish` stays on crates.io until the next
release. That is why the audit gates the release, not just each PR.

## Conventions

- Do not describe tinymind, or this crate as a design, as fixed-point only: both
  run networks in fixed point and in `float`/`double` (`f32`/`f64`).
- Project names are lowercase in prose: "tinymind" (the C++ library), "tinymind-rs"
  (this port). The crate is `tinymind`; write it in backticks when you mean the crate.
- **In user-facing text the C++ library is "tinymind (C++)"**, never a bare
  "tinymind" (Dan's call, 2026-11-08: readers could not tell the C++ library from
  this crate, which is also named `tinymind`). The project is "tinymind-rs", the
  crate `tinymind` in backticks. Covers README.md, CHANGELOG.md, docs/, the tool
  and bench READMEs, doc comments in src/ and examples/, and what the importers
  generate. `scripts/cpp-name.py` checks it in CI (`check-docs.sh`) and `--fix`
  rewrites bare mentions. One exception: "tinymind for Rust" names this project; it is the
  site's home-page title, which link previews (LinkedIn's card) show (Dan, 2026-11-10;
  the home page is `nav_exclude`d because just-the-docs uses the same title as its
  sidebar label). CLAUDE.md, code comments, commit messages and PRs are
  internal and may say "tinymind".
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
- CI (`.github/workflows/ci.yml`) runs three jobs on every PR and on pushes to `master`:
  `rust` (fmt, clippy, test, rustdoc with `-D warnings`, library builds for
  `thumbv7em-none-eabihf` and `thumbv6m-none-eabi`, and the tests again on the MSRV), `kani`
  (the proofs) and `cpp-equivalence` (checks out tinymind at
  `TINYMIND_REF` inside the workspace and runs `scripts/compare-with-cpp.sh`). Branch
  protection is meant to require `rust` and `cpp-equivalence` by job name, so do not rename them, and do not
  skip them with `paths-ignore` or a job-level `if:`: a required check that never
  reports blocks the PR forever. Those two jobs were shown to fail on planted defects
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
