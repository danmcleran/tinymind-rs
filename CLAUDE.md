# CLAUDE.md

Guidance for Claude Code (and any fresh agent) working in this repository.

## What this is

tinymind-rs (repo `tinymind-rs`, crate `tinymind`) is a Rust port of Dan's C++ library
[tinymind](https://github.com/danmcleran/tinymind) (checked out as a sibling at
`../tinymind`). It started as a learning project; on 2026-11-07 Dan decided to
publish it widely as free open source software, as tinymind is. Treat it as a
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

**Publishing:** `Cargo.toml` keeps `publish = false` until Dan decides to release.
Never run `cargo publish` without Dan's explicit go-ahead: a crates.io release
cannot be deleted, only yanked.

Dan is still learning Rust through this repo, so **explain the Rust idioms you use**
(ownership, traits vs C++ policies, const generics, `no_std`) in replies, and point
out where Rust forces a different shape than the C++. Code comments should serve
future users and maintainers; keep idiom tutorials out of the public API docs.

## Current state (2026-11-07)

- tinymind's `examples/xor` (2-4-1 MLP, Q16.16, tanh hidden + sigmoid output, SGD with
  momentum, 20000 steps) is ported and runs **bit-for-bit identically** to the C++
  build: the per-step weight trajectory (`nn_fixed_xor.txt`), learning curve,
  decision surface and stdout are byte-identical.
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
cargo build --release
cargo run --release --bin xor       # writes output/ (gitignored); args: [--format q16.16|q8.8] [out_dir] [lr] [momentum]
cargo run --release --bin iris      # args: [out_dir] [data file, default data/iris.data]
cargo test                          # unit tests + golden test against the C++ outputs
cargo clippy --all-targets -- -D warnings
cargo fmt --check
scripts/compare-with-cpp.sh         # builds ../tinymind's C++ xor (Q16.16, Q8.8) and iris and byte-diffs all outputs
python3 -I scripts/extract-luts.py  # regenerates src/lut.rs from ../tinymind's lookupTables.cpp
CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true cargo test --release --test qformat_oracle -- --ignored  # every Q8.8 pair (~15 s)
scripts/upstream-changes.sh         # tinymind commits since TINYMIND_REF touching ported sources
```

`cargo` comes from a user-level rustup install that `~/.profile`/`~/.bashrc` put on
PATH. A shell older than that install lacks it; use `~/.cargo/bin/cargo` then rather
than `source ~/.cargo/env && ...` (compound commands trip permission checks).

Gate before committing: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
`cargo test`, and `scripts/compare-with-cpp.sh` (needs `g++` and `../tinymind`).
CI runs the same gates; run them locally first anyway.

Formatting is rustfmt default (K&R braces). Dan prefers Allman braces in C++, but Rust
code follows rustfmt; do not fight the formatter.

## Layout (Rust module <- C++ source)

| Rust | C++ (tinymind) | Notes |
|------|----------------|-------|
| `src/qformat.rs` | `cpp/qformat.hpp`, `cpp/typeChooser.hpp` | `QValue<S, const FRAC: u32>` over `S` = `i32` (wide `i64`) or `i16` (wide `i32`); methods generated per storage type by `impl_qvalue!`, because stable Rust cannot pick a type from const-generic arithmetic (`QTypeChooser`); Truncate + Wrap policies only; `Fixed` trait = the implicit C++ "ValueType concept" |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp`, `cpp/tanh.hpp`, `cpp/sigmoid.hpp` | `Activation<V>` trait; `Relu` for every `Fixed` type; `Tanh`/`Sigmoid` for `Q16_16` and `Q8_8` |
| `src/lut.rs` | `cpp/lookupTables.cpp` | Q16.16 and Q8.8 tanh + sigmoid tables (96 entries each), generated by `scripts/extract-luts.py`, never hand-typed; CI checks they match `TINYMIND_REF` |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp` | One-hidden-layer MLP, batch 1, default policies; plain arrays, no placement-new buffers |
| `src/rand.rs` | (glibc) | Bit-exact glibc `srand`/`rand` (TYPE_3 additive feedback) |
| `src/bin/xor.rs` | `examples/xor/xor.cpp`, `xornet.h` | Driver, generic over the format (`--format q16.16\|q8.8`) |
| `src/bin/iris.rs`, `data/iris.data` | `examples/iris/iris.cpp`, `iris.data` | 4-8-3 MLP, ReLU hidden + sigmoid output, 3 outputs (exercises the MSE `/ OUT` path). Data is the UCI file, CC BY 4.0 (attribution in `NOTICE`), checked against tinymind's copy |
| `src/bin/common/cppfmt.rs` | (C++ iostreams) | `fmt_g` (`ostream << double`) and `fmt_fixed` (`std::fixed`), `#[path]`-included by each binary |
| `tests/qformat_oracle.rs` | `unit_test/qformat_oracle/qformat_oracle_unit_test.cpp` | Differential oracle: every `QValue` op vs an exact i64 reference; C++ edge values, splitmix64 sample, chains; ported formats only (signed, full width, Truncate + Wrap). Plus an exhaustive Q8.8 sweep (`#[ignore]`, CI runs it) |
| `tests/glibc_rand.rs` | (glibc) | `GlibcRand` vs the real `srand`/`rand` via FFI, 212 seeds incl. >= 2^31 (Linux/glibc only) |
| `tests/xor_golden.rs`, `tests/iris_golden.rs`, `tests/golden/` | | Goldens from the C++ runs (`q16_16/`, `q8_8/`, `iris/`); xor trajectory checked by FNV-1a hash; shared helper in `tests/common/mod.rs` |

The library (`src/lib.rs`) is `#![no_std]` and allocation-free, mirroring tinymind's
freestanding build. Only the `xor` binary and tests use `std`. Keep it that way: new
library code must not need `alloc`. It is also `#![forbid(unsafe_code)]`, and
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
   `src/bin/common/cppfmt.rs`. Get expected strings from a compiled C/C++ program,
   never from bash's `printf`, which parses its arguments as `long double`.
7. The C++ computes input-layer node deltas that nothing reads; the port skips them.
8. **Weight init.** The example's `generateRandomWeight` is
   `static_cast<FullWidthValueType>((rand() % (2 * half + 1)) - half)`: computed in
   `int`, then narrowed (`random_weight` in `src/bin/xor.rs`). Before tinymind #239
   the cast came first, which truncated `rand()` to `int16_t` for Q8.8 and put the
   weights in [-1.5, 0.5]. Porting Q8.8 found it; #239 fixed it in 13 examples with
   Q16.16 output unchanged, and the Q8.8 goldens were regenerated at `e78dd7a`.

Never trust output files lying around in the tinymind tree; regenerate the C++
reference (the compare script does). Stale ones were found there once (tinymind #237
fixed the docs that produced them).

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
  configurations (both formats, the xor and iris shapes); add new public API and
  new configurations to it. Planted panics (an assert, a dropped index clamp in
  the table lookup) fail it. To keep it passing, total functions over panicking
  ones: `pow2_neg(n > FRAC)` returns `ZERO` (the truncated value) and the weight
  getters return whole matrices instead of taking indices. Only the caller's
  weight-init closure can panic. An invalid format (`FRAC` 0 or >= the storage
  width) is a compile error wherever its arithmetic is used. Debug builds still
  have overflow-check panics; the oracle and the debug-built golden runs have
  never triggered one.
- **Sanitizers / Miri / MSan / TSan: not needed** while the library is
  `#![forbid(unsafe_code)]` and single-threaded; safe Rust cannot have the UB they
  detect. If `unsafe` ever enters (SIMD intrinsics), add Miri and the matching
  sanitizer for that code.
- **Not done yet:** Kani proofs (the CBMC counterpart), a coverage report, and
  running the goldens on Cortex-M under QEMU.

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

Natural next steps, roughly by size: more activations (capped/leaky ReLU, linear;
`examples/energy_efficiency` is ReLU + linear output on the same MLP), multiple
hidden layers (a trait-based layer chain instead of C++ `LayerChain`
templates), then an int8 inference layer (`cpp/qdense.hpp` family) verified against
tinymind's `unit_test/integration` goldens.

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

- Pages: Home, XOR example, Iris example, Q-format in Rust, Equivalence with tinymind, API
  reference. Keep them about what is ported; link to tinymind's site for concepts
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
