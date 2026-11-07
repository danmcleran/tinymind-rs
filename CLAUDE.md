# CLAUDE.md

Guidance for Claude Code (and any fresh agent) working in this repository.

## What this is

`rustymind` is a **learning project**: Dan is using it to learn Rust by porting a
minimal slice of his C++ library [tinymind](https://github.com/danmcleran/tinymind)
(checked out as a sibling at `../tinymind`). It is not a product and not a full port.
A full port was assessed and rejected (2026-10-22): stable Rust lacks
`generic_const_exprs`, so tinymind's compile-time storage-type selection
(`QTypeChooser`) and arithmetic layer shapes (`[T; IN - K + 1]`) do not translate,
Arm DSP intrinsics (SMLAD) are nightly-only, and a C ABI erases generics. tinymind
(C++) stays the primary product.

Because the goal is learning, **explain the Rust idioms you use** (ownership, traits vs
C++ policies, const generics, `no_std`) in code comments and in replies, and point
out where Rust forces a different shape than the C++.

## Current state (2026-11-07)

- tinymind's `examples/xor` (2-4-1 MLP, Q16.16, tanh hidden + sigmoid output, SGD with
  momentum, 20000 steps) is ported and runs **bit-for-bit identically** to the C++
  build: the per-step weight trajectory (`nn_fixed_xor.txt`), learning curve,
  decision surface and stdout are byte-identical.
- Ported against the tinymind commit in `TINYMIND_REF` (verified byte-identical there).

## Pull model: how work arrives here

Porting is **pulled** from this repo: a session started at the rustymind root reads
tinymind and brings changes over. Nothing is pushed into rustymind from tinymind
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
cargo run --release --bin xor       # writes output/ (gitignored); args: [out_dir] [lr] [momentum]
cargo test                          # unit tests + golden test against the C++ outputs
cargo clippy --all-targets -- -D warnings
cargo fmt --check
scripts/compare-with-cpp.sh         # builds ../tinymind's C++ xor and byte-diffs all outputs
scripts/upstream-changes.sh         # tinymind commits since TINYMIND_REF touching ported sources
```

`cargo` comes from a user-level rustup install that `~/.profile`/`~/.bashrc` put on
PATH. A shell older than that install lacks it; use `~/.cargo/bin/cargo` then rather
than `source ~/.cargo/env && ...` (compound commands trip permission checks).

Gate before committing: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
`cargo test`, and `scripts/compare-with-cpp.sh` (needs `g++` and `../tinymind`).

Formatting is rustfmt default (K&R braces). Dan prefers Allman braces in C++, but Rust
code follows rustfmt; do not fight the formatter.

## Layout (Rust module <- C++ source)

| Rust | C++ (tinymind) | Notes |
|------|----------------|-------|
| `src/qformat.rs` | `cpp/qformat.hpp` | `QValue<const FRAC: u32>` over `i32` only; Truncate + Wrap policies only; `Fixed` trait = the implicit C++ "ValueType concept" |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp` | `Activation<V>` trait; `Tanh`/`Sigmoid` implemented for `Q16_16` only |
| `src/lut.rs` | `cpp/lookupTables.cpp` | Q16.16 tanh + sigmoid tables (96 entries each), extracted by script, never hand-typed |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp` | One-hidden-layer MLP, batch 1, default policies; plain arrays, no placement-new buffers |
| `src/rand.rs` | (glibc) | Bit-exact glibc `srand`/`rand` (TYPE_3 additive feedback) |
| `src/bin/xor.rs` | `examples/xor/xor.cpp`, `xornet.h` | Driver + `fmt_g` (C++ `ostream << double` formatting) |
| `tests/xor_golden.rs`, `tests/golden/` | | Goldens from the C++ run; trajectory checked by FNV-1a hash |

The library (`src/lib.rs`) is `#![no_std]` and allocation-free, mirroring tinymind's
freestanding build. Only the `xor` binary and tests use `std`. Keep it that way: new
library code must not need `alloc`.

## Equivalence rules (hard-won; read before changing arithmetic)

Bit-exactness against the C++ is the acceptance test. These details all matter:

1. **Operation order.** Fixed-point multiply truncates, so `(a*b)*c != a*(b*c)` in
   general. Keep the C++ expression shape exactly, e.g. the weighted sum is
   `bias_term + (0 + t0 + t1 + ...)` and the weight update is
   `(lr*g + momentum*dw) + accel*prev_dw`.
2. **Multiply** = `i64` product, arithmetic `>> FRAC` (floors toward -inf), wrap to `i32`.
   **Divide** = `(a as i64 * 2^FRAC) / b` truncating toward zero; divide-by-zero gives
   `i32::MAX`/`i32::MIN`/0 by dividend sign. `QValue / int` (`div_int`) is a different
   operation from `QValue / QValue::from_int(n)`.
3. **Add/sub wrap** (`wrapping_add`), comparisons are on the raw `i32`.
4. **Two seeds.** The C++ example seeds the initial weights with `WEIGHT_INIT_SEED = 1`,
   builds the network, then re-seeds with 7 for the training samples. The Rust driver
   uses two generators. History: the C++ network used to be a global, built before
   `main()` called `srand(7)`, so its weights silently came from glibc's default seed 1.
   Porting found it; tinymind PR #236 made the init seed explicit in 11 examples with
   byte-identical output, so the goldens did not change.
5. **Random-draw count.** C++ output neurons own one unused connection whose
   initialisation consumes a `rand()` draw. `Neuron<V, 1>` keeps it so later draws line up.
6. **Printing.** `QValue` `Display` prints raw LSBs (like C++ `operator<<`). Doubles use
   `fmt_g` (printf `%g`, precision 6), not Rust's `{}`.
7. The C++ computes input-layer node deltas that nothing reads; the port skips them.

Never trust output files lying around in the tinymind tree; regenerate the C++
reference (the compare script does). Stale ones were found there once (tinymind #237
fixed the docs that produced them).

## Porting the next example

1. Read the C++ example and every policy it instantiates, including defaults hidden in
   template parameter lists (`FixedPointTransferFunctions` has ~8 defaulted policies).
2. Port only the code paths that configuration executes. Add Rust tests per module.
3. Extract any lookup tables with a script (write it in the session scratchpad, run with
   `python3 -I`), asserting counts and first/last values.
4. Build the C++ reference into a temp dir and byte-diff. Divergence on the first data
   line usually means RNG / init order; later divergence usually means operation order
   or a rounding/saturation policy.
5. Commit small goldens to `tests/golden/`; hash large ones.

Natural next steps, roughly by size: a Q8.8 variant (needs `i16` storage, so a second
storage type: a good exercise in traits or macros), another activation (ReLU needs no
table), multiple hidden layers (a trait-based layer chain instead of C++ `LayerChain`
templates), then an int8 inference layer (`cpp/qdense.hpp` family) verified against
tinymind's `unit_test/integration` goldens.

## Conventions

- Project names are lowercase in prose: "tinymind", "rustymind".
- Copyright: `Copyright (c) 2026 Dan McLeran` (LICENSE). Code translated from tinymind
  also carries tinymind's MIT notice (Intel 2020 + Dan McLeran 2023) via `NOTICE`.
  Do not add Intel copyright lines to new files.
- No dependencies unless there is a real reason; the point is to learn the language.
- Python, if needed, goes in a venv/pyenv, never `pip install --break-system-packages`.
- Repo is public at github.com/danmcleran/rustymind. Default branch `master`. No CI or
  branch protection yet: commits so far go straight to `master` after the local gates
  pass. (In tinymind, "commit and push" means the full PR flow with CI; that rule does
  not apply here unless Dan asks for it.)

## Working with Dan

These live in tinymind's session memory, which sessions started here do not load:

- Be critical, not agreeable: when an idea has a concrete failure mode, say so and
  propose something better.
- A zero exit status or green check is not evidence. Confirm the result directly, and
  show that any new test or gate fails on a planted defect before trusting it.
- Ask before outward-facing actions (pushing to a new remote, publishing, deleting).
- Use the lowercase project names in all prose.
