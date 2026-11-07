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
- Reference tinymind commit for the goldens: `4e74539`.

## Commands

```bash
source ~/.cargo/env                 # rustup is user-level; cargo is not on the default PATH
cargo build --release
cargo run --release --bin xor       # writes output/ (gitignored); args: [out_dir] [lr] [momentum]
cargo test                          # unit tests + golden test against the C++ outputs
cargo clippy --all-targets -- -D warnings
cargo fmt --check
scripts/compare-with-cpp.sh         # builds ../tinymind's C++ xor and byte-diffs all outputs
```

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

`examples/xor/nn_fixed_xor.txt` etc. inside the tinymind tree are **stale, untracked**
files from an older build (they differ from line 825). Always regenerate the C++
reference (the compare script does) rather than trusting those.

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
- Repo is public at github.com/danmcleran/rustymind. Default branch `master`.
