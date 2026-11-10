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
number of hidden layers, `Linear`, gradient clipping, `energy_efficiency`), 0.1.4
(2026-11-07, tag `v0.1.4`: Adam/RMSprop, `Sqrt`, the `anomaly` example, panic fixes), 0.1.5
(2026-11-08, tag `v0.1.5`: `tinymind::tree`, the tree importer, `gbdt_tabular_int8` and
`imported_trees`) and 0.2.0 (2026-11-09, tag `v0.2.0`: int8 conv/pool, `QAdd`,
`QMul`, `QPad2D`, `QConcat`, `InferenceNetwork`, `f32`/`f64` networks with
`AdamFloat`/`RmsPropFloat`, the optional `libm` feature, and the breaking rename
`Fixed` -> `Scalar` with `Fixed` kept as a deprecated alias). Agents run the
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
PYTHON=<python with numpy> scripts/compare-with-cpp.sh   # all C++ comparisons; the importer checks need numpy, scikit-learn, xgboost, lightgbm (CI pins them; here: a venv)
python3 -I scripts/extract-luts.py  # regenerates src/lut.rs from ../tinymind's lookupTables.cpp
CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true cargo test --release --test qformat_oracle -- --ignored  # every Q8.8 pair (~15 s)
scripts/upstream-changes.sh         # tinymind commits since TINYMIND_REF touching ported sources
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

| Rust | C++ (tinymind) | Notes |
|------|----------------|-------|
| `src/qformat.rs` | `cpp/qformat.hpp`, `cpp/typeChooser.hpp` | `QValue<S, const FRAC: u32>` over `S` = `i32` (wide `i64`) or `i16` (wide `i32`); methods generated per storage type by `impl_qvalue!`, because stable Rust cannot pick a type from const-generic arithmetic (`QTypeChooser`); Truncate + Wrap policies only; `Scalar` trait (was `Fixed`, now a deprecated alias) = the implicit C++ "ValueType concept", implemented for the Q formats, `f32` and `f64` |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp`, `cpp/tanh.hpp`, `cpp/sigmoid.hpp` | `Activation<V>` trait; `Relu` for every `Scalar` type; `Tanh`/`Sigmoid` for `Q16_16` and `Q8_8` |
| `src/lut.rs` | `cpp/lookupTables.cpp` | Q16.16 and Q8.8 tanh + sigmoid tables (96 entries each), generated by `scripts/extract-luts.py`, never hand-typed; CI checks they match `TINYMIND_REF` |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp` | One-hidden-layer MLP, batch 1, default policies; plain arrays, no placement-new buffers. `InferencePerceptron` = C++ `IsTrainable = false`: a separate type holding only weights (no per-neuron state, `const fn new`, `feed_forward(&self)`), same sum order as the trainable one; checked by `tests/inference_mlp.rs` (golden from `scripts/cpp/inference_mlp.cpp`: tinymind's untrainable net loaded through its setters, 4 configs) and unit tests vs `MultilayerPerceptron`. Whole-matrix weight setters, not per-index (no panics) |
| `src/network.rs` | `cpp/neuralnet.hpp` (`NeuralNetwork`, chain path: `InnerLayerChainBuilder`, `ChainNetworkDeltasCalculator`, `ChainNetworkGradientsCalculator`, `ChainBackPropConnectionWeightUpdater`) | Any number of hidden layers: `NeuralNetwork<L>` over a flat tuple of 1-8 `Dense<V, A, IN, OUT>` (impls generated by `impl_layers!`; sealed `Layers` trait; sizes chained by the impl's const params). Rust has no variadic generics, so no `HiddenLayers<...>`. Per-layer activation (C++ uses one for all hidden layers). Init order = C++ by source layer (inputs then bias), then one draw per output neuron. Deltas all computed before any update; update order irrelevant (each connection reads only itself). Checked by `tests/deep_mlp.rs` (golden from `scripts/cpp/deep_mlp.cpp`: tinymind's `NeuralNetwork` with `HiddenLayers<4,4>`, `<8,6,5>`, `<8,6>`, both formats, every weight hashed after every step) and unit tests vs `MultilayerPerceptron` **Inference-only form** (2026-11-09): `InferenceDense<V, A, IN, OUT>` (pub `weights`/`bias_weights`, `const fn new`, `forward` in the C++ sum order like `Dense`) and `InferenceNetwork<L>` (pub `layers`, `const fn new`, `feed_forward(&self)`) over tuples of 1-8, sealed `InferenceLayers` generated by the same `impl_layers!`; `LayersImpl::Inference` (bound to the same Inputs/Outputs) with `to_inference`/`set_from`, exposed as `NeuralNetwork::to_inference`/`set_weights`. Checked by `tests/inference_deep.rs` (golden from `scripts/cpp/inference_deep.cpp`: tinymind's `NeuralNetwork<..., HiddenLayers<...>, ..., false>` loaded through its setters, 5 configs x 8 weight sets x 16 inputs, plus the trainable C++ net agreeing); plants (no activation, bias of neuron 0, set_from skipping biases, first input only) fail it |
| `src/optimizer.rs` | `cpp/adam.hpp`, `cpp/rmsprop.hpp` (`SquareRootApproximation` -> `qformat::Sqrt`; `AdamOptimizerFloat`/`RmsPropOptimizerFloat` -> `AdamFloat`/`RmsPropFloat`, see Floating point below) | Optimizer = type parameter (`Opt = Sgd`, zero-sized, so no size cost; `new` exists for `Sgd` only, `with_optimizer` for any). Sealed `OptimizerImpl` (allow `private_interfaces`: it takes the crate-private `Connection`). Faithful quirks, reported to Dan, not on the site: C++ calls `step()` at the start of each *layer* update (output layer first), so Adam's `t` advances per layer; the adaptive path has no bias branch, so bias weights never change. Moments live in the delta-weight slots. `(1 - beta2) * g * g` is left-associative. No `Default` for `Adam`/`RmsProp`: tinymind's defaults are raw values tuned for Q8.8. `NeuralNetwork` now updates layers output-first (order was irrelevant for SGD, matters for Adam's `t`). Checked by the adaptive runs in `tests/deep_mlp.rs`; planted defects (per-step `t`, bias updates, input-first order, `g*g` grouped) each fail |
| `src/qlearn.rs`, `examples/maze.rs`, `examples/common/maze_driver.rs` | `cpp/qlearn.hpp` (`QLearner`, `QValueTablePolicy`, `ArgMaxPolicy`, `DefaultLearningPolicy`, `QLearningEnvironment`), `examples/maze` | 2026-11-09. `QLearner<V, E, STATES, ACTIONS, Q = QTable>` owns its `Environment<ACTIONS>` (user trait: `valid_actions` fills a buffer, `random_int(min, max)` = the C++ RandomNumberPolicy's `randInt`) and its `QValues<V>` (open trait; `QTable { pub values: [[V; A]; S] }`, out-of-range get 0 / set ignored). States and actions are `usize`, invalid = `None` (C++ uses the type's max). Learning rate, discount (`< ONE`) and exploration percent (`<= 100`, the C++ example policy's check) live in the learner; initial 0 (the C++ example policy leaves the percent uninitialized). The C++ `environment.takeAction(experience)` hook is dropped (the maze only stores it). `Experience` has no `state`: the C++ update reads `mState`, never `experience.state`. Faithful quirks: the future Q is `Q(s', chooseAction(s'))`, exploration included (draws RNG), not max; ArgMax with 2 actions picks a1 on ties; with 3+ (since tinymind #258, `TINYMIND_REF` 0425b66) the first of the largest Q values whatever its sign, and a random action only when every valid action ties (before #258 it started from `bestQValue = 0`, so non-positive Q values fell back to a random action); nothing computed when lr = 0 (no draws); with no current state the update only moves. The C++ maze uses unsigned Q16.16 (`QValue<16,16,false>`); the port uses signed `Q16_16` (all values non-negative, -1 reward only compared), proved by the sweep hashing the unsigned C++ table. C++ `Constants` has no float specialization, so its QLearner needs a Q format; ours also runs on f32/f64. Checked by `tests/maze_golden.rs` (stdout + 2 CSVs, byte-identical first run) and `tests/qlearn_sweep.rs` (golden from `scripts/cpp/qlearn_sweep.cpp`, UBSan clean: maze in UQ16.16 + 60 random envs in Q16.16/Q8.8 with 0/1/2/3+ valid actions, negative rewards, lr/df 0, exactly 1, > 1, exploration 0/100/150/random with decay, every 4th episode starting with an update before any action). Plants that fail it: argmax floor -2, and for the #258 rule: no all-tied check, ties to the last, a tie broken only by a larger value, the old 0 floor; 2-action tie to first, textbook max future, future drawn at lr 0, explore `<`, lr = 1 or df = 1 accepted, percent 150 accepted, update at episode start (needed the explicit-1 rate and no-state update cases: both passed before them). Equivalent mutant: old Q at episode start (computed, never stored). Kani `qlearner_never_panics` (304 checks, ~4 s; planted `valid[index]` fails it and panic-never). **Not ported yet: `QValueNeuralNetworkPolicy` (DQN)**. Until tinymind #258 its `getQValue(state, action)` ignored `state` and returned the online net's last outputs (shown by a probe: identical values for all six states); #258 feeds the state forward (cached until `setQValue` trains) and changed `examples/dqn_maze` to a one-hot state and a linear output. `QValues::q_value` takes `&self`; the DQN's needs a forward pass, so porting it changes that trait (unreleased, so no semver cost before the next release) |
| `src/recurrent.rs`, `examples/har_activity.rs`, `examples/common/har_driver.rs` | `cpp/neuralnet.hpp` (`LstmNeuralNetwork`, `LstmHiddenLayer`, `TrainableLstmHiddenLayerNeuron`, `ChainBackPropagationThruTimePolicy` and the recurrent chain calculators), `examples/har_activity` | 2026-11-09, against tinymind #257 (`TINYMIND_REF` 0425b66). `LstmNetwork<V, H, O, IN, HID, OUT, Opt = Sgd>`, one hidden layer (the C++ multi-layer recurrent chain is not ported: never initialised or trained, per the findings). Stable Rust cannot write `[_; 4 * HID]`, so gate connections are `[[Connection; 4]; HID]` (`GatedNeuron`); nesting order = the C++ flat index `n * 4 + g`, so init draws line up. Gate order candidate (H = TF hidden activation), input, forget, output (Sigmoid), cell output `Tanh(c) * o`; `c = (f * c_prev) + (cand * i)`. Init order: inputs, input bias, cells (OUT each), hidden bias, outputs (1 unused each), recurrent (last). Forward latches h[t-1] into the recurrent layer first (#257's `latchPreviousOutputs`), sums are bias + inputs + recurrent, accumulated source-outer (same per-target order). Train: output deltas, raw dL/dh (`calculateAndSetRawNodeDeltas`), `computeGateDeltas` with each activation's derivative (left-assoc products as written), gradients output/hidden-bias, recurrent (`h[t-1] * gate delta`, LSTM recurrent delta = gate delta), input and input bias; recurrent-layer and input node deltas skipped (nothing reads them). Updates layer by layer (output, recurrent, input) so Adam's `t` advances 3x per step as in the C++; bias never clipped; recurrent layer has no bias. `reset_state` zeroes c, h and the latched h. Checked by `tests/lstm_sweep.rs` (golden from `scripts/cpp/lstm_sweep.cpp`, UBSan clean: 7 configs incl. har's 3-16-4, Q8.8, ReLU/linear, asymmetric clip, Adam; sequences 1-12 with resets, every 7th step untrained, outputs + all weights hashed per step), 4 float LSTM runs in `tests/float_mlp.rs` (f64 golden, f32 live), and `tests/har_golden.rs` (stdout + 3 CSVs, byte-identical first run; 31 s debug). Plants that fail the sweep: in-place h write (R2), reset keeping h (R5), hidden bias untrained, forget delta with c[t], dL/dc regrouped, recurrent gradient with h[t] (R4), Adam stepping twice, input bias clipped, recurrent sum dropped, cell init reversed. har's C++ is built with `-ffp-contract=off` (double sin/cos data). Host timing (best of 5): Rust release 0.75 s vs C++ -O3 0.70 s. Kani `lstm_never_panics` (Q8.8 1-1-1, unwind 6, ~80 s; a reachable planted assert fails it; an unreachable one, `h != 1`, passed, so plants must be reachable). panic-never covers Q16.16 3-16-4, Q8.8 ReLU/linear, Adam, f32, f64 AdamFloat. Not ported: GRU, Elman, inference-only LSTM, `setWeights` |
| `src/rand.rs` | (glibc) | Bit-exact glibc `srand`/`rand` (TYPE_3 additive feedback) |
| `examples/xor.rs`, `examples/common/xor_driver.rs` | `examples/xor/xor.cpp`, `xornet.h` | Example `main` (arguments) + driver (the program, generic over the format, `--format q16.16\|q8.8`); the golden test includes the driver and runs it in-process |
| `src/int8.rs` | `cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qactivations.hpp`, `cpp/include/simd/simd_dispatch.hpp` (scalar dot) | Int8 affine inference: gemmlowp primitives, `Requantizer` (i8 output only; both C++ paths, fused and saturating), `QDense<'a, IN, OUT>` borrowing `&[[i8; IN]; OUT]` weights (flash-friendly), `qrelu`/`qrelu6`, 256-entry LUT. Integer only, no_std |
| (same, `QSoftmax`) | `cpp/qsoftmax_fixedpoint.hpp` | TFLite-exact int8 softmax (`QSoftmaxFixedPoint1D`), one row per `forward::<N>` (the C++ `Rows` loop is the caller's); `N` in `1..512` checked at compile time. tinymind's other softmax, `QSoftmax1D` (exp LUT + int64 divide, TFLite to within rounding), is deliberately not ported. Checked by `tests/softmax_tflite.rs` (TFLite Micro goldens and sweep hashes, extracted by script from tinymind's unit test) and `tests/softmax_sweep.rs` (golden from `scripts/cpp/softmax_sweep.cpp`: tinymind's kernel + `buildQSoftmaxFixedPointParams`, 504 calibrations, rows of 1 to 511) |
| `src/int8/calibration.rs` | `cpp/include/qcalibration.hpp`, `cpp/qactivations.hpp` | Feature `std`: `quantize_multiplier` (with a ported `frexp`), `build_requantizer`, `quantize`/`dequantize`, sigmoid/tanh LUT builders, `build_softmax` (`buildQSoftmaxFixedPointParams`) |
| (same, `QDensePerChannel`) | `cpp/qpointwiseconv2d.hpp` (1x1, per channel) | Dense with one requantizer per output, as TFLite quantizes dense layers; verified on the boards against tinymind's benchmark hash |
| `src/int8/conv.rs`, `src/int8/pool.rs` (re-exported from `int8`) | `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qpool2d.hpp` | `QConv2D`/`QConv2DPerChannel`, `QDepthwiseConv2D`, `QPointwiseConv2D`/`...PerChannel`, `QMaxPool2D`, `QAvgPool2D`, `QGlobalAvgPool2D`. NHWC nested arrays (`[[[i8; C]; W]; H]`, conv weights OHWI); kernel, strides and padding are layer const params (defaults: stride 1 / pool stride = window, no padding); `forward(&self, input, &mut output)` takes H, W, OH, OW from the arrays and checks `OH == (H+PT+PB-KH)/SH+1` etc. in an inline `const { assert!() }` (stable since 1.79), so a wrong size or invalid layer is a compile error: this is how stable Rust gets around `OutputHeight` arithmetic. Loops tuned on the boards (4C, then 2026-11-09): conv is pixel-outer (window read once for all filters; filter-outer measured 20% slower for the Cin=1 stem), now as `conv2d_gather` (C=1) or `conv2d_band` (C>=2); the original per-tap `conv2d` was removed; depthwise and pointwise run filter-outer with the zero point folded into the bias (`bias - zp*sum w`, equal mod 2^32), weights held across pixels, interior windows read as slices (padded ones take the general per-tap path), and the requantizer prepared once per filter: `int8::FastRequantizer` (crate-private) is the C++ FastRequantizer extended to any qmin/qmax (fused ReLU), proved equal to `Requantizer::apply` by Kani (`fast_requantizer_matches_apply`); filters without a fast form run a `#[cold]` out-of-line loop (one loop branching per output cost depthwise 27%). Pointwise on one pixel keeps the per-output form. Same `i32` sums as the C++ modulo 2^32. Checked by `tests/conv_sweep.rs` (golden from `scripts/cpp/conv_sweep.cpp`: 57 configurations x 3 trials, every C++ path incl. patch/row/tiles/odd filters/all-padding windows, both requantizer paths and the C++ FastRequantizer vs general form, pool ranges with qmin > qmax); ten planted defects each fail it. **16-bit band path** (2026-11-09, for ResNet-8): `QConv2D`/`QConv2DPerChannel` with `C >= BAND_MIN_CHANNELS` (2) run `conv2d_band`: per output row the KH input rows widen to an i16 band (`x - zp`), per (row, filter) the weights to i16, interior windows are KH `dot16` calls over contiguous KW*C, edges tap by tap, cold general-requantizer rows tap by tap (small code). Scratch on stack ~2*KH*(W+KW)*C B; not size-limited. LLVM's ARMParallelDSP pass makes `SMLAD` from `acc + x[i]*w[i] + x[i+1]*w[i+1]` over i16 loads (plain safe Rust, opt 3), but it analyses at most 16 loads (`-arm-parallel-dsp-load-limit`, default 16), and fully unrolled constant-length sums exceed it, so by default the code is ldrsh/ldrsh/smlabb (~4.7 c/MAC); `-C llvm-args=-arm-parallel-dsp-load-limit=512` gives pure ldr/ldr/smlad (opt-in, documented; not in bench config). Tried and dropped: packed u32 words (needs `[u32; W*C/2]`, inexpressible; assembling words from i16 folds back to ldrsh), `black_box` length + multi-exit loop (SMLAD but loop overhead ~2.75 c/MAC), unsigned byte lanes (UXTB16 trick: LLVM splits to per-byte uxtb, worse), 2 filters x 2 pixels (68.0M default, 48.1M with flag vs 45.9M, +68 KB flash). Stable Rust exposes no Arm target features in cfg, so no DSP gate: measured on the L031 a 10x10x8 3x3 conv 675,064 vs 835,734 cycles on the old path (F411 303,574 vs 451,012), so the band path is used everywhere; C=3 stem also faster (2.75M vs 3.88M). Kani `conv_band_never_panics` (C=2, unwind 6, 69 s); conv_sweep plants (odd tail, zp not subtracted, band rows off by one) fail. Not ported: `QPad2D` (tinymind's translator inserts it only when a padded band exceeds its buffer limit; these layers pad internally without one) |
| `src/int8/add.rs` (re-exported from `int8`) | `cpp/qadd.hpp`, `buildQAddParams` in `cpp/include/qcalibration.hpp` | `QAdd` (TFLite ADD): not size-parameterized; `forward<T: Tensor>(&a, &b, &mut out)` over the sealed `int8::Tensor` (`[i8; N]`, `[[i8; C]; W]`, `[[[i8; C]; W]; H]`), one `T` so shapes must match at compile time (compile_fail doctest). The C++ `(a - zp) * (1 << left_shift)` is UB outside `0..=23`; here a negative left shift is 0 and a larger one saturates (`saturating_shl`, the same shift as in `multiply_by_quantized_multiplier`, Kani `saturating_shl_saturates` vs an i128 spec); the sum wraps. `multiply_by_quantized_multiplier` keeps its own copy: calling `saturating_shl` from it moved code by 10 bytes and the F411 DS-CNN took 11% more cycles (17.88M vs 16.12M; with `-C llvm-args=-align-loops=16` both builds take 16.197M: F411 timings depended on hot-loop alignment against flash lines; the bench now aligns loops, see On-board checks). A Kani proof that the two shifts agree took 566 s (symbolic 32x32 multiply), too slow for CI. From 256 elements `forward` tabulates `scaled_a`/`scaled_b` for all 256 int8 values (2 KB stack) and uses `FastRequantizer`: ResNet-8 adds 185 -> 22 cycles/element; `add_sweep` checks the element-wise path (chunks of 8) equals the table path; plants (table b from a, table off by one) fail; panic-never covers both paths. `calibration::build_add((scale, zp), (scale, zp), (scale, zp), qmin, qmax)` = `buildQAddParams` (f64 from f32 scales, left shift 20). Note make_mlperf.py computes the ADD parameters differently (TFLM's float32 `twice_max` and denominator), so the translator must follow it, not `build_add`. Checked by `tests/add_sweep.rs` (golden from `scripts/cpp/add_sweep.cpp`, built with UBSan clean: 48 calibrated layers x all 65,536 input pairs, 64 direct random-parameter layers, 16 saturating the requantizer's left shift); plants (b with a's multiplier or zero point, left shift 0 as 1, min for max scale, real_out off by 1e-7, real_a in f32, wrapping instead of saturating left shift) each fail it. Before this sweep nothing exercised the left-shift saturation of `multiply_by_quantized_multiplier` (a wrapping plant passed every suite) |
| `src/int8/compose.rs` (re-exported from `int8`) | `cpp/qmul.hpp`, `cpp/qpad.hpp`, `cpp/qconcat.hpp`, `buildQMulRequantizer`/`buildRescaler` | `QMul` (`forward<T: Tensor>`, FastRequantizer prepared once; product fits i32), `QPad2D<PT, PB, PL, PR> { pad_value }` (const assert OH = H+PT+PB, OW = W+PL+PR with checked adds), `QConcat` (C++ `QConcat2_2D`; const assert CA, CB > 0 and C = CA + CB). `calibration::build_mul_requantizer` and `build_rescaler` are thin wrappers over `build_requantizer` (same f64 ratio as the C++). Checked by `tests/compose_sweep.rs` (golden from `scripts/cpp/compose_sweep.cpp`, UBSan clean: 32 calibrated QMul on all 65,536 pairs, 48 direct, 5 concat shapes x 4, 6 pad shapes x 3); plants (b with a's zp, pad rows offset by PL, pad with 0, concat b with a's requantizer, rescaler scale nudged, both compile_fail doctests made valid) fail; Kani `compose_never_panics` (22 s); panic-never covers all three. Translator support: PAD (H/W only, pad value = output zp, input/output quantization equal) -> QPad2D, MUL (same shapes; ratio via `rq`) -> QMul, CONCATENATION (2 maps on channels, all quantized as the output: TFLite int8 copies then, else rescales in float) -> QConcat with identity requantizers; checked end to end by `tools/import_tflite/test_ops.py` against LiteRT's reference kernels (ai-edge-litert 2.3.0, `OpResolverType.BUILTIN_REF`) on three models built by `tflite_build.py` (flatbuffers via the `tflite` package's builders, no TensorFlow); plants (MUL ratio +0.1%, MUL ReLU dropped (visible only because fused-ReLU outputs are calibrated before the ReLU), pad 0, pad L/R swapped, concat order swapped) fail it |
| `examples/kws_cortex_m_int8.rs`, `examples/common/kws_int8_driver.rs`, `examples/common/std_random.rs`, `scripts/cpp/kws_int8_dump.cpp` | `examples/kws_cortex_m_int8/kws_cortex_m_int8.cpp` | Feature `std`. Draws from `std::mt19937(42)` + libstdc++'s `uniform_real_distribution<float>` (`std_random.rs`: GCC 13's `generate_canonical<float,24>` is one draw, `float(u) / 2^32`, then `u * (b - a) + a` in f32; x86-64 has no FMA by default, but g++ 15 on CI's Ubuntu 26.04 targets FMA and contracts it, moving three depthwise scales by an ulp: the dump is built with `-ffp-contract=off`). Since tinymind #255 (ported, `TINYMIND_REF` 1389e8f) each layer's activation scale is fitted from a float reference forward pass over 32 calibration inputs (`RangeObserver`, `affine_params_symmetric`); the float sums follow the C++ loop order exactly. The C++ stdout starts with host-ns timings, so the check is a dump: `kws_int8_dump.cpp` `#define main`s and `#include`s the example source unchanged, captures its cout, prints input, weights, requantizers and every layer output, then the stdout from "Logits (int8" on; `tests/kws_int8_golden.rs` compares the driver's `dump()` plus that report. Two plants (31 calibration inputs; observer starting at 0) do not change this model's output, so `RangeObserver` has unit tests. Rust weight_bytes differ only for max pool (ZST 0 vs C++ 1). Planted defects (f64 distribution, per-channel scale off, transposed dw kernel, per-tensor dw requantizer) each fail it |
| `examples/int8_xor.rs`, `examples/common/int8_xor_driver.rs` | `examples/pytorch_quant/xor/xor_quant.cpp`, `weights.hpp` | Weights and scales copied from `weights.hpp` (the goldens catch any mistake) |
| `tools/import_pytorch/tinymind_import.py` | `apps/import_pytorch/tinymind_import.py` | PyTorch importer emitting Rust. Calibration/quantization/observers copied unchanged (Dense/ReLU/Sigmoid/Tanh/Softmax subset); new: integer requantizers (`buildRequantizer` on float32 scales), LUTs via libm `expf`/`tanhf` over ctypes, softmax params (`buildQSoftmaxFixedPointParams` on the float32 scale; tinymind's importer leaves softmax to the hand-written C++), `forward`, `from_torch`, rustfmt'd output. `test_parity.py` vs tinymind's module; `example_mlp.py` (no torch) and `torch_xor.py` (torch) |
| `tools/import_tflite/tinymind_import_tflite.py`, `mlperf.py`, `test_translator.py`, `bench/src/mlperf.rs` | `benchmarks/framework_comparison_f411/gen/make_mlperf.py` | int8 .tflite -> Rust module (4E). tinymind's helpers copied unchanged (`quantize_multiplier` with Python `round`, `act_range`, `rq`, `same_pads`, softmax params); reads the flatbuffer with the `tflite` package (pinned 2.18.0 + flatbuffers 25.12.19; no TensorFlow). Ops: CONV_2D, DEPTHWISE_CONV_2D, global AVERAGE_POOL_2D, per-tensor FULLY_CONNECTED, one-row SOFTMAX, RESHAPE, ADD (same shapes; `add_params` copied from make_mlperf.py's add(), TFLM's float32 `twice_max`, emitted by `add_literal`). The graph may branch: steps record input/output tensor indices (RESHAPE without a copy is an alias), and `forward` keeps feature maps in buffers per shape, a buffer reused once no later step reads its tensor (liveness), so a chain ping-pongs between two and a residual block keeps its skip tensor: nesting each layer in a block peaked at 3 tensors (the returned value got its own slot), 24.5 KB vs 16.3 KB for the DS-CNN. Since 2026-11-09 the maps share one byte arena planned first-fit by liveness (as make_mlperf.py's planner; ResNet-8 49,152 B = the C++'s); each step does `split_at_mut` around its output and views slots with generated `view`/`view_mut` (`as_chunks` + `first_chunk`, safe; the generated module needs Rust 1.88, the crate stays 1.85). Per-shape typed buffers had needed 86.5 KB. `mlperf.py kws` fetches the pinned DS-CNN (sha256; cached in target/tflite-models; Apache-2.0, never committed), runs make_mlperf.py's seeded inputs through cargo and compares with `TM_MLPERF_KWS_HASH` (0xF623A166) from tinymind's `fw/backend.hpp`; `--out/--data` write the bench files. `test_ops.py` (needs ai-edge-litert) checks PAD/MUL/CONCATENATION against TFLite's reference kernels on generated models; `mlperf.run_module(output_map=True)` flattens map outputs. `mlperf.py ic` does the same for ResNet-8 (`TM_MLPERF_IC_HASH` 0x8897F6ED; matched on the first run). Four planted translator bugs change the DS-CNN hash; on ResNet-8, ADD with b's multiplier for a, a's zero point for b, left shift 19, and the old ping-pong buffer rule (overwrites the skip tensor) each change it. Fused ReLU (also on ADD), the softmax radius and nudges to `twice_max` do not show on these models (every activation zero point is -128), so `test_translator.py` checks them (softmax params vs the 504-calibration golden, `add_params` vs the 48 calibrated `buildQAddParams` lines of tests/golden/add: the scaling by 2 and 2^20 is exact, so TFLM's float32 and the C++'s double agree). F411 DS-CNN: 13,225,965 cycles (C++ scalar 15,736,026, +SMLAD 8,193,064, TFLM CMSIS-NN 9,583,099), 46,416 B flash, 16,680 B RAM. F411 ResNet-8 (`mlperf-ic`): 64,407,536 cycles (C++ scalar 85,467,754, +SMLAD 26,753,191, TFLM CMSIS-NN 37,372,049), 130,184 B flash, 57,692 B RAM; with `-arm-parallel-dsp-load-limit=512` 41,401,469 cycles. Before the band path, ADD tables and arena: 91,085,417 / 101,076 B / 86,520 B. Per layer (default, before ADD tables): conv1/2 11.1M each, conv5 13.3M, conv9 11.4M, conv4 5.7M, conv8 6.6M, conv0 2.7M, adds 3.0/1.5/0.76M |
| `tools/import_pytorch` conv support (4D), `examples/imported_cnn.rs`, `examples/common/imported_cnn{,_flat}_model.rs`, `examples/common/imported_cnn_driver.rs`, `scripts/cpp/imported_cnn.cpp`, `tools/import_pytorch/{example_cnn,torch_cnn}.py` | (tinymind importer's Conv2D descriptor + C++ runtime) | `Conv2D` gains `stride`/`padding` (t,b,l,r)/`depthwise` (tinymind's leaves layer types to hand-written C++); new `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`, `Flatten` (pass-through: range = input's). `_plan` follows ("vec", n)/("map", h, w, c); conv emits `QConv2D`/`QDepthwiseConv2D` (per-channel rq array, all equal)/`QPointwiseConv2D` (1x1, stride 1, no pad) with the output size spelled out; a ReLU after a conv is folded into the requantizer's qmin (= qrelu after it). `from_torch(model, input_shape=(C,H,W))`: OIHW->OHWI, NCHW->NHWC, Linear-after-Flatten columns reordered, "same" padding; `torch_cnn.py` checks it against PyTorch (needs torch, local only; four planted layout bugs fail it). Parity: tinymind's importer has no pools/flatten, so `example_cnn.py --cpp` folds each into the next layer's forward (same ranges). The C++ check prints a hash of every intermediate tensor (the models' outputs vary little). A ReLU's range starts at 0, so its zero point is -128 unless a KL observer gives a symmetric range: `cnn`'s relu3 uses KL so the qmin fold is exercised. The float references (`conv2d_nhwc`, `dense_ref`, the average pools) accumulate in float64 without BLAS (`einsum(optimize=False)`) and round to float32: with float32 einsum/`@` the calibrated ranges depended on the CI runner's CPU (BLAS summation order), moved by an ulp, changed requantizer multipliers, and the regenerated `cnn` module differed on CI (2026-11-09; #50 had passed on another runner). `BatchNorm2D` after a `Conv2D` is folded by tinymind's `fold_conv_bn`/`fuse_layers` (copied, with the fused layer also keeping stride/padding/depthwise), including tinymind #256 (2026-11-09, `TINYMIND_REF` 02bea40): before it the fused layer kept `forward=conv.forward` and calibration fitted the pre-BN range (found here, reported). `import_pytorch_model` and `run_float` fuse first; `cnn` folds two BNs (one depthwise); `torch_cnn.py` also runs each conv from its folded weights. |
| `examples/imported_mlp.rs`, `examples/common/imported_{mlp,classifier}_model.rs`, `examples/common/imported_mlp_driver.rs`, `scripts/cpp/imported_mlp.cpp` | (tinymind importer + C++ runtime) | The importer's two example models (sigmoid MLP, softmax classifier); the C++ program runs tinymind's headers for the same models, with `QSoftmaxFixedPoint1D`. The `*_model.rs` files are generated: regenerate with `example_mlp.py [--model classifier] --rust`, never edit |
| `tools/import_trees/tinymind_import_trees.py` | `apps/import_trees/tinymind_import_trees.py` | Tree importer emitting Rust. Everything up to `export_trees` (FeatureQuant, readers, plan, threshold/leaf/base rules) copied unchanged from tinymind (only `namespace` -> `name`, docstrings); new: `_render` (Rust: `quantize_features`, pool form via `QGbdt` or one `QDecisionTree` per voting tree, nested-`if` `tree_k` fns, `const _: () = assert!(..is_valid())`), rustfmt, `verify` via cargo (temp crate, path dep, optional shared `target_dir`). `test_parity.py`: 18 models, integer model identical to tinymind's + tinymind's g++ verify + our cargo verify. `FeatureQuant.from_data` used to clip the zero point to `[qmin, qmax]` (found here 2026-11-08: features far from 0 saturated; iris sepal length all at 127); tinymind #254 stopped clipping it (bounded by `_ZP_LIMIT` so `quantize_features`' `i32` sum cannot overflow, else the feature's range is widened to include 0), ported here with the example modules regenerated; `test_parity.py` has an offset case and a far-offset (fallback) case |
| `examples/imported_trees.rs`, `examples/common/imported_trees_{driver,gbdt_model,forest_model,data}.rs`, `scripts/cpp/imported_trees.cpp` | (tinymind tree importer + C++ runtime) | Wine (sklearn `load_wine`, CC BY 4.0, `NOTICE`), 70/30 split: GradientBoosting (int8) and RandomForest (int16). `*_model.rs` and `imported_trees_data.rs` are generated by `tools/import_trees/example_trees.py`, never edit; CI retrains with pinned numpy/scipy/scikit-learn and checks they are current |
| `examples/energy_efficiency.rs`, `examples/common/energy_driver.rs`, `data/ENB2012_data.csv` | `examples/energy_efficiency/energy_efficiency.cpp`, `ENB2012_data.csv` | 8-16-2 regression: ReLU hidden, `Linear` output, `set_gradient_clip((-1, 1))`, lr 0.02, momentum and acceleration 0. Data is CRLF in tinymind's working tree (`eol=crlf`), so `.gitattributes` here pins the same; CC BY 4.0 (`NOTICE`). Gradient clipping (`cpp/gradientClipping.hpp`) applies to weights only: tinymind's `BiasNeuronConnectionWeightUpdater` uses the raw gradient (and no weight decay). This example never pushes a bias gradient past 1, so `scripts/cpp/deep_mlp.cpp` adds two clipped regressions that do (a planted bias clip fails them) |
| `examples/anomaly.rs`, `examples/common/anomaly_demo.rs`, `bench/src/anomaly.rs` | (none) | On-device anomaly detection: 8-4-8 tanh/linear Q16.16 autoencoder, SGD + clip, trained on synthetic normal windows (integer sine table + glibc rand), threshold = 2x max normal error. `no_std`, shared by host and board; hash pinned by `tests/anomaly_golden.rs` (Rust golden, documented exception). Results on `docs/anomaly.md` and `bench/README.md` |
| `src/tree.rs` | `cpp/qtree.hpp` (`QTreeNode`, `walkQTree`, `QDecisionTree`, `QGBDT`) | Runtime trees over a borrowed pool (`&[QTreeNode<T>; N]`), generic over threshold and input types (`Copy + Into<i32>`, compared in `i32` as the C++ casts). `QGBDT` -> `QGbdt` (Rust acronym style). Out-of-contract indices (child/root/feature/class), UB in the C++, are total here: the walk stops with 0, a bad class is skipped; `is_valid` is the `const fn` form of `qtreePoolIsValid` plus root/class checks. Sums wrap. `predict` carries the running best by value (indexing `scores[best]` kept a bounds check at opt s/z, found by the panic gate). Not ported: `QStatic*` and the Inline/Compact/Walk policies (C++ template expansion; Rust equivalent is importer codegen). Checked by `tests/tree_sweep.rs` (golden from `scripts/cpp/tree_sweep.cpp`: 72 random ensembles, i8/u8/i16, base scores, ties, inputs at thresholds); planted defects (`<` for `<=`, ties to the highest class, base score 1) each fail it |
| `examples/gbdt_tabular_int8.rs`, `examples/common/gbdt_driver.rs` | `examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp` | Feature `std` (`affine_params_asymmetric`, `quantize`). stdout, CSV and `--golden` report byte-identical. The C++ README says feature 1 "nudges the boundaries"; it never changes the winner (bands are pure vertical; 48 cells flip at f0 = -0.149) |
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
  seven minutes): 71 harnesses, for every input. For Q8.8, Q15.1, Q1.15, Q16.16, Q31.1
  and Q1.31: add/sub wrap, multiply floors then wraps, division never panics,
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
one idea from the report not done. Trees: the runtime (`src/tree.rs`, 3A) and
the importer (`tools/import_trees`, 3B) are done; tinymind's packed
`QTreeCompactEval` form is not. Int8 conv (priority 4): the layers are done
(4A, 2026-11-08: output sizes spelled out by the caller and checked at compile
time). 4B, `examples/kws_cortex_m_int8`, is done (every tensor and the logits
report, including tinymind #255's per-layer calibration). 4C, the
framework-comparison KWS CNN on both boards, is done and tuned (13-27% fewer
cycles than the C++ scalar build). 4D, convolutions in the PyTorch importer, is
done, including BatchNorm folding (tinymind #256 ported). The TFLite translator (4E) runs the MLPerf DS-CNN with tinymind's hash on the
F411, and ResNet-8 too (`QAdd` ported, ADD in the translator), with RAM and cycles tuned (arena, 16-bit band path, ADD tables). Possible next: the same 16-bit pairing for pointwise/depthwise/dense (the DS-CNN's pointwise layers), and 2x2 register blocking that LLVM does not spill. The inference-only
multi-layer network is done (`InferenceNetwork`, 2026-11-09). Report rank 6:
tabular Q-learning and the maze are done (2026-11-09); the DQN can be ported
now that tinymind #258 fixed `getQValue`. Rank 7 (LSTM/GRU/Elman from
`neuralnet.hpp`) waited for Dan's C++ fixes, which landed in tinymind #257
(2026-11-09, `68ddd9c`; the recurrent probe then showed every LSTM/GRU weight
changing). The LSTM and `har_activity` are ported (2026-11-09, see the layout
row); GRU and Elman are next. Found here, verified in the source and by a probe at 02bea40: (R1) `computeGateDeltas()` is never
called (dropped in 681b804), so LSTM/GRU gate weights (input, input bias,
recurrent) never change, only hidden->output trains, and the hidden->output
bias of LSTM/GRU is never trained (`NumberOfBiasNeuronsInLayer = 0`); (R2) the
forward pass writes each neuron's h_t into the recurrent layer inside the
loop, so later neurons mix h_t and h_{t-1}; (R3) the fixed-point GRU's
`ValueType(1)` is raw 1 = 2^-16; (R4) Elman's recurrent gradient uses h_t, not
the h_{t-1} fed in; (R5) LSTM `resetState` clears only the cell state.
Lower priority, from a read only: Adam `step()` 3x per recurrent train step,
`RecurrentConnectionDepth > 1` a no-op, batch bookkeeping ignores recurrent
gradients, `setWeights` copies only H of G*H input connections; #257 fixed
R1-R5 (plus the LSTM gate deltas' raw 1 and the GRU candidate's missing
reset-gate factor) but not these. Write-ups and probes for Dan:
`~/Documents/tinymind/tinymind-rs-findings-2026-11-09/`. Port against the
fixed code: `har_activity` (LSTM, default truncating Q16.16) is done; `gru_xor`,
`lstm_sinusoid` and `elman_temporal_xor` also need `RoundUpPolicy`, which is
not ported (a policy parameter on `QValue` is an API change).

Floating point (2026-11-09): `Scalar` (was `Fixed`) is implemented for `f32`/`f64` (`pow2_neg`
by exact halving, `div_int` = `self / n as $t`, as the C++ `value /= n`), so every
network type is generic over them; `Tanh`/`Sigmoid` for floats come from
`impl_float_activations!` (tanh, `1/(1+exp(-x))`, derivatives `1-y*y`,
`y*(1-y)`, as tinymind's float example and `nn_unit_test.cpp` write them) with
feature `std` (system libm: bit-exact with the C++ on glibc) or the optional
`libm` feature/dependency (`no_std`, not bit-exact; `all(libm, not(std))`, so
`--all-features` uses std). tinymind ships no float transfer-function bundle:
the library templates plus user specializations are its float path, copied
unchanged into `scripts/cpp/float_mlp.cpp` (needs `<new>` and
`fixedPointTransferFunctions.hpp` for `ZeroToleranceCalculator`; built with
`-ffp-contract=off`). The bundle's rates are 0.15/0.5/0.1; the Rust default rates
stay 1/4, 1/2, 1/16 for every type, so tests set them. `tests/float_mlp.rs`
(feature std) matches every step of five configs (f64/f32; 1-2 hidden layers;
tanh/ReLU; sigmoid/linear) on the first run. The f32 runs differ between glibc
2.39 (here) and CI's newer glibc (2.41 made tanhf/expf correctly rounded; f64
matched): so the committed golden holds only the f64 lines, and
compare-with-cpp.sh runs the test with `TINYMIND_FLOAT_MLP_CPP` = that host's C++
output to compare every line live (KEEP_CPP_OUTPUT copies only f64 lines); plants (tanh derivative as
(1-y)(1+y), div_int via reciprocal, sigmoid +eps/4) fail it. panic-never runs
float networks with `libm` (M0+ soft float and M4F). tinymind's
`AdamOptimizerFloat`/`RmsPropOptimizerFloat` are ported as `AdamFloat` /
`RmsPropFloat` (src/optimizer.rs, `impl_float_optimizers!` for f32 and f64;
pub f64 hyperparameters with the C++ values as `Default`; casts exactly as the
C++: `(beta as T) * m`, `((1 - beta2) as T) * g * g`, `m / ((1 - pow(beta, t)) as
T)`, `sqrt` in f64 then cast; pow/sqrt from std or libm via `float_math`). The C++
program attaches them as nn_unit_test.cpp does (bundle + `OptimizerPolicyType`
typedef), needs `-DTINYMIND_ENABLE_FLOAT=1`, and gives those runs learning rate
0.01 (0.15 saturated the f32 tanh run and killed a ReLU one). Four more configs,
all bit-identical live; plants (Q-style t<100 cut-off, beta1*m in f64, epsilon
outside the division, decay swapped) fail. The
value-type trait was renamed `Fixed` -> `Scalar` at Dan's request (2026-11-09):
`Fixed` stays as a `#[deprecated(since = "0.2.0")]` subtrait with a blanket
impl for every `Scalar`, so `V: Fixed` bounds compile with a warning (doctest on
`Fixed`; checked from an outside crate). Implementors of `Fixed` break, so the
next release is 0.2.0, not 0.1.6.

## Docs site (GitHub Pages)

`docs/` is a Jekyll site with the just-the-docs theme, the same setup as tinymind's
site, published at <https://danmcleran.github.io/tinymind-rs/> by
`.github/workflows/pages.yml`. That workflow builds on every PR (deploys only from
`master`) and adds the crate's rustdoc under `/api/`. Its `build site` job is not a
required check.

- Pages: Home, XOR example, Iris example, Energy example, Anomaly detection, Decision trees, Int8 inference, Q-format in Rust, On-board
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
  rewrites bare mentions. CLAUDE.md, code comments, commit messages and PRs are
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
