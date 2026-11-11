---
paths:
  - "src/recurrent.rs"
  - "examples/har_activity.rs"
  - "examples/common/har_driver.rs"
  - "tests/lstm_sweep.rs"
  - "tests/har_golden.rs"
  - "scripts/cpp/lstm_sweep.cpp"
---

# LSTM (`LstmNetwork`) and `har_activity`

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.
The history of the C++ recurrent fixes (R1-R5) and what is next (GRU, Elman)
is in `.claude/notes/roadmap.md`. The int8 cells are in `int8.md`.

## `src/recurrent.rs`, `examples/har_activity.rs`, `examples/common/har_driver.rs`

C++ source: `cpp/neuralnet.hpp` (`LstmNeuralNetwork`, `LstmHiddenLayer`, `TrainableLstmHiddenLayerNeuron`, `ChainBackPropagationThruTimePolicy` and the recurrent chain calculators), `examples/har_activity`

2026-11-09, against tinymind #257 (`TINYMIND_REF` 0425b66). `LstmNetwork<V, H, O, IN, HID, OUT, Opt = Sgd>`, one hidden layer (the C++ multi-layer recurrent chain is not ported; it trains since tinymind #263, so it can be). Stable Rust cannot write `[_; 4 * HID]`, so gate connections are `[[Connection; 4]; HID]` (`GatedNeuron`); nesting order = the C++ flat index `n * 4 + g`, so init draws line up. Gate order candidate (H = TF hidden activation), input, forget, output (Sigmoid), cell output `Tanh(c) * o`; `c = (f * c_prev) + (cand * i)`. Init order: inputs, input bias, cells (OUT each), hidden bias, outputs (1 unused each), recurrent (last). Forward latches h[t-1] into the recurrent layer first (#257's `latchPreviousOutputs`), sums are bias + inputs + recurrent, accumulated source-outer (same per-target order). Train: output deltas, raw dL/dh (`calculateAndSetRawNodeDeltas`), `computeGateDeltas` with each activation's derivative (left-assoc products as written), gradients output/hidden-bias, recurrent (`h[t-1] * gate delta`, LSTM recurrent delta = gate delta), input and input bias; recurrent-layer and input node deltas skipped (nothing reads them). Updates layer by layer (output, recurrent, input) with one optimizer step first (since #264); bias clipped like the weights (since #264); recurrent layer has no bias. `reset_state` zeroes c, h and the latched h. Checked by `tests/lstm_sweep.rs` (golden from `scripts/cpp/lstm_sweep.cpp`, UBSan clean: 7 configs incl. har's 3-16-4, Q8.8, ReLU/linear, asymmetric clip, Adam; sequences 1-12 with resets, every 7th step untrained, outputs + all weights hashed per step), 4 float LSTM runs in `tests/float_mlp.rs` (f64 golden, f32 live), and `tests/har_golden.rs` (stdout + 3 CSVs, byte-identical first run; 31 s debug). Plants that fail the sweep: in-place h write (R2), reset keeping h (R5), hidden bias untrained, forget delta with c[t], dL/dc regrouped, recurrent gradient with h[t] (R4), Adam stepping twice, recurrent sum dropped, cell init reversed. har's C++ is built with `-ffp-contract=off` (double sin/cos data). tinymind #262 (`965f00f`) rewrote har: classes walking/upstairs/downstairs/sitting (stairs are time reversals: sawtooth, only sample order separates them), lr 0.05 with no momentum (`HarNetworkInitializer`), trains every step (the zero-tolerance skip is gone), 60 epochs, nearest-centroid + chance baselines, `--readout-only` (input, input-bias and recurrent weights restored after every step; compared live by compare-with-cpp.sh, no golden). Ported byte-identical first run, 91.5% (readout-only 50%); 92.5% (185/200) since #264's bias-gradient clipping; #267 (SGD step rounded to nearest) raised the learning rate 0.05 -> 0.1 (at 0.05 the unbiased steps left 8 of 10 C++ seeds at chance): 76.5% (153/200, a weak data seed; the C++ README reports 89.6% over 10). C++ `SAWTOOTH_RMS` literal is 0x1.279a74590331cp-1, one ulp below 1/sqrt(3) rounded: copy literals, don't recompute (an ulp there is an equivalent mutant at this output resolution anyway). Host timing (best of 5, at 965f00f): Rust release 4.16 s vs C++ -O3 5.37 s. Kani `lstm_never_panics` (Q8.8 1-1-1, unwind 6, ~80 s; a reachable planted assert fails it; an unreachable one, `h != 1`, passed, so plants must be reachable). panic-never covers Q16.16 3-16-4, Q8.8 ReLU/linear, Adam, f32, f64 AdamFloat. Not ported: GRU, Elman, inference-only LSTM, `setWeights`
