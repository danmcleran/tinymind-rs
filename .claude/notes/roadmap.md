# Roadmap and history

Moved out of CLAUDE.md (2026-11-11). Not loaded automatically: read it when choosing
what to port next, and update it when a step is done.

## Next steps

- **FFT speed (done, 2026-11-12):** C++ timed on both boards (a scratch
  backend in tinymind's harness; gcc's -O3 code depends on the caller, best
  found): gcc -O3 F411 ~82,900, L031 ~101,600; -Os ~85,200 / 142,482. Rust
  (twiddle by twiddle, constant stage sizes, saturation skip on ARMv6-M):
  opt 3 41,991 / 91,430, opt s 44,255 / 96,784. **Open:** a bit-reversal table (~5K of the L031's ~9.3K reversal cycles, 512 B
  at N=256). Notes in `.claude/rules/int8.md`.

- **Training controls (done, 2026-11-14):** `tinymind::training`: L2 weight
  decay (`WeightDecay`) and step-decay schedule (`Scheduled` + `StepDecay`) as
  optimizer wrappers, and `EarlyStopping` (C++ `weightDecay.hpp`,
  `learningRateSchedule.hpp`, `earlyStopping.hpp`). Chosen as the highest-ROI
  next port (2026-11-14): small, strengthens the on-device training story. It
  unblocks the early-stopping half of `air_quality` and `lstm_sinusoid`
  (both also need `teacherForcing.hpp`; `lstm_sinusoid` also `RoundUpPolicy`).
  Other candidates from that review, in order: `predictive_maintenance` and
  `optical_digits` (plain Q16.16 MLPs, API already there), the PyTorch
  importer emitting the int8 LSTM, `QTreeCompactEval`, trainable GRU/Elman.

- **Head-only fine-tuning (done, 2026-11-13):** tinymind #277 added
  `DenseHead`/`PrototypeHead` and the gas_sensor_drift adaptation; ported as
  `tinymind::head` and `examples/gas_sensor_drift.rs` (notes in
  `.claude/rules/core.md`). This was report priority 2's last item. Timed on
  both boards with the C++ alongside (2026-11-13, workload `head`): level on
  the L031, 37% behind on the F411 step. The cheaper `mul_round_nearest` plus
  a never-inlined row update (2026-11-13) took the F411 step to 9,371 (27%
  behind) and the L031 to 51,252 (2% ahead); with the M4 machine scheduler off
  the F411 step is 8,001 (8% behind), so the rest is mostly
  llvm/llvm-project#241851. The MLP (xor) clips in a pass before the update
  since 2026-11-13 (no unswitched copy); `optimizer::update_all` (anomaly)
  still has two copies (+340 B on the L031): every fix tried cost cycles or
  bytes (`.claude/rules/bench.md`).

Next steps follow `reports/Embedded Rust ML priorities.md` (local, not committed):
Dan chose to port what matters most to embedded Rust developers, not the smallest
piece. Done: int8 dense + softmax, the PyTorch importer, multiple hidden layers
(2A), linear output + gradient clipping (2B, `examples/energy_efficiency`),
Adam/RMSprop (2C), the on-device anomaly-detection demo on both boards (2D).
Priority 2 is complete, head-only fine-tuning included (2026-11-13). Trees: the runtime (`src/tree.rs`, 3A) and
the importer (`tools/import_trees`, 3B) are done; tinymind's packed
`QTreeCompactEval` form is not. Int8 conv (priority 4): the layers are done
(4A, 2026-11-08: output sizes spelled out by the caller and checked at compile
time). 4B, `examples/kws_cortex_m_int8`, is done (every tensor and the logits
report, including tinymind #255's per-layer calibration). 4C, the
framework-comparison KWS CNN on both boards, is done and tuned (13-27% fewer
cycles than the C++ scalar build). 4D, convolutions in the PyTorch importer, is
done, including BatchNorm folding (tinymind #256 ported). The TFLite translator (4E) runs the MLPerf DS-CNN with tinymind's hash on the
F411, and ResNet-8 too (`QAdd` ported, ADD in the translator), with RAM and cycles tuned (arena, 16-bit band path, ADD tables). Possible next: the same 16-bit pairing for pointwise/depthwise/dense (the DS-CNN's pointwise layers), and 2x2 register blocking that LLVM does not spill. The int8 LSTM and GRU cells (`src/int8/recurrent.rs`) are done (2026-11-10); next for them: emitting them from the PyTorch importer (`nn.LSTM`/`nn.GRU`). The inference-only
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
reset-gate factor), and #263/#264 (2026-11-10) fixed these too: stacked
recurrent layers now train (single-layer outputs bit-identical, so porting the
multi-layer recurrent chain is now possible), `RecurrentConnectionDepth != 1`
is a static_assert, BatchSize > 1 averages per batch, setWeights copies every
weight, and the optimizer steps once per training step. Write-ups and probes for Dan:
`~/Documents/tinymind/tinymind-rs-findings-2026-11-09/`. Port against the
fixed code: `har_activity` (LSTM, default truncating Q16.16) is done; `gru_xor`,
`lstm_sinusoid` and `elman_temporal_xor` also need `RoundUpPolicy`, which is
not ported (a policy parameter on `QValue` is an API change).

### Performance without `unsafe` (2026-11-11)

Dan ruled out `unsafe` in any form, including an opt-in feature (see
CLAUDE.md). The `asm!` SMLAD experiment (F411: MLP 15,419, KWS 176,194,
DS-CNN 9.93M, ResNet-8 38.9M cycles) is therefore closed; Dan kept it for
reference as the branch `archive/exp-smlad-asm` on GitHub (never to be
merged). Its numbers only show the gap to the C++ +SMLAD build. Safe routes
left, roughly by expected gain:

- 16-bit pairs that LLVM's ARMParallelDSP pass turns into `SMLAD` (already the
  conv band path: ResNet-8 with `-arm-parallel-dsp-load-limit=512`), extended to
  dense and pointwise layers (the int8 MLP is 20,437 vs C++ +SMLAD 10,241).
- 2x2 register blocking where LLVM does not spill (pointwise done: #91).
- The two smaller regressions from `target-cpu=cortex-m4` (#89): KWS pointwise
  (C = 8) +15% for that layer, the 36-tree subset +7%. Not investigated.
- The small-Q16.16-layer spills (anomaly +37% under cortex-m4) are LLVM's
  CortexM4Model scheduler; no safe code shape fixed them (`.claude/rules/bench.md`).
  Filed upstream as llvm/llvm-project#241851 (2026-11-12); watch it, and
  re-measure the anomaly workload when a fix reaches rustc's LLVM.
- **Toolchain for board timings:** every table was re-measured on 2026-11-14
  with rustc 1.100.0 (LLVM 23.1.5), the stable CI runs. Most figures moved by
  0-0.3%; the largest moves were L031 KWS at opt s +0.9% and its max pool
  -4.9%. When stable moves again, update and re-measure every workload, not
  just the changed one, so a table never mixes compilers (raw notes in
  `.claude/rules/bench.md`).

## Release history (as CLAUDE.md recorded it; CHANGELOG.md is the full record)

Releases, each with a tag and a GitHub release: 0.1.0 (2026-11-07, `601098f`),
0.1.1 (2026-11-07, `ab43794`: int8 softmax), 0.1.2 (2026-11-07, `7dd0423`:
`InferencePerceptron`), 0.1.3 (2026-11-07, tag `v0.1.3`: `NeuralNetwork` with any
number of hidden layers, `Linear`, gradient clipping, `energy_efficiency`), 0.1.4
(2026-11-07, tag `v0.1.4`: Adam/RMSprop, `Sqrt`, the `anomaly` example, panic fixes), 0.1.5
(2026-11-08, tag `v0.1.5`: `tinymind::tree`, the tree importer, `gbdt_tabular_int8` and
`imported_trees`) and 0.2.0 (2026-11-09, tag `v0.2.0`: int8 conv/pool, `QAdd`,
`QMul`, `QPad2D`, `QConcat`, `InferenceNetwork`, `f32`/`f64` networks with
`AdamFloat`/`RmsPropFloat`, the optional `libm` feature, and the breaking rename
`Fixed` -> `Scalar` with `Fixed` kept as a deprecated alias) and 0.3.0 (2026-11-10,
tag `v0.3.0`: tabular Q-learning and the maze, the LSTM and `har_activity`,
the syncs to tinymind #261-#267: optimizers train biases with one step per
update, SGD momentum rounds toward zero and its step to nearest; `Default`
Adam/RmsProp constants, `mul_round_toward_zero`, `mul_round_nearest`; minor bump because results change and `Scalar` gained
a method), 0.3.1 (2026-11-11, tag `v0.3.1`: the int8 LSTM and GRU cells,
`int8::QLstmCell` and `int8::QGruCell`, with their calibration), and 0.3.2
(2026-11-12, tag `v0.3.2`: the quantized FFT, `int8::QFft1D` and
`calibration::build_fft_twiddles`; 2x2 pointwise pairs; QEMU runs in CI), and 0.3.3
(2026-11-12, tag `v0.3.3`: the deep Q-network `qlearn::QNetwork` and
`dqn_maze`; QDense one-rounding loop; FFT twiddle-outer, constant stage sizes
and the ARMv6-M saturation skip; TINYMIND_REF v18.2), and 0.3.4 (2026-11-13, tag `v0.3.4`:
head-only fine-tuning, `tinymind::head` with `DenseHead` and `PrototypeHead`,
`MultilayerPerceptron::hidden_values` and `gas_sensor_drift`; heads timed on
both boards; TINYMIND_REF 3d89270).
