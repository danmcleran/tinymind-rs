---
paths:
  - "tools/**"
  - "examples/imported_*.rs"
  - "examples/common/imported_*.rs"
  - "tests/imported_*.rs"
  - "scripts/cpp/imported_*.cpp"
  - "bench/src/mlperf.rs"
---

# Importers (PyTorch, TFLite, trees) and their examples

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.

## `tools/import_pytorch/tinymind_import.py`

C++ source: `apps/import_pytorch/tinymind_import.py`

PyTorch importer emitting Rust. Calibration/quantization/observers copied unchanged (Dense/ReLU/Sigmoid/Tanh/Softmax subset); new: integer requantizers (`buildRequantizer` on float32 scales), LUTs via libm `expf`/`tanhf` over ctypes, softmax params (`buildQSoftmaxFixedPointParams` on the float32 scale; tinymind's importer leaves softmax to the hand-written C++), `forward`, `from_torch`, rustfmt'd output. `test_parity.py` vs tinymind's module; `example_mlp.py` (no torch) and `torch_xor.py` (torch)

## `tools/import_pytorch` conv support (4D), `examples/imported_cnn.rs`, `examples/common/imported_cnn{,_flat}_model.rs`, `examples/common/imported_cnn_driver.rs`, `scripts/cpp/imported_cnn.cpp`, `tools/import_pytorch/{example_cnn,torch_cnn}.py`

C++ source: (tinymind importer's Conv2D descriptor + C++ runtime)

`Conv2D` gains `stride`/`padding` (t,b,l,r)/`depthwise` (tinymind's leaves layer types to hand-written C++); new `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`, `Flatten` (pass-through: range = input's). `_plan` follows ("vec", n)/("map", h, w, c); conv emits `QConv2D`/`QDepthwiseConv2D` (per-channel rq array, all equal)/`QPointwiseConv2D` (1x1, stride 1, no pad) with the output size spelled out; a ReLU after a conv is folded into the requantizer's qmin (= qrelu after it). `from_torch(model, input_shape=(C,H,W))`: OIHW->OHWI, NCHW->NHWC, Linear-after-Flatten columns reordered, "same" padding; `torch_cnn.py` checks it against PyTorch (needs torch, local only; four planted layout bugs fail it). Parity: tinymind's importer has no pools/flatten, so `example_cnn.py --cpp` folds each into the next layer's forward (same ranges). The C++ check prints a hash of every intermediate tensor (the models' outputs vary little). A ReLU's range starts at 0, so its zero point is -128 unless a KL observer gives a symmetric range: `cnn`'s relu3 uses KL so the qmin fold is exercised. The float references (`conv2d_nhwc`, `dense_ref`, the average pools) accumulate in float64 without BLAS (`einsum(optimize=False)`) and round to float32: with float32 einsum/`@` the calibrated ranges depended on the CI runner's CPU (BLAS summation order), moved by an ulp, changed requantizer multipliers, and the regenerated `cnn` module differed on CI (2026-11-09; #50 had passed on another runner). `BatchNorm2D` after a `Conv2D` is folded by tinymind's `fold_conv_bn`/`fuse_layers` (copied, with the fused layer also keeping stride/padding/depthwise), including tinymind #256 (2026-11-09, `TINYMIND_REF` 02bea40): before it the fused layer kept `forward=conv.forward` and calibration fitted the pre-BN range (found here, reported). `import_pytorch_model` and `run_float` fuse first; `cnn` folds two BNs (one depthwise); `torch_cnn.py` also runs each conv from its folded weights.

## `examples/imported_mlp.rs`, `examples/common/imported_{mlp,classifier}_model.rs`, `examples/common/imported_mlp_driver.rs`, `scripts/cpp/imported_mlp.cpp`

C++ source: (tinymind importer + C++ runtime)

The importer's two example models (sigmoid MLP, softmax classifier); the C++ program runs tinymind's headers for the same models, with `QSoftmaxFixedPoint1D`. The `*_model.rs` files are generated: regenerate with `example_mlp.py [--model classifier] --rust`, never edit

## `tools/import_tflite/tinymind_import_tflite.py`, `mlperf.py`, `test_translator.py`, `bench/src/mlperf.rs`

C++ source: `benchmarks/framework_comparison_f411/gen/make_mlperf.py`

int8 .tflite -> Rust module (4E). tinymind's helpers copied unchanged (`quantize_multiplier` with Python `round`, `act_range`, `rq`, `same_pads`, softmax params); reads the flatbuffer with the `tflite` package (pinned 2.18.0 + flatbuffers 25.12.19; no TensorFlow). Ops: CONV_2D, DEPTHWISE_CONV_2D, global AVERAGE_POOL_2D, per-tensor FULLY_CONNECTED, one-row SOFTMAX, RESHAPE, ADD (same shapes; `add_params` copied from make_mlperf.py's add(), TFLM's float32 `twice_max`, emitted by `add_literal`). The graph may branch: steps record input/output tensor indices (RESHAPE without a copy is an alias), and `forward` keeps feature maps in buffers per shape, a buffer reused once no later step reads its tensor (liveness), so a chain ping-pongs between two and a residual block keeps its skip tensor: nesting each layer in a block peaked at 3 tensors (the returned value got its own slot), 24.5 KB vs 16.3 KB for the DS-CNN. Since 2026-11-09 the maps share one byte arena planned first-fit by liveness (as make_mlperf.py's planner; ResNet-8 49,152 B = the C++'s); each step does `split_at_mut` around its output and views slots with generated `view`/`view_mut` (`as_chunks` + `first_chunk`, safe; the generated module needs Rust 1.88, the crate stays 1.85). Per-shape typed buffers had needed 86.5 KB. `mlperf.py kws` fetches the pinned DS-CNN (sha256; cached in target/tflite-models; Apache-2.0, never committed), runs make_mlperf.py's seeded inputs through cargo and compares with `TM_MLPERF_KWS_HASH` (0xF623A166) from tinymind's `fw/backend.hpp`; `--out/--data` write the bench files. `test_ops.py` (needs ai-edge-litert) checks PAD/MUL/CONCATENATION against TFLite's reference kernels on generated models; `mlperf.run_module(output_map=True)` flattens map outputs. `mlperf.py ic` does the same for ResNet-8 (`TM_MLPERF_IC_HASH` 0x8897F6ED; matched on the first run). Four planted translator bugs change the DS-CNN hash; on ResNet-8, ADD with b's multiplier for a, a's zero point for b, left shift 19, and the old ping-pong buffer rule (overwrites the skip tensor) each change it. Fused ReLU (also on ADD), the softmax radius and nudges to `twice_max` do not show on these models (every activation zero point is -128), so `test_translator.py` checks them (softmax params vs the 504-calibration golden, `add_params` vs the 48 calibrated `buildQAddParams` lines of tests/golden/add: the scaling by 2 and 2^20 is exact, so TFLM's float32 and the C++'s double agree). F411 DS-CNN: 13,225,965 cycles (C++ scalar 15,736,026, +SMLAD 8,193,064, TFLM CMSIS-NN 9,583,099), 46,416 B flash, 16,680 B RAM. F411 ResNet-8 (`mlperf-ic`): 64,407,536 cycles (C++ scalar 85,467,754, +SMLAD 26,753,191, TFLM CMSIS-NN 37,372,049), 130,184 B flash, 57,692 B RAM; with `-arm-parallel-dsp-load-limit=512` 41,401,469 cycles. Before the band path, ADD tables and arena: 91,085,417 / 101,076 B / 86,520 B. Per layer (default, before ADD tables): conv1/2 11.1M each, conv5 13.3M, conv9 11.4M, conv4 5.7M, conv8 6.6M, conv0 2.7M, adds 3.0/1.5/0.76M

## `tools/import_trees/tinymind_import_trees.py`

C++ source: `apps/import_trees/tinymind_import_trees.py`

Tree importer emitting Rust. Everything up to `export_trees` (FeatureQuant, readers, plan, threshold/leaf/base rules) copied unchanged from tinymind (only `namespace` -> `name`, docstrings); new: `_render` (Rust: `quantize_features`, pool form via `QGbdt` or one `QDecisionTree` per voting tree, nested-`if` `tree_k` fns, `const _: () = assert!(..is_valid())`), rustfmt, `verify` via cargo (temp crate, path dep, optional shared `target_dir`). `test_parity.py`: 18 models, integer model identical to tinymind's + tinymind's g++ verify + our cargo verify. `FeatureQuant.from_data` used to clip the zero point to `[qmin, qmax]` (found here 2026-11-08: features far from 0 saturated; iris sepal length all at 127); tinymind #254 stopped clipping it (bounded by `_ZP_LIMIT` so `quantize_features`' `i32` sum cannot overflow, else the feature's range is widened to include 0), ported here with the example modules regenerated; `test_parity.py` has an offset case and a far-offset (fallback) case

## `examples/imported_trees.rs`, `examples/common/imported_trees_{driver,gbdt_model,forest_model,data}.rs`, `scripts/cpp/imported_trees.cpp`

C++ source: (tinymind tree importer + C++ runtime)

Wine (sklearn `load_wine`, CC BY 4.0, `NOTICE`), 70/30 split: GradientBoosting (int8) and RandomForest (int16). `*_model.rs` and `imported_trees_data.rs` are generated by `tools/import_trees/example_trees.py`, never edit; CI retrains with pinned numpy/scipy/scikit-learn and checks they are current
