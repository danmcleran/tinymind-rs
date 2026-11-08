# Tree importer

Turns a trained scikit-learn, XGBoost or LightGBM tree model into a Rust module
for the `tinymind` crate: integer feature grids, integer split thresholds and
leaves, and `forward` / `predict` functions over `tinymind::tree`. The
generated module is integer-only and `no_std`.

It is a port of tinymind (C++)'s [`apps/import_trees`](https://github.com/danmcleran/tinymind/tree/master/apps/import_trees).
Reading the model, quantizing the features and thresholds, scaling the leaves
and measuring the base score are tinymind (C++)'s code, unchanged, so a model
imported here has the same integer trees as the same model imported with
tinymind (C++); only the output is Rust instead of a C++ header.

## Supported models

| Library | Models |
|---|---|
| scikit-learn | `DecisionTree{Classifier,Regressor}`, `RandomForest{Classifier,Regressor}`, `ExtraTrees{Classifier,Regressor}`, `GradientBoosting{Classifier,Regressor}` |
| XGBoost | `XGBClassifier`, `XGBRegressor`, `Booster` (gbtree) |
| LightGBM | `LGBMClassifier`, `LGBMRegressor`, `Booster` |

Not supported, and rejected with a `ValueError`: missing values (NaN),
categorical splits, multi-output trees, XGBoost dart / gblinear, LightGBM
linear trees and rf mode.

## Use

Needs Python 3.10+, numpy, and the library the model comes from.

```python
import sys; sys.path.insert(0, "tools/import_trees")
from tinymind_import_trees import FeatureQuant, export_trees

clf.fit(X_train, y_train)
model = export_trees(clf, X_train, "my_model.rs", "my_model",
                     quant=FeatureQuant.from_data(X_train, "int8"))
print(model.report)        # trees, nodes, agreement with the original
model.verify(X_test)       # cargo-builds the module and checks it bit for bit
print(model.agreement(X_test))
```

Then in the Rust crate:

```rust
mod my_model;

let q = my_model::quantize_features(&x);   // [f32; NUM_FEATURES] -> the integer grid
let class = my_model::predict(&q);         // or predict_inline(&q)
let raw = my_model::forward(&q);           // votes, raw scores or the regression sum
```

Every module has two forms of the same trees, with the same results:

| Form | What it is | Trade-off |
|---|---|---|
| `forward`, `predict` | The node pool walked by `tinymind::tree` (`QGbdt`, or one `QDecisionTree` per voting tree) | Smaller: 12 bytes per node and one shared loop |
| `forward_inline`, `predict_inline` | Every tree compiled to nested `if`s (tinymind (C++)'s `QTreeInlineEval`) | Faster: compares against constants; code grows with the node count |

What `forward` returns depends on the model:

| Model | `forward` | `predict` |
|---|---|---|
| Classifier forest or tree (hard vote) | votes per class | most votes, ties to the lowest class |
| Boosted classifier, soft-vote forest | `i32` raw score per class | largest score (multiclass) or score > 0 (binary) |
| Regressor | `i32` raw value; `raw / OUTPUT_SCALE` is the prediction | - |

`CLASS_LABELS` maps a predicted index back to the original label when the
labels are integers.

## How the integer model is built

These are tinymind (C++)'s rules (see its README for the details):

- **Features** go onto a per-feature integer grid, `q = round(x / scale) + zero_point`
  (ties away from zero, clamped), in `int8`, `uint8` or `int16`.
  `FeatureQuant.integer(n)` is for features that already are integers (ADC
  counts, pixels, bins): scale 1, and every split is reproduced exactly.
  `FeatureQuant.from_data(X, dtype)` calibrates each feature's min/max onto
  the whole grid; a range that excludes 0 gets a zero point outside the
  storage range (it is `i32` in the module). Only a feature so far from 0 that
  `quantize_features`' `i32` sum could overflow is widened to include 0.
- **Thresholds.** A split `x <= t` (scikit-learn, LightGBM) becomes
  `q <= floor(t/scale + zp)`, and XGBoost's `x < t` becomes
  `q <= ceil(t/scale + zp) - 1`. The one grid cell that straddles a threshold
  follows the majority of the calibration samples in it. A split that sends
  the whole grid one way is removed.
- **Leaves.** Classifier forests vote by default (one class per tree);
  `voting="soft"` reproduces scikit-learn's probability averaging, at one tree
  copy per class. Boosted models and regressors add `i32` leaves scaled by a
  power of two chosen so that no sum can overflow.
- **Base score.** Measured as the model's raw output minus the summed tree
  outputs on the calibration data; it must come out the same for every sample,
  which also checks the tree parsing against the library.

On calibrated float features, a sample within one grid step of a threshold can
flip; `model.agreement(X)` reports it on whatever data you give it, and should
be checked on held-out data.

## Example

`example_trees.py` trains two classifiers on the UCI Wine data (scikit-learn
ships it) and writes the modules the `imported_trees` example runs:

```bash
python3 -I tools/import_trees/example_trees.py --model gbdt --rust examples/common/imported_trees_gbdt_model.rs
cargo run --release --example imported_trees
```

## How it is checked

- `test_parity.py`: 18 models (every supported family, and features far from
  zero; int8, uint8, int16 and integer grids) exported by both importers. Every integer must be identical:
  nodes, thresholds, leaves, tree classes, base scores, output scale, grids.
  Then tinymind (C++)'s header is compiled with g++ and run under its three
  evaluation policies, and the Rust module is built with cargo and run in both
  forms; each must equal the shared integer reference on every sample,
  including `quantize_features`. Planted defects in the Rust generator (`<` for
  `<=`, swapped children, a rounding change) each fail it.
- `scripts/compare-with-cpp.sh` imports the example's two models with both
  importers, runs tinymind (C++)'s header on its C++ runtime and the Rust module on
  this crate, and byte-compares the outputs; it also regenerates the committed
  modules and checks they are current. CI runs it with the library versions
  pinned.
- Both of the example's generated modules (the gradient-boosted pool and the
  voting forest) are in the Cortex-M panic gate: no reachable panics.
