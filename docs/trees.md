---
title: Decision trees
layout: default
nav_order: 6
---

# Decision trees and gradient boosting

On tabular sensor data, a few shallow trees often do as well as a neural network at a fraction of the cost. A tree walk is a handful of integer compares and branches: no multiply-accumulates, no activation tables. The `tinymind::tree` module is tinymind's `cpp/qtree.hpp`: a quantized decision tree (`QDecisionTree`) and a gradient-boosted ensemble (`QGbdt`) whose trees share one node pool.

## How it works

- **Nodes.** A model is a flat array of `QTreeNode`s: a split names a feature, a threshold and the indices of its two children; a leaf holds an `i32` value. The arrays are borrowed, so a model can be `static` data in flash.
- **Thresholds on the feature grid.** Each threshold is quantized onto the same integer grid as its feature (`int8`, `uint8` or `int16`). An affine map is monotone, so `x_q <= t_q` makes the same decision as `x <= t` in the real domain, except for an input within one grid step of the threshold.
- **Ensembles.** Each tree adds its leaf value to one class's score, starting from optional base scores; the prediction is the class with the largest score, the lowest on a tie. One class makes a regression.
- **Malformed models.** Where the C++ would read out of bounds (a child, root or feature index outside the model), a walk here stops and contributes 0, so every function is total. `is_valid()` is a `const fn`, so a model in a `const` can be checked at compile time. The [API reference]({{ site.baseurl }}/api/tinymind/tree/) has a complete example.

## The example

`gbdt_tabular_int8` is tinymind's example of the same name. Five trees over two features score three classes: three trees split feature 0 at -0.15 and 0.25 into three bands, and two trees split feature 1 at -0.10 and 0.10. The thresholds are written as floats, as a trainer exports them, and quantized onto an int8 grid over [-1, 1]. The program runs a 48 x 48 grid of inputs through both the float trees and the int8 ensemble.

```bash
cargo run --release --features std --example gbdt_tabular_int8 -- output   # also writes output/gbdt_tabular_int8.csv
```

![The int8 ensemble's predicted class over the feature grid]({{ site.baseurl }}/assets/plots/gbdt_regions.png)

The int8 ensemble agrees with the float trees on 2,256 of the 2,304 grid points (97.92%). The 48 that differ are one column of the grid, at feature 0 = -0.149, which is within one int8 step of the -0.15 threshold: on the grid it rounds onto the threshold and goes left. The feature-1 trees add +5 or -3 to a score, never enough to change the winner here, so the regions are three vertical bands.

## How it is checked

- The example's three outputs (the report, the CSV and the compact golden report) are byte-identical to the C++ example's.
- A C++ program on tinymind's `QDecisionTree` and `QGBDT` (`scripts/cpp/tree_sweep.cpp`) runs 72 random ensembles: int8, uint8 and int16 thresholds, eight trees of depth up to 5 sharing a 512-node pool, with and without base scores, small leaf values so that classes tie, and inputs on or next to the thresholds. The Rust trees produce the same scores and predictions for every input.
- Kani proves that the walk, the ensemble and the validity checks never panic, for any node pool (indices in or out of range, cycles), root, class, base score and input. The Cortex-M panic gate links them too.

## Importing a trained model

`tools/import_trees` turns a scikit-learn, XGBoost or LightGBM model (single trees, random forests, extra trees, gradient boosting) into a Rust module: the feature grids, the integer trees, a `quantize_features` function for float samples, and `forward` and `predict`. It is a port of tinymind's tree importer; reading the model, placing the thresholds on the grid, scaling the leaves and measuring the base score are tinymind's code, unchanged, so both importers produce the same integer trees.

Each module has two forms of the same trees, with the same results: `forward`/`predict` walk a node pool with `tinymind::tree` (smaller), and `forward_inline`/`predict_inline` run the trees compiled to nested `if`s (faster, larger code). Usage and the conversion rules are in the [importer's README](https://github.com/danmcleran/tinymind-rs/tree/master/tools/import_trees).

The `imported_trees` example runs two scikit-learn classifiers for the UCI Wine data (13 chemical measurements, three cultivars), trained on 70% of it, on the other 54 samples:

```bash
cargo run --release --example imported_trees
```

| Model | Inputs | Trees | Correct on the 54 held-out samples |
|---|---|---|---|
| Gradient boosting, 30 rounds of depth-2 trees | int8 | 90 (one per class per round) | 53 (scikit-learn: 53) |
| Random forest, 15 voting trees of depth 4 | int16 | 15 | 52 (scikit-learn: 53) |

The same two models imported with tinymind's importer and run on tinymind's C++ runtime print byte-identical output, and so do 16 more models across every supported family when checked against the importers' shared integer reference.

## Not ported

tinymind's compile-time models (`QStaticDecisionTree`, `QStaticGBDT`) and their packed `QTreeCompactEval` form. The importer's two forms cover the walk and inline policies.
