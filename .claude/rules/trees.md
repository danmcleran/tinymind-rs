---
paths:
  - "src/tree.rs"
  - "examples/gbdt_tabular_int8.rs"
  - "examples/common/gbdt_driver.rs"
  - "tests/tree_sweep.rs"
  - "tests/gbdt_golden.rs"
  - "scripts/cpp/tree_sweep.cpp"
---

# Decision trees (`src/tree.rs`)

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.
The tree importer and `imported_trees` are in `importers.md`.

## `src/tree.rs`

C++ source: `cpp/qtree.hpp` (`QTreeNode`, `walkQTree`, `QDecisionTree`, `QGBDT`)

Runtime trees over a borrowed pool (`&[QTreeNode<T>; N]`), generic over threshold and input types (`Copy + Into<i32>`, compared in `i32` as the C++ casts). `QGBDT` -> `QGbdt` (Rust acronym style). Out-of-contract indices (child/root/feature/class), UB in the C++, are total here: the walk stops with 0, a bad class is skipped; `is_valid` is the `const fn` form of `qtreePoolIsValid` plus root/class checks. Sums wrap. `predict` carries the running best by value (indexing `scores[best]` kept a bounds check at opt s/z, found by the panic gate). Not ported: `QStatic*` and the Inline/Compact/Walk policies (C++ template expansion; Rust equivalent is importer codegen). Checked by `tests/tree_sweep.rs` (golden from `scripts/cpp/tree_sweep.cpp`: 72 random ensembles, i8/u8/i16, base scores, ties, inputs at thresholds); planted defects (`<` for `<=`, ties to the highest class, base score 1) each fail it

## `examples/gbdt_tabular_int8.rs`, `examples/common/gbdt_driver.rs`

C++ source: `examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp`

Feature `std` (`affine_params_asymmetric`, `quantize`). stdout, CSV and `--golden` report byte-identical. The C++ README says feature 1 "nudges the boundaries"; it never changes the winner (bands are pure vertical; 48 cells flip at f0 = -0.149)
