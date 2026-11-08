// Golden test for the imported-trees example.
//
// tests/golden/imported_trees/stdout.txt is the output of
// scripts/cpp/imported_trees.cpp: the same scikit-learn models imported by
// tinymind's own importer and run on tinymind's C++ runtime.
// scripts/compare-with-cpp.sh regenerates the models with both importers,
// checks the committed Rust modules are current and diffs both outputs.

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/imported_trees_driver.rs"]
mod imported_trees_driver;

mod common;

use common::assert_same;

#[test]
fn imported_trees_match_cpp_golden() {
    let mut out = Vec::new();
    imported_trees_driver::run(&mut out).expect("imported_trees example failed");
    let golden = include_bytes!("golden/imported_trees/stdout.txt");
    assert_same("stdout", &out, golden);
}
