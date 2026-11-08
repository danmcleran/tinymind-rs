// Golden test for the int8 GBDT example.
//
// The goldens in tests/golden/gbdt/ come from tinymind's C++
// `examples/gbdt_tabular_int8`, built with its Makefile's release flags (see
// `scripts/compare-with-cpp.sh`, which rebuilds it and diffs every file).

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/gbdt_driver.rs"]
mod gbdt_driver;

use std::fs;

use common::assert_same;

#[test]
fn gbdt_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-gbdt-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = env!("CARGO_MANIFEST_DIR");
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/gbdt/{f}")).unwrap();

    let mut stdout = Vec::new();
    assert!(gbdt_driver::run(&dir, false, &mut stdout).expect("gbdt example failed"));
    assert_same("stdout", &stdout, &golden("stdout.txt"));
    let csv = fs::read(dir.join("gbdt_tabular_int8.csv")).unwrap();
    assert_same(
        "gbdt_tabular_int8.csv",
        &csv,
        &golden("gbdt_tabular_int8.csv"),
    );

    let mut report = Vec::new();
    assert!(gbdt_driver::run(&dir, true, &mut report).expect("gbdt example failed"));
    assert_same(
        "golden report",
        &report,
        &golden("gbdt_tabular_int8.golden"),
    );
    let _ = fs::remove_dir_all(&dir);
}
