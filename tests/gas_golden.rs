// Golden test for the gas_sensor_drift example.
//
// The goldens in tests/golden/gas/ come from tinymind's C++
// `examples/gas_sensor_drift`, built with its Makefile's flags plus
// -ffp-contract=off (see `scripts/compare-with-cpp.sh`, which rebuilds it and
// diffs everything): the example synthesizes its data in double arithmetic,
// and a compiler that fuses multiply-adds would change it.

mod common;

#[path = "../examples/common/gas_driver.rs"]
mod gas_driver;

use std::fs;

use common::assert_same;

#[test]
fn gas_sensor_drift_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-gas-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let root = env!("CARGO_MANIFEST_DIR");
    let mut stdout = Vec::new();
    gas_driver::run(&dir, &mut stdout).expect("gas_sensor_drift failed");

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/gas/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in [
        "gas_loss.csv",
        "gas_confusion.csv",
        "gas_drift.csv",
        "gas_adapt.csv",
    ] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
