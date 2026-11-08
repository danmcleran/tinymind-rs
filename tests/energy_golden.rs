// Golden test for the energy_efficiency example.
//
// The goldens in tests/golden/energy/ come from tinymind's C++
// `examples/energy_efficiency`, built with its Makefile's release flags (see
// `scripts/compare-with-cpp.sh`, which rebuilds it and diffs everything). The
// input is data/ENB2012_data.csv, the file tinymind ships with the example.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/energy_driver.rs"]
mod energy_driver;

use std::fs;

use common::assert_same;

#[test]
fn energy_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-energy-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let root = env!("CARGO_MANIFEST_DIR");
    let mut stdout = Vec::new();
    let data = std::path::PathBuf::from(format!("{root}/data/ENB2012_data.csv"));
    energy_driver::run(&dir, &data, &mut stdout).expect("energy_efficiency example failed");

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/energy/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in ["energy_loss.csv", "energy_pred.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
