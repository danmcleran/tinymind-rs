// Golden test for the optical_digits example.
//
// The goldens in tests/golden/digits/ come from tinymind's C++
// `examples/optical_digits`, built with its Makefile's release flags (see
// `scripts/compare-with-cpp.sh`, which rebuilds it and diffs everything). The
// inputs are data/optdigits.tra and data/optdigits.tes, the files tinymind ships
// with the example.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/digits_driver.rs"]
mod digits_driver;

use std::fs;
use std::path::Path;

use common::assert_same;

#[test]
fn digits_matches_cpp_golden() {
    let root = env!("CARGO_MANIFEST_DIR");
    let data = Path::new(root).join("data");
    // The data files are left out of the published crate (Cargo.toml
    // `exclude`), so a test run from the crate's tarball has nothing to read.
    // A git checkout must have them.
    if !data.join("optdigits.tra").exists() {
        assert!(
            !Path::new(root).join(".git").exists(),
            "data/optdigits.tra is missing from the checkout"
        );
        eprintln!("skipping: data/optdigits.* not in this copy of the crate");
        return;
    }

    let dir = std::env::temp_dir().join(format!("tinymind-digits-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let mut stdout = Vec::new();
    digits_driver::run(&dir, &data, &mut stdout).expect("optical_digits example failed");

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/digits/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in [
        "digits_loss.csv",
        "digits_confusion.csv",
        "digits_samples.csv",
    ] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
