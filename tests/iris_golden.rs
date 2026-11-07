// Golden test for the iris example.
//
// The goldens in tests/golden/iris/ come from tinymind's C++ `examples/iris`,
// built with its Makefile's release flags (see `scripts/compare-with-cpp.sh`,
// which rebuilds it and diffs everything). The input is data/iris.data, the
// file tinymind ships with the example.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/iris_driver.rs"]
mod iris_driver;

use std::fs;

use common::assert_same;

#[test]
fn iris_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-iris-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let root = env!("CARGO_MANIFEST_DIR");
    let mut stdout = Vec::new();
    let data = std::path::PathBuf::from(format!("{root}/data/iris.data"));
    iris_driver::run(&dir, &data, &mut stdout).expect("iris example failed");

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/iris/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in ["iris_loss.csv", "iris_scatter.csv", "iris_confusion.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
