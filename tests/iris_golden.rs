// Golden test for the iris example.
//
// The goldens in tests/golden/iris/ come from tinymind's C++ `examples/iris`,
// built with its Makefile's release flags (see `scripts/compare-with-cpp.sh`,
// which rebuilds it and diffs everything). The input is data/iris.data, the
// file tinymind ships with the example.

mod common;

use std::fs;
use std::process::Command;

use common::assert_same;

#[test]
fn iris_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-iris-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let root = env!("CARGO_MANIFEST_DIR");
    let out = Command::new(env!("CARGO_BIN_EXE_iris"))
        .arg(&dir)
        .arg(format!("{root}/data/iris.data"))
        .output()
        .expect("failed to run iris binary");
    assert!(out.status.success(), "iris exited with {}", out.status);

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/iris/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &out.stdout, &golden("stdout.txt"));
    for f in ["iris_loss.csv", "iris_scatter.csv", "iris_confusion.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
