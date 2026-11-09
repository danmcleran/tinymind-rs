// Golden test for the maze example.
//
// The goldens in tests/golden/maze/ come from tinymind's C++ `examples/maze`,
// built with its Makefile's release flags (see `scripts/compare-with-cpp.sh`,
// which rebuilds it and diffs everything).

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/maze_driver.rs"]
mod maze_driver;

use std::fs;

use common::assert_same;

#[test]
fn maze_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-maze-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let mut stdout = Vec::new();
    maze_driver::run(&dir, &mut stdout).expect("maze example failed");

    let root = env!("CARGO_MANIFEST_DIR");
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/maze/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in ["maze_training.csv", "maze_test.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
