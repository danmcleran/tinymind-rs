// Golden test for the int8 XOR example.
//
// The goldens in tests/golden/int8_xor/ come from tinymind's C++
// `examples/pytorch_quant/xor`, built with its Makefile's release flags (see
// `scripts/compare-with-cpp.sh`, which rebuilds it and diffs both files).

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/int8_xor_driver.rs"]
mod int8_xor_driver;

use std::fs;

use common::assert_same;

#[test]
fn int8_xor_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-int8-xor-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let mut stdout = Vec::new();
    let correct = int8_xor_driver::run(&dir, &mut stdout).expect("int8_xor example failed");
    assert_eq!(correct, 4);

    let root = env!("CARGO_MANIFEST_DIR");
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/int8_xor/{f}")).unwrap();
    assert_same("stdout", &stdout, &golden("stdout.txt"));
    let surface = fs::read(dir.join("xor_decision_surface.csv")).unwrap();
    assert_same(
        "xor_decision_surface.csv",
        &surface,
        &golden("xor_decision_surface.csv"),
    );
    let _ = fs::remove_dir_all(&dir);
}
