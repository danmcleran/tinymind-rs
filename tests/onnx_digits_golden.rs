// Golden test for the onnx_digits_int8 example.
//
// The goldens in tests/golden/onnx_digits/ come from tinymind's C++
// `examples/onnx_digits_int8`, built with its Makefile's release flags plus
// -ffp-contract=off (see `scripts/compare-with-cpp.sh`, which rebuilds it and
// diffs every file). The per-digit results (291 KB) and logits (96 KB) are
// pinned here by byte length, line count and 64-bit FNV-1a hash instead of
// being committed. The generated models use the ONNX importer's arena
// option, which needs Rust 1.88: CI's MSRV job skips this test.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/onnx_digits_driver.rs"]
mod onnx_digits_driver;

use std::fs;

use common::assert_same;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn onnx_digits_matches_cpp_golden() {
    let root = env!("CARGO_MANIFEST_DIR");
    let test_set = std::path::Path::new(root).join("data/optdigits.tes");
    // The test set is left out of the published crate (Cargo.toml `exclude`),
    // so a test run from the crate's tarball has nothing to read. A git
    // checkout must have it.
    if !test_set.exists() {
        assert!(
            !std::path::Path::new(root).join(".git").exists(),
            "data/optdigits.tes is missing from the checkout"
        );
        eprintln!("skipping: data/optdigits.tes not in this copy of the crate");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "tinymind-onnx-digits-golden-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/onnx_digits/{f}")).unwrap();

    let mut stdout = Vec::new();
    onnx_digits_driver::run(&dir, &test_set, &mut stdout).expect("onnx_digits_int8 failed");
    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in ["onnx_digits_int8_plan.csv", "onnx_digits_int8_summary.csv"] {
        assert_same(f, &fs::read(dir.join(f)).unwrap(), &golden(f));
    }
    for (f, len, lines, hash) in [
        ("onnx_digits_int8.csv", 290_870, 1798, 0x718c_5c7f_bd6f_e087),
        (
            "onnx_digits_int8_logits.csv",
            96_259,
            2561,
            0x8a7e_1fd9_0d36_9c77,
        ),
    ] {
        let data = fs::read(dir.join(f)).unwrap();
        let newlines = data.iter().filter(|&&b| b == b'\n').count();
        assert_eq!((data.len(), newlines), (len, lines), "{f} length, lines");
        assert_eq!(fnv1a(&data), hash, "{f} FNV-1a");
    }
    let _ = fs::remove_dir_all(&dir);
}
