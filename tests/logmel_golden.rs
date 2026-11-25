// Golden test for the log-mel example.
//
// The goldens in tests/golden/logmel/ come from tinymind's C++
// `examples/logmel_vibration`, built with its Makefile's release flags plus
// -ffp-contract=off (see `scripts/compare-with-cpp.sh`, which rebuilds it and
// diffs every file). The two per-frame CSVs (183 KB and 69 KB) are pinned here
// by byte length, line count and 64-bit FNV-1a hash instead of being
// committed.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/logmel_driver.rs"]
mod logmel_driver;

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
fn logmel_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-logmel-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = env!("CARGO_MANIFEST_DIR");
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/logmel/{f}")).unwrap();

    let mut stdout = Vec::new();
    logmel_driver::run(&dir, &mut stdout).expect("logmel example failed");
    assert_same("stdout", &stdout, &golden("stdout.txt"));
    let f = "logmel_vibration_error.csv";
    assert_same(f, &fs::read(dir.join(f)).unwrap(), &golden(f));
    for (f, len, lines, hash) in [
        ("logmel_vibration.csv", 182_826, 300, 0x835e_7172_0124_3e52),
        (
            "logmel_vibration_mfcc.csv",
            69_410,
            300,
            0xa9ea_918f_07f8_52e1,
        ),
    ] {
        let data = fs::read(dir.join(f)).unwrap();
        let newlines = data.iter().filter(|&&b| b == b'\n').count();
        assert_eq!((data.len(), newlines), (len, lines), "{f} length, lines");
        assert_eq!(fnv1a(&data), hash, "{f} FNV-1a");
    }
    let _ = fs::remove_dir_all(&dir);
}
