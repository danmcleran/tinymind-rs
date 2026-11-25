// Golden test for the vibration anomaly example.
//
// The goldens in tests/golden/vibration/ come from tinymind's C++
// `examples/vibration_anomaly`, built with its Makefile's release flags plus
// -ffp-contract=off (see `scripts/compare-with-cpp.sh`, which rebuilds it and
// diffs every file). The per-frame CSV (390 KB) is pinned here by byte length,
// line count and 64-bit FNV-1a hash instead of being committed.
//
// One line differs from the C++ by design: the last line of stdout gives the
// trainer's size, which is 1,264 bytes in the C++ (its field order needs more
// padding) and 1,248 here, where the compiler orders the fields.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/vibration_driver.rs"]
mod vibration_driver;

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
fn vibration_matches_cpp_golden() {
    let dir =
        std::env::temp_dir().join(format!("tinymind-vibration-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = env!("CARGO_MANIFEST_DIR");
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/vibration/{f}")).unwrap();

    let mut stdout = Vec::new();
    vibration_driver::run(&dir, &mut stdout).expect("vibration example failed");
    let expected = String::from_utf8(golden("stdout.txt"))
        .unwrap()
        .replace("k-means trainer 1264 bytes", "k-means trainer 1248 bytes");
    assert_same("stdout", &stdout, expected.as_bytes());
    let f = "vibration_anomaly_summary.csv";
    assert_same(f, &fs::read(dir.join(f)).unwrap(), &golden(f));
    let data = fs::read(dir.join("vibration_anomaly.csv")).unwrap();
    let newlines = data.iter().filter(|&&b| b == b'\n').count();
    assert_eq!(
        (data.len(), newlines),
        (389_885, 7501),
        "vibration_anomaly.csv length, lines"
    );
    assert_eq!(
        fnv1a(&data),
        0xc047_37b3_891d_94b5,
        "vibration_anomaly.csv FNV-1a"
    );
    let _ = fs::remove_dir_all(&dir);
}
