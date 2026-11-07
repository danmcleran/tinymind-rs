// Golden test for the xor example.
//
// The goldens in tests/golden/ were produced by tinymind's C++ `examples/xor`
// at tinymind commit 4e74539. The 2.6 MB nn_fixed_xor.txt is not stored; it is
// checked by byte length, newline count and 64-bit FNV-1a hash instead.
// `scripts/compare-with-cpp.sh` rebuilds the C++ example and does the full
// byte-for-byte diff of all four files.

use std::fs;
use std::process::Command;

const NN_LEN: usize = 2_587_725;
const NN_LINES: usize = 20_001;
const NN_FNV1A: u64 = 0x0f6d_21c3_e3a5_076b;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn assert_same(name: &str, actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let a = String::from_utf8_lossy(actual);
    let e = String::from_utf8_lossy(expected);
    let mut al = a.split('\n');
    let mut el = e.split('\n');
    let mut n = 1;
    loop {
        match (al.next(), el.next()) {
            (Some(x), Some(y)) if x == y => n += 1,
            (x, y) => panic!("{name} differs at line {n}:\n  actual:   {x:?}\n  expected: {y:?}"),
        }
    }
}

#[test]
fn xor_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("rustymind-xor-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_xor"))
        .arg(&dir)
        .output()
        .expect("failed to run xor binary");
    assert!(out.status.success(), "xor exited with {}", out.status);

    let golden =
        |f: &str| fs::read(format!("{}/tests/golden/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap();

    let nn = fs::read(dir.join("nn_fixed_xor.txt")).unwrap();
    let train = fs::read(dir.join("xor_training.csv")).unwrap();
    let surface = fs::read(dir.join("xor_decision_surface.csv")).unwrap();
    let _ = fs::remove_dir_all(&dir);

    assert_same("stdout", &out.stdout, &golden("stdout.txt"));
    assert_same("xor_training.csv", &train, &golden("xor_training.csv"));
    assert_same(
        "xor_decision_surface.csv",
        &surface,
        &golden("xor_decision_surface.csv"),
    );

    assert_eq!(nn.len(), NN_LEN, "nn_fixed_xor.txt length");
    assert_eq!(
        nn.iter().filter(|&&b| b == b'\n').count(),
        NN_LINES,
        "nn_fixed_xor.txt line count"
    );
    assert_eq!(fnv1a(&nn), NN_FNV1A, "nn_fixed_xor.txt FNV-1a");
}
