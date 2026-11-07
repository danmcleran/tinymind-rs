// Golden tests for the xor example, one per format.
//
// The goldens in tests/golden/<format>/ come from tinymind's C++
// `examples/xor`: Q16.16 is the example as shipped (produced at tinymind commit
// 4e74539, output unchanged through TINYMIND_REF); Q8.8 is the same example with
// both bit counts in xornet.h set to 8, as `scripts/compare-with-cpp.sh` builds it
// (regenerated at tinymind e78dd7a: #239 changed the Q8.8 weight initialisation).
// The 2.4-2.6 MB nn_fixed_xor.txt is not stored; it is checked by byte length,
// newline count and 64-bit FNV-1a hash instead.
// `scripts/compare-with-cpp.sh` rebuilds the C++ example and does the full
// byte-for-byte diff of all four files in both formats.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/xor_driver.rs"]
mod xor_driver;

use std::fs;

use common::assert_same;

/// Expected `nn_fixed_xor.txt`: byte length, newline count, FNV-1a hash.
struct NnFile {
    len: usize,
    lines: usize,
    fnv1a: u64,
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn check(format: &str, golden_dir: &str, nn_expected: NnFile) {
    let dir = std::env::temp_dir().join(format!(
        "tinymind-xor-golden-{golden_dir}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    assert!(xor_driver::FORMATS.contains(&format));
    let mut stdout = Vec::new();
    xor_driver::run_format(format, &dir, None, None, &mut stdout).expect("xor example failed");

    let golden = |f: &str| {
        fs::read(format!(
            "{}/tests/golden/{golden_dir}/{f}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    };

    let nn = fs::read(dir.join("nn_fixed_xor.txt")).unwrap();
    let train = fs::read(dir.join("xor_training.csv")).unwrap();
    let surface = fs::read(dir.join("xor_decision_surface.csv")).unwrap();
    let _ = fs::remove_dir_all(&dir);

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    assert_same("xor_training.csv", &train, &golden("xor_training.csv"));
    assert_same(
        "xor_decision_surface.csv",
        &surface,
        &golden("xor_decision_surface.csv"),
    );

    assert_eq!(nn.len(), nn_expected.len, "nn_fixed_xor.txt length");
    assert_eq!(
        nn.iter().filter(|&&b| b == b'\n').count(),
        nn_expected.lines,
        "nn_fixed_xor.txt line count"
    );
    assert_eq!(fnv1a(&nn), nn_expected.fnv1a, "nn_fixed_xor.txt FNV-1a");
}

#[test]
fn xor_q16_16_matches_cpp_golden() {
    check(
        "q16.16",
        "q16_16",
        NnFile {
            len: 2_587_725,
            lines: 20_001,
            fnv1a: 0x0f6d_21c3_e3a5_076b,
        },
    );
}

#[test]
fn xor_q8_8_matches_cpp_golden() {
    check(
        "q8.8",
        "q8_8",
        NnFile {
            len: 2_383_566,
            lines: 20_001,
            fnv1a: 0x189a_6e57_c8c2_2955,
        },
    );
}
