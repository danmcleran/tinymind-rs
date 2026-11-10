// Golden test for the maze example.
//
// The goldens come from tinymind's C++ `examples/maze`, built with its
// Makefile's release flags plus -ffp-contract=off (see
// `scripts/compare-with-cpp.sh`, which rebuilds it and diffs every file).
// maze_qtable.csv and maze_test.csv are in tests/golden/maze/; stdout (about
// 570 KB) and maze_training.csv (5000 episodes) are pinned here by length,
// line count and 64-bit FNV-1a hash of the C++ files.

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/maze_driver.rs"]
mod maze_driver;

use std::fs;

use common::assert_same;

/// A large output file's byte length, newline count and FNV-1a hash.
struct Pinned {
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

fn check_pinned(name: &str, data: &[u8], expected: Pinned) {
    assert_eq!(data.len(), expected.len, "{name} length");
    let lines = data.iter().filter(|&&b| b == b'\n').count();
    assert_eq!(lines, expected.lines, "{name} line count");
    assert_eq!(fnv1a(data), expected.fnv1a, "{name} FNV-1a");
}

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

    for f in ["maze_qtable.csv", "maze_test.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    check_pinned("stdout", &stdout, STDOUT);
    check_pinned("maze_training.csv", &actual("maze_training.csv"), TRAINING);
    let _ = fs::remove_dir_all(&dir);
}

const STDOUT: Pinned = Pinned {
    len: 571_672,
    lines: 35_730,
    fnv1a: 0xb264_24d4_6535_08d4,
};

const TRAINING: Pinned = Pinned {
    len: 75_934,
    lines: 5_000,
    fnv1a: 0x3906_5f04_8d04_9b02,
};
