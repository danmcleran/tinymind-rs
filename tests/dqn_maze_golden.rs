// Golden test for the deep Q-network maze example.
//
// The goldens come from tinymind's C++ `examples/dqn_maze`, built with its
// Makefile's release flags plus -ffp-contract=off (see
// `scripts/compare-with-cpp.sh`, which rebuilds it and diffs every file).
// dqn_maze_qtable.csv and dqn_maze_test.csv are in tests/golden/dqn_maze/;
// stdout (about 1.1 MB) and dqn_maze_training.csv (10000 episodes) are pinned
// here by length, line count and 64-bit FNV-1a hash of the C++ files. The C++
// writes nothing to stderr (no episode runs out of steps).

mod common;

// The example's program, run in-process (examples have no CARGO_BIN_EXE_*).
#[path = "../examples/common/dqn_maze_driver.rs"]
mod dqn_maze_driver;

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
fn dqn_maze_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-dqn-maze-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    dqn_maze_driver::run(&dir, &mut stdout, &mut stderr).expect("dqn_maze example failed");

    let root = env!("CARGO_MANIFEST_DIR");
    let golden = |f: &str| fs::read(format!("{root}/tests/golden/dqn_maze/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    for f in ["dqn_maze_qtable.csv", "dqn_maze_test.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    check_pinned("stdout", &stdout, STDOUT);
    check_pinned(
        "dqn_maze_training.csv",
        &actual("dqn_maze_training.csv"),
        TRAINING,
    );
    assert!(
        stderr.is_empty(),
        "stderr: {}",
        String::from_utf8_lossy(&stderr)
    );
    let _ = fs::remove_dir_all(&dir);
}

const STDOUT: Pinned = Pinned {
    len: 1_123_651,
    lines: 70_157,
    fnv1a: 0xb96f_bed8_d861_4e23,
};

const TRAINING: Pinned = Pinned {
    len: 149_782,
    lines: 10_000,
    fnv1a: 0x02c6_46a3_c12e_6728,
};
