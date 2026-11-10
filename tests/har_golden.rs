// Golden test for the har_activity example.
//
// The goldens in tests/golden/har/ come from tinymind's C++
// `examples/har_activity`, built with its Makefile's flags plus
// -ffp-contract=off (see `scripts/compare-with-cpp.sh`, which rebuilds it and
// diffs everything): the example synthesizes its data in double arithmetic,
// and a compiler that fuses multiply-adds would change the windows.

mod common;

#[path = "../examples/common/har_driver.rs"]
mod har_driver;

use std::fs;

use common::assert_same;

#[test]
fn har_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-har-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let root = env!("CARGO_MANIFEST_DIR");
    let mut stdout = Vec::new();
    // No har.csv here, so the example synthesizes its windows.
    har_driver::run(&dir, &dir.join("har.csv"), false, &mut stdout).expect("har_activity failed");

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/har/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in ["har_samples.csv", "har_loss.csv", "har_confusion.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
