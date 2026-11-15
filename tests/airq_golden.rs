// Golden test for the air_quality example.
//
// The goldens in tests/golden/airq/ come from tinymind's C++
// `examples/air_quality`, built with its Makefile's flags plus
// -ffp-contract=off (the series is synthesized in double arithmetic, and a
// compiler that fused multiply-adds would change it), run with its defaults:
// up to 15000 epochs, stopping after 28 once no sample trains.
// `scripts/compare-with-cpp.sh` checks these goldens against a fresh C++ run.

mod common;

#[path = "../examples/common/airq_driver.rs"]
mod airq_driver;

use std::fs;

use common::assert_same;

#[test]
fn airq_matches_cpp_golden() {
    let dir = std::env::temp_dir().join(format!("tinymind-airq-golden-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let root = env!("CARGO_MANIFEST_DIR");
    let mut stdout = Vec::new();
    // No AirQualityUCI.csv there, so the example synthesizes its series.
    airq_driver::run(
        &dir,
        &dir.join("AirQualityUCI.csv"),
        airq_driver::TRAINING_EPOCHS,
        airq_driver::LEARNING_RATE,
        &mut stdout,
    )
    .expect("air_quality failed");

    let golden = |f: &str| fs::read(format!("{root}/tests/golden/airq/{f}")).unwrap();
    let actual = |f: &str| fs::read(dir.join(f)).unwrap();

    assert_same("stdout", &stdout, &golden("stdout.txt"));
    for f in ["airq_loss.csv", "airq_forecast.csv"] {
        assert_same(f, &actual(f), &golden(f));
    }
    let _ = fs::remove_dir_all(&dir);
}
