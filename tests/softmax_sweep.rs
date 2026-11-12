// QSoftmax and calibration::build_softmax against tinymind's C++
// (QSoftmaxFixedPoint1D, buildQSoftmaxFixedPointParams) over 504 calibrations
// (input scales (1 + j/8) * 2^e for e in -14..=6, betas 1, 0.5 and 2.5) and
// row lengths 1 to 511, 32 rows each. tests/golden/softmax/stdout.txt is the
// output of scripts/cpp/softmax_sweep.cpp (scripts/compare-with-cpp.sh rebuilds
// it and checks the golden is current).

// The sweep itself is in tests/sweeps/softmax.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/softmax.rs"]
mod softmax;

use tinymind::int8::QSoftmax;
use tinymind::int8::calibration::build_softmax;

struct Calibrate;

impl softmax::Calibrate for Calibrate {
    fn softmax(&mut self, input_scale: f64, beta: f64) -> QSoftmax {
        build_softmax(input_scale, beta)
    }
}

#[test]
fn softmax_sweep_matches_cpp() {
    let mut out = String::new();
    softmax::report(&mut out, &mut Calibrate);
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/softmax/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
