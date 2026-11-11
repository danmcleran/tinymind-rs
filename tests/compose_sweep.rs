// The int8 composition layers and their calibration against tinymind (C++)'s
// QMul (with buildQMulRequantizer), QConcat2_2D (with buildRescaler) and
// QPad2D. tests/golden/compose/stdout.txt is the output of
// scripts/cpp/compose_sweep.cpp, which draws the same scales, parameters and
// tensors from the same generator, in the same order;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// See that program for what the cases cover.

// The calibration functions need the `std` feature.
#![cfg(feature = "std")]

// The sweep itself is in tests/sweeps/compose.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/compose.rs"]
mod compose;

use tinymind::int8::Requantizer;
use tinymind::int8::calibration::{build_mul_requantizer, build_rescaler};

struct Calibrate;

impl compose::Calibrate for Calibrate {
    fn mul_requantizer(
        &mut self,
        sa: f32,
        sb: f32,
        so: f32,
        zo: i8,
        qmin: i32,
        qmax: i32,
    ) -> Requantizer {
        build_mul_requantizer(sa, sb, so, zo.into(), qmin, qmax)
    }

    fn rescaler(&mut self, s: f32, so: f32, zo: i8, qmin: i32, qmax: i32) -> Requantizer {
        build_rescaler(s, so, zo.into(), qmin, qmax)
    }
}

#[test]
fn compose_matches_cpp() {
    let mut out = String::new();
    compose::report(&mut out, &mut Calibrate);
    assert_eq!(out, include_str!("golden/compose/stdout.txt"));
}
