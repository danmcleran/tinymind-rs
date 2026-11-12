// The int8 elementwise addition and its calibration against tinymind (C++)'s
// QAdd and buildQAddParams. tests/golden/add/stdout.txt is the output of
// scripts/cpp/add_sweep.cpp, which draws the same scales and parameters from
// the same generator, in the same order, and runs the C++ layer;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// See that program for what the cases cover.

// The calibration functions need the `std` feature.
#![cfg(feature = "std")]

// The sweep itself is in tests/sweeps/add.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/add.rs"]
mod add;

use tinymind::int8::QAdd;
use tinymind::int8::calibration::build_add;

struct Calibrate;

impl add::Calibrate for Calibrate {
    fn add(&mut self, a: (f32, i8), b: (f32, i8), out: (f32, i8), qmin: i8, qmax: i8) -> QAdd {
        build_add(
            (a.0, a.1.into()),
            (b.0, b.1.into()),
            (out.0, out.1.into()),
            qmin.into(),
            qmax.into(),
        )
    }
}

#[test]
fn add_matches_cpp() {
    let mut out = String::new();
    add::report(&mut out, &mut Calibrate);
    assert_eq!(out, include_str!("golden/add/stdout.txt"));
}
