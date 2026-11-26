// The 16x8 layers (int16x8) against tinymind (C++): the 64-bit requantizer
// on every corner, QDense, QConv2D, QConv2DPerChannel, QDepthwiseConv2D
// (windows above and below 256 taps), QMaxPool2D, QAdd (calibrated with
// left shift 15, and random), apply_lut over all 65,536 inputs on random and
// built sigmoid / tanh tables, and QSoftmax from build_softmax.
// tests/golden/int16x8/stdout.txt is the output of
// scripts/cpp/int16x8_sweep.cpp (scripts/compare-with-cpp.sh rebuilds it and
// checks the golden is current); this file computes the same lines.

// The sweep itself is in tests/sweeps/int16x8.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/int16x8.rs"]
mod int16x8;

use tinymind::int16x8::calibration::{build_add, build_sigmoid_lut, build_softmax, build_tanh_lut};
use tinymind::int16x8::{LUT_SIZE, QAdd, QSoftmax};

struct Host;

impl int16x8::Calibrate for Host {
    fn add(&mut self, sa: f32, sb: f32, so: f32) -> QAdd {
        build_add((sa, 0), (sb, 0), (so, 0), -32768, 32767)
    }

    fn lut(&mut self, sigmoid: bool, in_scale: f64, in_zp: i32) -> Option<[i16; LUT_SIZE]> {
        Some(if sigmoid {
            build_sigmoid_lut(in_scale, in_zp, 1.0 / 32768.0)
        } else {
            build_tanh_lut(in_scale, in_zp, 1.0 / 32768.0)
        })
    }

    fn softmax(&mut self, scale: f64, beta: f64) -> QSoftmax {
        build_softmax(scale, beta)
    }
}

#[test]
fn int16x8_sweep_matches_cpp() {
    let mut out = String::new();
    int16x8::run(&mut out, &mut Host);
    let golden = include_str!("golden/int16x8/stdout.txt");
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
