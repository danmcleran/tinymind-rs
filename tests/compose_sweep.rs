// The int8 composition layers and their calibration against tinymind (C++)'s
// QMul (with buildQMulRequantizer), QConcat2_2D (with buildRescaler) and
// QPad2D. tests/golden/compose/stdout.txt is the output of
// scripts/cpp/compose_sweep.cpp, which draws the same scales, parameters and
// tensors from the same generator, in the same order;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// See that program for what the cases cover.

// The calibration functions need the `std` feature.
#![cfg(feature = "std")]

use std::fmt::Write;

use tinymind::int8::calibration::{build_mul_requantizer, build_rescaler};
use tinymind::int8::{QConcat, QMul, QPad2D, Requantizer};

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }

    fn i8(&mut self) -> i8 {
        (self.draw(256) as i32 - 128) as i8
    }

    /// (1 + d) * 2^-(16 + base + e): exact in f32.
    fn scale(&mut self, base: i32) -> f32 {
        let m = (1 + self.draw(1 << 16)) as f32;
        m * 2f32.powi(-16 - base - self.draw(4) as i32)
    }

    fn range(&mut self, zero_point: i8) -> (i32, i32) {
        let zp = i32::from(zero_point);
        match self.draw(4) {
            0 => (-128, 127),
            1 => (zp, 127),
            2 => ((zp - 20).max(-128), (zp + 20).min(127)),
            _ => (i32::from(self.i8()), i32::from(self.i8())),
        }
    }

    fn fill(&mut self, x: &mut [i8]) {
        for v in x {
            *v = self.i8();
        }
    }
}

fn fnv(h: u32, bytes: &[i8]) -> u32 {
    bytes.iter().fold(h, |h, &v| {
        (h ^ u32::from(v as u8)).wrapping_mul(0x0100_0193)
    })
}

fn rq_text(r: &Requantizer) -> String {
    format!(
        "rq={},{},{},{},{}",
        r.multiplier, r.shift, r.zero_point, r.qmin, r.qmax
    )
}

macro_rules! concat {
    ($rng:expr, $out:expr, $name:expr, $h:literal, $w:literal, $ca:literal, $cb:literal) => {
        for t in 0..4 {
            let input_a_zero_point = $rng.i8();
            let input_b_zero_point = $rng.i8();
            let (sa, sb, so) = ($rng.scale(10), $rng.scale(10), $rng.scale(10));
            let zo = $rng.i8();
            let (qmin, qmax) = $rng.range(zo);
            let layer = QConcat {
                input_a_zero_point,
                input_b_zero_point,
                requantizer_a: build_rescaler(sa, so, zo.into(), qmin, qmax),
                requantizer_b: build_rescaler(sb, so, zo.into(), qmin, qmax),
            };
            let mut a = [[[0i8; $ca]; $w]; $h];
            let mut b = [[[0i8; $cb]; $w]; $h];
            $rng.fill(a.as_flattened_mut().as_flattened_mut());
            $rng.fill(b.as_flattened_mut().as_flattened_mut());
            let mut y = [[[0i8; $ca + $cb]; $w]; $h];
            layer.forward(&a, &b, &mut y);
            writeln!(
                $out,
                "concat {} {t} scales=0x{:08X},0x{:08X},0x{:08X} zp={},{} {} {} hash=0x{:08X}",
                $name,
                sa.to_bits(),
                sb.to_bits(),
                so.to_bits(),
                input_a_zero_point,
                input_b_zero_point,
                rq_text(&layer.requantizer_a),
                rq_text(&layer.requantizer_b),
                fnv(0x811C_9DC5, y.as_flattened().as_flattened())
            )
            .unwrap();
        }
    };
}

macro_rules! pad {
    ($rng:expr, $out:expr, $name:expr, $h:literal, $w:literal, $c:literal,
     $pt:literal, $pb:literal, $pl:literal, $pr:literal) => {
        for t in 0..3 {
            let layer = QPad2D::<$pt, $pb, $pl, $pr> {
                pad_value: $rng.i8(),
            };
            let mut x = [[[0i8; $c]; $w]; $h];
            $rng.fill(x.as_flattened_mut().as_flattened_mut());
            let mut y = [[[0i8; $c]; $w + $pl + $pr]; $h + $pt + $pb];
            layer.forward(&x, &mut y);
            writeln!(
                $out,
                "pad {} {t} value={} hash=0x{:08X}",
                $name,
                layer.pad_value,
                fnv(0x811C_9DC5, y.as_flattened().as_flattened())
            )
            .unwrap();
        }
    };
}

#[test]
fn compose_matches_cpp() {
    let mut rng = Lcg(0x5EED_0003);
    let mut out = String::new();

    for k in 0..32 {
        let (sa, sb, so) = (rng.scale(4), rng.scale(4), rng.scale(6));
        let input_a_zero_point = rng.i8();
        let input_b_zero_point = rng.i8();
        let zo = rng.i8();
        let (qmin, qmax) = rng.range(zo);
        let mul = QMul {
            input_a_zero_point,
            input_b_zero_point,
            requantizer: build_mul_requantizer(sa, sb, so, zo.into(), qmin, qmax),
        };
        let mut b = [0i8; 256];
        for (j, v) in b.iter_mut().enumerate() {
            *v = (j as i32 - 128) as i8;
        }
        let mut h = 0x811C_9DC5;
        let mut y = [0i8; 256];
        for i in 0..256 {
            mul.forward(&[(i - 128) as i8; 256], &b, &mut y);
            h = fnv(h, &y);
        }
        writeln!(
            out,
            "mul calibrated {k} scales=0x{:08X},0x{:08X},0x{:08X} zp={},{} {} hash=0x{h:08X}",
            sa.to_bits(),
            sb.to_bits(),
            so.to_bits(),
            input_a_zero_point,
            input_b_zero_point,
            rq_text(&mul.requantizer)
        )
        .unwrap();
    }
    for k in 0..48 {
        let input_a_zero_point = rng.i8();
        let input_b_zero_point = rng.i8();
        let multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let shift = if rng.draw(3) == 0 {
            rng.draw(5) as i32 - 3
        } else {
            8 + rng.draw(8) as i32
        };
        let zero_point = rng.i8();
        let (qmin, qmax) = rng.range(zero_point);
        let mul = QMul {
            input_a_zero_point,
            input_b_zero_point,
            requantizer: Requantizer {
                multiplier,
                shift,
                zero_point,
                qmin: qmin as i8,
                qmax: qmax as i8,
            },
        };
        let (mut a, mut b) = ([0i8; 256], [0i8; 256]);
        rng.fill(&mut a);
        rng.fill(&mut b);
        let mut y = [0i8; 256];
        mul.forward(&a, &b, &mut y);
        writeln!(
            out,
            "mul direct {k} zp={},{} {} hash=0x{:08X}",
            input_a_zero_point,
            input_b_zero_point,
            rq_text(&mul.requantizer),
            fnv(0x811C_9DC5, &y)
        )
        .unwrap();
    }

    concat!(rng, out, "3x4_2+3", 3, 4, 2, 3);
    concat!(rng, out, "1x1_1+1", 1, 1, 1, 1);
    concat!(rng, out, "5x2_4+1", 5, 2, 4, 1);
    concat!(rng, out, "2x3_8+8", 2, 3, 8, 8);
    concat!(rng, out, "4x4_16+3", 4, 4, 16, 3);

    pad!(rng, out, "3x4x2_1111", 3, 4, 2, 1, 1, 1, 1);
    pad!(rng, out, "2x2x3_0230", 2, 2, 3, 0, 2, 3, 0);
    pad!(rng, out, "4x1x1_2001", 4, 1, 1, 2, 0, 0, 1);
    pad!(rng, out, "1x5x4_1100", 1, 5, 4, 1, 1, 0, 0);
    pad!(rng, out, "2x3x8_none", 2, 3, 8, 0, 0, 0, 0);
    pad!(rng, out, "3x3x1_4444", 3, 3, 1, 4, 4, 4, 4);

    assert_eq!(out, include_str!("golden/compose/stdout.txt"));
}
