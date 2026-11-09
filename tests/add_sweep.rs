// The int8 elementwise addition and its calibration against tinymind (C++)'s
// QAdd and buildQAddParams. tests/golden/add/stdout.txt is the output of
// scripts/cpp/add_sweep.cpp, which draws the same scales and parameters from
// the same generator, in the same order, and runs the C++ layer;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// See that program for what the cases cover.

use std::fmt::Write;

use tinymind::int8::calibration::build_add;
use tinymind::int8::{QAdd, Requantizer};

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }

    fn i8(&mut self) -> i8 {
        (self.draw(256) as i32 - 128) as i8
    }

    /// (1 + d) * 2^-(16 + e): exact in f32.
    fn scale(&mut self) -> f32 {
        let m = (1 + self.draw(1 << 16)) as f32;
        m * 2f32.powi(-16 - self.draw(4) as i32)
    }

    /// The output range: full, fused ReLU, narrow around the zero point, or
    /// any (so sometimes qmin > qmax).
    fn range(&mut self, zero_point: i8) -> (i8, i8) {
        match self.draw(4) {
            0 => (-128, 127),
            1 => (zero_point, 127),
            2 => (zero_point.saturating_sub(20), zero_point.saturating_add(20)),
            _ => (self.i8(), self.i8()),
        }
    }
}

fn fnv(h: u32, bytes: &[i8]) -> u32 {
    bytes.iter().fold(h, |h, &v| {
        (h ^ u32::from(v as u8)).wrapping_mul(0x0100_0193)
    })
}

fn print(out: &mut String, kind: &str, k: usize, add: &QAdd, h: u32) {
    let rq = &add.output_requantizer;
    writeln!(
        out,
        "{kind} {k} zp={},{} ls={} a={},{} b={},{} rq={},{},{},{},{} hash=0x{h:08X}",
        add.input_a_zero_point,
        add.input_b_zero_point,
        add.left_shift,
        add.input_a_multiplier,
        add.input_a_shift,
        add.input_b_multiplier,
        add.input_b_shift,
        rq.multiplier,
        rq.shift,
        rq.zero_point,
        rq.qmin,
        rq.qmax
    )
    .unwrap();
}

#[test]
fn add_matches_cpp() {
    let mut rng = Lcg(0x5EED_0002);
    let mut out = String::new();

    for k in 0..48 {
        let sa = rng.scale();
        let sb = if rng.draw(4) == 0 { sa } else { rng.scale() };
        let so = rng.scale();
        let za = rng.i8();
        let zb = rng.i8();
        let zo = rng.i8();
        let (qmin, qmax) = rng.range(zo);
        let add = build_add(
            (sa, za.into()),
            (sb, zb.into()),
            (so, zo.into()),
            qmin.into(),
            qmax.into(),
        );
        // Every (a, b) pair: a row per value of a, b running over all 256.
        let mut b = [0i8; 256];
        for (j, v) in b.iter_mut().enumerate() {
            *v = (j as i32 - 128) as i8;
        }
        let mut h = 0x811C_9DC5;
        let mut y = [0i8; 256];
        for i in 0..256 {
            add.forward(&[(i - 128) as i8; 256], &b, &mut y);
            h = fnv(h, &y);
        }
        writeln!(
            out,
            "scales 0x{:08X} 0x{:08X} 0x{:08X}",
            sa.to_bits(),
            sb.to_bits(),
            so.to_bits()
        )
        .unwrap();
        print(&mut out, "calibrated", k, &add, h);
    }

    for k in 0..64 {
        let input_a_zero_point = rng.i8();
        let input_b_zero_point = rng.i8();
        let left_shift = rng.draw(22) as i32;
        let input_a_multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let input_a_shift = rng.draw(12) as i32 - 1;
        let input_b_multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let input_b_shift = rng.draw(12) as i32 - 1;
        let multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let mut shift = left_shift - input_a_shift.min(input_b_shift) + rng.draw(5) as i32 - 2;
        if rng.draw(3) == 0 || shift < -3 {
            shift = rng.draw(5) as i32 - 3;
        }
        let zero_point = rng.i8();
        let (qmin, qmax) = rng.range(zero_point);
        let add = QAdd {
            input_a_zero_point,
            input_b_zero_point,
            left_shift,
            input_a_multiplier,
            input_a_shift,
            input_b_multiplier,
            input_b_shift,
            output_requantizer: Requantizer {
                multiplier,
                shift: shift.min(31),
                zero_point,
                qmin,
                qmax,
            },
        };
        let (mut a, mut b) = ([0i8; 256], [0i8; 256]);
        for i in 0..256 {
            a[i] = rng.i8();
            b[i] = rng.i8();
        }
        let mut y = [0i8; 256];
        add.forward(&a, &b, &mut y);
        print(&mut out, "direct", k, &add, fnv(0x811C_9DC5, &y));
        // Below 256 elements the layer rescales element by element instead of
        // through tables: the same outputs.
        for ((ya, yb), yc) in a
            .chunks_exact(8)
            .zip(b.chunks_exact(8))
            .zip(y.chunks_exact(8))
        {
            let (ya, yb): ([i8; 8], [i8; 8]) = (ya.try_into().unwrap(), yb.try_into().unwrap());
            let mut small = [0i8; 8];
            add.forward(&ya, &yb, &mut small);
            assert_eq!(&small[..], yc, "direct {k}: element-wise path differs");
        }
    }

    for k in 0..16 {
        let input_a_zero_point = rng.i8();
        let input_b_zero_point = rng.i8();
        let left_shift = 18 + rng.draw(4) as i32;
        let input_a_multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let input_a_shift = rng.draw(2) as i32 - 1;
        let input_b_multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let input_b_shift = rng.draw(2) as i32 - 1;
        let multiplier = ((1u32 << 30) + (rng.draw(1 << 24) << 6)) as i32;
        let shift = -1 - rng.draw(3) as i32;
        let zero_point = rng.i8();
        let add = QAdd {
            input_a_zero_point,
            input_b_zero_point,
            left_shift,
            input_a_multiplier,
            input_a_shift,
            input_b_multiplier,
            input_b_shift,
            output_requantizer: Requantizer {
                multiplier,
                shift,
                zero_point,
                qmin: -128,
                qmax: 127,
            },
        };
        let (mut a, mut b) = ([0i8; 256], [0i8; 256]);
        for i in 0..256 {
            a[i] = rng.i8();
            b[i] = rng.i8();
        }
        let mut y = [0i8; 256];
        add.forward(&a, &b, &mut y);
        print(&mut out, "saturating", k, &add, fnv(0x811C_9DC5, &y));
    }

    let golden = include_str!("golden/add/stdout.txt");
    assert_eq!(out, golden);
}
