// The 16x8 layers (int16x8) against tinymind (C++): the 64-bit requantizer
// on every corner, QDense, QConv2D, QConv2DPerChannel, QDepthwiseConv2D
// (windows above and below 256 taps), QMaxPool2D, QAdd (calibrated with
// left shift 15, and random), apply_lut over all 65,536 inputs on random and
// built sigmoid / tanh tables, and QSoftmax from build_softmax.
// tests/golden/int16x8/stdout.txt is the output of
// scripts/cpp/int16x8_sweep.cpp (scripts/compare-with-cpp.sh rebuilds it and
// checks the golden is current); this file computes the same lines.

use std::fmt::Write;

use tinymind::int16x8::calibration::{build_add, build_sigmoid_lut, build_softmax, build_tanh_lut};
use tinymind::int16x8::{
    LUT_SIZE, QAdd, QConv2D, QConv2DPerChannel, QDense, QDepthwiseConv2D, QMaxPool2D, Requantizer,
    Requantizer32, apply_lut,
};

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn draw(&mut self, n: u32) -> u32 {
        (self.next() >> 8) % n
    }
    fn i16(&mut self) -> i16 {
        (self.next() >> 16) as u16 as i16
    }
    fn i8(&mut self) -> i8 {
        (self.draw(256) as i32 - 128) as i8
    }
    fn i64(&mut self) -> i64 {
        let high = u64::from(self.next() >> 16);
        let mid = u64::from(self.next() >> 16);
        let low = u64::from(self.next() >> 16);
        ((high << 48) | (mid << 32) | (low << 16) | u64::from(self.next() >> 16)) as i64
    }
    fn multiplier(&mut self) -> i32 {
        ((1u32 << 30) + (self.draw(1 << 24) << 6)) as i32
    }
    fn scale(&mut self, min_exp: u32, spread: u32) -> f64 {
        let mantissa = 1.0 + f64::from(self.draw(1000)) / 1000.0;
        let e = (min_exp + self.draw(spread)) as i32;
        mantissa * 2f64.powi(-e)
    }

    fn rq(&mut self) -> Requantizer {
        let multiplier = self.multiplier();
        let shift = self.draw(15) as i32;
        let zero_point = if self.draw(2) == 0 { 0 } else { self.i16() / 4 };
        let range = self.draw(3);
        let qmin = match range {
            0 => zero_point,
            1 => -20000,
            _ => -32768,
        };
        let qmax = if range == 1 { 25000 } else { 32767 };
        Requantizer {
            multiplier,
            shift,
            zero_point,
            qmin,
            qmax,
        }
    }

    fn fine_rq(&mut self) -> Requantizer {
        let multiplier = self.multiplier();
        let shift = self.draw(4) as i32;
        Requantizer {
            multiplier,
            shift,
            zero_point: 0,
            qmin: -32768,
            qmax: 32767,
        }
    }

    /// Trial 1 of each layer runs small inputs through fine requantizers.
    fn input(&mut self, t: usize) -> i16 {
        let v = self.i16();
        if t == 1 { v >> 9 } else { v }
    }

    fn any_rq(&mut self) -> Requantizer {
        let multiplier = match self.draw(6) {
            0 => self.next() as i32,
            1 => 0x7FFF_0000 + self.draw(0x10000) as i32,
            2 => -(self.draw(1 << 30) as i32),
            _ => self.multiplier(),
        };
        let shift = self.draw(71) as i32 - 20;
        let zero_point = self.i16();
        let mut qmin = self.i16();
        let qmax = if self.draw(4) == 0 { self.i16() } else { 32767 };
        if self.draw(3) == 0 {
            qmin = -32768;
        }
        Requantizer {
            multiplier,
            shift,
            zero_point,
            qmin,
            qmax,
        }
    }

    fn biases<const N: usize>(&mut self) -> ([i64; N], bool) {
        let use_them = self.draw(4) != 0;
        (core::array::from_fn(|_| self.i64() >> 30), use_them)
    }
}

struct Hash(u32);

impl Hash {
    fn new() -> Self {
        Hash(0x811C_9DC5)
    }
    fn add(&mut self, v: i16) {
        let u = v as u16;
        self.0 = (self.0 ^ u32::from(u & 0xFF)).wrapping_mul(0x0100_0193);
        self.0 = (self.0 ^ u32::from(u >> 8)).wrapping_mul(0x0100_0193);
    }
    fn add_all<'a>(&mut self, vs: impl IntoIterator<Item = &'a i16>) {
        for &v in vs {
            self.add(v);
        }
    }
}

fn nested<const A: usize, const B: usize, const C: usize, T: Copy + Default>(
    mut f: impl FnMut() -> T,
) -> Box<[[[T; C]; B]; A]> {
    let mut t = Box::new([[[T::default(); C]; B]; A]);
    for v in t.iter_mut().flatten().flatten() {
        *v = f();
    }
    t
}

fn requantizers(out: &mut String, rng: &mut Lcg) {
    let mut h = Hash::new();
    for _ in 0..20000 {
        let rq = rng.any_rq();
        let kind = rng.draw(4);
        let mut acc = rng.i64();
        match kind {
            0 => acc >>= 16,
            1 => acc >>= 40,
            2 => {
                let v = i64::from(rng.i16());
                acc = v * (1i64 << rng.draw(40));
            }
            _ => {}
        }
        h.add(rq.apply(acc));
    }
    writeln!(out, "rq64 corners 0x{:08X}", h.0).unwrap();
}

fn dense<const IN: usize, const OUT: usize>(out: &mut String, rng: &mut Lcg, name: &str) {
    let mut h = Hash::new();
    for t in 0..4 {
        let zp = if t == 0 { 0 } else { rng.i16() };
        let x: [i16; IN] = core::array::from_fn(|_| rng.input(t));
        let mut w = Box::new([[0i8; IN]; OUT]);
        for v in w.iter_mut().flatten() {
            *v = rng.i8();
        }
        let (b, use_bias) = rng.biases::<OUT>();
        let layer: QDense<'_, IN, OUT> = QDense {
            weights: &w,
            biases: use_bias.then_some(&b),
            input_zero_point: zp,
            requantizer: if t == 1 { rng.fine_rq() } else { rng.rq() },
        };
        h.add_all(&layer.forward(&x));
    }
    writeln!(out, "dense {name} 0x{:08X}", h.0).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn conv<
    const PER_CHANNEL: bool,
    const H: usize,
    const W: usize,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const SH: usize,
    const SW: usize,
    const F: usize,
    const PT: usize,
    const PB: usize,
    const PL: usize,
    const PR: usize,
    const OH: usize,
    const OW: usize,
>(
    out: &mut String,
    rng: &mut Lcg,
    name: &str,
) {
    let mut h = Hash::new();
    for t in 0..3 {
        let zp = if t == 0 { 0 } else { rng.i16() };
        let x = nested::<H, W, C, i16>(|| rng.input(t));
        let mut w = Box::new([[[[0i8; C]; KW]; KH]; F]);
        for v in w.iter_mut().flatten().flatten().flatten() {
            *v = rng.i8();
        }
        let (b, use_bias) = rng.biases::<F>();
        let rqs: [Requantizer; F] =
            core::array::from_fn(|_| if t == 1 { rng.fine_rq() } else { rng.rq() });
        let mut y = Box::new([[[0i16; F]; OW]; OH]);
        let biases = use_bias.then_some(&b);
        if PER_CHANNEL {
            let layer: QConv2DPerChannel<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR> =
                QConv2DPerChannel {
                    weights: &w,
                    biases,
                    input_zero_point: zp,
                    requantizers: &rqs,
                };
            layer.forward(&x, &mut y);
        } else {
            let layer: QConv2D<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR> = QConv2D {
                weights: &w,
                biases,
                input_zero_point: zp,
                requantizer: rqs[0],
            };
            layer.forward(&x, &mut y);
        }
        h.add_all(y.iter().flatten().flatten());
    }
    let kind = if PER_CHANNEL {
        "conv-per-channel"
    } else {
        "conv"
    };
    writeln!(out, "{kind} {name} 0x{:08X}", h.0).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn depthwise<
    const H: usize,
    const W: usize,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const SH: usize,
    const SW: usize,
    const PT: usize,
    const PB: usize,
    const PL: usize,
    const PR: usize,
    const OH: usize,
    const OW: usize,
>(
    out: &mut String,
    rng: &mut Lcg,
    name: &str,
) {
    let mut h = Hash::new();
    for t in 0..3 {
        let zp = if t == 0 { 0 } else { rng.i16() };
        let x = nested::<H, W, C, i16>(|| rng.input(t));
        let w = nested::<C, KH, KW, i8>(|| rng.i8());
        let (b, use_bias) = rng.biases::<C>();
        let rqs: [Requantizer; C] =
            core::array::from_fn(|_| if t == 1 { rng.fine_rq() } else { rng.rq() });
        let layer: QDepthwiseConv2D<'_, C, KH, KW, SH, SW, PT, PB, PL, PR> = QDepthwiseConv2D {
            weights: &w,
            biases: use_bias.then_some(&b),
            input_zero_point: zp,
            requantizers: &rqs,
        };
        let mut y = Box::new([[[0i16; C]; OW]; OH]);
        layer.forward(&x, &mut y);
        h.add_all(y.iter().flatten().flatten());
    }
    writeln!(out, "depthwise {name} 0x{:08X}", h.0).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn max_pool<
    const H: usize,
    const W: usize,
    const C: usize,
    const PH: usize,
    const PW: usize,
    const SH: usize,
    const SW: usize,
    const OH: usize,
    const OW: usize,
>(
    out: &mut String,
    rng: &mut Lcg,
    name: &str,
) {
    let x = nested::<H, W, C, i16>(|| rng.i16());
    let mut y = Box::new([[[0i16; C]; OW]; OH]);
    QMaxPool2D::<PH, PW, SH, SW>.forward(&x, &mut y);
    let mut h = Hash::new();
    h.add_all(y.iter().flatten().flatten());
    writeln!(out, "maxpool {name} 0x{:08X}", h.0).unwrap();
}

fn adds(out: &mut String, rng: &mut Lcg) {
    const N: usize = 257;
    let mut h = Hash::new();
    for t in 0..24 {
        let add = if t < 16 {
            let sa = rng.scale(0, 14) as f32;
            let sb = rng.scale(0, 14) as f32;
            let so = rng.scale(0, 13) as f32;
            let add = build_add((sa, 0), (sb, 0), (so, 0), -32768, 32767);
            writeln!(
                out,
                "add calibrated {t}: {} {} {} {} {} {} {}",
                add.left_shift,
                add.input_a_multiplier,
                add.input_a_shift,
                add.input_b_multiplier,
                add.input_b_shift,
                add.output_requantizer.multiplier,
                add.output_requantizer.shift
            )
            .unwrap();
            add
        } else {
            let input_a_zero_point = (rng.draw(201) as i32 - 100) as i16;
            let input_b_zero_point = (rng.draw(201) as i32 - 100) as i16;
            let left_shift = 10 + rng.draw(6) as i32;
            let input_a_multiplier = rng.multiplier();
            let input_a_shift = rng.draw(4) as i32;
            let input_b_multiplier = rng.multiplier();
            let input_b_shift = rng.draw(4) as i32;
            let multiplier = rng.multiplier();
            let shift = rng.draw(20) as i32 - 2;
            let zero_point = rng.i16() / 8;
            let qmin = (-30000 + rng.draw(1000) as i32) as i16;
            let qmax = (30000 - rng.draw(1000) as i32) as i16;
            QAdd {
                input_a_zero_point,
                input_b_zero_point,
                left_shift,
                input_a_multiplier,
                input_a_shift,
                input_b_multiplier,
                input_b_shift,
                output_requantizer: Requantizer32 {
                    multiplier,
                    shift,
                    zero_point,
                    qmin,
                    qmax,
                },
            }
        };
        let mut a = [0i16; N];
        let mut b = [0i16; N];
        for i in 0..N {
            a[i] = if t < 16 { rng.i16() } else { rng.i16() / 2 };
            b[i] = if t < 16 { rng.i16() } else { rng.i16() / 2 };
        }
        let mut y = [0i16; N];
        add.forward(&a, &b, &mut y);
        h.add_all(&y);
    }
    writeln!(out, "add int16 0x{:08X}", h.0).unwrap();
}

fn luts(out: &mut String, rng: &mut Lcg) {
    let mut lut = [0i16; LUT_SIZE];
    for t in 0..6 {
        if t < 2 {
            for v in lut.iter_mut() {
                *v = rng.i16();
            }
        } else {
            let in_scale = rng.scale(9, 4);
            let in_zp = if t % 2 == 0 {
                0
            } else {
                i32::from(rng.i16()) / 4
            };
            lut = if t < 4 {
                build_sigmoid_lut(in_scale, in_zp, 1.0 / 32768.0)
            } else {
                build_tanh_lut(in_scale, in_zp, 1.0 / 32768.0)
            };
            let mut th = Hash::new();
            th.add_all(&lut);
            writeln!(out, "lut table {t} 0x{:08X}", th.0).unwrap();
        }
        let mut h = Hash::new();
        for x in i16::MIN..=i16::MAX {
            h.add(apply_lut(x, &lut));
        }
        writeln!(out, "lut apply {t} 0x{:08X}", h.0).unwrap();
    }
}

fn softmax_rows<const N: usize>(out: &mut String, rng: &mut Lcg, name: &str) {
    let mut h = Hash::new();
    for t in 0..8 {
        let scale = rng.scale(10, 6) as f32;
        let beta = if t % 3 == 0 {
            1.0f32
        } else {
            0.5f32 + rng.draw(100) as f32 / 50.0f32
        };
        let softmax = build_softmax(f64::from(scale), f64::from(beta));
        writeln!(
            out,
            "softmax {name} {t} params {} {}",
            softmax.input_multiplier, softmax.input_shift
        )
        .unwrap();
        let x: [i16; N] = core::array::from_fn(|i| {
            if t == 7 {
                (-32768 + 65535 * (i % 2) as i32) as i16
            } else {
                rng.i16()
            }
        });
        let mut y = [0i16; N];
        softmax.forward(&x, &mut y);
        h.add_all(&y);
    }
    writeln!(out, "softmax {name} 0x{:08X}", h.0).unwrap();
}

#[test]
fn int16x8_sweep_matches_cpp() {
    let mut out = String::new();
    let rng = &mut Lcg(0x1608_1608);
    requantizers(&mut out, rng);
    dense::<1, 1>(&mut out, rng, "1x1");
    dense::<7, 5>(&mut out, rng, "7x5");
    dense::<64, 10>(&mut out, rng, "64x10");
    dense::<300, 3>(&mut out, rng, "300x3");
    conv::<false, 5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1, 5, 6>(&mut out, rng, "3x3-pad");
    conv::<true, 5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1, 5, 6>(&mut out, rng, "3x3-pad");
    conv::<true, 7, 7, 1, 3, 2, 2, 1, 3, 2, 0, 1, 1, 4, 8>(&mut out, rng, "strided-asym");
    conv::<true, 4, 4, 9, 1, 1, 1, 1, 5, 0, 0, 0, 0, 4, 4>(&mut out, rng, "pointwise");
    conv::<false, 6, 5, 2, 2, 3, 2, 2, 6, 0, 1, 2, 0, 3, 3>(&mut out, rng, "odd");
    depthwise::<6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 6, 5>(&mut out, rng, "3x3-pad");
    depthwise::<8, 8, 4, 5, 5, 2, 1, 2, 2, 0, 1, 4, 5>(&mut out, rng, "5x5-strided");
    depthwise::<17, 18, 2, 17, 17, 1, 1, 0, 0, 1, 1, 1, 4>(&mut out, rng, "289-taps");
    max_pool::<6, 6, 3, 2, 2, 2, 2, 3, 3>(&mut out, rng, "2x2");
    max_pool::<7, 5, 2, 3, 2, 2, 1, 3, 4>(&mut out, rng, "3x2-strided");
    adds(&mut out, rng);
    luts(&mut out, rng);
    softmax_rows::<1>(&mut out, rng, "1");
    softmax_rows::<2>(&mut out, rng, "2");
    softmax_rows::<10>(&mut out, rng, "10");
    softmax_rows::<300>(&mut out, rng, "300");

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/int16x8/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
