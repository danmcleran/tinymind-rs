// The packed int4 layers against tinymind (C++) (cpp/qint4.hpp): QDenseInt4,
// QConv2DInt4 (unpacked and packed paths, odd windows that start filters
// mid-byte), QDepthwiseConv2DInt4, pack_int4 / unpack_int4 and
// calibration::quantize_int4_per_channel, over random tensors with the
// requantizers and draws of conv_sweep. Each layer is also compared with the
// int8 layer on the unpacked weights (same=1). tests/golden/int4/stdout.txt is
// the output of scripts/cpp/int4_sweep.cpp (scripts/compare-with-cpp.sh
// rebuilds it and checks the golden is current).

use std::fmt::Write;

use tinymind::int8::calibration::quantize_int4_per_channel;
use tinymind::int8::{
    QConv2DInt4, QConv2DPerChannel, QDenseInt4, QDensePerChannel, QDepthwiseConv2D,
    QDepthwiseConv2DInt4, Requantizer, pack_int4, unpack_int4,
};

const TRIALS: usize = 3;

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }

    fn i8(&mut self) -> i8 {
        (self.draw(256) as i32 - 128) as i8
    }

    fn tensor(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.i8()).collect()
    }

    fn packed<const P: usize>(&mut self) -> [u8; P] {
        core::array::from_fn(|_| self.draw(256) as u8)
    }

    fn biases<const N: usize>(&mut self) -> ([i32; N], bool) {
        let use_them = self.draw(4) != 0;
        let mut b = [0i32; N];
        for v in &mut b {
            *v = self.draw(1 << 17) as i32 - (1 << 16);
        }
        (b, use_them)
    }

    fn requantizer(&mut self) -> Requantizer {
        let multiplier = ((1u32 << 30) + (self.draw(1 << 24) << 6)) as i32;
        let shift = match self.draw(8) {
            0 => self.draw(6) as i32 - 3,
            1 => 20 + self.draw(12) as i32,
            _ => 6 + self.draw(8) as i32,
        };
        let zero_point = self.i8();
        let (qmin, qmax) = match self.draw(4) {
            0 => (zero_point, 127),
            1 => (-100, 90),
            _ => (-128, 127),
        };
        Requantizer {
            multiplier,
            shift,
            zero_point,
            qmin,
            qmax,
        }
    }

    fn requantizers<const N: usize>(&mut self) -> [Requantizer; N] {
        core::array::from_fn(|_| self.requantizer())
    }
}

struct Hash(u32);

impl Hash {
    fn add(&mut self, bytes: impl IntoIterator<Item = i8>) {
        for b in bytes {
            self.0 = (self.0 ^ u32::from(b as u8)).wrapping_mul(0x0100_0193);
        }
    }
}

fn unpacked<const P: usize>(packed: &[u8; P], n: usize) -> Vec<i8> {
    (0..n).map(|k| unpack_int4(packed, k).unwrap()).collect()
}

/// A nested array filled in memory order from a flat slice.
fn nested<const A: usize, const B: usize, const C: usize>(v: &[i8]) -> Box<[[[i8; C]; B]; A]> {
    let mut t = Box::new([[[0i8; C]; B]; A]);
    for (d, &s) in t.iter_mut().flatten().flatten().zip(v) {
        *d = s;
    }
    t
}

fn dense<const IN: usize, const OUT: usize, const P: usize>(
    out: &mut String,
    rng: &mut Lcg,
    name: &str,
) {
    let mut hash = Hash(0x811C_9DC5);
    let mut same = true;
    for _ in 0..TRIALS {
        let zp = rng.i8();
        let x: [i8; IN] = rng.tensor(IN).try_into().unwrap();
        let w: [u8; P] = rng.packed();
        let (b, use_bias) = rng.biases::<OUT>();
        let rqs = rng.requantizers::<OUT>();
        let biases = use_bias.then_some(&b);
        let layer: QDenseInt4<'_, IN, OUT, P> = QDenseInt4 {
            weights: &w,
            biases,
            input_zero_point: zp,
            requantizers: &rqs,
        };
        let y = layer.forward(&x);
        hash.add(y);
        let w8 = unpacked(&w, IN * OUT);
        let mut rows = [[0i8; IN]; OUT];
        for (d, &s) in rows.iter_mut().flatten().zip(&w8) {
            *d = s;
        }
        let reference = QDensePerChannel {
            weights: &rows,
            biases,
            input_zero_point: zp,
            requantizers: &rqs,
        };
        same &= reference.forward(&x) == y;
    }
    writeln!(out, "dense {name} 0x{:08X} same={}", hash.0, u8::from(same)).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn conv<
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
    const P: usize,
>(
    out: &mut String,
    rng: &mut Lcg,
    name: &str,
) {
    let mut hash = Hash(0x811C_9DC5);
    let mut same = true;
    for _ in 0..TRIALS {
        let zp = rng.i8();
        let x = nested::<H, W, C>(&rng.tensor(H * W * C));
        let w: [u8; P] = rng.packed();
        let (b, use_bias) = rng.biases::<F>();
        let rqs = rng.requantizers::<F>();
        let biases = use_bias.then_some(&b);
        let layer: QConv2DInt4<'_, C, KH, KW, F, P, SH, SW, PT, PB, PL, PR> = QConv2DInt4 {
            weights: &w,
            biases,
            input_zero_point: zp,
            requantizers: &rqs,
        };
        let mut y = Box::new([[[0i8; F]; OW]; OH]);
        layer.forward(&x, &mut y);
        hash.add(y.iter().flatten().flatten().copied());
        let w8 = unpacked(&w, F * KH * KW * C);
        let mut weights = Box::new([[[[0i8; C]; KW]; KH]; F]);
        for (d, &s) in weights.iter_mut().flatten().flatten().flatten().zip(&w8) {
            *d = s;
        }
        let reference: QConv2DPerChannel<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR> =
            QConv2DPerChannel {
                weights: &weights,
                biases,
                input_zero_point: zp,
                requantizers: &rqs,
            };
        let mut r = Box::new([[[0i8; F]; OW]; OH]);
        reference.forward(&x, &mut r);
        same &= r == y;
    }
    writeln!(
        out,
        "# {name} unpacks={}",
        u8::from(QConv2DInt4::<'_, C, KH, KW, F, P, SH, SW, PT, PB, PL, PR>::UNPACKS_WEIGHTS)
    )
    .unwrap();
    writeln!(out, "conv {name} 0x{:08X} same={}", hash.0, u8::from(same)).unwrap();
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
    const P: usize,
>(
    out: &mut String,
    rng: &mut Lcg,
    name: &str,
) {
    let mut hash = Hash(0x811C_9DC5);
    let mut same = true;
    for _ in 0..TRIALS {
        let zp = rng.i8();
        let x = nested::<H, W, C>(&rng.tensor(H * W * C));
        let w: [u8; P] = rng.packed();
        let (b, use_bias) = rng.biases::<C>();
        let rqs = rng.requantizers::<C>();
        let biases = use_bias.then_some(&b);
        let layer: QDepthwiseConv2DInt4<'_, C, KH, KW, P, SH, SW, PT, PB, PL, PR> =
            QDepthwiseConv2DInt4 {
                weights: &w,
                biases,
                input_zero_point: zp,
                requantizers: &rqs,
            };
        let mut y = Box::new([[[0i8; C]; OW]; OH]);
        layer.forward(&x, &mut y);
        hash.add(y.iter().flatten().flatten().copied());
        let weights = nested::<C, KH, KW>(&unpacked(&w, C * KH * KW));
        let reference: QDepthwiseConv2D<'_, C, KH, KW, SH, SW, PT, PB, PL, PR> = QDepthwiseConv2D {
            weights: &weights,
            biases,
            input_zero_point: zp,
            requantizers: &rqs,
        };
        let mut r = Box::new([[[0i8; C]; OW]; OH]);
        reference.forward(&x, &mut r);
        same &= r == y;
    }
    writeln!(
        out,
        "depthwise {name} 0x{:08X} same={}",
        hash.0,
        u8::from(same)
    )
    .unwrap();
}

fn packing(out: &mut String, rng: &mut Lcg) {
    let values: [i8; 35] =
        core::array::from_fn(|i| if i < 32 { (i % 16) as i8 - 8 } else { rng.i8() });
    let packed: [u8; 18] = pack_int4(&values);
    let hex: Vec<String> = packed.iter().map(|b| format!("{b:02x}")).collect();
    writeln!(out, "pack {}", hex.join(" ")).unwrap();
    let un: Vec<String> = (0..35)
        .map(|k| unpack_int4(&packed, k).unwrap().to_string())
        .collect();
    writeln!(out, "unpack {}", un.join(" ")).unwrap();
}

fn quantize(out: &mut String, rng: &mut Lcg) {
    let mut w = [[0f32; 9]; 4];
    for v in w.iter_mut().flatten() {
        *v = (rng.draw(2001) as i32 - 1000) as f32 / 1024.0;
    }
    w[2] = [0.0; 9];
    w[3] = [3.5, -0.25, 0.75, -1.25, 2.25, -3.25, 0.0, 1.75, -2.75];
    let (codes, scales) = quantize_int4_per_channel(&w);
    for (c, (row, scale)) in codes.iter().zip(scales).enumerate() {
        let codes: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        writeln!(
            out,
            "quantize c={c} scale={:08x} codes={}",
            scale.to_bits(),
            codes.join(",")
        )
        .unwrap();
    }
}

#[test]
fn int4_sweep_matches_cpp() {
    let mut out = String::new();
    let rng = &mut Lcg(0x5EED_0004);
    dense::<1, 1, 1>(&mut out, rng, "1x1");
    dense::<3, 2, 3>(&mut out, rng, "3x2");
    dense::<7, 5, 18>(&mut out, rng, "7x5");
    dense::<8, 4, 16>(&mut out, rng, "8x4");
    dense::<33, 10, 165>(&mut out, rng, "33x10");
    dense::<64, 16, 512>(&mut out, rng, "64x16");
    conv::<5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1, 5, 6, 54>(&mut out, rng, "unpack-3x3-pad");
    conv::<7, 7, 1, 3, 2, 2, 1, 3, 2, 0, 1, 1, 4, 8, 9>(&mut out, rng, "unpack-odd-window");
    conv::<4, 4, 5, 1, 1, 1, 1, 7, 0, 0, 0, 0, 4, 4, 18>(&mut out, rng, "unpack-pointwise");
    conv::<6, 6, 8, 3, 3, 1, 1, 16, 1, 1, 1, 1, 6, 6, 576>(&mut out, rng, "packed-patch");
    conv::<5, 6, 7, 3, 1, 2, 1, 50, 1, 1, 0, 0, 3, 6, 525>(&mut out, rng, "packed-patch-odd");
    conv::<5, 6, 15, 3, 3, 2, 1, 8, 1, 1, 1, 0, 3, 5, 540>(&mut out, rng, "packed-rows-odd");
    conv::<4, 5, 11, 1, 3, 1, 1, 32, 0, 0, 1, 1, 4, 5, 528>(&mut out, rng, "packed-one-row-odd");
    depthwise::<6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 6, 5, 14>(&mut out, rng, "3x3-pad");
    depthwise::<8, 8, 40, 5, 6, 2, 2, 2, 2, 3, 2, 4, 4, 600>(&mut out, rng, "large-strided");
    packing(&mut out, rng);
    quantize(&mut out, rng);

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/int4/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
