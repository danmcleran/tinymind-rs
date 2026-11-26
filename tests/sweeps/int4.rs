// The packed int4 sweep, shared by tests/int4_sweep.rs (on the host, against
// tinymind (C++)'s output, with calibration::quantize_int4_per_channel) and
// bench/src/sweeps.rs (on Cortex-M, against the same golden, without the
// float quantizer: its lines are skipped there). no_std and allocation-free.
// The C++ is scripts/cpp/int4_sweep.cpp.

use core::fmt::Write;

use tinymind::int8::{
    QConv2DInt4, QConv2DPerChannel, QDenseInt4, QDensePerChannel, QDepthwiseConv2D,
    QDepthwiseConv2DInt4, Requantizer, pack_int4, unpack_int4,
};

/// What the sweep needs from `int8::calibration` (std only): on the host
/// `quantize_int4_per_channel`, on the device nothing (`None`).
pub trait Quantize {
    /// `quantize_int4_per_channel(w)`.
    fn quantize(&mut self, w: &[[f32; 9]; 4]) -> Option<([[i8; 9]; 4], [f32; 4])>;
}

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

    /// Fill a slice in memory order, as the C++ fills a flat tensor.
    fn fill(&mut self, t: &mut [i8]) {
        for v in t {
            *v = self.i8();
        }
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

/// Unpack `packed` into `out` in memory order.
fn unpack_into<const P: usize>(packed: &[u8; P], out: &mut [i8]) {
    for (k, d) in out.iter_mut().enumerate() {
        *d = unpack_int4(packed, k).unwrap_or(0);
    }
}

fn dense<const IN: usize, const OUT: usize, const P: usize>(
    out: &mut impl Write,
    rng: &mut Lcg,
    name: &str,
) {
    let mut hash = Hash(0x811C_9DC5);
    let mut same = true;
    for _ in 0..TRIALS {
        let zp = rng.i8();
        let mut x = [0i8; IN];
        rng.fill(&mut x);
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
        let mut rows = [[0i8; IN]; OUT];
        unpack_into(&w, rows.as_flattened_mut());
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
    out: &mut impl Write,
    rng: &mut Lcg,
    name: &str,
) {
    let mut hash = Hash(0x811C_9DC5);
    let mut same = true;
    for _ in 0..TRIALS {
        let zp = rng.i8();
        let mut x = [[[0i8; C]; W]; H];
        rng.fill(x.as_flattened_mut().as_flattened_mut());
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
        let mut y = [[[0i8; F]; OW]; OH];
        layer.forward(&x, &mut y);
        hash.add(y.iter().flatten().flatten().copied());
        let mut weights = [[[[0i8; C]; KW]; KH]; F];
        unpack_into(
            &w,
            weights
                .as_flattened_mut()
                .as_flattened_mut()
                .as_flattened_mut(),
        );
        let reference: QConv2DPerChannel<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR> =
            QConv2DPerChannel {
                weights: &weights,
                biases,
                input_zero_point: zp,
                requantizers: &rqs,
            };
        let mut r = [[[0i8; F]; OW]; OH];
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
    out: &mut impl Write,
    rng: &mut Lcg,
    name: &str,
) {
    let mut hash = Hash(0x811C_9DC5);
    let mut same = true;
    for _ in 0..TRIALS {
        let zp = rng.i8();
        let mut x = [[[0i8; C]; W]; H];
        rng.fill(x.as_flattened_mut().as_flattened_mut());
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
        let mut y = [[[0i8; C]; OW]; OH];
        layer.forward(&x, &mut y);
        hash.add(y.iter().flatten().flatten().copied());
        let mut weights = [[[0i8; KW]; KH]; C];
        unpack_into(&w, weights.as_flattened_mut().as_flattened_mut());
        let reference: QDepthwiseConv2D<'_, C, KH, KW, SH, SW, PT, PB, PL, PR> = QDepthwiseConv2D {
            weights: &weights,
            biases,
            input_zero_point: zp,
            requantizers: &rqs,
        };
        let mut r = [[[0i8; C]; OW]; OH];
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

fn packing(out: &mut impl Write, rng: &mut Lcg) {
    let values: [i8; 35] =
        core::array::from_fn(|i| if i < 32 { (i % 16) as i8 - 8 } else { rng.i8() });
    let packed: [u8; 18] = pack_int4(&values);
    out.write_str("pack").unwrap();
    for b in packed {
        write!(out, " {b:02x}").unwrap();
    }
    out.write_str("\nunpack").unwrap();
    for k in 0..35 {
        write!(out, " {}", unpack_int4(&packed, k).unwrap_or(0)).unwrap();
    }
    out.write_str("\n").unwrap();
}

fn quantize(out: &mut impl Write, rng: &mut Lcg, cal: &mut impl Quantize) {
    let mut w = [[0f32; 9]; 4];
    for v in w.iter_mut().flatten() {
        *v = (rng.draw(2001) as i32 - 1000) as f32 / 1024.0;
    }
    w[2] = [0.0; 9];
    w[3] = [3.5, -0.25, 0.75, -1.25, 2.25, -3.25, 0.0, 1.75, -2.75];
    let Some((codes, scales)) = cal.quantize(&w) else {
        return;
    };
    for (c, (row, scale)) in codes.iter().zip(scales).enumerate() {
        write!(out, "quantize c={c} scale={:08x} codes=", scale.to_bits()).unwrap();
        for (k, v) in row.iter().enumerate() {
            write!(out, "{}{v}", if k == 0 { "" } else { "," }).unwrap();
        }
        out.write_str("\n").unwrap();
    }
}

/// The whole sweep, in the golden's order.
pub fn run(out: &mut impl Write, cal: &mut impl Quantize) {
    let rng = &mut Lcg(0x5EED_0004);
    dense::<1, 1, 1>(out, rng, "1x1");
    dense::<3, 2, 3>(out, rng, "3x2");
    dense::<7, 5, 18>(out, rng, "7x5");
    dense::<8, 4, 16>(out, rng, "8x4");
    dense::<33, 10, 165>(out, rng, "33x10");
    dense::<64, 16, 512>(out, rng, "64x16");
    conv::<5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1, 5, 6, 54>(out, rng, "unpack-3x3-pad");
    conv::<7, 7, 1, 3, 2, 2, 1, 3, 2, 0, 1, 1, 4, 8, 9>(out, rng, "unpack-odd-window");
    conv::<4, 4, 5, 1, 1, 1, 1, 7, 0, 0, 0, 0, 4, 4, 18>(out, rng, "unpack-pointwise");
    conv::<6, 6, 8, 3, 3, 1, 1, 16, 1, 1, 1, 1, 6, 6, 576>(out, rng, "packed-patch");
    conv::<5, 6, 7, 3, 1, 2, 1, 50, 1, 1, 0, 0, 3, 6, 525>(out, rng, "packed-patch-odd");
    conv::<5, 6, 15, 3, 3, 2, 1, 8, 1, 1, 1, 0, 3, 5, 540>(out, rng, "packed-rows-odd");
    conv::<4, 5, 11, 1, 3, 1, 1, 32, 0, 0, 1, 1, 4, 5, 528>(out, rng, "packed-one-row-odd");
    depthwise::<6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 6, 5, 14>(out, rng, "3x3-pad");
    depthwise::<8, 8, 40, 5, 6, 2, 2, 2, 2, 3, 2, 4, 4, 600>(out, rng, "large-strided");
    packing(out, rng);
    quantize(out, rng, cal);
}
