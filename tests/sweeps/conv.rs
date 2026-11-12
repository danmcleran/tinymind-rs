// The int8 convolution and pooling sweep, shared by tests/conv_sweep.rs (on
// the host, against tinymind (C++)'s output) and bench/src/sweeps.rs (on
// Cortex-M, against the same golden). no_std and allocation-free on the
// device: see `zeroed`.

use core::fmt::Write;

use tinymind::int8::{
    QAvgPool2D, QConv2D, QConv2DPerChannel, QDepthwiseConv2D, QGlobalAvgPool2D, QMaxPool2D,
    QPointwiseConv2D, QPointwiseConv2DPerChannel, Requantizer,
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

    /// Fill any nested `i8` array in memory order, as the C++ fills a flat one.
    fn fill<T: AsMutBytes + ?Sized>(&mut self, t: &mut T) {
        for b in t.bytes() {
            *b = self.i8();
        }
    }

    /// The biases, and whether this trial uses them (drawn first).
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

    fn range(&mut self) -> (i8, i8) {
        match self.draw(3) {
            0 => (-128, 127),
            1 => (-20, 20),
            _ => {
                let qmin = self.i8();
                (qmin, self.i8())
            }
        }
    }
}

/// The bytes of a nested `i8` array, in memory order.
trait AsMutBytes {
    fn bytes(&mut self) -> impl Iterator<Item = &mut i8>;
}

impl AsMutBytes for i8 {
    fn bytes(&mut self) -> impl Iterator<Item = &mut i8> {
        core::iter::once(self)
    }
}

impl<T: AsMutBytes, const N: usize> AsMutBytes for [T; N] {
    fn bytes(&mut self) -> impl Iterator<Item = &mut i8> {
        self.iter_mut().flat_map(|t| t.bytes())
    }
}

struct Hash(u32);

impl Hash {
    fn new() -> Self {
        Hash(0x811C_9DC5)
    }

    fn add<T: AsMutBytes>(&mut self, mut t: T) {
        for &mut b in t.bytes() {
            self.0 ^= u32::from(b as u8);
            self.0 = self.0.wrapping_mul(0x0100_0193);
        }
    }
}

/// Zeroed nested array. On the host it is heap-allocated: the larger tensors
/// would crowd a test thread's stack in a debug build. On the device (no heap)
/// it lives on the stack, which optimised builds share between the shapes.
#[cfg(not(target_os = "none"))]
fn zeroed<T: AsMutBytes + Copy, const N: usize>(zero: T) -> Box<[T; N]> {
    Box::new([zero; N])
}

#[cfg(target_os = "none")]
fn zeroed<T: AsMutBytes + Copy, const N: usize>(zero: T) -> OnStack<[T; N]> {
    OnStack([zero; N])
}

/// A value that dereferences like the host's `Box`, so the call sites are the
/// same on both.
#[cfg(target_os = "none")]
struct OnStack<T>(T);

#[cfg(target_os = "none")]
impl<T> core::ops::Deref for OnStack<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[cfg(target_os = "none")]
impl<T> core::ops::DerefMut for OnStack<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

macro_rules! conv {
    ($out:ident, $name:literal, $h:literal, $w:literal, $c:literal, $kh:literal, $kw:literal,
     $sh:literal, $sw:literal, $f:literal, $pt:literal, $pb:literal, $pl:literal, $pr:literal) => {{
        const OH: usize = ($h + $pt + $pb - $kh) / $sh + 1;
        const OW: usize = ($w + $pl + $pr - $kw) / $sw + 1;
        for per_channel in [false, true] {
            let r = &mut *$out.1;
            let mut hash = Hash::new();
            for _ in 0..TRIALS {
                let zp = r.i8();
                let mut input = zeroed::<[[i8; $c]; $w], $h>([[0; $c]; $w]);
                r.fill(&mut *input);
                let mut weights = zeroed::<[[[i8; $c]; $kw]; $kh], $f>([[[0; $c]; $kw]; $kh]);
                r.fill(&mut *weights);
                let (biases, use_biases) = r.biases::<$f>();
                let biases = use_biases.then_some(&biases);
                let mut output = zeroed::<[[i8; $f]; OW], OH>([[0; $f]; OW]);
                if per_channel {
                    let rqs = r.requantizers::<$f>();
                    let layer: QConv2DPerChannel<
                        '_,
                        $c,
                        $kh,
                        $kw,
                        $f,
                        $sh,
                        $sw,
                        $pt,
                        $pb,
                        $pl,
                        $pr,
                    > = QConv2DPerChannel {
                        weights: &weights,
                        biases,
                        input_zero_point: zp,
                        requantizers: &rqs,
                    };
                    layer.forward(&input, &mut output);
                } else {
                    let requantizer = r.requantizer();
                    let layer: QConv2D<'_, $c, $kh, $kw, $f, $sh, $sw, $pt, $pb, $pl, $pr> =
                        QConv2D {
                            weights: &weights,
                            biases,
                            input_zero_point: zp,
                            requantizer,
                        };
                    layer.forward(&input, &mut output);
                }
                hash.add(*output);
            }
            let kind = if per_channel {
                "conv_per_channel"
            } else {
                "conv"
            };
            writeln!($out.0, "{kind} {} 0x{:08X}", $name, hash.0).unwrap();
        }
    }};
}

macro_rules! depthwise {
    ($out:ident, $name:literal, $h:literal, $w:literal, $c:literal, $kh:literal, $kw:literal,
     $sh:literal, $sw:literal, $pt:literal, $pb:literal, $pl:literal, $pr:literal) => {{
        const OH: usize = ($h + $pt + $pb - $kh) / $sh + 1;
        const OW: usize = ($w + $pl + $pr - $kw) / $sw + 1;
        let r = &mut *$out.1;
        let mut hash = Hash::new();
        for _ in 0..TRIALS {
            let zp = r.i8();
            let mut input = zeroed::<[[i8; $c]; $w], $h>([[0; $c]; $w]);
            r.fill(&mut *input);
            let mut weights = [[[0i8; $kw]; $kh]; $c];
            r.fill(&mut weights);
            let (biases, use_biases) = r.biases::<$c>();
            let biases = use_biases.then_some(&biases);
            let rqs = r.requantizers::<$c>();
            let layer: QDepthwiseConv2D<'_, $c, $kh, $kw, $sh, $sw, $pt, $pb, $pl, $pr> =
                QDepthwiseConv2D {
                    weights: &weights,
                    biases,
                    input_zero_point: zp,
                    requantizers: &rqs,
                };
            let mut output = zeroed::<[[i8; $c]; OW], OH>([[0; $c]; OW]);
            layer.forward(&input, &mut output);
            hash.add(*output);
        }
        writeln!($out.0, "depthwise {} 0x{:08X}", $name, hash.0).unwrap();
    }};
}

macro_rules! pointwise {
    ($out:ident, $name:literal, $h:literal, $w:literal, $c:literal, $f:literal) => {{
        for per_channel in [false, true] {
            let r = &mut *$out.1;
            let mut hash = Hash::new();
            for _ in 0..TRIALS {
                let zp = r.i8();
                let mut input = zeroed::<[[i8; $c]; $w], $h>([[0; $c]; $w]);
                r.fill(&mut *input);
                let mut weights = [[0i8; $c]; $f];
                r.fill(&mut weights);
                let (biases, use_biases) = r.biases::<$f>();
                let biases = use_biases.then_some(&biases);
                let mut output = zeroed::<[[i8; $f]; $w], $h>([[0; $f]; $w]);
                if per_channel {
                    let rqs = r.requantizers::<$f>();
                    let layer = QPointwiseConv2DPerChannel {
                        weights: &weights,
                        biases,
                        input_zero_point: zp,
                        requantizers: &rqs,
                    };
                    layer.forward(&input, &mut output);
                } else {
                    let requantizer = r.requantizer();
                    let layer = QPointwiseConv2D {
                        weights: &weights,
                        biases,
                        input_zero_point: zp,
                        requantizer,
                    };
                    layer.forward(&input, &mut output);
                }
                hash.add(*output);
            }
            let kind = if per_channel {
                "pointwise_per_channel"
            } else {
                "pointwise"
            };
            writeln!($out.0, "{kind} {} 0x{:08X}", $name, hash.0).unwrap();
        }
    }};
}

macro_rules! pool {
    ($out:ident, $name:literal, $h:literal, $w:literal, $c:literal, $ph:literal, $pw:literal,
     $sh:literal, $sw:literal) => {{
        const OH: usize = ($h - $ph) / $sh + 1;
        const OW: usize = ($w - $pw) / $sw + 1;
        let r = &mut *$out.1;
        let mut hash = Hash::new();
        for _ in 0..TRIALS {
            let mut input = [[[0i8; $c]; $w]; $h];
            r.fill(&mut input);
            let mut output = [[[0i8; $c]; OW]; OH];
            QMaxPool2D::<$ph, $pw, $sh, $sw>.forward(&input, &mut output);
            hash.add(output);
        }
        writeln!($out.0, "max_pool {} 0x{:08X}", $name, hash.0).unwrap();

        let mut hash = Hash::new();
        for _ in 0..TRIALS {
            let mut input = [[[0i8; $c]; $w]; $h];
            r.fill(&mut input);
            let (qmin, qmax) = r.range();
            let mut output = [[[0i8; $c]; OW]; OH];
            QAvgPool2D::<$ph, $pw, $sh, $sw> { qmin, qmax }.forward(&input, &mut output);
            hash.add(output);
        }
        writeln!($out.0, "avg_pool {} 0x{:08X}", $name, hash.0).unwrap();
    }};
}

macro_rules! global_avg_pool {
    ($out:ident, $name:literal, $h:literal, $w:literal, $c:literal) => {{
        let r = &mut *$out.1;
        let mut hash = Hash::new();
        for _ in 0..TRIALS {
            let mut input = [[[0i8; $c]; $w]; $h];
            r.fill(&mut input);
            let (qmin, qmax) = r.range();
            hash.add(QGlobalAvgPool2D { qmin, qmax }.forward(&input));
        }
        writeln!($out.0, "global_avg_pool {} 0x{:08X}", $name, hash.0).unwrap();
    }};
}

/// Write the sweep's report (tests/golden/conv/stdout.txt) to `text`.
pub fn run<W: Write>(text: &mut W) {
    let mut rng = Lcg(0x5EED_0001);
    let out = (text, &mut rng);

    //          name             H   W   C  KH  KW SH SW  F PT PB PL PR
    conv!(out, "stem_20x20", 20, 20, 1, 3, 3, 1, 1, 8, 0, 0, 0, 0);
    conv!(out, "same_3x3", 7, 9, 3, 3, 3, 1, 1, 5, 1, 1, 1, 1);
    conv!(out, "rows_3x3x8", 10, 10, 8, 3, 3, 1, 1, 4, 0, 0, 0, 0);
    conv!(out, "rows_stride2", 9, 11, 8, 3, 3, 2, 2, 7, 1, 1, 1, 1);
    conv!(out, "asym_pad", 12, 5, 2, 4, 2, 3, 1, 3, 0, 2, 1, 0);
    conv!(out, "one_by_one", 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    conv!(out, "dscnn_stem", 49, 10, 1, 10, 4, 2, 2, 8, 4, 5, 1, 2);
    conv!(out, "tiles_1x5", 5, 40, 2, 1, 5, 1, 1, 6, 0, 0, 2, 2);
    conv!(out, "rows_padded", 3, 3, 16, 3, 3, 1, 1, 2, 1, 1, 1, 1);
    conv!(out, "stride_4x3", 8, 8, 4, 5, 5, 4, 3, 9, 0, 0, 0, 0);
    conv!(out, "stride_gt_k", 7, 7, 1, 2, 2, 3, 3, 2, 0, 1, 0, 1);
    conv!(out, "wide_pad", 4, 4, 2, 3, 3, 1, 1, 3, 2, 2, 2, 2);
    conv!(out, "all_pad", 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2);

    //               name           H   W   C KH KW SH SW PT PB PL PR
    depthwise!(out, "kws_3x3", 9, 9, 8, 3, 3, 1, 1, 0, 0, 0, 0);
    depthwise!(out, "same_3x3", 13, 5, 12, 3, 3, 1, 1, 1, 1, 1, 1);
    depthwise!(out, "asym_pad", 8, 7, 3, 3, 2, 2, 1, 0, 2, 1, 0);
    depthwise!(out, "big_kernel", 6, 6, 5, 5, 5, 1, 1, 0, 0, 0, 0);
    depthwise!(out, "one_by_one", 10, 4, 2, 1, 1, 3, 2, 0, 0, 0, 0);
    depthwise!(out, "stride_gt_k", 7, 7, 4, 2, 2, 3, 3, 1, 0, 0, 1);
    depthwise!(out, "all_pad", 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2);

    //               name         H  W   C   F
    pointwise!(out, "kws_8_16", 7, 7, 8, 16);
    pointwise!(out, "pixel", 1, 1, 16, 10);
    pointwise!(out, "wide", 5, 3, 40, 6);
    pointwise!(out, "odd", 3, 3, 33, 5);
    pointwise!(out, "tiny", 4, 1, 1, 1);

    //          name            H   W  C PH PW SH SW
    pool!(out, "kws_2x2", 18, 18, 8, 2, 2, 2, 2);
    pool!(out, "overlap_3x3", 7, 9, 3, 3, 3, 2, 2);
    pool!(out, "rect_2x3", 5, 5, 2, 2, 3, 1, 2);
    pool!(out, "big_window", 6, 4, 1, 4, 4, 1, 1);
    pool!(out, "tiled_3x3", 9, 9, 4, 3, 3, 3, 3);

    //                     name     H  W   C
    global_avg_pool!(out, "kws", 7, 7, 16);
    global_avg_pool!(out, "pixel", 1, 1, 3);
    global_avg_pool!(out, "rect", 5, 3, 10);
    global_avg_pool!(out, "odd", 25, 5, 8);
}
