//! 16x8 quantized inference, ported from tinymind (C++) v19.0's int16 paths
//! (`cpp/qaffine.hpp`, `cpp/qdense.hpp`, `cpp/qconv2d.hpp`,
//! `cpp/qdepthwiseconv2d.hpp`, `cpp/qpointwiseconv2d.hpp`, `cpp/qpool2d.hpp`,
//! `cpp/qadd.hpp`, `cpp/qactivations.hpp`, `cpp/qsoftmax_fixedpoint.hpp`):
//! TFLite's 16x8 scheme, int16 activations with int8 weights.
//!
//! Activations are `i16` with twice int8's resolution, for twice the
//! activation RAM; weights stay `i8` (symmetric, zero point 0) and biases and
//! accumulators are `i64`, as in TFLite and TFLite Micro. A layer computes
//! `acc[o] = bias[o] + sum_i weight[o][i] * (input[i] - input_zero_point)` in
//! `i64`, then a 64-bit [`Requantizer`] rescales it to the output's `i16`
//! grid exactly as TFLite's 64-bit `MultiplyByQuantizedMultiplier` does.
//! TFLite requires int16 tensors to have zero point 0; these layers accept
//! any.
//!
//! The layers have the [`int8`](crate::int8) layers' names, shapes and
//! conventions (NHWC nested arrays, kernel, strides and padding as const
//! parameters, output sizes checked at compile time): an int8 model becomes
//! a 16x8 one by changing the activation type, the bias type and the
//! requantizers, with no recalibration of the weights. [`QAdd`] adds two
//! int16 tensors (TFLite's int16 ADD), [`apply_lut`] runs TFLite's 513-entry
//! interpolated int16 table for sigmoid or tanh, and [`QSoftmax`] is
//! TFLite's int16 softmax, bit for bit. `calibration` (feature `std`) builds
//! the parameters from float scales.

use crate::int8::conv::{tap, window_fits};
use crate::int8::{
    multiply_by_quantized_multiplier, rounding_high_mul_divide_by_pot, saturating_shl,
};

#[cfg(feature = "std")]
pub mod calibration;
mod softmax_tables;

/// Rescales an `i64` accumulator to int16 storage (C++ `Requantizer<int64_t,
/// int16_t>`), exactly as TFLite's 64-bit `MultiplyByQuantizedMultiplier`
/// does, then adds the zero point and saturates to `[qmin, qmax]`:
///
/// ```text
/// m16    = multiplier < 0x7FFF0000 ? (multiplier + 2^15) >> 16 : 0x7FFF
/// scaled = (acc * m16 + 2^(14 + shift)) >> (15 + shift)
/// ```
///
/// The Q0.31 multiplier is reduced to Q0.15, so the product fits 64 bits for
/// any `|acc| < 2^47`, with one rounding. `shift` has the int8
/// [`Requantizer`](crate::int8::Requantizer)'s sign (positive divides).
/// TFLite's contract is `multiplier >= 0`, `-7 <= shift <= 31`, `|acc| <
/// 2^47` and a scaled value that fits `i32`; outside it the result is
/// defined: `acc` is clamped to `[-2^47, 2^47 - 1]`, `shift` to `-14..=47`,
/// and a scaled value beyond `i32` saturates before the zero point is added.
/// With `qmin > qmax` the result is `qmax`.
///
/// ```
/// use tinymind::int16x8::Requantizer;
///
/// // A ratio of 1/2: multiplier 2^30 (0.5 in Q0.31), shift 0.
/// let rq = Requantizer { multiplier: 1 << 30, shift: 0, zero_point: 0, qmin: -32768, qmax: 32767 };
/// assert_eq!(rq.apply(1000), 500);
/// assert_eq!(rq.apply(-3), -1); // -1.5 rounds half up
/// assert_eq!(rq.apply(1 << 40), 32767);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requantizer {
    /// Q0.31 fixed-point multiplier, normally in `[2^30, 2^31)`.
    pub multiplier: i32,
    /// Power-of-two exponent: positive divides, negative multiplies.
    pub shift: i32,
    /// Output tensor zero point (TFLite: 0).
    pub zero_point: i16,
    /// Lowest output value.
    pub qmin: i16,
    /// Highest output value.
    pub qmax: i16,
}

impl Requantizer {
    /// The scaled accumulator before the zero point: TFLite's 64-bit
    /// `MultiplyByQuantizedMultiplier`, saturated to `i32`.
    pub(crate) fn scale(&self, acc: i64) -> i32 {
        const LIMIT: i64 = 1 << 47;
        let x = acc.clamp(-LIMIT, LIMIT - 1);
        let s = self.shift.clamp(-14, 47);
        let reduced = if self.multiplier < 0x7FFF_0000 {
            ((i64::from(self.multiplier) + (1 << 15)) >> 16) as i32
        } else {
            0x7FFF
        };
        let total = (15 + s) as u32; // 1..=62
        // |x * reduced| < 2^62 and the rounding term is at most 2^61: no
        // overflow.
        let v = x * i64::from(reduced) + (1i64 << (total - 1));
        let q = v >> total;
        q.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }

    /// Rescale `acc`, add the zero point and saturate to `[qmin, qmax]`.
    pub fn apply(&self, acc: i64) -> i16 {
        // |zero_point|, |qmin| and |qmax| are below 2^16, so clamping to
        // +-2^30 first keeps the sum in i32 and changes no result.
        let rail = 1 << 30;
        let v = self.scale(acc).clamp(-rail, rail) + i32::from(self.zero_point);
        clamp16(v, self.qmin, self.qmax)
    }
}

/// `v` saturated to `[qmin, qmax]` with two `if`s, so `qmin > qmax` gives
/// `qmax` (as the C++) rather than a panic.
#[inline]
fn clamp16(v: i32, qmin: i16, qmax: i16) -> i16 {
    let mut v = v;
    if v < i32::from(qmin) {
        v = i32::from(qmin);
    }
    if v > i32::from(qmax) {
        v = i32::from(qmax);
    }
    v as i16
}

/// Rescales an `i32` sum to int16 storage (C++ `Requantizer<int32_t,
/// int16_t>`): the int8 [`Requantizer`](crate::int8::Requantizer)'s
/// arithmetic with int16 bounds. [`QAdd`]'s output uses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requantizer32 {
    /// Q0.31 fixed-point multiplier, normally in `[2^30, 2^31)`.
    pub multiplier: i32,
    /// Power-of-two exponent: positive divides, negative multiplies.
    pub shift: i32,
    /// Output tensor zero point (TFLite: 0).
    pub zero_point: i16,
    /// Lowest output value.
    pub qmin: i16,
    /// Highest output value.
    pub qmax: i16,
}

impl Requantizer32 {
    /// Rescale `acc`, add the zero point and saturate to `[qmin, qmax]`.
    pub fn apply(&self, acc: i32) -> i16 {
        let scaled = if self.multiplier >= 0 && (2..=31).contains(&self.shift) {
            rounding_high_mul_divide_by_pot(acc, self.multiplier, self.shift)
        } else {
            let rail = 1 << 30;
            multiply_by_quantized_multiplier(acc, self.multiplier, self.shift).clamp(-rail, rail)
        };
        clamp16(scaled + i32::from(self.zero_point), self.qmin, self.qmax)
    }
}

/// `sum_i (x[i] - zp) * w[i]` in `i64`.
#[inline]
fn dot(x: &[i16], w: &[i8], zp: i32) -> i64 {
    let mut acc = 0i64;
    for (&v, &wi) in x.iter().zip(w) {
        acc = acc.wrapping_add(i64::from((i32::from(v) - zp) * i32::from(wi)));
    }
    acc
}

/// `biases[i]`, or 0 without biases.
#[inline]
fn bias_of<const N: usize>(biases: Option<&[i64; N]>, i: usize) -> i64 {
    biases.and_then(|b| b.get(i)).copied().unwrap_or(0)
}

/// 16x8 dense layer, one requantizer for every output (C++ `QDense` with
/// int16 input and output, int8 weights and an int64 accumulator).
///
/// ```
/// use tinymind::int16x8::{QDense, Requantizer};
///
/// static WEIGHTS: [[i8; 2]; 1] = [[3, -2]];
/// const RQ: Requantizer = Requantizer { multiplier: 1 << 30, shift: 0, zero_point: 0, qmin: -32768, qmax: 32767 };
/// let dense: QDense<'_, 2, 1> =
///     QDense { weights: &WEIGHTS, biases: Some(&[100]), input_zero_point: 0, requantizer: RQ };
/// assert_eq!(dense.forward(&[1000, -500]), [2050]); // (100 + 3000 + 1000) / 2
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QDense<'a, const IN: usize, const OUT: usize> {
    /// `weights[o][i]`: weight from input `i` to output `o`.
    pub weights: &'a [[i8; IN]; OUT],
    /// One `i64` bias per output, or `None`.
    pub biases: Option<&'a [i64; OUT]>,
    /// The input tensor's zero point (TFLite: 0).
    pub input_zero_point: i16,
    /// Rescales each accumulator to the output tensor's grid.
    pub requantizer: Requantizer,
}

impl<const IN: usize, const OUT: usize> QDense<'_, IN, OUT> {
    /// One forward pass.
    pub fn forward(&self, input: &[i16; IN]) -> [i16; OUT] {
        let zp = i32::from(self.input_zero_point);
        let mut out = [0i16; OUT];
        for (o, (y, w)) in out.iter_mut().zip(self.weights).enumerate() {
            *y = self
                .requantizer
                .apply(bias_of(self.biases, o).wrapping_add(dot(input, w, zp)));
        }
        out
    }
}

/// [`QDense`] with one requantizer per output, as TFLite quantizes dense
/// layers.
#[derive(Clone, Copy, Debug)]
pub struct QDensePerChannel<'a, const IN: usize, const OUT: usize> {
    /// `weights[o][i]`: weight from input `i` to output `o`.
    pub weights: &'a [[i8; IN]; OUT],
    /// One `i64` bias per output, or `None`.
    pub biases: Option<&'a [i64; OUT]>,
    /// The input tensor's zero point (TFLite: 0).
    pub input_zero_point: i16,
    /// `requantizers[o]` rescales output `o`.
    pub requantizers: &'a [Requantizer; OUT],
}

impl<const IN: usize, const OUT: usize> QDensePerChannel<'_, IN, OUT> {
    /// One forward pass.
    pub fn forward(&self, input: &[i16; IN]) -> [i16; OUT] {
        let zp = i32::from(self.input_zero_point);
        let mut out = [0i16; OUT];
        for (o, ((y, w), rq)) in out
            .iter_mut()
            .zip(self.weights)
            .zip(self.requantizers)
            .enumerate()
        {
            *y = rq.apply(bias_of(self.biases, o).wrapping_add(dot(input, w, zp)));
        }
        out
    }
}

/// The convolution shared by [`QConv2D`] and [`QConv2DPerChannel`]: filter
/// `f`'s accumulator, tap by tap, with padded taps (the zero point) adding
/// nothing.
#[allow(clippy::too_many_arguments)]
fn conv2d<
    const C: usize,
    const KH: usize,
    const KW: usize,
    const F: usize,
    const H: usize,
    const W: usize,
    const OH: usize,
    const OW: usize,
>(
    input: &[[[i16; C]; W]; H],
    output: &mut [[[i16; F]; OW]; OH],
    weights: &[[[[i8; C]; KW]; KH]; F],
    biases: Option<&[i64; F]>,
    zp: i16,
    stride: (usize, usize),
    pad: (usize, usize),
    requantize: impl Fn(usize, i64) -> i16,
) {
    let zp = i32::from(zp);
    for (oh, row) in output.iter_mut().enumerate() {
        for (ow, pixel) in row.iter_mut().enumerate() {
            for (f, (y, filter)) in pixel.iter_mut().zip(weights).enumerate() {
                let mut acc = bias_of(biases, f);
                for (kh, taps) in filter.iter().enumerate() {
                    for (kw, w) in taps.iter().enumerate() {
                        if let Some(x) =
                            tap(input, oh * stride.0 + kh, ow * stride.1 + kw, pad.0, pad.1)
                        {
                            acc = acc.wrapping_add(dot(x, w, zp));
                        }
                    }
                }
                *y = requantize(f, acc);
            }
        }
    }
}

macro_rules! conv_layer {
    ($(#[$doc:meta])* $name:ident, $rq_field:ident: $rq_ty:ty, $rq_doc:literal, |$s:ident, $f:ident| $rq:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name<
            'a,
            const C: usize,
            const KH: usize,
            const KW: usize,
            const F: usize,
            const SH: usize = 1,
            const SW: usize = 1,
            const PT: usize = 0,
            const PB: usize = 0,
            const PL: usize = 0,
            const PR: usize = 0,
        > {
            /// OHWI weights, `[F][KH][KW][C]`.
            pub weights: &'a [[[[i8; C]; KW]; KH]; F],
            /// One `i64` bias per filter, or `None`.
            pub biases: Option<&'a [i64; F]>,
            /// The input tensor's zero point (also the padding's value).
            pub input_zero_point: i16,
            #[doc = $rq_doc]
            pub $rq_field: $rq_ty,
        }

        impl<
            const C: usize,
            const KH: usize,
            const KW: usize,
            const F: usize,
            const SH: usize,
            const SW: usize,
            const PT: usize,
            const PB: usize,
            const PL: usize,
            const PR: usize,
        > $name<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR>
        {
            /// One forward pass: `input` is `[H][W][C]`, `output`
            /// `[OH][OW][F]`, with `OH = (H + PT + PB - KH) / SH + 1` and
            /// likewise `OW`, or the call does not compile.
            pub fn forward<const H: usize, const W: usize, const OH: usize, const OW: usize>(
                &self,
                input: &[[[i16; C]; W]; H],
                output: &mut [[[i16; F]; OW]; OH],
            ) {
                const {
                    assert!(
                        C > 0
                            && F > 0
                            && window_fits(H, KH, SH, PT, PB, OH)
                            && window_fits(W, KW, SW, PL, PR, OW),
                        "the output size does not match the input, kernel, strides and padding, or the layer is invalid"
                    )
                };
                let $s = self;
                conv2d(
                    input,
                    output,
                    self.weights,
                    self.biases,
                    self.input_zero_point,
                    (SH, SW),
                    (PT, PL),
                    |$f, acc| $rq.map_or(0, |rq: &Requantizer| rq.apply(acc)),
                );
            }
        }
    };
}

conv_layer!(
    /// 16x8 2D convolution, NHWC, one requantizer for every filter (C++
    /// `QConv2D` with int16 activations): the int8
    /// [`QConv2D`](crate::int8::QConv2D)'s shapes and conventions.
    QConv2D,
    requantizer: Requantizer,
    "Rescales each accumulator to the output tensor's grid.",
    |s, _f| Some(&s.requantizer)
);

conv_layer!(
    /// [`QConv2D`] with one requantizer per filter, as TFLite quantizes
    /// convolutions. A 1x1 kernel is a pointwise convolution.
    QConv2DPerChannel,
    requantizers: &'a [Requantizer; F],
    "`requantizers[f]` rescales filter `f`.",
    |s, f| s.requantizers.get(f)
);

/// 16x8 depthwise 2D convolution: one `KH` x `KW` filter per channel,
/// channels not mixed, one requantizer per channel (C++ `QDepthwiseConv2D`
/// with int16 activations): the int8
/// [`QDepthwiseConv2D`](crate::int8::QDepthwiseConv2D)'s shapes and
/// conventions.
#[derive(Clone, Copy, Debug)]
pub struct QDepthwiseConv2D<
    'a,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const SH: usize = 1,
    const SW: usize = 1,
    const PT: usize = 0,
    const PB: usize = 0,
    const PL: usize = 0,
    const PR: usize = 0,
> {
    /// `[C][KH][KW]` weights.
    pub weights: &'a [[[i8; KW]; KH]; C],
    /// One `i64` bias per channel, or `None`.
    pub biases: Option<&'a [i64; C]>,
    /// The input tensor's zero point (also the padding's value).
    pub input_zero_point: i16,
    /// `requantizers[c]` rescales channel `c`.
    pub requantizers: &'a [Requantizer; C],
}

impl<
    const C: usize,
    const KH: usize,
    const KW: usize,
    const SH: usize,
    const SW: usize,
    const PT: usize,
    const PB: usize,
    const PL: usize,
    const PR: usize,
> QDepthwiseConv2D<'_, C, KH, KW, SH, SW, PT, PB, PL, PR>
{
    /// One forward pass: `input` is `[H][W][C]`, `output` `[OH][OW][C]`, with
    /// `OH = (H + PT + PB - KH) / SH + 1` and likewise `OW`, or the call does
    /// not compile.
    pub fn forward<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i16; C]; W]; H],
        output: &mut [[[i16; C]; OW]; OH],
    ) {
        const {
            assert!(
                C > 0 && window_fits(H, KH, SH, PT, PB, OH) && window_fits(W, KW, SW, PL, PR, OW),
                "QDepthwiseConv2D: the output size does not match the input, kernel, strides and padding, or the layer is invalid"
            )
        };
        let zp = i32::from(self.input_zero_point);
        for (oh, row) in output.iter_mut().enumerate() {
            for (ow, pixel) in row.iter_mut().enumerate() {
                for (c, ((y, filter), rq)) in pixel
                    .iter_mut()
                    .zip(self.weights)
                    .zip(self.requantizers)
                    .enumerate()
                {
                    let mut acc = bias_of(self.biases, c);
                    for (kh, taps) in filter.iter().enumerate() {
                        for (kw, &w) in taps.iter().enumerate() {
                            if let Some(x) = tap(input, oh * SH + kh, ow * SW + kw, PT, PL) {
                                let x = x.get(c).map_or(zp, |&v| i32::from(v));
                                acc = acc.wrapping_add(i64::from((x - zp) * i32::from(w)));
                            }
                        }
                    }
                    *y = rq.apply(acc);
                }
            }
        }
    }
}

/// 16x8 max pooling over `PH` x `PW` windows (C++ `QMaxPool2D` with int16
/// storage): the int8 [`QMaxPool2D`](crate::int8::QMaxPool2D)'s shapes, in
/// the input's scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct QMaxPool2D<const PH: usize, const PW: usize, const SH: usize = PH, const SW: usize = PW>;

impl<const PH: usize, const PW: usize, const SH: usize, const SW: usize>
    QMaxPool2D<PH, PW, SH, SW>
{
    /// One forward pass: `output` is `[(H - PH) / SH + 1][(W - PW) / SW +
    /// 1][C]`, or the call does not compile.
    pub fn forward<
        const H: usize,
        const W: usize,
        const C: usize,
        const OH: usize,
        const OW: usize,
    >(
        &self,
        input: &[[[i16; C]; W]; H],
        output: &mut [[[i16; C]; OW]; OH],
    ) {
        const {
            assert!(
                window_fits(H, PH, SH, 0, 0, OH) && window_fits(W, PW, SW, 0, 0, OW),
                "QMaxPool2D: the output size does not match the input, window and strides"
            )
        };
        for (oh, row) in output.iter_mut().enumerate() {
            for (ow, pixel) in row.iter_mut().enumerate() {
                for (c, y) in pixel.iter_mut().enumerate() {
                    let mut best = i16::MIN;
                    for kh in 0..PH {
                        for kw in 0..PW {
                            if let Some(&v) =
                                tap(input, oh * SH + kh, ow * SW + kw, 0, 0).and_then(|x| x.get(c))
                            {
                                best = best.max(v);
                            }
                        }
                    }
                    *y = best;
                }
            }
        }
    }
}

/// Element-wise addition of two int16 tensors with their own scales and
/// zero points (C++ `QAdd` with int16 storage; TFLite's int16 ADD, whose
/// `left_shift` is 15): the int8 [`QAdd`](crate::int8::QAdd)'s arithmetic
/// with an [`Requantizer32`] onto the int16 output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QAdd {
    /// Zero point of the first input (TFLite: 0).
    pub input_a_zero_point: i16,
    /// Zero point of the second input (TFLite: 0).
    pub input_b_zero_point: i16,
    /// Left shift applied to each input less its zero point before
    /// rescaling (TFLite: 15). A negative one is treated as 0; a shifted
    /// value beyond `i32` saturates.
    pub left_shift: i32,
    /// Q0.31 multiplier rescaling the first input to the common scale.
    pub input_a_multiplier: i32,
    /// Shift for `input_a_multiplier` (positive divides).
    pub input_a_shift: i32,
    /// Q0.31 multiplier rescaling the second input to the common scale.
    pub input_b_multiplier: i32,
    /// Shift for `input_b_multiplier`.
    pub input_b_shift: i32,
    /// Rescales the `i32` sum to the output.
    pub output_requantizer: Requantizer32,
}

impl QAdd {
    /// `out = a + b` element by element; the three tensors have the same
    /// shape (`[i16; N]`, `[[i16; C]; W]` or `[[[i16; C]; W]; H]`).
    pub fn forward<T: Tensor>(&self, a: &T, b: &T, out: &mut T) {
        self.add_elements(
            out.as_flat_mut()
                .iter_mut()
                .zip(a.as_flat())
                .zip(b.as_flat())
                .map(|((o, &x), &y)| (o, x, y)),
        );
    }

    /// `a = a + b` element by element: [`forward`](Self::forward) with its
    /// result written over the first input, as tinymind (C++)'s generated
    /// models run an addition whose first input dies there. Each element is
    /// read before it is written, so the result equals `forward`'s.
    pub fn forward_in_place<T: Tensor>(&self, a: &mut T, b: &T) {
        self.add_elements(a.as_flat_mut().iter_mut().zip(b.as_flat()).map(|(o, &y)| {
            let x = *o;
            (o, x, y)
        }));
    }

    /// `b = a + b` element by element: [`forward`](Self::forward) with its
    /// result written over the second input.
    pub fn forward_in_place_b<T: Tensor>(&self, a: &T, b: &mut T) {
        self.add_elements(b.as_flat_mut().iter_mut().zip(a.as_flat()).map(|(o, &x)| {
            let y = *o;
            (o, x, y)
        }));
    }

    /// The loop over (output slot, a, b) triples all three forms share.
    fn add_elements<'o>(&self, elements: impl Iterator<Item = (&'o mut i16, i16, i16)>) {
        let shift = self.left_shift.max(0).unsigned_abs();
        let scale = |x: i16, zp: i16, m: i32, s: i32| {
            multiply_by_quantized_multiplier(
                saturating_shl(i32::from(x) - i32::from(zp), shift),
                m,
                s,
            )
        };
        for (o, x, y) in elements {
            let sa = scale(
                x,
                self.input_a_zero_point,
                self.input_a_multiplier,
                self.input_a_shift,
            );
            let sb = scale(
                y,
                self.input_b_zero_point,
                self.input_b_multiplier,
                self.input_b_shift,
            );
            *o = self.output_requantizer.apply(sa.wrapping_add(sb));
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An int16 tensor [`QAdd`] can add: `[i16; N]`, `[[i16; C]; W]` or
/// `[[[i16; C]; W]; H]`. Sealed.
pub trait Tensor: sealed::Sealed {
    /// The elements in memory order.
    fn as_flat(&self) -> &[i16];
    /// The elements in memory order, mutable.
    fn as_flat_mut(&mut self) -> &mut [i16];
}

impl<const N: usize> sealed::Sealed for [i16; N] {}
impl<const N: usize> Tensor for [i16; N] {
    fn as_flat(&self) -> &[i16] {
        self
    }
    fn as_flat_mut(&mut self) -> &mut [i16] {
        self
    }
}

impl<const W: usize, const C: usize> sealed::Sealed for [[i16; C]; W] {}
impl<const W: usize, const C: usize> Tensor for [[i16; C]; W] {
    fn as_flat(&self) -> &[i16] {
        self.as_flattened()
    }
    fn as_flat_mut(&mut self) -> &mut [i16] {
        self.as_flattened_mut()
    }
}

impl<const H: usize, const W: usize, const C: usize> sealed::Sealed for [[[i16; C]; W]; H] {}
impl<const H: usize, const W: usize, const C: usize> Tensor for [[[i16; C]; W]; H] {
    fn as_flat(&self) -> &[i16] {
        self.as_flattened().as_flattened()
    }
    fn as_flat_mut(&mut self) -> &mut [i16] {
        self.as_flattened_mut().as_flattened_mut()
    }
}

/// Entries in an int16 activation table: 512 base values and one more for
/// the slope of the last interval.
pub const LUT_SIZE: usize = 513;

/// Two's-complement wrap to `i16`, as TFLite's int16 arithmetic does.
#[inline]
fn wrap16(v: i32) -> i32 {
    i32::from(v as i16)
}

/// Look up an int16 input in a 513-entry table (C++ `qApplyLUT16`, TFLite's
/// `LUTLookup`): the top 9 bits of the input pick an interval, the low 7
/// interpolate along it, rounding half up, with TFLite's int16 wrap of the
/// slope. `calibration::build_sigmoid_lut` and `build_tanh_lut` build
/// tables.
///
/// ```
/// use tinymind::int16x8::{apply_lut, LUT_SIZE};
///
/// // The identity table: entry i holds the input at its knot, -32768 + 128 i.
/// let lut: [i16; LUT_SIZE] = core::array::from_fn(|i| (-32768 + 128 * i as i32).min(32767) as i16);
/// assert_eq!(apply_lut(1234, &lut), 1234);
/// ```
pub fn apply_lut(x: i16, lut: &[i16; LUT_SIZE]) -> i16 {
    let biased = (i32::from(x) + 32768) as u32;
    let index = (biased >> 7) as usize; // 0..=511
    let offset = (biased & 0x7F) as i32;
    let base = lut.get(index).copied().map_or(0, i32::from);
    let next = lut.get(index + 1).copied().map_or(0, i32::from);
    let slope = wrap16(next - base);
    let delta = (slope * offset + 64) >> 7;
    wrap16(base + delta) as i16
}

/// int16 softmax over a row, bit-exact with TFLite's int16 SOFTMAX
/// (`reference_ops::SoftmaxInt16`; C++ `QSoftmaxInt16`): int16 input with
/// zero point 0, int16 output on `[0, 1)` with scale 2^-15 and zero point 0.
/// For each element `d = x - max` is rescaled by (`input_multiplier`,
/// `input_shift`) so that -65535 maps to -10, looked up in TFLite's exp
/// table over `[-10, 0]`, and multiplied by the reciprocal of the row's sum
/// from TFLite's `1 / (1 + x)` table; both tables are constants of the
/// crate. `calibration::build_softmax` derives the two parameters.
///
/// ```
/// use tinymind::int16x8::QSoftmax;
///
/// // Input scale 10 / 65535 (beta 1): multiplier 2^30 with shift -1 is 1.0.
/// let softmax = QSoftmax { input_multiplier: 1 << 30, input_shift: -1 };
/// let mut out = [0i16; 2];
/// softmax.forward(&[0, 0], &mut out);
/// assert_eq!(out, [16384, 16384]); // one half each
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QSoftmax {
    /// Q0.31 multiplier of `input_scale * beta / (10 / 65535)`.
    pub input_multiplier: i32,
    /// Its shift (positive divides; TFLite's `input_left_shift` is
    /// `-input_shift`).
    pub input_shift: i32,
}

impl QSoftmax {
    /// The softmax of one row of `N` values into `output`, `N` in
    /// `1..65536` (checked at compile time).
    pub fn forward<const N: usize>(&self, input: &[i16; N], output: &mut [i16; N]) {
        const { assert!(N > 0 && N < 65536, "a row of 1 to 65535 values") };
        let max = input.iter().copied().max().map_or(0, i32::from);
        let mut sum = 0i32;
        for (e, &x) in output.iter_mut().zip(input) {
            let scaled = multiply_by_quantized_multiplier(
                i32::from(x) - max,
                self.input_multiplier,
                self.input_shift,
            );
            // x - max <= 0, so scaled <= 0 and adding 32767 cannot overflow.
            let centred = scaled.saturating_add(32767).max(-32768);
            *e = apply_lut(centred as i16, &softmax_tables::EXP);
            sum = sum.wrapping_add(i32::from(*e));
        }
        // The row's maximum contributes the table's top entry, 32767, so the
        // sum is at least that and headroom_plus_one is at most 17.
        let headroom_plus_one = (sum as u32).leading_zeros().clamp(1, 30);
        let shifted =
            (((sum as u32) << (headroom_plus_one - 1)).wrapping_add(1 << 13) >> 14) as i32;
        let centred_sum = (shifted - ((1 << 15) + (1 << 16))).clamp(-32768, 32767);
        let reciprocal = i32::from(apply_lut(centred_sum as i16, &softmax_tables::RECIPROCAL));
        let right_shift = 31 - headroom_plus_one;
        let round = 1i32 << (right_shift - 1);
        for y in output.iter_mut() {
            let v = (i32::from(*y) * reciprocal).wrapping_add(round) >> right_shift;
            *y = v.clamp(0, 32767) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both in-place additions equal `forward`.
    #[test]
    fn in_place_add_equals_forward() {
        let mut state = 11u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for _ in 0..16 {
            let add = QAdd {
                input_a_zero_point: (next() >> 24) as u8 as i8 as i16,
                input_b_zero_point: (next() >> 24) as u8 as i8 as i16,
                left_shift: 15,
                input_a_multiplier: (1 << 30) + (next() >> 2) as i32,
                input_a_shift: (next() % 4) as i32,
                input_b_multiplier: (1 << 30) + (next() >> 2) as i32,
                input_b_shift: (next() % 4) as i32,
                output_requantizer: Requantizer32 {
                    multiplier: (1 << 30) + (next() >> 2) as i32,
                    shift: (next() % 20) as i32,
                    zero_point: 0,
                    qmin: -32768,
                    qmax: 32767,
                },
            };
            let a: [[i16; 3]; 5] =
                core::array::from_fn(|_| core::array::from_fn(|_| (next() >> 16) as i16));
            let b: [[i16; 3]; 5] =
                core::array::from_fn(|_| core::array::from_fn(|_| (next() >> 16) as i16));
            let mut sum = [[0i16; 3]; 5];
            add.forward(&a, &b, &mut sum);
            let mut in_a = a;
            add.forward_in_place(&mut in_a, &b);
            assert_eq!(in_a, sum);
            let mut in_b = b;
            add.forward_in_place_b(&a, &mut in_b);
            assert_eq!(in_b, sum);
        }
    }
}
