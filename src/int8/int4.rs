//! Packed int4 weights, ported from tinymind (C++) `cpp/qint4.hpp`: two
//! signed 4-bit weights per byte, half the flash of int8 weights.
//! Re-exported from [`int8`](super).
//!
//! Packing is TFLite's and ONNX's: a tensor of `n` int4 values is stored in
//! `(n + 1) / 2` bytes, element `k` in the low nibble of byte `k / 2` when `k`
//! is even and in the high nibble when `k` is odd. A weight matrix is packed
//! as one tensor, so with an odd number of inputs every second row starts in
//! the middle of a byte.
//!
//! The layers compute exactly what their int8 counterparts compute with the
//! same weight values stored as `i8`: [`QDenseInt4`] what
//! [`QDensePerChannel`](super::QDensePerChannel) computes,
//! [`QConv2DInt4`] what [`QConv2DPerChannel`] computes and
//! [`QDepthwiseConv2DInt4`] what [`QDepthwiseConv2D`] computes. Stable Rust
//! cannot compute the packed length `(n + 1) / 2` from the layer's other
//! parameters, so each layer takes it as one more const parameter, `PACKED`,
//! checked at compile time: a wrong length does not compile.

use super::conv::{tap, window_fits};
use super::{QConv2DPerChannel, QDepthwiseConv2D, Requantizer};

/// The largest weight tensor, in weights, that [`QConv2DInt4`] and
/// [`QDepthwiseConv2DInt4`] unpack onto the stack to run the int8 kernels
/// (C++ `TINYMIND_INT4_CONV_UNPACK_MAX_BYTES`). A larger layer computes from
/// the packed weights directly, with the same results.
pub const INT4_UNPACK_MAX_WEIGHTS: usize = 1024;

/// Bytes needed for `n` packed int4 values (C++ `int4PackedBytes`).
pub const fn int4_packed_bytes(n: usize) -> usize {
    n.div_ceil(2)
}

/// A 4-bit field as a signed value in `-8..=7`.
#[inline]
const fn sign_extend(nibble: u8) -> i8 {
    ((nibble & 15) ^ 8) as i8 - 8
}

/// Element `k` of a packed tensor whose length is known to cover it.
#[inline]
fn nibble<const P: usize>(packed: &[u8; P], k: usize) -> i8 {
    let byte = packed.get(k / 2).copied().unwrap_or(0);
    sign_extend(if k & 1 == 0 { byte } else { byte >> 4 })
}

/// Element `index` of a packed int4 tensor (C++ `unpackInt4`), or `None`
/// past its end.
///
/// ```
/// use tinymind::int8::unpack_int4;
///
/// let packed = [0x8F, 0x07]; // -1 and -8, then 7
/// assert_eq!(unpack_int4(&packed, 0), Some(-1));
/// assert_eq!(unpack_int4(&packed, 1), Some(-8));
/// assert_eq!(unpack_int4(&packed, 2), Some(7));
/// assert_eq!(unpack_int4(&packed, 4), None);
/// ```
pub fn unpack_int4(packed: &[u8], index: usize) -> Option<i8> {
    let byte = *packed.get(index / 2)?;
    Some(sign_extend(if index & 1 == 0 { byte } else { byte >> 4 }))
}

/// Pack `N` values into `PACKED = (N + 1) / 2` bytes (C++ `packInt4`). A
/// trailing high nibble is 0; values outside `-8..=7` keep their low 4 bits.
///
/// ```
/// use tinymind::int8::{pack_int4, unpack_int4};
///
/// let packed: [u8; 2] = pack_int4(&[-1, -8, 7]);
/// assert_eq!(packed, [0x8F, 0x07]);
/// assert_eq!(unpack_int4(&packed, 2), Some(7));
/// ```
pub fn pack_int4<const N: usize, const PACKED: usize>(values: &[i8; N]) -> [u8; PACKED] {
    const { assert!(PACKED == N.div_ceil(2), "PACKED must be (N + 1) / 2") };
    let mut packed = [0u8; PACKED];
    for (b, pair) in packed.iter_mut().zip(values.chunks(2)) {
        let low = pair.first().map_or(0, |&v| v as u8 & 15);
        let high = pair.get(1).map_or(0, |&v| v as u8 & 15);
        *b = low | (high << 4);
    }
    packed
}

/// `sum (x[i] - zp) * w[first + i]` over a packed tensor, in wrapping `i32`.
#[inline]
fn dot_int4<const P: usize>(x: &[i8], packed: &[u8; P], first: usize, zp: i32) -> i32 {
    let mut acc = 0i32;
    for (i, &v) in x.iter().enumerate() {
        let w = i32::from(nibble(packed, first + i));
        acc = acc.wrapping_add((i32::from(v) - zp) * w);
    }
    acc
}

/// Dense layer over packed int4 weights, one [`Requantizer`] per output (C++
/// `QDenseInt4`; int4 weights need per-channel scales, so for a per-tensor
/// layer pass the same requantizer for every output):
///
/// ```text
/// out[o] = requantizers[o].apply(bias[o] + sum_i (x[i] - zp) * w[o][i])
/// ```
///
/// `weights` is the `[OUT][IN]` matrix packed as one tensor,
/// `PACKED = (IN * OUT + 1) / 2` bytes. The result equals
/// [`QDensePerChannel`](super::QDensePerChannel) with the same weights as
/// `i8`.
///
/// ```
/// use tinymind::int8::{pack_int4, QDenseInt4, Requantizer};
///
/// const RQ: Requantizer = Requantizer { multiplier: 1 << 30, shift: 1, zero_point: 0, qmin: -128, qmax: 127 };
/// // Two outputs over three inputs: rows [1, -2, 3] and [-8, 7, 0].
/// static WEIGHTS: [u8; 3] = [0xE1, 0x83, 0x07];
/// assert_eq!(WEIGHTS, pack_int4::<6, 3>(&[1, -2, 3, -8, 7, 0]));
/// let dense: QDenseInt4<'_, 3, 2, 3> =
///     QDenseInt4 { weights: &WEIGHTS, biases: None, input_zero_point: 0, requantizers: &[RQ; 2] };
/// // [1 - 4 + 9, -8 + 14 + 0] = [6, 6], times 2^30 / 2^31 * 2^-1 = 1/4: 1.5 rounds to 2.
/// assert_eq!(dense.forward(&[1, 2, 3]), [2, 2]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QDenseInt4<'a, const IN: usize, const OUT: usize, const PACKED: usize> {
    /// The `[OUT][IN]` weights, packed two per byte.
    pub weights: &'a [u8; PACKED],
    /// One `i32` bias per output, or `None`.
    pub biases: Option<&'a [i32; OUT]>,
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// `requantizers[o]` rescales output `o`.
    pub requantizers: &'a [Requantizer; OUT],
}

impl<const IN: usize, const OUT: usize, const PACKED: usize> QDenseInt4<'_, IN, OUT, PACKED> {
    /// One forward pass.
    pub fn forward(&self, input: &[i8; IN]) -> [i8; OUT] {
        const {
            assert!(IN > 0 && OUT > 0, "at least one input and one output");
            assert!(IN < 1 << 20, "too many inputs for an i32 row sum");
            assert!(
                PACKED == (IN * OUT).div_ceil(2),
                "PACKED must be (IN * OUT + 1) / 2"
            );
        };
        let zp = i32::from(self.input_zero_point);
        let mut out = [0i8; OUT];
        for (o, (y, rq)) in out.iter_mut().zip(self.requantizers).enumerate() {
            let bias = self.biases.and_then(|b| b.get(o)).copied().unwrap_or(0);
            *y = rq.apply(bias.wrapping_add(dot_int4(input, self.weights, o * IN, zp)));
        }
        out
    }
}

/// 2D convolution over packed int4 weights, one [`Requantizer`] per filter,
/// NHWC (C++ `QConv2DInt4`): [`QConv2DPerChannel`]'s parameters and results,
/// with the OHWI weight tensor `[F][KH][KW][C]` packed as one tensor,
/// `PACKED = (F * KH * KW * C + 1) / 2` bytes (so with an odd window every
/// second filter starts mid-byte). A 1x1 kernel is a pointwise convolution.
///
/// A layer of up to [`INT4_UNPACK_MAX_WEIGHTS`] weights unpacks them onto the
/// stack and runs [`QConv2DPerChannel`] over the `i8` copy; a larger one
/// computes from the packed weights, tap by tap.
#[derive(Clone, Copy, Debug)]
pub struct QConv2DInt4<
    'a,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const F: usize,
    const PACKED: usize,
    const SH: usize = 1,
    const SW: usize = 1,
    const PT: usize = 0,
    const PB: usize = 0,
    const PL: usize = 0,
    const PR: usize = 0,
> {
    /// The `[F][KH][KW][C]` weights, packed two per byte.
    pub weights: &'a [u8; PACKED],
    /// One `i32` bias per filter, or `None`.
    pub biases: Option<&'a [i32; F]>,
    /// The input tensor's zero point (also the padding's value).
    pub input_zero_point: i8,
    /// `requantizers[f]` rescales filter `f`.
    pub requantizers: &'a [Requantizer; F],
}

impl<
    const C: usize,
    const KH: usize,
    const KW: usize,
    const F: usize,
    const PACKED: usize,
    const SH: usize,
    const SW: usize,
    const PT: usize,
    const PB: usize,
    const PL: usize,
    const PR: usize,
> QConv2DInt4<'_, C, KH, KW, F, PACKED, SH, SW, PT, PB, PL, PR>
{
    /// Whether the weights are unpacked onto the stack ([`QConv2DInt4`]).
    pub const UNPACKS_WEIGHTS: bool = F * KH * KW * C <= INT4_UNPACK_MAX_WEIGHTS;

    /// One forward pass: `input` is `[H][W][C]`, `output` `[OH][OW][F]`, with
    /// `OH = (H + PT + PB - KH) / SH + 1` and likewise `OW`, or the call does
    /// not compile.
    pub fn forward<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; F]; OW]; OH],
    ) {
        const {
            assert!(
                C > 0
                    && F > 0
                    && window_fits(H, KH, SH, PT, PB, OH)
                    && window_fits(W, KW, SW, PL, PR, OW),
                "QConv2DInt4: the output size does not match the input, kernel, strides and padding, or the layer is invalid"
            );
            assert!(
                PACKED == (F * KH * KW * C).div_ceil(2),
                "PACKED must be (F * KH * KW * C + 1) / 2"
            );
            assert!(
                KH * KW * C < 1 << 20,
                "too many weights per filter for an i32 sum"
            );
        };
        if Self::UNPACKS_WEIGHTS {
            self.forward_unpacked(input, output);
        } else {
            self.forward_packed(input, output);
        }
    }

    /// Unpacks the weights onto the stack and runs the int8 layer.
    #[inline(never)]
    fn forward_unpacked<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; F]; OW]; OH],
    ) {
        let mut weights = [[[[0i8; C]; KW]; KH]; F];
        let mut k = 0;
        for filter in weights.iter_mut() {
            for row in filter.iter_mut() {
                for tap in row.iter_mut() {
                    for w in tap.iter_mut() {
                        *w = nibble(self.weights, k);
                        k += 1;
                    }
                }
            }
        }
        let layer: QConv2DPerChannel<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR> =
            QConv2DPerChannel {
                weights: &weights,
                biases: self.biases,
                input_zero_point: self.input_zero_point,
                requantizers: self.requantizers,
            };
        layer.forward(input, output);
    }

    /// Computes from the packed weights, tap by tap.
    fn forward_packed<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; F]; OW]; OH],
    ) {
        let zp = i32::from(self.input_zero_point);
        for (oh, row) in output.iter_mut().enumerate() {
            for (ow, pixel) in row.iter_mut().enumerate() {
                for (f, (y, rq)) in pixel.iter_mut().zip(self.requantizers).enumerate() {
                    let mut acc = self.biases.and_then(|b| b.get(f)).copied().unwrap_or(0);
                    let mut k = f * KH * KW * C;
                    for kh in 0..KH {
                        for kw in 0..KW {
                            match tap(input, oh * SH + kh, ow * SW + kw, PT, PL) {
                                // Padding holds the zero point: no contribution.
                                None => k += C,
                                Some(x) => {
                                    acc = acc.wrapping_add(dot_int4(x, self.weights, k, zp));
                                    k += C;
                                }
                            }
                        }
                    }
                    *y = rq.apply(acc);
                }
            }
        }
    }
}

/// Depthwise 2D convolution over packed int4 weights (C++
/// `QDepthwiseConv2DInt4`): [`QDepthwiseConv2D`]'s parameters and results,
/// with the `[C][KH][KW]` weight tensor packed as one tensor,
/// `PACKED = (C * KH * KW + 1) / 2` bytes.
///
/// A layer of up to [`INT4_UNPACK_MAX_WEIGHTS`] weights unpacks them onto the
/// stack and runs [`QDepthwiseConv2D`]; a larger one computes from the packed
/// weights, tap by tap.
#[derive(Clone, Copy, Debug)]
pub struct QDepthwiseConv2DInt4<
    'a,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const PACKED: usize,
    const SH: usize = 1,
    const SW: usize = 1,
    const PT: usize = 0,
    const PB: usize = 0,
    const PL: usize = 0,
    const PR: usize = 0,
> {
    /// The `[C][KH][KW]` weights, packed two per byte.
    pub weights: &'a [u8; PACKED],
    /// One `i32` bias per channel, or `None`.
    pub biases: Option<&'a [i32; C]>,
    /// The input tensor's zero point (also the padding's value).
    pub input_zero_point: i8,
    /// `requantizers[c]` rescales channel `c`.
    pub requantizers: &'a [Requantizer; C],
}

impl<
    const C: usize,
    const KH: usize,
    const KW: usize,
    const PACKED: usize,
    const SH: usize,
    const SW: usize,
    const PT: usize,
    const PB: usize,
    const PL: usize,
    const PR: usize,
> QDepthwiseConv2DInt4<'_, C, KH, KW, PACKED, SH, SW, PT, PB, PL, PR>
{
    /// Whether the weights are unpacked onto the stack
    /// ([`QDepthwiseConv2DInt4`]).
    pub const UNPACKS_WEIGHTS: bool = C * KH * KW <= INT4_UNPACK_MAX_WEIGHTS;

    /// One forward pass: `input` is `[H][W][C]`, `output` `[OH][OW][C]`, with
    /// `OH = (H + PT + PB - KH) / SH + 1` and likewise `OW`, or the call does
    /// not compile.
    pub fn forward<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
    ) {
        const {
            assert!(
                C > 0 && window_fits(H, KH, SH, PT, PB, OH) && window_fits(W, KW, SW, PL, PR, OW),
                "QDepthwiseConv2DInt4: the output size does not match the input, kernel, strides and padding, or the layer is invalid"
            );
            assert!(
                PACKED == (C * KH * KW).div_ceil(2),
                "PACKED must be (C * KH * KW + 1) / 2"
            );
        };
        if Self::UNPACKS_WEIGHTS {
            self.forward_unpacked(input, output);
        } else {
            self.forward_packed(input, output);
        }
    }

    /// Unpacks the weights onto the stack and runs the int8 layer.
    #[inline(never)]
    fn forward_unpacked<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
    ) {
        let mut weights = [[[0i8; KW]; KH]; C];
        let mut k = 0;
        for channel in weights.iter_mut() {
            for row in channel.iter_mut() {
                for w in row.iter_mut() {
                    *w = nibble(self.weights, k);
                    k += 1;
                }
            }
        }
        let layer: QDepthwiseConv2D<'_, C, KH, KW, SH, SW, PT, PB, PL, PR> = QDepthwiseConv2D {
            weights: &weights,
            biases: self.biases,
            input_zero_point: self.input_zero_point,
            requantizers: self.requantizers,
        };
        layer.forward(input, output);
    }

    /// Computes from the packed weights, tap by tap.
    fn forward_packed<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
    ) {
        let zp = i32::from(self.input_zero_point);
        for (oh, row) in output.iter_mut().enumerate() {
            for (ow, pixel) in row.iter_mut().enumerate() {
                for (c, (y, rq)) in pixel.iter_mut().zip(self.requantizers).enumerate() {
                    let mut acc = self.biases.and_then(|b| b.get(c)).copied().unwrap_or(0);
                    for kh in 0..KH {
                        for kw in 0..KW {
                            if let Some(x) = tap(input, oh * SH + kh, ow * SW + kw, PT, PL) {
                                let x = x.get(c).copied().map_or(0, i32::from);
                                let w = i32::from(nibble(self.weights, (c * KH + kh) * KW + kw));
                                acc = acc.wrapping_add((x - zp) * w);
                            }
                        }
                    }
                    *y = rq.apply(acc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibbles_round_trip() {
        let values: [i8; 16] = core::array::from_fn(|i| i as i8 - 8);
        let packed: [u8; 8] = pack_int4(&values);
        for (k, &v) in values.iter().enumerate() {
            assert_eq!(unpack_int4(&packed, k), Some(v));
        }
        // Odd length: the trailing high nibble is 0.
        let packed: [u8; 2] = pack_int4(&[3, -4, -5]);
        assert_eq!(packed[1] >> 4, 0);
        // Out-of-range values keep their low 4 bits.
        let packed: [u8; 1] = pack_int4(&[17, -9]);
        assert_eq!(
            (unpack_int4(&packed, 0), unpack_int4(&packed, 1)),
            (Some(1), Some(7))
        );
    }
}
