//! Int8 convolutions over NHWC tensors, ported from tinymind (C++)
//! `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp` and `cpp/qpointwiseconv2d.hpp`.
//! Re-exported from [`int8`](super); the conventions are in its module docs.

use super::{Requantizer, dot_with_zero_point};

/// Whether `out` is the size a window of `kernel` at `stride` produces over
/// `input` padded by `before` and `after`, `(input + before + after - kernel)
/// / stride + 1`, on a valid layer: stride and kernel above zero, kernel no
/// larger than the padded input. Every layer checks it at compile time.
pub(crate) const fn window_fits(
    input: usize,
    kernel: usize,
    stride: usize,
    before: usize,
    after: usize,
    out: usize,
) -> bool {
    let Some(padded) = input.checked_add(before) else {
        return false;
    };
    let Some(padded) = padded.checked_add(after) else {
        return false;
    };
    stride > 0 && kernel > 0 && padded >= kernel && (padded - kernel) / stride + 1 == out
}

/// The input element at padded position (`row`, `column`), or `None` in the
/// padding.
#[inline]
pub(crate) fn tap<T, const W: usize, const H: usize>(
    input: &[[T; W]; H],
    row: usize,
    column: usize,
    pad_top: usize,
    pad_left: usize,
) -> Option<&T> {
    let r = row.checked_sub(pad_top)?;
    let c = column.checked_sub(pad_left)?;
    input.get(r)?.get(c)
}

/// `biases[i]`, or 0 without biases.
#[inline]
fn bias_of<const N: usize>(biases: Option<&[i32; N]>, i: usize) -> i32 {
    biases.map_or(0, |b| b.get(i).copied().unwrap_or(0))
}

/// A window's strides and its padding before the first row and column.
#[derive(Clone, Copy)]
pub(crate) struct Window {
    pub(crate) stride_h: usize,
    pub(crate) stride_w: usize,
    pub(crate) pad_top: usize,
    pub(crate) pad_left: usize,
}

/// The body of [`QConv2D`] and [`QConv2DPerChannel`]: `rqs` yields filter
/// `f`'s requantizer as its `f`-th item (the per-tensor layer repeats one).
///
/// `acc = bias + sum over the window of w * (x - zero_point)`. A padded tap
/// holds the zero point, as TFLite pads, so it adds nothing and is skipped.
/// tinymind (C++) gathers patches, tiles rows and pairs filters for speed;
/// those add the same products, and `i32` sums wrap the same in any order.
fn conv2d<
    'r,
    const H: usize,
    const W: usize,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const F: usize,
    const OH: usize,
    const OW: usize,
>(
    input: &[[[i8; C]; W]; H],
    output: &mut [[[i8; F]; OW]; OH],
    weights: &[[[[i8; C]; KW]; KH]; F],
    biases: Option<&[i32; F]>,
    zero_point: i8,
    rqs: impl Iterator<Item = &'r Requantizer> + Clone,
    window: Window,
) {
    for (oh, out_row) in output.iter_mut().enumerate() {
        let top = oh * window.stride_h;
        for (ow, out_pixel) in out_row.iter_mut().enumerate() {
            let left = ow * window.stride_w;
            let filters = out_pixel.iter_mut().zip(weights).zip(rqs.clone());
            for (f, ((out, filter), rq)) in filters.enumerate() {
                let mut acc = bias_of(biases, f);
                for (kh, filter_row) in filter.iter().enumerate() {
                    for (kw, taps) in filter_row.iter().enumerate() {
                        let x = tap(input, top + kh, left + kw, window.pad_top, window.pad_left);
                        if let Some(x) = x {
                            acc = acc.wrapping_add(dot_with_zero_point(x, taps, zero_point));
                        }
                    }
                }
                *out = rq.apply(acc);
            }
        }
    }
}

/// A quantized 2D convolution with one requantizer for the whole output (C++
/// `QConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F, PT,
/// PB, PL, PR>`).
///
/// `C` input channels, a `KH` x `KW` kernel, `F` filters, strides `SH` and
/// `SW` (default 1) and padding `PT`, `PB`, `PL`, `PR` rows and columns
/// (default 0, VALID). Weights are `weights[filter][row][column][channel]`
/// (OHWI, as TFLite stores them), symmetric int8; biases are in the
/// accumulator's scale. The layer borrows them, so they can live in flash.
///
/// ```
/// use tinymind::int8::{QConv2D, Requantizer};
///
/// // Two 2x2 filters over a 3x3 single-channel input: a 2x2x2 output.
/// static WEIGHTS: [[[[i8; 1]; 2]; 2]; 2] = [[[[1], [1]], [[1], [1]]], [[[1], [0]], [[0], [-1]]]];
/// let conv: QConv2D<'_, 1, 2, 2, 2> = QConv2D {
///     weights: &WEIGHTS,
///     biases: None,
///     input_zero_point: 0,
///     requantizer: Requantizer { multiplier: 1 << 30, shift: -1, zero_point: 0, qmin: -128, qmax: 127 },
/// };
/// let input = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]];
/// let mut output = [[[0i8; 2]; 2]; 2];
/// conv.forward(&input, &mut output);
/// // The sum of each window, and its top-left minus bottom-right.
/// assert_eq!(output, [[[12, -4], [16, -4]], [[24, -4], [28, -4]]]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QConv2D<
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
    /// `weights[f][kh][kw][c]`.
    pub weights: &'a [[[[i8; C]; KW]; KH]; F],
    /// One `i32` bias per filter, or `None`.
    pub biases: Option<&'a [i32; F]>,
    /// The input tensor's zero point, which also fills the padding.
    pub input_zero_point: i8,
    /// Rescales every accumulator to the output tensor's grid.
    pub requantizer: Requantizer,
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
> QConv2D<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR>
{
    /// One forward pass over an `H` x `W` x `C` input into an `OH` x `OW` x
    /// `F` output. `OH` and `OW` must be what the kernel, strides and padding
    /// produce (a compile error otherwise).
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
                "QConv2D: the output size does not match the input, kernel, strides and padding, or the layer is invalid"
            )
        };
        let window = Window {
            stride_h: SH,
            stride_w: SW,
            pad_top: PT,
            pad_left: PL,
        };
        let rqs = core::iter::repeat(&self.requantizer);
        conv2d(
            input,
            output,
            self.weights,
            self.biases,
            self.input_zero_point,
            rqs,
            window,
        );
    }
}

/// [`QConv2D`] with one requantizer per filter, the per-channel weight
/// quantization TFLite applies to convolutions (C++ `QConv2DPerChannel`).
#[derive(Clone, Copy, Debug)]
pub struct QConv2DPerChannel<
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
    /// `weights[f][kh][kw][c]`.
    pub weights: &'a [[[[i8; C]; KW]; KH]; F],
    /// One `i32` bias per filter, or `None`.
    pub biases: Option<&'a [i32; F]>,
    /// The input tensor's zero point, which also fills the padding.
    pub input_zero_point: i8,
    /// `requantizers[f]` rescales filter `f`'s outputs.
    pub requantizers: &'a [Requantizer; F],
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
> QConv2DPerChannel<'_, C, KH, KW, F, SH, SW, PT, PB, PL, PR>
{
    /// One forward pass; see [`QConv2D::forward`].
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
                "QConv2DPerChannel: the output size does not match the input, kernel, strides and padding, or the layer is invalid"
            )
        };
        let window = Window {
            stride_h: SH,
            stride_w: SW,
            pad_top: PT,
            pad_left: PL,
        };
        let rqs = self.requantizers.iter();
        conv2d(
            input,
            output,
            self.weights,
            self.biases,
            self.input_zero_point,
            rqs,
            window,
        );
    }
}

/// A quantized depthwise 2D convolution: one `KH` x `KW` filter per channel,
/// channels never mixed, one requantizer per channel as TFLite requires (C++
/// `QDepthwiseConv2D`).
///
/// Weights are `weights[channel][row][column]`; strides and padding as in
/// [`QConv2D`]. The output has the input's `C` channels.
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
    /// `weights[c][kh][kw]`.
    pub weights: &'a [[[i8; KW]; KH]; C],
    /// One `i32` bias per channel, or `None`.
    pub biases: Option<&'a [i32; C]>,
    /// The input tensor's zero point, which also fills the padding.
    pub input_zero_point: i8,
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
    /// One forward pass over an `H` x `W` x `C` input into an `OH` x `OW` x
    /// `C` output; `OH` and `OW` as in [`QConv2D::forward`].
    ///
    /// tinymind (C++) folds the zero point into the bias (`bias - zp * sum w`)
    /// and runs channel by channel; the `i32` sums are the same integers.
    pub fn forward<const H: usize, const W: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
    ) {
        const {
            assert!(
                C > 0 && window_fits(H, KH, SH, PT, PB, OH) && window_fits(W, KW, SW, PL, PR, OW),
                "QDepthwiseConv2D: the output size does not match the input, kernel, strides and padding, or the layer is invalid"
            )
        };
        let zero_point = i32::from(self.input_zero_point);
        for (oh, out_row) in output.iter_mut().enumerate() {
            let top = oh * SH;
            for (ow, out_pixel) in out_row.iter_mut().enumerate() {
                let left = ow * SW;
                let channels = out_pixel
                    .iter_mut()
                    .zip(self.weights)
                    .zip(self.requantizers);
                for (c, ((out, filter), rq)) in channels.enumerate() {
                    let mut acc = bias_of(self.biases, c);
                    for (kh, filter_row) in filter.iter().enumerate() {
                        for (kw, &w) in filter_row.iter().enumerate() {
                            let pixel = tap(input, top + kh, left + kw, PT, PL);
                            if let Some(&x) = pixel.and_then(|p| p.get(c)) {
                                acc = acc.wrapping_add(i32::from(w) * (i32::from(x) - zero_point));
                            }
                        }
                    }
                    *out = rq.apply(acc);
                }
            }
        }
    }
}

/// The body of [`QPointwiseConv2D`] and [`QPointwiseConv2DPerChannel`]: a
/// dense layer applied to every pixel.
fn pointwise<'r, const H: usize, const W: usize, const C: usize, const F: usize>(
    input: &[[[i8; C]; W]; H],
    output: &mut [[[i8; F]; W]; H],
    weights: &[[i8; C]; F],
    biases: Option<&[i32; F]>,
    zero_point: i8,
    rqs: impl Iterator<Item = &'r Requantizer> + Clone,
) {
    let pixels = output.iter_mut().flatten().zip(input.iter().flatten());
    for (out_pixel, in_pixel) in pixels {
        let filters = out_pixel.iter_mut().zip(weights).zip(rqs.clone());
        for (f, ((out, row), rq)) in filters.enumerate() {
            let acc =
                bias_of(biases, f).wrapping_add(dot_with_zero_point(in_pixel, row, zero_point));
            *out = rq.apply(acc);
        }
    }
}

/// A quantized 1x1 ("pointwise") convolution, the channel-mixing half of a
/// depthwise-separable block, with one requantizer for the whole output (C++
/// `QPointwiseConv2D`). Every pixel goes through the same `C`-to-`F` dense
/// layer; weights are `weights[filter][channel]`.
#[derive(Clone, Copy, Debug)]
pub struct QPointwiseConv2D<'a, const C: usize, const F: usize> {
    /// `weights[f][c]`.
    pub weights: &'a [[i8; C]; F],
    /// One `i32` bias per filter, or `None`.
    pub biases: Option<&'a [i32; F]>,
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// Rescales every accumulator to the output tensor's grid.
    pub requantizer: Requantizer,
}

impl<const C: usize, const F: usize> QPointwiseConv2D<'_, C, F> {
    /// One forward pass over an `H` x `W` x `C` input into an `H` x `W` x `F`
    /// output.
    pub fn forward<const H: usize, const W: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; F]; W]; H],
    ) {
        const {
            assert!(
                H > 0 && W > 0 && C > 0 && F > 0,
                "QPointwiseConv2D: every size must be above zero"
            )
        };
        let rqs = core::iter::repeat(&self.requantizer);
        pointwise(
            input,
            output,
            self.weights,
            self.biases,
            self.input_zero_point,
            rqs,
        );
    }
}

/// [`QPointwiseConv2D`] with one requantizer per filter (C++
/// `QPointwiseConv2DPerChannel`). For a single pixel this is
/// [`QDensePerChannel`](super::QDensePerChannel).
#[derive(Clone, Copy, Debug)]
pub struct QPointwiseConv2DPerChannel<'a, const C: usize, const F: usize> {
    /// `weights[f][c]`.
    pub weights: &'a [[i8; C]; F],
    /// One `i32` bias per filter, or `None`.
    pub biases: Option<&'a [i32; F]>,
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// `requantizers[f]` rescales filter `f`'s outputs.
    pub requantizers: &'a [Requantizer; F],
}

impl<const C: usize, const F: usize> QPointwiseConv2DPerChannel<'_, C, F> {
    /// One forward pass; see [`QPointwiseConv2D::forward`].
    pub fn forward<const H: usize, const W: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; F]; W]; H],
    ) {
        const {
            assert!(
                H > 0 && W > 0 && C > 0 && F > 0,
                "QPointwiseConv2DPerChannel: every size must be above zero"
            )
        };
        let rqs = self.requantizers.iter();
        pointwise(
            input,
            output,
            self.weights,
            self.biases,
            self.input_zero_point,
            rqs,
        );
    }
}
