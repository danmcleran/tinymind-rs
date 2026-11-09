//! Int8 convolutions over NHWC tensors, ported from tinymind (C++)
//! `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp` and `cpp/qpointwiseconv2d.hpp`.
//! Re-exported from [`int8`](super); the conventions are in its module docs.

use super::{FastRequantizer, Requantizer, dot_with_zero_point};

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
    /// Any padding at all. Without it every window lies inside the input, so
    /// a layer needs no code for windows that reach into the padding.
    pub(crate) padded: bool,
}

// The bodies of `QConv2D` and `QConv2DPerChannel` (`conv2d_gather`,
// `conv2d_band`) take `rqs` yielding filter `f`'s requantizer as its `f`-th
// item (the per-tensor layer repeats one). Both compute `acc = bias + sum
// over the window of w * (x - zero_point)`; a padded tap holds the zero
// point, as TFLite pads, so it adds nothing. tinymind (C++) gathers patches,
// tiles rows and pairs filters for speed; those add the same products, and
// `i32` sums wrap the same in any order.

/// The convolution for layers with fewer than [`BAND_MIN_CHANNELS`] input
/// channels (a stem over one channel): per output pixel the window is gathered once
/// into `win`, the padding filled with the zero point, so every filter runs a
/// plain multiply-accumulate against it with the zero point folded into its
/// bias (a padded tap then adds `w * zp - w * zp = 0`).
/// Filters run in pairs, so each window value loaded serves two. Same `i32`
/// sums modulo 2^32.
#[allow(clippy::too_many_arguments)]
fn conv2d_gather<
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
    let folded: [i32; F] = core::array::from_fn(|f| {
        let taps = weights
            .get(f)
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .copied();
        folded_bias(bias_of(biases, f), zero_point, taps)
    });
    let requantize = |f: usize, acc: i32| rqs.clone().nth(f).map_or(0, |rq| rq.apply(acc));
    let mut win = [[[zero_point; C]; KW]; KH];
    for (oh, out_row) in output.iter_mut().enumerate() {
        let top = oh * window.stride_h;
        for (ow, out_pixel) in out_row.iter_mut().enumerate() {
            let left = ow * window.stride_w;
            for (kh, win_row) in win.iter_mut().enumerate() {
                for (kw, taps) in win_row.iter_mut().enumerate() {
                    *taps = *tap(input, top + kh, left + kw, window.pad_top, window.pad_left)
                        .unwrap_or(&[zero_point; C]);
                }
            }
            let x = win.as_flattened().as_flattened();
            let mut f = 0;
            while f + 1 < F {
                if let (Some(w0), Some(w1)) = (weights.get(f), weights.get(f + 1)) {
                    let (a0, a1) = dot2(
                        x,
                        w0.as_flattened().as_flattened(),
                        w1.as_flattened().as_flattened(),
                    );
                    let b0 = folded.get(f).copied().unwrap_or(0);
                    let b1 = folded.get(f + 1).copied().unwrap_or(0);
                    if let Some(out) = out_pixel.get_mut(f) {
                        *out = requantize(f, b0.wrapping_add(a0));
                    }
                    if let Some(out) = out_pixel.get_mut(f + 1) {
                        *out = requantize(f + 1, b1.wrapping_add(a1));
                    }
                }
                f += 2;
            }
            if let (Some(wf), Some(out), Some(&b)) =
                (weights.get(f), out_pixel.get_mut(f), folded.get(f))
            {
                let wf = wf.as_flattened().as_flattened();
                let (a, _) = dot2(x, wf, wf);
                *out = requantize(f, b.wrapping_add(a));
            }
        }
    }
}

/// Input channels from which [`QConv2D`] and [`QConv2DPerChannel`] take
/// [`conv2d_band`]. With fewer (a stem over one or a few channels) the window
/// is too short to pair, and reading each window once for all filters wins.
const BAND_MIN_CHANNELS: usize = 2;

/// `acc + sum x[i] * w[i]` over the shorter of the two rows, wrapping. Written
/// as pairs of products added to one accumulator: on Cortex-M cores with the
/// DSP extension (M4, M7, M33), LLVM's Arm parallel-DSP pass turns each pair
/// of adjacent `i16` products into one `SMLAD`, whose sum wraps like
/// `wrapping_add`.
#[inline(always)]
fn dot16(mut acc: i32, x: &[i16], w: &[i16]) -> i32 {
    let n = x.len().min(w.len());
    let mut i = 0;
    while i + 1 < n && i + 1 < x.len() && i + 1 < w.len() {
        let p0 = i32::from(x[i]) * i32::from(w[i]);
        let p1 = i32::from(x[i + 1]) * i32::from(w[i + 1]);
        acc = acc.wrapping_add(p0).wrapping_add(p1);
        i += 2;
    }
    if let (Some(&x), Some(&w)) = (x.get(i), w.get(i)) {
        if i < n {
            acc = acc.wrapping_add(i32::from(x) * i32::from(w));
        }
    }
    acc
}

/// The convolution for layers with several input channels, in 16 bits: per output
/// row, the `KH` input rows it reads are widened once into `band` as `x -
/// zero_point` (`i16`), and per filter its weights into `wide`; each window
/// inside the input is then `KH` dot products over contiguous `KW * C`
/// elements ([`dot16`]). Windows reaching into the padding add tap by tap.
/// Widening costs one pass over the rows and over the weights per output row,
/// small against the `OW * KH * KW * C` products per filter; the scratch is
/// `2 * KH * (W + KW) * C` bytes of stack. Same `i32` sums modulo 2^32.
#[allow(clippy::too_many_arguments)]
fn conv2d_band<
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
    let mut band = [[[0i16; C]; W]; KH];
    let mut valid = [false; KH];
    let mut wide = [[[0i16; C]; KW]; KH];
    let zp = i16::from(zero_point);
    for (oh, out_row) in output.iter_mut().enumerate() {
        let top = oh * window.stride_h;
        for (kh, (band_row, ok)) in band.iter_mut().zip(valid.iter_mut()).enumerate() {
            let row = (top + kh)
                .checked_sub(window.pad_top)
                .and_then(|r| input.get(r));
            *ok = row.is_some();
            if let Some(row) = row {
                for (b, &x) in band_row
                    .as_flattened_mut()
                    .iter_mut()
                    .zip(row.as_flattened())
                {
                    *b = i16::from(x) - zp;
                }
            }
        }
        for (f, (filter, rq)) in weights.iter().zip(rqs.clone()).enumerate() {
            for (wide_row, filter_row) in wide.iter_mut().zip(filter) {
                for (w16, &w) in wide_row
                    .as_flattened_mut()
                    .iter_mut()
                    .zip(filter_row.as_flattened())
                {
                    *w16 = i16::from(w);
                }
            }
            let rows = BandRows {
                band: &band,
                valid: &valid,
                wide: &wide,
                bias: bias_of(biases, f),
                window,
            };
            match FastRequantizer::new(rq) {
                Some(fast) => rows.run(out_row, f, |acc| fast.apply(acc)),
                None => rows.run_general(out_row, f, rq),
            }
        }
    }
}

/// One output row of one filter in [`conv2d_band`].
struct BandRows<'t, const W: usize, const C: usize, const KH: usize, const KW: usize> {
    band: &'t [[[i16; C]; W]; KH],
    valid: &'t [bool; KH],
    wide: &'t [[[i16; C]; KW]; KH],
    bias: i32,
    window: Window,
}

impl<const W: usize, const C: usize, const KH: usize, const KW: usize> BandRows<'_, W, C, KH, KW> {
    /// The row for a requantizer without a prepared form, out of line and
    /// tap by tap (the same sums as `run`, in less code): rare, so small.
    #[cold]
    #[inline(never)]
    fn run_general<const F: usize, const OW: usize>(
        &self,
        out_row: &mut [[i8; F]; OW],
        f: usize,
        rq: &Requantizer,
    ) {
        for (ow, pixel) in out_row.iter_mut().enumerate() {
            let acc = self.edge(ow * self.window.stride_w);
            if let Some(out) = pixel.get_mut(f) {
                *out = rq.apply(acc);
            }
        }
    }

    /// `bias + sum over the window` at padded column `left`, tap by tap: for
    /// windows reaching into the padding.
    #[inline(always)]
    fn edge(&self, left: usize) -> i32 {
        let mut acc = self.bias;
        for ((band_row, &ok), wide_row) in self.band.iter().zip(self.valid).zip(self.wide) {
            if !ok {
                continue;
            }
            for (kw, taps) in wide_row.iter().enumerate() {
                let x = (left + kw)
                    .checked_sub(self.window.pad_left)
                    .and_then(|c| band_row.get(c));
                if let Some(x) = x {
                    acc = dot16(acc, x, taps);
                }
            }
        }
        acc
    }

    #[inline(always)]
    fn run<const F: usize, const OW: usize>(
        &self,
        out_row: &mut [[i8; F]; OW],
        f: usize,
        requantize: impl Fn(i32) -> i8,
    ) {
        let pad_left = self.window.pad_left;
        for (ow, pixel) in out_row.iter_mut().enumerate() {
            let left = ow * self.window.stride_w;
            let mut acc = self.bias;
            let interior = left
                .checked_sub(pad_left)
                .and_then(|c| self.band[0].get(c..c + KW).map(|_| c));
            if let Some(c0) = interior {
                for ((band_row, &ok), wide_row) in self.band.iter().zip(self.valid).zip(self.wide) {
                    if let (true, Some(xs)) = (ok, band_row.get(c0..c0 + KW)) {
                        acc = dot16(acc, xs.as_flattened(), wide_row.as_flattened());
                    }
                }
            } else {
                acc = self.edge(left);
            }
            if let Some(out) = pixel.get_mut(f) {
                *out = requantize(acc);
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
            padded: PT + PB + PL + PR != 0,
        };
        let rqs = core::iter::repeat(&self.requantizer);
        if C >= BAND_MIN_CHANNELS {
            conv2d_band(
                input,
                output,
                self.weights,
                self.biases,
                self.input_zero_point,
                rqs,
                window,
            );
        } else {
            conv2d_gather(
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
            padded: PT + PB + PL + PR != 0,
        };
        let rqs = self.requantizers.iter();
        if C >= BAND_MIN_CHANNELS {
            conv2d_band(
                input,
                output,
                self.weights,
                self.biases,
                self.input_zero_point,
                rqs,
                window,
            );
        } else {
            conv2d_gather(
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
    /// Channel by channel, as tinymind (C++) runs it: each channel's taps, its
    /// bias folded with the zero point (`bias - zp * sum w`) and its prepared
    /// requantizer stay put across the pixels. The `i32` sums are the same
    /// integers as the per-tap form, modulo 2^32.
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
        let window = Window {
            stride_h: SH,
            stride_w: SW,
            pad_top: PT,
            pad_left: PL,
            padded: PT + PB + PL + PR != 0,
        };
        let zero_point = self.input_zero_point;
        let channels = self.weights.iter().zip(self.requantizers);
        for (c, (taps, rq)) in channels.enumerate() {
            let taps = *taps;
            let bias = bias_of(self.biases, c);
            let folded = folded_bias(bias, zero_point, taps.iter().flatten().copied());
            let k = DepthwiseChannel {
                c,
                taps: &taps,
                bias,
                folded,
                zero_point,
                window,
            };
            // The pixel loop once per requantizer form, as the C++ writes it.
            match FastRequantizer::new(rq) {
                Some(fast) => k.run(input, output, |acc| fast.apply(acc)),
                None => k.run_general(input, output, *rq),
            }
        }
    }
}

/// One depthwise channel: its taps, its bias as given and folded with the
/// zero point (see [`folded_bias`]).
struct DepthwiseChannel<'t, const KH: usize, const KW: usize> {
    c: usize,
    taps: &'t [[i8; KW]; KH],
    bias: i32,
    folded: i32,
    zero_point: i8,
    window: Window,
}

impl<const KH: usize, const KW: usize> DepthwiseChannel<'_, KH, KW> {
    /// [`run`](Self::run) for a requantizer without a prepared form (a
    /// scale ratio of 0.25 or more, a negative multiplier): rare, so kept out
    /// of line and cold, which keeps the common loop's code from doubling.
    #[cold]
    #[inline(never)]
    fn run_general<
        const H: usize,
        const W: usize,
        const C: usize,
        const OH: usize,
        const OW: usize,
    >(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
        rq: Requantizer,
    ) {
        self.run(input, output, |acc| rq.apply(acc));
    }

    /// Every output pixel of this channel. A window wholly inside the input
    /// reads its rows and pixels as slices and runs a plain multiply-accumulate
    /// on the folded bias; one that reaches into the padding takes the
    /// general form, skipping the padded taps.
    #[inline(always)]
    fn run<const H: usize, const W: usize, const C: usize, const OH: usize, const OW: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
        requantize: impl Fn(i32) -> i8,
    ) {
        let Window {
            stride_h,
            stride_w,
            pad_top,
            pad_left,
            padded,
        } = self.window;
        let c = self.c;
        for (oh, out_row) in output.iter_mut().enumerate() {
            let top = oh * stride_h;
            for (ow, out_pixel) in out_row.iter_mut().enumerate() {
                let left = ow * stride_w;
                let rows = top.checked_sub(pad_top).and_then(|r| input.get(r..r + KH));
                let column = left.checked_sub(pad_left).filter(|&col| col + KW <= W);
                let acc = match (rows, column) {
                    (Some(rows), Some(col)) => {
                        let mut acc = self.folded;
                        for (row, tap_row) in rows.iter().zip(self.taps) {
                            if let Some(pixels) = row.get(col..col + KW) {
                                for (pixel, &w) in pixels.iter().zip(tap_row) {
                                    if let Some(&x) = pixel.get(c) {
                                        acc = acc.wrapping_add(i32::from(w) * i32::from(x));
                                    }
                                }
                            }
                        }
                        acc
                    }
                    // Unreachable without padding (every window lies inside
                    // the input); this arm keeps the general path out of an
                    // unpadded layer's code.
                    _ if !padded => self.folded,
                    _ => {
                        let zp = i32::from(self.zero_point);
                        let mut acc = self.bias;
                        for (kh, tap_row) in self.taps.iter().enumerate() {
                            for (kw, &w) in tap_row.iter().enumerate() {
                                let pixel = tap(input, top + kh, left + kw, pad_top, pad_left);
                                if let Some(&x) = pixel.and_then(|p| p.get(c)) {
                                    acc = acc.wrapping_add(i32::from(w) * (i32::from(x) - zp));
                                }
                            }
                        }
                        acc
                    }
                };
                if let Some(out) = out_pixel.get_mut(c) {
                    *out = requantize(acc);
                }
            }
        }
    }
}

/// `sum w * x` in `i32`, wrapping. With the zero point folded into the bias
/// (see [`folded_bias`]) this is the whole per-output dot product.
#[inline]
fn dot<const N: usize>(x: &[i8; N], w: &[i8; N]) -> i32 {
    x.iter().zip(w).fold(0i32, |acc, (&x, &w)| {
        acc.wrapping_add(i32::from(w) * i32::from(x))
    })
}

/// `bias - zero_point * sum w`: since `sum w * (x - zp) == sum w * x - zp *
/// sum w`, a filter whose taps all read the input (no padding) can run a plain
/// multiply-accumulate. `i32` arithmetic wraps, so the two forms are equal
/// modulo 2^32, as the C++'s are.
#[inline]
fn folded_bias(bias: i32, zero_point: i8, weights: impl IntoIterator<Item = i8>) -> i32 {
    let sum = weights
        .into_iter()
        .fold(0i32, |s, w| s.wrapping_add(i32::from(w)));
    bias.wrapping_sub(i32::from(zero_point).wrapping_mul(sum))
}

/// The body of [`QPointwiseConv2D`] and [`QPointwiseConv2DPerChannel`]: a
/// dense layer applied to every pixel. Filter by filter, as the C++ runs it:
/// each filter's weights, folded bias and requantizer stay in registers or on
/// the stack across the pixels.
fn pointwise<'r, const H: usize, const W: usize, const C: usize, const F: usize>(
    input: &[[[i8; C]; W]; H],
    output: &mut [[[i8; F]; W]; H],
    weights: &[[i8; C]; F],
    biases: Option<&[i32; F]>,
    zero_point: i8,
    rqs: impl Iterator<Item = &'r Requantizer>,
) {
    if H * W == 1 {
        // A single pixel (a dense layer): one pass, output by output, so the
        // folded bias is not worth its sum of the weights.
        let pixel = output
            .iter_mut()
            .flatten()
            .next()
            .zip(input.iter().flatten().next());
        if let Some((out_pixel, in_pixel)) = pixel {
            let filters = out_pixel.iter_mut().zip(weights).zip(rqs);
            for (f, ((out, row), rq)) in filters.enumerate() {
                let acc =
                    bias_of(biases, f).wrapping_add(dot_with_zero_point(in_pixel, row, zero_point));
                *out = rq.apply(acc);
            }
        }
        return;
    }
    for (f, (row, rq)) in weights.iter().zip(rqs).enumerate() {
        let w = *row;
        let bias = folded_bias(bias_of(biases, f), zero_point, w);
        // The pixel loop once per requantizer form, so neither tests the form
        // per output.
        match FastRequantizer::new(rq) {
            Some(fast) => pointwise_filter(input, output, f, &w, bias, |acc| fast.apply(acc)),
            None => pointwise_filter_general(input, output, f, &w, bias, *rq),
        }
    }
}

/// [`pointwise_filter`] for a requantizer without a prepared form: rare, so
/// kept out of line and cold (see [`DepthwiseChannel::run_general`]).
#[cold]
#[inline(never)]
fn pointwise_filter_general<const H: usize, const W: usize, const C: usize, const F: usize>(
    input: &[[[i8; C]; W]; H],
    output: &mut [[[i8; F]; W]; H],
    f: usize,
    w: &[i8; C],
    bias: i32,
    rq: Requantizer,
) {
    pointwise_filter(input, output, f, w, bias, |acc| rq.apply(acc));
}

/// One pointwise filter over every pixel, two pixels at a time: each weight
/// loaded serves both (four would spill registers on a Cortex-M4). Pixels are
/// taken from the flattened rows; an odd last pixel runs alone.
#[inline(always)]
fn pointwise_filter<const H: usize, const W: usize, const C: usize, const F: usize>(
    input: &[[[i8; C]; W]; H],
    output: &mut [[[i8; F]; W]; H],
    f: usize,
    w: &[i8; C],
    bias: i32,
    requantize: impl Fn(i32) -> i8,
) {
    let inputs = input.as_flattened();
    let outputs = output.as_flattened_mut();
    let mut ins = inputs.chunks_exact(2);
    let mut outs = outputs.chunks_exact_mut(2);
    for (out_pair, in_pair) in (&mut outs).zip(&mut ins) {
        if let ([o0, o1], [x0, x1]) = (out_pair, in_pair) {
            let (a0, a1) = dot2_pixels(x0, x1, w);
            if let Some(out) = o0.get_mut(f) {
                *out = requantize(bias.wrapping_add(a0));
            }
            if let Some(out) = o1.get_mut(f) {
                *out = requantize(bias.wrapping_add(a1));
            }
        }
    }
    for (out_pixel, in_pixel) in outs.into_remainder().iter_mut().zip(ins.remainder()) {
        if let Some(out) = out_pixel.get_mut(f) {
            *out = requantize(bias.wrapping_add(dot(in_pixel, w)));
        }
    }
}

/// `(sum x0 * w, sum x1 * w)`, wrapping: two pixels against one filter.
#[inline(always)]
fn dot2_pixels<const N: usize>(x0: &[i8; N], x1: &[i8; N], w: &[i8; N]) -> (i32, i32) {
    x0.iter()
        .zip(x1)
        .zip(w)
        .fold((0i32, 0i32), |(a0, a1), ((&x0, &x1), &w)| {
            let w = i32::from(w);
            (
                a0.wrapping_add(w * i32::from(x0)),
                a1.wrapping_add(w * i32::from(x1)),
            )
        })
}

/// `(sum x * w0, sum x * w1)` over the shortest of the three, wrapping: one
/// window against two filters (an odd last filter passes itself twice).
#[inline(always)]
fn dot2(x: &[i8], w0: &[i8], w1: &[i8]) -> (i32, i32) {
    x.iter()
        .zip(w0)
        .zip(w1)
        .fold((0i32, 0i32), |(a0, a1), ((&x, &w0), &w1)| {
            let x = i32::from(x);
            (
                a0.wrapping_add(x * i32::from(w0)),
                a1.wrapping_add(x * i32::from(w1)),
            )
        })
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
