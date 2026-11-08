//! Int8 pooling over NHWC tensors, ported from tinymind (C++) `cpp/qpool2d.hpp`.
//! Re-exported from [`int8`](super).
//!
//! Input and output share one scale and zero point: the maximum of the
//! integers is the integer of the maximum, and the mean of `q` rounded to the
//! nearest integer is the mean of the real values on the same grid. No
//! requantizer is needed; follow with one to change the scale.

use super::conv::{tap, window_fits};

/// `numerator / denominator` rounded half away from zero, 0 for a zero
/// denominator (C++ `detail::roundedDivide`). Total: where the C++ would
/// overflow (a numerator within `denominator / 2` of the `i32` rails) the sum
/// wraps, and `i32::MIN / -1` gives 0.
pub(crate) fn rounded_divide(numerator: i32, denominator: i32) -> i32 {
    let half = denominator / 2;
    let biased = if numerator >= 0 {
        numerator.wrapping_add(half)
    } else {
        numerator.wrapping_sub(half)
    };
    biased.checked_div(denominator).unwrap_or(0)
}

/// Clamp to `[qmin, qmax]` with two separate tests, as the C++ does: with
/// `qmin > qmax` the result is `qmax` (and `clamp` would panic).
fn saturate(v: i32, qmin: i8, qmax: i8) -> i8 {
    let mut v = v;
    if v < i32::from(qmin) {
        v = i32::from(qmin);
    }
    if v > i32::from(qmax) {
        v = i32::from(qmax);
    }
    v as i8
}

/// Max pooling over `PH` x `PW` windows at strides `SH`, `SW` (default: the
/// window size, non-overlapping), no padding (C++ `QMaxPool2D`).
#[derive(Clone, Copy, Debug, Default)]
pub struct QMaxPool2D<const PH: usize, const PW: usize, const SH: usize = PH, const SW: usize = PW>;

impl<const PH: usize, const PW: usize, const SH: usize, const SW: usize>
    QMaxPool2D<PH, PW, SH, SW>
{
    /// One forward pass over an `H` x `W` x `C` input into an `OH` x `OW` x
    /// `C` output, with `OH = (H - PH) / SH + 1` and likewise `OW` (a compile
    /// error otherwise).
    pub fn forward<
        const H: usize,
        const W: usize,
        const C: usize,
        const OH: usize,
        const OW: usize,
    >(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
    ) {
        const {
            assert!(
                C > 0 && window_fits(H, PH, SH, 0, 0, OH) && window_fits(W, PW, SW, 0, 0, OW),
                "QMaxPool2D: the output size does not match the input, window and strides, or the layer is invalid"
            )
        };
        for (oh, out_row) in output.iter_mut().enumerate() {
            for (ow, out_pixel) in out_row.iter_mut().enumerate() {
                for (c, out) in out_pixel.iter_mut().enumerate() {
                    let mut max = i8::MIN;
                    for kh in 0..PH {
                        for kw in 0..PW {
                            let x = tap(input, oh * SH + kh, ow * SW + kw, 0, 0);
                            if let Some(&x) = x.and_then(|p| p.get(c)) {
                                max = max.max(x);
                            }
                        }
                    }
                    *out = max;
                }
            }
        }
    }
}

/// Average pooling over `PH` x `PW` windows at strides `SH`, `SW` (default: the
/// window size), no padding: the window's sum divided by its area, rounded
/// half away from zero, then clamped to `[qmin, qmax]` (C++ `QAvgPool2D`).
#[derive(Clone, Copy, Debug)]
pub struct QAvgPool2D<const PH: usize, const PW: usize, const SH: usize = PH, const SW: usize = PW>
{
    /// Lowest output value.
    pub qmin: i8,
    /// Highest output value.
    pub qmax: i8,
}

impl<const PH: usize, const PW: usize, const SH: usize, const SW: usize>
    QAvgPool2D<PH, PW, SH, SW>
{
    /// One forward pass; sizes as in [`QMaxPool2D::forward`].
    pub fn forward<
        const H: usize,
        const W: usize,
        const C: usize,
        const OH: usize,
        const OW: usize,
    >(
        &self,
        input: &[[[i8; C]; W]; H],
        output: &mut [[[i8; C]; OW]; OH],
    ) {
        const {
            assert!(
                C > 0 && window_fits(H, PH, SH, 0, 0, OH) && window_fits(W, PW, SW, 0, 0, OW),
                "QAvgPool2D: the output size does not match the input, window and strides, or the layer is invalid"
            )
        };
        // As the C++ casts the window area to int32_t.
        let area = const { PH.wrapping_mul(PW) as i32 };
        for (oh, out_row) in output.iter_mut().enumerate() {
            for (ow, out_pixel) in out_row.iter_mut().enumerate() {
                for (c, out) in out_pixel.iter_mut().enumerate() {
                    let mut sum = 0i32;
                    for kh in 0..PH {
                        for kw in 0..PW {
                            let x = tap(input, oh * SH + kh, ow * SW + kw, 0, 0);
                            if let Some(&x) = x.and_then(|p| p.get(c)) {
                                sum = sum.wrapping_add(i32::from(x));
                            }
                        }
                    }
                    *out = saturate(rounded_divide(sum, area), self.qmin, self.qmax);
                }
            }
        }
    }
}

/// Global average pooling: one value per channel, the mean over every pixel,
/// rounded and clamped as in [`QAvgPool2D`] (C++ `QGlobalAvgPool2D`). Its
/// output can feed a dense head in place of a large flatten-to-dense layer.
#[derive(Clone, Copy, Debug)]
pub struct QGlobalAvgPool2D {
    /// Lowest output value.
    pub qmin: i8,
    /// Highest output value.
    pub qmax: i8,
}

impl QGlobalAvgPool2D {
    /// One forward pass over an `H` x `W` x `C` input.
    pub fn forward<const H: usize, const W: usize, const C: usize>(
        &self,
        input: &[[[i8; C]; W]; H],
    ) -> [i8; C] {
        const {
            assert!(
                H > 0 && W > 0 && C > 0,
                "QGlobalAvgPool2D: every size must be above zero"
            )
        };
        let area = const { H.wrapping_mul(W) as i32 };
        // Channel by channel, as the C++ runs it, so no per-channel sums are
        // kept on the stack.
        let mut output = [0i8; C];
        for (c, out) in output.iter_mut().enumerate() {
            let sum = input
                .iter()
                .flatten()
                .filter_map(|pixel| pixel.get(c))
                .fold(0i32, |sum, &x| sum.wrapping_add(i32::from(x)));
            *out = saturate(rounded_divide(sum, area), self.qmin, self.qmax);
        }
        output
    }
}
