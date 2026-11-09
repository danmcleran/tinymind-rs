//! Int8 composition layers, ported from tinymind (C++) `cpp/qmul.hpp`,
//! `cpp/qpad.hpp` and `cpp/qconcat.hpp`: elementwise multiplication, explicit
//! padding and channel concatenation. Re-exported from [`int8`](super).

use super::{FastRequantizer, Requantizer, Tensor};

/// Elementwise multiplication of two int8 tensors with their own zero points
/// (C++ `QMul`, TFLite MUL): squeeze-and-excite gating in MobileNet-V3 and
/// EfficientNet. `out = requantize((a - zp_a) * (b - zp_b))`, where the
/// requantizer encodes `scale_a * scale_b / scale_out`
/// (`calibration::build_mul_requantizer`).
///
/// ```
/// use tinymind::int8::{QMul, Requantizer};
///
/// // Scales 1/4, 1/4 and 1/16: the product of the real values, in quarters
/// // times quarters, is exact on the output grid (ratio 1, shift -1 then 1/2).
/// let mul = QMul {
///     input_a_zero_point: 0,
///     input_b_zero_point: 2,
///     requantizer: Requantizer { multiplier: 1 << 30, shift: -1, zero_point: 0, qmin: -128, qmax: 127 },
/// };
/// let a = [[4i8, -8], [3, 0]];
/// let b = [[6i8, 4], [2, 9]];
/// let mut y = [[0i8; 2]; 2];
/// mul.forward(&a, &b, &mut y);
/// assert_eq!(y, [[16, -16], [0, 0]]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QMul {
    /// Zero point of the first input.
    pub input_a_zero_point: i8,
    /// Zero point of the second input.
    pub input_b_zero_point: i8,
    /// Rescales each product to the output: `scale_a * scale_b / scale_out`,
    /// the output zero point and the saturation range.
    pub requantizer: Requantizer,
}

impl QMul {
    /// `out = a * b` element by element. The three tensors have the same type,
    /// so the same shape (see [`Tensor`]).
    pub fn forward<T: Tensor>(&self, a: &T, b: &T, out: &mut T) {
        let (a, b, out) = (a.as_flat(), b.as_flat(), out.as_flat_mut());
        let (za, zb) = (
            i32::from(self.input_a_zero_point),
            i32::from(self.input_b_zero_point),
        );
        // |a - zp_a| * |b - zp_b| <= 255 * 255: the product fits an i32.
        let products = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (i32::from(x) - za) * (i32::from(y) - zb));
        match FastRequantizer::new(&self.requantizer) {
            Some(rq) => {
                for (o, p) in out.iter_mut().zip(products) {
                    *o = rq.apply(p);
                }
            }
            None => {
                for (o, p) in out.iter_mut().zip(products) {
                    *o = self.requantizer.apply(p);
                }
            }
        }
    }
}

/// `a + b + c == d`, without overflowing.
const fn sums_to(a: usize, b: usize, c: usize, d: usize) -> bool {
    match a.checked_add(b) {
        Some(ab) => match ab.checked_add(c) {
            Some(abc) => abc == d,
            None => false,
        },
        None => false,
    }
}

/// Explicit spatial padding of an NHWC tensor with a constant (C++ `QPad2D`,
/// TFLite PAD): `PT` rows above, `PB` below, `PL` columns on the left and
/// `PR` on the right, each holding `pad_value` in every channel. The
/// convolution layers pad by themselves; this is for models with a separate
/// PAD operator. The output must be `H + PT + PB` by `W + PL + PR`, or the call
/// does not compile.
///
/// ```
/// use tinymind::int8::QPad2D;
///
/// let input = [[[1i8, 2]; 2]; 1];
/// let mut padded = [[[0i8; 2]; 3]; 2]; // one row below, one column on the left
/// QPad2D::<0, 1, 1, 0> { pad_value: -7 }.forward(&input, &mut padded);
/// assert_eq!(padded, [[[-7, -7], [1, 2], [1, 2]], [[-7, -7]; 3]]);
/// ```
///
/// A padded size that does not add up does not compile:
///
/// ```compile_fail
/// # use tinymind::int8::QPad2D;
/// let input = [[[1i8, 2]; 2]; 1];
/// let mut padded = [[[0i8; 2]; 4]; 2]; // 4 columns: 2 + 1 + 0 is 3
/// QPad2D::<0, 1, 1, 0> { pad_value: -7 }.forward(&input, &mut padded);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QPad2D<const PT: usize, const PB: usize, const PL: usize, const PR: usize> {
    /// The value every padded element holds: the tensor's zero point for a
    /// real 0.0, as TFLite pads.
    pub pad_value: i8,
}

impl<const PT: usize, const PB: usize, const PL: usize, const PR: usize> QPad2D<PT, PB, PL, PR> {
    /// Pad an `H` x `W` x `C` input into an `OH` x `OW` x `C` output.
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
                sums_to(H, PT, PB, OH) && sums_to(W, PL, PR, OW),
                "QPad2D: the output size is not the input size plus the padding"
            )
        };
        for (oh, out_row) in output.iter_mut().enumerate() {
            let row = oh.checked_sub(PT).and_then(|r| input.get(r));
            for (ow, pixel) in out_row.iter_mut().enumerate() {
                let x = row.and_then(|r| ow.checked_sub(PL).and_then(|c| r.get(c)));
                *pixel = x.copied().unwrap_or([self.pad_value; C]);
            }
        }
    }
}

/// Concatenation of two NHWC tensors along the channel axis (C++
/// `QConcat2_2D`, TFLite CONCATENATION on the last axis): the output's
/// channels are the first input's, then the second's, each rescaled onto the
/// output grid by its own requantizer (`calibration::build_rescaler`). The
/// output's channel count must be `CA + CB`, or the call does not compile.
///
/// ```
/// use tinymind::int8::{QConcat, Requantizer};
///
/// // Both inputs already on the output grid: ratio 1 (multiplier 2^30, shift -1).
/// let same = Requantizer { multiplier: 1 << 30, shift: -1, zero_point: 0, qmin: -128, qmax: 127 };
/// let concat = QConcat { input_a_zero_point: 0, input_b_zero_point: 0, requantizer_a: same, requantizer_b: same };
/// let a = [[[1i8]; 2]; 1];
/// let b = [[[2i8, 3]; 2]; 1];
/// let mut y = [[[0i8; 3]; 2]; 1];
/// concat.forward(&a, &b, &mut y);
/// assert_eq!(y, [[[1, 2, 3]; 2]]);
/// ```
///
/// An output whose channels are not the two inputs' together does not compile:
///
/// ```compile_fail
/// # use tinymind::int8::{QConcat, Requantizer};
/// # let same = Requantizer { multiplier: 1 << 30, shift: -1, zero_point: 0, qmin: -128, qmax: 127 };
/// # let concat = QConcat { input_a_zero_point: 0, input_b_zero_point: 0, requantizer_a: same, requantizer_b: same };
/// let a = [[[1i8]; 2]; 1];
/// let b = [[[2i8, 3]; 2]; 1];
/// let mut y = [[[0i8; 4]; 2]; 1]; // 4 channels: 1 + 2 is 3
/// concat.forward(&a, &b, &mut y);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QConcat {
    /// Zero point of the first input.
    pub input_a_zero_point: i8,
    /// Zero point of the second input.
    pub input_b_zero_point: i8,
    /// Rescales the first input to the output: `scale_a / scale_out`.
    pub requantizer_a: Requantizer,
    /// Rescales the second input to the output: `scale_b / scale_out`.
    pub requantizer_b: Requantizer,
}

impl QConcat {
    /// Concatenate `a` (`CA` channels) and `b` (`CB` channels) into `out`
    /// (`C = CA + CB` channels), pixel by pixel.
    pub fn forward<
        const H: usize,
        const W: usize,
        const CA: usize,
        const CB: usize,
        const C: usize,
    >(
        &self,
        a: &[[[i8; CA]; W]; H],
        b: &[[[i8; CB]; W]; H],
        out: &mut [[[i8; C]; W]; H],
    ) {
        const {
            assert!(
                CA > 0 && CB > 0 && sums_to(CA, CB, 0, C),
                "QConcat: the output's channels must be the two inputs' channels together"
            )
        };
        let (za, zb) = (
            i32::from(self.input_a_zero_point),
            i32::from(self.input_b_zero_point),
        );
        let pixels = out
            .as_flattened_mut()
            .iter_mut()
            .zip(a.as_flattened())
            .zip(b.as_flattened());
        for ((y, xa), xb) in pixels {
            let (ya, yb) = y.split_at_mut(CA.min(C));
            for (o, &x) in ya.iter_mut().zip(xa) {
                *o = self.requantizer_a.apply(i32::from(x) - za);
            }
            for (o, &x) in yb.iter_mut().zip(xb) {
                *o = self.requantizer_b.apply(i32::from(x) - zb);
            }
        }
    }
}
