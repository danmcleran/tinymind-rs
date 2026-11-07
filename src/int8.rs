//! Int8 affine-quantized inference, ported from tinymind `cpp/qaffine.hpp`,
//! `cpp/qdense.hpp` and `cpp/qactivations.hpp`.
//!
//! This is the TFLite / CMSIS-NN quantization model, separate from the Q-format
//! [`QValue`](crate::QValue) pipeline: a real value is `scale * (q - zero_point)`,
//! with int8 activations (asymmetric, any zero point), int8 weights (symmetric,
//! zero point 0) and int32 biases and accumulators. Everything here is pure
//! integer arithmetic at run time; turning float scales into the integer
//! parameters is host-side work, in `calibration` (`std` feature).
//!
//! A layer computes `acc[o] = bias[o] + sum_i weight[o][i] * (input[i] - input_zero_point)`
//! in `i32`, then a [`Requantizer`] rescales `acc` to the output tensor's int8 grid
//! with a fixed-point multiplier, adds the output zero point and saturates.
//!
//! Results match tinymind's C++ (and therefore TFLite's reference kernels, which
//! tinymind matches) bit for bit. Where the C++ would hit undefined behaviour
//! (an `i32` accumulator overflow, a shift out of its documented range), this
//! port wraps or clamps instead, so every function is total and none panics.

#[cfg(feature = "std")]
pub mod calibration;

/// Saturating rounding doubling high multiply: `(a * b + 2^30) >> 31`, the
/// Q0.31 multiply of gemmlowp and TFLite, with `i32::MIN * i32::MIN` saturated
/// to `i32::MAX` (C++ `saturatingRoundingDoublingHighMul`).
pub fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    let overflow = a == i32::MIN && b == i32::MIN;
    // |a * b| <= 2^62, so adding 2^30 cannot overflow i64.
    let ab = i64::from(a) * i64::from(b);
    let high = ((ab + (1i64 << 30)) >> 31) as i32;
    if overflow { i32::MAX } else { high }
}

/// `x / 2^exponent`, rounded to nearest with gemmlowp's tie rule (C++
/// `roundingDivideByPOT`). An `exponent` of zero or less returns `x`; the
/// documented range ends at 31, and larger exponents are treated as 31.
pub fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent <= 0 {
        return x;
    }
    let exponent = exponent.min(31) as u32;
    let mask = ((1u32 << exponent) - 1) as i32;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

/// `rounding_divide_by_pot(saturating_rounding_doubling_high_mul(x, multiplier),
/// exponent)` in one 64-bit rounding (C++ `roundingHighMulDivideByPOT`). Only
/// valid for `multiplier >= 0` and `2 <= exponent <= 31`, which
/// [`Requantizer::apply`] checks before calling it; the two forms then agree bit
/// for bit (a Kani proof checks that).
pub(crate) fn rounding_high_mul_divide_by_pot(x: i32, multiplier: i32, exponent: i32) -> i32 {
    // nudge = 2^(30+e) + 2^30 - [x < 0] * 2^31, assembled from 32-bit halves.
    let negative = (x as u32) >> 31;
    let nudge_high = (1u32 << (exponent - 2)).wrapping_sub(negative);
    let nudge_low = (negative << 31) | 0x4000_0000;
    let nudge = (u64::from(nudge_high) << 32) | u64::from(nudge_low);
    // The sum is formed modulo 2^64; its true value fits i64.
    let v = nudge.wrapping_add((i64::from(x) * i64::from(multiplier)) as u64);
    ((v >> 32) as u32 as i32) >> (exponent - 1)
}

/// Rescale an `i32` by a Q0.31 `multiplier` and a power-of-two `shift`, without
/// zero point or saturation to storage (gemmlowp's "MultiplyByQuantizedMultiplier",
/// C++ `multiplyByQuantizedMultiplier`). A negative `shift` left-shifts `x` first,
/// saturating to the `i32` range; a positive one divides the product with
/// rounding.
pub fn multiply_by_quantized_multiplier(x: i32, multiplier: i32, shift: i32) -> i32 {
    let left_shift = if shift < 0 { shift.unsigned_abs() } else { 0 };
    let right_shift = shift.max(0);

    let shifted = if left_shift > 30 {
        // Any non-zero x saturates; 2^31 itself is not representable.
        if x > 0 {
            i32::MAX
        } else if x < 0 {
            i32::MIN
        } else {
            0
        }
    } else if left_shift > 0 {
        // x * 2^left_shift fits i32 exactly when x lies in
        // [-2^(31 - left_shift), 2^(31 - left_shift) - 1].
        let limit = i32::MAX >> left_shift;
        if x > limit {
            i32::MAX
        } else if x < -limit - 1 {
            i32::MIN
        } else {
            x * (1 << left_shift)
        }
    } else {
        x
    };

    rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(shifted, multiplier),
        right_shift,
    )
}

/// Rescales an `i32` accumulator to int8 storage (C++ `Requantizer<int32_t,
/// int8_t>`).
///
/// `multiplier` (Q0.31) and `shift` encode the ratio
/// `input_scale * weight_scale / output_scale`, and a negative `shift` means the
/// ratio is above one. `zero_point` is the output tensor's, and the result
/// saturates to `[qmin, qmax]` (usually `[-128, 127]`; raise `qmin` to the zero
/// point to fuse a ReLU). Build one from float scales with
/// `calibration::build_requantizer` on the host (`std` feature), or write the
/// five integers out as constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requantizer {
    /// Q0.31 fixed-point multiplier, normally in `[2^30, 2^31)`.
    pub multiplier: i32,
    /// Power-of-two exponent: positive divides, negative multiplies.
    pub shift: i32,
    /// Output tensor zero point.
    pub zero_point: i8,
    /// Lowest output value.
    pub qmin: i8,
    /// Highest output value.
    pub qmax: i8,
}

impl Requantizer {
    /// Rescale `acc`, add the zero point and saturate to `[qmin, qmax]`.
    pub fn apply(&self, acc: i32) -> i8 {
        let scaled = if self.multiplier >= 0 && (2..=31).contains(&self.shift) {
            // The common case (a scale ratio below 0.25): one rounding, equal to
            // the two-step form below bit for bit.
            rounding_high_mul_divide_by_pot(acc, self.multiplier, self.shift)
        } else {
            // This path can reach the i32 rails, where adding the zero point
            // would overflow; clamping to +-2^30 first gives the same final
            // result for int8 storage.
            let rail = 1 << 30;
            multiply_by_quantized_multiplier(acc, self.multiplier, self.shift).clamp(-rail, rail)
        };
        let mut with_zp = scaled + i32::from(self.zero_point);
        // Two separate tests, as in the C++: with qmin > qmax the result is
        // qmax (and `clamp` would panic).
        if with_zp < i32::from(self.qmin) {
            with_zp = i32::from(self.qmin);
        }
        if with_zp > i32::from(self.qmax) {
            with_zp = i32::from(self.qmax);
        }
        with_zp as i8
    }
}

/// `sum_i w[i] * (x[i] - zero_point)` in `i32` (C++
/// `dotProductWithZeroPoint`, scalar path). The sum wraps on overflow, which
/// needs vectors far longer than any layer here.
fn dot_with_zero_point<const N: usize>(x: &[i8; N], w: &[i8; N], zero_point: i8) -> i32 {
    x.iter().zip(w).fold(0i32, |acc, (&x, &w)| {
        acc.wrapping_add(i32::from(w) * (i32::from(x) - i32::from(zero_point)))
    })
}

/// A quantized fully connected layer with `IN` int8 inputs and `OUT` int8
/// outputs (C++ `QDense<int8_t, int8_t, int32_t, int8_t, IN, OUT>`).
///
/// The layer borrows its weights and biases, so they can live in flash as
/// `static` or `const` data. Weights are `weights[output][input]` (row-major,
/// as PyTorch and TFLite store them), symmetric int8. Biases are in the
/// accumulator's scale (`input_scale * weight_scale`); `None` means no bias.
#[derive(Clone, Copy, Debug)]
pub struct QDense<'a, const IN: usize, const OUT: usize> {
    /// `weights[o][i]`: weight from input `i` to output `o`.
    pub weights: &'a [[i8; IN]; OUT],
    /// One `i32` bias per output, or `None`.
    pub biases: Option<&'a [i32; OUT]>,
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// Rescales each accumulator to the output tensor's grid.
    pub requantizer: Requantizer,
}

impl<const IN: usize, const OUT: usize> QDense<'_, IN, OUT> {
    /// One forward pass.
    pub fn forward(&self, input: &[i8; IN]) -> [i8; OUT] {
        let mut output = [0i8; OUT];
        for (o, (out, row)) in output.iter_mut().zip(self.weights).enumerate() {
            let bias = self.biases.map_or(0, |b| b.get(o).copied().unwrap_or(0));
            let acc = bias.wrapping_add(dot_with_zero_point(input, row, self.input_zero_point));
            *out = self.requantizer.apply(acc);
        }
        output
    }
}

/// Quantized ReLU: clamp at the tensor's zero point, which represents 0.0
/// (C++ `qrelu`).
pub fn qrelu(x: i8, zero_point: i8) -> i8 {
    if x < zero_point { zero_point } else { x }
}

/// Quantized ReLU6: clamp to `[zero_point, q_six]`, where `q_six` is 6.0 in the
/// tensor's grid (C++ `qrelu6`).
pub fn qrelu6(x: i8, zero_point: i8, q_six: i8) -> i8 {
    if x < zero_point {
        zero_point
    } else if x > q_six {
        q_six
    } else {
        x
    }
}

/// Entries in an int8 activation lookup table: one per int8 input.
pub const LUT_SIZE: usize = 256;

/// Index of `x` in an activation lookup table: -128 maps to 0, 0 to 128 and 127
/// to 255 (C++ `qActivationLUTIndex`).
pub fn lut_index(x: i8) -> usize {
    usize::from((x as u8).wrapping_add(128))
}

/// Apply a 256-entry int8 lookup table, such as a sigmoid or tanh table built by
/// `calibration::build_sigmoid_lut` (C++ `qApplyLUT`).
pub fn apply_lut(x: i8, lut: &[i8; LUT_SIZE]) -> i8 {
    lut[lut_index(x)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// About 1.0 in Q0.31.
    const Q31_ONE: i32 = i32::MAX;

    fn i8_requant(multiplier: i32, shift: i32, zero_point: i8) -> Requantizer {
        Requantizer {
            multiplier,
            shift,
            zero_point,
            qmin: -128,
            qmax: 127,
        }
    }

    // Known answers from tinymind unit_test/quantization.
    #[test]
    fn high_mul_saturates_min_squared() {
        assert_eq!(
            saturating_rounding_doubling_high_mul(i32::MIN, i32::MIN),
            i32::MAX
        );
    }

    #[test]
    fn rounding_divide_by_pot_cases() {
        assert_eq!(rounding_divide_by_pot(1234, 0), 1234);
        assert_eq!(rounding_divide_by_pot(-1234, 0), -1234);
        assert_eq!(rounding_divide_by_pot(1234, -3), 1234);
        for (x, expected) in [
            (8, 1),
            (11, 1),
            (12, 2),
            (15, 2),
            (-8, -1),
            (-12, -2),
            (-11, -1),
            (-13, -2),
        ] {
            assert_eq!(rounding_divide_by_pot(x, 3), expected, "{x} / 8");
        }
    }

    #[test]
    fn requantizer_known_answers() {
        let identity = i8_requant(Q31_ONE, 0, 0);
        assert_eq!(
            [identity.apply(0), identity.apply(50), identity.apply(-50)],
            [0, 50, -50]
        );
        assert_eq!([identity.apply(1000), identity.apply(-1000)], [127, -128]);
        let halve = i8_requant(Q31_ONE, 1, 0);
        assert_eq!(
            [halve.apply(100), halve.apply(-100), halve.apply(0)],
            [50, -50, 0]
        );
        let double = i8_requant(Q31_ONE, -1, 0);
        assert_eq!([double.apply(30), double.apply(-30)], [60, -60]);
        let zp = i8_requant(Q31_ONE, 0, 10);
        assert_eq!(
            [
                zp.apply(0),
                zp.apply(50),
                zp.apply(-50),
                zp.apply(200),
                zp.apply(-200)
            ],
            [10, 60, -40, 127, -128]
        );
    }

    /// tinymind's `requantizer_saturating_path_at_int32_rails`, int8 part: the
    /// saturating path's +-2^30 clamp gives the exact result, the rescaled value
    /// plus the zero point clamped in i64.
    #[test]
    fn requantizer_saturating_path_at_rails() {
        let accs = [
            i32::MIN,
            i32::MIN + 1,
            -(1 << 30) - 1,
            -(1 << 30),
            -(1 << 30) + 1,
            -70000,
            -1,
            0,
            1,
            70000,
            (1 << 30) - 1,
            1 << 30,
            (1 << 30) + 1,
            i32::MAX - 1,
            i32::MAX,
        ];
        let multipliers = [0, 1, 1 << 30, Q31_ONE, -1, i32::MIN];
        let zero_points: [i8; 5] = [-128, -127, 0, 126, 127];
        let bounds: [[i8; 2]; 4] = [[-128, 127], [0, 127], [-128, 0], [0, 0]];
        let mut cases = 0;
        for shift in -31..=1 {
            for multiplier in multipliers {
                for zero_point in zero_points {
                    for [qmin, qmax] in bounds {
                        let rq = Requantizer {
                            multiplier,
                            shift,
                            zero_point,
                            qmin,
                            qmax,
                        };
                        for acc in accs {
                            let sum =
                                i64::from(multiply_by_quantized_multiplier(acc, multiplier, shift))
                                    + i64::from(zero_point);
                            let expected = sum.clamp(i64::from(qmin), i64::from(qmax));
                            assert_eq!(i64::from(rq.apply(acc)), expected);
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 10_000);
    }

    #[test]
    fn dense_known_answers() {
        let unit = i8_requant(Q31_ONE, 0, 0);
        let w1 = [[1, 1]];
        let layer = QDense {
            weights: &w1,
            biases: Some(&[0]),
            input_zero_point: 0,
            requantizer: unit,
        };
        assert_eq!(layer.forward(&[3, 5]), [8]);
        let w2 = [[2, 1], [-1, 3]];
        let layer = QDense {
            weights: &w2,
            biases: Some(&[0, 0]),
            input_zero_point: 0,
            requantizer: unit,
        };
        assert_eq!(layer.forward(&[1, 2]), [4, 5]);
        let layer = QDense {
            weights: &w1,
            biases: Some(&[10]),
            input_zero_point: 0,
            requantizer: unit,
        };
        assert_eq!(layer.forward(&[3, 5]), [18]);
        let w3 = [[2]];
        let layer = QDense {
            weights: &w3,
            biases: None,
            input_zero_point: 0,
            requantizer: unit,
        };
        assert_eq!(layer.forward(&[4]), [8]);
    }

    #[test]
    fn activations() {
        assert_eq!([qrelu(-5, 0), qrelu(5, 0), qrelu(-128, -10)], [0, 5, -10]);
        assert_eq!(
            [qrelu6(-5, 0, 60), qrelu6(30, 0, 60), qrelu6(100, 0, 60)],
            [0, 30, 60]
        );
        assert_eq!(
            [lut_index(-128), lut_index(0), lut_index(127)],
            [0, 128, 255]
        );
        let mut lut = [0i8; LUT_SIZE];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = (i as i32 - 128) as i8;
        }
        assert_eq!(apply_lut(-7, &lut), -7);
    }
}
