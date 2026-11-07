//! Host-side calibration: turning float scales into the integer parameters the
//! int8 layers use (C++ `quantizeMultiplier`, `buildRequantizer`, `quantize`,
//! `dequantize`, `buildQSigmoidLUT`, `buildQTanhLUT`,
//! `buildQSoftmaxFixedPointParams`).
//!
//! Needs the `std` feature, for `f32::exp` and `f32::tanh`; tinymind gates the
//! same functions on `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`. On a device,
//! run these on the host and embed the results as constants. The float
//! arithmetic follows the C++ step for step, so the tables and requantizers match
//! the C++ ones on the same platform: `exp` and `tanh` come from the system math
//! library in both.

use super::{LUT_SIZE, QSoftmax, Requantizer};

/// C `frexp`: `x = mantissa * 2^exponent` with `|mantissa|` in `[0.5, 1)`.
/// Zero, infinities and NaN come back unchanged with exponent 0.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale into the normal range first.
        let (m, e) = frexp(x * f64::from_bits(0x4350_0000_0000_0000)); // 2^54
        return (m, e - 54);
    }
    let mantissa = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (mantissa, biased - 1022)
}

/// Decompose a positive real scale ratio into a Q0.31 multiplier and a shift
/// (C++ `quantizeMultiplier`). Returns `(multiplier, shift)`: `shift < 0` for a
/// ratio above 1, `shift > 0` below 1. A zero ratio gives `(0, 0)`.
pub fn quantize_multiplier(ratio: f64) -> (i32, i32) {
    if ratio == 0.0 {
        return (0, 0);
    }
    let (mantissa, mut exponent) = frexp(ratio);
    // C++ llround: round half away from zero, as f64::round does.
    let mut q = (mantissa * (1i64 << 31) as f64).round() as i64;
    // The mantissa can round up to exactly 2^31; renormalise.
    if q == 1i64 << 31 {
        q >>= 1;
        exponent += 1;
    }
    (q as i32, -exponent)
}

/// Build a [`Requantizer`] for `(input_scale * weight_scale) / output_scale`
/// (C++ `buildRequantizer<int8_t>`). The zero point and bounds are narrowed to
/// `i8` as the C++ `static_cast` does.
pub fn build_requantizer(
    input_scale: f32,
    weight_scale: f32,
    output_scale: f32,
    output_zero_point: i32,
    qmin: i32,
    qmax: i32,
) -> Requantizer {
    let ratio = (f64::from(input_scale) * f64::from(weight_scale)) / f64::from(output_scale);
    let (multiplier, shift) = quantize_multiplier(ratio);
    Requantizer {
        multiplier,
        shift,
        zero_point: output_zero_point as i8,
        qmin: qmin as i8,
        qmax: qmax as i8,
    }
}

/// Quantize a real value to int8: `round(x / scale) + zero_point`, ties away
/// from zero (as TFLite), saturated to `[qmin, qmax]` (C++ `quantize<int8_t>`).
pub fn quantize(x: f32, scale: f32, zero_point: i32, qmin: i32, qmax: i32) -> i8 {
    let q = (f64::from(x) / f64::from(scale)).round() + f64::from(zero_point);
    let mut qi = q as i64;
    if qi < i64::from(qmin) {
        qi = i64::from(qmin);
    }
    if qi > i64::from(qmax) {
        qi = i64::from(qmax);
    }
    qi as i8
}

/// The real value an int8 stands for: `scale * (q - zero_point)` (C++
/// `dequantize<int8_t>`).
pub fn dequantize(q: i8, scale: f32, zero_point: i32) -> f32 {
    scale * (i32::from(q) - zero_point) as f32
}

/// Round to nearest, ties away from zero, and saturate to int8 (C++
/// `qSaturateRoundToInt8`).
fn saturate_round_to_i8(v: f32) -> i8 {
    (f64::from(v).round() as i64).clamp(-128, 127) as i8
}

/// A 256-entry lookup table for `f` from the input tensor's grid to the output
/// tensor's: dequantize each int8, apply `f`, requantize.
fn build_lut(
    input_scale: f32,
    input_zero_point: i32,
    output_scale: f32,
    output_zero_point: i32,
    f: impl Fn(f32) -> f32,
) -> [i8; LUT_SIZE] {
    core::array::from_fn(|index| {
        let x = index as i32 - 128;
        let x_real = input_scale * (x - input_zero_point) as f32;
        let y_q = output_zero_point as f32 + f(x_real) / output_scale;
        saturate_round_to_i8(y_q)
    })
}

/// Sigmoid lookup table, `1 / (1 + exp(-x))` (C++ `buildQSigmoidLUT`). An output
/// grid of scale 1/256 and zero point -128 uses the full int8 range.
pub fn build_sigmoid_lut(
    input_scale: f32,
    input_zero_point: i32,
    output_scale: f32,
    output_zero_point: i32,
) -> [i8; LUT_SIZE] {
    build_lut(
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        |x| 1.0 / (1.0 + (-x).exp()),
    )
}

/// Tanh lookup table (C++ `buildQTanhLUT`). An output grid of scale 1/128 and
/// zero point 0 uses the full int8 range.
pub fn build_tanh_lut(
    input_scale: f32,
    input_zero_point: i32,
    output_scale: f32,
    output_zero_point: i32,
) -> [i8; LUT_SIZE] {
    build_lut(
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        f32::tanh,
    )
}

/// [`QSoftmax`] parameters for an input tensor of scale `input_scale` and a
/// softmax of `beta` (1.0 for a plain softmax), computed as TFLite Micro does
/// for an int8 softmax (`PreprocessSoftmaxScaling` with 5 integer bits, then
/// `CalculateInputRadius`; C++ `buildQSoftmaxFixedPointParams`).
pub fn build_softmax(input_scale: f64, beta: f64) -> QSoftmax {
    const INPUT_INTEGER_BITS: i32 = 5;
    let max_real_multiplier = ((1i64 << 31) - 1) as f64;
    let mut real_multiplier = beta * input_scale * (1i64 << (31 - INPUT_INTEGER_BITS)) as f64;
    if real_multiplier > max_real_multiplier {
        real_multiplier = max_real_multiplier;
    }

    // quantize_multiplier returns shift = -exponent; a multiplier above one is
    // a left shift.
    let (input_multiplier, shift) = quantize_multiplier(real_multiplier);
    let input_left_shift = -shift;

    // 2^input_left_shift is exact in f64 (the C++ shifts an int64).
    let max_input_rescaled = f64::from((1 << INPUT_INTEGER_BITS) - 1)
        * (1i64 << (31 - INPUT_INTEGER_BITS)) as f64
        / 2f64.powi(input_left_shift);
    QSoftmax {
        input_multiplier,
        input_left_shift,
        diff_min: (max_input_rescaled.floor() as i32).saturating_neg(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known answers from tinymind unit_test/quantization.
    #[test]
    fn quantize_multiplier_cases() {
        assert_eq!(quantize_multiplier(0.0), (0, 0));
        // frexp(1.0) = 0.5 * 2^1.
        assert_eq!(quantize_multiplier(1.0), (1 << 30, -1));
        assert_eq!(quantize_multiplier(0.5), (1 << 30, 0));
        // A mantissa that rounds up to 2^31 is renormalised.
        assert_eq!(quantize_multiplier(1.0 - 1e-10), (1 << 30, -1));
        let unity = build_requantizer(1.0, 1.0, 1.0, 0, -128, 127);
        assert_eq!(unity.apply(42), 42);
        let half = build_requantizer(1.0, 0.5, 1.0, 0, -128, 127);
        assert_eq!(half.apply(100), 50);
    }

    #[test]
    fn frexp_matches_c() {
        for (x, m, e) in [
            (1.0, 0.5, 1),
            (0.75, 0.75, 0),
            (3.0, 0.75, 2),
            (-3.0, -0.75, 2),
            (f64::MIN_POSITIVE, 0.5, -1021),
            (f64::MIN_POSITIVE / 4.0, 0.5, -1023),
        ] {
            assert_eq!(frexp(x), (m, e), "frexp({x})");
        }
    }

    #[test]
    fn quantize_rounds_half_away_and_saturates() {
        assert_eq!(quantize(0.5, 1.0, 0, -128, 127), 1);
        assert_eq!(quantize(-0.5, 1.0, 0, -128, 127), -1);
        assert_eq!(quantize(1000.0, 1.0, 0, -128, 127), 127);
        assert_eq!(quantize(-1000.0, 1.0, 0, -128, 127), -128);
        assert_eq!(dequantize(-128, 0.5, -128), 0.0);
    }
}
