//! Host-side calibration for the 16x8 layers: float scales to the integer
//! parameters (C++ `buildQLUT16`, `buildQSigmoidLUT16`, `buildQTanhLUT16`,
//! `buildQSoftmaxInt16LUTs`, `buildQSoftmaxInt16Params`, and
//! `buildQAddParams` with TFLite's int16 left shift of 15). Needs the `std`
//! feature, for `exp` and `tanh`; on a device, run these on the host and
//! embed the results as constants. The float arithmetic follows the C++ step
//! for step, so the tables match the C++ ones on the same platform.

use super::{LUT_SIZE, QAdd, QSoftmax, Requantizer, Requantizer32};
use crate::int8::calibration::quantize_multiplier;

/// A [`Requantizer`] for `(input_scale * weight_scale) / output_scale`: the
/// ratio in `f64` from the `f32` scales, as
/// [`int8::calibration::build_requantizer`](crate::int8::calibration::build_requantizer)
/// forms it. TFLite's int16 tensors have zero point 0 and the full range.
///
/// ```
/// use tinymind::int16x8::calibration::build_requantizer;
///
/// let rq = build_requantizer(0.5, 0.25, 0.25, 0, -32768, 32767);
/// assert_eq!(rq.apply(1000), 500);
/// ```
pub fn build_requantizer(
    input_scale: f32,
    weight_scale: f32,
    output_scale: f32,
    output_zero_point: i16,
    qmin: i16,
    qmax: i16,
) -> Requantizer {
    let ratio = (f64::from(input_scale) * f64::from(weight_scale)) / f64::from(output_scale);
    let (multiplier, shift) = quantize_multiplier(ratio);
    Requantizer {
        multiplier,
        shift,
        zero_point: output_zero_point,
        qmin,
        qmax,
    }
}

/// Build an int16 [`QAdd`] for inputs and output `(scale, zero point)` (C++
/// `buildQAddParams` with TFLite's int16 left shift of 15): both inputs are
/// rescaled to twice the larger input scale, the ratios formed in `f64` from
/// the `f32` scales.
pub fn build_add(
    input_a: (f32, i16),
    input_b: (f32, i16),
    output: (f32, i16),
    qmin: i16,
    qmax: i16,
) -> QAdd {
    const LEFT_SHIFT: i32 = 15;
    let (a_scale, b_scale) = (input_a.0, input_b.0);
    // The C++ comparison, so equal or NaN scales take the same branch.
    let max_in = if a_scale > b_scale { a_scale } else { b_scale };
    let twice_max = 2.0 * f64::from(max_in);
    let real_a = f64::from(a_scale) / twice_max;
    let real_b = f64::from(b_scale) / twice_max;
    let real_out = twice_max / ((1i64 << LEFT_SHIFT) as f64 * f64::from(output.0));
    let (input_a_multiplier, input_a_shift) = quantize_multiplier(real_a);
    let (input_b_multiplier, input_b_shift) = quantize_multiplier(real_b);
    let (multiplier, shift) = quantize_multiplier(real_out);
    QAdd {
        input_a_zero_point: input_a.1,
        input_b_zero_point: input_b.1,
        left_shift: LEFT_SHIFT,
        input_a_multiplier,
        input_a_shift,
        input_b_multiplier,
        input_b_shift,
        output_requantizer: Requantizer32 {
            multiplier,
            shift,
            zero_point: output.1,
            qmin,
            qmax,
        },
    }
}

/// A 513-entry table for [`apply_lut`](super::apply_lut) with `transform(x)`,
/// from the input grid (`input_scale`, `input_zero_point`) to the symmetric
/// int16 output grid of `output_scale` (C++ `buildQLUT16`). Entry `i` is the
/// function at the knot `-32768 + 128 i`, less half the linear
/// interpolation's error at the interval's midpoint, as TFLite's
/// `LUTPopulateInt16` does, but with the samples exactly on the knots.
pub fn build_lut(
    input_scale: f64,
    input_zero_point: i32,
    output_scale: f64,
    transform: impl Fn(f64) -> f64,
) -> [i16; LUT_SIZE] {
    const STEPS: i32 = 512;
    let inv = 1.0 / output_scale;
    let knot = |i: i32| input_scale * f64::from(-32768 + 128 * i - input_zero_point);
    core::array::from_fn(|i| {
        let i = i as i32;
        let x = knot(i);
        let sample = (transform(x) * inv).round();
        let mut v = sample;
        if i < STEPS {
            let x_next = knot(i + 1);
            let mid_interp = ((transform(x_next) * inv + sample) / 2.0).round();
            let mid = (transform((x + x_next) / 2.0) * inv).round();
            v = sample - ((mid_interp - mid) / 2.0).round();
        }
        v.clamp(-32768.0, 32767.0) as i16
    })
}

/// The int16 sigmoid table (C++ `buildQSigmoidLUT16`); an output scale of
/// 2^-15 covers `(0, 1)`.
pub fn build_sigmoid_lut(
    input_scale: f64,
    input_zero_point: i32,
    output_scale: f64,
) -> [i16; LUT_SIZE] {
    build_lut(input_scale, input_zero_point, output_scale, |x| {
        1.0 / (1.0 + (-x).exp())
    })
}

/// The int16 tanh table (C++ `buildQTanhLUT16`); an output scale of 2^-15
/// covers `(-1, 1)`.
pub fn build_tanh_lut(
    input_scale: f64,
    input_zero_point: i32,
    output_scale: f64,
) -> [i16; LUT_SIZE] {
    build_lut(input_scale, input_zero_point, output_scale, f64::tanh)
}

/// TFLite's `LUTPopulateInt16<double>`, operation for operation: 512 samples
/// spaced `(input_max - input_min) / 512` apart, each less half the linear
/// interpolation's error at its midpoint, and a 513th at `input_max` (C++
/// `detail::populateTFLiteLUT16`).
fn populate_tflite_lut(
    input_scale: f64,
    input_zero_point: i32,
    output_scale: f64,
    output_zero_point: i32,
    transform: impl Fn(f64) -> f64,
) -> [i16; LUT_SIZE] {
    let input_min = input_scale * f64::from(-32768 - input_zero_point);
    let input_max = input_scale * f64::from(32767 - input_zero_point);
    let output_min = output_scale * f64::from(-32768 - output_zero_point);
    let output_max = output_scale * f64::from(32767 - output_zero_point);
    let nb_steps = 512;
    let step = (input_max - input_min) / f64::from(nb_steps);
    let half_step = step / 2.0;
    let output_scaling_inv = f64::from(32767 - (-32768) + 1) / (output_max - output_min);
    let mut lut = [0i16; LUT_SIZE];
    for (i, entry) in lut.iter_mut().enumerate().take(nb_steps as usize) {
        let i = i as f64;
        let val = transform(input_min + i * step);
        let val_midpoint = transform(input_min + i * step + half_step);
        let val_next = transform(input_min + (i + 1.0) * step);
        let sample_val = (val * output_scaling_inv).round();
        let midpoint_interp_val =
            ((val_next * output_scaling_inv + (val * output_scaling_inv).round()) / 2.0).round();
        let midpoint_val = (val_midpoint * output_scaling_inv).round();
        let midpoint_err = midpoint_interp_val - midpoint_val;
        let bias = (midpoint_err / 2.0).round();
        *entry = (sample_val - bias).clamp(-32768.0, 32767.0) as i16;
    }
    let last = (transform(input_max) * output_scaling_inv).round();
    lut[nb_steps as usize] = last.clamp(-32768.0, 32767.0) as i16;
    lut
}

/// [`QSoftmax`]'s two tables, `(exp, reciprocal)`, built as TFLite's
/// `SoftmaxPrepare` builds them for an int16 softmax (C++
/// `buildQSoftmaxInt16LUTs`): `exp(x)` over `[-10, 0]` and `1 / (1 + x)` over
/// `[0, 1]`. The layer uses constant copies; this builds them on the host to
/// check those copies.
pub fn build_softmax_luts() -> ([i16; LUT_SIZE], [i16; LUT_SIZE]) {
    let range = f64::from(32767 - (-32768));
    (
        populate_tflite_lut(10.0 / range, 32767, 2.0 / range, 0, f64::exp),
        populate_tflite_lut(1.0 / range, -32768, 2.0 / range, 0, |x| 1.0 / (1.0 + x)),
    )
}

/// The [`QSoftmax`] for an int16 input of scale `input_scale` (zero point 0)
/// and softmax `beta`, as TFLite computes its parameters (C++
/// `buildQSoftmaxInt16Params`): `input_scale * beta / (10 / 65535)`. Pass the
/// `f32` scale and beta widened to `f64`, as TFLite does, to get its
/// multiplier exactly.
pub fn build_softmax(input_scale: f64, beta: f64) -> QSoftmax {
    let (input_multiplier, input_shift) =
        quantize_multiplier(input_scale * beta / (10.0 / 65535.0));
    QSoftmax {
        input_multiplier,
        input_shift,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_tables_match_the_builder() {
        let (exp, reciprocal) = build_softmax_luts();
        assert_eq!(exp, super::super::softmax_tables::EXP);
        assert_eq!(reciprocal, super::super::softmax_tables::RECIPROCAL);
    }
}
