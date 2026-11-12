//! Host-side calibration: turning float scales into the integer parameters the
//! int8 layers use (C++ `quantizeMultiplier`, `buildRequantizer`, `quantize`,
//! `dequantize`, `buildQSigmoidLUT`, `buildQTanhLUT`,
//! `buildQSoftmaxFixedPointParams`, `computeAffineParamsAsymmetric`,
//! `computeAffineParamsSymmetric`, `computePerChannelSymmetricScales`, the
//! `RangeObserver`, `buildQAddParams`, `buildRescaler`,
//! `buildQMulRequantizer`, `buildQLSTMParams`, `buildQGRUParams`,
//! `quantizeQLSTMBiases`, `quantizeQGRUBiases`, `buildQFFTTwiddles`).
//!
//! Needs the `std` feature, for `f32::exp` and `f32::tanh`; tinymind (C++) gates the
//! same functions on `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`. On a device,
//! run these on the host and embed the results as constants. The float
//! arithmetic follows the C++ step for step, so the tables and requantizers match
//! the C++ ones on the same platform: `exp` and `tanh` come from the system math
//! library in both.

use super::{LUT_SIZE, QAdd, QSoftmax, Requantizer};
#[cfg(doc)]
use super::{QFft1D, QGruCell, QLstmCell};

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
///
/// The shift always lies in `-31..=31`, the [`Requantizer`]'s contract. A ratio
/// below 2^-32 rounds every `i32` accumulator to 0 and comes back as `(0, 0)`;
/// a ratio of 2^31 or more gets shift -31, which already saturates every
/// non-zero accumulator.
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
    let shift = -exponent;
    if shift > 31 {
        (0, 0)
    } else {
        (q as i32, shift.max(-31))
    }
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

/// A requantizer for a pure rescale, `input_scale / output_scale`, with no
/// weight: for [`QConcat`](super::QConcat) inputs and anything else that
/// copies values from one grid to another (C++ `buildRescaler`, which is
/// `buildRequantizer` with a weight scale of 1).
pub fn build_rescaler(
    input_scale: f32,
    output_scale: f32,
    output_zero_point: i32,
    qmin: i32,
    qmax: i32,
) -> Requantizer {
    build_requantizer(
        input_scale,
        1.0,
        output_scale,
        output_zero_point,
        qmin,
        qmax,
    )
}

/// The requantizer of a [`QMul`](super::QMul): `input_a_scale * input_b_scale /
/// output_scale`, in `f64` from the `f32` scales (C++ `buildQMulRequantizer`;
/// the same ratio as [`build_requantizer`] with the second input's scale as the
/// weight scale).
pub fn build_mul_requantizer(
    input_a_scale: f32,
    input_b_scale: f32,
    output_scale: f32,
    output_zero_point: i32,
    qmin: i32,
    qmax: i32,
) -> Requantizer {
    build_requantizer(
        input_a_scale,
        input_b_scale,
        output_scale,
        output_zero_point,
        qmin,
        qmax,
    )
}

/// Build a [`QAdd`] for inputs `(scale, zero point)` `input_a` and `input_b`
/// and output `output` (C++ `buildQAddParams` with its default left shift of 20,
/// TFLite's). Both inputs are rescaled to twice the larger input scale; ratios
/// are formed in `f64` from the `f32` scales, as the C++ does. Zero points and
/// bounds are narrowed to `i8`; raise `qmin` to the output zero point to fuse a
/// ReLU.
pub fn build_add(
    input_a: (f32, i32),
    input_b: (f32, i32),
    output: (f32, i32),
    qmin: i32,
    qmax: i32,
) -> QAdd {
    const LEFT_SHIFT: i32 = 20;
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
        input_a_zero_point: input_a.1 as i8,
        input_b_zero_point: input_b.1 as i8,
        left_shift: LEFT_SHIFT,
        input_a_multiplier,
        input_a_shift,
        input_b_multiplier,
        input_b_shift,
        output_requantizer: Requantizer {
            multiplier,
            shift,
            zero_point: output.1 as i8,
            qmin: qmin as i8,
            qmax: qmax as i8,
        },
    }
}

/// Symmetric affine parameters `(scale, 0)` covering `[fmin, fmax]`: the scale
/// maps the larger magnitude to `qmax` (C++ `computeAffineParamsSymmetric`).
/// A range of zeros, or a `qmax` of 0 or below, gives scale 1.0.
pub fn affine_params_symmetric(fmin: f32, fmax: f32, qmax: i32) -> (f32, i32) {
    // Comparisons as the C++ writes them, so a NaN takes the same branch.
    let abs_min = if fmin < 0.0 { -fmin } else { fmin };
    let abs_max = if fmax < 0.0 { -fmax } else { fmax };
    let absmax = if abs_min > abs_max { abs_min } else { abs_max };
    if absmax == 0.0 || qmax <= 0 {
        return (1.0, 0);
    }
    (absmax / qmax as f32, 0)
}

/// The running minimum and maximum of the values it has observed, for fitting
/// an activation's scale from calibration inputs (C++ `RangeObserver`). Before
/// any observation the range is `[0, 0]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RangeObserver {
    /// Smallest value observed.
    pub min_value: f32,
    /// Largest value observed.
    pub max_value: f32,
    /// Whether anything has been observed.
    pub has_data: bool,
}

impl RangeObserver {
    /// An observer with nothing observed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Widen the range to include `x`. The first value sets both ends.
    pub fn observe(&mut self, x: f32) {
        if !self.has_data {
            self.min_value = x;
            self.max_value = x;
            self.has_data = true;
        } else {
            if x < self.min_value {
                self.min_value = x;
            }
            if x > self.max_value {
                self.max_value = x;
            }
        }
    }

    /// [`observe`](Self::observe) each value in order.
    pub fn observe_all(&mut self, xs: &[f32]) {
        for &x in xs {
            self.observe(x);
        }
    }
}

/// Asymmetric per-tensor affine parameters `(scale, zero_point)` that map the
/// real range `[fmin, fmax]`, widened to include 0, onto `[qmin, qmax]` (C++
/// `computeAffineParamsAsymmetric`). An empty range gives `(1.0, qmin)`.
pub fn affine_params_asymmetric(fmin: f32, fmax: f32, qmin: i32, qmax: i32) -> (f32, i32) {
    // `if`s rather than min/max, which treat NaN differently from the C++.
    let fmin = if fmin > 0.0 { 0.0 } else { fmin };
    let fmax = if fmax < 0.0 { 0.0 } else { fmax };
    let qrange = qmax.wrapping_sub(qmin) as f32;
    if fmax == fmin || qrange == 0.0 {
        return (1.0, qmin);
    }
    let scale = (fmax - fmin) / qrange;
    // C++ lround: half away from zero, as f64::round.
    let mut zp = (f64::from(qmin) - f64::from(fmin) / f64::from(scale)).round() as i64;
    if zp < i64::from(qmin) {
        zp = i64::from(qmin);
    }
    if zp > i64::from(qmax) {
        zp = i64::from(qmax);
    }
    (scale, zp as i32)
}

/// One symmetric weight scale per output channel, `max |w| / qmax` over each
/// channel's weights (C++ `computePerChannelSymmetricScales`), as TFLite
/// quantizes depthwise and convolution weights. A channel of zeros, or a
/// `qmax` of 0 or below, gets scale 1.0.
pub fn per_channel_symmetric_scales<const C: usize, const N: usize>(
    weights: &[[f32; N]; C],
    qmax: i32,
) -> [f32; C] {
    let mut scales = [1.0f32; C];
    for (scale, channel) in scales.iter_mut().zip(weights) {
        // Comparisons as the C++ writes them, so a NaN is skipped the same way.
        let mut absmax = 0.0f32;
        for &v in channel {
            let a = if v < 0.0 { -v } else { v };
            if a > absmax {
                absmax = a;
            }
        }
        if absmax != 0.0 && qmax > 0 {
            *scale = absmax / qmax as f32;
        }
    }
    scales
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

/// Output scale of the sigmoid LUTs the recurrent cells read: 1/256, with zero
/// point -128 (C++ `kQLstmSigmoidOutputScale`). Build them with
/// `build_sigmoid_lut(lut_input_scale, 0, SIGMOID_LUT_OUTPUT_SCALE, -128)`.
pub const SIGMOID_LUT_OUTPUT_SCALE: f32 = 1.0 / 256.0;

/// Output scale of the tanh LUTs the recurrent cells read: 1/128, with zero
/// point 0 (C++ `kQLstmTanhOutputScale`). Build them with
/// `build_tanh_lut(lut_input_scale, 0, TANH_LUT_OUTPUT_SCALE, 0)`.
pub const TANH_LUT_OUTPUT_SCALE: f32 = 1.0 / 128.0;

/// The float scales a [`QLstmCell`] is calibrated from (C++ `QLSTMScales`).
/// Gates are in the order i, f, g, o.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LstmScales {
    /// Scale of the input tensor.
    pub input_scale: f32,
    /// Scale of the hidden state.
    pub hidden_scale: f32,
    /// Scale of the cell state.
    pub cell_scale: f32,
    /// Scale of the LUTs' input grid, shared by every gate and the cell's tanh.
    pub lut_input_scale: f32,
    /// Per gate, the scale of the symmetric input weights.
    pub input_weight_scales: [f32; 4],
    /// Per gate, the scale of the symmetric recurrent weights.
    pub recurrent_weight_scales: [f32; 4],
}

/// The multipliers and shifts of a [`QLstmCell`] (C++ `QLSTMParams`), with the
/// same field names as the cell. `output_multiplier` and `output_shift` go in
/// the cell's `output_requantizer`, with the hidden zero point and bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LstmParams {
    /// Per gate, the cell's `input_to_lut_multiplier`.
    pub input_to_lut_multiplier: [i32; 4],
    /// Per gate, the cell's `input_to_lut_shift`.
    pub input_to_lut_shift: [i32; 4],
    /// Per gate, the cell's `recurrent_to_lut_multiplier`.
    pub recurrent_to_lut_multiplier: [i32; 4],
    /// Per gate, the cell's `recurrent_to_lut_shift`.
    pub recurrent_to_lut_shift: [i32; 4],
    /// The cell's `f_times_c_multiplier`.
    pub f_times_c_multiplier: i32,
    /// The cell's `f_times_c_shift`.
    pub f_times_c_shift: i32,
    /// The cell's `i_times_g_multiplier`.
    pub i_times_g_multiplier: i32,
    /// The cell's `i_times_g_shift`.
    pub i_times_g_shift: i32,
    /// The cell's `cell_to_tanh_multiplier`.
    pub cell_to_tanh_multiplier: i32,
    /// The cell's `cell_to_tanh_shift`.
    pub cell_to_tanh_shift: i32,
    /// The `multiplier` of the cell's `output_requantizer`.
    pub output_multiplier: i32,
    /// The `shift` of the cell's `output_requantizer`.
    pub output_shift: i32,
}

/// Per gate, the multipliers rescaling the input and recurrent sums to the LUT
/// input grid: `input_scale * weight_scale / lut_input_scale`, in `f64`.
#[allow(clippy::type_complexity)]
fn gate_rescales<const G: usize>(
    input_scale: f32,
    hidden_scale: f32,
    lut_input_scale: f32,
    input_weight_scales: &[f32; G],
    recurrent_weight_scales: &[f32; G],
) -> ([i32; G], [i32; G], [i32; G], [i32; G]) {
    let mut out = ([0; G], [0; G], [0; G], [0; G]);
    let lut = f64::from(lut_input_scale);
    for g in 0..G {
        let r_in = f64::from(input_scale) * f64::from(input_weight_scales[g]) / lut;
        (out.0[g], out.1[g]) = quantize_multiplier(r_in);
        let r_rec = f64::from(hidden_scale) * f64::from(recurrent_weight_scales[g]) / lut;
        (out.2[g], out.3[g]) = quantize_multiplier(r_rec);
    }
    out
}

/// Build a [`QLstmCell`]'s multipliers and shifts from its float scales (C++
/// `buildQLSTMParams`). The ratios are formed in `f64` from the `f32` scales,
/// as the C++ does.
///
/// ```
/// use tinymind::int8::calibration::{
///     LstmScales, SIGMOID_LUT_OUTPUT_SCALE, TANH_LUT_OUTPUT_SCALE, build_lstm_params,
///     build_sigmoid_lut, build_tanh_lut, quantize, quantize_gate_biases,
/// };
/// use tinymind::int8::{QLstmCell, Requantizer};
///
/// // A float LSTM with 1 input and 2 hidden units.
/// let w_in = [[[0.5f32], [-0.25]], [[0.3], [0.1]], [[-0.4], [0.2]], [[0.25], [0.5]]];
/// let w_rec = [[[0.1f32, 0.2], [0.0, -0.1]]; 4];
/// let bias = [[0.0f32, 0.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
///
/// let scales = LstmScales {
///     input_scale: 1.0 / 127.0,
///     hidden_scale: 1.0 / 127.0,
///     cell_scale: 4.0 / 127.0,
///     lut_input_scale: 8.0 / 127.0,
///     input_weight_scales: [0.5 / 127.0, 0.3 / 127.0, 0.4 / 127.0, 0.5 / 127.0],
///     recurrent_weight_scales: [0.2 / 127.0; 4],
/// };
/// let p = build_lstm_params(&scales);
/// let q = |w: f32, s: f32| quantize(w, s, 0, -127, 127);
/// let input_weights: [[[i8; 1]; 2]; 4] =
///     core::array::from_fn(|g| w_in[g].map(|row| row.map(|w| q(w, scales.input_weight_scales[g]))));
/// let recurrent_weights: [[[i8; 2]; 2]; 4] = core::array::from_fn(|g| {
///     w_rec[g].map(|row| row.map(|w| q(w, scales.recurrent_weight_scales[g])))
/// });
/// let biases = quantize_gate_biases(&bias, scales.lut_input_scale);
/// let sigmoid = build_sigmoid_lut(scales.lut_input_scale, 0, SIGMOID_LUT_OUTPUT_SCALE, -128);
/// let tanh = build_tanh_lut(scales.lut_input_scale, 0, TANH_LUT_OUTPUT_SCALE, 0);
///
/// let cell: QLstmCell<'_, 1, 2> = QLstmCell {
///     input_weights: &input_weights,
///     recurrent_weights: &recurrent_weights,
///     biases: &biases,
///     input_zero_point: 0,
///     hidden_zero_point: 0,
///     cell_zero_point: 0,
///     input_to_lut_multiplier: p.input_to_lut_multiplier,
///     input_to_lut_shift: p.input_to_lut_shift,
///     recurrent_to_lut_multiplier: p.recurrent_to_lut_multiplier,
///     recurrent_to_lut_shift: p.recurrent_to_lut_shift,
///     sigmoid_lut: &sigmoid,
///     tanh_lut: &tanh,
///     f_times_c_multiplier: p.f_times_c_multiplier,
///     f_times_c_shift: p.f_times_c_shift,
///     i_times_g_multiplier: p.i_times_g_multiplier,
///     i_times_g_shift: p.i_times_g_shift,
///     cell_qmin: -127,
///     cell_qmax: 127,
///     cell_to_tanh_multiplier: p.cell_to_tanh_multiplier,
///     cell_to_tanh_shift: p.cell_to_tanh_shift,
///     tanh_cell_lut: &tanh,
///     output_requantizer: Requantizer {
///         multiplier: p.output_multiplier,
///         shift: p.output_shift,
///         zero_point: 0,
///         qmin: -128,
///         qmax: 127,
///     },
/// };
/// let (mut h, mut c) = ([0i8; 2], [0i8; 2]);
/// for x in [0.5f32, -0.3, 0.7] {
///     cell.forward(&[quantize(x, scales.input_scale, 0, -128, 127)], &mut h, &mut c);
/// }
/// println!("h = {:?}", h.map(|v| f32::from(v) * scales.hidden_scale));
/// ```
pub fn build_lstm_params(scales: &LstmScales) -> LstmParams {
    let (in_m, in_s, rec_m, rec_s) = gate_rescales(
        scales.input_scale,
        scales.hidden_scale,
        scales.lut_input_scale,
        &scales.input_weight_scales,
        &scales.recurrent_weight_scales,
    );
    let sig = f64::from(SIGMOID_LUT_OUTPUT_SCALE);
    let tanh = f64::from(TANH_LUT_OUTPUT_SCALE);
    let c = f64::from(scales.cell_scale);
    let h = f64::from(scales.hidden_scale);
    let (f_times_c_multiplier, f_times_c_shift) = quantize_multiplier(sig * c / c);
    let (i_times_g_multiplier, i_times_g_shift) = quantize_multiplier(sig * tanh / c);
    let (cell_to_tanh_multiplier, cell_to_tanh_shift) =
        quantize_multiplier(c / f64::from(scales.lut_input_scale));
    let (output_multiplier, output_shift) = quantize_multiplier(sig * tanh / h);
    LstmParams {
        input_to_lut_multiplier: in_m,
        input_to_lut_shift: in_s,
        recurrent_to_lut_multiplier: rec_m,
        recurrent_to_lut_shift: rec_s,
        f_times_c_multiplier,
        f_times_c_shift,
        i_times_g_multiplier,
        i_times_g_shift,
        cell_to_tanh_multiplier,
        cell_to_tanh_shift,
        output_multiplier,
        output_shift,
    }
}

/// The float scales a [`QGruCell`] is calibrated from (C++ `QGRUScales`).
/// Gates are in the order r, z, n.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GruScales {
    /// Scale of the input tensor.
    pub input_scale: f32,
    /// Scale of the hidden state.
    pub hidden_scale: f32,
    /// Scale of the LUTs' input grid, shared by every gate.
    pub lut_input_scale: f32,
    /// Per gate, the scale of the symmetric input weights.
    pub input_weight_scales: [f32; 3],
    /// Per gate, the scale of the symmetric recurrent weights.
    pub recurrent_weight_scales: [f32; 3],
}

/// The multipliers and shifts of a [`QGruCell`] (C++ `QGRUParams`), with the
/// same field names as the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GruParams {
    /// Per gate, the cell's `input_to_lut_multiplier`.
    pub input_to_lut_multiplier: [i32; 3],
    /// Per gate, the cell's `input_to_lut_shift`.
    pub input_to_lut_shift: [i32; 3],
    /// Per gate, the cell's `recurrent_to_lut_multiplier`.
    pub recurrent_to_lut_multiplier: [i32; 3],
    /// Per gate, the cell's `recurrent_to_lut_shift`.
    pub recurrent_to_lut_shift: [i32; 3],
    /// The cell's `r_times_h_multiplier`.
    pub r_times_h_multiplier: i32,
    /// The cell's `r_times_h_shift`.
    pub r_times_h_shift: i32,
    /// The cell's `one_minus_z_times_n_multiplier`.
    pub one_minus_z_times_n_multiplier: i32,
    /// The cell's `one_minus_z_times_n_shift`.
    pub one_minus_z_times_n_shift: i32,
    /// The cell's `z_times_h_multiplier`.
    pub z_times_h_multiplier: i32,
    /// The cell's `z_times_h_shift`.
    pub z_times_h_shift: i32,
}

/// Build a [`QGruCell`]'s multipliers and shifts from its float scales (C++
/// `buildQGRUParams`). Use the LUTs and biases as for [`build_lstm_params`].
pub fn build_gru_params(scales: &GruScales) -> GruParams {
    let (in_m, in_s, rec_m, rec_s) = gate_rescales(
        scales.input_scale,
        scales.hidden_scale,
        scales.lut_input_scale,
        &scales.input_weight_scales,
        &scales.recurrent_weight_scales,
    );
    let sig = f64::from(SIGMOID_LUT_OUTPUT_SCALE);
    let tanh = f64::from(TANH_LUT_OUTPUT_SCALE);
    let h = f64::from(scales.hidden_scale);
    let (r_times_h_multiplier, r_times_h_shift) = quantize_multiplier(sig);
    let (one_minus_z_times_n_multiplier, one_minus_z_times_n_shift) =
        quantize_multiplier(sig * tanh / h);
    let (z_times_h_multiplier, z_times_h_shift) = quantize_multiplier(sig);
    GruParams {
        input_to_lut_multiplier: in_m,
        input_to_lut_shift: in_s,
        recurrent_to_lut_multiplier: rec_m,
        recurrent_to_lut_shift: rec_s,
        r_times_h_multiplier,
        r_times_h_shift,
        one_minus_z_times_n_multiplier,
        one_minus_z_times_n_shift,
        z_times_h_multiplier,
        z_times_h_shift,
    }
}

/// Quantize a recurrent cell's float biases, `biases[gate][unit]`, into the LUT
/// input grid: `round(b / lut_input_scale)` in `f64`, ties away from zero
/// (C++ `quantizeQLSTMBiases` with `G` = 4, `quantizeQGRUBiases` with `G` = 3).
/// A value beyond the `i32` range wraps, as the C++ conversion from `long`
/// does.
pub fn quantize_gate_biases<const G: usize, const HID: usize>(
    biases: &[[f32; HID]; G],
    lut_input_scale: f32,
) -> [[i32; HID]; G] {
    biases
        .map(|gate| gate.map(|b| (f64::from(b) / f64::from(lut_input_scale)).round() as i64 as i32))
}

/// The Q1.15 twiddle tables of a [`QFft1D`]`<N, HALF>`: `cos(-2πk/N)` and
/// `sin(-2πk/N)` for `k` in `0..N/2`, times 32768, rounded to nearest (ties
/// away from zero) and clamped to `i16` (C++ `buildQFFTTwiddles`). Returns
/// `(cos, sin)`.
///
/// `1.0` (`k = 0`) clamps to 32767, the convention of TFLite and CMSIS-DSP;
/// `-1.0` (the sine at `k = N/4`) is exactly -32768. `cos` and `sin` come from
/// the system math library, as in the C++, so the tables match the C++ ones on
/// the same platform. The same checks on `N` and `HALF` as the layer's apply.
///
/// ```
/// use tinymind::int8::QFft1D;
/// use tinymind::int8::calibration::build_fft_twiddles;
///
/// let (cos, sin) = build_fft_twiddles::<8, 4>();
/// assert_eq!(cos, [32767, 23170, 0, -23170]);
/// assert_eq!(sin, [0, -23170, -32768, -23170]);
/// let fft: QFft1D<'_, 8, 4> = QFft1D { twiddle_cos: &cos, twiddle_sin: &sin };
/// # let _ = fft;
/// ```
pub fn build_fft_twiddles<const N: usize, const HALF: usize>() -> ([i16; HALF], [i16; HALF]) {
    const {
        assert!(N >= 2, "FFT length must be at least 2");
        assert!(N.is_power_of_two(), "FFT length must be a power of two");
        assert!(HALF == N / 2, "HALF must be N / 2");
    }
    // C `lround`, then the C++ clamp to the i16 range.
    let q15 = |x: f64| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
    let mut cos = [0i16; HALF];
    let mut sin = [0i16; HALF];
    for (k, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
        // The C++ literal 6.283185307179586476925286766559 is TAU as an f64.
        let phase = -core::f64::consts::TAU * k as f64 / N as f64;
        *c = q15(phase.cos());
        *s = q15(phase.sin());
    }
    (cos, sin)
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
        // The shift stays in -31..=31 (tinymind #247); values from the C++.
        assert_eq!(quantize_multiplier(2f64.powi(-31)), (1 << 30, 30));
        assert_eq!(quantize_multiplier(2f64.powi(-32)), (1 << 30, 31));
        assert_eq!(quantize_multiplier(0.75 * 2f64.powi(-32)), (0, 0));
        assert_eq!(quantize_multiplier(2f64.powi(-40)), (0, 0));
        assert_eq!(quantize_multiplier(1e-12), (0, 0));
        assert_eq!(quantize_multiplier(2f64.powi(30)), (1 << 30, -31));
        assert_eq!(quantize_multiplier(2f64.powi(31)), (1 << 30, -31));
        assert_eq!(
            quantize_multiplier(0.75 * 2f64.powi(32)),
            (1_610_612_736, -31)
        );
        assert_eq!(quantize_multiplier(2f64.powi(40)), (1 << 30, -31));
        assert_eq!(quantize_multiplier(1e15), (1_907_348_633, -31));
        // A ratio below 2^-32 rounds every accumulator to 0, so the output is
        // the zero point; a left shift past 31 saturates exactly as 31 does.
        let rq = |(multiplier, shift)| Requantizer {
            multiplier,
            shift,
            zero_point: 3,
            qmin: -128,
            qmax: 127,
        };
        for acc in [i32::MIN, -1 << 20, -2, -1, 0, 1, 2, 1 << 20, i32::MAX] {
            assert_eq!(rq(quantize_multiplier(2f64.powi(-40))).apply(acc), 3);
            assert_eq!(rq((1 << 30, -41)).apply(acc), rq((1 << 30, -31)).apply(acc));
        }
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

    #[test]
    fn range_observer_starts_from_its_first_value() {
        let mut obs = RangeObserver::new();
        assert!(!obs.has_data);
        obs.observe(5.0);
        assert_eq!((obs.min_value, obs.max_value), (5.0, 5.0));
        obs.observe_all(&[7.0, 6.0, 2.5]);
        assert_eq!((obs.min_value, obs.max_value), (2.5, 7.0));
        assert!(obs.has_data);
    }

    #[test]
    fn symmetric_params_cover_the_larger_magnitude() {
        assert_eq!(affine_params_symmetric(-4.0, 2.0, 127), (4.0 / 127.0, 0));
        assert_eq!(affine_params_symmetric(-1.0, 3.0, 127), (3.0 / 127.0, 0));
        assert_eq!(affine_params_symmetric(0.0, 0.0, 127), (1.0, 0));
        assert_eq!(affine_params_symmetric(-1.0, 1.0, 0), (1.0, 0));
    }
}
