//! Int8 affine-quantized inference, ported from tinymind (C++) `cpp/qaffine.hpp`,
//! `cpp/qdense.hpp`, `cpp/qconv2d.hpp`, `cpp/qdepthwiseconv2d.hpp`,
//! `cpp/qpointwiseconv2d.hpp`, `cpp/qpool2d.hpp`, `cpp/qadd.hpp`, `cpp/qmul.hpp`,
//! `cpp/qpad.hpp`, `cpp/qconcat.hpp`, `cpp/qactivations.hpp`,
//! `cpp/qsoftmax_fixedpoint.hpp`, `cpp/qlstm.hpp`, `cpp/qgru.hpp` and
//! `cpp/qfft1d.hpp`.
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
//! Layers run fastest as `const` or `static` items, as the importers generate
//! them: the compiler then folds their weights and requantizers into the code.
//! A layer built from values known only at run time gives the same results
//! more slowly; on a Cortex-M4 a convolution over one input channel took about
//! twice as long.
//!
//! # Convolution and pooling
//!
//! [`QConv2D`] (and [`QConv2DPerChannel`]), [`QDepthwiseConv2D`],
//! [`QPointwiseConv2D`] (and [`QPointwiseConv2DPerChannel`]), [`QMaxPool2D`],
//! [`QAvgPool2D`] and [`QGlobalAvgPool2D`] work on NHWC tensors without the batch
//! axis, as nested arrays: an input is `[[[i8; C]; W]; H]`, indexed
//! `[row][column][channel]`. Kernel, strides and padding are const parameters of
//! the layer; the input and output sizes come from the arrays passed to
//! `forward`, which writes into a caller-owned output so tensors can share
//! buffers. The output must have the size the layer produces, `(H + PT + PB -
//! KH) / SH + 1` rows and likewise columns, or the call does not compile; an
//! invalid layer (a zero stride, a kernel larger than the padded input) does not
//! compile either. Padding holds the input zero point, as TFLite pads, so a
//! padded tap contributes nothing. Pooling keeps the input's scale and zero
//! point: no requantizer.
//!
//! [`QAdd`] adds two tensors of the same shape with their own scales and zero
//! points (TFLite ADD), for residual connections, and [`QMul`] multiplies them
//! (TFLite MUL, squeeze-and-excite gating). [`QPad2D`] pads a tensor with a
//! constant and [`QConcat`] joins two along the channel axis.
//!
//! # Recurrent cells
//!
//! [`QLstmCell`] and [`QGruCell`] run one time step of an int8 LSTM or GRU
//! cell, updating the caller's hidden state (and the LSTM's cell state, `i8`
//! or `i16`) in place; the caller runs the time loop over a sequence. Weights
//! are gate-major, `[gate][unit][input]`: PyTorch's `nn.LSTM` stacks its gates
//! in the same order (i, f, g, o), so its weight matrices map onto them
//! directly (add its two bias vectors, `bias_ih` and `bias_hh`, before
//! quantizing them). The GRU applies its reset gate before the recurrent
//! multiply, as Keras does with `reset_after=False` (whose kernels store the
//! gates as update, reset, new, so reorder them); PyTorch's `nn.GRU` and Keras' default
//! apply it after, which is a different function. The gate
//! activations come from 256-entry sigmoid and tanh tables with fixed output
//! grids; `calibration` builds the tables and the rescaling parameters from
//! float scales.
//!
//! # FFT
//!
//! [`QFft1D`] is a radix-2 FFT for vibration, audio or IMU front ends: 16-bit
//! samples, Q1.15 twiddle tables borrowed from flash, in place on the caller's
//! real and imaginary buffers. Its bin magnitudes come out in `i32`, ready for a
//! [`Requantizer`] onto an int8 grid; `calibration` builds the twiddle tables.
//!
//! A convolution over two or more input channels widens its input rows and
//! each filter's weights to 16 bits and adds the products in pairs. On cores
//! with the DSP extension (Cortex-M4, M7, M33), LLVM can turn each pair into
//! one `SMLAD` instruction, from this plain Rust, but by default it gives up
//! on long unrolled sums. Building with `-C
//! llvm-args=-arm-parallel-dsp-load-limit=512` (in the target's `rustflags`)
//! lets it pair them: on an STM32F411 the MLPerf Tiny ResNet-8 then takes 36%
//! fewer cycles. It is an LLVM internal option, not a stable Rust one.
//!
//! ```
//! use tinymind::int8::{QConv2D, QMaxPool2D, Requantizer};
//!
//! // Eight 3x3 filters over a 20x20 single-channel input, then 2x2 max pooling.
//! static WEIGHTS: [[[[i8; 1]; 3]; 3]; 8] = [[[[1]; 3]; 3]; 8];
//! const RQ: Requantizer = Requantizer { multiplier: 1 << 30, shift: 3, zero_point: 0, qmin: -128, qmax: 127 };
//! let conv: QConv2D<'_, 1, 3, 3, 8> =
//!     QConv2D { weights: &WEIGHTS, biases: None, input_zero_point: 0, requantizer: RQ };
//!
//! let input = [[[2i8; 1]; 20]; 20];
//! let mut features = [[[0i8; 8]; 18]; 18]; // 3x3, stride 1, no padding: 18 x 18
//! conv.forward(&input, &mut features);
//! let mut pooled = [[[0i8; 8]; 9]; 9]; // 2x2 windows: 9 x 9
//! QMaxPool2D::<2, 2>.forward(&features, &mut pooled);
//! // Each window sums nine 2s; the requantizer scales by 1/16: 18/16 rounds to 1.
//! assert_eq!(pooled[0][0], [1; 8]);
//! ```
//!
//! An output of the wrong size does not compile:
//!
//! ```compile_fail
//! # use tinymind::int8::{QConv2D, Requantizer};
//! # static WEIGHTS: [[[[i8; 1]; 3]; 3]; 8] = [[[[1]; 3]; 3]; 8];
//! # const RQ: Requantizer = Requantizer { multiplier: 1 << 30, shift: 3, zero_point: 0, qmin: -128, qmax: 127 };
//! # let conv: QConv2D<'_, 1, 3, 3, 8> =
//! #     QConv2D { weights: &WEIGHTS, biases: None, input_zero_point: 0, requantizer: RQ };
//! let input = [[[2i8; 1]; 20]; 20];
//! let mut features = [[[0i8; 8]; 17]; 18]; // 17 columns: the layer makes 18
//! conv.forward(&input, &mut features);
//! ```
//!
//! Results match tinymind (C++) bit for bit, and through it TFLite's reference
//! kernels for the layers TFLite has (the recurrent cells' table-based gates
//! are tinymind (C++)'s own, not TFLite's integer LSTM kernel). Where the C++ would hit undefined behaviour
//! (an `i32` accumulator overflow, a shift out of its documented range), this
//! port wraps or clamps instead, so every function is total and none panics.

pub(crate) mod add;
#[cfg(feature = "std")]
pub mod calibration;
pub(crate) mod compose;
pub(crate) mod conv;
pub(crate) mod fft;
pub(crate) mod pool;
pub(crate) mod recurrent;

pub use add::{QAdd, Tensor};
pub use compose::{QConcat, QMul, QPad2D};
pub use conv::{
    QConv2D, QConv2DPerChannel, QDepthwiseConv2D, QPointwiseConv2D, QPointwiseConv2DPerChannel,
};
pub use fft::QFft1D;
pub use pool::{QAvgPool2D, QGlobalAvgPool2D, QMaxPool2D};
pub use recurrent::{CellStorage, QGruCell, QLstmCell};

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

/// `x * 2^n`, saturated to the `i32` range, for [`QAdd`]: the left shift in
/// [`multiply_by_quantized_multiplier`].
pub(crate) fn saturating_shl(x: i32, n: u32) -> i32 {
    if n > 30 {
        // Any non-zero x saturates; 2^31 itself is not representable.
        if x > 0 {
            i32::MAX
        } else if x < 0 {
            i32::MIN
        } else {
            0
        }
    } else if n > 0 {
        // x * 2^n fits i32 exactly when x lies in [-2^(31 - n), 2^(31 - n) - 1].
        let limit = i32::MAX >> n;
        if x > limit {
            i32::MAX
        } else if x < -limit - 1 {
            i32::MIN
        } else {
            x * (1 << n)
        }
    } else {
        x
    }
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
    /// Power-of-two exponent: positive divides, negative multiplies. tinymind (C++)'s
    /// contract is `-31..=31` (what `calibration::quantize_multiplier` returns);
    /// outside it the result is still defined (see the module docs).
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

/// [`Requantizer::apply`] with the work that does not depend on the
/// accumulator done once (C++ `detail::FastRequantizer`): on the fused path
/// the output zero point folds into the rounding nudge. The C++ form takes
/// only the full int8 range; this one also takes the narrower ranges of a fused
/// ReLU, saturating in `apply`'s order. Layers prepare one per filter
/// and run their pixel loop once per form. A Kani proof checks it equals
/// `apply` for every accumulator wherever [`FastRequantizer::new`] accepts the
/// requantizer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FastRequantizer {
    multiplier: i32,
    nudge_high: u32,
    high_shift: u32,
    qmin: i32,
    qmax: i32,
}

impl FastRequantizer {
    /// The fast form, or `None` where `apply` must take its general path: a
    /// negative multiplier, a shift outside `2..=31`, or a zero-point term
    /// above 2^29 (which could carry out of the nudge's high word).
    pub(crate) fn new(rq: &Requantizer) -> Option<Self> {
        if rq.multiplier < 0 || !(2..=31).contains(&rq.shift) {
            return None;
        }
        let zp_term = i64::from(rq.zero_point) * (1i64 << (rq.shift - 1));
        if zp_term.abs() > 1 << 29 {
            return None;
        }
        Some(FastRequantizer {
            multiplier: rq.multiplier,
            nudge_high: ((1i64 << (rq.shift - 2)) + zp_term) as u32,
            high_shift: (rq.shift - 1) as u32,
            qmin: rq.qmin.into(),
            qmax: rq.qmax.into(),
        })
    }

    /// `rq.apply(acc)` for the requantizer this was prepared from.
    #[inline]
    pub(crate) fn apply(&self, acc: i32) -> i8 {
        let negative = (acc as u32) >> 31;
        let high = self.nudge_high.wrapping_sub(negative);
        let low = (negative << 31) | 0x4000_0000;
        let nudge = (u64::from(high) << 32) | u64::from(low);
        let v = nudge.wrapping_add((i64::from(acc) * i64::from(self.multiplier)) as u64);
        let mut r = ((v >> 32) as u32 as i32) >> (self.high_shift & 31);
        // Two tests in `apply`'s order: with qmin > qmax the result is qmax.
        if r < self.qmin {
            r = self.qmin;
        }
        if r > self.qmax {
            r = self.qmax;
        }
        r as i8
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

/// A quantized fully connected layer with one [`Requantizer`] per output: the
/// per-channel weight quantization TFLite applies to dense layers by default
/// (C++ `QPointwiseConv2DPerChannel` over a 1x1 input, which is a dense layer).
/// Otherwise the same as [`QDense`].
#[derive(Clone, Copy, Debug)]
pub struct QDensePerChannel<'a, const IN: usize, const OUT: usize> {
    /// `weights[o][i]`: weight from input `i` to output `o`.
    pub weights: &'a [[i8; IN]; OUT],
    /// One `i32` bias per output, or `None`.
    pub biases: Option<&'a [i32; OUT]>,
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// `requantizers[o]` rescales output `o`.
    pub requantizers: &'a [Requantizer; OUT],
}

impl<const IN: usize, const OUT: usize> QDensePerChannel<'_, IN, OUT> {
    /// One forward pass.
    pub fn forward(&self, input: &[i8; IN]) -> [i8; OUT] {
        let mut output = [0i8; OUT];
        let rows = self.weights.iter().zip(self.requantizers);
        for (o, (out, (row, rq))) in output.iter_mut().zip(rows).enumerate() {
            let bias = self.biases.map_or(0, |b| b.get(o).copied().unwrap_or(0));
            let acc = bias.wrapping_add(dot_with_zero_point(input, row, self.input_zero_point));
            *out = rq.apply(acc);
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

/// `x * 2^exponent`, saturating to the `i32` range, for `1 <= exponent <= 30`
/// (gemmlowp `SaturatingRoundingMultiplyByPOT`, C++
/// `fixedPointMulByPOTSaturating`).
fn mul_by_pot_saturating(x: i32, exponent: u32) -> i32 {
    let threshold = ((1u32 << (31 - exponent)) - 1) as i32;
    if x > threshold {
        i32::MAX
    } else if x < -threshold {
        i32::MIN
    } else {
        x * (1 << exponent)
    }
}

/// `exp(a)` for `a <= 0`: `a` in Q5.26, the result in Q0.31 (gemmlowp
/// `exp_on_negative_values`, C++ `fixedPointExpOnNegativeValues`). A Taylor
/// series around -1/8 covers the fraction within a quarter; each whole quarter
/// multiplies in a constant `exp(-2^k)`.
pub(crate) fn exp_on_negative_values(a: i32) -> i32 {
    const ONE_QUARTER: i32 = 1 << 24;
    // exp(-1/8) in Q0.31.
    const EXP_OF_MINUS_ONE_EIGHTH: i32 = 1_895_147_668;
    // exp(-1/4), exp(-1/2), exp(-1), exp(-2), exp(-4), exp(-8), exp(-16) in Q0.31.
    const EXP_OF_MINUS_POWER_OF_TWO: [i32; 7] = [
        1_672_461_947,
        1_302_514_674,
        790_015_084,
        290_630_308,
        39_332_535,
        720_401,
        242,
    ];

    let a_mod_quarter_minus_one_quarter = (a & (ONE_QUARTER - 1)) - ONE_QUARTER;

    // exp on [-1/4, 0): Taylor series around -1/8, in Q0.31. These values are
    // bounded whatever `a` is (an exhaustive test covers every fraction).
    let x = a_mod_quarter_minus_one_quarter * 32 + (1 << 28);
    let x2 = saturating_rounding_doubling_high_mul(x, x);
    let x3 = saturating_rounding_doubling_high_mul(x2, x);
    let x4 = saturating_rounding_doubling_high_mul(x2, x2);
    let x4_over_4 = rounding_divide_by_pot(x4, 2);
    let series = rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x4_over_4 + x3, 715_827_883) + x2,
        1,
    );
    let mut result = EXP_OF_MINUS_ONE_EIGHTH
        + saturating_rounding_doubling_high_mul(EXP_OF_MINUS_ONE_EIGHTH, x + series);

    // The whole quarters. Wraps only for a > 0, outside the domain.
    let remainder = a_mod_quarter_minus_one_quarter.wrapping_sub(a);
    for (k, &factor) in EXP_OF_MINUS_POWER_OF_TWO.iter().enumerate() {
        if remainder & (1 << (24 + k)) != 0 {
            result = saturating_rounding_doubling_high_mul(result, factor);
        }
    }

    if a == 0 { i32::MAX } else { result }
}

/// `1 / (1 + a)` for `a` in `[0, 1)`, both in Q0.31, by three Newton-Raphson
/// steps (gemmlowp `one_over_one_plus_x_for_x_in_0_1`, C++
/// `fixedPointOneOverOnePlusX`).
pub(crate) fn one_over_one_plus_x(a: i32) -> i32 {
    // (1 + a) / 2, rounded half away from zero; fits i32 for every a.
    let sum = i64::from(a) + i64::from(i32::MAX);
    let half_denominator = ((sum + if sum >= 0 { 1 } else { -1 }) / 2) as i32;

    // Newton-Raphson in Q2.29 from the initial guess 48/17 - 32/17 * d. The
    // sums stay in range for `a` in the domain; outside it they wrap.
    let mut x =
        1_515_870_810 + saturating_rounding_doubling_high_mul(half_denominator, -1_010_580_540);
    let one: i32 = 1 << 29;
    for _ in 0..3 {
        let one_minus_half_denominator_times_x =
            one.wrapping_sub(saturating_rounding_doubling_high_mul(half_denominator, x));
        x = x.wrapping_add(mul_by_pot_saturating(
            saturating_rounding_doubling_high_mul(x, one_minus_half_denominator_times_x),
            2,
        ));
    }
    mul_by_pot_saturating(x, 1)
}

/// Int8 softmax over a row of logits, bit-exact with TFLite's int8 softmax
/// (the TFLite Micro reference kernel, which CMSIS-NN's `arm_softmax_s8` also
/// reproduces), in gemmlowp fixed point (C++ `QSoftmaxFixedPoint1D`).
///
/// The output is on TFLite's probability grid, scale 1/256 and zero point
/// -128: -128 is 0 and 127 is 255/256. The three parameters depend only on the
/// input tensor's scale and the softmax's beta; compute them on the host with
/// `calibration::build_softmax` (`std` feature), as TFLite Micro does, or write
/// them out as constants.
///
/// For each row: subtract the maximum, take `exp((x - max) * input_scale *
/// beta)` in Q5.26 -> Q0.31 for every `x - max >= diff_min` (the rest
/// contribute 0), sum the terms in Q12.19, invert the sum by Newton-Raphson and
/// scale each term by it. Pure integer arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QSoftmax {
    /// Q0.31 multiplier of `input_scale * beta`.
    pub input_multiplier: i32,
    /// Left shift applied with the multiplier, `0..=31`.
    pub input_left_shift: i32,
    /// At most 0: a logit this far or further below the row's maximum gets
    /// probability 0.
    pub diff_min: i32,
}

impl QSoftmax {
    /// Softmax over one row of `N` int8 logits in the input tensor's grid. `N`
    /// must be in `1..512` (a compile error otherwise), as in TFLite: the sum
    /// of the exponentials must fit its Q12.19 accumulator.
    pub fn forward<const N: usize>(&self, input: &[i8; N]) -> [i8; N] {
        const { assert!(N > 0 && N < 512, "QSoftmax needs 1 to 511 logits per row") };
        const ACCUMULATION_INTEGER_BITS: i32 = 12;

        // Out of its documented range the shift is clamped and the scaled
        // difference wraps (the C++ would be undefined there); in range,
        // `diff * 2^shift` cannot overflow for any `diff >= diff_min` from
        // `calibration::build_softmax`.
        let left_shift = self.input_left_shift.clamp(0, 31) as u32;
        let exp_of = |diff: i32| {
            let scaled = saturating_rounding_doubling_high_mul(
                diff.wrapping_mul(1 << left_shift),
                self.input_multiplier,
            );
            exp_on_negative_values(scaled)
        };

        let max = i32::from(input.iter().copied().fold(i8::MIN, i8::max));

        // Each term is at most 1.0 = 2^19 in Q12.19, so with fewer than 512
        // terms the sum stays below 2^28.
        let mut sum_of_exps: i32 = 0;
        for &x in input {
            let diff = i32::from(x) - max;
            if diff >= self.diff_min {
                sum_of_exps = sum_of_exps.wrapping_add(rounding_divide_by_pot(
                    exp_of(diff),
                    ACCUMULATION_INTEGER_BITS,
                ));
            }
        }

        // The maximum contributes exactly 1.0, so the sum is at least 2^19:
        // normalise it to [1, 2) and invert. (A sum of 0, possible only with a
        // positive diff_min, leaves every output at -128.)
        let headroom_plus_one = (sum_of_exps as u32).leading_zeros();
        let shifted_sum_minus_one = (sum_of_exps as u32)
            .checked_shl(headroom_plus_one)
            .unwrap_or(0)
            .wrapping_sub(1 << 31) as i32;
        let shifted_scale = one_over_one_plus_x(shifted_sum_minus_one);
        let exponent = (ACCUMULATION_INTEGER_BITS - headroom_plus_one as i32) + 31 - 8;

        let mut output = [-128i8; N];
        for (out, &x) in output.iter_mut().zip(input) {
            let diff = i32::from(x) - max;
            if diff >= self.diff_min {
                // exponent >= 3, so the rounded value is within +-2^28.
                let q = rounding_divide_by_pot(
                    saturating_rounding_doubling_high_mul(shifted_scale, exp_of(diff)),
                    exponent,
                ) - 128;
                *out = q.clamp(-128, 127) as i8;
            }
        }
        output
    }
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

    /// tinymind (C++)'s `requantizer_saturating_path_at_int32_rails`, int8 part: the
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
    fn per_channel_dense_matches_dense_with_shared_requantizer() {
        let rq = Requantizer {
            multiplier: 1_200_000_000,
            shift: 6,
            zero_point: -3,
            qmin: -128,
            qmax: 127,
        };
        let weights = [[12, -7, 100], [-128, 5, 9]];
        let biases = [1000, -2000];
        let input = [-128, 0, 77];
        let shared = QDense {
            weights: &weights,
            biases: Some(&biases),
            input_zero_point: -5,
            requantizer: rq,
        };
        let per_channel = QDensePerChannel {
            weights: &weights,
            biases: Some(&biases),
            input_zero_point: -5,
            requantizers: &[rq, rq],
        };
        assert_eq!(per_channel.forward(&input), shared.forward(&input));
        // A different requantizer on output 1 changes only output 1.
        let other = Requantizer { shift: 4, ..rq };
        let mixed = QDensePerChannel {
            requantizers: &[rq, other],
            ..per_channel
        };
        let (a, b) = (mixed.forward(&input), shared.forward(&input));
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
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

    /// Every fraction the Taylor step of `exp_on_negative_values` can see (its
    /// result depends on `a` only through `a mod 2^24`), with overflow checks on
    /// in this debug build: none of its plain additions overflows. Also exp(0)
    /// is 1 and the result never increases as `a` falls.
    #[test]
    fn softmax_exp_every_fraction() {
        let mut previous = exp_on_negative_values(0);
        assert_eq!(previous, i32::MAX);
        for a in (-(1 << 24)..0).rev() {
            let e = exp_on_negative_values(a);
            assert!(
                e > 0 && e <= previous,
                "exp({a}) = {e}, previous {previous}"
            );
            previous = e;
        }
        // Accuracy against f64 at whole and fractional points in Q5.26.
        for a in [-(1 << 26), -(3 << 24), -(5 << 26) - 12_345, -(31 << 26)] {
            let got = f64::from(exp_on_negative_values(a)) / 2f64.powi(31);
            let want = (f64::from(a) / 2f64.powi(26)).exp();
            // gemmlowp (and so TFLite) is accurate to about 1e-7 here.
            assert!((got - want).abs() < 1e-6, "exp({a}): {got} vs {want}");
        }
    }

    /// `one_over_one_plus_x` across its domain [0, 1) in Q0.31, against f64.
    #[test]
    fn softmax_reciprocal() {
        for a in (0..i32::MAX).step_by(4093).chain([i32::MAX]) {
            let got = f64::from(one_over_one_plus_x(a)) / 2f64.powi(31);
            let want = 1.0 / (1.0 + f64::from(a) / 2f64.powi(31));
            assert!((got - want).abs() < 1e-8, "1/(1+{a}): {got} vs {want}");
        }
    }

    #[test]
    fn softmax_basics() {
        // Params for input scale 0.1, beta 1 (TFLite Micro computes these).
        let softmax = QSoftmax {
            input_multiplier: 1_717_986_918,
            input_left_shift: 23,
            diff_min: -248,
        };
        // One logit: probability 1, which saturates to 127.
        assert_eq!(softmax.forward(&[5]), [127]);
        // Order is kept, and the probabilities sum to about 1 (256 steps).
        let y = softmax.forward(&[10, 20, 30]);
        assert!(y[0] < y[1] && y[1] < y[2]);
        let total: i32 = y.iter().map(|&v| i32::from(v) + 128).sum();
        assert!((254..=258).contains(&total), "{y:?}");
        // A logit at or below diff_min from the maximum gets exactly 0.
        assert_eq!(softmax.forward(&[127, -128])[1], -128);
    }
}
