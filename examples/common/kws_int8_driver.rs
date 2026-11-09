// The program of tinymind (C++)'s examples/kws_cortex_m_int8: a keyword-spotting
// shaped int8 network with synthetic weights, calibrated on the host and run
// layer by layer.
//
//   input [20 x 20 x 1]
//     -> QConv2D 3x3, 8 filters           -> [18 x 18 x 8]
//     -> QMaxPool2D 2x2                   -> [9 x 9 x 8]
//     -> QDepthwiseConv2D 3x3 (per-chan)  -> [7 x 7 x 8]
//     -> QPointwiseConv2D 8 -> 16         -> [7 x 7 x 16]
//     -> QGlobalAvgPool2D                 -> [16]
//     -> QPointwiseConv2D (dense) 16 -> 10 logits
//
// Weights and inputs are drawn from std::mt19937(42) through libstdc++'s
// uniform_real_distribution<float>, in the C++ order. Each layer's activation
// scale is fitted from a float reference forward pass over 32 calibration
// inputs (RangeObserver), as the C++ does since tinymind (C++) #255. Every
// tensor and the logits report match the C++ byte for byte
// (tests/kws_int8_golden.rs). Shared by examples/kws_cortex_m_int8.rs and the
// golden test.

use std::io::{self, Write};
use std::mem::size_of_val;

use tinymind::int8::calibration::{
    RangeObserver, affine_params_symmetric, build_requantizer, dequantize,
    per_channel_symmetric_scales, quantize,
};
use tinymind::int8::{
    QConv2D, QDepthwiseConv2D, QGlobalAvgPool2D, QMaxPool2D, QPointwiseConv2D, Requantizer,
};

#[path = "std_random.rs"]
pub mod std_random;
use std_random::{Mt19937, uniform_f32};

const QMIN: i32 = -127;
const QMAX: i32 = 127;
const ZERO_POINT: i32 = 0; // symmetric int8 activations
const IN_SCALE: f32 = 1.0 / 127.0; // the input covers [-1, 1]
const W_SCALE: f32 = 0.1 / 127.0; // weights drawn from [-0.1, 0.1]
/// Calibration inputs, drawn after the weights.
const CALIBRATION_SAMPLES: usize = 32;

/// Every tensor of one run: the quantized input, weights and requantizers, and
/// each layer's output.
pub struct Tensors {
    pub input: [[[i8; 1]; 20]; 20],
    pub w_conv1: [[[[i8; 1]; 3]; 3]; 8],
    pub rq_conv1: Requantizer,
    pub w_dw: [[[i8; 3]; 3]; 8],
    pub rq_dw: [Requantizer; 8],
    pub w_pw: [[i8; 8]; 16],
    pub rq_pw: Requantizer,
    pub w_dense: [[i8; 16]; 10],
    pub rq_dense: Requantizer,
    pub conv1: [[[i8; 8]; 18]; 18],
    pub pool1: [[[i8; 8]; 9]; 9],
    pub dw: [[[i8; 8]; 7]; 7],
    pub pw: [[[i8; 16]; 7]; 7],
    pub gap: [i8; 16],
    pub dense: [[[i8; 10]; 1]; 1],
}

/// The bytes of a nested `i8` array in memory order (row-major, as the C++
/// lays out its flat buffers).
pub trait Bytes {
    fn bytes(&self) -> Vec<i8>;
}

impl Bytes for i8 {
    fn bytes(&self) -> Vec<i8> {
        vec![*self]
    }
}

impl<T: Bytes, const N: usize> Bytes for [T; N] {
    fn bytes(&self) -> Vec<i8> {
        self.iter().flat_map(Bytes::bytes).collect()
    }
}

/// The float model: the same weights before quantization (biases are 0), in
/// the int8 layers' layouts.
struct FloatModel {
    conv1: [[[f32; 3]; 3]; 8],
    dw: [[[f32; 3]; 3]; 8],
    pw: [[f32; 8]; 16],
    dense: [[f32; 16]; 10],
}

/// The float outputs the calibration observes.
struct FloatActivations {
    conv1: [[[f32; 8]; 18]; 18],
    dw: [[[f32; 8]; 7]; 7],
    pw: [[[f32; 16]; 7]; 7],
    dense: [f32; 10],
}

/// The C++ example's float reference forward pass, summing in its order (an
/// f32 sum depends on it; Rust does not contract `a * b + c` into a fused
/// multiply-add, and the C++ reference is built with `-ffp-contract=off`).
fn float_forward(m: &FloatModel, input: &[[f32; 20]; 20]) -> FloatActivations {
    let mut conv1 = [[[0.0f32; 8]; 18]; 18];
    for (y, row) in conv1.iter_mut().enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            for (out, w) in pixel.iter_mut().zip(&m.conv1) {
                let mut acc = 0.0f32;
                for (ky, w_row) in w.iter().enumerate() {
                    for (kx, &wv) in w_row.iter().enumerate() {
                        acc += input[y + ky][x + kx] * wv;
                    }
                }
                *out = acc;
            }
        }
    }
    let mut pool1 = [[[0.0f32; 8]; 9]; 9];
    for (y, row) in pool1.iter_mut().enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            for (c, out) in pixel.iter_mut().enumerate() {
                let mut v = conv1[2 * y][2 * x][c];
                for py in 0..2 {
                    for px in 0..2 {
                        let candidate = conv1[2 * y + py][2 * x + px][c];
                        if candidate > v {
                            v = candidate;
                        }
                    }
                }
                *out = v;
            }
        }
    }
    let mut dw = [[[0.0f32; 8]; 7]; 7];
    for (y, row) in dw.iter_mut().enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            for (c, (out, w)) in pixel.iter_mut().zip(&m.dw).enumerate() {
                let mut acc = 0.0f32;
                for (ky, w_row) in w.iter().enumerate() {
                    for (kx, &wv) in w_row.iter().enumerate() {
                        acc += pool1[y + ky][x + kx][c] * wv;
                    }
                }
                *out = acc;
            }
        }
    }
    let mut pw = [[[0.0f32; 16]; 7]; 7];
    for (out_pixel, in_pixel) in pw.iter_mut().flatten().zip(dw.iter().flatten()) {
        for (out, w) in out_pixel.iter_mut().zip(&m.pw) {
            let mut acc = 0.0f32;
            for (&x, &wv) in in_pixel.iter().zip(w) {
                acc += x * wv;
            }
            *out = acc;
        }
    }
    let mut gap = [0.0f32; 16];
    for (f, out) in gap.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for pixel in pw.iter().flatten() {
            sum += pixel[f];
        }
        *out = sum / 49.0;
    }
    let mut dense = [0.0f32; 10];
    for (out, w) in dense.iter_mut().zip(&m.dense) {
        let mut acc = 0.0f32;
        for (&x, &wv) in gap.iter().zip(w) {
            acc += x * wv;
        }
        *out = acc;
    }
    FloatActivations {
        conv1,
        dw,
        pw,
        dense,
    }
}

/// The symmetric int8 scale covering an observed range.
fn activation_scale(obs: &RangeObserver) -> f32 {
    affine_params_symmetric(obs.min_value, obs.max_value, QMAX).0
}

/// The first index of the largest value.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Build the network as the C++ example does and run it once. `clock` returns
/// a timestamp in the C++ host counter's unit (nanoseconds, as `u32`); the
/// report goes to `out` in the C++ example's format, with this build's own
/// sizes and timings.
pub fn run(out: &mut impl Write, mut clock: impl FnMut() -> u32) -> io::Result<Tensors> {
    let mut rng = Mt19937::new(42);

    // Input: 400 draws from [-1, 1), kept in float for the reference pass.
    let mut input_f = [[0.0f32; 20]; 20];
    let mut input = [[[0i8; 1]; 20]; 20];
    for (f, q) in input_f.iter_mut().flatten().zip(input.iter_mut().flatten()) {
        *f = uniform_f32(&mut rng, -1.0, 1.0);
        *q = [quantize(*f, IN_SCALE, ZERO_POINT, QMIN, QMAX)];
    }

    // Weights. Conv1, pointwise and dense use one symmetric weight scale per
    // tensor; depthwise one per channel (TFLite mandate).
    let mut m = FloatModel {
        conv1: [[[0.0; 3]; 3]; 8],
        dw: [[[0.0; 3]; 3]; 8],
        pw: [[0.0; 8]; 16],
        dense: [[0.0; 16]; 10],
    };
    for w in m.conv1.iter_mut().flatten().flatten() {
        *w = uniform_f32(&mut rng, -0.1, 0.1);
    }
    for w in m.dw.iter_mut().flatten().flatten() {
        *w = uniform_f32(&mut rng, -0.1, 0.1);
    }
    for w in m.pw.iter_mut().flatten() {
        *w = uniform_f32(&mut rng, -0.1, 0.1);
    }
    for w in m.dense.iter_mut().flatten() {
        *w = uniform_f32(&mut rng, -0.1, 0.1);
    }
    let q = |v: f32, scale: f32| quantize(v, scale, 0, QMIN, QMAX);
    let w_conv1 = m.conv1.map(|f| f.map(|r| r.map(|v| [q(v, W_SCALE)])));
    let dw_by_channel = m.dw.map(|c| {
        let mut flat = [0.0f32; 9];
        for (o, &v) in flat.iter_mut().zip(c.iter().flatten()) {
            *o = v;
        }
        flat
    });
    let dw_scales = per_channel_symmetric_scales(&dw_by_channel, QMAX);
    let mut w_dw = [[[0i8; 3]; 3]; 8];
    for ((out, w), &scale) in w_dw.iter_mut().zip(&m.dw).zip(&dw_scales) {
        *out = w.map(|r| r.map(|v| q(v, scale)));
    }
    let w_pw = m.pw.map(|r| r.map(|v| q(v, W_SCALE)));
    let w_dense = m.dense.map(|r| r.map(|v| q(v, W_SCALE)));

    // Activation scales: the float model over calibration inputs gives each
    // layer's output the symmetric scale covering its observed range. Max
    // pooling and global average pooling have no requantizer, so their
    // outputs keep their input's scale.
    let mut obs_conv1 = RangeObserver::new();
    let mut obs_dw = RangeObserver::new();
    let mut obs_pw = RangeObserver::new();
    let mut obs_dense = RangeObserver::new();
    for _ in 0..CALIBRATION_SAMPLES {
        let mut cal = [[0.0f32; 20]; 20];
        for v in cal.iter_mut().flatten() {
            *v = uniform_f32(&mut rng, -1.0, 1.0);
        }
        let a = float_forward(&m, &cal);
        a.conv1
            .iter()
            .flatten()
            .for_each(|p| obs_conv1.observe_all(p));
        a.dw.iter().flatten().for_each(|p| obs_dw.observe_all(p));
        a.pw.iter().flatten().for_each(|p| obs_pw.observe_all(p));
        obs_dense.observe_all(&a.dense);
    }
    let conv1_scale = activation_scale(&obs_conv1);
    let dw_scale = activation_scale(&obs_dw);
    let pw_scale = activation_scale(&obs_pw);
    let dense_scale = activation_scale(&obs_dense);

    let rq = |input: f32, weight: f32, output: f32| {
        build_requantizer(input, weight, output, ZERO_POINT, QMIN, QMAX)
    };
    let rq_conv1 = rq(IN_SCALE, W_SCALE, conv1_scale);
    // Depthwise reads the max pool's output, at conv1's scale.
    let rq_dw = dw_scales.map(|s| rq(conv1_scale, s, dw_scale));
    let rq_pw = rq(dw_scale, W_SCALE, pw_scale);
    // Dense reads the average pool's output, at the pointwise layer's scale.
    let rq_dense = rq(pw_scale, W_SCALE, dense_scale);

    // All-zero biases, as the C++ sets them.
    let (b_conv1, b_dw, b_pw, b_dense) = ([0i32; 8], [0i32; 8], [0i32; 16], [0i32; 10]);
    let zp = ZERO_POINT as i8;
    let conv1: QConv2D<'_, 1, 3, 3, 8> = QConv2D {
        weights: &w_conv1,
        biases: Some(&b_conv1),
        input_zero_point: zp,
        requantizer: rq_conv1,
    };
    let pool1 = QMaxPool2D::<2, 2>;
    let dw: QDepthwiseConv2D<'_, 8, 3, 3> = QDepthwiseConv2D {
        weights: &w_dw,
        biases: Some(&b_dw),
        input_zero_point: zp,
        requantizers: &rq_dw,
    };
    let pw = QPointwiseConv2D {
        weights: &w_pw,
        biases: Some(&b_pw),
        input_zero_point: zp,
        requantizer: rq_pw,
    };
    let gap = QGlobalAvgPool2D {
        qmin: QMIN as i8,
        qmax: QMAX as i8,
    };
    let dense = QPointwiseConv2D {
        weights: &w_dense,
        biases: Some(&b_dense),
        input_zero_point: zp,
        requantizer: rq_dense,
    };

    let mut t = Tensors {
        input,
        w_conv1,
        rq_conv1,
        w_dw,
        rq_dw,
        w_pw,
        rq_pw,
        w_dense,
        rq_dense,
        conv1: [[[0; 8]; 18]; 18],
        pool1: [[[0; 8]; 9]; 9],
        dw: [[[0; 8]; 7]; 7],
        pw: [[[0; 16]; 7]; 7],
        gap: [0; 16],
        dense: [[[0; 10]; 1]; 1],
    };

    let t0 = clock();
    conv1.forward(&t.input, &mut t.conv1);
    let t1 = clock();
    pool1.forward(&t.conv1, &mut t.pool1);
    let t2 = clock();
    dw.forward(&t.pool1, &mut t.dw);
    let t3 = clock();
    pw.forward(&t.dw, &mut t.pw);
    let t4 = clock();
    t.gap = gap.forward(&t.pw);
    let t5 = clock();
    dense.forward(&[[t.gap]], &mut t.dense);
    let t6 = clock();

    // Weight bytes as the C++ counts them: the layer itself plus the weight,
    // bias and requantizer arrays it points to. (The max-pool layer is
    // zero-sized here; the C++ empty struct is one byte.)
    let conv1_bytes = size_of_val(&conv1) + size_of_val(&t.w_conv1) + size_of_val(&b_conv1);
    let dw_bytes =
        size_of_val(&dw) + size_of_val(&t.w_dw) + size_of_val(&b_dw) + size_of_val(&t.rq_dw);
    let pw_bytes = size_of_val(&pw) + size_of_val(&t.w_pw) + size_of_val(&b_pw);
    let dense_bytes = size_of_val(&dense) + size_of_val(&t.w_dense) + size_of_val(&b_dense);
    let rows = [
        (
            "qconv2d_3x3_8",
            conv1_bytes,
            size_of_val(&t.conv1),
            t1.wrapping_sub(t0),
        ),
        (
            "qmaxpool2d_2x2",
            size_of_val(&pool1),
            size_of_val(&t.pool1),
            t2.wrapping_sub(t1),
        ),
        (
            "qdwconv2d_3x3",
            dw_bytes,
            size_of_val(&t.dw),
            t3.wrapping_sub(t2),
        ),
        (
            "qpwconv2d_8x16",
            pw_bytes,
            size_of_val(&t.pw),
            t4.wrapping_sub(t3),
        ),
        (
            "qglobal_avgpool2d",
            size_of_val(&gap),
            size_of_val(&t.gap),
            t5.wrapping_sub(t4),
        ),
        (
            "qdense_16x10",
            dense_bytes,
            size_of_val(&t.dense),
            t6.wrapping_sub(t5),
        ),
    ];
    writeln!(out, "name,weight_bytes,activation_bytes,cycles")?;
    for (name, weights, activations, cycles) in rows {
        writeln!(out, "{name},{weights},{activations},{cycles}")?;
    }
    let total_weights: usize = rows.iter().map(|r| r.1).sum();
    writeln!(out, "\nSummary:")?;
    writeln!(out, "  total weight bytes     : {total_weights}")?;
    writeln!(out, "  peak activation bytes  : {}", size_of_val(&t.conv1))?;
    writeln!(out, "  total inference cycles : {}", t6.wrapping_sub(t0))?;

    // The int8 logits against the float model on the same input. The weights
    // are untrained, so the class index means nothing; the comparison shows
    // the int8 pipeline tracks the float one.
    let reference = float_forward(&m, &input_f).dense;
    let mut logits = [0.0f32; 10];
    let mut max_err = 0.0f32;
    let (mut ref_min, mut ref_max) = (reference[0], reference[0]);
    writeln!(out, "\nLogits (int8 -> dequantized, float reference):")?;
    let rows = logits.iter_mut().zip(&t.dense[0][0]).zip(&reference);
    for (c, ((logit, &q), &r)) in rows.enumerate() {
        *logit = dequantize(q, dense_scale, ZERO_POINT);
        let err = (*logit - r).abs();
        if err > max_err {
            max_err = err;
        }
        if r < ref_min {
            ref_min = r;
        }
        if r > ref_max {
            ref_max = r;
        }
        writeln!(out, "  class {c}: {q:4} -> {logit:9.6}   float {r:9.6}")?;
    }
    let share = 100.0f32 * max_err / (ref_max - ref_min);
    writeln!(
        out,
        "max |int8 - float| = {max_err:.6} ({share:.1}% of the float logits' range)"
    )?;
    writeln!(
        out,
        "argmax (untrained weights): int8 {}, float {}",
        argmax(&logits),
        argmax(&reference)
    )?;
    Ok(t)
}

/// One line of the dump: a name, then its values.
fn line(name: &str, values: impl IntoIterator<Item = i64>) -> String {
    let mut l = String::from(name);
    for v in values {
        l.push_str(&format!(" {v}"));
    }
    l.push('\n');
    l
}

/// Every tensor in the format of scripts/cpp/kws_int8_dump.cpp: one line per
/// tensor (name, then its values in memory order) or requantizer (multiplier,
/// shift, zero point, qmin, qmax).
pub fn dump(t: &Tensors) -> String {
    let tensor = |name: &str, v: Vec<i8>| line(name, v.into_iter().map(i64::from));
    let rq = |name: &str, r: &Requantizer| {
        let v = [
            r.multiplier,
            r.shift,
            r.zero_point.into(),
            r.qmin.into(),
            r.qmax.into(),
        ];
        line(name, v.map(i64::from))
    };
    let mut s = String::new();
    s += &tensor("input", t.input.bytes());
    s += &tensor("w_conv1", t.w_conv1.bytes());
    s += &rq("rq_conv1", &t.rq_conv1);
    s += &tensor("w_dw", t.w_dw.bytes());
    for r in &t.rq_dw {
        s += &rq("rq_dw", r);
    }
    s += &tensor("w_pw", t.w_pw.bytes());
    s += &rq("rq_pw", &t.rq_pw);
    s += &tensor("w_dense", t.w_dense.bytes());
    s += &rq("rq_dense", &t.rq_dense);
    s += &tensor("conv1", t.conv1.bytes());
    s += &tensor("pool1", t.pool1.bytes());
    s += &tensor("dw", t.dw.bytes());
    s += &tensor("pw", t.pw.bytes());
    s += &tensor("gap", t.gap.bytes());
    s += &tensor("dense", t.dense.bytes());
    s
}
