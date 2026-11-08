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
// uniform_real_distribution<float>, in the C++ order, and quantized with the
// same calibration; every tensor matches the C++ byte for byte
// (tests/kws_int8_golden.rs). Shared by examples/kws_cortex_m_int8.rs and the
// golden test.

use std::io::{self, Write};
use std::mem::size_of_val;

use tinymind::int8::calibration::{build_requantizer, per_channel_symmetric_scales, quantize};
use tinymind::int8::{
    QConv2D, QDepthwiseConv2D, QGlobalAvgPool2D, QMaxPool2D, QPointwiseConv2D, Requantizer,
};

#[path = "std_random.rs"]
pub mod std_random;
use std_random::{Mt19937, uniform_f32};

const QMIN: i32 = -127;
const QMAX: i32 = 127;
const ZERO_POINT: i32 = 0; // symmetric int8 activations
const ACT_SCALE: f32 = 1.0 / 127.0; // activations cover [-1, 1]
const W_SCALE: f32 = 0.1 / 127.0; // weights drawn from [-0.1, 0.1]

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

/// `n` draws from `[a, b)`, quantized to int8 with `scale` (C++
/// `calibratePerTensorWeights` / `quantizeBuffer` with zero point 0).
fn quantized_draws<const N: usize>(rng: &mut Mt19937, a: f32, b: f32, scale: f32) -> [i8; N] {
    let mut q = [0i8; N];
    for v in &mut q {
        *v = quantize(uniform_f32(rng, a, b), scale, ZERO_POINT, QMIN, QMAX);
    }
    q
}

/// Copy a flat array into a nested one of the same size, in memory order.
fn nest<T: Bytes + Copy, const N: usize>(
    flat: &[i8],
    zero: T,
    set: impl Fn(&mut T, &[i8]),
) -> [T; N] {
    let mut out = [zero; N];
    let per = flat.len() / N;
    for (o, chunk) in out.iter_mut().zip(flat.chunks(per)) {
        set(o, chunk);
    }
    out
}

/// Build the network as the C++ example does and run it once. `clock` returns
/// a timestamp in the C++ host counter's unit (nanoseconds, as `u32`); the
/// report goes to `out` in the C++ example's format, with this build's own
/// sizes and timings.
pub fn run(out: &mut impl Write, mut clock: impl FnMut() -> u32) -> io::Result<Tensors> {
    let mut rng = Mt19937::new(42);

    // Input: 400 draws from [-1, 1).
    let input_flat: [i8; 400] = quantized_draws(&mut rng, -1.0, 1.0, ACT_SCALE);
    let input: [[[i8; 1]; 20]; 20] = nest(&input_flat, [[0; 1]; 20], |row, c| {
        for (px, &v) in row.iter_mut().zip(c) {
            *px = [v];
        }
    });

    // Conv1: per-tensor symmetric weights (OHWI, one input channel).
    let conv1_flat: [i8; 72] = quantized_draws(&mut rng, -0.1, 0.1, W_SCALE);
    let w_conv1: [[[[i8; 1]; 3]; 3]; 8] = nest(&conv1_flat, [[[0; 1]; 3]; 3], |f, c| {
        for (k, &v) in c.iter().enumerate() {
            f[k / 3][k % 3] = [v];
        }
    });
    let rq_conv1 = build_requantizer(ACT_SCALE, W_SCALE, ACT_SCALE, ZERO_POINT, QMIN, QMAX);

    // Depthwise: per-channel symmetric scales from the float weights.
    let mut dw_float = [[0.0f32; 9]; 8];
    for v in dw_float.iter_mut().flatten() {
        *v = uniform_f32(&mut rng, -0.1, 0.1);
    }
    let dw_scales = per_channel_symmetric_scales(&dw_float, QMAX);
    let mut w_dw = [[[0i8; 3]; 3]; 8];
    let mut rq_dw = [rq_conv1; 8];
    for c in 0..8 {
        for (k, &v) in dw_float[c].iter().enumerate() {
            w_dw[c][k / 3][k % 3] = quantize(v, dw_scales[c], 0, QMIN, QMAX);
        }
        rq_dw[c] = build_requantizer(ACT_SCALE, dw_scales[c], ACT_SCALE, ZERO_POINT, QMIN, QMAX);
    }

    // Pointwise 8 -> 16 and dense 16 -> 10: per-tensor.
    let pw_flat: [i8; 128] = quantized_draws(&mut rng, -0.1, 0.1, W_SCALE);
    let w_pw: [[i8; 8]; 16] = nest(&pw_flat, [0; 8], |r, c| r.copy_from_slice(c));
    let rq_pw = build_requantizer(ACT_SCALE, W_SCALE, ACT_SCALE, ZERO_POINT, QMIN, QMAX);
    let dense_flat: [i8; 160] = quantized_draws(&mut rng, -0.1, 0.1, W_SCALE);
    let w_dense: [[i8; 16]; 10] = nest(&dense_flat, [0; 16], |r, c| r.copy_from_slice(c));
    let rq_dense = build_requantizer(ACT_SCALE, W_SCALE, ACT_SCALE, ZERO_POINT, QMIN, QMAX);

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

    write!(out, "\nLogits (int8): ")?;
    let logits = t.dense[0][0];
    let mut argmax = 0;
    for (c, &v) in logits.iter().enumerate() {
        if v > logits[argmax] {
            argmax = c;
        }
        write!(out, "{v} ")?;
    }
    writeln!(
        out,
        "\nPredicted class (untrained random weights): {argmax}"
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
