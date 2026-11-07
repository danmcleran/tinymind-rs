//! The int8 XOR example's program, shared by `examples/int8_xor.rs` and
//! `tests/int8_xor_golden.rs` (which runs it in-process against the C++
//! goldens). Port of tinymind `examples/pytorch_quant/xor/xor_quant.cpp`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::int8::calibration::{build_requantizer, build_sigmoid_lut, dequantize, quantize};
use tinymind::int8::{LUT_SIZE, QDense, apply_lut, qrelu};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::{fmt_fixed, fmt_g};

/// The trained model: tinymind `examples/pytorch_quant/xor/weights.hpp`, emitted
/// by its `xor_quant.py` from a PyTorch model. Copied by hand; any difference
/// fails the golden test and `scripts/compare-with-cpp.sh`.
// The float literals keep the C++ digits so the files compare by eye; they
// round to the same f32 values.
#[allow(clippy::excessive_precision)]
mod weights {
    pub const NUM_INPUTS: usize = 2;
    pub const HIDDEN_SIZE: usize = 4;
    pub const NUM_OUTPUTS: usize = 1;

    // Activation calibration (per-tensor, asymmetric int8).
    pub const INPUT_SCALE: f32 = 0.003_921_568_627_450_98; // 1/255
    pub const INPUT_ZERO_POINT: i32 = -128;
    pub const HIDDEN_SCALE: f32 = 0.007_843_137_254_901_96; // 2/255
    pub const HIDDEN_ZERO_POINT: i32 = -128;
    pub const LOGIT_SCALE: f32 = 0.015_686_274_509_803_92; // 4/255
    pub const LOGIT_ZERO_POINT: i32 = -1;
    pub const SIGMOID_OUT_SCALE: f32 = 0.003_906_25; // 1/256
    pub const SIGMOID_OUT_ZERO_POINT: i32 = -128;

    // Weight calibration (per-tensor, symmetric int8).
    pub const FC1_WEIGHT_SCALE: f32 = 0.007_874_015_748_031_496; // 1/127
    pub const FC2_WEIGHT_SCALE: f32 = 0.031_496_062_992_125_984; // 4/127

    pub static FC1_WEIGHTS: [[i8; NUM_INPUTS]; HIDDEN_SIZE] =
        [[127, 127], [127, 127], [127, 127], [127, 127]];
    pub static FC1_BIASES: [i32; HIDDEN_SIZE] = [0, -32385, 0, -32385];
    pub static FC2_WEIGHTS: [[i8; HIDDEN_SIZE]; NUM_OUTPUTS] = [[64, -127, 64, -127]];
    pub static FC2_BIASES: [i32; NUM_OUTPUTS] = [-8096];
}

use weights::*;

const XOR: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const XOR_LABELS: [i32; 4] = [0, 1, 1, 0];

/// The two-layer int8 network: fc1, quantized ReLU, fc2, sigmoid table.
struct Model {
    fc1: QDense<'static, NUM_INPUTS, HIDDEN_SIZE>,
    fc2: QDense<'static, HIDDEN_SIZE, NUM_OUTPUTS>,
    sigmoid: [i8; LUT_SIZE],
}

impl Model {
    /// Rebuild the requantizers and the sigmoid table from the float scales, as
    /// the C++ example does. An inference-only firmware build would embed the
    /// resulting integers instead.
    fn new() -> Self {
        Model {
            fc1: QDense {
                weights: &FC1_WEIGHTS,
                biases: Some(&FC1_BIASES),
                input_zero_point: INPUT_ZERO_POINT as i8,
                requantizer: build_requantizer(
                    INPUT_SCALE,
                    FC1_WEIGHT_SCALE,
                    HIDDEN_SCALE,
                    HIDDEN_ZERO_POINT,
                    -128,
                    127,
                ),
            },
            fc2: QDense {
                weights: &FC2_WEIGHTS,
                biases: Some(&FC2_BIASES),
                input_zero_point: HIDDEN_ZERO_POINT as i8,
                requantizer: build_requantizer(
                    HIDDEN_SCALE,
                    FC2_WEIGHT_SCALE,
                    LOGIT_SCALE,
                    LOGIT_ZERO_POINT,
                    -128,
                    127,
                ),
            },
            sigmoid: build_sigmoid_lut(
                LOGIT_SCALE,
                LOGIT_ZERO_POINT,
                SIGMOID_OUT_SCALE,
                SIGMOID_OUT_ZERO_POINT,
            ),
        }
    }

    fn quantize_input(x: [f32; NUM_INPUTS]) -> [i8; NUM_INPUTS] {
        x.map(|v| quantize(v, INPUT_SCALE, INPUT_ZERO_POINT, -128, 127))
    }

    /// Returns the int8 logit and the int8 sigmoid output.
    fn predict(&self, input: &[i8; NUM_INPUTS]) -> (i8, i8) {
        let hidden = self
            .fc1
            .forward(input)
            .map(|h| qrelu(h, HIDDEN_ZERO_POINT as i8));
        let logit = self.fc2.forward(&hidden)[0];
        (logit, apply_lut(logit, &self.sigmoid))
    }
}

/// Run the example: print the four XOR cases to `stdout` and write
/// `xor_decision_surface.csv` into `out_dir`. Returns how many of the four
/// were classified correctly.
pub fn run(out_dir: &Path, stdout: &mut dyn Write) -> io::Result<usize> {
    let model = Model::new();

    writeln!(stdout, "PyTorch -> tinymind int8 XOR inference")?;
    writeln!(
        stdout,
        "  input_scale={} input_zp={} hidden_scale={} hidden_zp={}",
        fmt_fixed(INPUT_SCALE.into(), 4),
        INPUT_ZERO_POINT,
        fmt_fixed(HIDDEN_SCALE.into(), 4),
        HIDDEN_ZERO_POINT
    )?;
    writeln!(
        stdout,
        "  logit_scale={} logit_zp={} out_scale={} out_zp={}\n",
        fmt_fixed(LOGIT_SCALE.into(), 4),
        LOGIT_ZERO_POINT,
        fmt_fixed(SIGMOID_OUT_SCALE.into(), 4),
        SIGMOID_OUT_ZERO_POINT
    )?;

    let mut correct = 0;
    for (x, label) in XOR.iter().zip(XOR_LABELS) {
        let input = Model::quantize_input(*x);
        let (logit, out) = model.predict(&input);
        let prob = dequantize(out, SIGMOID_OUT_SCALE, SIGMOID_OUT_ZERO_POINT);
        let predicted = i32::from(prob > 0.5);
        if predicted == label {
            correct += 1;
        }
        writeln!(
            stdout,
            "  in=({},{})  q_input=({},{})  q_logit={}  prob={}  pred={}  label={}",
            fmt_fixed(x[0].into(), 4),
            fmt_fixed(x[1].into(), 4),
            input[0],
            input[1],
            logit,
            fmt_fixed(prob.into(), 4),
            predicted,
            label
        )?;
    }
    writeln!(stdout, "\nint8 XOR accuracy: {correct}/4")?;

    // Decision surface over the [0,1]^2 grid, in f32 as the C++ computes it.
    let mut surf = BufWriter::new(File::create(out_dir.join("xor_decision_surface.csv"))?);
    writeln!(surf, "x0,x1,prob")?;
    const G: i32 = 41;
    for a in 0..G {
        for b in 0..G {
            let x0 = a as f32 / (G - 1) as f32;
            let x1 = b as f32 / (G - 1) as f32;
            let (_, out) = model.predict(&Model::quantize_input([x0, x1]));
            let prob = dequantize(out, SIGMOID_OUT_SCALE, SIGMOID_OUT_ZERO_POINT);
            writeln!(
                surf,
                "{},{},{}",
                fmt_g(x0.into()),
                fmt_g(x1.into()),
                fmt_g(prob.into())
            )?;
        }
    }
    surf.flush()?;
    stdout.flush()?;
    Ok(correct)
}
