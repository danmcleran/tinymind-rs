//! The imported_mlp example's program, shared by `examples/imported_mlp.rs` and
//! `tests/imported_mlp_golden.rs` (which runs it in-process against the C++
//! output).

use std::io::{self, Write};

use tinymind::int8::calibration::quantize;

#[path = "imported_mlp_model.rs"]
mod model;

#[path = "imported_classifier_model.rs"]
mod classifier;

/// What the C++ check prints. For the MLP: each layer's integer requantizer, a
/// hash of the sigmoid table, and the int8 outputs for 16 test inputs. For the
/// classifier: its softmax parameters, then the probabilities and the predicted
/// class for 16 test inputs.
pub fn run(out: &mut dyn Write) -> io::Result<()> {
    run_mlp(out)?;
    run_classifier(out)
}

fn run_classifier(out: &mut dyn Write) -> io::Result<()> {
    let p = classifier::PROBS;
    writeln!(
        out,
        "classifier softmax={},{},{}",
        p.input_multiplier, p.input_left_shift, p.diff_min
    )?;
    for s in 0..16 {
        let input: [i8; classifier::NUM_INPUTS] = core::array::from_fn(|k| {
            let x = ((s * 19 + k * 11) % 33) as f32 / 16.0 - 1.0;
            quantize(
                x,
                classifier::INPUT_SCALE,
                i32::from(classifier::INPUT_ZERO_POINT),
                -128,
                127,
            )
        });
        let y = classifier::forward(&input);
        // The first of the most probable classes, as the C++ loop finds it.
        let mut best = 0;
        for (k, &v) in y.iter().enumerate() {
            if v > y[best] {
                best = k;
            }
        }
        writeln!(
            out,
            "in={},{},{},{} probs={},{},{} class={best}",
            input[0], input[1], input[2], input[3], y[0], y[1], y[2]
        )?;
    }
    Ok(())
}

fn run_mlp(out: &mut dyn Write) -> io::Result<()> {
    for (name, rq) in [
        ("fc1", model::FC1.requantizer),
        ("fc2", model::FC2.requantizer),
        ("fc3", model::FC3.requantizer),
    ] {
        writeln!(
            out,
            "{name} rq={},{},{},{},{}",
            rq.multiplier, rq.shift, rq.zero_point, rq.qmin, rq.qmax
        )?;
    }
    let mut h: u32 = 0x811C_9DC5;
    for v in model::OUT_LUT {
        h ^= u32::from(v as u8);
        h = h.wrapping_mul(0x0100_0193);
    }
    writeln!(out, "lut fnv=0x{h:08X}")?;

    for s in 0..16 {
        let input: [i8; model::NUM_INPUTS] = core::array::from_fn(|k| {
            let x = ((s * 23 + k * 29) % 37) as f32 / 18.0 - 1.0;
            quantize(
                x,
                model::INPUT_SCALE,
                i32::from(model::INPUT_ZERO_POINT),
                -128,
                127,
            )
        });
        let y = model::forward(&input);
        writeln!(
            out,
            "in={},{},{} out={},{}",
            input[0], input[1], input[2], y[0], y[1]
        )?;
    }
    Ok(())
}
