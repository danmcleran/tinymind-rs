//! The onnx_digits_int8 example's program, shared by
//! `examples/onnx_digits_int8.rs` (which parses the command line) and
//! `tests/onnx_digits_golden.rs` (which runs it in-process against the C++
//! goldens). Port of tinymind (C++) `examples/onnx_digits_int8/onnx_digits_int8.cpp`;
//! see `examples/onnx_digits_int8.rs`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::fmt_fixed;

#[path = "onnx_digits_model.rs"]
mod model;
#[path = "onnx_digits_16x8_model.rs"]
mod model16;
#[path = "onnx_digits_int4_model.rs"]
mod model4;
#[path = "onnx_digits_reference.rs"]
mod reference;

const RESULTS_CSV: &str = "onnx_digits_int8.csv";
const PLAN_CSV: &str = "onnx_digits_int8_plan.csv";
const SUMMARY_CSV: &str = "onnx_digits_int8_summary.csv";
const LOGITS_CSV: &str = "onnx_digits_int8_logits.csv";

const PIXELS: usize = 64;
const CLASSES: usize = 10;

/// One test digit: its pixels (0..16) and label.
struct Digit {
    pixels: [u8; PIXELS],
    label: usize,
}

/// `optdigits.tes`: 64 pixel values and the label per line.
fn load(path: &Path) -> io::Result<Vec<Digit>> {
    let bad = |what: String| io::Error::new(io::ErrorKind::InvalidData, what);
    let text = fs::read_to_string(path)
        .map_err(|e| io::Error::new(e.kind(), format!("cannot open {}", path.display())))?;
    let mut digits = Vec::new();
    for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let values: Vec<u8> = line
            .split(',')
            .map(|v| v.trim().parse::<u8>())
            .collect::<Result<_, _>>()
            .map_err(|_| bad(format!("{}: bad value in row {n}", path.display())))?;
        let (label, pixels) = values
            .split_last()
            .ok_or_else(|| bad(format!("{}: empty row {n}", path.display())))?;
        let pixels: [u8; PIXELS] = pixels.try_into().map_err(|_| {
            bad(format!(
                "{}: row {n} has {} pixels",
                path.display(),
                pixels.len()
            ))
        })?;
        if pixels.iter().any(|&p| p > 16) {
            return Err(bad(format!("{}: bad pixel in row {n}", path.display())));
        }
        digits.push(Digit {
            pixels,
            label: usize::from(*label),
        });
    }
    if digits.len() < reference::TEST_COUNT {
        return Err(bad(format!(
            "{}: {} rows, expected {}",
            path.display(),
            digits.len(),
            reference::TEST_COUNT
        )));
    }
    digits.truncate(reference::TEST_COUNT);
    Ok(digits)
}

/// The first index of the largest logit, as the C++ `argmax`.
fn argmax<T: PartialOrd + Copy>(logits: &[T; CLASSES]) -> usize {
    let mut best = 0;
    for c in 1..CLASSES {
        if logits[c] > logits[best] {
            best = c;
        }
    }
    best
}

/// Run the three generated models on every test digit in `test_set`; write
/// the four CSV files into `out_dir` and the summary to `stdout`.
pub fn run(out_dir: &Path, test_set: &Path, stdout: &mut dyn Write) -> io::Result<()> {
    let digits = load(test_set)?;
    let mut arena: model::Arena = [0; model::ARENA_BYTES];
    let mut arena4: model4::Arena = [0; model4::ARENA_BYTES];
    let mut arena16: model16::Arena = [0; model16::ARENA_BYTES / 2];

    let mut results = BufWriter::new(File::create(out_dir.join(RESULTS_CSV))?);
    write!(
        results,
        "index,label,tinymind,tinymind_int4,tinymind_16x8,onnxruntime,float"
    )?;
    for p in 0..PIXELS {
        write!(results, ",p{p}")?;
    }
    writeln!(results)?;
    let mut logits_csv = BufWriter::new(File::create(out_dir.join(LOGITS_CSV))?);
    writeln!(logits_csv, "index,class,reference,int8,int16x8")?;

    let (mut correct, mut correct4, mut agree4, mut agree_ort) = (0usize, 0usize, 0usize, 0usize);
    let (mut correct_ort, mut correct_float, mut correct16) = (0usize, 0usize, 0usize);
    let (mut agree_float, mut agree_float16) = (0usize, 0usize);
    let (mut error8, mut error16, mut worst8, mut worst16) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (n, digit) in digits.iter().enumerate() {
        // One input channel, so the row-major 8x8 image is already NHWC.
        let image: model::Input = core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                [reference::PIXEL_TO_INPUT[usize::from(digit.pixels[r * 8 + c])]]
            })
        });
        let image16: model16::Input = core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                [reference::PIXEL_TO_INPUT16[usize::from(digit.pixels[r * 8 + c])]]
            })
        });
        let label = digit.label;
        let logits = model::forward(&image, &mut arena);
        let predicted = argmax(&logits);
        let logits16 = model16::forward(&image16, &mut arena16);
        let predicted16 = argmax(&logits16);
        let float_class = usize::from(reference::FLOAT_PREDICTION[n]);
        correct16 += usize::from(predicted16 == label);
        agree_float += usize::from(predicted == float_class);
        agree_float16 += usize::from(predicted16 == float_class);
        if n < reference::REFERENCE_LOGIT_COUNT {
            // Logits in real units, against the int8-weight model run in float.
            for c in 0..CLASSES {
                let r = f64::from(reference::REFERENCE_LOGITS[n * CLASSES + c]);
                let v8 = f64::from(i32::from(logits[c]) - i32::from(model::OUTPUT_ZERO_POINT))
                    * f64::from(model::OUTPUT_SCALE);
                let v16 = f64::from(logits16[c]) * f64::from(model16::OUTPUT_SCALE);
                let e8 = if v8 > r { v8 - r } else { r - v8 };
                let e16 = if v16 > r { v16 - r } else { r - v16 };
                error8 += e8;
                error16 += e16;
                worst8 = if e8 > worst8 { e8 } else { worst8 };
                worst16 = if e16 > worst16 { e16 } else { worst16 };
                writeln!(
                    logits_csv,
                    "{n},{c},{},{},{}",
                    fmt_fixed(r, 6),
                    fmt_fixed(v8, 6),
                    fmt_fixed(v16, 6)
                )?;
            }
        }
        let predicted4 = argmax(&model4::forward(&image, &mut arena4));
        correct4 += usize::from(predicted4 == label);
        agree4 += usize::from(predicted4 == predicted);
        let ort = usize::from(reference::ORT_PREDICTION[n]);
        correct += usize::from(predicted == label);
        agree_ort += usize::from(predicted == ort);
        correct_ort += usize::from(ort == label);
        correct_float += usize::from(float_class == label);
        write!(
            results,
            "{n},{label},{predicted},{predicted4},{predicted16},{ort},{float_class}"
        )?;
        for p in digit.pixels {
            write!(results, ",{p}")?;
        }
        writeln!(results)?;
    }
    results.flush()?;
    logits_csv.flush()?;
    let reference_count = (reference::REFERENCE_LOGIT_COUNT * CLASSES) as f64;

    let mut plan = BufWriter::new(File::create(out_dir.join(PLAN_CSV))?);
    writeln!(plan, "offset,size,first_layer,last_layer,outputs")?;
    for b in model::BUFFERS {
        writeln!(
            plan,
            "{},{},{},{},{}",
            b.offset, b.size, b.first_layer, b.last_layer, b.outputs
        )?;
    }
    plan.flush()?;

    let count = reference::TEST_COUNT;
    let total = count as f64;
    let accuracy = |k: usize| fmt_fixed(k as f64 / total, 4);
    let mut summary = BufWriter::new(File::create(out_dir.join(SUMMARY_CSV))?);
    writeln!(summary, "metric,value")?;
    writeln!(summary, "layers,{}", model::LAYER_NAMES.len())?;
    writeln!(summary, "arena_bytes,{}", model::ARENA_BYTES)?;
    writeln!(summary, "unshared_bytes,{}", model::UNSHARED_BYTES)?;
    writeln!(summary, "busiest_layer_bytes,{}", model::PEAK_LIVE_BYTES)?;
    writeln!(summary, "weight_bytes_int8,{}", model::WEIGHT_BYTES)?;
    writeln!(summary, "weight_bytes_int4,{}", model4::WEIGHT_BYTES)?;
    writeln!(summary, "accuracy_int8,{}", accuracy(correct))?;
    writeln!(summary, "accuracy_int4,{}", accuracy(correct4))?;
    writeln!(summary, "accuracy_16x8,{}", accuracy(correct16))?;
    writeln!(summary, "arena_bytes_16x8,{}", model16::ARENA_BYTES)?;
    writeln!(summary, "same_class_as_float_int8,{agree_float}")?;
    writeln!(summary, "same_class_as_float_16x8,{agree_float16}")?;
    writeln!(
        summary,
        "mean_logit_error_int8,{}",
        fmt_fixed(error8 / reference_count, 6)
    )?;
    writeln!(
        summary,
        "mean_logit_error_16x8,{}",
        fmt_fixed(error16 / reference_count, 6)
    )?;
    writeln!(summary, "max_logit_error_int8,{}", fmt_fixed(worst8, 6))?;
    writeln!(summary, "max_logit_error_16x8,{}", fmt_fixed(worst16, 6))?;
    summary.flush()?;

    writeln!(stdout, "test digits:                    {count}")?;
    writeln!(
        stdout,
        "tinymind int8 accuracy:         {}",
        accuracy(correct)
    )?;
    writeln!(
        stdout,
        "ONNX Runtime int8 accuracy:     {}",
        accuracy(correct_ort)
    )?;
    writeln!(
        stdout,
        "float (PyTorch) accuracy:       {}",
        accuracy(correct_float)
    )?;
    writeln!(
        stdout,
        "same class as ONNX Runtime:     {agree_ort} of {count}"
    )?;
    writeln!(
        stdout,
        "activation arena:               {} bytes (unshared {}, busiest layer {})",
        model::ARENA_BYTES,
        model::UNSHARED_BYTES,
        model::PEAK_LIVE_BYTES
    )?;
    writeln!(
        stdout,
        "int4 weights accuracy:          {} (same class as int8 on {agree4} of {count})",
        accuracy(correct4)
    )?;
    writeln!(
        stdout,
        "weights:                        {} bytes int8, {} bytes in int4",
        model::WEIGHT_BYTES,
        model4::WEIGHT_BYTES
    )?;
    writeln!(
        stdout,
        "16x8 accuracy:                  {} (arena {} bytes, same weights)",
        accuracy(correct16),
        model16::ARENA_BYTES
    )?;
    writeln!(
        stdout,
        "same class as the float model:  int8 {agree_float}, 16x8 {agree_float16} of {count}"
    )?;
    writeln!(
        stdout,
        "logit error vs float activations (first {} digits): mean {} int8, {} 16x8; max {} int8, {} 16x8",
        reference::REFERENCE_LOGIT_COUNT,
        fmt_fixed(error8 / reference_count, 4),
        fmt_fixed(error16 / reference_count, 5),
        fmt_fixed(worst8, 4),
        fmt_fixed(worst16, 5)
    )?;
    writeln!(
        stdout,
        "wrote {RESULTS_CSV}, {PLAN_CSV}, {SUMMARY_CSV} and {LOGITS_CSV}"
    )?;
    Ok(())
}
