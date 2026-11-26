//! Port of tinymind (C++) `examples/onnx_digits_int8`: a residual int8 CNN
//! reads the UCI optical digits (8x8 bitmaps, 1,797 test digits), run from
//! Rust modules that `tools/import_onnx` generated from the ONNX models that
//! example commits, in three variants: int8, every weight layer on packed int4
//! weights (fine-tuned with int4 quantization-aware training), and int16
//! activations with the int8 weights (16x8). Each model runs in one activation
//! arena planned by tinymind (C++)'s planner (1,024 bytes for the int8 and
//! int4 models, 2,048 for 16x8). The program writes the same files the C++
//! example writes, byte for byte:
//!
//! - stdout: accuracies, agreement with ONNX Runtime and the float model, the
//!   arena and weight sizes, the logit error against float activations
//! - `onnx_digits_int8.csv`: per digit, the label, the three models' classes,
//!   ONNX Runtime's and the float model's, and the pixels
//! - `onnx_digits_int8_logits.csv`: the int8 and 16x8 logits in real units
//!   next to the float-activation reference, first 256 digits
//! - `onnx_digits_int8_plan.csv`: the arena plan
//! - `onnx_digits_int8_summary.csv`: the figures above as CSV
//!
//! Run: `cargo run --release --example onnx_digits_int8 --
//! [output_dir] [test_set]`. The default output dir is `output`; the default
//! test set is this repository's `data/optdigits.tes` (not in the published
//! crate; clone the repository to run this example). The generated modules
//! use the importer's arena option, which needs Rust 1.88.

use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

#[path = "common/onnx_digits_driver.rs"]
mod onnx_digits_driver;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let test_set = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/optdigits.tes"))
    });
    fs::create_dir_all(&out_dir)?;

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    if let Err(e) = onnx_digits_driver::run(&out_dir, &test_set, &mut stdout) {
        eprintln!("onnx_digits_int8: {e}");
        std::process::exit(1);
    }
    stdout.flush()
}
