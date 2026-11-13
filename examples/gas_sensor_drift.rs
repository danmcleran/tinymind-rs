//! Port of tinymind (C++) `examples/gas_sensor_drift/gas_sensor_drift.cpp`: a
//! 128-32-6 Q16.16 perceptron (ReLU hidden, sigmoid output) classifies six
//! gases from a synthetic 16-sensor array whose sensors drift over 10 batches,
//! then adapts to each drifted batch on the device from a few labelled samples
//! per class, with a [`head`](tinymind::head) on top of the frozen hidden
//! layer. It writes the same files the C++ example writes, byte for byte:
//!
//! - stdout: the training log, the drift curve, the check that the backbone
//!   plus head reproduces the trained network's hidden values (predictions
//!   differ only where two sigmoid outputs tie), and batch 10 accuracy per
//!   adaptation method and number of labelled samples
//! - `gas_loss.csv`: mean |error| every 2000 training steps
//! - `gas_confusion.csv`: the batch-1 validation confusion matrix
//! - `gas_drift.csv`: accuracy on each batch, trained on batch 1 only
//! - `gas_adapt.csv`: accuracy per method, batch and number of labelled
//!   samples per class (mean and spread over 10 draws), and on batch 1 after
//!   adapting
//!
//! Run: `cargo run --release --example gas_sensor_drift -- [output_dir]`. The
//! default output dir is `output`. The data is synthetic (drawn from glibc's
//! `rand()`), so there is no file to read.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/gas_driver.rs"]
mod gas_driver;

fn main() -> io::Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "output".into()));
    fs::create_dir_all(&out_dir)?;

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    if let Err(e) = gas_driver::run(&out_dir, &mut stdout) {
        eprintln!("gas_sensor_drift: {e}");
        std::process::exit(1);
    }
    Ok(())
}
