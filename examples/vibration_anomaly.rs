//! Port of tinymind (C++) `examples/vibration_anomaly`: vibration anomaly
//! detection that learns "normal" on the device. A synthetic machine's stream
//! goes frame by frame through [`QLogMelSpectrogram`], becomes int8 features,
//! and trains a [`QKMeansAnomalyTrainer`] on normal running; the learned
//! model then scores normal running and three growing faults (bearing,
//! imbalance, looseness), beside a Euclidean k-means and a diagonal Gaussian.
//! Its output is byte-identical to the C++ example's except the size of the
//! trainer in the last line of stdout (1,248 bytes here, 1,264 in the C++,
//! whose field order needs more padding):
//!
//! - stdout: thresholds, false-alarm rates, and per fault the AUC, the alarm
//!   rate and the severity of the first sustained alarm
//! - `vibration_anomaly.csv`: every test frame's three scores and thresholds
//! - `vibration_anomaly_summary.csv`: the stdout figures as a table
//!
//! Run: `cargo run --release --features std --example vibration_anomaly --
//! [output_dir]` (default `output`).
//!
//! [`QLogMelSpectrogram`]: tinymind::int8::QLogMelSpectrogram
//! [`QKMeansAnomalyTrainer`]: tinymind::int8::QKMeansAnomalyTrainer

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/vibration_driver.rs"]
mod vibration_driver;

fn main() -> io::Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "output".into()));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    vibration_driver::run(&out_dir, &mut stdout)
}
