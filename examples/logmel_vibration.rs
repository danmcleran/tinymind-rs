//! Port of tinymind (C++) `examples/logmel_vibration`: the fixed-point log-mel
//! spectrogram ([`QLogMelSpectrogram`]) and MFCCs ([`QMfcc`]) of a synthetic
//! machine-vibration recording whose level ramps from -50 dBFS to -6 dBFS,
//! compared frame by frame with the float definition they implement. Its
//! output is byte-identical to the C++ example's:
//!
//! - stdout: the error against the float reference by band level
//! - `logmel_vibration.csv`: per frame, the fixed-point and float bands
//! - `logmel_vibration_mfcc.csv`: per frame, the fixed-point and float MFCCs
//! - `logmel_vibration_error.csv`: the error summary
//!
//! Run: `cargo run --release --features std --example logmel_vibration --
//! [output_dir]` (default `output`).
//!
//! [`QLogMelSpectrogram`]: tinymind::int8::QLogMelSpectrogram
//! [`QMfcc`]: tinymind::int8::QMfcc

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/logmel_driver.rs"]
mod logmel_driver;

fn main() -> io::Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "output".into()));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    logmel_driver::run(&out_dir, &mut stdout)
}
