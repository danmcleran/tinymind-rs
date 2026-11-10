//! Port of tinymind (C++) `examples/har_activity`: an LSTM classifies windows
//! of accelerometer samples into four activities (walking, walking upstairs,
//! walking downstairs, sitting), reading each window one sample at a time, and
//! the program writes the same files the C++ example writes, byte for byte:
//!
//! - stdout: the training log (mean |error| per epoch), the test accuracy and
//!   two baselines (nearest centroid on per-axis mean and standard deviation,
//!   which ignores sample order, and chance)
//! - `har_samples.csv`: one synthesized window per activity
//! - `har_loss.csv`: mean |error| per epoch
//! - `har_confusion.csv`: the test confusion matrix
//!
//! Network: 3 inputs -> 16 LSTM cells -> 4 sigmoid outputs, Q16.16, trained
//! at learning rate 0.05 without momentum, gradients clipped to [-1, 1]. The
//! windows are synthesized unless a `har.csv` file in the C++ example's format
//! (`window,t,ax,ay,az,label`) is found, as in the C++.
//!
//! Run: `cargo run --release --example har_activity -- [--readout-only]
//! [output_dir] [har.csv]` (default `output`, and `har.csv` in the current
//! directory). `--readout-only`, as in the C++, keeps the LSTM's input, gate
//! and recurrent weights at their initial values and trains only the
//! hidden-to-output weights.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/har_driver.rs"]
mod har_driver;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let readout_only = args.next_if(|a| a == "--readout-only").is_some();
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let csv = PathBuf::from(args.next().unwrap_or_else(|| "har.csv".into()));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    har_driver::run(&out_dir, &csv, readout_only, &mut stdout)
}
