//! Port of tinymind (C++) `examples/air_quality`: an LSTM learns to forecast
//! the next hour's pollutant concentration from the current hour, and the
//! program writes the same files the C++ example writes, byte for byte:
//!
//! - stdout: the series used, the training summary and the forecast's mean
//!   absolute error
//! - `airq_loss.csv`: mean |error| per epoch
//! - `airq_forecast.csv`: true and predicted value for each held-out hour
//!
//! Network: 1 input -> 16 LSTM cells (tanh) -> 1 sigmoid output, Q16.16,
//! gradients clipped to [-1, 1], learning rate 0.5. The series is synthetic (an
//! hourly CO-like signal with two rush-hour peaks a day) unless
//! `AirQualityUCI.csv` from the UCI Air Quality dataset is found in the output
//! directory, as the C++ looks for it in its working directory.
//!
//! Run: `cargo run --release --example air_quality -- [output_dir] [epochs]
//! [learning_rate]` (defaults `output`, 15000 and 0.5, as in the C++). The
//! full run takes about 20 seconds.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/airq_driver.rs"]
mod airq_driver;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    // As the C++'s atoi/atof: an argument that does not parse gives 0.
    let epochs = args
        .next()
        .map_or(airq_driver::TRAINING_EPOCHS, |a| a.parse().unwrap_or(0));
    let learning_rate = args
        .next()
        .map_or(airq_driver::LEARNING_RATE, |a| a.parse().unwrap_or(0.0));
    fs::create_dir_all(&out_dir)?;
    let data = out_dir.join("AirQualityUCI.csv");
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    airq_driver::run(&out_dir, &data, epochs, learning_rate, &mut stdout)
}
