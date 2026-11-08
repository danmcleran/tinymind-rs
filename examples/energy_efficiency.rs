//! Port of tinymind `examples/energy_efficiency/energy_efficiency.cpp`: an
//! 8-16-2 Q16.16 regression (ReLU hidden, linear output, gradients clipped to
//! [-1, 1]) predicts a building's heating and cooling load from eight design
//! features, and writes the same files the C++ example writes, byte for byte:
//!
//! - stdout: the training log and the test MAE and R² per target
//! - `energy_loss.csv`: mean |error| every 2000 training steps
//! - `energy_pred.csv`: true and predicted loads per test sample
//!
//! Run: `cargo run --release --example energy_efficiency -- [output_dir]
//! [ENB2012_data.csv]`. The default output dir is `output`; the default data
//! file is this repository's `data/ENB2012_data.csv`, the UCI file tinymind
//! ships with the example.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/energy_driver.rs"]
mod energy_driver;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let data_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/ENB2012_data.csv"
        ))
    });
    fs::create_dir_all(&out_dir)?;

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    if let Err(e) = energy_driver::run(&out_dir, &data_path, &mut stdout) {
        eprintln!("energy_efficiency: {e}");
        std::process::exit(1);
    }
    Ok(())
}
