//! Port of tinymind `examples/iris/iris.cpp`: a 4-8-3 Q16.16 perceptron (ReLU
//! hidden, sigmoid output) classifies iris species, and writes the same files the
//! C++ example writes, byte for byte:
//!
//! - stdout: the training log and the test accuracy
//! - `iris_loss.csv`: mean |error| every 1000 training steps
//! - `iris_scatter.csv`: petal length and width, true and predicted class, per
//!   test sample
//! - `iris_confusion.csv`: the test-set confusion matrix
//!
//! Run: `cargo run --release --example iris -- [output_dir] [iris.data]`. The
//! default output dir is `output`; the default data file is this repository's
//! `data/iris.data`, the UCI file tinymind ships with the example.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/iris_driver.rs"]
mod iris_driver;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let data_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/iris.data")));
    fs::create_dir_all(&out_dir)?;

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    if let Err(e) = iris_driver::run(&out_dir, &data_path, &mut stdout) {
        eprintln!("iris: {e}");
        std::process::exit(1);
    }
    Ok(())
}
