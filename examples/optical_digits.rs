//! Port of tinymind (C++) `examples/optical_digits/optical_digits.cpp`: a 64-32-10
//! Q16.16 perceptron (ReLU hidden, sigmoid output) learns to read handwritten
//! digits from 8x8 bitmaps, and writes the same files the C++ example writes,
//! byte for byte:
//!
//! - stdout: the training log and the test accuracy
//! - `digits_loss.csv`: mean |error| every 2000 training steps
//! - `digits_confusion.csv`: the 10x10 test-set confusion matrix
//! - `digits_samples.csv`: up to 40 test images (64 pixels, true and predicted
//!   digit), misclassified ones first
//!
//! Run: `cargo run --release --example optical_digits -- [output_dir]
//! [data_dir]`. The default output dir is `output`; the default data dir is this
//! repository's `data/`, which holds `optdigits.tra` (3823 training images) and
//! `optdigits.tes` (1797 test images), the UCI files tinymind (C++) ships with
//! the example. They are not in the published crate (too large); clone the
//! repository to run this example.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/digits_driver.rs"]
mod digits_driver;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let data_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data")));
    fs::create_dir_all(&out_dir)?;

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    if let Err(e) = digits_driver::run(&out_dir, &data_dir, &mut stdout) {
        eprintln!("optical_digits: {e}");
        std::process::exit(1);
    }
    Ok(())
}
