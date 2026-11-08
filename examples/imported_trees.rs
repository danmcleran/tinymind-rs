//! Two scikit-learn tree models for the UCI Wine data, imported with
//! `tools/import_trees` and run with `tinymind::tree`: a gradient-boosted
//! classifier on int8 inputs and a voting random forest on int16 inputs. The
//! output is byte-identical to tinymind's C++ runtime running the same models
//! imported with tinymind's importer (`scripts/compare-with-cpp.sh`).
//!
//! Run: `cargo run --release --example imported_trees`

use std::io::{self, BufWriter};

#[path = "common/imported_trees_driver.rs"]
mod imported_trees_driver;

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    imported_trees_driver::run(&mut stdout)
}
