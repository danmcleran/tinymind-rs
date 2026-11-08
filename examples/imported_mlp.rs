//! Runs two models written by the PyTorch importer (`tools/import_pytorch`), the
//! importer's worked examples: a 3-8-4-2 MLP with a sigmoid output
//! (`examples/common/imported_mlp_model.rs`) and a 4-10-3 classifier ending in
//! softmax (`examples/common/imported_classifier_model.rs`). The output is
//! byte-identical to the same models imported with tinymind (C++)'s own importer and run
//! on its C++ runtime (`scripts/compare-with-cpp.sh`).
//!
//! Run: `cargo run --release --features std --example imported_mlp`. The
//! generated module itself is `no_std`; `std` is only for quantizing the float
//! test inputs here.

use std::io::{self, BufWriter, Write};

#[path = "common/imported_mlp_driver.rs"]
mod imported_mlp_driver;

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    imported_mlp_driver::run(&mut out)?;
    out.flush()
}
