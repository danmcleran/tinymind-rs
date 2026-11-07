//! Runs a model written by the PyTorch importer (`tools/import_pytorch`): the
//! importer's worked example, a 3-8-4-2 MLP, generated as
//! `examples/common/imported_mlp_model.rs`. The output is byte-identical to the
//! same model imported with tinymind's own importer and run on its C++ runtime
//! (`scripts/compare-with-cpp.sh`).
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
