//! Port of tinymind (C++) `examples/pytorch_quant/xor`: a 2-4-1 network trained in
//! PyTorch and quantized to int8 runs through two [`QDense`] layers, a quantized
//! ReLU and a sigmoid lookup table. Its output is byte-identical to the C++
//! example's:
//!
//! - stdout: the calibration and the four XOR cases
//! - `xor_decision_surface.csv`: the network's output over a 41x41 input grid
//!
//! Run: `cargo run --release --features std --example int8_xor -- [output_dir]`
//! (default `output`). Exits with status 1 unless all four cases are correct, as
//! the C++ example does.
//!
//! [`QDense`]: tinymind::int8::QDense

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/int8_xor_driver.rs"]
mod int8_xor_driver;

fn main() -> io::Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "output".into()));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    let correct = int8_xor_driver::run(&out_dir, &mut stdout)?;
    drop(stdout);
    if correct != 4 {
        std::process::exit(1);
    }
    Ok(())
}
