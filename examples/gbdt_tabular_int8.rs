//! Port of tinymind `examples/gbdt_tabular_int8`: a five-tree gradient-boosted
//! ensemble (two features, three classes) quantized to int8 and run through
//! [`QGbdt`]. Its output is byte-identical to the C++ example's:
//!
//! - stdout: the grid, the int8-versus-float agreement and probe predictions
//!   (with `--golden`, the C++ example's compact golden report instead)
//! - `gbdt_tabular_int8.csv`: the int8 ensemble's class over a 48x48 grid
//!
//! Run: `cargo run --release --features std --example gbdt_tabular_int8 --
//! [--golden] [output_dir]` (default `output`). Exits with status 1 if the
//! agreement is below 97%, as the C++ example does.
//!
//! [`QGbdt`]: tinymind::tree::QGbdt

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/gbdt_driver.rs"]
mod gbdt_driver;

fn main() -> io::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let golden = args.first().is_some_and(|a| a == "--golden");
    if golden {
        args.remove(0);
    }
    let out_dir = PathBuf::from(args.first().map_or("output", String::as_str));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    let passed = gbdt_driver::run(&out_dir, golden, &mut stdout)?;
    drop(stdout);
    if !passed {
        std::process::exit(1);
    }
    Ok(())
}
