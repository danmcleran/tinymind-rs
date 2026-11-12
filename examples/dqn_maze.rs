//! tinymind (C++)'s `examples/dqn_maze`: a deep Q-network learns the way out
//! of a six-room maze. Writes dqn_maze_training.csv, dqn_maze_qtable.csv and
//! dqn_maze_test.csv to the output directory (default `output/`).
//!
//! ```bash
//! cargo run --release --example dqn_maze -- output
//! ```

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/dqn_maze_driver.rs"]
mod dqn_maze_driver;

fn main() -> io::Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "output".into()));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    let stderr = io::stderr();
    let mut stderr = stderr.lock();
    dqn_maze_driver::run(&out_dir, &mut stdout, &mut stderr)
}
