//! Port of tinymind (C++) `examples/maze`: a Q-learner finds the way out of a
//! six-room maze, and the program writes the same files the C++ example
//! writes, byte for byte:
//!
//! - stdout: each episode's start room and actions
//! - `maze_training.csv`: the path of each of the 500 training episodes
//! - `maze_test.csv`: the path of each of the 100 test episodes
//!
//! Run: `cargo run --release --example maze -- [output_dir]` (default `output`).

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/maze_driver.rs"]
mod maze_driver;

fn main() -> io::Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "output".into()));
    fs::create_dir_all(&out_dir)?;
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    maze_driver::run(&out_dir, &mut stdout)
}
