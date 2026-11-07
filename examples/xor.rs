//! Port of tinymind `examples/xor/xor.cpp`: train a 2-4-1 perceptron (tanh
//! hidden, sigmoid output) on XOR and write the same files the C++ example
//! writes, byte for byte:
//!
//! - `nn_fixed_xor.txt`: every weight, the target, the output and the error after
//!   each of the 20000 training steps (raw LSBs)
//! - `xor_training.csv`: the 4-case XOR error every 100 steps
//! - `xor_decision_surface.csv`: the trained network over a 41x41 input grid
//!
//! Run: `cargo run --release --example xor -- [--format q16.16|q8.8] [output_dir]
//! [learning_rate] [momentum]`.
//! The default format is Q16.16, as in the C++ example; Q8.8 matches the C++
//! example built with `NUMBER_OF_FIXED_BITS` and `NUMBER_OF_FRACTIONAL_BITS` set
//! to 8 in `xornet.h`. The default output dir is `output`.

use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

#[path = "common/xor_driver.rs"]
mod xor_driver;

fn main() -> io::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let format = if args.first().map(String::as_str) == Some("--format") {
        if args.len() < 2 {
            usage("--format needs a value");
        }
        let f = args.remove(1);
        args.remove(0);
        f
    } else {
        "q16.16".into()
    };
    let mut args = args.into_iter();
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let learning_rate = args.next().map(|s| parse_arg(&s));
    let momentum = args.next().map(|s| parse_arg(&s));
    if args.next().is_some() {
        usage("too many arguments");
    }
    if !xor_driver::FORMATS.contains(&format.as_str()) {
        usage(&format!("unknown format {format:?}"));
    }
    fs::create_dir_all(&out_dir)?;

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    xor_driver::run_format(&format, &out_dir, learning_rate, momentum, &mut stdout)
}

fn usage(problem: &str) -> ! {
    eprintln!("xor: {problem}");
    eprintln!("usage: xor [--format q16.16|q8.8] [output_dir] [learning_rate] [momentum]");
    std::process::exit(2);
}

fn parse_arg(s: &str) -> f64 {
    s.parse()
        .unwrap_or_else(|_| usage(&format!("expected a number, got {s:?}")))
}
