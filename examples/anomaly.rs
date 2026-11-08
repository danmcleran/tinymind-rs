//! Anomaly detection with an autoencoder trained on the device: see
//! `examples/common/anomaly_demo.rs`. The same program runs on the boards in
//! `bench/` (feature `anomaly`) and must print the same hash.
//!
//! Run: `cargo run --release --example anomaly -- [output_dir]`. With an output
//! directory it also writes `anomaly_scores.csv`: the kind and reconstruction
//! error (raw Q16.16) of every test window, with the threshold.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write as _};
use std::path::PathBuf;

#[path = "common/anomaly_demo.rs"]
mod anomaly_demo;

fn main() -> io::Result<()> {
    let out_dir = std::env::args().nth(1).map(PathBuf::from);
    let mut scores = Vec::new();
    let mut report = String::new();
    anomaly_demo::run(&mut report, &mut |kind, error| {
        scores.push((kind.to_owned(), error))
    })
    .expect("formatting to a String cannot fail");
    io::stdout().write_all(report.as_bytes())?;

    if let Some(dir) = out_dir {
        fs::create_dir_all(&dir)?;
        let threshold = report
            .lines()
            .find_map(|l| l.strip_prefix("threshold="))
            .and_then(|l| l.split_whitespace().next())
            .unwrap_or("0");
        let mut csv = BufWriter::new(File::create(dir.join("anomaly_scores.csv"))?);
        writeln!(csv, "kind,error,threshold")?;
        for (kind, error) in scores {
            writeln!(csv, "{kind},{error},{threshold}")?;
        }
        csv.flush()?;
    }
    Ok(())
}
