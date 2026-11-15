//! Runs two LSTM models written by the PyTorch importer (`tools/import_pytorch`),
//! the importer's worked examples (`tools/import_pytorch/example_lstm.py`): a
//! sequence classifier (12 steps of 3 features, 8 LSTM units with an int16
//! cell state, a dense layer and softmax; `examples/common/imported_lstm_model.rs`)
//! and a sequence regressor (10 steps of 2 features, 5 units with an int8 cell
//! state, two dense layers and tanh; `examples/common/imported_lstm_i8_model.rs`).
//! tinymind (C++)'s importer has no LSTM, so the check is against its runtime:
//! `scripts/cpp/imported_lstm.cpp` builds every integer parameter with
//! tinymind (C++)'s calibration functions from the same float scales and runs
//! the models on its `QLSTMCell` and `QDense`, and the output is
//! byte-identical (`scripts/compare-with-cpp.sh`).
//!
//! Run: `cargo run --release --features std --example imported_lstm`. The
//! generated modules are `no_std`; `std` is only for quantizing the float test
//! inputs here.

use std::io::{self, BufWriter, Write};

#[path = "common/imported_lstm_driver.rs"]
mod imported_lstm_driver;

fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    imported_lstm_driver::run(&mut out)?;
    out.flush()
}
