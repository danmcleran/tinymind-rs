//! Port of tinymind (C++)'s `examples/kws_cortex_m_int8`: a keyword-spotting
//! shaped int8 network (convolution, max pooling, depthwise and pointwise
//! convolution, global average pooling, a dense head) with synthetic weights,
//! run once with a per-layer report. Every tensor matches the C++ byte for
//! byte; the sizes and timings in the report are this build's own.
//!
//! ```text
//! cargo run --release --features std --example kws_cortex_m_int8
//! ```

// The tensor dump and the tensors themselves are for the golden test.
#[allow(dead_code)]
#[path = "common/kws_int8_driver.rs"]
mod kws_int8_driver;

use std::time::Instant;

fn main() {
    // The C++ host build reports nanoseconds since its first reading.
    let start = Instant::now();
    let clock = || start.elapsed().as_nanos() as u32;
    let stdout = std::io::stdout();
    if let Err(e) = kws_int8_driver::run(&mut stdout.lock(), clock) {
        eprintln!("kws_cortex_m_int8: {e}");
        std::process::exit(1);
    }
}
