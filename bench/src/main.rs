//! On-board checks for the tinymind crate. Each image runs one workload and
//! prints a key=value report over the ST-LINK virtual COM port, in the format of
//! tinymind's benchmark harness (`# begin bench` ... `# end bench`).

#![no_std]
#![no_main]

#[cfg(all(feature = "mlperf-kws", feature = "mlperf-ic"))]
compile_error!("one MLPerf model per image: enable mlperf-kws or mlperf-ic, not both");

#[cfg(feature = "anomaly")]
mod anomaly;
mod board;
#[cfg(feature = "kws")]
mod kws;
#[cfg(not(any(
    feature = "xor-q16",
    feature = "xor-q8",
    feature = "anomaly",
    feature = "trees",
    feature = "kws",
    feature = "mlperf-kws",
    feature = "mlperf-ic"
)))]
mod mlp;
#[cfg(any(feature = "mlperf-kws", feature = "mlperf-ic"))]
mod mlperf;
#[cfg(feature = "trees")]
mod trees;
#[cfg(any(feature = "xor-q16", feature = "xor-q8"))]
mod xor;

use core::fmt::Write;

use board::Uart;
use cortex_m_rt::entry;

/// Timed repetitions per measurement; the median is reported (as tinymind does).
pub const REPS: usize = 33;

/// Median of `REPS` timings.
pub fn median(mut t: [u32; REPS]) -> (u32, u32, u32) {
    t.sort_unstable();
    (t[0], t[REPS / 2], t[REPS - 1])
}

#[entry]
fn main() -> ! {
    board::init();
    board::stack_paint();
    let mut out = Uart;
    let _ = writeln!(out, "\n# begin bench");
    let _ = writeln!(out, "board={}", board::NAME);
    let _ = writeln!(out, "mhz={}", board::MHZ);
    #[cfg(not(any(
        feature = "xor-q16",
        feature = "xor-q8",
        feature = "anomaly",
        feature = "trees",
        feature = "kws",
        feature = "mlperf-kws",
        feature = "mlperf-ic"
    )))]
    mlp::run(&mut out);
    #[cfg(feature = "kws")]
    kws::run(&mut out);
    #[cfg(any(feature = "mlperf-kws", feature = "mlperf-ic"))]
    mlperf::run(&mut out);
    #[cfg(feature = "trees")]
    trees::run(&mut out);
    #[cfg(feature = "anomaly")]
    anomaly::run(&mut out);
    #[cfg(feature = "xor-q16")]
    xor::run::<tinymind::Q16_16>(&mut out, "xor-q16.16", xor::Q16_EXPECT);
    #[cfg(feature = "xor-q8")]
    xor::run::<tinymind::Q8_8>(&mut out, "xor-q8.8", xor::Q8_EXPECT);
    let _ = writeln!(out, "stack_bytes={}", board::stack_used());
    let _ = writeln!(out, "static_ram={}", board::static_ram());
    let _ = writeln!(out, "# end bench");
    board::flush();
    loop {
        cortex_m::asm::nop();
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    let _ = writeln!(Uart, "\n# PANIC");
    loop {
        cortex_m::asm::nop();
    }
}
