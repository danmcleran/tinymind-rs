//! The anomaly-detection demo (examples/common/anomaly_demo.rs) on the board:
//! the autoencoder trains on the device, then sets its threshold and screens
//! normal and faulty windows. The report is the host example's, line for line;
//! the hash covers every trained weight and every score, so a match proves the
//! board trained and decided bit for bit as the host did.

use core::fmt::Write;
use core::hint::black_box;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[path = "../../examples/common/anomaly_demo.rs"]
mod anomaly_demo;
use anomaly_demo::{
    Kind, Signal, TRAIN_STEPS, evaluate, new_autoencoder, score, train, train_step,
};

/// The host run's hash (`cargo run --release --example anomaly`).
const EXPECT_HASH: u32 = 0x427B_ACE2;

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=anomaly");
    let mut nn = new_autoencoder(1);
    let mut signal = Signal::new(7);

    // Training on the device, window generation included.
    let a = board::cycles();
    let recent = train(&mut nn, &mut signal, TRAIN_STEPS);
    let total = board::cycles().wrapping_sub(a);
    let _ = writeln!(out, "train_steps={TRAIN_STEPS}\ntrain_cycles={total}");
    let _ = writeln!(out, "train_cycles_per_step={}", total / TRAIN_STEPS);
    let _ = writeln!(out, "train_recent_error={recent}");

    // The report, as the host prints it.
    let hash = evaluate(&mut nn, &mut signal, out, &mut |_, _| {}).unwrap_or(0);
    let _ = writeln!(
        out,
        "expect_hash=0x{EXPECT_HASH:08X}\nhash_match={}",
        u32::from(hash == EXPECT_HASH)
    );

    // Timings on fixed windows (after the hash, so they do not affect it): one
    // detection (forward pass and error) and one training step.
    let w = black_box(signal.window(Kind::Normal));
    let mut t = [0u32; REPS];
    for slot in t.iter_mut() {
        let a = board::cycles();
        black_box(score(&mut nn, &w));
        *slot = board::cycles().wrapping_sub(a);
    }
    let (min, med, max) = median(t);
    let _ = writeln!(out, "detect_min={min}\ndetect_med={med}\ndetect_max={max}");
    for slot in t.iter_mut() {
        let a = board::cycles();
        black_box(train_step(&mut nn, &w));
        *slot = board::cycles().wrapping_sub(a);
    }
    let (min, med, max) = median(t);
    let _ = writeln!(out, "step_min={min}\nstep_med={med}\nstep_max={max}");
}
