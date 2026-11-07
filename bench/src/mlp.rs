//! tinymind's int8 digits MLP (64-32-16-10, per-channel requantizers, ReLU fused
//! into qmin), on the same weights and test samples as its C++ benchmark. The
//! output hash must equal tinymind's expected hash, which tinymind's C++ and TFLM
//! both produce on these boards.
//!
//! Without the `mlp` feature this is the baseline image: the same harness with a
//! do-nothing network and a one-sample dummy test set, as tinymind's baseline
//! backend. Network size = image - baseline image - test-set bytes.

use core::fmt::Write;
use core::hint::black_box;

#[cfg(feature = "mlp")]
use tinymind::int8::QDensePerChannel;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[cfg(feature = "mlp")]
#[path = "gen/mlp.rs"]
mod model;
#[cfg(feature = "mlp")]
use model::*;

#[cfg(not(feature = "mlp"))]
const N_TEST: usize = 1;
#[cfg(not(feature = "mlp"))]
static X: [[i8; 64]; N_TEST] = [[0; 64]];
#[cfg(not(feature = "mlp"))]
static Y: [u8; N_TEST] = [0];
#[cfg(not(feature = "mlp"))]
const EXPECT_HASH: u32 = 0;
#[cfg(not(feature = "mlp"))]
const NAME: &str = "baseline";
#[cfg(feature = "mlp")]
const NAME: &str = "mlp-int8-per-channel";

#[cfg(not(feature = "mlp"))]
#[inline(never)]
fn infer(x: &[i8; 64]) -> [i8; 10] {
    black_box(x);
    [0; 10]
}

#[cfg(feature = "mlp")]
const L0: QDensePerChannel<'static, 64, 32> = QDensePerChannel {
    weights: &W_L0,
    biases: Some(&B_L0),
    input_zero_point: ZP_L0,
    requantizers: &R_L0,
};
#[cfg(feature = "mlp")]
const L1: QDensePerChannel<'static, 32, 16> = QDensePerChannel {
    weights: &W_L1,
    biases: Some(&B_L1),
    input_zero_point: ZP_L1,
    requantizers: &R_L1,
};
#[cfg(feature = "mlp")]
const L2: QDensePerChannel<'static, 16, 10> = QDensePerChannel {
    weights: &W_L2,
    biases: Some(&B_L2),
    input_zero_point: ZP_L2,
    requantizers: &R_L2,
};

#[cfg(feature = "mlp")]
#[inline(never)]
fn infer(x: &[i8; 64]) -> [i8; 10] {
    L2.forward(&L1.forward(&L0.forward(x)))
}

/// Time `f` on sample `r % N_TEST` for each repetition.
fn time(mut f: impl FnMut(&[i8; 64])) -> (u32, u32, u32) {
    let mut t = [0u32; REPS];
    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box(&X[r % N_TEST]);
        let a = board::cycles();
        f(x);
        *slot = board::cycles().wrapping_sub(a);
    }
    median(t)
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name={NAME}");
    black_box(infer(black_box(&X[0]))); // warm-up
    let (min, med, max) = time(|x| {
        black_box(infer(x));
    });
    let _ = writeln!(out, "infer_min={min}\ninfer_med={med}\ninfer_max={max}");

    // Per layer, on the activations each layer actually sees.
    #[cfg(feature = "profile")]
    {
        let (_, l0, _) = time(|x| {
            black_box(L0.forward(x));
        });
        let a0 = L0.forward(&X[0]);
        let a1 = L1.forward(&a0);
        let (_, l1, _) = time(|_| {
            black_box(L1.forward(black_box(&a0)));
        });
        let (_, l2, _) = time(|_| {
            black_box(L2.forward(black_box(&a1)));
        });
        let _ = writeln!(out, "layer.0.fc0={l0}\nlayer.1.fc1={l1}\nlayer.2.fc2={l2}");
    }

    // Whole test set: tinymind's FNV-1a hash over every output byte, and accuracy.
    let mut h: u32 = 0x811C_9DC5;
    let mut correct = 0u32;
    for (x, &label) in X.iter().zip(&Y) {
        let y = infer(x);
        for &v in &y {
            h ^= u32::from(v as u8);
            h = h.wrapping_mul(0x0100_0193);
        }
        let mut best = 0;
        for k in 1..y.len() {
            if y[k] > y[best] {
                best = k;
            }
        }
        correct += u32::from(best == usize::from(label));
    }
    let _ = writeln!(out, "n_test={N_TEST}\ncorrect={correct}");
    let _ = writeln!(out, "hash=0x{h:08X}\nexpect_hash=0x{EXPECT_HASH:08X}");
    let _ = writeln!(out, "hash_match={}", u32::from(h == EXPECT_HASH));
}
