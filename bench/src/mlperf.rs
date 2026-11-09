//! The MLPerf Tiny keyword-spotting reference model (DS-CNN, int8), translated
//! from its .tflite by tools/import_tflite (bench/src/gen, not committed: the
//! model is Apache-2.0, MLCommons). The output hash over make_mlperf.py's 8
//! seeded random inputs must equal tinymind's (`TM_MLPERF_KWS_HASH`), which
//! tinymind's C++ and TFLite's reference kernels produce. F411 only: the model
//! does not fit the L031 (tinymind's figures say the same).
//!
//! Generate first: `python3 -I tools/import_tflite/mlperf.py kws
//! --out bench/src/gen/mlperf_kws.rs --data bench/src/gen/mlperf_kws_data.rs`.

use core::fmt::Write;
use core::hint::black_box;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[path = "gen/mlperf_kws.rs"]
mod model;
#[path = "gen/mlperf_kws_data.rs"]
mod data;
use data::*;

#[inline(never)]
fn infer(x: &model::Input) -> model::Output {
    model::forward(x)
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=mlperf-kws-ds-cnn");
    black_box(infer(black_box(&X[0]))); // warm-up
    let mut t = [0u32; REPS];
    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box(&X[r % N_TEST]);
        let a = board::cycles();
        black_box(infer(x));
        *slot = board::cycles().wrapping_sub(a);
    }
    let (min, med, max) = median(t);
    let _ = writeln!(out, "infer_min={min}\ninfer_med={med}\ninfer_max={max}");

    // Every test input: tinymind's FNV-1a hash over every output byte.
    let mut h: u32 = 0x811C_9DC5;
    for x in &X {
        for v in infer(x) {
            h ^= u32::from(v as u8);
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    let _ = writeln!(out, "n_test={N_TEST}");
    let _ = writeln!(
        out,
        "hash=0x{h:08X}\nexpect_hash=0x{EXPECT_HASH:08X}\nhash_match={}",
        u32::from(h == EXPECT_HASH)
    );
}
