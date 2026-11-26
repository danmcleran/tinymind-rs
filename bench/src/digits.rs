//! The digits workloads of tinymind (C++)'s `benchmarks/tier1_onboard`: the
//! generated model of its `examples/onnx_digits_int8` in three variants,
//! int8, every weight layer on packed int4 weights, and int16 activations
//! with int8 weights (16x8), each running in the arena tinymind (C++)'s
//! planner gives it. The models are the modules `tools/import_onnx`
//! generates from that example's ONNX models (examples/common/onnx_digits_*,
//! which tests/onnx_digits_golden.rs checks against the C++ example), and the
//! inputs the 8 test digits of tinymind (C++)'s gen_tables.cpp
//! (tests/golden/features/tables.h). Each workload's output hash over those
//! digits must equal the one gen_tables.cpp printed. Timed as the C++ times
//! it: one untimed call, then 33 calls cycling through the 8 inputs, the
//! median reported. One image per variant (features `digits-int8`,
//! `digits-int4`, `digits-16x8`), as the C++ builds one per group.

use core::fmt::Write;
use core::hint::black_box;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[allow(dead_code)]
mod tables {
    include!(concat!(env!("OUT_DIR"), "/features_tables.rs"));
}

#[allow(dead_code)]
#[path = "../../examples/common/onnx_digits_reference.rs"]
mod reference;

#[cfg(feature = "digits-int8")]
#[path = "../../examples/common/onnx_digits_model.rs"]
mod model;
#[cfg(feature = "digits-int8")]
const NAME: &str = "digits_int8";
#[cfg(feature = "digits-int8")]
const HASH: u32 = tables::HASH_INT8;

#[cfg(feature = "digits-int4")]
#[path = "../../examples/common/onnx_digits_int4_model.rs"]
mod model;
#[cfg(feature = "digits-int4")]
const NAME: &str = "digits_int4";
#[cfg(feature = "digits-int4")]
const HASH: u32 = tables::HASH_INT4;

#[cfg(feature = "digits-16x8")]
#[path = "../../examples/common/onnx_digits_16x8_model.rs"]
mod model;
#[cfg(feature = "digits-16x8")]
const NAME: &str = "digits_16x8";
#[cfg(feature = "digits-16x8")]
const HASH: u32 = tables::HASH_16X8;

/// A test digit on the model's input grid (C++ `digitsInput8` /
/// `digitsInput16`).
fn input(pixels: &[u8; 64]) -> model::Input {
    core::array::from_fn(|r| {
        core::array::from_fn(|c| {
            let p = usize::from(pixels[r * 8 + c]);
            #[cfg(not(feature = "digits-16x8"))]
            let v = reference::PIXEL_TO_INPUT[p];
            #[cfg(feature = "digits-16x8")]
            let v = reference::PIXEL_TO_INPUT16[p];
            [v]
        })
    })
}

/// The timed call (C++ `runDigits*`, `noinline` there too).
#[inline(never)]
fn run_one(x: &model::Input, arena: &mut model::Arena) -> model::Output {
    model::forward(x, arena)
}

/// FNV-1a over the outputs' bytes, little end first (C++ `t1::fnv`).
fn fnv(mut h: u32, out: &model::Output) -> u32 {
    for &v in out {
        let mut u = v as u32;
        for _ in 0..core::mem::size_of_val(&v) {
            h = (h ^ (u & 0xFF)).wrapping_mul(16_777_619);
            u >>= 8;
        }
    }
    h
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(
        out,
        "name={NAME}\narena_bytes={}\nweight_bytes={}",
        model::ARENA_BYTES,
        model::WEIGHT_BYTES
    );
    let inputs: [model::Input; 8] = core::array::from_fn(|s| input(&tables::DIGIT_PIXELS[s]));
    #[cfg(not(feature = "digits-16x8"))]
    let mut arena: model::Arena = [0; model::ARENA_BYTES];
    #[cfg(feature = "digits-16x8")]
    let mut arena: model::Arena = [0; model::ARENA_BYTES / 2];
    let mut h = 2_166_136_261u32;
    for x in &inputs {
        h = fnv(h, &run_one(x, &mut arena));
    }
    let mut t = [0u32; REPS];
    black_box(run_one(black_box(&inputs[0]), &mut arena));
    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box(&inputs[r % inputs.len()]);
        let a = board::cycles();
        let y = run_one(x, &mut arena);
        *slot = board::cycles().wrapping_sub(a);
        black_box(y);
    }
    let (min, med, max) = median(t);
    let _ = writeln!(
        out,
        "{NAME}_min={min}\n{NAME}_med={med}\n{NAME}_max={max}\n{NAME}_match={}",
        u32::from(h == HASH)
    );
}
