//! The features workload of tinymind (C++)'s `benchmarks/tier1_onboard`: a
//! 256-sample log-mel frame (32 bands, pre-emphasis 0.97) and its 13 MFCCs,
//! k-means (32 features, 8 clusters) and Gaussian (4 components) anomaly
//! scores, and k-means learning steps, on `tests/sweeps/features_bench.rs`.
//! The tables are tinymind (C++)'s (tests/golden/features/tables.h, which
//! tests/features_bench.rs checks this crate's calibration reproduces), and
//! each workload's hash over its inputs must equal the one gen_tables.cpp
//! printed. Timed as the C++ times it: frames regenerated outside the timed
//! region, 33 runs cycling through the 8 inputs, the median reported.
//!
//! Feature `features` runs everything (one image, as the C++); at opt-level 3
//! that overflows the L031's 32 KB, so there `features-mel` (log-mel and
//! MFCC) and `features-anomaly` (the scores and learning) run as two images.

use core::fmt::Write;
use core::hint::black_box;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[allow(dead_code)]
#[path = "../../tests/sweeps/features_bench.rs"]
mod features_bench;
use features_bench::*;

#[allow(dead_code)]
mod tables {
    include!(concat!(env!("OUT_DIR"), "/features_tables.rs"));
}
use tables::*;

#[cfg(feature = "features-mel")]
static MEL: Mel<'static> = Mel {
    window: &WINDOW,
    twiddle_cos: &TW_COS,
    twiddle_sin: &TW_SIN,
    band_start: &BAND_START,
    band_length: &BAND_LENGTH,
    weights: &MEL_WEIGHTS,
    preemphasis: PREEMPHASIS,
    log_floor: -32768,
};
#[cfg(feature = "features-mel")]
static MFCC: Mfcc<'static> = Mfcc { dct: &DCT };
#[cfg(feature = "features-anomaly")]
static KMEANS: KMeans<'static> = KMeans {
    centroids: &CENTROIDS,
    inv_radius_sq: &INV_RADIUS,
    feature_weight: Some(&FEATURE_WEIGHT),
};
#[cfg(feature = "features-anomaly")]
static GAUSSIAN: Gaussian<'static> = Gaussian {
    means: &GAUSS_MEANS,
    precision: &GAUSS_PRECISION,
    bias: &GAUSS_BIAS,
};

fn report(out: &mut Uart, name: &str, t: [u32; REPS], matched: bool) {
    let (min, med, max) = median(t);
    let _ = writeln!(
        out,
        "{name}_min={min}\n{name}_med={med}\n{name}_max={max}\n{name}_match={}",
        u32::from(matched)
    );
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(
        out,
        "name=features\nframe={MEL_N}\nbands={BANDS}\ncoeffs={COEFFS}"
    );
    #[cfg(feature = "features-mel")]
    run_mel_workloads(out);
    #[cfg(feature = "features-anomaly")]
    run_anomaly_workloads(out);
}

#[cfg(feature = "features-mel")]
fn run_mel_workloads(out: &mut Uart) {
    let mut ws = Workspace::new();
    let mut frame = [0i16; MEL_N];
    let h = hash_mel(&MEL, &MFCC, &mut ws, &mut frame);
    let mut t = [0u32; REPS];
    let mut log_mel = [0i16; BANDS];
    for (r, slot) in t.iter_mut().enumerate() {
        make_frame(&mut frame, 1 + (r % FRAMES) as u32);
        let a = board::cycles();
        run_mel(black_box(&MEL), &frame, &mut log_mel, &mut ws);
        *slot = board::cycles().wrapping_sub(a);
    }
    report(out, "logmel_256x32", t, h[0] == HASH_MEL);

    let mut coeffs = [0i32; COEFFS];
    for slot in t.iter_mut() {
        let a = board::cycles();
        run_mfcc(black_box(&MFCC), &log_mel, &mut coeffs);
        *slot = board::cycles().wrapping_sub(a);
    }
    black_box(&coeffs);
    report(out, "mfcc_32x13", t, h[1] == HASH_MFCC);
}

#[cfg(feature = "features-anomaly")]
fn run_anomaly_workloads(out: &mut Uart) {
    let h = hash_scores(&KMEANS, &GAUSSIAN);
    let features: [[i8; FEATURES]; FRAMES] = core::array::from_fn(|s| make_features(s as u32 + 1));
    let mut t = [0u32; REPS];
    for (r, slot) in t.iter_mut().enumerate() {
        let x = &features[r % FRAMES];
        let a = board::cycles();
        black_box(run_kmeans(black_box(&KMEANS), x));
        *slot = board::cycles().wrapping_sub(a);
    }
    report(out, "kmeans_score_32x8", t, h[0] == HASH_KMEANS);

    for (r, slot) in t.iter_mut().enumerate() {
        let x = &features[r % FRAMES];
        let a = board::cycles();
        black_box(run_gaussian(black_box(&GAUSSIAN), x));
        *slot = board::cycles().wrapping_sub(a);
    }
    report(out, "gaussian_score_32x4", t, h[1] == HASH_GAUSSIAN);

    // Learning: the clusters are seeded first, so every timed call is a full
    // update (nearest centroid, then a step towards the sample).
    let mut trainer = Trainer::new();
    trainer_warmup(&mut trainer);
    for (r, slot) in t.iter_mut().enumerate() {
        let x = &features[r % FRAMES];
        let a = board::cycles();
        run_learn(&mut trainer, x);
        *slot = board::cycles().wrapping_sub(a);
    }
    report(
        out,
        "kmeans_learn_32x8",
        t,
        hash_trainer(&trainer) == HASH_LEARN,
    );
}
