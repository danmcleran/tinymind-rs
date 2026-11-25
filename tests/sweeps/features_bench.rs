// The features workload of tinymind (C++)'s benchmarks/tier1_onboard
// (tier1_common.hpp): a 256-sample log-mel frame (32 bands) and its 13 MFCCs,
// k-means and Gaussian anomaly scores over 32 features, and k-means learning
// steps, on the same synthetic inputs and with the same hashes. Shared by
// tests/features_bench.rs (on the host, with tables this crate builds, against
// the golden tables and hashes tinymind (C++)'s gen_tables.cpp writes) and
// bench/src/features.rs (on the boards, with the golden's tables). no_std and
// allocation-free.

use tinymind::int8::{
    QDiagGaussianAnomaly, QKMeansAnomaly, QKMeansAnomalyTrainer, QLogMelSpectrogram, QMelWorkspace,
    QMfcc,
};

/// Inputs per workload, cycled through the timed runs.
pub const FRAMES: usize = 8;
pub const MEL_N: usize = 256;
pub const HALF: usize = MEL_N / 2;
pub const BINS: usize = MEL_N / 2 + 1;
pub const BANDS: usize = 32;
pub const COEFFS: usize = 13;
pub const FEATURES: usize = 32;
pub const CLUSTERS: usize = 8;
pub const COMPONENTS: usize = 4;
/// 0.97 in Q15.
pub const PREEMPHASIS: i16 = 31785;

pub type Mel<'a> = QLogMelSpectrogram<'a, MEL_N, HALF, BANDS>;
pub type Mfcc<'a> = QMfcc<'a, BANDS, COEFFS>;
pub type KMeans<'a> = QKMeansAnomaly<'a, FEATURES, CLUSTERS>;
pub type Trainer = QKMeansAnomalyTrainer<FEATURES, CLUSTERS>;
pub type Gaussian<'a> = QDiagGaussianAnomaly<'a, FEATURES, COMPONENTS>;
pub type Workspace = QMelWorkspace<MEL_N, BINS>;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// FNV-1a over values' little-endian bytes.
pub fn fnv<const W: usize>(mut h: u32, bytes: impl IntoIterator<Item = [u8; W]>) -> u32 {
    for b in bytes {
        for byte in b {
            h = (h ^ u32::from(byte)).wrapping_mul(16_777_619);
        }
    }
    h
}

pub const FNV_OFFSET: u32 = 0x811C_9DC5;

/// Frame `seed` of a vibration-like signal: two integer triangle waves and
/// noise, at a level that falls 6 dB per frame (C++ `makeFrame`).
#[inline(never)]
pub fn make_frame(x: &mut [i16; MEL_N], seed: u32) {
    let mut state = seed.wrapping_mul(7919);
    let shift = (seed % FRAMES as u32) as i32;
    let (mut a, mut b) = (0i32, 0i32);
    for v in x.iter_mut() {
        a = (a + 1311) & 0xFFFF;
        b = (b + 4096 + seed as i32 * 17) & 0xFFFF;
        let ta = if a < 32768 { a - 16384 } else { 49152 - a };
        let tb = if b < 32768 { b - 16384 } else { 49152 - b };
        let noise = i32::from((lcg(&mut state) >> 16) as u16 as i16) >> 3;
        *v = ((ta + (tb >> 1) + noise) >> shift).clamp(-32768, 32767) as i16;
    }
}

/// Feature vector `seed`: around one of the clusters, with noise (C++
/// `makeFeatures`).
#[inline(never)]
pub fn make_features(seed: u32) -> [i8; FEATURES] {
    let mut state = seed.wrapping_mul(104_729);
    let cluster = (seed % CLUSTERS as u32) as i32;
    core::array::from_fn(|f| {
        let centre = ((cluster * 37 + f as i32 * 11) % 160) - 80;
        let noise = (lcg(&mut state) >> 27) as i32 - 16;
        (centre + noise).clamp(-128, 127) as i8
    })
}

#[inline(never)]
pub fn run_mel(
    mel: &Mel<'_>,
    frame: &[i16; MEL_N],
    log_mel: &mut [i16; BANDS],
    ws: &mut Workspace,
) {
    mel.forward(frame, 0, log_mel, ws);
}

#[inline(never)]
pub fn run_mfcc(mfcc: &Mfcc<'_>, log_mel: &[i16; BANDS], out: &mut [i32; COEFFS]) {
    mfcc.forward(log_mel, out);
}

#[inline(never)]
pub fn run_kmeans(km: &KMeans<'_>, x: &[i8; FEATURES]) -> i32 {
    km.score(x)
}

#[inline(never)]
pub fn run_gaussian(g: &Gaussian<'_>, x: &[i8; FEATURES]) -> i32 {
    g.score(x)
}

#[inline(never)]
pub fn run_learn(trainer: &mut Trainer, x: &[i8; FEATURES]) {
    trainer.learn(x);
}

/// The mel and MFCC hashes over the `FRAMES` inputs (the first two of C++
/// `hashFeatures`).
#[inline(never)]
pub fn hash_mel(
    mel: &Mel<'_>,
    mfcc: &Mfcc<'_>,
    ws: &mut Workspace,
    frame: &mut [i16; MEL_N],
) -> [u32; 2] {
    let mut h = [FNV_OFFSET; 2];
    for s in 0..FRAMES as u32 {
        let mut log_mel = [0i16; BANDS];
        let mut coeffs = [0i32; COEFFS];
        make_frame(frame, s + 1);
        run_mel(mel, frame, &mut log_mel, ws);
        h[0] = fnv(h[0], log_mel.map(i16::to_le_bytes));
        run_mfcc(mfcc, &log_mel, &mut coeffs);
        h[1] = fnv(h[1], coeffs.map(i32::to_le_bytes));
    }
    h
}

/// The k-means and Gaussian hashes over the `FRAMES` inputs (the last two of
/// C++ `hashFeatures`).
#[inline(never)]
pub fn hash_scores(km: &KMeans<'_>, gauss: &Gaussian<'_>) -> [u32; 2] {
    let mut h = [FNV_OFFSET; 2];
    for s in 0..FRAMES as u32 {
        let x = make_features(s + 1);
        h[0] = fnv(h[0], [run_kmeans(km, &x).to_le_bytes()]);
        h[1] = fnv(h[1], [run_gaussian(gauss, &x).to_le_bytes()]);
    }
    h
}

/// Seed the clusters, as the board does before timing learning steps (C++
/// `trainerWarmup`).
#[inline(never)]
pub fn trainer_warmup(trainer: &mut Trainer) {
    trainer.reset();
    for s in 0..CLUSTERS as u32 {
        trainer.learn(&make_features(100 + s));
    }
}

/// The centroids and learn counts (C++ `hashTrainer`).
#[inline(never)]
pub fn hash_trainer(trainer: &Trainer) -> u32 {
    let mut h = FNV_OFFSET;
    for (centroid, &count) in trainer.centroids().iter().zip(trainer.learn_counts()) {
        h = fnv(h, centroid.map(|v| [v as u8]));
        h = fnv(h, [count.to_le_bytes()]);
    }
    h
}
