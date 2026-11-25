// The features workload of tinymind (C++)'s on-board benchmark
// (benchmarks/tier1_onboard: log-mel, MFCC, k-means and Gaussian scores,
// k-means learning) on the host: every table this crate's calibration builds
// (the window, twiddles, mel filters, DCT, the k-means model a device would
// learn and calibrate, the Gaussian mixture) must equal the one tinymind
// (C++)'s gen_tables.cpp wrote, and every workload hash must equal its hash.
// tests/golden/features/tables.h is gen_tables.cpp's output, unchanged
// (scripts/compare-with-cpp.sh rebuilds it and checks the golden is
// current); bench/src/features.rs runs the same workload on the boards from
// that header's tables.

#[path = "sweeps/features_bench.rs"]
mod features_bench;

use features_bench::*;
use tinymind::int8::calibration::{
    build_dct_matrix, build_diag_gaussian_anomaly, build_fft_twiddles, build_hann_window,
    build_mel_filterbank,
};

/// The values of `static const <type> <name>[n] = {...};` in the golden.
fn array(header: &str, name: &str) -> Vec<i64> {
    let line = header
        .lines()
        .find(|l| l.contains(&format!(" {name}[")))
        .unwrap_or_else(|| panic!("{name} not in the golden"));
    let body = line.split('{').nth(1).unwrap().split('}').next().unwrap();
    body.split(',').map(|v| v.trim().parse().unwrap()).collect()
}

fn hash(header: &str, name: &str) -> u32 {
    let line = header
        .lines()
        .find(|l| l.starts_with(&format!("#define {name} ")))
        .unwrap_or_else(|| panic!("{name} not in the golden"));
    u32::from_str_radix(
        line.rsplit(' ')
            .next()
            .unwrap()
            .trim_start_matches("0x")
            .trim_end_matches('u'),
        16,
    )
    .unwrap()
}

fn same<T: Copy + Into<i64>>(header: &str, name: &str, ours: &[T]) {
    let ours: Vec<i64> = ours.iter().map(|&v| v.into()).collect();
    assert_eq!(ours, array(header, name), "{name}");
}

#[test]
fn features_tables_and_hashes_match_cpp() {
    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/features/tables.h"
    ))
    .unwrap();

    // Log-mel front end: 256 samples at 3.2 kHz, 32 bands from 5 to 1600 Hz.
    let window = build_hann_window::<MEL_N>(true);
    let (cos, sin) = build_fft_twiddles::<MEL_N, HALF>();
    let fb = build_mel_filterbank::<MEL_N, BANDS, { 2 * BINS }>(3200.0, 5.0, 1600.0).unwrap();
    let dct = build_dct_matrix::<BANDS, COEFFS>();
    same(&header, "kWindow", &window);
    same(&header, "kTwCos", &cos);
    same(&header, "kTwSin", &sin);
    same(&header, "kBandStart", &fb.band_start);
    same(&header, "kBandLength", &fb.band_length);
    same(&header, "kMelWeights", fb.weights());
    same(&header, "kDct", dct.as_flattened());
    let mel: Mel<'_> = Mel {
        window: &window,
        twiddle_cos: &cos,
        twiddle_sin: &sin,
        band_start: &fb.band_start,
        band_length: &fb.band_length,
        weights: fb.weights(),
        preemphasis: PREEMPHASIS,
        log_floor: -32768,
    };
    let mfcc: Mfcc<'_> = Mfcc { dct: &dct };

    // k-means, trained and calibrated as a device would.
    let mut trainer = Box::new(Trainer::new());
    for s in 0..400 {
        trainer.learn(&make_features(1000 + s));
    }
    for s in 0..200 {
        trainer.calibrate(&make_features(5000 + s));
    }
    let centroids = *trainer.centroids();
    let inv = *trainer.model().inv_radius_sq;
    let weights = *trainer.feature_weights();
    same(&header, "kCentroids", centroids.as_flattened());
    same(&header, "kInvRadius", &inv);
    same(&header, "kFeatureWeight", &weights);
    let km: KMeans<'_> = KMeans {
        centroids: &centroids,
        inv_radius_sq: &inv,
        feature_weight: Some(&weights),
    };

    // A diagonal Gaussian mixture over the first COMPONENTS clusters.
    let means: [[f64; FEATURES]; COMPONENTS] =
        core::array::from_fn(|k| core::array::from_fn(|f| ((k * 37 + f * 11) % 160) as f64 - 80.0));
    let (g_means, g_precision, g_bias) = build_diag_gaussian_anomaly(
        &means,
        &[[85.0; FEATURES]; COMPONENTS],
        &[1.0 / COMPONENTS as f64; COMPONENTS],
        1.0,
        0,
        0.25,
    );
    same(&header, "kGaussMeans", g_means.as_flattened());
    same(&header, "kGaussPrecision", g_precision.as_flattened());
    same(&header, "kGaussBias", &g_bias);
    let gauss: Gaussian<'_> = Gaussian {
        means: &g_means,
        precision: &g_precision,
        bias: &g_bias,
    };

    let mut ws = Box::new(Workspace::new());
    let mut frame = [0i16; MEL_N];
    let [mel_hash, mfcc_hash] = hash_mel(&mel, &mfcc, &mut ws, &mut frame);
    let [kmeans_hash, gaussian_hash] = hash_scores(&km, &gauss);
    assert_eq!(mel_hash, hash(&header, "HASH_MEL"), "mel");
    assert_eq!(mfcc_hash, hash(&header, "HASH_MFCC"), "mfcc");
    assert_eq!(kmeans_hash, hash(&header, "HASH_KMEANS"), "kmeans");
    assert_eq!(gaussian_hash, hash(&header, "HASH_GAUSSIAN"), "gaussian");

    // The board's learning sequence: warm up, then 33 learn() calls.
    trainer_warmup(&mut trainer);
    for r in 0..33u32 {
        run_learn(&mut trainer, &make_features(1 + r % FRAMES as u32));
    }
    assert_eq!(hash_trainer(&trainer), hash(&header, "HASH_LEARN"), "learn");
}
