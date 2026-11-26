// QKMeansAnomaly, QKMeansAnomalyTrainer, QDiagGaussianAnomaly and
// calibration::build_diag_gaussian_anomaly against tinymind (C++)
// (cpp/qanomaly.hpp): scores on direct random tables, the trainer's state and
// scores after clustered streams (step caps, weighting on and off, the edge
// cases), and the Gaussian score on random and on built tables.
// tests/golden/qanomaly/stdout.txt is the output of
// scripts/cpp/anomaly_sweep.cpp (scripts/compare-with-cpp.sh rebuilds it and
// checks the golden is current); this file computes the same lines.
// The sweep itself is in tests/sweeps/qanomaly.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/qanomaly.rs"]
mod qanomaly;

use tinymind::int8::calibration::build_diag_gaussian_anomaly;

struct Host;

impl qanomaly::Calibrate for Host {
    fn gaussian<const F: usize, const K: usize>(
        &mut self,
        means: &[[f64; F]; K],
        variances: &[[f64; F]; K],
        weights: &[f64; K],
        scale: f32,
        zero_point: i32,
        floor: f64,
    ) -> Option<qanomaly::Gaussian<F, K>> {
        Some(build_diag_gaussian_anomaly(
            means, variances, weights, scale, zero_point, floor,
        ))
    }
}

#[test]
fn anomaly_sweep_matches_cpp() {
    let mut out = String::new();
    qanomaly::run(&mut out, &mut Host);
    let golden = include_str!("golden/qanomaly/stdout.txt");
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
