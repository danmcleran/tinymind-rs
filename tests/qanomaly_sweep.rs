// QKMeansAnomaly, QKMeansAnomalyTrainer, QDiagGaussianAnomaly and
// calibration::build_diag_gaussian_anomaly against tinymind (C++)
// (cpp/qanomaly.hpp): scores on direct random tables, the trainer's state and
// scores after clustered streams (step caps, weighting on and off, the edge
// cases), and the Gaussian score on random and on built tables.
// tests/golden/qanomaly/stdout.txt is the output of
// scripts/cpp/anomaly_sweep.cpp (scripts/compare-with-cpp.sh rebuilds it and
// checks the golden is current); this file computes the same lines.
use std::fmt::Write;

use tinymind::int8::calibration::build_diag_gaussian_anomaly;
use tinymind::int8::{QDiagGaussianAnomaly, QKMeansAnomaly, QKMeansAnomalyTrainer};

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn i8(&mut self) -> i8 {
        (self.next() >> 24) as u8 as i8
    }
    fn u16(&mut self) -> u16 {
        (self.next() >> 16) as u16
    }
    fn u32(&mut self) -> u32 {
        let high = self.next() >> 16;
        (high << 16) | (self.next() >> 16)
    }
    fn unit(&mut self) -> f64 {
        f64::from(self.next() >> 8) / 16_777_216.0
    }
}

fn hash_bytes<'a>(bytes: impl IntoIterator<Item = &'a u8>, mut hash: u32) -> u32 {
    for &b in bytes {
        hash = (hash ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    hash
}

const FNV: u32 = 0x811C_9DC5;

fn sample<const F: usize>(rng: &mut Rng, kind: usize) -> [i8; F] {
    core::array::from_fn(|_| match kind {
        0 => -128,
        1 => 127,
        _ => rng.i8(),
    })
}

fn join<T: std::fmt::Display>(xs: impl IntoIterator<Item = T>) -> String {
    xs.into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn direct_kmeans<const F: usize, const K: usize>(out: &mut String, rng: &mut Rng, name: &str) {
    let mut centroids = [[0i8; F]; K];
    let mut inv = [0u32; K];
    let mut weights = [0u16; F];
    for trial in 0..4 {
        for k in 0..K {
            for c in centroids[k].iter_mut() {
                *c = rng.i8();
            }
            inv[k] = match rng.next() >> 29 {
                0 => 0,
                1 => u32::MAX,
                2 => rng.next() >> 22,
                _ => rng.u32(),
            };
        }
        for w in weights.iter_mut() {
            *w = if trial == 3 { 65535 } else { rng.u16() };
        }
        let weighted = trial % 2 == 1;
        let model: QKMeansAnomaly<'_, F, K> = QKMeansAnomaly {
            centroids: &centroids,
            inv_radius_sq: &inv,
            feature_weight: if weighted { Some(&weights) } else { None },
        };
        let scores: Vec<i32> = (0..12).map(|s| model.score(&sample::<F>(rng, s))).collect();
        writeln!(
            out,
            "kmeans {name} trial={trial} weighted={} scores={}",
            u8::from(weighted),
            join(scores)
        )
        .unwrap();
    }
}

struct Stream<const F: usize, const K: usize> {
    centers: [[i8; F]; K],
    spread: [i32; F],
}

impl<const F: usize, const K: usize> Stream<F, K> {
    fn new(rng: &mut Rng, constant: bool) -> Self {
        let common = (i32::from(rng.i8()) / 2) as i8;
        let mut centers = [[0i8; F]; K];
        for c in centers.iter_mut().flatten() {
            *c = if constant {
                common
            } else {
                (i32::from(rng.i8()) / 2) as i8
            };
        }
        let spread = core::array::from_fn(|f| {
            if constant || f == 1 {
                0
            } else if F >= 512 {
                if f == 0 { 127 } else { 1 }
            } else {
                1 + (rng.next() >> 28) as i32
            }
        });
        Stream { centers, spread }
    }

    fn draw(&self, rng: &mut Rng) -> [i8; F] {
        let k = ((rng.next() >> 16) as usize) % K;
        core::array::from_fn(|f| {
            let s = self.spread[f];
            let noise = ((rng.next() >> 16) % (2 * s + 1) as u32) as i32 - s;
            (i32::from(self.centers[k][f]) + noise).clamp(-128, 127) as i8
        })
    }
}

fn print_trainer<const F: usize, const K: usize>(
    out: &mut String,
    name: &str,
    run: usize,
    t: &QKMeansAnomalyTrainer<F, K>,
) {
    let bytes: Vec<u8> = t.centroids().iter().flatten().map(|&v| v as u8).collect();
    let weight_bytes: Vec<u8> = t
        .feature_weights()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    writeln!(
        out,
        "trainer {name} run={run} seeded={} counts={} calibration={} centroids={:08x} weights={} weight_hash={:08x} inv={}",
        t.seeded_clusters(),
        join(t.learn_counts()),
        join(t.calibration_counts()),
        hash_bytes(&bytes, FNV),
        join(t.feature_weights().iter().take(8)),
        hash_bytes(&weight_bytes, FNV),
        join(t.model().inv_radius_sq)
    )
    .unwrap();
}

fn trainers<const F: usize, const K: usize>(out: &mut String, rng: &mut Rng, name: &str) {
    const CAPS: [u16; 4] = [65535, 0, 5, 40];
    let mut t = QKMeansAnomalyTrainer::<F, K>::new();
    for run in 0..9 {
        let stream = Stream::<F, K>::new(rng, run == 8);
        t.reset();
        if run != 0 {
            t.set_maximum_count(CAPS[run % 4]);
        }
        t.set_feature_weighting(!(4..6).contains(&run));
        let learn_count = if run == 5 { K / 2 } else { 50 + 37 * run };
        if run == 7 {
            t.calibrate(&stream.draw(rng));
            print_trainer(out, name, run, &t);
            t.reset();
        }
        for _ in 0..learn_count {
            t.learn(&stream.draw(rng));
        }
        for _ in 0..40 + 3 * run {
            t.calibrate(&stream.draw(rng));
        }
        t.learn(&stream.draw(rng));
        print_trainer(out, name, run, &t);
        let model = t.model();
        let scores: Vec<i32> = (0..16)
            .map(|s| {
                let x = if s < 8 {
                    stream.draw(rng)
                } else {
                    sample::<F>(rng, s - 8)
                };
                model.score(&x)
            })
            .collect();
        writeln!(out, "trainer {name} run={run} scores={}", join(scores)).unwrap();
    }
}

fn gaussians<const F: usize, const K: usize>(out: &mut String, rng: &mut Rng, name: &str) {
    let mut means = [[0i8; F]; K];
    let mut precision = [[0u32; F]; K];
    let mut bias = [0i32; K];
    for trial in 0..6 {
        if trial < 3 {
            for k in 0..K {
                for f in 0..F {
                    means[k][f] = rng.i8();
                    precision[k][f] = if trial == 2 { u32::MAX } else { rng.u32() };
                }
                bias[k] = if trial == 1 {
                    rng.u32() as i32
                } else {
                    (rng.next() >> 20) as i32 - 2048
                };
            }
            if trial == 2 {
                bias[0] = i32::MIN;
            }
        } else {
            let mut fm = [[0f64; F]; K];
            let mut fv = [[0f64; F]; K];
            let mut fw = [0f64; K];
            let scale = (0.01 + rng.unit()) as f32;
            let zp = (rng.next() >> 26) as i32 - 32;
            let mut total = 0.0;
            for k in 0..K {
                for f in 0..F {
                    fm[k][f] = (rng.unit() - 0.5) * 300.0 * f64::from(scale);
                    let u = rng.unit();
                    fv[k][f] = if u < 0.2 {
                        1e-9
                    } else {
                        u * u * 400.0 * f64::from(scale) * f64::from(scale)
                    };
                }
                fw[k] = 0.05 + rng.unit();
                total += fw[k];
            }
            for w in fw.iter_mut() {
                *w /= total;
            }
            let floor = if trial == 5 { 0.0001 } else { 0.25 };
            (means, precision, bias) = build_diag_gaussian_anomaly(&fm, &fv, &fw, scale, zp, floor);
            let mean_bytes: Vec<u8> = means.iter().flatten().map(|&v| v as u8).collect();
            let precision_bytes: Vec<u8> = precision
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            writeln!(
                out,
                "gaussian {name} trial={trial} built means={:08x} precision={:08x} bias={}",
                hash_bytes(&mean_bytes, FNV),
                hash_bytes(&precision_bytes, FNV),
                join(bias)
            )
            .unwrap();
        }
        let model: QDiagGaussianAnomaly<'_, F, K> = QDiagGaussianAnomaly {
            means: &means,
            precision: &precision,
            bias: &bias,
        };
        let scores: Vec<i32> = (0..12).map(|s| model.score(&sample::<F>(rng, s))).collect();
        writeln!(out, "gaussian {name} trial={trial} scores={}", join(scores)).unwrap();
    }
}

#[test]
fn anomaly_sweep_matches_cpp() {
    let mut out = String::new();
    let rng = &mut Rng(1);
    direct_kmeans::<1, 1>(&mut out, rng, "1x1");
    direct_kmeans::<4, 3>(&mut out, rng, "4x3");
    direct_kmeans::<7, 2>(&mut out, rng, "7x2");
    direct_kmeans::<32, 8>(&mut out, rng, "32x8");
    trainers::<1, 1>(&mut out, rng, "1x1");
    trainers::<4, 3>(&mut out, rng, "4x3");
    trainers::<7, 2>(&mut out, rng, "7x2");
    trainers::<32, 8>(&mut out, rng, "32x8");
    trainers::<600, 2>(&mut out, rng, "600x2");
    gaussians::<1, 1>(&mut out, rng, "1x1");
    gaussians::<4, 3>(&mut out, rng, "4x3");
    gaussians::<32, 2>(&mut out, rng, "32x2");

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/qanomaly/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
