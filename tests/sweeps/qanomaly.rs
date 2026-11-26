// The anomaly-score sweep, shared by tests/qanomaly_sweep.rs (on the host,
// against tinymind (C++)'s output, with
// calibration::build_diag_gaussian_anomaly) and bench/src/sweeps.rs (on
// Cortex-M, against the same golden; the built Gaussians are only hashed
// there, so their trials run on the host only). no_std and allocation-free.
// The C++ is scripts/cpp/anomaly_sweep.cpp.

use core::fmt::Write;

use tinymind::int8::{QDiagGaussianAnomaly, QKMeansAnomaly, QKMeansAnomalyTrainer};

/// The means, precisions and biases of a built diagonal Gaussian.
pub type Gaussian<const F: usize, const K: usize> = ([[i8; F]; K], [[u32; F]; K], [i32; K]);

/// What the sweep needs from `int8::calibration` (std only).
pub trait Calibrate {
    /// `build_diag_gaussian_anomaly(...)`; `None` skips the built trials.
    fn gaussian<const F: usize, const K: usize>(
        &mut self,
        means: &[[f64; F]; K],
        variances: &[[f64; F]; K],
        weights: &[f64; K],
        scale: f32,
        zero_point: i32,
        floor: f64,
    ) -> Option<Gaussian<F, K>>;
}

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

fn hash_bytes(bytes: impl IntoIterator<Item = u8>, mut hash: u32) -> u32 {
    for b in bytes {
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

/// Values separated by commas, written as they are formatted (no allocation).
struct Join<I>(I);

impl<I: Iterator<Item = T> + Clone, T: core::fmt::Display> core::fmt::Display for Join<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, x) in self.0.clone().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{x}")?;
        }
        Ok(())
    }
}

fn join<T: core::fmt::Display, I: IntoIterator<Item = T>>(xs: I) -> Join<I::IntoIter>
where
    I::IntoIter: Clone,
{
    Join(xs.into_iter())
}

fn direct_kmeans<const F: usize, const K: usize>(out: &mut impl Write, rng: &mut Rng, name: &str) {
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
        let scores: [i32; 12] = core::array::from_fn(|s| model.score(&sample::<F>(rng, s)));
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
    out: &mut impl Write,
    name: &str,
    run: usize,
    t: &QKMeansAnomalyTrainer<F, K>,
) {
    let bytes = t.centroids().iter().flatten().map(|&v| v as u8);
    let weight_bytes = t.feature_weights().iter().flat_map(|w| w.to_le_bytes());
    writeln!(
        out,
        "trainer {name} run={run} seeded={} counts={} calibration={} centroids={:08x} weights={} weight_hash={:08x} inv={}",
        t.seeded_clusters(),
        join(t.learn_counts()),
        join(t.calibration_counts()),
        hash_bytes(bytes, FNV),
        join(t.feature_weights().iter().take(8)),
        hash_bytes(weight_bytes, FNV),
        join(t.model().inv_radius_sq)
    )
    .unwrap();
}

fn trainers<const F: usize, const K: usize>(out: &mut impl Write, rng: &mut Rng, name: &str) {
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
        let scores: [i32; 16] = core::array::from_fn(|s| {
            let x = if s < 8 {
                stream.draw(rng)
            } else {
                sample::<F>(rng, s - 8)
            };
            model.score(&x)
        });
        writeln!(out, "trainer {name} run={run} scores={}", join(scores)).unwrap();
    }
}

fn gaussians<const F: usize, const K: usize>(
    out: &mut impl Write,
    rng: &mut Rng,
    cal: &mut impl Calibrate,
    name: &str,
) {
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
            // Without a builder (the device) the previous model scores the
            // samples, so the draws stay in step; those lines are host-only.
            if let Some(built) = cal.gaussian(&fm, &fv, &fw, scale, zp, floor) {
                (means, precision, bias) = built;
                let mean_bytes = means.iter().flatten().map(|&v| v as u8);
                let precision_bytes = precision.iter().flatten().flat_map(|v| v.to_le_bytes());
                writeln!(
                    out,
                    "gaussian {name} trial={trial} built means={:08x} precision={:08x} bias={}",
                    hash_bytes(mean_bytes, FNV),
                    hash_bytes(precision_bytes, FNV),
                    join(bias)
                )
                .unwrap();
            }
        }
        let model: QDiagGaussianAnomaly<'_, F, K> = QDiagGaussianAnomaly {
            means: &means,
            precision: &precision,
            bias: &bias,
        };
        let scores: [i32; 12] = core::array::from_fn(|s| model.score(&sample::<F>(rng, s)));
        writeln!(out, "gaussian {name} trial={trial} scores={}", join(scores)).unwrap();
    }
}

/// The whole sweep, in the golden's order.
pub fn run(out: &mut impl Write, cal: &mut impl Calibrate) {
    let rng = &mut Rng(1);
    direct_kmeans::<1, 1>(out, rng, "1x1");
    direct_kmeans::<4, 3>(out, rng, "4x3");
    direct_kmeans::<7, 2>(out, rng, "7x2");
    direct_kmeans::<32, 8>(out, rng, "32x8");
    trainers::<1, 1>(out, rng, "1x1");
    trainers::<4, 3>(out, rng, "4x3");
    trainers::<7, 2>(out, rng, "7x2");
    trainers::<32, 8>(out, rng, "32x8");
    trainers::<600, 2>(out, rng, "600x2");
    gaussians::<1, 1>(out, rng, cal, "1x1");
    gaussians::<4, 3>(out, rng, cal, "4x3");
    gaussians::<32, 2>(out, rng, cal, "32x2");
}
