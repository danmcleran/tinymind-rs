//! The vibration anomaly example's program, shared by
//! `examples/vibration_anomaly.rs` and `tests/vibration_golden.rs` (which runs
//! it in-process against the C++ goldens). Port of tinymind (C++)
//! `examples/vibration_anomaly/vibration_anomaly.cpp`.
//!
//! A synthetic machine (3,200 samples per second, int16) runs at a speed that
//! wanders between 28 and 32 Hz and a load that moves its level by +-6 dB. The
//! stream is processed as a sensor node would, frame by frame (256 samples,
//! hop 128), with no buffering:
//!
//! 1. [`QLogMelSpectrogram`] gives 32 log-mel bands (Q8), less the frame's
//!    mean band (the spectrum's shape, not its level), mapped onto int8
//!    features at 1/8 nat per step
//! 2. learn, 120 s of normal running: [`QKMeansAnomalyTrainer`] learns 8
//!    centroids, one frame at a time
//! 3. calibrate, 60 s: the trainer measures each cluster's radius
//! 4. threshold, 60 s: the largest score seen, times 1.25, is the alarm
//!    threshold
//! 5. test: normal running, then three faults whose severity grows from 0 to
//!    full over 60 s each, with normal running in between: an outer-race
//!    bearing defect, imbalance and mechanical looseness
//!
//! For comparison, the same trainer with feature weighting off learns
//! alongside, and a one-component [`QDiagGaussianAnomaly`] is fitted on the
//! host to the same learning frames.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::int8::calibration::{
    build_diag_gaussian_anomaly, build_fft_twiddles, build_hann_window, build_mel_filterbank,
};
use tinymind::int8::{
    QDiagGaussianAnomaly, QKMeansAnomalyTrainer, QLogMelSpectrogram, QMelWorkspace,
};

#[allow(dead_code)]
#[path = "std_random.rs"]
mod std_random;
use std_random::{Mt19937, NormalF64};

const RATE: f64 = 3200.0;
const N: usize = 256;
const HALF: usize = N / 2;
const BINS: usize = N / 2 + 1;
const HOP: usize = 128;
const BANDS: usize = 32;
const CLUSTERS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Condition {
    Normal = 0,
    Bearing = 1,
    Imbalance = 2,
    Looseness = 3,
}

const CONDITION_NAME: [&str; 4] = ["normal", "bearing", "imbalance", "looseness"];
const FAULTS: [Condition; 3] = [
    Condition::Bearing,
    Condition::Imbalance,
    Condition::Looseness,
];

struct Segment {
    phase: &'static str,
    condition: Condition,
    seconds: f64,
}

const SEGMENTS: [Segment; 9] = [
    Segment {
        phase: "learn",
        condition: Condition::Normal,
        seconds: 120.0,
    },
    Segment {
        phase: "calibrate",
        condition: Condition::Normal,
        seconds: 60.0,
    },
    Segment {
        phase: "threshold",
        condition: Condition::Normal,
        seconds: 60.0,
    },
    Segment {
        phase: "test",
        condition: Condition::Normal,
        seconds: 60.0,
    },
    Segment {
        phase: "test",
        condition: Condition::Bearing,
        seconds: 60.0,
    },
    Segment {
        phase: "test",
        condition: Condition::Normal,
        seconds: 30.0,
    },
    Segment {
        phase: "test",
        condition: Condition::Imbalance,
        seconds: 60.0,
    },
    Segment {
        phase: "test",
        condition: Condition::Normal,
        seconds: 30.0,
    },
    Segment {
        phase: "test",
        condition: Condition::Looseness,
        seconds: 60.0,
    },
];

/// The machine: state carried across segments so the stream is continuous.
struct Machine {
    rng: Mt19937,
    gauss: NormalF64,
    phase: f64,
    speed: f64,
    load_db: f64,
    ring: f64,
    ring_prev: f64,
    impact_phase: f64,
}

impl Machine {
    fn new() -> Self {
        Machine {
            rng: Mt19937::new(11),
            gauss: NormalF64::new(0.0, 1.0),
            phase: 0.0,
            speed: 30.0,
            load_db: 0.0,
            ring: 0.0,
            ring_prev: 0.0,
            impact_phase: 0.0,
        }
    }

    fn next(&mut self, condition: Condition, severity: f64) -> i16 {
        // Slow random walks of speed and load.
        self.speed += 0.0005 * self.gauss.sample(&mut self.rng);
        self.speed = self.speed.clamp(28.0, 32.0);
        self.load_db += 0.002 * self.gauss.sample(&mut self.rng);
        self.load_db = self.load_db.clamp(-6.0, 6.0);

        self.phase += 2.0 * PI * self.speed / RATE;
        if self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        let phase = self.phase;
        let mut s = 0.0;
        let one_x = if condition == Condition::Imbalance {
            1.0 + 2.0 * severity
        } else {
            1.0
        };
        s += one_x * phase.sin();
        s += 0.35 * (2.0 * phase + 0.4).sin();
        s += 0.15 * (3.0 * phase + 1.1).sin();
        s += 0.08 * (4.0 * phase + 2.0).sin();
        if condition == Condition::Looseness {
            s += 0.25 * severity * (0.5 * phase + 0.3).sin();
            for h in 5..=12 {
                let h = f64::from(h);
                s += 0.12 * severity * (h * phase + 0.7 * h).sin();
            }
        }
        // Outer-race defect: impacts at 3.57x shaft speed ring a damped
        // 900 Hz resonance.
        let mut impact = 0.0;
        if condition == Condition::Bearing {
            self.impact_phase += 3.57 * self.speed / RATE;
            if self.impact_phase >= 1.0 {
                self.impact_phase -= 1.0;
                impact = 4.0 * severity;
            }
        }
        let omega = 2.0 * PI * 900.0 / RATE;
        let r = 0.985;
        let next_ring = 2.0 * r * omega.cos() * self.ring - r * r * self.ring_prev + impact;
        self.ring_prev = self.ring;
        self.ring = next_ring;
        s += 0.3 * self.ring;
        s += 0.04 * self.gauss.sample(&mut self.rng);

        let amplitude = 10f64.powf((self.load_db - 20.0) / 20.0) * 32767.0 / 2.0;
        (s * amplitude).round().clamp(-32768.0, 32767.0) as i16
    }
}

/// Log-mel (Q8 nats) to int8 features at 1/8 nat per step, relative to the
/// frame's mean band: `q = (ln E_m - mean_m ln E_m) * 8`. Removing the mean
/// keeps the spectrum's shape and drops its overall level, which the load
/// moves by +-6 dB in normal running.
fn to_features(log_mel: &[i16; BANDS]) -> [i8; BANDS] {
    let mut mean = 0i32;
    for &v in log_mel {
        mean += i32::from(v);
    }
    mean /= BANDS as i32;
    log_mel.map(|v| ((i32::from(v) - mean + 16) >> 5).clamp(-128, 127) as i8)
}

struct Frame {
    time: f64,
    condition: Condition,
    severity: f64,
    /// Weighted k-means, Euclidean k-means, Gaussian.
    score: [i32; 3],
}

/// The probability that an anomalous score beats a normal one (ties count
/// half).
fn auc(normal: &[i32], anomalous: &[i32]) -> f64 {
    let mut sorted = normal.to_vec();
    sorted.sort_unstable();
    let mut wins = 0.0;
    for &a in anomalous {
        let below = sorted.partition_point(|&v| v < a);
        let upto = sorted.partition_point(|&v| v <= a);
        wins += below as f64 + 0.5 * (upto - below) as f64;
    }
    wins / (anomalous.len() as f64 * sorted.len() as f64)
}

/// Runs the example: writes `vibration_anomaly.csv` and
/// `vibration_anomaly_summary.csv` into `out_dir` and the C++ example's
/// report to `stdout`.
pub fn run(out_dir: &Path, stdout: &mut impl Write) -> io::Result<()> {
    let window = build_hann_window::<N>(true);
    let (cos, sin) = build_fft_twiddles::<N, HALF>();
    let fb = build_mel_filterbank::<N, BANDS, { 2 * BINS }>(RATE, 5.0, 1600.0)
        .expect("2 * BINS weights always suffice");
    let mel: QLogMelSpectrogram<'_, N, HALF, BANDS> = QLogMelSpectrogram {
        window: &window,
        twiddle_cos: &cos,
        twiddle_sin: &sin,
        band_start: &fb.band_start,
        band_length: &fb.band_length,
        weights: fb.weights(),
        preemphasis: 0,
        log_floor: -32768,
    };
    let mut workspace = QMelWorkspace::<N, BINS>::new();
    let mut trainer = QKMeansAnomalyTrainer::<BANDS, CLUSTERS>::new();
    // The same trainer with a plain Euclidean distance, for comparison.
    let mut euclidean = QKMeansAnomalyTrainer::<BANDS, CLUSTERS>::new();
    euclidean.set_feature_weighting(false);

    // Host-side statistics of the learning frames, for the Gaussian.
    let mut sum = [0f64; BANDS];
    let mut sum_squares = [0f64; BANDS];
    let mut learn_frames = 0usize;
    let mut gauss_means = [[0i8; BANDS]; 1];
    let mut gauss_precision = [[0u32; BANDS]; 1];
    let mut gauss_bias = [0i32; 1];
    let mut gaussian_built = false;
    // Largest score in the threshold phase, and the threshold, per detector.
    let mut max_score = [0i32, 0, i32::MIN];
    let mut threshold = [0i32; 3];
    let mut test_frames: Vec<Frame> = Vec::new();

    let mut machine = Machine::new();
    let mut ring_buffer = [0i16; N];
    let mut filled = 0usize;
    let mut since_frame = 0usize;
    let mut previous_sample = 0i16;
    let mut time = 0.0f64;
    let mut last_phase = "";
    for seg in &SEGMENTS {
        let phase = seg.phase;
        if phase == "threshold" && last_phase != "threshold" {
            // Fit the Gaussian to the learning frames (float, on the host).
            let scale = 0.125;
            let lf = learn_frames as f64;
            let mut means = [[0f64; BANDS]; 1];
            let mut variances = [[0f64; BANDS]; 1];
            for m in 0..BANDS {
                means[0][m] = scale * (sum[m] / lf);
                variances[0][m] =
                    scale * scale * (sum_squares[m] / lf - (sum[m] / lf) * (sum[m] / lf));
            }
            (gauss_means, gauss_precision, gauss_bias) =
                build_diag_gaussian_anomaly(&means, &variances, &[1.0], scale as f32, 0, 0.25);
            gaussian_built = true;
        }
        if phase == "test" && last_phase != "test" {
            threshold[0] = (1.25 * f64::from(max_score[0])) as i32;
            threshold[1] = (1.25 * f64::from(max_score[1])) as i32;
            // The Gaussian's score is a log-likelihood: give it the same 25%
            // margin on the part of the score above its minimum, the bias.
            threshold[2] = gauss_bias[0]
                .wrapping_add((1.25 * f64::from(max_score[2].wrapping_sub(gauss_bias[0]))) as i32);
        }
        last_phase = phase;

        let samples = (seg.seconds * RATE) as usize;
        for n in 0..samples {
            let severity = n as f64 / samples as f64;
            let x = machine.next(seg.condition, severity);
            time += 1.0 / RATE;
            // Slide the frame: keep the last N samples in order.
            if filled < N {
                ring_buffer[filled] = x;
                filled += 1;
            } else {
                ring_buffer.copy_within(1.., 0);
                ring_buffer[N - 1] = x;
            }
            if filled < N {
                continue;
            }
            since_frame += 1;
            if since_frame < HOP {
                continue;
            }
            since_frame = 0;

            let mut log_mel = [0i16; BANDS];
            mel.forward(&ring_buffer, previous_sample, &mut log_mel, &mut workspace);
            previous_sample = ring_buffer[HOP - 1];
            let features = to_features(&log_mel);

            if phase == "learn" {
                trainer.learn(&features);
                euclidean.learn(&features);
                for (m, &f) in features.iter().enumerate() {
                    sum[m] += f64::from(f);
                    sum_squares[m] += f64::from(f) * f64::from(f);
                }
                learn_frames += 1;
                continue;
            }
            if phase == "calibrate" {
                trainer.calibrate(&features);
                euclidean.calibrate(&features);
                continue;
            }
            let gaussian: QDiagGaussianAnomaly<'_, BANDS, 1> = QDiagGaussianAnomaly {
                means: &gauss_means,
                precision: &gauss_precision,
                bias: &gauss_bias,
            };
            let scores = [
                trainer.model().score(&features),
                euclidean.model().score(&features),
                if gaussian_built {
                    gaussian.score(&features)
                } else {
                    0
                },
            ];
            if phase == "threshold" {
                for d in 0..3 {
                    max_score[d] = max_score[d].max(scores[d]);
                }
                continue;
            }
            test_frames.push(Frame {
                time,
                condition: seg.condition,
                severity: if seg.condition == Condition::Normal {
                    0.0
                } else {
                    severity
                },
                score: scores,
            });
        }
    }

    let mut csv = BufWriter::new(File::create(out_dir.join("vibration_anomaly.csv"))?);
    let mut summary = BufWriter::new(File::create(out_dir.join("vibration_anomaly_summary.csv"))?);
    writeln!(
        csv,
        "time_s,condition,severity,kmeans_score,kmeans_threshold,euclidean_score,euclidean_threshold,gaussian_score,gaussian_threshold"
    )?;
    for f in &test_frames {
        writeln!(
            csv,
            "{:.3},{},{:.4},{},{},{},{},{},{}",
            f.time,
            CONDITION_NAME[f.condition as usize],
            f.severity,
            f.score[0],
            threshold[0],
            f.score[1],
            threshold[1],
            f.score[2],
            threshold[2]
        )?;
    }
    csv.flush()?;

    writeln!(
        stdout,
        "learned {} clusters from {learn_frames} frames; thresholds: k-means {} and {} (Q8 radii^2), Gaussian {} (Q8 nats)",
        trainer.seeded_clusters(),
        threshold[0],
        threshold[1],
        threshold[2]
    )?;
    writeln!(
        summary,
        "detector,condition,frames,auc,alarm_rate,first_sustained_alarm_severity"
    )?;
    let detector_name = ["kmeans", "euclidean", "gaussian"];
    for d in 0..3 {
        let mut normal_scores = Vec::new();
        let mut normal_alarms = 0usize;
        for f in test_frames
            .iter()
            .filter(|f| f.condition == Condition::Normal)
        {
            normal_scores.push(f.score[d]);
            normal_alarms += usize::from(f.score[d] > threshold[d]);
        }
        let false_alarm_rate = normal_alarms as f64 / normal_scores.len() as f64;
        writeln!(
            summary,
            "{},normal,{},,{false_alarm_rate:.4},",
            detector_name[d],
            normal_scores.len()
        )?;
        writeln!(
            stdout,
            "{:<9}  normal running: {} frames, false alarms {:.2}%",
            detector_name[d],
            normal_scores.len(),
            100.0 * false_alarm_rate
        )?;
        for c in FAULTS {
            let mut scores = Vec::new();
            let mut alarms = 0usize;
            let mut first_sustained = -1.0f64;
            let mut run = 0;
            for f in test_frames.iter().filter(|f| f.condition == c) {
                let v = f.score[d];
                let alarm = v > threshold[d];
                scores.push(v);
                alarms += usize::from(alarm);
                // Sustained: 8 alarms in a row (about a third of a second).
                run = if alarm { run + 1 } else { 0 };
                if run == 8 && first_sustained < 0.0 {
                    first_sustained = f.severity;
                }
            }
            let a = auc(&normal_scores, &scores);
            writeln!(
                summary,
                "{},{},{},{a:.4},{:.4},{first_sustained:.4}",
                detector_name[d],
                CONDITION_NAME[c as usize],
                scores.len(),
                alarms as f64 / scores.len() as f64
            )?;
            writeln!(
                stdout,
                "{:<9}  {:<9}: AUC {a:.3}, alarms on {:.1}% of frames, first sustained alarm at severity {first_sustained:.2}",
                detector_name[d],
                CONDITION_NAME[c as usize],
                100.0 * alarms as f64 / scores.len() as f64
            )?;
        }
    }
    summary.flush()?;
    writeln!(
        stdout,
        "on-device state: k-means trainer {} bytes, log-mel workspace {} bytes",
        size_of::<QKMeansAnomalyTrainer<BANDS, CLUSTERS>>(),
        size_of::<QMelWorkspace<N, BINS>>()
    )?;
    Ok(())
}
