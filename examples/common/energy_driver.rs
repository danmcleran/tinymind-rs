//! The energy_efficiency example's program, shared by
//! `examples/energy_efficiency.rs` (which parses the command line) and
//! `tests/energy_golden.rs` (which runs it in-process against the C++ goldens).
//! Port of tinymind `examples/energy_efficiency/energy_efficiency.cpp`; see
//! `examples/energy_efficiency.rs`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::rand::GlibcRand;
use tinymind::{Linear, MultilayerPerceptron, Q16_16, Relu};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::{fmt_fixed, fmt_g};

const NUMBER_OF_INPUTS: usize = 8;
const NUMBER_OF_NEURONS_PER_HIDDEN_LAYER: usize = 16;
const NUMBER_OF_OUTPUTS: usize = 2;

/// ReLU hidden, linear output: a regression.
type NeuralNetwork = MultilayerPerceptron<
    Q16_16,
    Relu,
    Linear,
    NUMBER_OF_INPUTS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
>;

/// C++ `WEIGHT_INIT_SEED`: seeds the initial weights only.
const WEIGHT_INIT_SEED: u32 = 1;
/// Seeds the shuffle and the training-sample draws.
const RANDOM_SEED: u32 = 7;
const ITERATIONS: u32 = 40_000;
const REPORT_EVERY: u32 = 2_000;
/// Targets are z-scores divided by this, so a linear output's gradients stay
/// inside Q16.16.
const TARGET_SCALE: f64 = 6.0;

#[derive(Clone, Copy)]
struct Sample {
    /// X1..X8: building design features.
    feat: [f64; 8],
    /// Y1 heating load, Y2 cooling load.
    load: [f64; 2],
}

/// Per-column z-score statistics fit on the training split.
struct Stats<const N: usize> {
    mean: [f64; N],
    stdev: [f64; N],
}

/// Train on `data_path`, evaluate, write the two CSV files into `out_dir` and
/// the training log and test metrics to `stdout`.
pub fn run(out_dir: &Path, data_path: &Path, stdout: &mut dyn Write) -> io::Result<()> {
    // The C++ seeds rand() with WEIGHT_INIT_SEED, builds the network (drawing
    // its weights), then re-seeds with 7. Two generators here.
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let mut nn = NeuralNetwork::new(|| random_weight(init_rng.next_i32()));

    let Some(mut data) = load_csv(data_path) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("could not open or parse {}", data_path.display()),
        ));
    };
    writeln!(stdout, "Loaded {} rows from ENB2012_data.csv.", data.len())?;

    // Deterministic shuffle + 80/20 split.
    let mut rng = GlibcRand::new(RANDOM_SEED);
    for i in (2..=data.len()).rev() {
        let j = rng.next_i32() as usize % i;
        data.swap(i - 1, j);
    }
    let n_train = data.len() * 4 / 5;
    let (train, test) = data.split_at(n_train);
    writeln!(stdout, "Train: {}  Test: {}", train.len(), test.len())?;

    let feat_stats = fit_stats(train, |s| s.feat);
    let target_stats = fit_stats(train, |s| s.load);

    // A modest learning rate, no momentum or acceleration, and gradients
    // clipped to [-1, 1] (C++ GradientClipByValue<ValueType>): a linear output's
    // unbounded gradient otherwise saturates Q16.16.
    nn.set_learning_rate(to_q(0.02));
    nn.set_momentum_rate(to_q(0.0));
    nn.set_acceleration_rate(to_q(0.0));
    nn.set_gradient_clip(Some((Q16_16::from_int(-1), Q16_16::ONE)));

    // ---- Training ----
    let mut loss_csv = BufWriter::new(File::create(out_dir.join("energy_loss.csv"))?);
    writeln!(loss_csv, "iter,avg_err")?;
    let mut err_sum = 0.0;
    for it in 0..ITERATIONS {
        let s = &train[rng.next_i32() as usize % train.len()];
        let input = to_input(s, &feat_stats);
        let target = to_target(s, &target_stats);

        nn.feed_forward(&input);
        let err = nn.calculate_error(&target);
        if !err.is_within_zero_tolerance() {
            nn.train(&target);
        }
        err_sum += err.to_f64().abs();

        if (it + 1) % REPORT_EVERY == 0 {
            let avg = err_sum / f64::from(REPORT_EVERY);
            writeln!(
                stdout,
                "iter {:>6}   avg|err| = {}",
                it + 1,
                fmt_fixed(avg, 4)
            )?;
            writeln!(loss_csv, "{},{}", it + 1, fmt_g(avg))?;
            err_sum = 0.0;
        }
    }
    loss_csv.flush()?;

    // ---- Evaluation (real load units) ----
    let mut pred = BufWriter::new(File::create(out_dir.join("energy_pred.csv"))?);
    writeln!(pred, "sample,y1_true,y1_pred,y2_true,y2_pred")?;
    let mut sum_abs = [0.0; 2];
    let mut sum_sq_res = [0.0; 2];
    let mut sum_true = [0.0; 2];
    let mut sum_sq_true = [0.0; 2];
    for (i, s) in test.iter().enumerate() {
        nn.feed_forward(&to_input(s, &feat_stats));
        let learned = nn.learned_values();
        let mut y_pred = [0.0; 2];
        for t in 0..2 {
            y_pred[t] = from_target(learned[t].to_f64(), t, &target_stats);
            let r = y_pred[t] - s.load[t];
            sum_abs[t] += r.abs();
            sum_sq_res[t] += r * r;
            sum_true[t] += s.load[t];
            sum_sq_true[t] += s.load[t] * s.load[t];
        }
        writeln!(
            pred,
            "{i},{},{},{},{}",
            fmt_g(s.load[0]),
            fmt_g(y_pred[0]),
            fmt_g(s.load[1]),
            fmt_g(y_pred[1])
        )?;
    }
    pred.flush()?;

    let n = test.len() as f64;
    let mut mae = [0.0; 2];
    let mut r2 = [0.0; 2];
    for t in 0..2 {
        mae[t] = sum_abs[t] / n;
        let mean_true = sum_true[t] / n;
        let ss_tot = sum_sq_true[t] - n * mean_true * mean_true;
        r2[t] = if ss_tot > 1e-9 {
            1.0 - sum_sq_res[t] / ss_tot
        } else {
            0.0
        };
    }
    writeln!(
        stdout,
        "\nHeating load (Y1):  MAE={}  R2={}",
        fmt_fixed(mae[0], 4),
        fmt_fixed(r2[0], 4)
    )?;
    writeln!(
        stdout,
        "Cooling load (Y2):  MAE={}  R2={}",
        fmt_fixed(mae[1], 4),
        fmt_fixed(r2[1], 4)
    )?;
    stdout.flush()
}

/// Symmetric small init in [-0.5, 0.5] raw Q16.16 units (C++ `RandomNumberGenerator`).
fn random_weight(rand: i32) -> Q16_16 {
    let half = Q16_16::ONE.raw() / 2;
    Q16_16::from_raw(rand % (2 * half + 1) - half)
}

/// C++ `toQ`: `lround(v * 2^16)`. `f64::round` rounds halves away from zero, as
/// `lround` does.
fn to_q(v: f64) -> Q16_16 {
    Q16_16::from_raw((v * 65536.0).round() as i32)
}

/// z-score, then /3 so typical values sit inside Q16.16's stable range.
fn to_input(s: &Sample, stats: &Stats<8>) -> [Q16_16; NUMBER_OF_INPUTS] {
    core::array::from_fn(|f| to_q((s.feat[f] - stats.mean[f]) / stats.stdev[f] / 3.0))
}

/// z-score, then /TARGET_SCALE.
fn to_target(s: &Sample, stats: &Stats<2>) -> [Q16_16; NUMBER_OF_OUTPUTS] {
    core::array::from_fn(|t| to_q((s.load[t] - stats.mean[t]) / stats.stdev[t] / TARGET_SCALE))
}

/// A network output back in real load units.
fn from_target(z: f64, t: usize, stats: &Stats<2>) -> f64 {
    z * TARGET_SCALE * stats.stdev[t] + stats.mean[t]
}

/// Mean and population standard deviation per column, summed in the C++ order.
fn fit_stats<const N: usize>(train: &[Sample], column: impl Fn(&Sample) -> [f64; N]) -> Stats<N> {
    let mut mean = [0.0; N];
    let mut stdev = [0.0; N];
    for s in train {
        for (m, x) in mean.iter_mut().zip(column(s)) {
            *m += x;
        }
    }
    let n = train.len() as f64;
    for m in &mut mean {
        *m /= n;
    }
    for s in train {
        for ((sd, x), m) in stdev.iter_mut().zip(column(s)).zip(mean) {
            let d = x - m;
            *sd += d * d;
        }
    }
    for sd in &mut stdev {
        *sd = (*sd / n).sqrt();
        if *sd < 1e-6 {
            *sd = 1.0;
        }
    }
    Stats { mean, stdev }
}

/// C++ `loadCsv`: skip the header; X1..X8, Y1, Y2 per line (the file has two
/// empty trailing columns); blank lines, short lines and lines with an empty
/// first cell are skipped. `None` if the file cannot be read, a number does not
/// parse, or no row is usable.
fn load_csv(path: &Path) -> Option<Vec<Sample>> {
    let text = fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() < 10 || cells[0].is_empty() {
            continue;
        }
        let mut feat = [0.0; 8];
        for (f, cell) in feat.iter_mut().zip(&cells) {
            *f = cell.trim().parse().ok()?;
        }
        let load = [cells[8].trim().parse().ok()?, cells[9].trim().parse().ok()?];
        out.push(Sample { feat, load });
    }
    (!out.is_empty()).then_some(out)
}
