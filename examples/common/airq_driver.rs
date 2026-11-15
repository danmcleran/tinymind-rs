//! The program behind `examples/air_quality.rs` (and `tests/airq_golden.rs`):
//! tinymind (C++)'s `examples/air_quality/air_quality.cpp`, step for step.
//!
//! An LSTM (1 input -> 16 cells (tanh) -> 1 sigmoid output, Q16.16, gradients
//! clipped to [-1, 1]) learns to predict the next hour's pollutant
//! concentration from the current hour. The series is the UCI Air Quality
//! dataset's CO(GT) column when `AirQualityUCI.csv` is given, otherwise a
//! synthetic hourly series (two rush-hour peaks a day, a slow weekly trend and
//! noise from glibc `rand()` seeded with 7). Training resets the recurrent
//! state every 24 hours and runs over everything but the last 200 hours, which
//! are then forecast one step ahead.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::rand::GlibcRand;
use tinymind::recurrent::LstmNetwork;
use tinymind::{Q16_16, Sigmoid, Tanh};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::fmt_g;

/// C++ `TRAINING_EPOCHS`, the default upper bound on the number of epochs.
pub const TRAINING_EPOCHS: u32 = 15_000;
/// C++ `LEARNING_RATE`, the default learning rate.
pub const LEARNING_RATE: f64 = 0.5;
/// 100 days of hourly samples (synthetic series).
const NUM_SAMPLES: usize = 2400;
const HOURS_PER_DAY: usize = 24;
/// The last 200 hours are held out for the one-step forecast.
const HOLDOUT_HOURS: usize = 200;
/// The recurrent state is reset every day during training.
const TRAIN_WINDOW: usize = 24;
const WEIGHT_INIT_SEED: u32 = 1;
const RANDOM_SEED: u32 = 7;
/// The series is normalised into [0.1, 0.9], inside the sigmoid's responsive
/// band, rather than [0, 1].
const NORM_LO: f64 = 0.1;
const NORM_SPAN: f64 = 0.8;

type Network = LstmNetwork<Q16_16, Tanh, Sigmoid, 1, 16, 1>;

/// Train for `epochs` epochs at `learning_rate` on the CO(GT) column of
/// `data_path` (a UCI `AirQualityUCI.csv`) or, if it cannot be read or has
/// fewer than 300 values, on the synthetic series; write `airq_loss.csv` and
/// `airq_forecast.csv` into `out_dir` and the log to `stdout`.
pub fn run(
    out_dir: &Path,
    data_path: &Path,
    epochs: u32,
    learning_rate: f64,
    stdout: &mut dyn Write,
) -> io::Result<()> {
    // The C++ seeds rand() with WEIGHT_INIT_SEED, builds the network (drawing
    // its weights), then re-seeds with 7 for the synthetic series. Two
    // generators here.
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let half = Q16_16::ONE.raw() / 2;
    let mut nn = Network::new(|| Q16_16::from_raw(init_rng.next_i32() % (2 * half + 1) - half));
    // GradientClipByValue<ValueType>: [-1, 1]. Momentum and acceleration keep
    // tinymind (C++)'s defaults.
    nn.set_gradient_clip(Some((Q16_16::from_int(-1), Q16_16::ONE)));
    let mut rng = GlibcRand::new(RANDOM_SEED);
    nn.set_learning_rate(to_q(learning_rate));

    let raw = match load_real_series(data_path) {
        Some(raw) => {
            writeln!(
                stdout,
                "Loaded real AirQualityUCI.csv ({} valid CO(GT) samples).",
                raw.len()
            )?;
            raw
        }
        None => {
            let raw = synth_series(&mut rng);
            writeln!(
                stdout,
                "Using synthetic hourly air-quality series ({} samples).",
                raw.len()
            )?;
            raw
        }
    };
    let num_samples = raw.len();

    let mut data_min = raw[0];
    let mut data_max = raw[0];
    for &v in &raw {
        if v < data_min {
            data_min = v;
        }
        if v > data_max {
            data_max = v;
        }
    }
    let range = if data_max > data_min {
        data_max - data_min
    } else {
        1.0
    };
    let norm: Vec<f64> = raw
        .iter()
        .map(|&v| NORM_LO + NORM_SPAN * (v - data_min) / range)
        .collect();
    let series: Vec<Q16_16> = norm.iter().map(|&n| to_q(n)).collect();

    let holdout = if num_samples > HOLDOUT_HOURS + 1 {
        HOLDOUT_HOURS
    } else {
        num_samples / 4
    };
    let train_end = num_samples - holdout;

    writeln!(
        stdout,
        "Training LSTM forecaster: up to {epochs} epochs over {} sequential pairs (holdout={holdout} hours)...",
        train_end - 1
    )?;

    let mut loss_csv = BufWriter::new(File::create(out_dir.join("airq_loss.csv"))?);
    writeln!(loss_csv, "epoch,avg_err")?;
    // Training stops at the first epoch in which no sample trains: every
    // error is inside the zero-tolerance band (1/128 of full scale), so the
    // weights did not move. The state is reset at the start of every epoch and
    // nothing else changes between epochs, so every later epoch would repeat
    // that one exactly; `epochs` is only an upper bound.
    let mut epochs_run = 0;
    for epoch in 0..epochs {
        let mut abs_err = 0.0;
        let mut count = 0usize;
        let mut trained = 0usize;
        nn.reset_state();
        for t in 0..train_end - 1 {
            if t > 0 && t % TRAIN_WINDOW == 0 {
                nn.reset_state();
            }
            let target = [series[t + 1]];
            nn.feed_forward(&[series[t]]);
            let error = nn.calculate_error(&target);
            abs_err += from_q(error).abs();
            count += 1;
            if !error.is_within_zero_tolerance() {
                nn.train(&target);
                trained += 1;
            }
        }
        let avg = if count > 0 {
            abs_err / count as f64
        } else {
            0.0
        };
        writeln!(loss_csv, "{epoch},{}", fmt_g(avg))?;
        epochs_run = epoch + 1;
        if trained == 0 {
            break;
        }
    }
    loss_csv.flush()?;
    write!(stdout, "Training complete after {epochs_run} epochs")?;
    if epochs_run < epochs {
        write!(
            stdout,
            " (every error inside the zero-tolerance band; further epochs would not change the weights)"
        )?;
    }
    writeln!(stdout, ".")?;

    // One-step-ahead forecast over the held-out tail: warm the state on the
    // day before it, then predict each held-out hour from the true current one.
    nn.reset_state();
    let warm_start = train_end.saturating_sub(TRAIN_WINDOW);
    for &x in &series[warm_start..train_end - 1] {
        nn.feed_forward(&[x]);
    }
    let mut fc = BufWriter::new(File::create(out_dir.join("airq_forecast.csv"))?);
    writeln!(fc, "hour,true,predicted")?;
    let mut sum_abs_err = 0.0;
    let mut eval_count = 0usize;
    for t in train_end - 1..num_samples - 1 {
        nn.feed_forward(&[series[t]]);
        let pred_norm = from_q(nn.learned_values()[0]);
        let true_norm = norm[t + 1];
        let pred_real = data_min + ((pred_norm - NORM_LO) / NORM_SPAN) * range;
        let true_real = data_min + ((true_norm - NORM_LO) / NORM_SPAN) * range;
        writeln!(fc, "{},{},{}", t + 1, fmt_g(true_real), fmt_g(pred_real))?;
        sum_abs_err += (pred_real - true_real).abs();
        eval_count += 1;
    }
    fc.flush()?;

    let mae = if eval_count > 0 {
        sum_abs_err / eval_count as f64
    } else {
        0.0
    };
    writeln!(
        stdout,
        "One-step-ahead forecast over {eval_count} held-out hours. MAE = {} (real units; series range {}..{}).",
        fmt_g(mae),
        fmt_g(data_min),
        fmt_g(data_max)
    )?;
    writeln!(stdout, "Wrote airq_loss.csv and airq_forecast.csv.")?;
    stdout.flush()
}

/// C++ `toQ`: `v * 2^16`, truncated toward zero.
fn to_q(v: f64) -> Q16_16 {
    Q16_16::from_raw((v * 65536.0) as i32)
}

fn from_q(v: Q16_16) -> f64 {
    f64::from(v.raw()) / 65536.0
}

/// C++ `synthSeries`: a daily baseline with morning and evening rush-hour
/// peaks, a weekly trend and mild noise, one value per hour.
fn synth_series(rng: &mut GlibcRand) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..NUM_SAMPLES)
        .map(|t| {
            let hour = (t % HOURS_PER_DAY) as f64;
            let phase = 2.0 * PI * hour / HOURS_PER_DAY as f64;
            // gcc folds `std::pow(x, 2.0)` into `x * x`, so square directly.
            let morning = (hour - 8.0) / 2.0;
            let evening = (hour - 18.0) / 2.2;
            let daily = 1.6
                + 0.9 * (-0.5 * (morning * morning)).exp()
                + 1.1 * (-0.5 * (evening * evening)).exp()
                + 0.3 * phase.sin();
            let days = t as f64 / HOURS_PER_DAY as f64;
            let trend = 0.25 * (2.0 * PI * days / 7.0).sin();
            let noise = 0.04 * (f64::from(rng.next_i32()) / f64::from(i32::MAX) - 0.5);
            daily + trend + noise
        })
        .collect()
}

/// C++ `loadRealSeries`: the UCI file is `;`-separated with decimal commas and
/// marks missing values with -200. Reads the CO(GT) column (the third field),
/// skipping the header, rows without it and missing values. `None` if the file
/// cannot be read or has fewer than 300 values.
fn load_real_series(path: &Path) -> Option<Vec<f64>> {
    let bytes = fs::read(path).ok()?;
    let mut out = Vec::new();
    for line in bytes.split(|&b| b == b'\n').skip(1) {
        // Decimal commas become dots, `\r` is dropped, `;` separates fields.
        let line: String = line
            .iter()
            .filter(|&&b| b != b'\r')
            .map(|&b| if b == b',' { '.' } else { char::from(b) })
            .collect();
        let Some(field) = line.split(';').nth(2) else {
            continue;
        };
        let Some(value) = parse_leading_f64(field) else {
            continue;
        };
        if value <= -200.0 {
            continue;
        }
        out.push(value);
    }
    (out.len() >= 300).then_some(out)
}

/// `std::istringstream >> double`: skip leading whitespace, then read the
/// longest prefix that is a decimal number. `None` if there is none.
fn parse_leading_f64(s: &str) -> Option<f64> {
    let s = s.trim_start();
    let b = s.as_bytes();
    let digits = |mut i: usize| {
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        i
    };
    let mut end = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let int_end = digits(end);
    let mut mantissa_digits = int_end - end;
    end = int_end;
    if b.get(end) == Some(&b'.') {
        let frac_end = digits(end + 1);
        mantissa_digits += frac_end - end - 1;
        end = frac_end;
    }
    if mantissa_digits == 0 {
        return None;
    }
    if matches!(b.get(end), Some(b'e' | b'E')) {
        let mut exp = end + 1;
        if matches!(b.get(exp), Some(b'+' | b'-')) {
            exp += 1;
        }
        let exp_end = digits(exp);
        if exp_end > exp {
            end = exp_end;
        }
    }
    s[..end].parse().ok()
}
