//! The optical_digits example's program, shared by `examples/optical_digits.rs`
//! (which parses the command line) and `tests/digits_golden.rs` (which runs it
//! in-process against the C++ goldens). Port of tinymind (C++)
//! `examples/optical_digits/optical_digits.cpp`; see `examples/optical_digits.rs`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::rand::GlibcRand;
use tinymind::{MultilayerPerceptron, Q16_16, Relu, Sigmoid};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::{fmt_fixed, fmt_g};

const NUMBER_OF_PIXELS: usize = 64;
const NUMBER_OF_INPUTS: usize = NUMBER_OF_PIXELS;
const NUMBER_OF_NEURONS_PER_HIDDEN_LAYER: usize = 32;
const NUMBER_OF_OUTPUTS: usize = 10;

type NeuralNetwork = MultilayerPerceptron<
    Q16_16,
    Relu,
    Sigmoid,
    NUMBER_OF_INPUTS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
>;

/// C++ `WEIGHT_INIT_SEED`: seeds the initial weights only.
const WEIGHT_INIT_SEED: u32 = 1;
/// Seeds the training-sample draws.
const RANDOM_SEED: u32 = 7;
const ITERATIONS: u32 = 60_000;
const REPORT_EVERY: u32 = 2_000;
/// Test images written to `digits_samples.csv`, at most half of them
/// misclassified.
const MAX_SAMPLES: usize = 40;

struct Sample {
    /// Pixel intensities 0..16 of the 8x8 bitmap, row-major.
    pixel: [i32; NUMBER_OF_PIXELS],
    /// The digit, 0..9.
    label: usize,
}

/// Per-pixel z-score statistics fit on the training split.
struct FeatureStats {
    mean: [f64; NUMBER_OF_PIXELS],
    stdev: [f64; NUMBER_OF_PIXELS],
}

/// Train on `optdigits.tra` and test on `optdigits.tes`, both read from
/// `data_dir`; write the three CSV files into `out_dir` and the training log and
/// accuracy to `stdout`.
pub fn run(out_dir: &Path, data_dir: &Path, stdout: &mut dyn Write) -> io::Result<()> {
    // The C++ seeds rand() with WEIGHT_INIT_SEED, builds the network (drawing
    // its weights), then re-seeds with 7. Two generators here.
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let mut nn = NeuralNetwork::new(|| random_weight(init_rng.next_i32()));
    let mut rng = GlibcRand::new(RANDOM_SEED);

    let load = |name: &str| {
        let path = data_dir.join(name);
        load_csv(&path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("could not open or parse {}", path.display()),
            )
        })
    };
    let train = load("optdigits.tra")?;
    let test = load("optdigits.tes")?;
    writeln!(
        stdout,
        "Loaded {} train rows (optdigits.tra), {} test rows (optdigits.tes).",
        train.len(),
        test.len()
    )?;

    let stats = fit_stats(&train);

    // ---- Training ----
    let mut loss_csv = BufWriter::new(File::create(out_dir.join("digits_loss.csv"))?);
    writeln!(loss_csv, "iter,avg_err")?;
    let mut err_sum = 0.0;
    for it in 0..ITERATIONS {
        let s = &train[rng.next_i32() as usize % train.len()];
        let input = to_input(s, &stats);
        let target: [Q16_16; NUMBER_OF_OUTPUTS] =
            core::array::from_fn(|k| to_q(if s.label == k { 1.0 } else { 0.0 }));

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

    // ---- Evaluation ----
    let mut confusion = [[0u32; NUMBER_OF_OUTPUTS]; NUMBER_OF_OUTPUTS];
    let mut pred_of = Vec::with_capacity(test.len());
    let mut correct = 0;
    for s in &test {
        nn.feed_forward(&to_input(s, &stats));
        let learned = nn.learned_values();
        // First index wins ties, as the C++ strict `>` scan does.
        let mut pred = 0;
        let mut best = learned[0].to_f64();
        for (k, value) in learned.iter().enumerate().skip(1) {
            let v = value.to_f64();
            if v > best {
                best = v;
                pred = k;
            }
        }
        pred_of.push(pred);
        confusion[s.label][pred] += 1;
        if pred == s.label {
            correct += 1;
        }
    }

    let mut cm = BufWriter::new(File::create(out_dir.join("digits_confusion.csv"))?);
    write!(cm, "actual")?;
    for c in 0..NUMBER_OF_OUTPUTS {
        write!(cm, ",pred_{c}")?;
    }
    writeln!(cm)?;
    for (r, row) in confusion.iter().enumerate() {
        write!(cm, "{r}")?;
        for count in row {
            write!(cm, ",{count}")?;
        }
        writeln!(cm)?;
    }
    cm.flush()?;

    // ---- Image-grid sample: misclassified images first (up to half the
    // grid, the hard cases worth looking at), then correct ones ----
    let misses = (0..test.len()).filter(|&i| pred_of[i] != test[i].label);
    let mut chosen: Vec<usize> = misses.take(MAX_SAMPLES / 2).collect();
    let hits = (0..test.len()).filter(|&i| pred_of[i] == test[i].label);
    let room = MAX_SAMPLES - chosen.len();
    chosen.extend(hits.take(room));

    let mut samples = BufWriter::new(File::create(out_dir.join("digits_samples.csv"))?);
    for p in 0..NUMBER_OF_PIXELS {
        write!(samples, "pixel{p},")?;
    }
    writeln!(samples, "true,pred")?;
    for idx in chosen {
        let s = &test[idx];
        for p in s.pixel {
            write!(samples, "{p},")?;
        }
        writeln!(samples, "{},{}", s.label, pred_of[idx])?;
    }
    samples.flush()?;

    let acc = f64::from(correct) / test.len() as f64;
    writeln!(
        stdout,
        "\nTest accuracy: {}  ({correct}/{})",
        fmt_fixed(acc, 4),
        test.len()
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
fn to_input(s: &Sample, stats: &FeatureStats) -> [Q16_16; NUMBER_OF_INPUTS] {
    core::array::from_fn(|f| to_q((f64::from(s.pixel[f]) - stats.mean[f]) / stats.stdev[f] / 3.0))
}

/// Mean and population standard deviation per pixel, summed in the C++ order.
/// Border pixels that are always 0 get a standard deviation of 1, so they map
/// to a constant 0 input instead of a division by ~0.
fn fit_stats(train: &[Sample]) -> FeatureStats {
    let mut mean = [0.0; NUMBER_OF_PIXELS];
    let mut stdev = [0.0; NUMBER_OF_PIXELS];
    for s in train {
        for (m, x) in mean.iter_mut().zip(s.pixel) {
            *m += f64::from(x);
        }
    }
    let n = train.len() as f64;
    for m in &mut mean {
        *m /= n;
    }
    for s in train {
        for ((sd, x), m) in stdev.iter_mut().zip(s.pixel).zip(mean) {
            let d = f64::from(x) - m;
            *sd += d * d;
        }
    }
    for sd in &mut stdev {
        *sd = (*sd / n).sqrt();
        if *sd < 1e-6 {
            *sd = 1.0;
        }
    }
    FeatureStats { mean, stdev }
}

/// C++ `loadCsv` (UCI format): 64 pixel values and the digit per line; blank
/// lines, short lines and labels outside 0..9 are skipped. `None` if the file
/// cannot be read, a number does not parse, or no row is usable.
fn load_csv(path: &Path) -> Option<Vec<Sample>> {
    let text = fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split(',').collect();
        if line.is_empty() || cells.len() < NUMBER_OF_PIXELS + 1 {
            continue;
        }
        let mut pixel = [0; NUMBER_OF_PIXELS];
        for (p, cell) in pixel.iter_mut().zip(&cells) {
            *p = cell.trim().parse().ok()?;
        }
        let label: i32 = cells[NUMBER_OF_PIXELS].trim().parse().ok()?;
        let Ok(label @ 0..=9) = usize::try_from(label) else {
            continue;
        };
        out.push(Sample { pixel, label });
    }
    (!out.is_empty()).then_some(out)
}
