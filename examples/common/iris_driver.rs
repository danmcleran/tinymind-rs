//! The iris example's program, shared by `examples/iris.rs` (which parses the
//! command line) and `tests/iris_golden.rs` (which runs it in-process against the
//! C++ goldens). Port of tinymind (C++) `examples/iris/iris.cpp`; see `examples/iris.rs`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::rand::GlibcRand;
use tinymind::{MultilayerPerceptron, Q16_16, Relu, Sigmoid};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::{fmt_fixed, fmt_g};

const NUMBER_OF_INPUTS: usize = 4;
const NUMBER_OF_NEURONS_PER_HIDDEN_LAYER: usize = 8;
const NUMBER_OF_OUTPUTS: usize = 3;

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
/// Seeds the shuffle and the training-sample draws.
const RANDOM_SEED: u32 = 7;
const ITERATIONS: u32 = 30_000;
const REPORT_EVERY: u32 = 1_000;
const CLASS_NAMES: [&str; NUMBER_OF_OUTPUTS] = ["setosa", "versicolor", "virginica"];

#[derive(Clone, Copy)]
struct Sample {
    /// Sepal length and width, petal length and width (cm).
    feat: [f64; 4],
    /// 0 = setosa, 1 = versicolor, 2 = virginica.
    label: usize,
}

/// Per-feature z-score statistics fit on the training split.
struct FeatureStats {
    mean: [f64; 4],
    stdev: [f64; 4],
}

/// Train on `data_path`, evaluate, write the three CSV files into `out_dir` and
/// the training log and accuracy to `stdout`.
pub fn run(out_dir: &Path, data_path: &Path, stdout: &mut dyn Write) -> io::Result<()> {
    // As in the xor port: the C++ seeds rand() with WEIGHT_INIT_SEED, builds the
    // network (drawing its weights), then re-seeds with 7. Two generators here.
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let mut nn = NeuralNetwork::new(|| random_weight(init_rng.next_i32()));

    let Some(mut data) = load_csv(data_path) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("could not open or parse {}", data_path.display()),
        ));
    };
    // The C++ names the file it opens from the working directory.
    writeln!(stdout, "Loaded {} rows from iris.data.", data.len())?;

    // Deterministic shuffle + 80/20 split (the C++ re-seeds with 7 just before).
    let mut rng = GlibcRand::new(RANDOM_SEED);
    for i in (2..=data.len()).rev() {
        let j = rng.next_i32() as usize % i;
        data.swap(i - 1, j);
    }
    let n_train = data.len() * 4 / 5;
    let (train, test) = data.split_at(n_train);
    writeln!(stdout, "Train: {}  Test: {}", train.len(), test.len())?;

    let stats = fit_stats(train);

    // ---- Training ----
    let mut loss_csv = BufWriter::new(File::create(out_dir.join("iris_loss.csv"))?);
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
    let mut scatter = BufWriter::new(File::create(out_dir.join("iris_scatter.csv"))?);
    writeln!(scatter, "petal_len,petal_width,true_class,pred_class")?;
    let mut correct = 0;
    for s in test {
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
        confusion[s.label][pred] += 1;
        if pred == s.label {
            correct += 1;
        }
        writeln!(
            scatter,
            "{},{},{},{}",
            fmt_g(s.feat[2]),
            fmt_g(s.feat[3]),
            s.label,
            pred
        )?;
    }
    scatter.flush()?;

    let mut cm = BufWriter::new(File::create(out_dir.join("iris_confusion.csv"))?);
    writeln!(cm, "actual,pred_setosa,pred_versicolor,pred_virginica")?;
    for (name, row) in CLASS_NAMES.iter().zip(&confusion) {
        write!(cm, "{name}")?;
        for count in row {
            write!(cm, ",{count}")?;
        }
        writeln!(cm)?;
    }
    cm.flush()?;

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
    core::array::from_fn(|f| to_q((s.feat[f] - stats.mean[f]) / stats.stdev[f] / 3.0))
}

/// Mean and population standard deviation per feature, summed in the C++ order.
fn fit_stats(train: &[Sample]) -> FeatureStats {
    let mut mean = [0.0; 4];
    let mut stdev = [0.0; 4];
    for s in train {
        for (m, x) in mean.iter_mut().zip(s.feat) {
            *m += x;
        }
    }
    let n = train.len() as f64;
    for m in &mut mean {
        *m /= n;
    }
    for s in train {
        for ((sd, x), m) in stdev.iter_mut().zip(s.feat).zip(mean) {
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
    FeatureStats { mean, stdev }
}

/// C++ `loadCsv`: 4 numbers and a species name per line; blank lines, short lines
/// and unknown species are skipped. `None` if the file cannot be read or has no
/// usable rows.
fn load_csv(path: &Path) -> Option<Vec<Sample>> {
    let text = fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split(',').collect();
        if line.is_empty() || cells.len() < 5 {
            continue;
        }
        let mut feat = [0.0; 4];
        for (f, cell) in feat.iter_mut().zip(&cells) {
            *f = cell.trim().parse().ok()?;
        }
        let name = cells[4];
        let label = if name.contains("setosa") {
            0
        } else if name.contains("versicolor") {
            1
        } else if name.contains("virginica") {
            2
        } else {
            continue;
        };
        out.push(Sample { feat, label });
    }
    (!out.is_empty()).then_some(out)
}
