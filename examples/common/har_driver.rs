//! The program behind `examples/har_activity.rs` (and `tests/har_golden.rs`):
//! tinymind (C++)'s `examples/har_activity/har_activity.cpp`, step for step.
//!
//! An LSTM reads a window of 32 tri-axial accelerometer samples one step at a
//! time and classifies the window into one of four activities at the last
//! step: walking, walking upstairs, sitting, standing. The windows are
//! synthesized (sines with noise from glibc `rand()` seeded with 7), unless a
//! `har.csv` file in the C++ example's long format is given. Network: 3 inputs
//! -> 16 LSTM cells (tanh) -> 4 sigmoid outputs, Q16.16, gradients clipped to
//! [-1, 1], trained toward the window's label at every step for 45 epochs.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use tinymind::rand::GlibcRand;
use tinymind::recurrent::LstmNetwork;
use tinymind::{Q16_16, Sigmoid, Tanh};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::{fmt_fixed, fmt_g};

pub const WINDOW_LENGTH: usize = 32;
const NUMBER_OF_OUTPUTS: usize = 4;
const TRAIN_PER_CLASS: usize = 200;
const TEST_PER_CLASS: usize = 50;
const EPOCHS: usize = 45;
const WEIGHT_INIT_SEED: u32 = 1;
const RANDOM_SEED: u32 = 7;

pub const CLASS_NAMES: [&str; NUMBER_OF_OUTPUTS] = ["walking", "upstairs", "sitting", "standing"];

pub type Network = LstmNetwork<Q16_16, Tanh, Sigmoid, 3, 16, NUMBER_OF_OUTPUTS>;

/// glibc's `RAND_MAX`.
const RAND_MAX: f64 = 2_147_483_647.0;

/// One window: `WINDOW_LENGTH` samples of (ax, ay, az) and its activity.
#[derive(Clone, Copy)]
pub struct Window {
    pub ax: [f64; WINDOW_LENGTH],
    pub ay: [f64; WINDOW_LENGTH],
    pub az: [f64; WINDOW_LENGTH],
    pub label: i32,
}

impl Window {
    fn zeroed() -> Self {
        Window {
            ax: [0.0; WINDOW_LENGTH],
            ay: [0.0; WINDOW_LENGTH],
            az: [0.0; WINDOW_LENGTH],
            label: 0,
        }
    }
}

/// The C++ `noise(amp)`: uniform in [-amp, amp] from one `rand()`.
fn noise(rng: &mut GlibcRand, amp: f64) -> f64 {
    amp * ((2.0 * f64::from(rng.next_i32()) / RAND_MAX) - 1.0)
}

/// The C++ `synthesizeWindow`, operation for operation.
pub fn synthesize_window(rng: &mut GlibcRand, label: i32) -> Window {
    const GSCALE: f64 = 1.5;
    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    let n = WINDOW_LENGTH as f64;
    let phase = noise(rng, TWO_PI);
    let mut w = Window::zeroed();
    for t in 0..WINDOW_LENGTH {
        let tt = t as f64;
        let (mut ax, mut ay, mut az) = (0.0, 0.0, 0.0);
        match label {
            0 => {
                let f = TWO_PI * 2.0 / n * 2.0;
                ax = 1.0 + 0.35 * (f * tt + phase).sin() + 0.12 * (2.0 * f * tt + phase).sin();
                ay = 0.30 * (f * tt + phase + 1.0).sin() + 0.10 * (2.0 * f * tt + phase).sin();
                az = 0.30 * (f * tt + phase).cos() + 0.10 * (2.0 * f * tt + phase).cos();
                ax += noise(rng, 0.05);
                ay += noise(rng, 0.05);
                az += noise(rng, 0.05);
            }
            1 => {
                let f = TWO_PI * 1.0 / n * 2.0;
                ax = 0.55 + 0.25 * (f * tt + phase).sin();
                ay = 0.30 * (f * tt + phase + 0.5).sin();
                az = 0.55 + 0.70 * (f * tt + phase).sin() + 0.20 * (2.0 * f * tt + phase).sin();
                ax += noise(rng, 0.06);
                ay += noise(rng, 0.06);
                az += noise(rng, 0.06);
            }
            2 => {
                ax = 0.0 + noise(rng, 0.04);
                ay = 1.0 + noise(rng, 0.04);
                az = 0.0 + noise(rng, 0.04);
            }
            3 => {
                ax = 0.0 + noise(rng, 0.04);
                ay = 0.0 + noise(rng, 0.04);
                az = 1.0 + noise(rng, 0.04);
            }
            _ => {}
        }
        w.ax[t] = ax / GSCALE;
        w.ay[t] = ay / GSCALE;
        w.az[t] = az / GSCALE;
    }
    w.label = label;
    w
}

/// The C++ `loadCsv`: rows `window,t,ax,ay,az,label` after a header line,
/// grouped into windows by consecutive window ids. Returns `None` if the file
/// cannot be opened or holds no window. Two differences, both where the C++
/// is undefined or aborts: a sample missing from the first window is 0 (the
/// C++ reads an uninitialized value), and a cell that does not parse skips its
/// row (`std::stoi` / `std::stod` throw and end the C++ program).
pub fn load_csv(path: &Path) -> Option<Vec<Window>> {
    let file = File::open(path).ok()?;
    let mut windows = Vec::new();
    let mut current_id = None;
    let mut cur = Window::zeroed();
    let mut header = true;
    for line in BufReader::new(file).lines() {
        let line = line.ok()?;
        if line.is_empty() {
            continue;
        }
        if header {
            header = false;
            continue;
        }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() < 6 {
            continue;
        }
        let parsed = (|| {
            Some((
                cells[0].trim().parse::<i32>().ok()?,
                cells[1].trim().parse::<usize>().ok()?,
                cells[2].trim().parse::<f64>().ok()?,
                cells[3].trim().parse::<f64>().ok()?,
                cells[4].trim().parse::<f64>().ok()?,
                cells[5].trim().parse::<i32>().ok()?,
            ))
        })();
        let Some((id, t, ax, ay, az, label)) = parsed else {
            continue;
        };
        if t >= WINDOW_LENGTH {
            continue;
        }
        if current_id != Some(id) {
            if current_id.is_some() {
                windows.push(cur);
            }
            current_id = Some(id);
        }
        cur.ax[t] = ax;
        cur.ay[t] = ay;
        cur.az[t] = az;
        cur.label = label;
    }
    if current_id.is_some() {
        windows.push(cur);
    }
    if windows.is_empty() {
        None
    } else {
        Some(windows)
    }
}

/// The C++ `toQ`: `lround(v * 2^16)`.
fn to_q(v: f64) -> Q16_16 {
    Q16_16::from_raw((v * 65536.0).round() as i32)
}

fn from_q(q: Q16_16) -> f64 {
    f64::from(q.raw()) / 65536.0
}

/// The C++ `processWindow`: reset the state, feed the window one step at a
/// time (training toward the label at every step when `train`), add the last
/// step's |error| to `err_accum` and return the arg-max class at the last step.
pub fn process_window(nn: &mut Network, w: &Window, train: bool, err_accum: &mut f64) -> usize {
    let target: [Q16_16; NUMBER_OF_OUTPUTS] =
        core::array::from_fn(|k| to_q(if w.label == k as i32 { 1.0 } else { 0.0 }));
    nn.reset_state();
    for t in 0..WINDOW_LENGTH {
        nn.feed_forward(&[to_q(w.ax[t]), to_q(w.ay[t]), to_q(w.az[t])]);
        let err = nn.calculate_error(&target);
        if t == WINDOW_LENGTH - 1 {
            *err_accum += from_q(err).abs();
        }
        if train && !err.is_within_zero_tolerance() {
            nn.train(&target);
        }
    }
    let learned = nn.learned_values();
    let mut pred = 0;
    let mut best = from_q(learned[0]);
    for (k, &v) in learned.iter().enumerate().skip(1) {
        let v = from_q(v);
        if v > best {
            best = v;
            pred = k;
        }
    }
    pred
}

/// The network with the C++ example's initial weights: `rand()` seeded with
/// `WEIGHT_INIT_SEED`, each weight symmetric in [-0.25, 0.25].
pub fn new_network() -> Network {
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let quarter = 1 << 14;
    let mut nn =
        Network::new(|| Q16_16::from_raw(init_rng.next_i32() % (2 * quarter + 1) - quarter));
    // GradientClipByValue<ValueType>: [-1, 1].
    nn.set_gradient_clip(Some((Q16_16::from_int(-1), Q16_16::ONE)));
    nn
}

/// Fisher-Yates with `rand() % i`, as the C++ shuffles.
fn shuffle(rng: &mut GlibcRand, v: &mut [Window]) {
    for i in (2..=v.len()).rev() {
        let j = rng.next_i32() as usize % i;
        v.swap(i - 1, j);
    }
}

/// Runs the example: writes `har_samples.csv`, `har_loss.csv` and
/// `har_confusion.csv` to `out_dir` and the log to `stdout`. `csv` is the
/// optional real-data file (the C++ example reads `har.csv`).
pub fn run(out_dir: &Path, csv: &Path, stdout: &mut dyn Write) -> io::Result<()> {
    let mut nn = new_network();
    let mut rng = GlibcRand::new(RANDOM_SEED);

    let mut train = Vec::new();
    let mut test = Vec::new();
    if let Some(mut loaded) = load_csv(csv) {
        writeln!(stdout, "Loaded {} windows from har.csv.", loaded.len())?;
        shuffle(&mut rng, &mut loaded);
        let n_train = (loaded.len() * 4) / 5;
        test = loaded.split_off(n_train);
        train = loaded;
    } else {
        writeln!(
            stdout,
            "har.csv not found -- synthesizing accelerometer windows."
        )?;
        for label in 0..NUMBER_OF_OUTPUTS as i32 {
            for _ in 0..TRAIN_PER_CLASS {
                train.push(synthesize_window(&mut rng, label));
            }
            for _ in 0..TEST_PER_CLASS {
                test.push(synthesize_window(&mut rng, label));
            }
        }
    }
    writeln!(stdout, "Train: {}  Test: {}", train.len(), test.len())?;

    {
        let mut samples = BufWriter::new(fs::File::create(out_dir.join("har_samples.csv"))?);
        writeln!(samples, "class,t,ax,ay,az")?;
        for (label, name) in CLASS_NAMES.iter().enumerate() {
            let w = synthesize_window(&mut rng, label as i32);
            for t in 0..WINDOW_LENGTH {
                writeln!(
                    samples,
                    "{name},{t},{},{},{}",
                    fmt_g(w.ax[t]),
                    fmt_g(w.ay[t]),
                    fmt_g(w.az[t])
                )?;
            }
        }
        samples.flush()?;
    }

    let mut loss = BufWriter::new(fs::File::create(out_dir.join("har_loss.csv"))?);
    writeln!(loss, "epoch,avg_err")?;
    for epoch in 0..EPOCHS {
        shuffle(&mut rng, &mut train);
        let mut err_sum = 0.0;
        for w in &train {
            process_window(&mut nn, w, true, &mut err_sum);
        }
        let avg = if train.is_empty() {
            0.0
        } else {
            err_sum / train.len() as f64
        };
        writeln!(
            stdout,
            "epoch {:>3}   avg|err| = {}",
            epoch + 1,
            fmt_fixed(avg, 4)
        )?;
        writeln!(loss, "{},{}", epoch + 1, fmt_g(avg))?;
    }
    loss.flush()?;

    let mut confusion = [[0u32; NUMBER_OF_OUTPUTS]; NUMBER_OF_OUTPUTS];
    let mut eval_err = 0.0;
    let mut correct = 0usize;
    for w in &test {
        let pred = process_window(&mut nn, w, false, &mut eval_err);
        if let Some(row) = usize::try_from(w.label)
            .ok()
            .and_then(|l| confusion.get_mut(l))
        {
            row[pred] += 1;
        }
        if pred as i32 == w.label {
            correct += 1;
        }
    }

    {
        let mut cm = BufWriter::new(fs::File::create(out_dir.join("har_confusion.csv"))?);
        writeln!(
            cm,
            "actual,pred_walk,pred_upstairs,pred_sitting,pred_standing"
        )?;
        for (name, row) in CLASS_NAMES.iter().zip(&confusion) {
            write!(cm, "{name}")?;
            for c in row {
                write!(cm, ",{c}")?;
            }
            writeln!(cm)?;
        }
        cm.flush()?;
    }

    let acc = correct as f64 / test.len() as f64;
    writeln!(
        stdout,
        "\nTest accuracy: {}  ({}/{})",
        fmt_fixed(acc, 4),
        correct,
        test.len()
    )?;
    stdout.flush()
}
