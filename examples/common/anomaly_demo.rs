//! On-device anomaly detection with a dense autoencoder: the program shared by
//! `examples/anomaly.rs` (host) and the bench firmware (`bench/`, on the
//! boards), which run it bit for bit identically. `no_std`, integer only.
//!
//! A machine's vibration is simulated as a sine tone with random phase and
//! amplitude plus noise, sampled in windows of 8. An 8-4-8 autoencoder learns to
//! reconstruct *normal* windows only, on the device; a window whose
//! reconstruction error is far above anything seen on normal data is flagged.
//! Four kinds of fault are then injected: the tone at the wrong frequency, an
//! added harmonic, an impulse spike, and a stalled sensor.
//!
//! The signal is synthetic, so the demo needs no dataset and the same program
//! runs on a 32 KB part. Everything is deterministic (glibc `rand()`
//! re-implementation, integer sine table), so the host and the boards produce
//! the same report and the same hash.

use core::fmt::{self, Write};

use tinymind::rand::GlibcRand;
use tinymind::{Dense, Linear, NeuralNetwork, Q16_16, Tanh};

/// Samples per window, the autoencoder's input and output width.
pub const WINDOW: usize = 8;
/// Bottleneck width.
pub const CODE: usize = 4;
/// Training steps, each on a fresh normal window.
pub const TRAIN_STEPS: u32 = 3_000;
/// Normal windows used to set the threshold, and normal test windows.
pub const NORMAL_WINDOWS: u32 = 256;
/// Test windows per fault type.
pub const FAULT_WINDOWS: u32 = 64;

/// The autoencoder: 8 inputs -> 4 tanh -> 8 linear outputs.
pub type Autoencoder = NeuralNetwork<(
    Dense<Q16_16, Tanh, WINDOW, CODE>,
    Dense<Q16_16, Linear, CODE, WINDOW>,
)>;

/// One period of a sine in Q16.16 (sin(2 pi k / 64) * 65536, rounded).
const SINE: [i32; 64] = [
    0, 6424, 12785, 19024, 25080, 30893, 36410, 41576, 46341, 50660, 54491, 57798, 60547, 62714,
    64277, 65220, 65536, 65220, 64277, 62714, 60547, 57798, 54491, 50660, 46341, 41576, 36410,
    30893, 25080, 19024, 12785, 6424, 0, -6424, -12785, -19024, -25080, -30893, -36410, -41576,
    -46341, -50660, -54491, -57798, -60547, -62714, -64277, -65220, -65536, -65220, -64277, -62714,
    -60547, -57798, -54491, -50660, -46341, -41576, -36410, -30893, -25080, -19024, -12785, -6424,
];

/// The kinds of window the demo generates.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Normal,
    /// The tone at a different frequency.
    Frequency,
    /// The normal tone plus its third harmonic.
    Harmonic,
    /// The normal tone plus a single large impulse.
    Spike,
    /// The sensor stalls: the window holds one value.
    Stall,
}

pub const FAULTS: [(Kind, &str); 4] = [
    (Kind::Frequency, "frequency"),
    (Kind::Harmonic, "harmonic"),
    (Kind::Spike, "spike"),
    (Kind::Stall, "stall"),
];

/// Deterministic signal source.
pub struct Signal(GlibcRand);

impl Signal {
    pub fn new(seed: u32) -> Self {
        Self(GlibcRand::new(seed))
    }

    /// A uniform integer in 0..n.
    fn below(&mut self, n: u32) -> i32 {
        (self.0.next_i32() as u32 % n) as i32
    }

    /// One window of the given kind, in Q16.16, within about [-1, 1].
    pub fn window(&mut self, kind: Kind) -> [Q16_16; WINDOW] {
        // Normal: step 3 through the table (period 21.3 samples), amplitude
        // 0.4 to 0.6, any phase, noise within +-0.02.
        let step = if kind == Kind::Frequency { 9 } else { 3 };
        let amplitude = 26_214 + self.below(13_108); // 0.4 + [0, 0.2)
        let phase = self.below(64);
        let mut w = [Q16_16::ZERO; WINDOW];
        for (n, s) in w.iter_mut().enumerate() {
            let k = (phase + n as i32 * step) as usize % 64;
            let mut v = Q16_16::from_raw(amplitude) * Q16_16::from_raw(SINE[k]);
            if kind == Kind::Harmonic {
                let k3 = (phase + 3 * n as i32 * step) as usize % 64;
                v += Q16_16::from_raw(19_661) * Q16_16::from_raw(SINE[k3]); // + 0.3 sin(3x)
            }
            let noise = self.below(2_623) - 1_311; // +-0.02
            *s = v + Q16_16::from_raw(noise);
        }
        match kind {
            Kind::Spike => {
                let at = self.below(WINDOW as u32) as usize;
                w[at] += Q16_16::from_raw(52_429); // + 0.8
            }
            Kind::Stall => {
                let held = w[0];
                w = [held; WINDOW];
            }
            _ => {}
        }
        w
    }
}

/// The reconstruction error of one window: tinymind's mean squared error of
/// the output against the input.
#[inline(never)]
pub fn score(nn: &mut Autoencoder, w: &[Q16_16; WINDOW]) -> Q16_16 {
    nn.feed_forward(w);
    nn.calculate_error(w)
}

/// Initial weights in [-0.5, 0.5], as tinymind's examples draw them.
pub fn new_autoencoder(seed: u32) -> Autoencoder {
    let mut rng = GlibcRand::new(seed);
    let half = Q16_16::ONE.raw() / 2;
    let mut nn = Autoencoder::new(|| Q16_16::from_raw(rng.next_i32() % (2 * half + 1) - half));
    nn.set_learning_rate(Q16_16::from_raw(1 << 11)); // 1/32
    nn.set_momentum_rate(Q16_16::from_raw(1 << 15)); // 1/2
    nn.set_acceleration_rate(Q16_16::ZERO);
    nn.set_gradient_clip(Some((Q16_16::from_int(-1), Q16_16::ONE)));
    nn
}

/// One training step on window `w` (forward pass, error, back-propagation);
/// returns the error before the step. Out of line, so the training loop and the
/// board's timing loop share one copy (the 32 KB board has no room for two).
#[inline(never)]
pub fn train_step(nn: &mut Autoencoder, w: &[Q16_16; WINDOW]) -> Q16_16 {
    let error = score(nn, w);
    nn.train(w);
    error
}

/// Train on `steps` normal windows; returns the mean error (raw) over the last
/// 500 steps.
#[inline(never)]
pub fn train(nn: &mut Autoencoder, signal: &mut Signal, steps: u32) -> i32 {
    // Errors at the end of training are a few hundred raw units at most, so
    // the sum of 500 fits easily; saturate rather than wrap if it ever does not.
    let mut recent = 0i32;
    for step in 0..steps {
        let w = signal.window(Kind::Normal);
        let error = train_step(nn, &w);
        if step + 500 >= steps {
            recent = recent.saturating_add(error.raw());
        }
    }
    recent / 500
}

/// 32-bit FNV-1a over everything written, to compare host and board runs.
pub struct Fnv(pub u32);

impl Fnv {
    pub fn new() -> Self {
        Self(0x811C_9DC5)
    }
    pub fn add(&mut self, raw: i32) {
        for b in raw.to_le_bytes() {
            self.0 ^= u32::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0193);
        }
    }
}

/// Everything after training: set the threshold from normal windows, test
/// normal and faulty windows, and print the report. `each` sees every test
/// window's kind name and score (raw), for a plot. Returns the hash of the
/// trained weights and every score.
#[inline(never)]
pub fn evaluate(
    nn: &mut Autoencoder,
    signal: &mut Signal,
    out: &mut dyn Write,
    each: &mut dyn FnMut(&str, i32),
) -> Result<u32, fmt::Error> {
    let mut hash = Fnv::new();
    let (l0, l1) = nn.layers();
    for row in l0.weights() {
        for w in row {
            hash.add(w.raw());
        }
    }
    for w in l0.bias_weights() {
        hash.add(w.raw());
    }
    for row in l1.weights() {
        for w in row {
            hash.add(w.raw());
        }
    }
    for w in l1.bias_weights() {
        hash.add(w.raw());
    }

    // Threshold: twice the largest error on normal validation windows.
    let mut max_normal = 0;
    for _ in 0..NORMAL_WINDOWS {
        let e = score(nn, &signal.window(Kind::Normal)).raw();
        hash.add(e);
        max_normal = max_normal.max(e);
    }
    let threshold = max_normal.saturating_mul(2);
    writeln!(
        out,
        "threshold={threshold} (2 x the largest error on {NORMAL_WINDOWS} normal windows, raw Q16.16)"
    )?;

    let mut flagged = |kind: Kind, name: &str, n: u32, hash: &mut Fnv| {
        let mut count = 0;
        for _ in 0..n {
            let e = score(nn, &signal.window(kind)).raw();
            hash.add(e);
            each(name, e);
            count += u32::from(e > threshold);
        }
        count
    };
    let fp = flagged(Kind::Normal, "normal", NORMAL_WINDOWS, &mut hash);
    writeln!(out, "normal: flagged {fp}/{NORMAL_WINDOWS} (false alarms)")?;
    for (kind, name) in FAULTS {
        let tp = flagged(kind, name, FAULT_WINDOWS, &mut hash);
        writeln!(out, "{name}: flagged {tp}/{FAULT_WINDOWS}")?;
    }
    writeln!(out, "hash=0x{:08X}", hash.0)?;
    Ok(hash.0)
}

/// The whole demo: train on normal windows, then evaluate. Seeds are fixed.
/// (The board firmware calls the pieces itself, to time them.)
#[allow(dead_code)]
pub fn run(out: &mut dyn Write, each: &mut dyn FnMut(&str, i32)) -> Result<u32, fmt::Error> {
    let mut nn = new_autoencoder(1);
    let mut signal = Signal::new(7);
    writeln!(
        out,
        "autoencoder {WINDOW}-{CODE}-{WINDOW} (tanh, linear), Q16.16, {TRAIN_STEPS} training steps on normal windows"
    )?;
    let recent = train(&mut nn, &mut signal, TRAIN_STEPS);
    writeln!(
        out,
        "training error, mean of the last 500 steps={recent} (raw Q16.16)"
    )?;
    evaluate(&mut nn, &mut signal, out, each)
}
