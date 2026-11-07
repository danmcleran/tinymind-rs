//! Port of tinymind `examples/xor/xor.cpp`: train a 2-4-1 Q16.16 perceptron
//! (tanh hidden, sigmoid output) on XOR and write the same files the C++ example
//! writes, byte for byte:
//!
//! - `nn_fixed_xor.txt`: every weight, the target, the output and the error after
//!   each of the 20000 training steps (raw Q16.16 LSBs)
//! - `xor_training.csv`: the 4-case XOR error every 100 steps
//! - `xor_decision_surface.csv`: the trained network over a 41x41 input grid
//!
//! Usage: `xor [output_dir] [learning_rate] [momentum]` (default output dir: `output`).

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use rustymind::rand::GlibcRand;
use rustymind::{MultilayerPerceptron, Q16_16, Sigmoid, Tanh};

const TRAINING_ITERATIONS: u32 = 20_000;
const NUM_SAMPLES_AVG_ERROR: u32 = 100;
const RANDOM_SEED: u32 = 7;
/// The seed glibc's rand() uses when srand() has not been called.
const GLIBC_DEFAULT_SEED: u32 = 1;

const NUMBER_OF_INPUTS: usize = 2;
const NUMBER_OF_NEURONS_PER_HIDDEN_LAYER: usize = 4;
const NUMBER_OF_OUTPUTS: usize = 1;

type NeuralNetwork = MultilayerPerceptron<
    Q16_16,
    Tanh,
    Sigmoid,
    NUMBER_OF_INPUTS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
>;

const XOR_CASES: [[i32; 3]; 4] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let learning_rate = args.next().map(|s| parse_arg(&s));
    let momentum = args.next().map(|s| parse_arg(&s));
    fs::create_dir_all(&out_dir)?;

    // C++ uses one global rand() for both weight init and training samples, but
    // the network is a *global object*: its constructor (which draws the initial
    // weights) runs during static initialisation, before main() calls
    // srand(7). So the weights come from glibc's unseeded stream (seed 1) and only
    // the training samples come from seed 7. Rust has no life-before-main, so the
    // two streams are made explicit here.
    let mut init_rng = GlibcRand::new(GLIBC_DEFAULT_SEED);
    let mut nn = NeuralNetwork::new(|| generate_random_weight(&mut init_rng));
    let mut rng = GlibcRand::new(RANDOM_SEED);
    if let Some(lr) = learning_rate {
        nn.set_learning_rate(Q16_16::from_f64_trunc(lr));
    }
    if let Some(m) = momentum {
        nn.set_momentum_rate(Q16_16::from_f64_trunc(m));
    }

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    let mut results = BufWriter::new(File::create(out_dir.join("nn_fixed_xor.txt"))?);
    let mut curve = BufWriter::new(File::create(out_dir.join("xor_training.csv"))?);
    writeln!(curve, "iteration,avg_error")?;

    write_header(&mut results)?;

    let mut avg_error = Q16_16::ZERO;
    for i in 0..TRAINING_ITERATIONS {
        let (inputs, target) = generate_xor_training_value(&mut rng);

        nn.feed_forward(&inputs);
        let error = nn.calculate_error(&target);
        if !error.is_within_zero_tolerance() {
            nn.train(&target);
        }
        let learned = nn.learned_values();

        store_network_properties(&nn, &mut results, &target, &learned)?;
        writeln!(results, "{error}")?;

        // Accumulate |error| in raw LSBs; the average divides it back down.
        avg_error += Q16_16::from_raw(error.raw().wrapping_abs());
        if (i + 1) % NUM_SAMPLES_AVG_ERROR == 0 {
            avg_error = avg_error / Q16_16::from_int(NUM_SAMPLES_AVG_ERROR as i32);
            writeln!(stdout, "Iteration: {} Average Error: {}", i + 1, avg_error)?;
            writeln!(curve, "{},{}", i + 1, fmt_g(evaluate_xor_error(&mut nn)))?;
            avg_error = Q16_16::ZERO;
        }
    }

    // Success = every XOR pattern lands on the correct side of 0.5.
    let mut correct = 0;
    for [a, b, expected] in XOR_CASES {
        let pred = predict(&mut nn, Q16_16::from_int(a), Q16_16::from_int(b));
        let ok = (pred >= 0.5) == (expected == 1);
        correct += u32::from(ok);
        writeln!(
            stdout,
            "  {a} ^ {b} = {expected}  pred={}{}",
            fmt_g(pred),
            if ok { "  ok" } else { "  WRONG" }
        )?;
    }
    writeln!(
        stdout,
        "{correct}/4 patterns classified correctly. Final 4-case error: {}",
        fmt_g(evaluate_xor_error(&mut nn))
    )?;

    // Decision surface over the [0,1]^2 grid.
    let mut surf = BufWriter::new(File::create(out_dir.join("xor_decision_surface.csv"))?);
    writeln!(surf, "x0,x1,prob")?;
    const G: i32 = 41;
    for a in 0..G {
        for b in 0..G {
            let x0 = f64::from(a) / f64::from(G - 1);
            let x1 = f64::from(b) / f64::from(G - 1);
            let prob = predict(
                &mut nn,
                Q16_16::from_f64_trunc(x0),
                Q16_16::from_f64_trunc(x1),
            );
            writeln!(surf, "{},{},{}", fmt_g(x0), fmt_g(x1), fmt_g(prob))?;
        }
    }

    // Flush explicitly so a write error is reported rather than lost in Drop.
    results.flush()?;
    curve.flush()?;
    surf.flush()?;
    stdout.flush()
}

fn parse_arg(s: &str) -> f64 {
    s.parse()
        .unwrap_or_else(|_| panic!("expected a number, got {s:?}"))
}

/// Symmetric small init in [-0.5, 0.5] raw Q16.16 units (C++ `RandomNumberGenerator`).
fn generate_random_weight(rng: &mut GlibcRand) -> Q16_16 {
    let half = Q16_16::ONE.raw() / 2;
    Q16_16::from_raw(rng.next_i32() % (2 * half + 1) - half)
}

fn generate_xor_training_value(rng: &mut GlibcRand) -> ([Q16_16; 2], [Q16_16; 1]) {
    let x = rng.next_i32() & 1;
    let y = rng.next_i32() & 1;
    (
        [Q16_16::from_int(x), Q16_16::from_int(y)],
        [Q16_16::from_int(x ^ y)],
    )
}

fn predict(nn: &mut NeuralNetwork, x0: Q16_16, x1: Q16_16) -> f64 {
    nn.feed_forward(&[x0, x1]);
    nn.learned_values()[0].to_f64()
}

/// Mean |prediction - target| over the four XOR cases, in real units.
fn evaluate_xor_error(nn: &mut NeuralNetwork) -> f64 {
    let mut sum = 0.0;
    for [a, b, expected] in XOR_CASES {
        let pred = predict(nn, Q16_16::from_int(a), Q16_16::from_int(b));
        sum += (pred - f64::from(expected)).abs();
    }
    sum / 4.0
}

fn write_header(out: &mut impl Write) -> io::Result<()> {
    for i in 0..NUMBER_OF_INPUTS {
        for h in 0..NUMBER_OF_NEURONS_PER_HIDDEN_LAYER {
            write!(out, "Input{i}{h}Weight,")?;
        }
    }
    for h in 0..NUMBER_OF_NEURONS_PER_HIDDEN_LAYER {
        write!(out, "InputBias0{h}Weight,")?;
    }
    for h in 0..NUMBER_OF_NEURONS_PER_HIDDEN_LAYER {
        for o in 0..NUMBER_OF_OUTPUTS {
            write!(out, "Hidden0{h}{o}Weight,")?;
        }
    }
    for o in 0..NUMBER_OF_OUTPUTS {
        write!(out, "Hidden0Bias{o}Weight,")?;
    }
    for o in 0..NUMBER_OF_OUTPUTS {
        write!(out, "Expected{o},")?;
    }
    for o in 0..NUMBER_OF_OUTPUTS {
        write!(out, "Learned{o},")?;
    }
    writeln!(out, "Error")
}

fn store_network_properties(
    nn: &NeuralNetwork,
    out: &mut impl Write,
    expected: &[Q16_16; NUMBER_OF_OUTPUTS],
    learned: &[Q16_16; NUMBER_OF_OUTPUTS],
) -> io::Result<()> {
    for i in 0..NUMBER_OF_INPUTS {
        for h in 0..NUMBER_OF_NEURONS_PER_HIDDEN_LAYER {
            write!(out, "{},", nn.input_weight(i, h))?;
        }
    }
    for h in 0..NUMBER_OF_NEURONS_PER_HIDDEN_LAYER {
        write!(out, "{},", nn.input_bias_weight(h))?;
    }
    for h in 0..NUMBER_OF_NEURONS_PER_HIDDEN_LAYER {
        for o in 0..NUMBER_OF_OUTPUTS {
            write!(out, "{},", nn.hidden_weight(h, o))?;
        }
    }
    for o in 0..NUMBER_OF_OUTPUTS {
        write!(out, "{},", nn.hidden_bias_weight(o))?;
    }
    for value in expected.iter().chain(learned) {
        write!(out, "{value},")?;
    }
    Ok(())
}

/// Format like C++ `std::ostream << double` with default flags, which is
/// `printf("%g")` with precision 6. Rust's `{}` prints the shortest round-trip
/// form instead (`0.08288574218750001`-style), so the C++ rule is rebuilt here:
/// round to 6 significant digits, use scientific notation when the exponent is
/// below -4 or at least 6, and strip trailing zeros.
fn fmt_g(v: f64) -> String {
    const PRECISION: i32 = 6;
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }

    // `{:.5e}` rounds to 6 significant digits and tells us the decimal exponent
    // *after* rounding (e.g. 9.999995 -> "1.00000e1"), which is what %g uses.
    let sci = format!("{:.*e}", (PRECISION - 1) as usize, v);
    let (mantissa, exponent) = sci.split_once('e').expect("{:e} always has an exponent");
    let exponent: i32 = exponent.parse().expect("exponent is an integer");

    if !(-4..PRECISION).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{}e{sign}{:02}",
            strip_trailing_zeros(mantissa),
            exponent.abs()
        )
    } else {
        let decimals = (PRECISION - 1 - exponent) as usize;
        strip_trailing_zeros(&format!("{v:.decimals$}")).to_string()
    }
}

fn strip_trailing_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_g;

    #[test]
    fn fmt_g_matches_printf() {
        // Expected strings are what glibc printf("%g") produces.
        let cases = [
            (0.0, "0"),
            (1.0, "1"),
            (0.5, "0.5"),
            (0.025, "0.025"),
            (0.0828857421875, "0.0828857"),
            (0.911819458, "0.911819"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (9.9999996, "10"),
            (-2.5, "-2.5"),
        ];
        for (v, expected) in cases {
            assert_eq!(fmt_g(v), expected, "fmt_g({v})");
        }
    }
}
