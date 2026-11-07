//! Port of tinymind `examples/xor/xor.cpp`: train a 2-4-1 perceptron (tanh
//! hidden, sigmoid output) on XOR and write the same files the C++ example
//! writes, byte for byte:
//!
//! - `nn_fixed_xor.txt`: every weight, the target, the output and the error after
//!   each of the 20000 training steps (raw LSBs)
//! - `xor_training.csv`: the 4-case XOR error every 100 steps
//! - `xor_decision_surface.csv`: the trained network over a 41x41 input grid
//!
//! Usage: `xor [--format q16.16|q8.8] [output_dir] [learning_rate] [momentum]`.
//! The default format is Q16.16, as in the C++ example; Q8.8 matches the C++
//! example built with `NUMBER_OF_FIXED_BITS` and `NUMBER_OF_FRACTIONAL_BITS` set
//! to 8 in `xornet.h`. The default output dir is `output`.

use std::fmt::Display;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use tinymind::rand::GlibcRand;
use tinymind::{Activation, Fixed, MultilayerPerceptron, Q8_8, Q16_16, Sigmoid, Tanh};

const TRAINING_ITERATIONS: u32 = 20_000;
const NUM_SAMPLES_AVG_ERROR: u32 = 100;
const RANDOM_SEED: u32 = 7;
/// C++ `WEIGHT_INIT_SEED`: seeds the initial weights only.
const WEIGHT_INIT_SEED: u32 = 1;

const NUMBER_OF_INPUTS: usize = 2;
const NUMBER_OF_NEURONS_PER_HIDDEN_LAYER: usize = 4;
const NUMBER_OF_OUTPUTS: usize = 1;

type NeuralNetwork<V> = MultilayerPerceptron<
    V,
    Tanh,
    Sigmoid,
    NUMBER_OF_INPUTS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
>;

const XOR_CASES: [[i32; 3]; 4] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];

/// What the driver needs from a value type beyond [`Fixed`]. Both formats have
/// these as inherent methods; the trait lets [`run`] be generic over them.
trait Format: Fixed + Display {
    fn from_int(n: i32) -> Self;
    fn from_f64_trunc(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn is_within_zero_tolerance(self) -> bool;
    /// C++ `ValueType(abs(error.getValue()))`.
    fn abs(self) -> Self;
    /// C++ `RandomNumberGenerator::generateRandomWeight` given one `rand()` draw.
    fn random_weight(rand: i32) -> Self;
}

macro_rules! impl_format {
    ($v:ty, $s:ty) => {
        impl Format for $v {
            fn from_int(n: i32) -> Self {
                <$v>::from_int(n)
            }
            fn from_f64_trunc(x: f64) -> Self {
                <$v>::from_f64_trunc(x)
            }
            fn to_f64(self) -> f64 {
                <$v>::to_f64(self)
            }
            fn is_within_zero_tolerance(self) -> bool {
                <$v>::is_within_zero_tolerance(self)
            }
            fn abs(self) -> Self {
                <$v>::from_raw(self.raw().wrapping_abs())
            }
            fn random_weight(rand: i32) -> Self {
                // C++: `static_cast<FullWidthValueType>((rand() % (2 * half + 1)) - half)`
                // with `half` = 0.5: a weight in [-0.5, 0.5], computed in int and
                // then narrowed. (Before tinymind #239 the cast came first, which
                // truncated rand() for 16-bit storage.)
                let half = i32::from(<$v>::ONE.raw() / 2);
                let weight = rand % (2 * half + 1) - half;
                <$v>::from_raw(weight as $s)
            }
        }
    };
}

impl_format!(Q16_16, i32);
impl_format!(Q8_8, i16);

fn main() -> io::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let format = if args.first().map(String::as_str) == Some("--format") {
        if args.len() < 2 {
            usage("--format needs a value");
        }
        let f = args.remove(1);
        args.remove(0);
        f
    } else {
        "q16.16".into()
    };
    let mut args = args.into_iter();
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output".into()));
    let learning_rate = args.next().map(|s| parse_arg(&s));
    let momentum = args.next().map(|s| parse_arg(&s));
    if args.next().is_some() {
        usage("too many arguments");
    }
    fs::create_dir_all(&out_dir)?;

    match format.as_str() {
        "q16.16" => run::<Q16_16>(&out_dir, learning_rate, momentum),
        "q8.8" => run::<Q8_8>(&out_dir, learning_rate, momentum),
        other => usage(&format!("unknown format {other:?}")),
    }
}

fn usage(problem: &str) -> ! {
    eprintln!("xor: {problem}");
    eprintln!("usage: xor [--format q16.16|q8.8] [output_dir] [learning_rate] [momentum]");
    std::process::exit(2);
}

fn run<V: Format>(
    out_dir: &Path,
    learning_rate: Option<f64>,
    momentum: Option<f64>,
) -> io::Result<()>
where
    Tanh: Activation<V>,
    Sigmoid: Activation<V>,
{
    // The C++ example seeds rand() twice: srand(WEIGHT_INIT_SEED = 1), construct
    // the network (which draws the initial weights), then srand(7) for the
    // training samples. Re-seeding a global generator becomes two independent
    // generator values in Rust. (The two seeds exist because the C++ network used
    // to be a global, built before main() seeded rand(), i.e. from glibc's default
    // seed of 1; tinymind PR #236 made that explicit without changing any output.)
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let mut nn = NeuralNetwork::<V>::new(|| V::random_weight(init_rng.next_i32()));
    let mut rng = GlibcRand::new(RANDOM_SEED);
    if let Some(lr) = learning_rate {
        nn.set_learning_rate(V::from_f64_trunc(lr));
    }
    if let Some(m) = momentum {
        nn.set_momentum_rate(V::from_f64_trunc(m));
    }

    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    let mut results = BufWriter::new(File::create(out_dir.join("nn_fixed_xor.txt"))?);
    let mut curve = BufWriter::new(File::create(out_dir.join("xor_training.csv"))?);
    writeln!(curve, "iteration,avg_error")?;

    write_header(&mut results)?;

    let mut avg_error = V::ZERO;
    for i in 0..TRAINING_ITERATIONS {
        let (inputs, target) = generate_xor_training_value::<V>(&mut rng);

        nn.feed_forward(&inputs);
        let error = nn.calculate_error(&target);
        if !error.is_within_zero_tolerance() {
            nn.train(&target);
        }
        let learned = nn.learned_values();

        store_network_properties(&nn, &mut results, &target, &learned)?;
        writeln!(results, "{error}")?;

        // Accumulate |error| in raw LSBs; the average divides it back down.
        avg_error += error.abs();
        if (i + 1) % NUM_SAMPLES_AVG_ERROR == 0 {
            avg_error = avg_error / V::from_int(NUM_SAMPLES_AVG_ERROR as i32);
            writeln!(stdout, "Iteration: {} Average Error: {}", i + 1, avg_error)?;
            writeln!(curve, "{},{}", i + 1, fmt_g(evaluate_xor_error(&mut nn)))?;
            avg_error = V::ZERO;
        }
    }

    // Success = every XOR pattern lands on the correct side of 0.5.
    let mut correct = 0;
    for [a, b, expected] in XOR_CASES {
        let pred = predict(&mut nn, V::from_int(a), V::from_int(b));
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
            let prob = predict(&mut nn, V::from_f64_trunc(x0), V::from_f64_trunc(x1));
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
        .unwrap_or_else(|_| usage(&format!("expected a number, got {s:?}")))
}

fn generate_xor_training_value<V: Format>(rng: &mut GlibcRand) -> ([V; 2], [V; 1]) {
    let x = rng.next_i32() & 1;
    let y = rng.next_i32() & 1;
    ([V::from_int(x), V::from_int(y)], [V::from_int(x ^ y)])
}

fn predict<V: Format>(nn: &mut NeuralNetwork<V>, x0: V, x1: V) -> f64
where
    Tanh: Activation<V>,
    Sigmoid: Activation<V>,
{
    nn.feed_forward(&[x0, x1]);
    nn.learned_values()[0].to_f64()
}

/// Mean |prediction - target| over the four XOR cases, in real units.
fn evaluate_xor_error<V: Format>(nn: &mut NeuralNetwork<V>) -> f64
where
    Tanh: Activation<V>,
    Sigmoid: Activation<V>,
{
    let mut sum = 0.0;
    for [a, b, expected] in XOR_CASES {
        let pred = predict(nn, V::from_int(a), V::from_int(b));
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

fn store_network_properties<V: Format>(
    nn: &NeuralNetwork<V>,
    out: &mut impl Write,
    expected: &[V; NUMBER_OF_OUTPUTS],
    learned: &[V; NUMBER_OF_OUTPUTS],
) -> io::Result<()>
where
    Tanh: Activation<V>,
    Sigmoid: Activation<V>,
{
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
