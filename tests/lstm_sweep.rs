// LstmNetwork against tinymind's LstmNeuralNetwork. tests/golden/lstm/stdout.txt
// is the output of scripts/cpp/lstm_sweep.cpp, which trains the same networks
// on the same sequences from the same initial weights; scripts/compare-with-cpp.sh
// rebuilds it and checks the golden is current. See that program for what is
// covered.

use std::fmt::Write;

use tinymind::optimizer::Optimizer;
use tinymind::rand::GlibcRand;
use tinymind::recurrent::LstmNetwork;
use tinymind::{Activation, Adam, Linear, Q8_8, Q16_16, Relu, Scalar, Sgd, Sigmoid, Tanh};

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
}

fn absorb(hash: &mut u32, raw: i32) {
    for byte in raw.to_le_bytes() {
        *hash ^= u32::from(byte);
        *hash = hash.wrapping_mul(0x0100_0193);
    }
}

trait Raw: Scalar {
    const ONE_RAW: i32;
    fn from_raw_i32(r: i32) -> Self;
    fn to_raw_i32(self) -> i32;
    /// C++ `ZeroToleranceCalculator`.
    fn within_zero_tolerance(self) -> bool;
}

impl Raw for Q16_16 {
    const ONE_RAW: i32 = 1 << 16;
    fn from_raw_i32(r: i32) -> Self {
        Q16_16::from_raw(r)
    }
    fn to_raw_i32(self) -> i32 {
        self.raw()
    }
    fn within_zero_tolerance(self) -> bool {
        self.is_within_zero_tolerance()
    }
}

impl Raw for Q8_8 {
    const ONE_RAW: i32 = 1 << 8;
    fn from_raw_i32(r: i32) -> Self {
        Q8_8::from_raw(r as i16)
    }
    fn to_raw_i32(self) -> i32 {
        i32::from(self.raw())
    }
    fn within_zero_tolerance(self) -> bool {
        self.is_within_zero_tolerance()
    }
}

/// The C++ `QuarterWeights`: `(rand() % (2 * quarter + 1)) - quarter` in int.
fn quarter_weight<V: Raw>(rng: &mut GlibcRand) -> V {
    let quarter = V::ONE_RAW / 4;
    V::from_raw_i32(rng.next_i32() % (2 * quarter + 1) - quarter)
}

fn sample<V: Raw, const IN: usize, const OUT: usize>(
    lcg: &mut Lcg,
    previous: &mut i32,
    target_scale: i32,
) -> ([V; IN], [V; OUT]) {
    let one = V::ONE_RAW;
    let mut v = [0i32; IN];
    for x in &mut v {
        *x = ((lcg.next() >> 8) % (2 * one) as u32) as i32 - one;
    }
    let t = core::array::from_fn(|o| {
        let bit = match o {
            0 => *previous > 0,
            1 => (*previous > 0) != (v[0] > 0),
            _ => v[o % IN] > *previous,
        };
        V::from_raw_i32(i32::from(bit) * one * target_scale)
    });
    *previous = v[0];
    (v.map(V::from_raw_i32), t)
}

fn absorb_weights<V: Raw, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>(
    nn: &LstmNetwork<V, H, O, IN, HID, OUT, Opt>,
    hash: &mut u32,
) {
    for row in nn.input_weights() {
        for w in row.iter().flatten() {
            absorb(hash, w.to_raw_i32());
        }
    }
    for w in nn.input_bias_weights().iter().flatten() {
        absorb(hash, w.to_raw_i32());
    }
    for row in nn.recurrent_weights() {
        for w in row.iter().flatten() {
            absorb(hash, w.to_raw_i32());
        }
    }
    for w in nn.hidden_weights().iter().flatten() {
        absorb(hash, w.to_raw_i32());
    }
    for w in nn.hidden_bias_weights() {
        absorb(hash, w.to_raw_i32());
    }
}

#[allow(clippy::too_many_arguments)]
fn run<V, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>(
    out: &mut String,
    name: &str,
    seed: u32,
    steps: i32,
    lr_raw: i32,
    target_scale: i32,
    clip: Option<(V, V)>,
    optimizer: Opt,
) where
    V: Raw,
    H: Activation<V>,
    O: Activation<V>,
    Sigmoid: Activation<V>,
    Tanh: Activation<V>,
    Opt: Optimizer<V>,
{
    let mut rng = GlibcRand::new(seed);
    let mut nn = LstmNetwork::<V, H, O, IN, HID, OUT, Opt>::with_optimizer(
        || quarter_weight(&mut rng),
        optimizer,
    );
    if lr_raw != 0 {
        nn.set_learning_rate(V::from_raw_i32(lr_raw));
    }
    nn.set_gradient_clip(clip);
    let mut lcg = Lcg(seed);
    let mut hash = 0x811C_9DC5u32;
    let mut step = 0;
    let mut trained = 0;
    while step < steps {
        let length = 1 + ((lcg.next() >> 8) % 12) as i32;
        let mut previous = 0;
        nn.reset_state();
        let mut k = 0;
        while k < length && step < steps {
            k += 1;
            step += 1;
            let (x, t) = sample::<V, IN, OUT>(&mut lcg, &mut previous, target_scale);
            nn.feed_forward(&x);
            let error = nn.calculate_error(&t);
            if step % 7 != 0 && !error.within_zero_tolerance() {
                nn.train(&t);
                trained += 1;
            }
            for y in nn.learned_values() {
                absorb(&mut hash, y.to_raw_i32());
            }
            absorb_weights(&nn, &mut hash);
            if step % 250 == 0 {
                writeln!(
                    out,
                    "{name} step={step} trained={trained} error={} hash=0x{hash:08X}",
                    error.to_raw_i32()
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn lstm_matches_cpp_golden() {
    let mut out = String::new();
    let unit16 = Some((Q16_16::from_int(-1), Q16_16::ONE));
    let unit8 = Some((Q8_8::from_int(-1), Q8_8::ONE));
    // GradientClipByValue<Q8, 2, 0x80>: QValue(-2, 0x80) = -1.5, QValue(2, 0x80) = 2.5.
    let asymmetric8 = Some((
        Q8_8::from_raw((-2 << 8) | 0x80),
        Q8_8::from_raw((2 << 8) | 0x80),
    ));
    let adam16 = Adam::new(
        Q16_16::from_raw(58982),
        Q16_16::from_raw(65470),
        Q16_16::from_raw(16),
    );

    run::<Q16_16, Tanh, Sigmoid, 3, 16, 4, _>(
        &mut out,
        "q16.16 3-16-4 tanh/sigmoid clip[-1,1]",
        1,
        1500,
        0,
        1,
        unit16,
        Sgd,
    );
    run::<Q16_16, Tanh, Sigmoid, 2, 6, 3, _>(
        &mut out,
        "q16.16 2-6-3 tanh/sigmoid",
        2,
        2000,
        0,
        1,
        None,
        Sgd,
    );
    run::<Q16_16, Relu, Linear, 2, 5, 2, _>(
        &mut out,
        "q16.16 2-5-2 relu/linear clip[-1,1]",
        3,
        2000,
        1 << 10,
        3,
        unit16,
        Sgd,
    );
    run::<Q8_8, Tanh, Sigmoid, 2, 4, 2, _>(
        &mut out,
        "q8.8 2-4-2 tanh/sigmoid clip[-1,1]",
        4,
        2000,
        0,
        1,
        unit8,
        Sgd,
    );
    run::<Q8_8, Relu, Sigmoid, 3, 6, 3, _>(
        &mut out,
        "q8.8 3-6-3 relu/sigmoid clip[-1.5,2.5]",
        5,
        2000,
        0,
        1,
        asymmetric8,
        Sgd,
    );
    run::<Q16_16, Tanh, Sigmoid, 1, 1, 1, _>(
        &mut out,
        "q16.16 1-1-1 tanh/sigmoid",
        6,
        1000,
        0,
        1,
        None,
        Sgd,
    );
    run::<Q16_16, Tanh, Sigmoid, 2, 4, 2, _>(
        &mut out,
        "q16.16 2-4-2 tanh/sigmoid adam",
        7,
        2000,
        1 << 11,
        1,
        None,
        adam16,
    );

    let golden = include_str!("golden/lstm/stdout.txt");
    for (i, (a, g)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(a, g, "line {}", i + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count(), "line count");
}
