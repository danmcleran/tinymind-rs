// NeuralNetwork with two and three hidden layers against tinymind's
// NeuralNetwork<..., HiddenLayers<...>, ...>. tests/golden/deep_mlp/stdout.txt is
// the output of scripts/cpp/deep_mlp.cpp: the same initial weights (glibc
// rand(), as tinymind's examples draw them), the same samples, and an FNV-1a
// hash of every weight after every training step. scripts/compare-with-cpp.sh
// rebuilds it and checks the golden is current.

use std::fmt::Write;

use tinymind::rand::GlibcRand;
use tinymind::{Dense, Fixed, Layers, NeuralNetwork, Q8_8, Q16_16, Relu, Sigmoid, Tanh};

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

/// A format's raw-value conversions.
trait Raw: Fixed + PartialEq + core::fmt::Debug {
    const ONE_RAW: i32;
    fn from_raw_i32(r: i32) -> Self;
    fn to_raw_i32(self) -> i32;
}

impl Raw for Q16_16 {
    const ONE_RAW: i32 = 1 << 16;
    fn from_raw_i32(r: i32) -> Self {
        Q16_16::from_raw(r)
    }
    fn to_raw_i32(self) -> i32 {
        self.raw()
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
}

/// The C++ sample: XOR of two bits for 2 inputs; for 4 inputs, inputs in
/// [0, 1) and a one-hot target for the largest of three pair sums.
fn sample<V: Raw, const IN: usize, const OUT: usize>(lcg: &mut Lcg) -> ([V; IN], [V; OUT]) {
    let one = V::ONE_RAW;
    if IN == 2 {
        let a = ((lcg.next() >> 16) & 1) as i32;
        let b = ((lcg.next() >> 16) & 1) as i32;
        let x = core::array::from_fn(|i| V::from_raw_i32([a, b][i] * one));
        let t = core::array::from_fn(|_| V::from_raw_i32((a ^ b) * one));
        return (x, t);
    }
    let v: [i32; IN] = core::array::from_fn(|_| ((lcg.next() >> 8) % one as u32) as i32);
    let s = [v[0] + v[1], v[2] + v[3], v[0] + v[3]];
    let mut best = 0;
    for k in 1..3 {
        if s[k] > s[best] {
            best = k;
        }
    }
    let x = v.map(V::from_raw_i32);
    let t = core::array::from_fn(|o| V::from_raw_i32(if o == best { one } else { 0 }));
    (x, t)
}

/// Absorb every weight of a network, layer by layer: the weights by source,
/// then the bias weights.
macro_rules! absorb_layers {
    ($hash:expr, $nn:expr, $($idx:tt),+) => {{
        let layers = $nn.layers();
        $(
            for row in layers.$idx.weights() {
                for w in row {
                    absorb($hash, w.to_raw_i32());
                }
            }
            for w in layers.$idx.bias_weights() {
                absorb($hash, w.to_raw_i32());
            }
        )+
    }};
}

macro_rules! run {
    ($out:expr, $name:expr, $v:ty, $in:expr, $outn:expr, $layers:ty, [$($idx:tt),+], $seed:expr, $steps:expr) => {{
        let one = <$v as Raw>::ONE_RAW;
        let half = one / 2;
        let mut rng = GlibcRand::new($seed);
        let mut nn = NeuralNetwork::<$layers>::new(|| {
            <$v>::from_raw_i32(rng.next_i32() % (2 * half + 1) - half)
        });
        let mut lcg = Lcg($seed);
        let mut hash: u32 = 0x811C_9DC5;
        for step in 1..=$steps {
            let (x, t) = sample::<$v, $in, $outn>(&mut lcg);
            nn.feed_forward(&x);
            let error = nn.calculate_error(&t);
            nn.train(&t);
            absorb_layers!(&mut hash, nn, $($idx),+);
            if step % 500 == 0 {
                writeln!($out, "{} step={step} error={} hash=0x{hash:08X}", $name, error.to_raw_i32()).unwrap();
            }
        }
        for k in 0..4 {
            let (x, _) = sample::<$v, $in, $outn>(&mut lcg);
            nn.feed_forward(&x);
            write!($out, "{} final{k}", $name).unwrap();
            for y in nn.learned_values() {
                write!($out, " {}", y.to_raw_i32()).unwrap();
            }
            writeln!($out).unwrap();
        }
    }};
}

fn assert_layers<L: Layers>() {}

#[test]
fn deep_mlp_matches_cpp() {
    type Xor16 = (
        Dense<Q16_16, Tanh, 2, 4>,
        Dense<Q16_16, Tanh, 4, 4>,
        Dense<Q16_16, Sigmoid, 4, 1>,
    );
    type Xor8 = (
        Dense<Q8_8, Tanh, 2, 4>,
        Dense<Q8_8, Tanh, 4, 4>,
        Dense<Q8_8, Sigmoid, 4, 1>,
    );
    type Deep16 = (
        Dense<Q16_16, Relu, 4, 8>,
        Dense<Q16_16, Relu, 8, 6>,
        Dense<Q16_16, Relu, 6, 5>,
        Dense<Q16_16, Sigmoid, 5, 3>,
    );
    type Deep8 = (
        Dense<Q8_8, Relu, 4, 8>,
        Dense<Q8_8, Relu, 8, 6>,
        Dense<Q8_8, Sigmoid, 6, 3>,
    );
    assert_layers::<Deep16>();

    let mut out = String::new();
    run!(
        out,
        "q16.16 2-4-4-1",
        Q16_16,
        2,
        1,
        Xor16,
        [0, 1, 2],
        1,
        4000
    );
    run!(out, "q8.8 2-4-4-1", Q8_8, 2, 1, Xor8, [0, 1, 2], 2, 4000);
    run!(
        out,
        "q16.16 4-8-6-5-3",
        Q16_16,
        4,
        3,
        Deep16,
        [0, 1, 2, 3],
        3,
        3000
    );
    run!(out, "q8.8 4-8-6-3", Q8_8, 4, 3, Deep8, [0, 1, 2], 4, 3000);

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/deep_mlp/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
