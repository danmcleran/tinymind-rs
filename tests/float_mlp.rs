// NeuralNetwork over f32 and f64 against tinymind (C++)'s NeuralNetwork
// instantiated with float and double. tests/golden/float_mlp/stdout.txt is the
// output of scripts/cpp/float_mlp.cpp (built with -ffp-contract=off), which
// trains nine networks with tinymind (C++)'s floating-point activations and
// transfer-function bundle (learning rate 0.15, momentum 0.5, acceleration
// 0.1), five with SGD and four with its AdamOptimizerFloat and
// RmsPropOptimizerFloat (learning rate 0.01), and hashes every weight after
// every step; scripts/compare-with-cpp.sh
// rebuilds it and checks the golden is current. tanh and exp come from the
// platform's math library on both sides (feature `std`), so on a glibc host
// every step is bit-identical. The committed golden holds the f64 lines only;
// see the end of the test.
#![cfg(feature = "std")]

use std::fmt::Write;

use tinymind::rand::GlibcRand;
use tinymind::{
    AdamFloat, Dense, Linear, NeuralNetwork, Optimizer, Relu, RmsPropFloat, Scalar, Sgd, Sigmoid,
    Tanh,
};

/// What the test needs from f32 and f64 beyond `Scalar`.
trait Float: Scalar + core::fmt::Debug {
    fn from_f64(x: f64) -> Self;
    fn from_i32(x: i32) -> Self;
    fn absorb(self, hash: &mut u32);
    fn bits(self) -> String;
}

impl Float for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn from_i32(x: i32) -> Self {
        f64::from(x)
    }
    fn absorb(self, hash: &mut u32) {
        for b in self.to_bits().to_le_bytes() {
            *hash = (*hash ^ u32::from(b)).wrapping_mul(0x0100_0193);
        }
    }
    fn bits(self) -> String {
        format!("{:016X}", self.to_bits())
    }
}

impl Float for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn from_i32(x: i32) -> Self {
        x as f32
    }
    fn absorb(self, hash: &mut u32) {
        for b in self.to_bits().to_le_bytes() {
            *hash = (*hash ^ u32::from(b)).wrapping_mul(0x0100_0193);
        }
    }
    fn bits(self) -> String {
        format!("{:08X}", self.to_bits())
    }
}

/// Every weight in the C++ program's order: each layer's weights by source,
/// then its bias weights, from the input side.
trait AllWeights<V> {
    fn all_weights(&self) -> Vec<V>;
}

impl<V: Copy, A0, A1, const I: usize, const H: usize, const O: usize> AllWeights<V>
    for (Dense<V, A0, I, H>, Dense<V, A1, H, O>)
{
    fn all_weights(&self) -> Vec<V> {
        let mut w: Vec<V> = self.0.weights().iter().flatten().copied().collect();
        w.extend(self.0.bias_weights());
        w.extend(self.1.weights().iter().flatten().copied());
        w.extend(self.1.bias_weights());
        w
    }
}

impl<V: Copy, A0, A1, A2, const I: usize, const H1: usize, const H2: usize, const O: usize>
    AllWeights<V>
    for (
        Dense<V, A0, I, H1>,
        Dense<V, A1, H1, H2>,
        Dense<V, A2, H2, O>,
    )
{
    fn all_weights(&self) -> Vec<V> {
        let mut w: Vec<V> = self.0.weights().iter().flatten().copied().collect();
        w.extend(self.0.bias_weights());
        w.extend(self.1.weights().iter().flatten().copied());
        w.extend(self.1.bias_weights());
        w.extend(self.2.weights().iter().flatten().copied());
        w.extend(self.2.bias_weights());
        w
    }
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
}

/// The C++ program's sample for `IN` inputs (see its `sample`).
fn sample<V: Float, const IN: usize, const OUT: usize>(g: &mut Lcg) -> ([V; IN], [V; OUT]) {
    let mut x = [V::ZERO; IN];
    let mut t = [V::ZERO; OUT];
    if IN == 2 {
        let a = ((g.next() >> 16) & 1) as i32;
        let b = ((g.next() >> 16) & 1) as i32;
        x[0] = V::from_i32(a);
        x[1] = V::from_i32(b);
        t[0] = V::from_i32(a ^ b);
        return (x, t);
    }
    let mut v = [0i32; IN];
    for vi in &mut v {
        *vi = ((g.next() >> 8) % 65_536) as i32;
    }
    if IN == 3 {
        for (xi, &vi) in x.iter_mut().zip(&v) {
            *xi = V::from_i32(vi - 32_768) / V::from_i32(32_768);
        }
        t[0] = V::from_i32(v[0] - v[2]) / V::from_i32(65_536);
        t[1 % OUT] = V::from_i32(v[1] + v[2]) / V::from_i32(131_072);
        return (x, t);
    }
    for (xi, &vi) in x.iter_mut().zip(&v) {
        *xi = V::from_i32(vi) / V::from_i32(65_536);
    }
    let s = [v[0] + v[1], v[2] + v[3], v[0] + v[3]];
    let mut best = 0;
    for k in 1..3 {
        if s[k] > s[best] {
            best = k;
        }
    }
    for (o, to) in t.iter_mut().enumerate() {
        *to = V::from_i32(i32::from(o == best));
    }
    (x, t)
}

fn run<V, L, const IN: usize, const OUT: usize>(name: &str, seed: u32, steps: u32, out: &mut String)
where
    V: Float,
    L: tinymind::Layers<Value = V, Inputs = [V; IN], Outputs = [V; OUT]> + Default + AllWeights<V>,
    Sgd: Optimizer<V>,
{
    run_with::<V, L, Sgd, IN, OUT>(name, seed, steps, Sgd, 0.15, out);
}

/// One configuration with optimizer `opt` and learning rate `lr` (the C++
/// program gives the adaptive runs 0.01).
fn run_with<V, L, O, const IN: usize, const OUT: usize>(
    name: &str,
    seed: u32,
    steps: u32,
    opt: O,
    lr: f64,
    out: &mut String,
) where
    V: Float,
    L: tinymind::Layers<Value = V, Inputs = [V; IN], Outputs = [V; OUT]> + Default + AllWeights<V>,
    O: Optimizer<V>,
{
    let mut rand = GlibcRand::new(seed);
    let mut nn: NeuralNetwork<L, O> = NeuralNetwork::with_optimizer(
        || V::from_f64(f64::from(rand.next_i32()) / f64::from(i32::MAX) - 0.5),
        opt,
    );
    // tinymind (C++)'s floating-point transfer-function bundle.
    nn.set_learning_rate(V::from_f64(lr));
    nn.set_momentum_rate(V::from_f64(0.5));
    nn.set_acceleration_rate(V::from_f64(0.1));
    let mut g = Lcg(seed);
    let mut hash = 0x811C_9DC5u32;
    for step in 1..=steps {
        let (x, t) = sample::<V, IN, OUT>(&mut g);
        nn.feed_forward(&x);
        let error = nn.calculate_error(&t);
        nn.train(&t);
        for w in nn.layers().all_weights() {
            w.absorb(&mut hash);
        }
        if step % 500 == 0 {
            writeln!(
                out,
                "{name} step={step} error= {} hash=0x{hash:08X}",
                error.bits()
            )
            .unwrap();
        }
    }
    for k in 0..4 {
        let (x, _) = sample::<V, IN, OUT>(&mut g);
        nn.feed_forward(&x);
        write!(out, "{name} final{k}").unwrap();
        for y in nn.learned_values() {
            write!(out, " {}", y.bits()).unwrap();
        }
        writeln!(out).unwrap();
    }
}

#[test]
fn float_networks_match_cpp() {
    let mut out = String::new();
    run::<f64, (Dense<f64, Tanh, 2, 4>, Dense<f64, Sigmoid, 4, 1>), 2, 1>(
        "f64 2-4-1 tanh/sigmoid",
        1,
        3000,
        &mut out,
    );
    run::<
        f32,
        (
            Dense<f32, Tanh, 2, 4>,
            Dense<f32, Tanh, 4, 4>,
            Dense<f32, Sigmoid, 4, 1>,
        ),
        2,
        1,
    >("f32 2-4-4-1 tanh/sigmoid", 2, 3000, &mut out);
    run::<
        f64,
        (
            Dense<f64, Relu, 4, 8>,
            Dense<f64, Relu, 8, 6>,
            Dense<f64, Sigmoid, 6, 3>,
        ),
        4,
        3,
    >("f64 4-8-6-3 relu/sigmoid", 3, 2000, &mut out);
    run::<f32, (Dense<f32, Tanh, 4, 8>, Dense<f32, Sigmoid, 8, 3>), 4, 3>(
        "f32 4-8-3 tanh/sigmoid",
        4,
        2000,
        &mut out,
    );
    run::<f64, (Dense<f64, Relu, 3, 6>, Dense<f64, Linear, 6, 2>), 3, 2>(
        "f64 3-6-2 relu/linear",
        5,
        2000,
        &mut out,
    );
    // tinymind (C++)'s floating-point Adam and RMSprop.
    run_with::<f64, (Dense<f64, Tanh, 2, 4>, Dense<f64, Sigmoid, 4, 1>), _, 2, 1>(
        "f64 2-4-1 tanh/sigmoid adam",
        6,
        2000,
        AdamFloat::default(),
        0.01,
        &mut out,
    );
    run_with::<
        f32,
        (
            Dense<f32, Tanh, 4, 8>,
            Dense<f32, Tanh, 8, 6>,
            Dense<f32, Sigmoid, 6, 3>,
        ),
        _,
        4,
        3,
    >(
        "f32 4-8-6-3 tanh/sigmoid adam",
        7,
        2000,
        AdamFloat::default(),
        0.01,
        &mut out,
    );
    run_with::<f64, (Dense<f64, Relu, 3, 6>, Dense<f64, Linear, 6, 2>), _, 3, 2>(
        "f64 3-6-2 relu/linear rmsprop",
        8,
        2000,
        RmsPropFloat::default(),
        0.01,
        &mut out,
    );
    run_with::<
        f32,
        (
            Dense<f32, Tanh, 2, 4>,
            Dense<f32, Tanh, 4, 4>,
            Dense<f32, Sigmoid, 4, 1>,
        ),
        _,
        2,
        1,
    >(
        "f32 2-4-4-1 tanh/sigmoid rmsprop",
        9,
        2000,
        RmsPropFloat::default(),
        0.01,
        &mut out,
    );
    // With TINYMIND_FLOAT_MLP_CPP set (scripts/compare-with-cpp.sh sets it),
    // every line is compared with the C++ program's output from the same
    // machine. Otherwise the f64 lines are compared with the committed golden:
    // the f32 ones depend on the C library's tanhf and expf, which glibc 2.41
    // replaced with correctly rounded versions, so they differ between hosts.
    match std::env::var("TINYMIND_FLOAT_MLP_CPP") {
        Ok(path) => {
            let cpp = std::fs::read_to_string(path).expect("the C++ program's output");
            assert_eq!(out, cpp);
        }
        Err(_) => {
            let f64_lines: String = out
                .lines()
                .filter(|l| l.starts_with("f64 "))
                .map(|l| format!("{l}\n"))
                .collect();
            assert_eq!(f64_lines, include_str!("golden/float_mlp/stdout.txt"));
        }
    }
}
