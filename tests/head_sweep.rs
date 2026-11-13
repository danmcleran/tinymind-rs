// DenseHead and PrototypeHead against tinymind's C++ (DenseHead,
// PrototypeHead): nine dense heads (Q16.16, Q8.8, f64 and f32; softmax
// cross-entropy and squared error; one with logits far beyond the exp table,
// one wrapping in Q8.8), the softmax on its own, and seven prototype heads
// (squared Euclidean and Manhattan; one whose squared distances wrap), every
// parameter and output hashed after every step.
// tests/golden/head/stdout.txt is the output of scripts/cpp/head_sweep.cpp
// (scripts/compare-with-cpp.sh rebuilds it and checks the golden is current).
// Needs `std`: the f64 softmax uses the platform's exp, as the C++ does.
#![cfg(feature = "std")]

use std::fmt::Write;

use tinymind::head::{
    DenseHead, Distance, HeadLoss, ManhattanDistance, PrototypeHead, SoftmaxCrossEntropyLoss,
    SquaredErrorLoss, SquaredEuclideanDistance,
};
use tinymind::{Q8_8, Q16_16, Scalar};

#[path = "sweeps/head_bench.rs"]
mod head_bench;

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }

    /// A whole number in [-range, range].
    fn draw(&mut self, range: i32) -> i32 {
        (self.next() % (2 * range + 1) as u32) as i32 - range
    }

    /// A value in [-units, units], in raw steps.
    fn value<V: Vt>(&mut self, units: i32) -> V {
        V::make(self.draw(units * V::ONE_RAW))
    }
}

const FNV_OFFSET: u32 = 0x811C_9DC5;

fn fnv(mut hash: u32, bits: u64) -> u32 {
    for b in bits.to_le_bytes() {
        hash = (hash ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    hash
}

fn hash_values<V: Vt>(x: &[V]) -> u32 {
    x.iter().fold(FNV_OFFSET, |h, v| fnv(h, v.bits()))
}

/// One value type: how to build a value from raw steps, its bits, its name.
trait Vt: Scalar {
    const ONE_RAW: i32;
    const NAME: &'static str;
    fn make(raw: i32) -> Self;
    fn bits(self) -> u64;
}

impl Vt for Q16_16 {
    const ONE_RAW: i32 = 1 << 16;
    const NAME: &'static str = "q16.16";
    fn make(raw: i32) -> Self {
        Q16_16::from_raw(raw)
    }
    fn bits(self) -> u64 {
        i64::from(self.raw()) as u64
    }
}

impl Vt for Q8_8 {
    const ONE_RAW: i32 = 1 << 8;
    const NAME: &'static str = "q8.8";
    fn make(raw: i32) -> Self {
        Q8_8::from_raw(raw as i16)
    }
    fn bits(self) -> u64 {
        i64::from(self.raw()) as u64
    }
}

impl Vt for f64 {
    const ONE_RAW: i32 = 1 << 16;
    const NAME: &'static str = "f64";
    fn make(raw: i32) -> Self {
        f64::from(raw) / 65536.0
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl Vt for f32 {
    const ONE_RAW: i32 = 1 << 16;
    const NAME: &'static str = "f32";
    fn make(raw: i32) -> Self {
        raw as f32 / 65536.0
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

fn hash_head<V: Vt, const IN: usize, const OUT: usize, L>(head: &DenseHead<V, IN, OUT, L>) -> u32 {
    let mut hash = FNV_OFFSET;
    for (row, b) in head.weights.iter().zip(&head.biases) {
        for w in row {
            hash = fnv(hash, w.bits());
        }
        hash = fnv(hash, b.bits());
    }
    hash
}

fn dense_run<V: Vt, const IN: usize, const OUT: usize, L: HeadLoss<V>>(
    out: &mut String,
    loss: &str,
    seed: u32,
    input_units: i32,
    lr_raw: i32,
) {
    let mut rng = Lcg(seed);
    let mut head = DenseHead::<V, IN, OUT, L>::new();
    writeln!(
        out,
        "dense {} {loss} {IN}x{OUT} zero {:08x}",
        V::NAME,
        hash_head(&head)
    )
    .unwrap();
    head.initialize_weights(|| rng.value(1));
    writeln!(out, "dense init {:08x}", hash_head(&head)).unwrap();

    let rate = V::make(lr_raw);
    for step in 0..64 {
        let input: [V; IN] = core::array::from_fn(|_| rng.value(input_units));
        let mut input_deltas = [V::make(0); IN];
        let kind = rng.next() % 4;
        let result;
        if kind == 3 {
            let targets: [V; OUT] = core::array::from_fn(|_| rng.value(1));
            let logits = head.forward(&input);
            let deltas = head.output_deltas(&logits, &targets);
            input_deltas = head.backward(&deltas);
            head.update(&deltas, &input, rate);
            result = head.predict(&input);
            write!(
                out,
                "step {step} parts logits {:08x} deltas {:08x}",
                hash_values(&logits),
                hash_values(&deltas)
            )
            .unwrap();
        } else {
            let class = (rng.next() % (OUT as u32 + 1)) as usize;
            let deltas = if kind == 0 {
                None
            } else {
                Some(&mut input_deltas)
            };
            result = head.train_class(&input, class, rate, deltas);
            write!(out, "step {step} class {class}").unwrap();
        }
        writeln!(
            out,
            " pred {result} head {:08x} in {:08x}",
            hash_head(&head),
            hash_values(&input_deltas)
        )
        .unwrap();
    }

    let mut copy = DenseHead::<V, IN, OUT, L>::new();
    copy.load_weights(&head.weights, &head.biases);
    head.clear();
    let probe: [V; IN] = core::array::from_fn(|_| rng.value(input_units));
    // A cleared head ties every logit: the first class wins.
    writeln!(
        out,
        "dense loaded {:08x} cleared {:08x} pred {}",
        hash_head(&copy),
        hash_head(&head),
        head.predict(&probe)
    )
    .unwrap();
}

fn softmax_run<V: Vt + tinymind::head::SoftmaxExp, const N: usize>(
    out: &mut String,
    seed: u32,
    units: i32,
) {
    let mut rng = Lcg(seed);
    for k in 0..32 {
        let logits: [V; N] = core::array::from_fn(|_| rng.value(units));
        let p = SoftmaxCrossEntropyLoss::probabilities(&logits);
        writeln!(out, "softmax {} {N} #{k} {:08x}", V::NAME, hash_values(&p)).unwrap();
    }
}

fn prototype_run<V: Vt, const IN: usize, const C: usize, D: Distance<V>>(
    out: &mut String,
    distance: &str,
    seed: u32,
    input_units: i32,
) {
    let mut rng = Lcg(seed);
    let mut head = PrototypeHead::<V, IN, C, D>::new();
    writeln!(out, "prototype {} {distance} {IN}x{C}", V::NAME).unwrap();

    for step in 0..64 {
        let input: [V; IN] = core::array::from_fn(|_| rng.value(input_units));
        let op = rng.next() % 16;
        let class = (rng.next() % C as u32) as usize;
        if op < 9 {
            head.learn(&input, class);
            write!(out, "step {step} learn {class}").unwrap();
        } else if op == 9 {
            let cap = [0u16, 1, 3, 65535][(rng.next() % 4) as usize];
            head.set_maximum_count(cap);
            write!(out, "step {step} cap {cap}").unwrap();
        } else if op == 10 {
            let count = (rng.next() % 6) as u16;
            head.set_prototype(class, &input, count);
            write!(out, "step {step} set {class} {count}").unwrap();
        } else if op == 11 {
            head.forget(class);
            write!(out, "step {step} forget {class}").unwrap();
        } else if op == 12 && rng.next() % 4 == 0 {
            head.clear();
            write!(out, "step {step} clear").unwrap();
        } else {
            write!(out, "step {step} query").unwrap();
        }

        let query: [V; IN] = core::array::from_fn(|_| rng.value(input_units));
        let mut distances = FNV_OFFSET;
        let mut prototypes = FNV_OFFSET;
        for c in 0..C {
            distances = fnv(distances, head.distance(&query, c).unwrap().bits());
            for p in head.prototype(c).unwrap() {
                prototypes = fnv(prototypes, p.bits());
            }
        }
        write!(
            out,
            " pred {} max {} counts",
            head.predict(&query),
            head.maximum_count()
        )
        .unwrap();
        for c in 0..C {
            write!(out, " {}", head.count(c)).unwrap();
        }
        writeln!(out, " d {distances:08x} p {prototypes:08x}").unwrap();
    }

    // Two classes with the same prototype tie: the first one wins.
    let shared: [V; IN] = core::array::from_fn(|_| rng.value(input_units));
    head.clear();
    head.set_prototype(C - 1, &shared, 1);
    head.set_prototype(1, &shared, 2);
    writeln!(out, "tie pred {}", head.predict(&shared)).unwrap();
}

#[test]
fn head_sweep_matches_cpp() {
    type Sce = SoftmaxCrossEntropyLoss;
    type Se = SquaredErrorLoss;
    type Euclid = SquaredEuclideanDistance;
    type Manhattan = ManhattanDistance;

    let mut out = String::new();
    dense_run::<Q16_16, 8, 4, Sce>(&mut out, "sce", 1, 3, 1 << 13);
    dense_run::<Q16_16, 8, 4, Se>(&mut out, "se", 2, 3, 1 << 11);
    dense_run::<Q16_16, 3, 5, Sce>(&mut out, "sce", 3, 40, 1 << 14);
    dense_run::<Q8_8, 4, 3, Sce>(&mut out, "sce", 4, 2, 1 << 5);
    dense_run::<Q8_8, 4, 3, Se>(&mut out, "se", 5, 8, 1 << 6);
    dense_run::<Q8_8, 1, 1, Sce>(&mut out, "sce", 6, 2, 1 << 5);
    dense_run::<f64, 8, 4, Sce>(&mut out, "sce", 7, 3, 1 << 13);
    dense_run::<f64, 8, 4, Se>(&mut out, "se", 8, 3, 1 << 11);
    dense_run::<f32, 8, 4, Se>(&mut out, "se", 9, 3, 1 << 11);

    softmax_run::<Q16_16, 5>(&mut out, 10, 10);
    softmax_run::<Q8_8, 5>(&mut out, 11, 10);
    softmax_run::<f64, 5>(&mut out, 12, 10);

    prototype_run::<Q16_16, 8, 4, Euclid>(&mut out, "euclid", 20, 3);
    prototype_run::<Q16_16, 8, 4, Manhattan>(&mut out, "manhattan", 21, 50);
    prototype_run::<Q16_16, 4, 3, Euclid>(&mut out, "euclid", 22, 100);
    prototype_run::<Q8_8, 4, 3, Euclid>(&mut out, "euclid", 23, 2);
    prototype_run::<Q8_8, 4, 3, Manhattan>(&mut out, "manhattan", 24, 20);
    prototype_run::<f64, 8, 4, Euclid>(&mut out, "euclid", 25, 3);
    prototype_run::<f32, 8, 4, Manhattan>(&mut out, "manhattan", 26, 3);

    // The on-board workload (bench/src/head.rs), checked here on the host.
    let data = head_bench::data();
    let (hashes, correct) = head_bench::hashes(&data);
    for (name, hash) in head_bench::NAMES.iter().zip(hashes) {
        write!(out, "bench {name} {hash:08x}").unwrap();
        if *name == "head_adapt" {
            write!(out, " correct {correct}").unwrap();
        }
        writeln!(out).unwrap();
    }

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/head/stdout.txt"
    ))
    .unwrap();
    // The boards read their expected hashes back with this parser.
    for (name, hash) in head_bench::NAMES.iter().zip(hashes) {
        assert_eq!(head_bench::expected(&golden, name), Some(hash), "{name}");
    }
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
