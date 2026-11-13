// The fine-tuning heads' on-board workload, shared by tests/head_sweep.rs (on
// the host, against tinymind (C++)'s `bench` lines in
// tests/golden/head/stdout.txt) and bench/src/head.rs (on the boards, timed,
// against the same lines). no_std and allocation-free. The C++ is `benchRun`
// in scripts/cpp/head_sweep.cpp.
//
// Q16.16 embeddings of 32 values for 6 classes, 5 samples each, drawn from the
// sweep's generator, and a shipped dense head drawn after them. The data is
// built by a `const fn`, so on a board it lives in flash.

use tinymind::Q16_16;
use tinymind::head::{DenseHead, PrototypeHead};

pub const IN: usize = 32;
pub const OUT: usize = 6;
pub const N: usize = 30;
pub const EPOCHS: usize = 30;
/// 0.05 in Q16.16 (`lround(0.05 * 2^16)`).
pub const RATE: Q16_16 = Q16_16::from_raw(3277);

pub type Head = DenseHead<Q16_16, IN, OUT>;
pub type Prototypes = PrototypeHead<Q16_16, IN, OUT>;

pub struct Data {
    pub x: [[Q16_16; IN]; N],
    pub label: [usize; N],
    pub shipped: Head,
}

const ONE: i32 = 1 << 16;

/// The sweep's generator: a whole number in [-range, range].
const fn draw(state: &mut u32, range: i32) -> i32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) % (2 * range + 1) as u32) as i32 - range
}

pub const fn data() -> Data {
    let mut state = 2026u32;
    let mut x = [[Q16_16::from_raw(0); IN]; N];
    let mut label = [0usize; N];
    let mut s = 0;
    while s < N {
        label[s] = s % OUT;
        let mut i = 0;
        while i < IN {
            let base = ((label[s] * 5 + i * 3) % 8) as i32 * (ONE / 4);
            x[s][i] = Q16_16::from_raw(base + draw(&mut state, ONE / 4));
            i += 1;
        }
        s += 1;
    }
    // C++ `initializeWeights`: every weight, output by output, biases 0.
    let mut weights = [[Q16_16::from_raw(0); IN]; OUT];
    let mut o = 0;
    while o < OUT {
        let mut i = 0;
        while i < IN {
            weights[o][i] = Q16_16::from_raw(draw(&mut state, ONE));
            i += 1;
        }
        o += 1;
    }
    Data {
        x,
        label,
        shipped: DenseHead::from_weights(weights, [Q16_16::from_raw(0); OUT]),
    }
}

pub const FNV_OFFSET: u32 = 0x811C_9DC5;

/// FNV-1a over the 8 little-endian bytes of `bits` (the sweep's hash).
pub fn fnv(mut hash: u32, bits: u64) -> u32 {
    for b in bits.to_le_bytes() {
        hash = (hash ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    hash
}

fn q(v: Q16_16) -> u64 {
    i64::from(v.raw()) as u64
}

pub fn hash_dense(head: &Head, mut hash: u32) -> u32 {
    for (row, &b) in head.weights.iter().zip(&head.biases) {
        for &w in row {
            hash = fnv(hash, q(w));
        }
        hash = fnv(hash, q(b));
    }
    hash
}

pub fn hash_prototypes(p: &Prototypes, mut hash: u32) -> u32 {
    for c in 0..OUT {
        if let Some(prototype) = p.prototype(c) {
            for &v in prototype {
                hash = fnv(hash, q(v));
            }
        }
        hash = fnv(hash, u64::from(p.count(c)));
    }
    hash
}

/// The prototypes of all `N` samples.
pub fn learned(d: &Data) -> Prototypes {
    let mut p = Prototypes::new();
    for (x, &label) in d.x.iter().zip(&d.label) {
        p.learn(x, label);
    }
    p
}

/// `EPOCHS` passes of `train_class` over the samples, in order.
pub fn adapt(head: &mut Head, d: &Data) {
    for _ in 0..EPOCHS {
        for (x, &label) in d.x.iter().zip(&d.label) {
            head.train_class(x, label, RATE, None);
        }
    }
}

/// The five hashes the C++ prints, in its order (head_step, head_predict,
/// proto_learn, proto_predict, head_adapt), and how many samples the adapted
/// head classifies correctly.
pub fn hashes(d: &Data) -> ([u32; 5], usize) {
    let learned = learned(d);
    let (mut step, mut predict, mut learn, mut nearest) =
        (FNV_OFFSET, FNV_OFFSET, FNV_OFFSET, FNV_OFFSET);
    for (x, &label) in d.x.iter().zip(&d.label) {
        let mut head = d.shipped;
        head.train_class(x, label, RATE, None);
        step = hash_dense(&head, step);
        predict = fnv(predict, d.shipped.predict(x) as u64);
        let mut p = learned;
        p.learn(x, label);
        learn = hash_prototypes(&p, learn);
        nearest = fnv(nearest, learned.predict(x) as u64);
    }
    let mut adapted = d.shipped;
    adapt(&mut adapted, d);
    let correct =
        d.x.iter()
            .zip(&d.label)
            .filter(|(x, label)| adapted.predict(x) == **label)
            .count();
    (
        [
            step,
            predict,
            learn,
            nearest,
            hash_dense(&adapted, FNV_OFFSET),
        ],
        correct,
    )
}

pub const NAMES: [&str; 5] = [
    "head_step",
    "head_predict",
    "proto_learn",
    "proto_predict",
    "head_adapt",
];

/// The hash the golden's `bench <name> <hex>` line gives.
pub fn expected(golden: &str, name: &str) -> Option<u32> {
    golden.lines().find_map(|line| {
        let rest = line.strip_prefix("bench ")?.strip_prefix(name)?;
        let hex = rest.strip_prefix(' ')?.split(' ').next()?;
        u32::from_str_radix(hex, 16).ok()
    })
}
