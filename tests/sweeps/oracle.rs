// The Q-format oracle, shared by tests/qformat_oracle.rs (on the host, with
// the soak's seed and scale and the exhaustive Q8.8 sweep) and
// bench/src/sweeps.rs (on Cortex-M, default sample): every `QValue` operation
// against an exact reference computed in wide integers. See
// tests/qformat_oracle.rs for what it covers. no_std and allocation-free.

use core::fmt;

use tinymind::{QValue, Scalar};

pub type Wide = i64;

// === Exact reference arithmetic =============================================

/// A signed format whose width is its storage width.
#[derive(Clone, Copy, Debug)]
pub struct Format {
    fixed_bits: u32,
    frac_bits: u32,
}

impl Format {
    fn width(self) -> u32 {
        self.fixed_bits + self.frac_bits
    }

    fn one(self) -> Wide {
        1 << self.frac_bits
    }

    fn min(self) -> Wide {
        -(1 << (self.width() - 1))
    }

    fn max(self) -> Wide {
        (1 << (self.width() - 1)) - 1
    }

    fn in_range(self, v: Wide) -> bool {
        v >= self.min() && v <= self.max()
    }

    /// Wrap modulo 2^width into the signed range (WrapPolicy).
    fn wrap(self, x: Wide) -> Wide {
        wrap_to(x, self.width())
    }

    fn add(self, a: Wide, b: Wide) -> Wide {
        self.wrap(a + b)
    }

    fn sub(self, a: Wide, b: Wide) -> Wide {
        self.wrap(a - b)
    }

    /// TruncatePolicy: the exact product, floored to the format's resolution.
    fn mul(self, a: Wide, b: Wide) -> Wide {
        self.wrap((a * b).div_euclid(self.one()))
    }

    fn div_by_zero(self, a: Wide) -> Wide {
        match a.signum() {
            1 => self.max(),
            -1 => self.min(),
            _ => 0,
        }
    }

    /// Quotient truncated towards zero, as integer division does.
    fn div(self, a: Wide, b: Wide) -> Wide {
        if b == 0 {
            self.div_by_zero(a)
        } else {
            self.wrap(a * self.one() / b)
        }
    }

    /// Integer divisor: the scaled quotient a*2^F / (n*2^F) is a / n.
    fn div_int(self, a: Wide, n: Wide) -> Wide {
        if n == 0 {
            self.div_by_zero(a)
        } else {
            self.wrap(a / n)
        }
    }

    fn compare_mask(a: Wide, b: Wide) -> Wide {
        Wide::from(a == b)
            | Wide::from(a != b) << 1
            | Wide::from(a < b) << 2
            | Wide::from(a > b) << 3
            | Wide::from(a <= b) << 4
            | Wide::from(a >= b) << 5
    }

    /// C++ `fromParts(n, 0)`: the fixed part reduced to `fixed_bits`, then scaled.
    fn int_value(self, n: Wide) -> Wide {
        self.wrap(wrap_to(n, self.fixed_bits) * self.one())
    }

    /// C++ `ZeroToleranceCalculator`: |v| < 2^(FRAC - 7), or < 2 for FRAC <= 7.
    fn within_zero_tolerance(self, v: Wide) -> bool {
        let shift = if self.frac_bits > 7 {
            self.frac_bits - 7
        } else {
            1
        };
        v.abs() < 1 << shift
    }
}

/// x reduced modulo 2^bits into the signed range of that width.
fn wrap_to(x: Wide, bits: u32) -> Wide {
    let m = 1i64 << bits;
    let low = x.rem_euclid(m);
    if low >= m / 2 { low - m } else { low }
}

// === The operations under test ==============================================

/// `QValue` seen through wide integers. Implemented for every `i16` and `i32`
/// format; the methods called are the crate's public API.
pub trait Oracle: Scalar + Ord {
    const FORMAT: Format;
    fn from_wide(v: Wide) -> Self;
    fn to_wide(self) -> Wide;
    fn q_from_int(n: i32) -> Self;
    fn q_within_zero_tolerance(self) -> bool;
    fn q_to_f64(self) -> f64;
    fn q_from_f64_trunc(x: f64) -> Self;
    fn q_one() -> Self;
    fn q_zero() -> Self;
}

macro_rules! impl_oracle {
    ($s:ty) => {
        impl<const FRAC: u32> Oracle for QValue<$s, FRAC> {
            const FORMAT: Format = Format {
                fixed_bits: <$s>::BITS - FRAC,
                frac_bits: FRAC,
            };
            fn from_wide(v: Wide) -> Self {
                Self::from_raw(v as $s)
            }
            fn to_wide(self) -> Wide {
                Wide::from(self.raw())
            }
            fn q_from_int(n: i32) -> Self {
                Self::from_int(n)
            }
            fn q_within_zero_tolerance(self) -> bool {
                self.is_within_zero_tolerance()
            }
            fn q_to_f64(self) -> f64 {
                self.to_f64()
            }
            fn q_from_f64_trunc(x: f64) -> Self {
                Self::from_f64_trunc(x)
            }
            fn q_one() -> Self {
                Self::ONE
            }
            fn q_zero() -> Self {
                Self::ZERO
            }
        }
    };
}

impl_oracle!(i16);
impl_oracle!(i32);

fn add<Q: Oracle>(a: Wide, b: Wide) -> Wide {
    (Q::from_wide(a) + Q::from_wide(b)).to_wide()
}
fn sub<Q: Oracle>(a: Wide, b: Wide) -> Wide {
    (Q::from_wide(a) - Q::from_wide(b)).to_wide()
}
fn mul<Q: Oracle>(a: Wide, b: Wide) -> Wide {
    (Q::from_wide(a) * Q::from_wide(b)).to_wide()
}
fn div<Q: Oracle>(a: Wide, b: Wide) -> Wide {
    (Q::from_wide(a) / Q::from_wide(b)).to_wide()
}
fn add_assign<Q: Oracle>(a: Wide, b: Wide) -> Wide {
    let mut q = Q::from_wide(a);
    q += Q::from_wide(b);
    q.to_wide()
}
fn compare<Q: Oracle>(a: Wide, b: Wide) -> Wide {
    let (x, y) = (Q::from_wide(a), Q::from_wide(b));
    Wide::from(x == y)
        | Wide::from(x != y) << 1
        | Wide::from(x < y) << 2
        | Wide::from(x > y) << 3
        | Wide::from(x <= y) << 4
        | Wide::from(x >= y) << 5
}
fn div_int<Q: Oracle>(a: Wide, n: Wide) -> Wide {
    Q::from_wide(a).div_int(n as i32).to_wide()
}

// === Fixed-capacity lists ===================================================

/// A short list of distinct values (the edge values and integer operands: at
/// most 31 candidates each).
#[derive(Clone, Copy)]
pub struct Values {
    buf: [Wide; 32],
    len: usize,
}

impl Values {
    fn new() -> Self {
        Values {
            buf: [0; 32],
            len: 0,
        }
    }

    fn push(&mut self, v: Wide) {
        if let Some(slot) = self.buf.get_mut(self.len) {
            *slot = v;
            self.len += 1;
        }
    }

    fn contains(&self, v: Wide) -> bool {
        self.buf[..self.len].contains(&v)
    }

    pub fn iter(&self) -> impl Iterator<Item = Wide> + Clone + '_ {
        self.buf[..self.len].iter().copied()
    }
}

// === Tally ==================================================================

/// How many failures a tally keeps for its report.
const KEEP: usize = 20;

/// One failed check.
#[derive(Clone, Copy)]
pub struct Failure {
    format: Format,
    op: &'static str,
    inputs: [Wide; 3],
    n_inputs: usize,
    actual: Wide,
    expected: Wide,
}

impl fmt::Display for Failure {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let f = self.format;
        write!(
            out,
            "Q{}.{} {}{:?}: got {}, expected {}",
            f.fixed_bits,
            f.frac_bits,
            self.op,
            &self.inputs[..self.n_inputs],
            self.actual,
            self.expected
        )
    }
}

#[derive(Default)]
pub struct Tally {
    pub checks: u64,
    pub failures: u64,
    first: [Option<Failure>; KEEP],
}

impl Tally {
    fn check(
        &mut self,
        f: Format,
        op: &'static str,
        inputs: &[Wide],
        actual: Wide,
        expected: Wide,
    ) {
        self.checks += 1;
        if actual != expected {
            self.failures += 1;
            let mut kept = [0; 3];
            let n_inputs = inputs.len().min(3);
            kept[..n_inputs].copy_from_slice(&inputs[..n_inputs]);
            self.keep(Failure {
                format: f,
                op,
                inputs: kept,
                n_inputs,
                actual,
                expected,
            });
        }
    }

    fn keep(&mut self, failure: Failure) {
        if let Some(slot) = self.first.iter_mut().find(|s| s.is_none()) {
            *slot = Some(failure);
        }
    }

    pub fn merge(&mut self, other: Tally) {
        self.checks += other.checks;
        self.failures += other.failures;
        for failure in other.first.into_iter().flatten() {
            self.keep(failure);
        }
    }

    /// The first failures (up to 20), in the order they happened.
    pub fn first_failures(&self) -> impl Iterator<Item = &Failure> {
        self.first.iter().flatten()
    }
}

// === Default sample =========================================================

/// Checks in the default sample (seed 0, scale 1), pinned so the device and
/// the host prove they ran all of it.
pub const CHECKS_16BIT: u64 = 105_792;
pub const CHECKS_32BIT: u64 = 134_485;

/// The 16-bit storage formats: Q8.8, Q1.15, Q7.9, Q15.1.
pub fn run_16bit(seed: u64, scale: u64) -> Tally {
    let mut t = Tally::default();
    t.merge(run_format::<QValue<i16, 8>>(seed, scale));
    t.merge(run_format::<QValue<i16, 15>>(seed, scale));
    t.merge(run_format::<QValue<i16, 9>>(seed, scale));
    t.merge(run_format::<QValue<i16, 1>>(seed, scale));
    t
}

/// The 32-bit storage formats: Q16.16, Q24.8, Q8.24, Q1.31, Q31.1.
pub fn run_32bit(seed: u64, scale: u64) -> Tally {
    let mut t = Tally::default();
    t.merge(run_format::<QValue<i32, 16>>(seed, scale));
    t.merge(run_format::<QValue<i32, 8>>(seed, scale));
    t.merge(run_format::<QValue<i32, 24>>(seed, scale));
    t.merge(run_format::<QValue<i32, 31>>(seed, scale));
    t.merge(run_format::<QValue<i32, 1>>(seed, scale));
    t
}

// === Inputs =================================================================

/// splitmix64: small, fixed-seed, identical on every platform.
#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The C++ oracle's per-format seed for a signed, Truncate, Wrap format.
fn seed_for(f: Format, seed: u64) -> u64 {
    let base =
        0x5EED_0000u64 ^ (u64::from(f.fixed_bits) << 24) ^ (u64::from(f.frac_bits) << 16) ^ 0x100;
    base ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Values where fixed-point arithmetic tends to break: zero, one LSB, one, one
/// half, the bounds and their neighbours, and the largest whole number.
fn edge_values(f: Format) -> Values {
    let one = f.one();
    let half = one / 2;
    let (lo, hi) = (f.min(), f.max());
    let clipped = ((1i64 << (f.fixed_bits - 1)) - 1) * one;
    let candidates = [
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        one,
        -one,
        half,
        -half,
        one + 1,
        one - 1,
        -one - 1,
        2 * one,
        -2 * one,
        one + half,
        -one - half,
        lo,
        lo + 1,
        lo + 2,
        hi,
        hi - 1,
        hi - 2,
        lo / 2,
        hi / 2,
        clipped,
        clipped + 1,
        clipped - 1,
        -clipped,
        -clipped - 1,
    ];
    let mut values = Values::new();
    for v in candidates {
        if f.in_range(v) && !values.contains(v) {
            values.push(v);
        }
    }
    values
}

fn random_value(f: Format, rng: &mut Rng) -> Wide {
    let (lo, hi) = (f.min(), f.max());
    match rng.next() % 4 {
        0 => {
            // Small magnitude: a few units either side of zero.
            let bits = (f.frac_bits + 3).min(f.width() - 1);
            let span = 1i64 << bits;
            let v = (rng.next() % span as u64) as Wide - span / 2;
            if f.in_range(v) { v } else { 0 }
        }
        1 => {
            // Near a bound.
            let offset = (rng.next() % 8) as Wide;
            if rng.next() & 1 != 0 {
                hi - offset
            } else {
                lo + offset
            }
        }
        // Uniform over the format's bit patterns.
        _ => wrap_to(rng.next() as i64, f.width()),
    }
}

pub fn int_operands(f: Format) -> Values {
    let p = |e: u32| 1i64 << e;
    let candidates = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        7,
        -7,
        100,
        -100,
        200,
        -200,
        1000,
        -1000,
        32767,
        -32768,
        65536,
        -65536,
        Wide::from(i32::MAX),
        Wide::from(i32::MIN),
        p(f.fixed_bits - 1) - 1,
        p(f.fixed_bits - 1),
        -p(f.fixed_bits - 1),
        -p(f.fixed_bits - 1) - 1,
        p(f.fixed_bits) - 1,
        p(f.fixed_bits),
    ];
    let mut values = Values::new();
    for v in candidates {
        if (Wide::from(i32::MIN)..=Wide::from(i32::MAX)).contains(&v) && !values.contains(v) {
            values.push(v);
        }
    }
    values
}

/// A random int operand: small, near a power of two, or anywhere in i32.
fn random_int(rng: &mut Rng) -> Wide {
    match rng.next() % 3 {
        0 => (rng.next() % 2049) as Wide - 1024,
        1 => {
            let magnitude = (1i64 << (rng.next() % 31)) + ((rng.next() % 3) as Wide - 1);
            if rng.next() & 1 != 0 {
                magnitude
            } else {
                -magnitude
            }
        }
        _ => wrap_to(rng.next() as i64, 32),
    }
}

// === Engine =================================================================

const RANDOM_PAIRS: u64 = 3000;
const RANDOM_INT_PAIRS: u64 = 1000;
const RANDOM_WORDS: u64 = 400;
const CHAINS: u64 = 100;
const CHAIN_LENGTH: u64 = 24;

pub fn check_pair<Q: Oracle>(t: &mut Tally, a: Wide, b: Wide) {
    let f = Q::FORMAT;
    let ab = [a, b];
    t.check(f, "add", &ab, add::<Q>(a, b), f.add(a, b));
    t.check(f, "addAssign", &ab, add_assign::<Q>(a, b), f.add(a, b));
    t.check(f, "sub", &ab, sub::<Q>(a, b), f.sub(a, b));
    t.check(f, "mul", &ab, mul::<Q>(a, b), f.mul(a, b));
    t.check(
        f,
        "compare",
        &ab,
        compare::<Q>(a, b),
        Format::compare_mask(a, b),
    );
    t.check(f, "div", &ab, div::<Q>(a, b), f.div(a, b));
}

pub fn check_int<Q: Oracle>(t: &mut Tally, a: Wide, n: Wide) {
    let f = Q::FORMAT;
    t.check(f, "divInt", &[a, n], div_int::<Q>(a, n), f.div_int(a, n));
}

/// Operations of one value: the Rust-only helpers have no C++ oracle
/// counterpart, so their reference is the definition each one documents.
pub fn check_value<Q: Oracle>(t: &mut Tally, a: Wide) {
    let f = Q::FORMAT;
    let q = Q::from_wide(a);
    t.check(f, "rawRoundTrip", &[a], q.to_wide(), a);
    t.check(
        f,
        "isWithinZeroTolerance",
        &[a],
        Wide::from(q.q_within_zero_tolerance()),
        Wide::from(f.within_zero_tolerance(a)),
    );
    // raw / 2^FRAC is exact in f64 for up to 32-bit storage.
    let real = q.q_to_f64();
    t.check(f, "toF64", &[a], (real * f.one() as f64) as Wide, a);
    t.check(
        f,
        "fromF64Trunc",
        &[a],
        Q::q_from_f64_trunc(real).to_wide(),
        a,
    );
}

fn check_representation<Q: Oracle>(t: &mut Tally, rng: &mut Rng, scale: u64) {
    let f = Q::FORMAT;
    t.check(f, "ZERO", &[], Q::q_zero().to_wide(), 0);
    t.check(f, "Scalar::ZERO", &[], <Q as Scalar>::ZERO.to_wide(), 0);
    t.check(f, "ONE", &[], Q::q_one().to_wide(), f.int_value(1));
    t.check(
        f,
        "Scalar::ONE",
        &[],
        <Q as Scalar>::ONE.to_wide(),
        f.int_value(1),
    );
    t.check(f, "default", &[], Q::default().to_wide(), 0);
    // Beyond FRAC, 2^-n truncates to zero.
    let beyond = [
        f.frac_bits + 1,
        f.frac_bits + 2,
        32,
        33,
        63,
        64,
        100,
        u32::MAX,
    ];
    for n in (0..=f.frac_bits).chain(beyond) {
        let expected = if n > f.frac_bits {
            0
        } else {
            f.wrap(1i64 << (f.frac_bits - n))
        };
        t.check(
            f,
            "pow2Neg",
            &[n.into()],
            Q::pow2_neg(n).to_wide(),
            expected,
        );
    }

    // The fixed operands, then RANDOM_WORDS * scale random ones (drawing and
    // checking interleaved: the checks draw nothing, so the sample is the same).
    let fixed = int_operands(f);
    let random = (0..RANDOM_WORDS * scale).map(|_| random_int(rng));
    for n in fixed.iter().chain(random) {
        t.check(
            f,
            "fromInt",
            &[n],
            Q::q_from_int(n as i32).to_wide(),
            f.int_value(n),
        );
    }
}

/// One value carried through a random sequence of operations. Each step starts
/// from the value the port actually produced, so a defect is reported at the
/// step that causes it instead of cascading through the rest.
fn check_chain<Q: Oracle>(t: &mut Tally, rng: &mut Rng) {
    let f = Q::FORMAT;
    let mut a = random_value(f, rng);
    for _ in 0..CHAIN_LENGTH {
        let b = random_value(f, rng);
        let n = random_int(rng);
        let (op, actual, expected) = match rng.next() % 5 {
            0 => ("chain add", add::<Q>(a, b), f.add(a, b)),
            1 => ("chain sub", sub::<Q>(a, b), f.sub(a, b)),
            2 => ("chain mul", mul::<Q>(a, b), f.mul(a, b)),
            3 => ("chain div", div::<Q>(a, b), f.div(a, b)),
            _ => ("chain divInt", div_int::<Q>(a, n), f.div_int(a, n)),
        };
        t.check(f, op, &[a, b, n], actual, expected);
        a = actual;
    }
}

/// Every check of one format, for a seed and scale (the soak's knobs; 0 and 1
/// by default).
pub fn run_format<Q: Oracle>(seed: u64, scale: u64) -> Tally {
    let f = Q::FORMAT;
    let mut rng = Rng(seed_for(f, seed));
    let mut t = Tally::default();

    // The single-value checks below take the edges and then 48 * scale random
    // values, drawn here, before the pairs. Keep the generator's state and
    // skip ahead, then replay those draws when they are checked: the sample is
    // the same, without a buffer that grows with the scale.
    let edges = edge_values(f);
    let values_rng = rng.clone();
    for _ in 0..48 * scale {
        random_value(f, &mut rng);
    }

    for a in edges.iter() {
        for b in edges.iter() {
            check_pair::<Q>(&mut t, a, b);
        }
    }
    for _ in 0..RANDOM_PAIRS * scale {
        let (a, b) = (random_value(f, &mut rng), random_value(f, &mut rng));
        check_pair::<Q>(&mut t, a, b);
    }

    let mut replay = values_rng;
    let random = (0..48 * scale).map(|_| random_value(f, &mut replay));
    for a in edges.iter().chain(random) {
        check_value::<Q>(&mut t, a);
    }

    let ints = int_operands(f);
    for a in edges.iter() {
        for n in ints.iter() {
            check_int::<Q>(&mut t, a, n);
        }
    }
    for _ in 0..RANDOM_INT_PAIRS * scale {
        let (a, n) = (random_value(f, &mut rng), random_int(&mut rng));
        check_int::<Q>(&mut t, a, n);
    }

    check_representation::<Q>(&mut t, &mut rng, scale);

    for _ in 0..CHAINS * scale {
        check_chain::<Q>(&mut t, &mut rng);
    }
    t
}
