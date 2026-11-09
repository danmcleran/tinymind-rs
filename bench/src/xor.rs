//! tinymind's xor example trained on the board: 20,000 SGD-with-momentum steps
//! of a 2-4-1 tanh/sigmoid network. The device hashes the exact bytes of the
//! nn_fixed_xor.txt the example writes (64-bit FNV-1a, as tests/xor_golden.rs
//! does), so a match proves every weight of every step is bit-exact with the
//! C++ and host runs.

use core::fmt::{self, Display, Write};
use core::hint::black_box;

use tinymind::rand::GlibcRand;
use tinymind::{Activation, Scalar, MultilayerPerceptron, Q8_8, Q16_16, Sigmoid, Tanh};

use crate::board::{self, Uart};
use crate::{REPS, median};

const STEPS: u32 = 20_000;

/// The golden trajectory: byte length, line count, 64-bit FNV-1a
/// (tests/xor_golden.rs).
pub struct Expect {
    len: u64,
    lines: u64,
    fnv: u64,
}
#[cfg(feature = "xor-q16")]
pub const Q16_EXPECT: Expect = Expect {
    len: 2_587_725,
    lines: 20_001,
    fnv: 0x0f6d_21c3_e3a5_076b,
};
#[cfg(feature = "xor-q8")]
pub const Q8_EXPECT: Expect = Expect {
    len: 2_383_566,
    lines: 20_001,
    fnv: 0x189a_6e57_c8c2_2955,
};

/// What the example needs from a format (examples/common/xor_driver.rs).
pub trait Format: Scalar + Display {
    fn from_int(n: i32) -> Self;
    fn within_zero_tolerance(self) -> bool;
    fn random_weight(rand: i32) -> Self;
}

impl Format for Q16_16 {
    fn from_int(n: i32) -> Self {
        Q16_16::from_int(n)
    }
    fn within_zero_tolerance(self) -> bool {
        self.is_within_zero_tolerance()
    }
    fn random_weight(rand: i32) -> Self {
        let half = Q16_16::ONE.raw() / 2;
        Q16_16::from_raw(rand % (2 * half + 1) - half)
    }
}

impl Format for Q8_8 {
    fn from_int(n: i32) -> Self {
        Q8_8::from_int(n)
    }
    fn within_zero_tolerance(self) -> bool {
        self.is_within_zero_tolerance()
    }
    fn random_weight(rand: i32) -> Self {
        let half = i32::from(Q8_8::ONE.raw() / 2);
        Q8_8::from_raw((rand % (2 * half + 1) - half) as i16)
    }
}

type Net<V> = MultilayerPerceptron<V, Tanh, Sigmoid, 2, 4, 1>;

/// 64-bit FNV-1a over everything written to it, with byte and line counts.
struct Fnv64 {
    h: u64,
    len: u64,
    lines: u64,
}

impl Write for Fnv64 {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            self.h ^= u64::from(b);
            self.h = self.h.wrapping_mul(0x0000_0100_0000_01b3);
            self.len += 1;
            self.lines += u64::from(b == b'\n');
        }
        Ok(())
    }
}

fn header(w: &mut impl Write) -> fmt::Result {
    for i in 0..2 {
        for h in 0..4 {
            write!(w, "Input{i}{h}Weight,")?;
        }
    }
    for h in 0..4 {
        write!(w, "InputBias0{h}Weight,")?;
    }
    for h in 0..4 {
        write!(w, "Hidden0{h}0Weight,")?;
    }
    writeln!(w, "Hidden0Bias0Weight,Expected0,Learned0,Error")
}

/// Train for STEPS steps; `line` sees each step's properties if given.
fn train<V: Format>(mut line: Option<&mut Fnv64>) -> Net<V>
where
    Tanh: Activation<V>,
    Sigmoid: Activation<V>,
{
    let mut init = GlibcRand::new(1);
    let mut nn = Net::<V>::new(|| V::random_weight(init.next_i32()));
    let mut rng = GlibcRand::new(7);
    for _ in 0..STEPS {
        let (x, y) = (rng.next_i32() & 1, rng.next_i32() & 1);
        let target = [V::from_int(x ^ y)];
        nn.feed_forward(&[V::from_int(x), V::from_int(y)]);
        let error = nn.calculate_error(&target);
        if !error.within_zero_tolerance() {
            nn.train(&target);
        }
        if let Some(w) = line.as_deref_mut() {
            for row in nn.input_weights() {
                for v in row {
                    let _ = write!(w, "{v},");
                }
            }
            for v in nn.input_bias_weights() {
                let _ = write!(w, "{v},");
            }
            for row in nn.hidden_weights() {
                for v in row {
                    let _ = write!(w, "{v},");
                }
            }
            for v in nn.hidden_bias_weights() {
                let _ = write!(w, "{v},");
            }
            let _ = writeln!(w, "{},{},{}", target[0], nn.learned_values()[0], error);
        }
    }
    nn
}

pub fn run<V: Format>(out: &mut Uart, name: &str, expect: Expect)
where
    Tanh: Activation<V>,
    Sigmoid: Activation<V>,
{
    let _ = writeln!(out, "name={name}");

    // Training speed: the plain loop, no hashing.
    let a = board::cycles();
    let mut nn = black_box(train::<V>(None));
    let total = board::cycles().wrapping_sub(a);
    let _ = writeln!(out, "train_steps={STEPS}\ntrain_cycles={total}");
    let _ = writeln!(out, "train_cycles_per_step={}", total / STEPS);

    // Inference speed of the trained network.
    let mut t = [0u32; REPS];
    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box([
            V::from_int((r & 1) as i32),
            V::from_int(((r >> 1) & 1) as i32),
        ]);
        let a = board::cycles();
        nn.feed_forward(&x);
        black_box(nn.learned_values());
        *slot = board::cycles().wrapping_sub(a);
    }
    let (min, med, max) = median(t);
    let _ = writeln!(out, "infer_min={min}\ninfer_med={med}\ninfer_max={max}");

    // Exactness: hash the trajectory the example would write.
    let mut hasher = Fnv64 {
        h: 0xcbf2_9ce4_8422_2325,
        len: 0,
        lines: 0,
    };
    let _ = header(&mut hasher);
    train::<V>(Some(&mut hasher));
    let _ = writeln!(
        out,
        "trajectory_bytes={}\ntrajectory_lines={}\ntrajectory_fnv64=0x{:016x}",
        hasher.len, hasher.lines, hasher.h
    );
    let _ = writeln!(out, "expect_fnv64=0x{:016x}", expect.fnv);
    let ok = hasher.h == expect.fnv && hasher.len == expect.len && hasher.lines == expect.lines;
    let _ = writeln!(out, "trajectory_match={}", u32::from(ok));
}
