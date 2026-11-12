// The int8 softmax sweep, shared by tests/softmax_sweep.rs (on the host,
// against tinymind (C++)'s output, calibrating with build_softmax) and
// bench/src/sweeps.rs (on Cortex-M, against the same golden, with the C++'s
// parameters read from it). no_std and allocation-free.

use core::fmt::Write;

use tinymind::int8::QSoftmax;

/// What the sweep needs from `int8::calibration` (std only): on the host the
/// real `build_softmax`, on the device the C++'s parameters read from the
/// golden.
pub trait Calibrate {
    /// `build_softmax(input_scale, beta)`.
    fn softmax(&mut self, input_scale: f64, beta: f64) -> QSoftmax;
}

/// 2^e as an f64 (e in -1022..=1023), exactly, without `powi` (std only).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

struct Logits(u32);

impl Logits {
    fn next(&mut self, row: usize) -> i8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = self.0 >> 24;
        if row % 4 == 3 {
            (96 + (v & 7)) as i8
        } else {
            (v as i32 - 128) as i8
        }
    }
}

fn sweep_rows<const N: usize>(softmax: &QSoftmax, logits: &mut Logits, mut hash: u32) -> u32 {
    for r in 0..32 {
        let input: [i8; N] = core::array::from_fn(|_| logits.next(r));
        for y in softmax.forward(&input) {
            hash ^= u32::from(y as u8);
            hash = hash.wrapping_mul(0x0100_0193);
        }
    }
    hash
}

/// Write the sweep's report (tests/golden/softmax/stdout.txt).
pub fn report(out: &mut impl Write, cal: &mut impl Calibrate) {
    let betas = [1.0, 0.5, 2.5];
    for e in -14i32..=6 {
        for j in 0..8 {
            for (b, &beta) in betas.iter().enumerate() {
                let scale = (1.0 + f64::from(j) / 8.0) * pow2(e);
                let p = cal.softmax(scale, beta);
                let mut logits = Logits(((e + 14) * 64 + j * 4 + b as i32) as u32);
                let mut hash = 0x811C_9DC5;
                hash = sweep_rows::<1>(&p, &mut logits, hash);
                hash = sweep_rows::<2>(&p, &mut logits, hash);
                hash = sweep_rows::<3>(&p, &mut logits, hash);
                hash = sweep_rows::<10>(&p, &mut logits, hash);
                hash = sweep_rows::<12>(&p, &mut logits, hash);
                hash = sweep_rows::<64>(&p, &mut logits, hash);
                hash = sweep_rows::<511>(&p, &mut logits, hash);
                writeln!(
                    out,
                    "e={e} j={j} beta={b} params={},{},{} hash=0x{hash:08X}",
                    p.input_multiplier, p.input_left_shift, p.diff_min
                )
                .unwrap();
            }
        }
    }
}
