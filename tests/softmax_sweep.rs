// QSoftmax and calibration::build_softmax against tinymind's C++
// (QSoftmaxFixedPoint1D, buildQSoftmaxFixedPointParams) over 504 calibrations
// (input scales (1 + j/8) * 2^e for e in -14..=6, betas 1, 0.5 and 2.5) and
// row lengths 1 to 511, 32 rows each. tests/golden/softmax/stdout.txt is the
// output of scripts/cpp/softmax_sweep.cpp (scripts/compare-with-cpp.sh rebuilds
// it and checks the golden is current).

use std::fmt::Write;

use tinymind::int8::QSoftmax;
use tinymind::int8::calibration::build_softmax;

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

#[test]
fn softmax_sweep_matches_cpp() {
    let betas = [1.0, 0.5, 2.5];
    let mut out = String::new();
    for e in -14i32..=6 {
        for j in 0..8 {
            for (b, &beta) in betas.iter().enumerate() {
                let scale = (1.0 + f64::from(j) / 8.0) * 2f64.powi(e);
                let p = build_softmax(scale, beta);
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
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/softmax/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
