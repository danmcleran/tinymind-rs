//! Kani proofs: properties checked for *every* input, not a sample. Run with
//! `cargo kani` (see CLAUDE.md for setup). Compiled only under `cfg(kani)`.
//!
//! Every harness also gets Kani's built-in checks for free: no panic, no
//! arithmetic overflow, no out-of-range shift, no out-of-bounds index.
//!
//! The specifications avoid the operations they check. "Wraps" is stated as a
//! congruence modulo 2^bits; "floors" and "truncates towards zero" are stated as
//! inequalities on a remainder, with the exact quotient as a free (`kani::any`)
//! variable constrained to be the only value that satisfies them. That keeps a
//! proof from restating the implementation, and keeps division (expensive for
//! the solver) out of the specification.

use crate::activation::{Activation, Relu, Sigmoid, Tanh};
use crate::lut::{SIGMOID_Q8_8, SIGMOID_Q16_16, TANH_Q8_8, TANH_Q16_16};
use crate::qformat::{Q8_8, Q16_16};

/// `x ≡ y (mod 2^bits)`.
fn congruent(x: i64, y: i64, bits: u32) -> bool {
    (x.wrapping_sub(y)).rem_euclid(1i64 << bits) == 0
}

/// Proofs for one format, generated per storage type and fractional-bit count.
macro_rules! format_proofs {
    ($module:ident, $s:ty, $frac:expr) => {
        mod $module {
            use super::congruent;
            use crate::qformat::{Fixed, QValue};

            type Q = QValue<$s, $frac>;
            const FRAC: u32 = $frac;
            const BITS: u32 = <$s>::BITS;

            fn any() -> Q {
                Q::from_raw(kani::any())
            }

            fn wide(q: Q) -> i64 {
                i64::from(q.raw())
            }

            fn bound_for_sign(a: Q) -> i64 {
                if a.raw() > 0 {
                    i64::from(<$s>::MAX)
                } else if a.raw() < 0 {
                    i64::from(<$s>::MIN)
                } else {
                    0
                }
            }

            #[kani::proof]
            fn add_and_sub_wrap() {
                let (a, b) = (any(), any());
                let mut c = a;
                c += b;
                assert!(congruent(wide(a + b), wide(a) + wide(b), BITS));
                assert!(congruent(wide(c), wide(a) + wide(b), BITS));
                assert!(congruent(wide(a - b), wide(a) - wide(b), BITS));
            }

            /// TruncatePolicy: the exact product floored to the format's
            /// resolution, then wrapped.
            #[kani::proof]
            fn mul_floors_then_wraps() {
                let (a, b) = (any(), any());
                // In i128 so the specification itself cannot overflow.
                let p = i128::from(wide(a)) * i128::from(wide(b));
                // f is the floor of p / 2^FRAC: the only integer with
                // f * 2^FRAC <= p < (f + 1) * 2^FRAC.
                let f: i64 = kani::any();
                let scaled = i128::from(f) * (1i128 << FRAC);
                kani::assume(scaled <= p && p - scaled < (1i128 << FRAC));
                assert!(congruent(wide(a * b), f, BITS));
            }

            /// Division never panics, for any operands (Kani's built-in overflow
            /// and divide-by-zero checks; no functional specification).
            #[kani::proof]
            fn div_never_panics() {
                let _ = any() / any();
                let _ = any().div_int(kani::any());
            }

            /// `from_int(n)` is n * 2^FRAC wrapped (the fixed part truncated).
            #[kani::proof]
            fn from_int_scales_then_wraps() {
                let n: i32 = kani::any();
                assert!(congruent(
                    wide(Q::from_int(n)),
                    i64::from(n) * (1i64 << FRAC),
                    BITS
                ));
            }

            /// `pow2_neg(n)` is 2^-n in the format: 2^(FRAC - n) raw, wrapped, or
            /// zero once 2^-n is below the resolution.
            #[kani::proof]
            fn pow2_neg_is_exact_or_zero() {
                let n: u32 = kani::any();
                let r = wide(Q::pow2_neg(n));
                if n > FRAC {
                    assert!(r == 0);
                } else {
                    assert!(congruent(r << n, 1i64 << FRAC, BITS + n));
                }
            }

            #[kani::proof]
            fn zero_tolerance_matches_definition() {
                let a = any();
                let tolerance = 1i64 << if FRAC > 7 { FRAC - 7 } else { 1 };
                assert!(a.is_within_zero_tolerance() == (wide(a).abs() < tolerance));
            }

            #[kani::proof]
            fn comparisons_are_on_the_raw_value() {
                let (a, b) = (any(), any());
                assert!((a < b) == (a.raw() < b.raw()));
                assert!((a == b) == (a.raw() == b.raw()));
            }
        }
    };
}

format_proofs!(q16_16, i32, 16);
format_proofs!(q8_8, i16, 8);
// Formats at the edges of the storage width: no fractional room, no integer room.
format_proofs!(q31_1, i32, 1);
format_proofs!(q1_31, i32, 31);
format_proofs!(q15_1, i16, 1);
format_proofs!(q1_15, i16, 15);

/// For every input, a table activation returns a value between its first and
/// last table entries (interpolation never overshoots), and the derivative
/// formula does not panic.
macro_rules! activation_proofs {
    ($name:ident, $q:ty, $s:ty, $act:ty, $table:expr) => {
        #[kani::proof]
        fn $name() {
            let x = <$q>::from_raw(kani::any());
            let y = <$act>::activate(x).raw();
            let first = $table[0] as $s;
            let last = $table[$table.len() - 1] as $s;
            assert!(first <= y && y <= last);
            let _ = <$act>::derivative(<$q>::from_raw(kani::any()));
        }
    };
}

activation_proofs!(tanh_q16_16_in_range, Q16_16, i32, Tanh, TANH_Q16_16);
activation_proofs!(
    sigmoid_q16_16_in_range,
    Q16_16,
    i32,
    Sigmoid,
    SIGMOID_Q16_16
);
activation_proofs!(tanh_q8_8_in_range, Q8_8, i16, Tanh, TANH_Q8_8);
activation_proofs!(sigmoid_q8_8_in_range, Q8_8, i16, Sigmoid, SIGMOID_Q8_8);

/// ReLU: max(0, x), derivative 0 or 1, for every input in both formats.
#[kani::proof]
fn relu_definition() {
    let x = Q16_16::from_raw(kani::any());
    let y = Relu::activate(x);
    assert!(y.raw() == x.raw().max(0));
    let d = Relu::derivative(y);
    assert!(d == Q16_16::ZERO || d == Q16_16::ONE);
    let x = Q8_8::from_raw(kani::any());
    assert!(Relu::activate(x).raw() == x.raw().max(0));
}
