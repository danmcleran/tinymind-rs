//! Signed fixed-point Q-format arithmetic, ported from tinymind `cpp/qformat.hpp`.
//!
//! The C++ `QValue<FixedBits, FractionalBits, IsSigned, RoundingPolicy, SaturatePolicy>`
//! picks its storage type from `FixedBits + FractionalBits` at compile time
//! (`QTypeChooser`). Stable Rust cannot compute a type from const-generic
//! arithmetic (that needs the nightly `generic_const_exprs` feature), so here the
//! storage type is an explicit parameter: `QValue<i32, 16>` is Q16.16 and
//! `QValue<i16, 8>` is Q8.8. The methods and operators are generated for each
//! supported storage type by a macro, each with the wider type C++ uses for
//! products and quotients (`i64` for `i32`, `i32` for `i16`).
//!
//! Only tinymind's default policies are ported: `TruncatePolicy` (multiplication
//! drops the low bits) and `WrapPolicy` (addition and subtraction wrap modulo the
//! storage width).

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Sub};

/// A signed fixed-point value stored in `S` (`i16` or `i32`) with `FRAC`
/// fractional bits.
///
/// The derived `PartialOrd`/`Ord` compare the raw integer, which is exactly what
/// the C++ comparison operators do (they compare `getValue()`).
///
/// `FRAC` must be at least 1 and less than the storage width; anything else is a
/// compile-time error wherever the format's arithmetic is used:
///
/// ```compile_fail
/// // 40 fractional bits do not fit in i32 storage.
/// let x = tinymind::QValue::<i32, 40>::default();
/// let _ = x.to_f64();
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QValue<S, const FRAC: u32>(S);

/// Q16.16: 16 integer bits (including sign), 16 fractional bits, `i32` storage.
pub type Q16_16 = QValue<i32, 16>;

/// Q8.8: 8 integer bits (including sign), 8 fractional bits, `i16` storage.
pub type Q8_8 = QValue<i16, 8>;

/// Prints the raw LSBs, matching the C++ `operator<<` (not the real value).
impl<S: fmt::Display, const FRAC: u32> fmt::Display for QValue<S, FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The arithmetic a network needs from its value type. This is the Rust
/// counterpart of the implicit "ValueType concept" that the C++ templates rely on:
/// in C++ a missing operator is a template instantiation error, here it is a
/// trait-bound error at the call site.
pub trait Fixed:
    Copy
    + Default
    + PartialOrd
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// 0.0
    const ZERO: Self;
    /// 1.0
    const ONE: Self;
    /// `2^-n`
    fn pow2_neg(n: u32) -> Self;
    /// Divide by a whole number (C++ `operator/=(int)`).
    fn div_int(self, n: i32) -> Self;
}

/// Square root of a non-negative value (C++ `SquareRootApproximation`):
/// integer Newton iteration on `raw << FRAC` in the wide type, which gives the
/// floor of the exact root in raw units. Zero and negative values give zero. The
/// Adam and RMSprop optimizers use it.
pub trait Sqrt {
    /// `floor(sqrt(self))` in this format; 0 for `self <= 0`.
    fn sqrt(self) -> Self;
}

/// Generates everything that depends on the storage type: `$s` is the storage,
/// `$wide` the C++ `MultiplicationResultFullWidthValueType` /
/// `DivisionResultFullWidthValueType` for it (twice the width).
macro_rules! impl_qvalue {
    ($s:ty, $wide:ty) => {
        impl<const FRAC: u32> QValue<$s, FRAC> {
            /// Compile-time format check, evaluated once per instantiated format.
            /// The C++ library does the same job with `static_assert`. Every method
            /// whose arithmetic depends on `FRAC` evaluates it, so an invalid format
            /// such as `QValue<i32, 40>` fails to compile wherever it is used, even
            /// when the value came from `Default`.
            const FORMAT_OK: () = assert!(
                FRAC >= 1 && FRAC < <$s>::BITS,
                "QValue needs at least 1 fractional bit and fewer than the storage width"
            );

            /// Number of fractional bits (`FRAC`).
            pub const FRACTIONAL_BITS: u32 = FRAC;
            /// Number of integer bits, including the sign bit (storage width - `FRAC`).
            pub const FIXED_BITS: u32 = <$s>::BITS - FRAC;
            /// 0.0
            pub const ZERO: Self = Self::from_raw(0);
            /// 1.0. When `FRAC` is one less than the storage width (Q1.15, Q1.31)
            /// there are no integer bits besides the sign, 1.0 is not
            /// representable, and this wraps to -1.0 (the storage type's minimum).
            pub const ONE: Self = Self::from_int(1);

            /// Build from raw LSBs (C++ `QValue(FullWidthFieldType)` / `setValue(raw)`).
            pub const fn from_raw(raw: $s) -> Self {
                let () = Self::FORMAT_OK;
                Self(raw)
            }

            /// Build from a whole number (C++ `QValue(fixedPart, 0)`). The fixed part
            /// is truncated to `FIXED_BITS`, as the C++ `fromParts` does, which is the
            /// same as shifting left and letting the high bits fall off: in Q8.8, 200
            /// becomes -56.
            pub const fn from_int(n: i32) -> Self {
                let () = Self::FORMAT_OK;
                Self(((n as u32) << FRAC) as $s)
            }

            /// `2^-n`, used for tinymind's default learning/momentum/acceleration rates.
            ///
            /// For `n > FRAC`, `2^-n` is below the format's resolution and the result
            /// is [`ZERO`](Self::ZERO): the value truncated to the format, as
            /// TruncatePolicy rounds every other result. `pow2_neg(0)` is
            /// [`ONE`](Self::ONE), with the same caveat for formats without integer bits.
            pub const fn pow2_neg(n: u32) -> Self {
                if n > FRAC {
                    return Self::from_raw(0);
                }
                Self::from_raw((1 as $s) << (FRAC - n))
            }

            /// Raw LSBs (C++ `getValue()`).
            pub const fn raw(self) -> $s {
                self.0
            }

            /// Real-number view, computed as the C++ example does: `raw / 2^FRAC` in `f64`.
            pub fn to_f64(self) -> f64 {
                let () = Self::FORMAT_OK;
                f64::from(self.0) / f64::from(1u32 << FRAC)
            }

            /// Truncating conversion from `f64` (C++
            /// `setValue(static_cast<FullWidthValueType>(x * 2^FRAC))`). Out-of-range
            /// input saturates; in C++ it is undefined behaviour.
            pub fn from_f64_trunc(x: f64) -> Self {
                Self::from_raw((x * f64::from(1u32 << FRAC)) as $s)
            }

            /// Divide by a whole number, truncating toward zero (C++
            /// `QValue::operator/=(int)`). This is *not* the same as dividing by
            /// `QValue::from_int(n)`.
            pub const fn div_int(self, n: i32) -> Self {
                if n == 0 {
                    return Self::div_by_zero(self.0);
                }
                // C++ applyInteger works in int64 for every format up to 32 bits.
                Self((self.0 as i64 / n as i64) as $s)
            }

            /// C++ `ZeroToleranceCalculator`: true when `|raw| < 1 << (FRAC - 7)` for
            /// formats with more than 7 fractional bits (512 LSBs for Q16.16, 2 for
            /// Q8.8), else `|raw| < 2`.
            pub const fn is_within_zero_tolerance(self) -> bool {
                let () = Self::FORMAT_OK;
                let shift = if FRAC > 7 { FRAC - 7 } else { 1 };
                let tolerance = (1 as $s) << shift;
                self.0 == 0 || (self.0 < tolerance && self.0 > -tolerance)
            }

            /// The bound a quotient by zero saturates to: max for a positive dividend,
            /// min for a negative one, 0 for 0 (C++ `WrapPolicy` division case).
            const fn div_by_zero(dividend: $s) -> Self {
                if dividend > 0 {
                    Self(<$s>::MAX)
                } else if dividend < 0 {
                    Self(<$s>::MIN)
                } else {
                    Self(0)
                }
            }
        }

        impl<const FRAC: u32> Add for QValue<$s, FRAC> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                // C++ adds in the unsigned counterpart to get defined wrap-around;
                // Rust says what it means directly.
                Self(self.0.wrapping_add(rhs.0))
            }
        }

        impl<const FRAC: u32> AddAssign for QValue<$s, FRAC> {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<const FRAC: u32> Sub for QValue<$s, FRAC> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0.wrapping_sub(rhs.0))
            }
        }

        impl<const FRAC: u32> Mul for QValue<$s, FRAC> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                let () = Self::FORMAT_OK;
                // Widen, truncate (arithmetic shift right floors toward -inf), then
                // wrap back to the storage width.
                let product = <$wide>::from(self.0) * <$wide>::from(rhs.0);
                Self((product >> FRAC) as $s)
            }
        }

        impl<const FRAC: u32> Div for QValue<$s, FRAC> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                let () = Self::FORMAT_OK;
                if rhs.0 == 0 {
                    return Self::div_by_zero(self.0);
                }
                // Scale the dividend up by 2^FRAC in the wide type, divide
                // (truncating toward zero, like C++), then wrap back.
                let dividend = <$wide>::from(self.0) * ((1 as $wide) << FRAC);
                Self((dividend / <$wide>::from(rhs.0)) as $s)
            }
        }

        impl<const FRAC: u32> Sqrt for QValue<$s, FRAC> {
            fn sqrt(self) -> Self {
                let () = Self::FORMAT_OK;
                if self.0 <= 0 {
                    return Self(0);
                }
                // n < 2^(storage bits - 1 + FRAC), so it and every sum below fit
                // the wide type, and x stays >= 1.
                let n = <$wide>::from(self.0) << FRAC;
                let mut x = n;
                let mut y = (x + 1) / 2;
                while y < x {
                    x = y;
                    // x >= 1 here, but the compiler cannot prove it; the
                    // fallback is unreachable and keeps the function panic-free.
                    y = (x + n.checked_div(x).unwrap_or(0)) / 2;
                }
                // n < 2^(2 * storage bits - 2), so the root is below 2^(storage
                // bits - 1) and the narrowing (the C++ static_cast) never truncates.
                Self(x as $s)
            }
        }

        impl<const FRAC: u32> Fixed for QValue<$s, FRAC> {
            const ZERO: Self = Self::ZERO;
            const ONE: Self = Self::ONE;
            fn pow2_neg(n: u32) -> Self {
                Self::pow2_neg(n)
            }
            fn div_int(self, n: i32) -> Self {
                self.div_int(n)
            }
        }
    };
}

impl_qvalue!(i32, i64);
impl_qvalue!(i16, i32);

#[cfg(test)]
mod tests {
    use super::*;

    /// `sqrt` is the floor of the exact root of `raw << FRAC`: r*r <= n < (r+1)^2.
    fn check_floor_root(raw: i64, frac: u32, root: i64) {
        let n = raw << frac;
        assert!(
            root * root <= n && (root + 1) * (root + 1) > n,
            "raw {raw}: root {root}"
        );
    }

    #[test]
    fn sqrt_every_q8_8_value() {
        for raw in i16::MIN..=i16::MAX {
            let r = i64::from(Q8_8::from_raw(raw).sqrt().raw());
            if raw <= 0 {
                assert_eq!(r, 0);
            } else {
                check_floor_root(i64::from(raw), 8, r);
            }
        }
        assert_eq!(Q8_8::from_int(4).sqrt(), Q8_8::from_int(2));
    }

    #[test]
    fn sqrt_q16_16_sample_and_edges() {
        let mut x: u32 = 1;
        let edges = [1, 2, 3, 65_535, 65_536, 65_537, i32::MAX - 1, i32::MAX];
        for raw in edges.into_iter().chain((0..100_000).map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 1) as i32
        })) {
            check_floor_root(
                i64::from(raw),
                16,
                i64::from(Q16_16::from_raw(raw).sqrt().raw()),
            );
        }
        assert_eq!(Q16_16::from_int(9).sqrt(), Q16_16::from_int(3));
        assert_eq!(Q16_16::from_int(-9).sqrt(), Q16_16::ZERO);
        // Formats with one integer bit: the root of a value near 1 is near 1 and
        // does not fit; it wraps as the C++ static_cast does, without panicking.
        let _ = QValue::<i32, 31>::from_raw(i32::MAX).sqrt();
        let _ = QValue::<i16, 15>::from_raw(i16::MAX).sqrt();
    }

    #[test]
    fn constants() {
        assert_eq!(Q16_16::ONE.raw(), 65536);
        assert_eq!(Q16_16::from_int(-1).raw(), -65536);
        assert_eq!(Q16_16::pow2_neg(2).raw(), 16384);
        assert_eq!(Q8_8::ONE.raw(), 256);
        assert_eq!(Q8_8::from_int(-1).raw(), -256);
        assert_eq!(Q8_8::pow2_neg(4).raw(), 16);
        assert_eq!((Q8_8::FIXED_BITS, Q8_8::FRACTIONAL_BITS), (8, 8));
    }

    #[test]
    fn from_int_truncates_the_fixed_part() {
        // tinymind's own example (qformat.hpp, applyInteger comment): Q8.8 200 -> -56.
        assert_eq!(Q8_8::from_int(200), Q8_8::from_int(-56));
        assert_eq!(Q8_8::from_int(127).raw(), 127 * 256);
    }

    #[test]
    fn multiply_truncates_toward_negative_infinity() {
        // -1 LSB * 0.5 = -0.5 LSB, which an arithmetic shift floors to -1 LSB.
        let r = Q16_16::from_raw(-1) * Q16_16::pow2_neg(1);
        assert_eq!(r.raw(), -1);
        let r = Q16_16::from_raw(1) * Q16_16::pow2_neg(1);
        assert_eq!(r.raw(), 0);
        assert_eq!((Q8_8::from_raw(-1) * Q8_8::pow2_neg(1)).raw(), -1);
        assert_eq!((Q8_8::from_raw(1) * Q8_8::pow2_neg(1)).raw(), 0);
    }

    #[test]
    fn multiply_wraps_to_the_storage_width() {
        // 100.0 * 2.0 = 200.0 does not fit Q8.8; the product wraps to -56.0.
        let r = Q8_8::from_int(100) * Q8_8::from_int(2);
        assert_eq!(r, Q8_8::from_int(-56));
    }

    #[test]
    fn divide_truncates_toward_zero_and_saturates_on_zero() {
        assert_eq!((Q16_16::from_raw(-3) / Q16_16::from_int(2)).raw(), -1);
        assert_eq!((Q16_16::ONE / Q16_16::ZERO).raw(), i32::MAX);
        assert_eq!((Q16_16::from_int(-1) / Q16_16::ZERO).raw(), i32::MIN);
        assert_eq!((Q16_16::ZERO / Q16_16::ZERO).raw(), 0);
        assert_eq!((Q8_8::from_raw(-3) / Q8_8::from_int(2)).raw(), -1);
        assert_eq!((Q8_8::ONE / Q8_8::ZERO).raw(), i16::MAX);
        assert_eq!((Q8_8::from_int(-1) / Q8_8::ZERO).raw(), i16::MIN);
        assert_eq!(Q8_8::ONE.div_int(0).raw(), i16::MAX);
    }

    #[test]
    fn pow2_neg_range() {
        assert_eq!(Q16_16::pow2_neg(16).raw(), 1);
        assert_eq!(Q16_16::pow2_neg(0), Q16_16::ONE);
        assert_eq!(Q8_8::pow2_neg(8).raw(), 1);
    }

    #[test]
    fn pow2_neg_below_resolution_truncates_to_zero() {
        for n in [17, 20, 32, 63, 64, u32::MAX] {
            assert_eq!(
                Q16_16::pow2_neg(core::hint::black_box(n)),
                Q16_16::ZERO,
                "n = {n}"
            );
        }
        assert_eq!(Q8_8::pow2_neg(9), Q8_8::ZERO);
    }

    #[test]
    fn add_wraps() {
        let r = Q16_16::from_raw(i32::MAX) + Q16_16::from_raw(1);
        assert_eq!(r.raw(), i32::MIN);
        let r = Q8_8::from_raw(i16::MAX) + Q8_8::from_raw(1);
        assert_eq!(r.raw(), i16::MIN);
    }

    #[test]
    fn zero_tolerance() {
        assert!(Q16_16::from_raw(511).is_within_zero_tolerance());
        assert!(Q16_16::from_raw(-511).is_within_zero_tolerance());
        assert!(!Q16_16::from_raw(512).is_within_zero_tolerance());
        assert!(Q8_8::from_raw(1).is_within_zero_tolerance());
        assert!(Q8_8::from_raw(-1).is_within_zero_tolerance());
        assert!(!Q8_8::from_raw(2).is_within_zero_tolerance());
        assert!(!Q8_8::from_raw(-2).is_within_zero_tolerance());
    }
}
