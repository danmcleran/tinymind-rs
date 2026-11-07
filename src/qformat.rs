//! Signed fixed-point Q-format arithmetic, ported from tinymind `cpp/qformat.hpp`.
//!
//! The C++ `QValue<FixedBits, FractionalBits, IsSigned, RoundingPolicy, SaturatePolicy>`
//! picks its storage type from `FixedBits + FractionalBits` at compile time
//! (`QTypeChooser`). Stable Rust cannot compute a type from const-generic
//! arithmetic (that needs the nightly `generic_const_exprs` feature), so this port
//! fixes the storage at `i32` and is generic only over the number of fractional
//! bits: `QValue<16>` is Q16.16, `QValue<24>` would be Q8.24, and so on.
//!
//! Only tinymind's default policies are ported: `TruncatePolicy` (multiplication
//! drops the low bits) and `WrapPolicy` (addition and subtraction wrap modulo 2^32).

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Sub};

/// A signed 32-bit fixed-point value with `FRAC` fractional bits.
///
/// The derived `PartialOrd`/`Ord` compare the raw `i32`, which is exactly what the
/// C++ comparison operators do (they compare `getValue()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QValue<const FRAC: u32>(i32);

/// The format used by the XOR example: 16 integer bits (including sign), 16 fractional bits.
pub type Q16_16 = QValue<16>;

impl<const FRAC: u32> QValue<FRAC> {
    /// Compile-time format check, evaluated once per instantiated `FRAC`. The C++
    /// library does the same job with `static_assert`.
    const FORMAT_OK: () = assert!(
        FRAC >= 1 && FRAC <= 31,
        "QValue needs 1..=31 fractional bits"
    );

    pub const FRACTIONAL_BITS: u32 = FRAC;
    pub const FIXED_BITS: u32 = 32 - FRAC;
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self::from_int(1);

    /// Build from raw LSBs (C++ `QValue(FullWidthFieldType)` / `setValue(raw)`).
    pub const fn from_raw(raw: i32) -> Self {
        let () = Self::FORMAT_OK;
        Self(raw)
    }

    /// Build from a whole number (C++ `QValue(fixedPart, 0)`). The fixed part is
    /// truncated to `FIXED_BITS`, as the C++ `fromParts` does, which is the same as
    /// shifting left and letting the high bits fall off.
    pub const fn from_int(n: i32) -> Self {
        let () = Self::FORMAT_OK;
        Self(((n as u32) << FRAC) as i32)
    }

    /// `2^-n`, used for tinymind's default learning/momentum/acceleration rates.
    pub const fn pow2_neg(n: u32) -> Self {
        Self::from_raw(1 << (FRAC - n))
    }

    /// Raw LSBs (C++ `getValue()`).
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Real-number view, computed as the C++ example does: `raw / 2^FRAC` in `f64`.
    pub fn to_f64(self) -> f64 {
        f64::from(self.0) / f64::from(1u32 << FRAC)
    }

    /// Truncating conversion from `f64` (C++ `setValue(static_cast<int32_t>(x * 2^FRAC))`).
    pub fn from_f64_trunc(x: f64) -> Self {
        Self::from_raw((x * f64::from(1u32 << FRAC)) as i32)
    }

    /// Divide by a whole number, truncating toward zero (C++ `QValue::operator/=(int)`).
    /// Note this is *not* the same as dividing by `QValue::from_int(n)`.
    pub const fn div_int(self, n: i32) -> Self {
        if n == 0 {
            return Self::div_by_zero(self.0);
        }
        Self((self.0 as i64 / n as i64) as i32)
    }

    /// C++ `ZeroToleranceCalculator`: true when `|raw| < 1 << (FRAC - 7)` for
    /// formats with more than 7 fractional bits (512 LSBs for Q16.16), else
    /// `|raw| < 2`.
    pub const fn is_within_zero_tolerance(self) -> bool {
        let shift = if FRAC > 7 { FRAC - 7 } else { 1 };
        let tolerance = 1i32 << shift;
        self.0 == 0 || (self.0 < tolerance && self.0 > -tolerance)
    }

    /// The bound a quotient by zero saturates to: max for a positive dividend,
    /// min for a negative one, 0 for 0 (C++ `WrapPolicy` division case).
    const fn div_by_zero(dividend: i32) -> Self {
        if dividend > 0 {
            Self(i32::MAX)
        } else if dividend < 0 {
            Self(i32::MIN)
        } else {
            Self(0)
        }
    }
}

impl<const FRAC: u32> Add for QValue<FRAC> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // C++ adds in the unsigned counterpart to get defined wrap-around; Rust
        // says what it means directly.
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl<const FRAC: u32> AddAssign for QValue<FRAC> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const FRAC: u32> Sub for QValue<FRAC> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl<const FRAC: u32> Mul for QValue<FRAC> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // Widen to i64 (C++ MultiplicationResultFullWidthValueType), truncate
        // (arithmetic shift right), then wrap back to the 32-bit word.
        let product = i64::from(self.0) * i64::from(rhs.0);
        Self(((product + ((1i64 << FRAC) >> 1)) >> FRAC) as i32)
    }
}

impl<const FRAC: u32> Div for QValue<FRAC> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Self::div_by_zero(self.0);
        }
        // Scale the dividend up by 2^FRAC in i64, divide (truncating toward zero,
        // like C++), then wrap back to 32 bits.
        let dividend = i64::from(self.0) * (1i64 << FRAC);
        Self((dividend / i64::from(rhs.0)) as i32)
    }
}

/// Prints the raw LSBs, matching the C++ `operator<<` (not the real value).
impl<const FRAC: u32> fmt::Display for QValue<FRAC> {
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
    const ZERO: Self;
    const ONE: Self;
    /// `2^-n`
    fn pow2_neg(n: u32) -> Self;
    /// Divide by a whole number (C++ `operator/=(int)`).
    fn div_int(self, n: i32) -> Self;
}

impl<const FRAC: u32> Fixed for QValue<FRAC> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    fn pow2_neg(n: u32) -> Self {
        Self::pow2_neg(n)
    }
    fn div_int(self, n: i32) -> Self {
        self.div_int(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(Q16_16::ONE.raw(), 65536);
        assert_eq!(Q16_16::from_int(-1).raw(), -65536);
        assert_eq!(Q16_16::pow2_neg(2).raw(), 16384);
    }

    #[test]
    fn multiply_truncates_toward_negative_infinity() {
        // -1 LSB * 0.5 = -0.5 LSB, which an arithmetic shift floors to -1 LSB.
        let r = Q16_16::from_raw(-1) * Q16_16::pow2_neg(1);
        assert_eq!(r.raw(), -1);
        let r = Q16_16::from_raw(1) * Q16_16::pow2_neg(1);
        assert_eq!(r.raw(), 0);
    }

    #[test]
    fn divide_truncates_toward_zero_and_saturates_on_zero() {
        assert_eq!((Q16_16::from_raw(-3) / Q16_16::from_int(2)).raw(), -1);
        assert_eq!((Q16_16::ONE / Q16_16::ZERO).raw(), i32::MAX);
        assert_eq!((Q16_16::from_int(-1) / Q16_16::ZERO).raw(), i32::MIN);
        assert_eq!((Q16_16::ZERO / Q16_16::ZERO).raw(), 0);
    }

    #[test]
    fn add_wraps() {
        let r = Q16_16::from_raw(i32::MAX) + Q16_16::from_raw(1);
        assert_eq!(r.raw(), i32::MIN);
    }

    #[test]
    fn zero_tolerance() {
        assert!(Q16_16::from_raw(511).is_within_zero_tolerance());
        assert!(Q16_16::from_raw(-511).is_within_zero_tolerance());
        assert!(!Q16_16::from_raw(512).is_within_zero_tolerance());
    }
}
