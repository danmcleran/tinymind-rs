//! Activation policies, ported from tinymind `cpp/activationFunctions.hpp`,
//! `cpp/lookupTable.hpp` and `cpp/interpolate.hpp`.
//!
//! C++ policy classes with static member functions become zero-sized Rust types
//! implementing a trait. The trait is generic over the value type and implemented
//! only for `Q16_16`, because the lookup tables are format-specific: asking for
//! `Tanh` on another format is a compile error, as it is in C++ when the matching
//! `TINYMIND_USE_TANH_x_y` table is not compiled in.

use crate::lut::{NUMBER_OF_ACTIVATION_TABLE_VALUES, SIGMOID_Q16_16, TANH_Q16_16};
use crate::qformat::{Q16_16, QValue};

/// A neuron activation function and its derivative.
pub trait Activation<V> {
    /// The activation of the weighted input sum `x`.
    fn activate(x: V) -> V;
    /// Derivative expressed in terms of the activation's *output* `y`, the way
    /// tinymind computes it during back-propagation.
    fn derivative(y: V) -> V;
}

/// Hyperbolic tangent via the 96-entry lookup table.
pub struct Tanh;

/// Logistic sigmoid via the 96-entry lookup table.
pub struct Sigmoid;

impl Activation<Q16_16> for Tanh {
    fn activate(x: Q16_16) -> Q16_16 {
        lookup(x, &TANH_Q16_16)
    }

    fn derivative(y: Q16_16) -> Q16_16 {
        Q16_16::ONE - (y * y)
    }
}

impl Activation<Q16_16> for Sigmoid {
    fn activate(x: Q16_16) -> Q16_16 {
        lookup(x, &SIGMOID_Q16_16)
    }

    fn derivative(y: Q16_16) -> Q16_16 {
        y * (Q16_16::ONE - y)
    }
}

/// Table domain is [-6, 6] with knots 1/8 apart (`cpp/activation.hpp`).
const MIN_X_TABLE_VALUE: i32 = -6;
const MAX_X_TABLE_VALUE: i32 = 6;
const ACTIVATION_DELTA_SHIFT: u32 = 3;
const MAX_ACTIVATION_INDEX: usize = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;

/// Piecewise-linear table lookup (C++ `LookupTable::getValue`).
fn lookup<const FRAC: u32>(
    x: QValue<FRAC>,
    table: &[u32; NUMBER_OF_ACTIVATION_TABLE_VALUES],
) -> QValue<FRAC> {
    // The table entries are stored as u32 bit patterns; reinterpret as i32 raw LSBs.
    let entry = |i: usize| QValue::<FRAC>::from_raw(table[i] as i32);

    let min_x = MIN_X_TABLE_VALUE << FRAC;
    let max_x = MAX_X_TABLE_VALUE << FRAC;
    let delta_x = 1i32 << (FRAC - ACTIVATION_DELTA_SHIFT);
    let v = x.raw();

    if v <= min_x {
        return entry(0);
    }
    if v >= max_x {
        return entry(MAX_ACTIVATION_INDEX);
    }

    let lower = ((v - min_x) / delta_x) as usize;
    if lower >= MAX_ACTIVATION_INDEX {
        return entry(MAX_ACTIVATION_INDEX);
    }
    let upper = lower + 1;
    let lower_x = min_x + (lower as i32) * delta_x;
    let upper_x = min_x + (upper as i32) * delta_x;

    if table[lower] == table[upper] {
        entry(upper)
    } else if v == lower_x {
        // x sits on a knot: the table value is exact.
        entry(lower)
    } else {
        linear_interpolation(
            x,
            QValue::from_raw(lower_x),
            QValue::from_raw(upper_x),
            entry(lower),
            entry(upper),
        )
    }
}

/// C++ `linearInterpolation`: `y0 + (y1 - y0) * ((x - x0) / (x1 - x0))`, in
/// fixed point, with the same operation order (it matters for truncation).
fn linear_interpolation<const FRAC: u32>(
    x: QValue<FRAC>,
    x0: QValue<FRAC>,
    x1: QValue<FRAC>,
    y0: QValue<FRAC>,
    y1: QValue<FRAC>,
) -> QValue<FRAC> {
    if x1 == x0 {
        return y0;
    }
    y0 + ((y1 - y0) * ((x - x0) / (x1 - x0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knots_and_clamps() {
        assert_eq!(Tanh::activate(Q16_16::ZERO).raw(), 0);
        assert_eq!(
            Sigmoid::activate(Q16_16::from_int(-100)).raw(),
            SIGMOID_Q16_16[0] as i32
        );
        assert_eq!(
            Sigmoid::activate(Q16_16::from_int(100)).raw(),
            SIGMOID_Q16_16[MAX_ACTIVATION_INDEX] as i32
        );
        // sigmoid(0) is the knot at index 48 (= 6 / (1/8)).
        assert_eq!(
            Sigmoid::activate(Q16_16::ZERO).raw(),
            SIGMOID_Q16_16[48] as i32
        );
    }

    #[test]
    fn interpolates_between_knots() {
        // Halfway between the knots at 0 and 1/8.
        let x = Q16_16::from_raw(4096);
        let y0 = TANH_Q16_16[48] as i32;
        let y1 = TANH_Q16_16[49] as i32;
        let got = Tanh::activate(x).raw();
        assert!(got > y0 && got < y1);
    }
}
