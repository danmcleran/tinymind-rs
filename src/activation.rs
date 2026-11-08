//! Activation policies, ported from tinymind `cpp/activationFunctions.hpp`,
//! `cpp/lookupTable.hpp` and `cpp/interpolate.hpp`.
//!
//! C++ policy classes with static member functions become zero-sized Rust types
//! implementing a trait. `Relu` and `Linear` need no table and work for every
//! value type.
//! `Tanh` and `Sigmoid` are implemented only for the formats that have lookup
//! tables (`Q16_16`, `Q8_8`): asking for `Tanh` on another format is a compile
//! error, as it is in C++ when the matching `TINYMIND_USE_TANH_x_y` table is not
//! compiled in.

use crate::lut::{
    NUMBER_OF_ACTIVATION_TABLE_VALUES, SIGMOID_Q8_8, SIGMOID_Q16_16, TANH_Q8_8, TANH_Q16_16,
};
use crate::qformat::{Fixed, Q8_8, Q16_16, QValue};

/// A neuron activation function and its derivative.
pub trait Activation<V> {
    /// The activation of the weighted input sum `x`.
    fn activate(x: V) -> V;
    /// Derivative expressed in terms of the activation's *output* `y`, the way
    /// tinymind computes it during back-propagation.
    fn derivative(y: V) -> V;
}

/// Rectified linear unit, `max(0, x)` (C++ `ReluActivationPolicy`).
pub struct Relu;

/// The identity, for a regression output layer (C++ `LinearActivationPolicy`).
/// Its gradient is unbounded; see `set_gradient_clip` on the networks.
pub struct Linear;

/// Hyperbolic tangent via the 96-entry lookup table.
pub struct Tanh;

/// Logistic sigmoid via the 96-entry lookup table.
pub struct Sigmoid;

impl<V: Fixed> Activation<V> for Relu {
    fn activate(x: V) -> V {
        if x <= V::ZERO { V::ZERO } else { x }
    }

    /// 1 where the neuron is active, else 0. The output is positive exactly when
    /// the input was, so deciding from the output gives the C++ result.
    fn derivative(y: V) -> V {
        if y <= V::ZERO { V::ZERO } else { V::ONE }
    }
}

impl<V: Fixed> Activation<V> for Linear {
    fn activate(x: V) -> V {
        x
    }

    /// Always 1.0 (the C++ returns `Constants<ValueType>::one()`, a real 1.0).
    fn derivative(_y: V) -> V {
        V::ONE
    }
}

/// `Tanh` and `Sigmoid` for one format, given its two tables.
macro_rules! impl_table_activations {
    ($v:ty, $tanh:expr, $sigmoid:expr) => {
        impl Activation<$v> for Tanh {
            fn activate(x: $v) -> $v {
                lookup(x, &$tanh)
            }

            fn derivative(y: $v) -> $v {
                <$v>::ONE - (y * y)
            }
        }

        impl Activation<$v> for Sigmoid {
            fn activate(x: $v) -> $v {
                lookup(x, &$sigmoid)
            }

            fn derivative(y: $v) -> $v {
                y * (<$v>::ONE - y)
            }
        }
    };
}

impl_table_activations!(Q16_16, TANH_Q16_16, SIGMOID_Q16_16);
impl_table_activations!(Q8_8, TANH_Q8_8, SIGMOID_Q8_8);

/// Table domain is [-6, 6] with knots 1/8 apart (`cpp/activation.hpp`).
const MIN_X_TABLE_VALUE: i32 = -6;
const MAX_X_TABLE_VALUE: i32 = 6;
const ACTIVATION_DELTA_SHIFT: u32 = 3;
const MAX_ACTIVATION_INDEX: usize = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;

/// What [`lookup`] needs from a format beyond [`Fixed`]: its raw value widened
/// to `i32` (exact for both storage types; every intermediate stays far inside
/// `i32`), and the reverse for values known to fit.
trait TableFormat: Fixed {
    const FRAC: u32;
    fn raw_i32(self) -> i32;
    fn from_raw_i32(raw: i32) -> Self;
}

impl<const FRAC: u32> TableFormat for QValue<i32, FRAC> {
    const FRAC: u32 = FRAC;
    fn raw_i32(self) -> i32 {
        self.raw()
    }
    fn from_raw_i32(raw: i32) -> Self {
        Self::from_raw(raw)
    }
}

impl<const FRAC: u32> TableFormat for QValue<i16, FRAC> {
    const FRAC: u32 = FRAC;
    fn raw_i32(self) -> i32 {
        i32::from(self.raw())
    }
    fn from_raw_i32(raw: i32) -> Self {
        Self::from_raw(raw as i16)
    }
}

/// Piecewise-linear table lookup (C++ `LookupTable::getValue`). Table entries
/// are the unsigned bit patterns of the raw values (`uint16_t`/`uint32_t` in
/// C++); the cast through `i32` and back to the storage type reinterprets them.
fn lookup<V: TableFormat, T: Copy + Into<u32>>(
    x: V,
    table: &[T; NUMBER_OF_ACTIVATION_TABLE_VALUES],
) -> V {
    let entry = |i: usize| V::from_raw_i32(table[i].into() as i32);

    let min_x = MIN_X_TABLE_VALUE << V::FRAC;
    let max_x = MAX_X_TABLE_VALUE << V::FRAC;
    let delta_x = 1i32 << (V::FRAC - ACTIVATION_DELTA_SHIFT);
    let v = x.raw_i32();

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

    if table[lower].into() == table[upper].into() {
        entry(upper)
    } else if v == lower_x {
        // x sits on a knot: the table value is exact.
        entry(lower)
    } else {
        linear_interpolation(
            x,
            V::from_raw_i32(lower_x),
            V::from_raw_i32(upper_x),
            entry(lower),
            entry(upper),
        )
    }
}

/// C++ `linearInterpolation`: `y0 + (y1 - y0) * ((x - x0) / (x1 - x0))`, in
/// fixed point, with the same operation order (it matters for truncation).
fn linear_interpolation<V: Fixed>(x: V, x0: V, x1: V, y0: V, y1: V) -> V {
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

    #[test]
    fn q8_8_knots_and_clamps() {
        // A Q8.8 table entry's u16 bit pattern is the i16 raw value.
        assert_eq!(Tanh::activate(Q8_8::ZERO).raw(), TANH_Q8_8[48] as i16);
        assert_eq!(
            Tanh::activate(Q8_8::from_int(-100)).raw(),
            TANH_Q8_8[0] as i16
        );
        assert_eq!(TANH_Q8_8[0] as i16, -255);
        assert_eq!(
            Sigmoid::activate(Q8_8::from_int(100)).raw(),
            SIGMOID_Q8_8[MAX_ACTIVATION_INDEX] as i16
        );
        assert_eq!(Sigmoid::activate(Q8_8::ZERO).raw(), 128); // 0.5
    }

    #[test]
    fn relu() {
        for (x, y, dy) in [(-3, 0, 0), (0, 0, 0), (1, 1, 1), (5, 5, 1)] {
            let x16 = Q16_16::from_int(x);
            assert_eq!(Relu::activate(x16), Q16_16::from_int(y));
            assert_eq!(Relu::derivative(Relu::activate(x16)), Q16_16::from_int(dy));
            let x8 = Q8_8::from_int(x);
            assert_eq!(Relu::activate(x8), Q8_8::from_int(y));
            assert_eq!(Relu::derivative(Relu::activate(x8)), Q8_8::from_int(dy));
        }
        // The smallest positive value passes through.
        assert_eq!(Relu::activate(Q16_16::from_raw(1)).raw(), 1);
        assert_eq!(Relu::activate(Q16_16::from_raw(-1)).raw(), 0);
    }
}
