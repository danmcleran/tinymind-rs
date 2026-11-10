//! Weight-update rules for the trainable networks, ported from tinymind (C++)
//! `cpp/adam.hpp` and `cpp/rmsprop.hpp` (C++ `TransferFunctionOptimizerPolicy`).
//!
//! A network's optimizer is a type parameter, as in C++: [`Sgd`] (the default,
//! zero-sized, so a network without an adaptive optimizer pays nothing for this),
//! [`Adam`] or [`RmsProp`], or for `f32` and `f64` networks `AdamFloat` and
//! `RmsPropFloat` (C++ `AdamOptimizerFloat`, `RmsPropOptimizerFloat`; feature
//! `std` or `libm`). Pass an adaptive optimizer with its hyperparameters to
//! `with_optimizer` on [`NeuralNetwork`](crate::NeuralNetwork) or
//! [`MultilayerPerceptron`](crate::MultilayerPerceptron).
//!
//! Every optimizer updates the bias weights as it updates the other weights,
//! and the network's gradient clipping applies to both. Adam's step count `t`
//! advances once per training step, before any weight moves.
//!
//! The first and second moments live in the connection's delta-weight slots, as
//! in tinymind (C++), so an adaptive optimizer needs no extra memory per weight.

// The sealed trait's methods take the crate-private `Connection`. The trait
// cannot be named outside the crate (it lives in a private module), so nothing
// private is usable from outside; the lint cannot see that.
#![allow(private_interfaces)]

use crate::neuralnet::{Connection, clip_gradient, update_connection};
use crate::qformat::{Scalar, Sqrt};

/// Stochastic gradient descent with momentum and acceleration (the default;
/// C++ `NullOptimizerPolicy`). The rates are the network's
/// `set_learning_rate`, `set_momentum_rate` and `set_acceleration_rate`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sgd;

/// Adam (C++ `AdamOptimizer`): per-weight first and second moment estimates
/// with bias correction by `1 - beta^t`, `beta^t` kept as a running product
/// (one multiply per training step). Uses the network's learning rate; the
/// momentum and acceleration rates do not apply.
///
/// `Adam::<Q16_16>::default()` and `Adam::<Q8_8>::default()` are the
/// defaults of tinymind (C++) for those formats: beta1 0.9, beta2 0.99 and epsilon
/// `sqrt(LSB / (1 - beta2))`, each rounded to the format. Not the textbook
/// 0.999 and 1e-8: in fixed point `(1 - beta2) * g * g` rounds to zero for
/// gradients below that epsilon, and a smaller epsilon then lets steps grow
/// far beyond the learning rate. Q8.8 has too few fractional bits for Adam
/// (its default epsilon is 0.625).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adam<V> {
    /// Decay of the first moment (C++ `beta1`).
    pub beta1: V,
    /// Decay of the second moment (C++ `beta2`).
    pub beta2: V,
    /// Added to the root of the second moment.
    pub epsilon: V,
    timestep: usize,
    /// `(beta1^t, beta2^t)`; `None` at step 0, where both are 1.
    powers: Option<(V, V)>,
}

impl<V> Adam<V> {
    /// Adam with these hyperparameters, at step 0.
    pub const fn new(beta1: V, beta2: V, epsilon: V) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            timestep: 0,
            powers: None,
        }
    }

    /// The step count bias correction uses: the number of training steps so far.
    pub fn timestep(&self) -> usize {
        self.timestep
    }
}

impl Default for Adam<crate::Q16_16> {
    /// tinymind (C++)'s Q16.16 defaults: beta1 58982/65536, beta2
    /// 64881/65536, epsilon 2560/65536 (0.039).
    fn default() -> Self {
        use crate::Q16_16;
        Self::new(
            Q16_16::from_raw(58_982),
            Q16_16::from_raw(64_881),
            Q16_16::from_raw(2_560),
        )
    }
}

impl Default for Adam<crate::Q8_8> {
    /// tinymind (C++)'s Q8.8 defaults: beta1 230/256, beta2 253/256,
    /// epsilon 160/256 (0.625).
    fn default() -> Self {
        use crate::Q8_8;
        Self::new(
            Q8_8::from_raw(230),
            Q8_8::from_raw(253),
            Q8_8::from_raw(160),
        )
    }
}

/// RMSprop (C++ `RmsPropOptimizer`): each weight's step is scaled by a running
/// root mean square of its gradients. Uses the network's learning rate only.
/// `RmsProp::<Q16_16>::default()` and `RmsProp::<Q8_8>::default()` are
/// tinymind (C++)'s defaults: decay 0.9 and epsilon 1e-8, each rounded to the
/// format and kept above 0 (epsilon is then one LSB).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsProp<V> {
    /// Decay of the running mean of squared gradients.
    pub decay: V,
    /// Added to the root of that mean.
    pub epsilon: V,
}

impl<V> RmsProp<V> {
    /// RMSprop with these hyperparameters.
    pub const fn new(decay: V, epsilon: V) -> Self {
        Self { decay, epsilon }
    }
}

impl Default for RmsProp<crate::Q16_16> {
    /// tinymind (C++)'s Q16.16 defaults: decay 58982/65536, epsilon 1/65536.
    fn default() -> Self {
        use crate::Q16_16;
        Self::new(Q16_16::from_raw(58_982), Q16_16::from_raw(1))
    }
}

impl Default for RmsProp<crate::Q8_8> {
    /// tinymind (C++)'s Q8.8 defaults: decay 230/256, epsilon 1/256.
    fn default() -> Self {
        use crate::Q8_8;
        Self::new(Q8_8::from_raw(230), Q8_8::from_raw(1))
    }
}

mod sealed {
    use super::*;

    /// What a network needs from its optimizer. Sealed: the crate implements it
    /// for [`Sgd`], [`Adam`] and [`RmsProp`], and the floating-point optimizers.
    pub trait OptimizerImpl<V> {
        /// C++ `step()`: called once per training step, before any weight
        /// moves (`beginTrainingStep`).
        fn begin_step(&mut self);
        /// Update one weight or bias weight from its gradient (clipped by
        /// `clip` if set). Since tinymind (C++) #264 every optimizer updates
        /// bias weights by the same rule as the other weights.
        fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>);
    }
}

/// An optimizer a trainable network can use: [`Sgd`], [`Adam`] or [`RmsProp`],
/// and for floating-point networks `AdamFloat` or `RmsPropFloat`.
pub trait Optimizer<V>: sealed::OptimizerImpl<V> {}

impl<V, T: sealed::OptimizerImpl<V>> Optimizer<V> for T {}

impl<V: Scalar> sealed::OptimizerImpl<V> for Sgd {
    fn begin_step(&mut self) {}

    fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>) {
        update_connection(c, rates, clip);
    }
}

/// One optimizer step on every connection of `connections` (weights or bias
/// weights: since tinymind (C++) #264 both follow the same rule). Never
/// inlined, so a network has one copy of the update code per optimizer
/// however many layers call it: with the code inlined at every weight and
/// bias loop the Cortex-M0+ anomaly image no longer fit its 32 KB of flash,
/// and a single chained-iterator loop instead cost the Cortex-M4 32% more
/// cycles per training step.
#[inline(never)]
pub(crate) fn update_all<V: Copy, Opt: Optimizer<V>>(
    optimizer: &Opt,
    connections: &mut [Connection<V>],
    rates: (V, V, V),
    clip: Option<(V, V)>,
) {
    for c in connections {
        optimizer.update_weight(c, rates, clip);
    }
}

fn clipped<V: Scalar>(gradient: V, clip: Option<(V, V)>) -> V {
    match clip {
        Some(range) => clip_gradient(gradient, range),
        None => gradient,
    }
}

impl<V: Scalar + Sqrt> Adam<V> {
    /// C++ `AdamOptimizer::computeUpdate`, every expression in its order:
    /// updates the moments in the connection's delta-weight slots and returns
    /// `lr * m_hat / (sqrt(v_hat) + epsilon)`.
    fn compute_update(&self, c: &mut Connection<V>, gradient: V, lr: V) -> V {
        let one = V::ONE;
        // First moment in the delta-weight slot, second in the previous one.
        let m = self.beta1 * c.delta_weight + (one - self.beta1) * gradient;
        let v = self.beta2 * c.previous_delta_weight + (one - self.beta2) * gradient * gradient;
        c.delta_weight = m;
        c.previous_delta_weight = v;

        let mut m_hat = m;
        let mut v_hat = v;
        // Before the first step both corrections are 1 - 1 = 0 and the moments
        // are used as they are.
        let (beta1_power, beta2_power) = self.powers.unwrap_or((one, one));
        let beta1_correction = one - beta1_power;
        let beta2_correction = one - beta2_power;
        if beta1_correction > V::ZERO {
            m_hat = m / beta1_correction;
        }
        if beta2_correction > V::ZERO {
            v_hat = v / beta2_correction;
        }
        lr * m_hat / (v_hat.sqrt() + self.epsilon)
    }
}

impl<V: Scalar + Sqrt> sealed::OptimizerImpl<V> for Adam<V> {
    fn begin_step(&mut self) {
        self.timestep = self.timestep.wrapping_add(1);
        let (beta1_power, beta2_power) = self.powers.unwrap_or((V::ONE, V::ONE));
        self.powers = Some((beta1_power * self.beta1, beta2_power * self.beta2));
    }

    fn update_weight(&self, c: &mut Connection<V>, (lr, _, _): (V, V, V), clip: Option<(V, V)>) {
        let update = self.compute_update(c, clipped(c.gradient, clip), lr);
        c.weight += update;
    }
}

impl<V: Scalar + Sqrt> RmsProp<V> {
    /// C++ `RmsPropOptimizer::computeUpdate`: the running mean of squared
    /// gradients in the second-moment slot, and `lr * g / (sqrt(v) + epsilon)`.
    fn compute_update(&self, c: &mut Connection<V>, gradient: V, lr: V) -> V {
        let v = self.decay * c.previous_delta_weight + (V::ONE - self.decay) * gradient * gradient;
        c.previous_delta_weight = v;
        lr * gradient / (v.sqrt() + self.epsilon)
    }
}

impl<V: Scalar + Sqrt> sealed::OptimizerImpl<V> for RmsProp<V> {
    /// C++ counts steps for RMSprop too but never reads the count.
    fn begin_step(&mut self) {}

    fn update_weight(&self, c: &mut Connection<V>, (lr, _, _): (V, V, V), clip: Option<(V, V)>) {
        let update = self.compute_update(c, clipped(c.gradient, clip), lr);
        c.weight += update;
    }
}

/// Adam for floating-point networks (C++ `AdamOptimizerFloat`): the
/// hyperparameters are `f64`, as in the C++, cast to the network's type where
/// they are used, and from the first training step bias correction divides by
/// `1 - beta^t`, with `beta^t` a running product in `f64`. The square root is
/// taken in `f64`.
///
/// Implemented for `f32` and `f64` networks; needs the `std` feature (`pow`
/// and `sqrt` from the platform, bit-identical to the C++ on a glibc host) or
/// `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamFloat {
    /// Decay of the first moment (C++ default 0.9).
    pub beta1: f64,
    /// Decay of the second moment (C++ default 0.999).
    pub beta2: f64,
    /// Added to the root of the second moment (C++ default 1e-8).
    pub epsilon: f64,
    timestep: usize,
    beta1_power: f64,
    beta2_power: f64,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Default for AdamFloat {
    /// tinymind (C++)'s values: 0.9, 0.999, 1e-8.
    fn default() -> Self {
        Self::new(0.9, 0.999, 1e-8)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl AdamFloat {
    /// Adam with these hyperparameters, at step 0.
    pub const fn new(beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            timestep: 0,
            beta1_power: 1.0,
            beta2_power: 1.0,
        }
    }

    /// The step count bias correction uses: the number of training steps so far.
    pub fn timestep(&self) -> usize {
        self.timestep
    }
}

/// RMSprop for floating-point networks (C++ `RmsPropOptimizerFloat`): the
/// hyperparameters are `f64`, cast to the network's type where they are used,
/// and the square root is taken in `f64`.
///
/// Implemented for `f32` and `f64` networks; needs the `std` feature or `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsPropFloat {
    /// Decay of the running mean of squared gradients (C++ default 0.9).
    pub decay: f64,
    /// Added to the root of that mean (C++ default 1e-8).
    pub epsilon: f64,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Default for RmsPropFloat {
    /// tinymind (C++)'s values: 0.9, 1e-8.
    fn default() -> Self {
        Self::new(0.9, 1e-8)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl RmsPropFloat {
    /// RMSprop with these hyperparameters.
    pub const fn new(decay: f64, epsilon: f64) -> Self {
        Self { decay, epsilon }
    }
}

/// `sqrt(x)` in `f64`: the platform's with `std`, the `libm` crate's without.
#[cfg(feature = "std")]
mod float_math {
    pub(super) fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }
}

#[cfg(all(feature = "libm", not(feature = "std")))]
mod float_math {
    pub(super) fn sqrt(x: f64) -> f64 {
        libm::sqrt(x)
    }
}

/// The shared part of a floating-point optimizer's weight and bias updates,
/// per float type (a trait so the `f32` and `f64` versions can coexist).
#[cfg(any(feature = "std", feature = "libm"))]
trait FloatUpdate<T> {
    /// Updates the connection's moments and returns the step to add.
    fn compute_update_float(&self, c: &mut Connection<T>, gradient: T, lr: T) -> T;
}

/// The floating-point optimizers for one float type, every expression in the
/// C++ order and with its casts.
#[cfg(any(feature = "std", feature = "libm"))]
macro_rules! impl_float_optimizers {
    ($t:ty) => {
        impl FloatUpdate<$t> for AdamFloat {
            /// C++ `AdamOptimizerFloat::computeUpdate`.
            fn compute_update_float(&self, c: &mut Connection<$t>, gradient: $t, lr: $t) -> $t {
                let m = (self.beta1 as $t) * c.delta_weight + ((1.0 - self.beta1) as $t) * gradient;
                let v = (self.beta2 as $t) * c.previous_delta_weight
                    + ((1.0 - self.beta2) as $t) * gradient * gradient;
                c.delta_weight = m;
                c.previous_delta_weight = v;
                let mut m_hat = m;
                let mut v_hat = v;
                if self.timestep > 0 {
                    m_hat = m / ((1.0 - self.beta1_power) as $t);
                    v_hat = v / ((1.0 - self.beta2_power) as $t);
                }
                let sqrt_v = float_math::sqrt(f64::from(v_hat)) as $t;
                lr * m_hat / (sqrt_v + self.epsilon as $t)
            }
        }

        impl sealed::OptimizerImpl<$t> for AdamFloat {
            fn begin_step(&mut self) {
                self.timestep = self.timestep.wrapping_add(1);
                self.beta1_power *= self.beta1;
                self.beta2_power *= self.beta2;
            }

            fn update_weight(
                &self,
                c: &mut Connection<$t>,
                (lr, _, _): ($t, $t, $t),
                clip: Option<($t, $t)>,
            ) {
                let update = self.compute_update_float(c, clipped(c.gradient, clip), lr);
                c.weight += update;
            }
        }

        impl FloatUpdate<$t> for RmsPropFloat {
            /// C++ `RmsPropOptimizerFloat::computeUpdate`.
            fn compute_update_float(&self, c: &mut Connection<$t>, gradient: $t, lr: $t) -> $t {
                let v = (self.decay as $t) * c.previous_delta_weight
                    + ((1.0 - self.decay) as $t) * gradient * gradient;
                c.previous_delta_weight = v;
                let sqrt_v = float_math::sqrt(f64::from(v)) as $t;
                lr * gradient / (sqrt_v + self.epsilon as $t)
            }
        }

        impl sealed::OptimizerImpl<$t> for RmsPropFloat {
            /// C++ counts steps for RMSprop too but never reads the count.
            fn begin_step(&mut self) {}

            fn update_weight(
                &self,
                c: &mut Connection<$t>,
                (lr, _, _): ($t, $t, $t),
                clip: Option<($t, $t)>,
            ) {
                let update = self.compute_update_float(c, clipped(c.gradient, clip), lr);
                c.weight += update;
            }
        }
    };
}

#[cfg(any(feature = "std", feature = "libm"))]
impl_float_optimizers!(f32);
#[cfg(any(feature = "std", feature = "libm"))]
impl_float_optimizers!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Q8_8, Q16_16};

    /// The defaults are tinymind (C++)'s `QFractionLsbs<Q, 9, 10>`,
    /// `QFractionLsbs<Q, 99, 100>`, `QAdamEpsilonLsbs<Q, 100>` and
    /// `QFractionLsbs<Q, 1, 100000000>`, printed by a C++ program at
    /// tinymind (C++) a40979f. The Q8.8 ones are also checked by `tests/deep_mlp.rs`, whose
    /// C++ runs use `AdamOptimizer<Q8>` and `RmsPropOptimizer<Q8>`.
    #[test]
    fn defaults_are_tinymind_cpp_constants() {
        let a = Adam::<Q16_16>::default();
        assert_eq!(
            (a.beta1.raw(), a.beta2.raw(), a.epsilon.raw()),
            (58_982, 64_881, 2_560)
        );
        let a = Adam::<Q8_8>::default();
        assert_eq!(
            (a.beta1.raw(), a.beta2.raw(), a.epsilon.raw()),
            (230, 253, 160)
        );
        let r = RmsProp::<Q16_16>::default();
        assert_eq!((r.decay.raw(), r.epsilon.raw()), (58_982, 1));
        let r = RmsProp::<Q8_8>::default();
        assert_eq!((r.decay.raw(), r.epsilon.raw()), (230, 1));
    }
}
