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
//! Faithful to tinymind (C++), including two details that differ from textbook Adam:
//!
//! - The step count `t` used for bias correction advances once per *layer* update,
//!   starting from the output layer, not once per training step.
//! - The adaptive optimizers update the weights only. The bias weights keep their
//!   initial values (tinymind (C++)'s adaptive update path has no bias branch).
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
/// with bias correction while its step count (layer updates, see the module
/// docs) is below 100. Uses the network's learning
/// rate; the momentum and acceleration rates do not apply.
///
/// C++ `AdamOptimizer<ValueType, B1I, B1F, B2I, B2F, EI, EF>` takes each
/// hyperparameter as `QValue(int, frac)`, which is the raw value `(int << FRAC) |
/// frac`. Its defaults (`frac` 230, 255 and 1 with `int` 0) are 0.898, 0.996 and
/// 1/256 in Q8.8 but 230 / 65536 and so on in Q16.16, so there is deliberately no
/// default here: write the values for your format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adam<V> {
    /// Decay of the first moment (C++ `beta1`).
    pub beta1: V,
    /// Decay of the second moment (C++ `beta2`).
    pub beta2: V,
    /// Added to the root of the second moment, and the threshold below which
    /// bias correction is skipped.
    pub epsilon: V,
    timestep: usize,
}

impl<V> Adam<V> {
    /// Adam with these hyperparameters, at step 0.
    pub const fn new(beta1: V, beta2: V, epsilon: V) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            timestep: 0,
        }
    }

    /// The step count bias correction uses: the number of layer updates so far.
    pub fn timestep(&self) -> usize {
        self.timestep
    }
}

/// RMSprop (C++ `RmsPropOptimizer`): each weight's step is scaled by a running
/// root mean square of its gradients. Uses the network's learning rate only. As
/// with [`Adam`], write the hyperparameters for your format (C++ defaults:
/// `decay` raw 230, `epsilon` raw 1).
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

mod sealed {
    use super::*;

    /// What a network needs from its optimizer. Sealed: the crate implements it
    /// for [`Sgd`], [`Adam`] and [`RmsProp`], and the floating-point optimizers.
    pub trait OptimizerImpl<V> {
        /// C++ `step()`: called once at the start of each layer's update.
        fn begin_layer(&mut self);
        /// Update one weight from its gradient (clipped by `clip` if set).
        fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>);
        /// Update one bias weight (never clipped).
        fn update_bias(&self, c: &mut Connection<V>, rates: (V, V, V));
    }
}

/// An optimizer a trainable network can use: [`Sgd`], [`Adam`] or [`RmsProp`],
/// and for floating-point networks `AdamFloat` or `RmsPropFloat`.
pub trait Optimizer<V>: sealed::OptimizerImpl<V> {}

impl<V, T: sealed::OptimizerImpl<V>> Optimizer<V> for T {}

impl<V: Scalar> sealed::OptimizerImpl<V> for Sgd {
    fn begin_layer(&mut self) {}

    fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>) {
        update_connection(c, rates, clip);
    }

    fn update_bias(&self, c: &mut Connection<V>, rates: (V, V, V)) {
        update_connection(c, rates, None);
    }
}

fn clipped<V: Scalar>(gradient: V, clip: Option<(V, V)>) -> V {
    match clip {
        Some(range) => clip_gradient(gradient, range),
        None => gradient,
    }
}

impl<V: Scalar + Sqrt> sealed::OptimizerImpl<V> for Adam<V> {
    fn begin_layer(&mut self) {
        self.timestep = self.timestep.wrapping_add(1);
    }

    /// C++ `AdamOptimizer::updateWeights`, every expression in its order.
    fn update_weight(&self, c: &mut Connection<V>, (lr, _, _): (V, V, V), clip: Option<(V, V)>) {
        let one = V::ONE;
        let gradient = clipped(c.gradient, clip);
        // First moment in the delta-weight slot, second in the previous one.
        let m = self.beta1 * c.delta_weight + (one - self.beta1) * gradient;
        let v = self.beta2 * c.previous_delta_weight + (one - self.beta2) * gradient * gradient;
        c.delta_weight = m;
        c.previous_delta_weight = v;

        let mut m_hat = m;
        let mut v_hat = v;
        if self.timestep < 100 {
            let mut beta1_power = one;
            let mut beta2_power = one;
            for _ in 0..self.timestep {
                beta1_power = beta1_power * self.beta1;
                beta2_power = beta2_power * self.beta2;
            }
            let beta1_correction = one - beta1_power;
            let beta2_correction = one - beta2_power;
            if beta1_correction > self.epsilon {
                m_hat = m / beta1_correction;
            }
            if beta2_correction > self.epsilon {
                v_hat = v / beta2_correction;
            }
        }

        let update = lr * m_hat / (v_hat.sqrt() + self.epsilon);
        c.weight += update;
    }

    /// tinymind (C++)'s adaptive path does not update bias weights.
    fn update_bias(&self, _c: &mut Connection<V>, _rates: (V, V, V)) {}
}

impl<V: Scalar + Sqrt> sealed::OptimizerImpl<V> for RmsProp<V> {
    /// C++ counts steps for RMSprop too but never reads the count.
    fn begin_layer(&mut self) {}

    /// C++ `RmsPropOptimizer::updateWeights`, every expression in its order.
    fn update_weight(&self, c: &mut Connection<V>, (lr, _, _): (V, V, V), clip: Option<(V, V)>) {
        let gradient = clipped(c.gradient, clip);
        // The running mean of squared gradients lives in the second-moment slot.
        let v = self.decay * c.previous_delta_weight + (V::ONE - self.decay) * gradient * gradient;
        c.previous_delta_weight = v;
        let update = lr * gradient / (v.sqrt() + self.epsilon);
        c.weight += update;
    }

    /// tinymind (C++)'s adaptive path does not update bias weights.
    fn update_bias(&self, _c: &mut Connection<V>, _rates: (V, V, V)) {}
}

/// Adam for floating-point networks (C++ `AdamOptimizerFloat`): the
/// hyperparameters are `f64`, as in the C++, cast to the network's type where
/// they are used, and bias correction divides by `1 - beta^t` on every step,
/// with `beta^t` from `pow` in `f64` (no cut-off at step 100 and no epsilon
/// threshold, unlike [`Adam`]). The square root is taken in `f64`. As with
/// [`Adam`], `t` counts layer updates and bias weights are not updated.
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
        }
    }

    /// The step count bias correction uses: the number of layer updates so far.
    pub fn timestep(&self) -> usize {
        self.timestep
    }
}

/// RMSprop for floating-point networks (C++ `RmsPropOptimizerFloat`): the
/// hyperparameters are `f64`, cast to the network's type where they are used,
/// and the square root is taken in `f64`. Bias weights are not updated.
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

/// `x^y` and `sqrt(x)` in `f64`: the platform's with `std`, the `libm`
/// crate's without.
#[cfg(feature = "std")]
mod float_math {
    pub(super) fn pow(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
    pub(super) fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }
}

#[cfg(all(feature = "libm", not(feature = "std")))]
mod float_math {
    pub(super) fn pow(x: f64, y: f64) -> f64 {
        libm::pow(x, y)
    }
    pub(super) fn sqrt(x: f64) -> f64 {
        libm::sqrt(x)
    }
}

/// The floating-point optimizers for one float type, every expression in the
/// C++ order and with its casts.
#[cfg(any(feature = "std", feature = "libm"))]
macro_rules! impl_float_optimizers {
    ($t:ty) => {
        impl sealed::OptimizerImpl<$t> for AdamFloat {
            fn begin_layer(&mut self) {
                self.timestep = self.timestep.wrapping_add(1);
            }

            /// C++ `AdamOptimizerFloat::updateWeights`.
            fn update_weight(
                &self,
                c: &mut Connection<$t>,
                (lr, _, _): ($t, $t, $t),
                clip: Option<($t, $t)>,
            ) {
                let gradient = clipped(c.gradient, clip);
                let m = (self.beta1 as $t) * c.delta_weight + ((1.0 - self.beta1) as $t) * gradient;
                let v = (self.beta2 as $t) * c.previous_delta_weight
                    + ((1.0 - self.beta2) as $t) * gradient * gradient;
                c.delta_weight = m;
                c.previous_delta_weight = v;
                let t = self.timestep as f64;
                let beta1_power = float_math::pow(self.beta1, t);
                let beta2_power = float_math::pow(self.beta2, t);
                let m_hat = m / ((1.0 - beta1_power) as $t);
                let v_hat = v / ((1.0 - beta2_power) as $t);
                let sqrt_v = float_math::sqrt(f64::from(v_hat)) as $t;
                let update = lr * m_hat / (sqrt_v + self.epsilon as $t);
                c.weight += update;
            }

            /// tinymind (C++)'s adaptive path does not update bias weights.
            fn update_bias(&self, _c: &mut Connection<$t>, _rates: ($t, $t, $t)) {}
        }

        impl sealed::OptimizerImpl<$t> for RmsPropFloat {
            /// C++ counts steps for RMSprop too but never reads the count.
            fn begin_layer(&mut self) {}

            /// C++ `RmsPropOptimizerFloat::updateWeights`.
            fn update_weight(
                &self,
                c: &mut Connection<$t>,
                (lr, _, _): ($t, $t, $t),
                clip: Option<($t, $t)>,
            ) {
                let gradient = clipped(c.gradient, clip);
                let v = (self.decay as $t) * c.previous_delta_weight
                    + ((1.0 - self.decay) as $t) * gradient * gradient;
                c.previous_delta_weight = v;
                let sqrt_v = float_math::sqrt(f64::from(v)) as $t;
                let update = lr * gradient / (sqrt_v + self.epsilon as $t);
                c.weight += update;
            }

            /// tinymind (C++)'s adaptive path does not update bias weights.
            fn update_bias(&self, _c: &mut Connection<$t>, _rates: ($t, $t, $t)) {}
        }
    };
}

#[cfg(any(feature = "std", feature = "libm"))]
impl_float_optimizers!(f32);
#[cfg(any(feature = "std", feature = "libm"))]
impl_float_optimizers!(f64);
