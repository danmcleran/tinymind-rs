//! Weight-update rules for the trainable networks, ported from tinymind
//! `cpp/adam.hpp` and `cpp/rmsprop.hpp` (C++ `TransferFunctionOptimizerPolicy`).
//!
//! A network's optimizer is a type parameter, as in C++: [`Sgd`] (the default,
//! zero-sized, so a network without an adaptive optimizer pays nothing for this),
//! [`Adam`] or [`RmsProp`]. Pass an adaptive optimizer with its hyperparameters to
//! `with_optimizer` on [`NeuralNetwork`](crate::NeuralNetwork) or
//! [`MultilayerPerceptron`](crate::MultilayerPerceptron).
//!
//! Faithful to tinymind, including two details that differ from textbook Adam:
//!
//! - The step count `t` used for bias correction advances once per *layer* update,
//!   starting from the output layer, not once per training step.
//! - The adaptive optimizers update the weights only. The bias weights keep their
//!   initial values (tinymind's adaptive update path has no bias branch).
//!
//! The first and second moments live in the connection's delta-weight slots, as
//! in tinymind, so an adaptive optimizer needs no extra memory per weight.

// The sealed trait's methods take the crate-private `Connection`. The trait
// cannot be named outside the crate (it lives in a private module), so nothing
// private is usable from outside; the lint cannot see that.
#![allow(private_interfaces)]

use crate::neuralnet::{Connection, clip_gradient, update_connection};
use crate::qformat::{Fixed, Sqrt};

/// Stochastic gradient descent with momentum and acceleration (the default;
/// C++ `NullOptimizerPolicy`). The rates are the network's
/// `set_learning_rate`, `set_momentum_rate` and `set_acceleration_rate`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sgd;

/// Adam (C++ `AdamOptimizer`): per-weight first and second moment estimates
/// with bias correction for the first 100 steps. Uses the network's learning
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
    /// for [`Sgd`], [`Adam`] and [`RmsProp`].
    pub trait OptimizerImpl<V> {
        /// C++ `step()`: called once at the start of each layer's update.
        fn begin_layer(&mut self);
        /// Update one weight from its gradient (clipped by `clip` if set).
        fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>);
        /// Update one bias weight (never clipped).
        fn update_bias(&self, c: &mut Connection<V>, rates: (V, V, V));
    }
}

/// An optimizer a trainable network can use: [`Sgd`], [`Adam`] or [`RmsProp`].
pub trait Optimizer<V>: sealed::OptimizerImpl<V> {}

impl<V, T: sealed::OptimizerImpl<V>> Optimizer<V> for T {}

impl<V: Fixed> sealed::OptimizerImpl<V> for Sgd {
    fn begin_layer(&mut self) {}

    fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>) {
        update_connection(c, rates, clip);
    }

    fn update_bias(&self, c: &mut Connection<V>, rates: (V, V, V)) {
        update_connection(c, rates, None);
    }
}

fn clipped<V: Fixed>(gradient: V, clip: Option<(V, V)>) -> V {
    match clip {
        Some(range) => clip_gradient(gradient, range),
        None => gradient,
    }
}

impl<V: Fixed + Sqrt> sealed::OptimizerImpl<V> for Adam<V> {
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

    /// tinymind's adaptive path does not update bias weights.
    fn update_bias(&self, _c: &mut Connection<V>, _rates: (V, V, V)) {}
}

impl<V: Fixed + Sqrt> sealed::OptimizerImpl<V> for RmsProp<V> {
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

    /// tinymind's adaptive path does not update bias weights.
    fn update_bias(&self, _c: &mut Connection<V>, _rates: (V, V, V)) {}
}
