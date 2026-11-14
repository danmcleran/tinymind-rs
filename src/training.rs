//! Training controls for the trainable networks, ported from tinymind (C++)
//! `cpp/weightDecay.hpp`, `cpp/learningRateSchedule.hpp` and
//! `cpp/earlyStopping.hpp`.
//!
//! - **[`WeightDecay`]** (L2) shrinks every connection weight, but no bias
//!   weight, by `learning rate * lambda * weight` before each update. With
//!   SGD this is an L2 penalty `(lambda / 2) * w^2` in the loss; under Adam
//!   and RMSprop the same shrink comes before the adaptive step, which makes
//!   it decoupled weight decay (AdamW). In fixed point it keeps weights from
//!   growing until they overflow.
//! - **[`Scheduled`]** with a [`StepDecay`] multiplies the learning rate by a
//!   factor every `interval` training steps. Read the current rate with the
//!   network's `learning_rate()`. Any other schedule is a call to
//!   `set_learning_rate` between steps.
//! - **[`EarlyStopping`]** watches an error (training or validation) and says
//!   when it has not improved for `patience` checks in a row. It is not tied
//!   to a network; the training loop asks it.
//!
//! [`WeightDecay`] and [`Scheduled`] wrap an optimizer ([`Sgd`](crate::Sgd),
//! [`Adam`](crate::Adam), ...) and are passed to `with_optimizer` on
//! [`MultilayerPerceptron`](crate::MultilayerPerceptron),
//! [`NeuralNetwork`](crate::NeuralNetwork) or
//! [`LstmNetwork`](crate::recurrent::LstmNetwork), as tinymind (C++) takes
//! `L2WeightDecay` and `StepDecaySchedule` as policies. A network built
//! without them compiles no decay or schedule code and stores nothing for
//! them. With them, results are bit-identical to the C++ network built with
//! the same policies.
//!
//! ```
//! use tinymind::training::{Scheduled, StepDecay, WeightDecay};
//! use tinymind::{MultilayerPerceptron, Q16_16, Sgd, Sigmoid, Tanh};
//!
//! // tinymind (C++)'s defaults for Q16.16: lambda 66/65536 (0.001), and the
//! // rate times 0.9 every 1000 steps.
//! let optimizer = Scheduled::new(
//!     WeightDecay::new(Sgd, Q16_16::from_raw(66)),
//!     StepDecay::<Q16_16>::default(),
//! );
//! let mut seed = 1u32;
//! let mut nn = MultilayerPerceptron::<Q16_16, Tanh, Sigmoid, 2, 4, 1, _>::with_optimizer(
//!     || {
//!         seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
//!         Q16_16::from_raw((seed >> 16) as i32 % 32_769 - 16_384)
//!     },
//!     optimizer,
//! );
//!
//! let samples = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)];
//! for step in 0..2_500 {
//!     let (x, t) = samples[step % 4];
//!     nn.feed_forward(&x.map(Q16_16::from_int));
//!     nn.train(&[Q16_16::from_int(t)]);
//! }
//! // Two decays so far: 1/4 * 0.9 * 0.9, each product floored.
//! let factor = Q16_16::from_raw(58_982);
//! assert_eq!(nn.learning_rate(), Q16_16::from_raw(16_384) * factor * factor);
//! assert_eq!(nn.optimizer().schedule.steps_since_decay(), 500);
//! ```

// The sealed optimizer trait's methods take the crate-private `Connection`;
// see `crate::optimizer`.
#![allow(private_interfaces)]

use crate::neuralnet::Connection;
use crate::optimizer::sealed::OptimizerImpl;
use crate::qformat::Scalar;

/// L2 weight decay around an optimizer (C++ `L2WeightDecay`): before each
/// update every connection weight, but no bias weight, becomes
/// `w - (learning rate * lambda) * w`, then `optimizer` takes its step.
///
/// tinymind (C++)'s default lambda is 0.001 rounded to the format and kept at
/// one LSB or more: `Q16_16::from_raw(66)` and `Q8_8::from_raw(1)` (0.001 is
/// below one Q8.8 LSB). A lambda of 0 gives the same results as `optimizer`
/// alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightDecay<Opt, V> {
    /// The optimizer that takes each step after the decay.
    pub optimizer: Opt,
    /// The decay rate.
    pub lambda: V,
}

impl<Opt, V> WeightDecay<Opt, V> {
    /// `optimizer` with L2 weight decay `lambda`.
    pub const fn new(optimizer: Opt, lambda: V) -> Self {
        Self { optimizer, lambda }
    }
}

impl<V: Scalar, Opt: OptimizerImpl<V>> OptimizerImpl<V> for WeightDecay<Opt, V> {
    fn begin_step(&mut self) {
        self.optimizer.begin_step();
    }

    fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>) {
        self.optimizer.update_weight(c, rates, clip);
    }

    fn weight_decay(&self) -> Option<V> {
        Some(self.lambda)
    }

    fn next_learning_rate(&mut self, rate: V) -> V {
        self.optimizer.next_learning_rate(rate)
    }
}

/// A learning-rate schedule around an optimizer (C++ `StepDecaySchedule` as
/// the `LearningRateSchedulePolicy`): after each update the network's
/// learning rate becomes `schedule.step(rate)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scheduled<Opt, V> {
    /// The optimizer that takes each step.
    pub optimizer: Opt,
    /// The schedule, with its step count.
    pub schedule: StepDecay<V>,
}

impl<Opt, V> Scheduled<Opt, V> {
    /// `optimizer` with the learning rate following `schedule`.
    pub const fn new(optimizer: Opt, schedule: StepDecay<V>) -> Self {
        Self {
            optimizer,
            schedule,
        }
    }
}

impl<V: Scalar, Opt: OptimizerImpl<V>> OptimizerImpl<V> for Scheduled<Opt, V> {
    fn begin_step(&mut self) {
        self.optimizer.begin_step();
    }

    fn update_weight(&self, c: &mut Connection<V>, rates: (V, V, V), clip: Option<(V, V)>) {
        self.optimizer.update_weight(c, rates, clip);
    }

    fn weight_decay(&self) -> Option<V> {
        self.optimizer.weight_decay()
    }

    fn next_learning_rate(&mut self, rate: V) -> V {
        let rate = self.optimizer.next_learning_rate(rate);
        self.schedule.step(rate)
    }
}

/// Step-decay learning-rate schedule (C++ `StepDecaySchedule`): every
/// `interval` training steps the learning rate is multiplied by `factor`.
///
/// A network trained with [`Scheduled`] calls [`step`](Self::step) after each
/// weight update, so the rate used by update `n` (counting from 1) is the initial
/// rate times `factor^((n - 1) / interval)`, each product rounded as the
/// value type rounds. An interval of 0 decays after every step, as 1 does.
///
/// `StepDecay::<Q16_16>::default()` and `StepDecay::<Q8_8>::default()` are
/// tinymind (C++)'s defaults: interval 1000 and factor 0.9 rounded to the format
/// (58982/65536 and 230/256). The counts are `u32` (C++ `size_t`), so the
/// schedule is the same size on a microcontroller and a 64-bit host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepDecay<V> {
    /// Training steps between two decays.
    pub interval: u32,
    /// What the learning rate is multiplied by at each decay.
    pub factor: V,
    pub(crate) count: u32,
}

impl<V> StepDecay<V> {
    /// A schedule that multiplies the rate by `factor` every `interval` steps,
    /// starting its count at 0.
    pub const fn new(interval: u32, factor: V) -> Self {
        Self {
            interval,
            factor,
            count: 0,
        }
    }

    /// Steps counted since the last decay (or since the start).
    pub fn steps_since_decay(&self) -> u32 {
        self.count
    }
}

impl<V: Scalar> StepDecay<V> {
    /// Count one training step and return the learning rate for the next one:
    /// `rate * factor` when this step completes an interval, else `rate`.
    pub fn step(&mut self, rate: V) -> V {
        self.count = self.count.saturating_add(1);
        if self.count >= self.interval {
            self.count = 0;
            return rate * self.factor;
        }
        rate
    }
}

impl Default for StepDecay<crate::Q16_16> {
    /// tinymind (C++)'s Q16.16 default: interval 1000, factor 58982/65536.
    fn default() -> Self {
        Self::new(1000, crate::Q16_16::from_raw(58_982))
    }
}

impl Default for StepDecay<crate::Q8_8> {
    /// tinymind (C++)'s Q8.8 default: interval 1000, factor 230/256.
    fn default() -> Self {
        Self::new(1000, crate::Q8_8::from_raw(230))
    }
}

/// Early stopping (C++ `EarlyStopping`): reports when the error has not
/// improved for `patience` checks in a row.
///
/// The first error checked becomes the best. After that, an error below the
/// best replaces it and resets the count; any other error (equal, larger, or
/// a floating-point NaN) adds one to the count, and
/// [`should_stop`](Self::should_stop) returns `true` once the count reaches
/// `patience`, and keeps returning `true` until an error improves on the best.
///
/// The default patience is 100, tinymind (C++)'s default.
///
/// ```
/// use tinymind::training::EarlyStopping;
///
/// let mut stopper = EarlyStopping::new(2);
/// let errors = [0.9, 0.5, 0.6, 0.4, 0.4, 0.7, 0.3];
/// let stop = errors.iter().position(|&e| stopper.should_stop(e));
/// assert_eq!(stop, Some(5)); // 0.4 and then 0.7 did not improve on 0.4
/// assert_eq!(stopper.best_error(), Some(0.4));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EarlyStopping<V> {
    /// Checks without improvement after which training should stop.
    pub patience: u32,
    pub(crate) best: Option<V>,
    pub(crate) counter: u32,
}

impl<V> EarlyStopping<V> {
    /// Early stopping after `patience` checks without improvement.
    pub const fn new(patience: u32) -> Self {
        Self {
            patience,
            best: None,
            counter: 0,
        }
    }

    /// Forget the best error and the count, as before the first check.
    pub fn reset(&mut self) {
        self.best = None;
        self.counter = 0;
    }

    /// Checks since the error last improved.
    pub fn patience_counter(&self) -> u32 {
        self.counter
    }
}

impl<V: Copy> EarlyStopping<V> {
    /// The lowest error seen so far, or `None` before the first check.
    pub fn best_error(&self) -> Option<V> {
        self.best
    }
}

impl<V: Copy + PartialOrd> EarlyStopping<V> {
    /// Report `error` (training or validation) and return whether training
    /// should stop: `true` when `patience` checks in a row, this one included,
    /// have not improved on the best error.
    pub fn should_stop(&mut self, error: V) -> bool {
        match self.best {
            None => {
                self.best = Some(error);
                self.counter = 0;
                false
            }
            Some(best) if error < best => {
                self.best = Some(error);
                self.counter = 0;
                false
            }
            Some(_) => {
                // Saturating: past u32::MAX checks the answer stays `true`.
                self.counter = self.counter.saturating_add(1);
                self.counter >= self.patience
            }
        }
    }
}

impl<V> Default for EarlyStopping<V> {
    /// Patience 100, tinymind (C++)'s default.
    fn default() -> Self {
        Self::new(100)
    }
}

/// C++ `L2WeightDecay::applyDecay` on every connection of `connections`:
/// `w - (learning rate * lambda) * w`, where `factor` is the product
/// `learning rate * lambda` (the C++ evaluates it left to right, so it is the
/// same value for every weight). Never inlined, so a network has one copy
/// however many layers call it.
#[inline(never)]
pub(crate) fn decay_weights<V: Scalar>(connections: &mut [Connection<V>], factor: V) {
    for c in connections {
        c.weight = c.weight - factor * c.weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Q8_8, Q16_16};

    /// The defaults are tinymind (C++)'s `QFractionLsbs<Q, 9, 10>` with
    /// `StepInterval = 1000`.
    #[test]
    fn step_decay_defaults_are_tinymind_cpp_constants() {
        let s = StepDecay::<Q16_16>::default();
        assert_eq!((s.interval, s.factor.raw()), (1000, 58_982));
        let s = StepDecay::<Q8_8>::default();
        assert_eq!((s.interval, s.factor.raw()), (1000, 230));
    }

    #[test]
    fn step_decay_multiplies_every_interval() {
        let half = Q16_16::from_raw(1 << 15);
        let mut s = StepDecay::new(3, half);
        let mut rate = Q16_16::ONE;
        let mut rates = [0; 7];
        for r in &mut rates {
            rate = s.step(rate);
            *r = rate.raw();
        }
        assert_eq!(
            rates,
            [65_536, 65_536, 32_768, 32_768, 32_768, 16_384, 16_384]
        );
        assert_eq!(s.steps_since_decay(), 1);
    }

    #[test]
    fn step_decay_interval_zero_decays_every_step() {
        let mut s = StepDecay::new(0, 0.5f64);
        assert_eq!(s.step(1.0), 0.5);
        assert_eq!(s.step(0.5), 0.25);
        assert_eq!(s.steps_since_decay(), 0);
    }

    #[test]
    fn early_stopping_counts_non_improving_checks() {
        let mut e = EarlyStopping::new(2);
        assert_eq!(e.best_error(), None);
        assert!(!e.should_stop(5));
        assert!(!e.should_stop(5)); // equal: count 1
        assert!(e.should_stop(6)); // count 2
        assert!(e.should_stop(7)); // still true
        assert!(!e.should_stop(4)); // improved
        assert_eq!((e.best_error(), e.patience_counter()), (Some(4), 0));
        e.reset();
        assert_eq!((e.best_error(), e.patience_counter()), (None, 0));
        assert!(!e.should_stop(9));
    }

    #[test]
    fn early_stopping_counts_nan_as_no_improvement() {
        let mut e = EarlyStopping::new(1);
        assert!(!e.should_stop(1.0f32));
        assert!(e.should_stop(f32::NAN));
        assert_eq!(e.best_error(), Some(1.0));
    }

    /// Lambda 0 and a factor of 1 change nothing: the wrapped optimizer
    /// trains every network type exactly as the plain one does, in fixed
    /// and floating point.
    #[test]
    fn neutral_wrappers_train_like_the_plain_optimizer() {
        use crate::recurrent::LstmNetwork;
        use crate::{Dense, Linear, MultilayerPerceptron, NeuralNetwork, Relu, Sgd, Sigmoid, Tanh};

        fn weights(seed: &mut u32) -> i32 {
            *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (*seed >> 16) as i32 % 32_769 - 16_384
        }
        let neutral =
            |l: Q16_16| Scheduled::new(WeightDecay::new(Sgd, l), StepDecay::new(1, Q16_16::ONE));

        let (mut s1, mut s2) = (1u32, 1u32);
        let mut plain = MultilayerPerceptron::<Q16_16, Tanh, Sigmoid, 2, 4, 1>::new(|| {
            Q16_16::from_raw(weights(&mut s1))
        });
        let mut wrapped = MultilayerPerceptron::<Q16_16, Tanh, Sigmoid, 2, 4, 1, _>::with_optimizer(
            || Q16_16::from_raw(weights(&mut s2)),
            neutral(Q16_16::ZERO),
        );
        let (mut s3, mut s4) = (2u32, 2u32);
        let mut deep_plain =
            NeuralNetwork::<(Dense<f64, Relu, 2, 3>, Dense<f64, Linear, 3, 1>)>::new(|| {
                f64::from(weights(&mut s3)) / 65_536.0
            });
        let mut deep_wrapped =
            NeuralNetwork::<(Dense<f64, Relu, 2, 3>, Dense<f64, Linear, 3, 1>), _>::with_optimizer(
                || f64::from(weights(&mut s4)) / 65_536.0,
                Scheduled::new(WeightDecay::new(Sgd, 0.0), StepDecay::new(1, 1.0)),
            );
        let (mut s5, mut s6) = (3u32, 3u32);
        let mut lstm_plain = LstmNetwork::<Q16_16, Tanh, Sigmoid, 2, 3, 1>::new(|| {
            Q16_16::from_raw(weights(&mut s5))
        });
        let mut lstm_wrapped = LstmNetwork::<Q16_16, Tanh, Sigmoid, 2, 3, 1, _>::with_optimizer(
            || Q16_16::from_raw(weights(&mut s6)),
            neutral(Q16_16::ZERO),
        );
        for step in 0..200 {
            let (a, b) = (step % 2, (step / 2) % 2);
            let x = [Q16_16::from_int(a), Q16_16::from_int(b)];
            let t = [Q16_16::from_int(a ^ b)];
            plain.feed_forward(&x);
            plain.train(&t);
            wrapped.feed_forward(&x);
            wrapped.train(&t);
            lstm_plain.feed_forward(&x);
            lstm_plain.train(&t);
            lstm_wrapped.feed_forward(&x);
            lstm_wrapped.train(&t);
            let xf = [f64::from(a), f64::from(b)];
            let tf = [f64::from(a - b)];
            deep_plain.feed_forward(&xf);
            deep_plain.train(&tf);
            deep_wrapped.feed_forward(&xf);
            deep_wrapped.train(&tf);
        }
        assert_eq!(plain.input_weights(), wrapped.input_weights());
        assert_eq!(plain.hidden_weights(), wrapped.hidden_weights());
        assert_eq!(plain.learning_rate(), wrapped.learning_rate());
        assert_eq!(lstm_plain.input_weights(), lstm_wrapped.input_weights());
        assert_eq!(
            lstm_plain.recurrent_weights(),
            lstm_wrapped.recurrent_weights()
        );
        assert_eq!(lstm_plain.hidden_weights(), lstm_wrapped.hidden_weights());
        assert_eq!(
            deep_plain.layers().0.weights(),
            deep_wrapped.layers().0.weights()
        );
        assert_eq!(
            deep_plain.layers().1.weights(),
            deep_wrapped.layers().1.weights()
        );

        // And a nonzero lambda does change the weights (bias weights alone
        // are left to the optimizer).
        let mut s7 = 1u32;
        let mut decayed = MultilayerPerceptron::<Q16_16, Tanh, Sigmoid, 2, 4, 1, _>::with_optimizer(
            || Q16_16::from_raw(weights(&mut s7)),
            neutral(Q16_16::from_raw(655)),
        );
        for step in 0..200 {
            let (a, b) = (step % 2, (step / 2) % 2);
            decayed.feed_forward(&[Q16_16::from_int(a), Q16_16::from_int(b)]);
            decayed.train(&[Q16_16::from_int(a ^ b)]);
        }
        assert_ne!(plain.input_weights(), decayed.input_weights());
    }

    /// Wrappers pass the inner optimizer's decay and schedule through, so
    /// nested schedules both apply (inner first) and the outer `WeightDecay`
    /// wins.
    #[test]
    fn wrappers_compose() {
        use crate::Sgd;
        let half = Q16_16::from_raw(1 << 15);
        let quarter = Q16_16::from_raw(1 << 14);
        let mut opt = Scheduled::new(
            WeightDecay::new(Scheduled::new(Sgd, StepDecay::new(1, half)), Q16_16::ONE),
            StepDecay::new(2, quarter),
        );
        let one = Q16_16::ONE;
        assert_eq!(opt.next_learning_rate(one), half); // inner only
        assert_eq!(opt.next_learning_rate(one), Q16_16::from_raw(1 << 13)); // both
        assert_eq!(opt.weight_decay(), Some(Q16_16::ONE));
        let opt = WeightDecay::new(WeightDecay::new(Sgd, half), quarter);
        assert_eq!(opt.weight_decay(), Some(quarter));
        assert_eq!(OptimizerImpl::<Q16_16>::weight_decay(&Sgd), None);
    }

    #[test]
    fn weight_decay_matches_the_cpp_expression() {
        // lr 1/4, lambda 0.01 (655 raw): factor (1/4 * 655/65536) = 163 raw,
        // floored; w = 2: 2 - 163/65536 * 2 = 131072 - 326.
        let factor = Q16_16::from_raw(1 << 14) * Q16_16::from_raw(655);
        assert_eq!(factor.raw(), 163);
        let mut c = [Connection {
            weight: Q16_16::from_int(2),
            ..Connection::default()
        }];
        decay_weights(&mut c, factor);
        assert_eq!(c[0].weight.raw(), 131_072 - 326);
    }
}
