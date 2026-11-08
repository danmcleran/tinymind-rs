//! A trainable feed-forward network with any number of hidden layers, ported
//! from tinymind's `NeuralNetwork<ValueType, Inputs, HiddenLayers<...>,
//! Outputs, ...>` (`cpp/neuralnet.hpp`, the chain-based path).
//!
//! The C++ network takes its hidden-layer sizes as a variadic template
//! (`HiddenLayers<8, 6>`). Rust has no variadic generics, so here a network is
//! a tuple of [`Dense`] layers, each sized by const generics:
//!
//! ```
//! use tinymind::{Dense, NeuralNetwork, Q16_16, Relu, Sigmoid};
//!
//! // 4 inputs -> 8 ReLU -> 6 ReLU -> 3 sigmoid outputs.
//! type Net = NeuralNetwork<(
//!     Dense<Q16_16, Relu, 4, 8>,
//!     Dense<Q16_16, Relu, 8, 6>,
//!     Dense<Q16_16, Sigmoid, 6, 3>,
//! )>;
//!
//! let mut seed = 1u32;
//! let mut nn = Net::new(|| {
//!     // Any source of small initial weights; here a tiny LCG in [-0.5, 0.5].
//!     seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
//!     Q16_16::from_raw((seed >> 16) as i32 % 32_769 - 16_384)
//! });
//! let x = [Q16_16::ONE, Q16_16::ZERO, Q16_16::ONE, Q16_16::ZERO];
//! let target = [Q16_16::ONE, Q16_16::ZERO, Q16_16::ZERO];
//! for _ in 0..100 {
//!     nn.feed_forward(&x);
//!     nn.train(&target);
//! }
//! nn.feed_forward(&x);
//! let [a, b, c] = nn.learned_values();
//! assert!(a > b && a > c);
//! ```
//!
//! Adjacent layers must agree (`Dense<_, _, 4, 8>` then `Dense<_, _, 8, _>`),
//! which the compiler checks. Each layer has its own activation; tinymind's
//! network uses one activation for every hidden layer, which is the special
//! case where they are all the same. Tuples of 1 to 8 layers are supported.
//!
//! Training is SGD with momentum, and every result is bit-identical to the C++
//! network with the same layer sizes, activations and initial weights (and to
//! [`MultilayerPerceptron`](crate::MultilayerPerceptron) for one hidden layer).

use core::marker::PhantomData;

use crate::activation::Activation;
use crate::neuralnet::{Connection, update_connection};
use crate::qformat::Fixed;

/// A fully connected layer: `IN` inputs (plus a bias) to `OUT` neurons with
/// activation `A`, and the training state of every connection.
///
/// A layer is a building block for [`NeuralNetwork`]; on its own it only
/// exposes its weights.
pub struct Dense<V, A, const IN: usize, const OUT: usize> {
    /// `weights[i][o]`: from input `i` to neuron `o`, stored by source as the
    /// C++ stores a neuron's outgoing connections.
    weights: [[Connection<V>; OUT]; IN],
    /// From the bias input (always 1) to neuron `o`.
    bias: [Connection<V>; OUT],
    outputs: [V; OUT],
    deltas: [V; OUT],
    activation: PhantomData<fn() -> A>,
}

impl<V: Fixed, A, const IN: usize, const OUT: usize> Default for Dense<V, A, IN, OUT> {
    fn default() -> Self {
        Self {
            weights: [[Connection::default(); OUT]; IN],
            bias: [Connection::default(); OUT],
            outputs: [V::ZERO; OUT],
            deltas: [V::ZERO; OUT],
            activation: PhantomData,
        }
    }
}

impl<V: Copy, A, const IN: usize, const OUT: usize> Dense<V, A, IN, OUT> {
    /// The weights: `weights()[input][neuron]`.
    pub fn weights(&self) -> [[V; OUT]; IN] {
        self.weights.map(|row| row.map(|c| c.weight))
    }

    /// The bias weights: `bias_weights()[neuron]`.
    pub fn bias_weights(&self) -> [V; OUT] {
        self.bias.map(|c| c.weight)
    }

    /// Replace the weights, laid out as [`weights`](Self::weights) returns
    /// them. The training state (delta weights, gradients) is kept.
    pub fn set_weights(&mut self, weights: &[[V; OUT]; IN]) {
        for (row, new) in self.weights.iter_mut().zip(weights) {
            for (c, &w) in row.iter_mut().zip(new) {
                c.weight = w;
            }
        }
    }

    /// Replace the bias weights.
    pub fn set_bias_weights(&mut self, weights: &[V; OUT]) {
        for (c, &w) in self.bias.iter_mut().zip(weights) {
            c.weight = w;
        }
    }

    /// The neurons' activations from the latest forward pass.
    pub fn outputs(&self) -> [V; OUT] {
        self.outputs
    }
}

impl<V: Fixed, A: Activation<V>, const IN: usize, const OUT: usize> Dense<V, A, IN, OUT> {
    /// C++ `initializeWeights` for this layer's sources: each input's outgoing
    /// connections in order (one random draw each, then the delta weights
    /// zeroed), then the bias neuron's.
    fn initialize(&mut self, random_weight: &mut dyn FnMut() -> V) {
        for row in self
            .weights
            .iter_mut()
            .chain(core::iter::once(&mut self.bias))
        {
            for c in row.iter_mut() {
                c.weight = random_weight();
            }
            for c in row.iter_mut() {
                c.set_delta_weight(V::ZERO);
            }
        }
    }

    /// `out[o] = A(1 * bias[o] + (0 + in[0] * w[0][o] + in[1] * w[1][o] + ...))`,
    /// the C++ sum order.
    fn forward(&mut self, input: &[V; IN]) {
        for (o, out) in self.outputs.iter_mut().enumerate() {
            let mut sum = V::ONE * self.bias[o].weight;
            let mut layer_sum = V::ZERO;
            for (&x, row) in input.iter().zip(&self.weights) {
                layer_sum += x * row[o].weight;
            }
            sum += layer_sum;
            *out = A::activate(sum);
        }
    }

    /// Output-layer deltas: `(target - out) * A'(out)`.
    fn output_deltas(&mut self, targets: &[V; OUT]) {
        for ((delta, &out), &target) in self.deltas.iter_mut().zip(&self.outputs).zip(targets) {
            *delta = (target - out) * A::derivative(out);
        }
    }

    /// Hidden-layer deltas from the error the next layer sends back:
    /// `signal * A'(out)`.
    fn hidden_deltas(&mut self, signal: &[V; OUT]) {
        for ((delta, &out), &s) in self.deltas.iter_mut().zip(&self.outputs).zip(signal) {
            *delta = s * A::derivative(out);
        }
    }

    /// The error this layer sends back to each input: `sum_o w[i][o] *
    /// delta[o]`, summed from zero in neuron order (C++ `calculateAndSetNodeDeltas`).
    fn back_signal(&self) -> [V; IN] {
        self.weights.map(|row| {
            let mut sum = V::ZERO;
            for (c, &delta) in row.iter().zip(&self.deltas) {
                sum += c.weight * delta;
            }
            sum
        })
    }

    /// `gradient = source output * delta`; the bias input is 1.
    fn gradients(&mut self, input: &[V; IN]) {
        for (row, &x) in self.weights.iter_mut().zip(input) {
            for (c, &delta) in row.iter_mut().zip(&self.deltas) {
                c.gradient = x * delta;
            }
        }
        for (c, &delta) in self.bias.iter_mut().zip(&self.deltas) {
            c.gradient = V::ONE * delta;
        }
    }

    /// One SGD-with-momentum step on every connection. Each update reads only
    /// its own connection, so the order does not matter.
    fn update(&mut self, rates: (V, V, V)) {
        for row in self
            .weights
            .iter_mut()
            .chain(core::iter::once(&mut self.bias))
        {
            for c in row.iter_mut() {
                update_connection(c, rates);
            }
        }
    }
}

mod sealed {
    /// The per-network operations, implemented for tuples of [`Dense`] layers
    /// whose sizes chain. Sealed: the set of layer types is the crate's to grow.
    ///
    /// [`Dense`]: super::Dense
    pub trait LayersImpl {
        type Value: Copy;
        type Inputs: Copy;
        type Outputs: Copy;

        fn zero_inputs() -> Self::Inputs;
        fn initialize(&mut self, random_weight: &mut dyn FnMut() -> Self::Value);
        fn feed_forward(&mut self, inputs: &Self::Inputs);
        fn outputs(&self) -> Self::Outputs;
        fn calculate_error(&self, targets: &Self::Outputs) -> Self::Value;
        fn train(
            &mut self,
            inputs: &Self::Inputs,
            targets: &Self::Outputs,
            rates: (Self::Value, Self::Value, Self::Value),
        );
        /// Number of output neurons: each owns one unused C++ connection whose
        /// initialisation consumes a random draw.
        const OUTPUTS: usize;
    }
}

/// A stack of layers [`NeuralNetwork`] can run: a tuple of 1 to 8 [`Dense`]
/// layers in which each layer's `OUT` is the next layer's `IN`. Implemented
/// by the crate only.
pub trait Layers: sealed::LayersImpl {}

impl<T: sealed::LayersImpl> Layers for T {}

/// Mean-squared error against `targets` (C++ `MeanSquaredErrorCalculator`):
/// the sum of squared differences, divided by the output count when it is more
/// than one.
fn mean_squared_error<V: Fixed, const OUT: usize>(outputs: &[V; OUT], targets: &[V; OUT]) -> V {
    let mut error = V::ZERO;
    for (&target, &out) in targets.iter().zip(outputs) {
        let delta = target - out;
        error += delta * delta;
    }
    if OUT > 1 {
        error = error.div_int(OUT as i32);
    }
    error
}

/// Implements the layer-stack operations for one tuple arity. `fwd` lists the
/// adjacent pairs `prev > next` from the input side, `back` the same pairs from
/// the output side.
macro_rules! impl_layers {
    (
        [$($idx:tt: $act:ident, $din:ident, $dout:ident);+]
        first $first:tt, $d0:ident; last $last:tt, $dn:ident;
        fwd [$($prev:tt > $next:tt),*]; back [$($hi:tt > $lo:tt),*];
        consts [$($d:ident),+]
    ) => {
        impl<V: Fixed, $($act: Activation<V>,)+ $(const $d: usize,)+> sealed::LayersImpl
            for ($(Dense<V, $act, $din, $dout>,)+)
        {
            type Value = V;
            type Inputs = [V; $d0];
            type Outputs = [V; $dn];
            const OUTPUTS: usize = $dn;

            fn zero_inputs() -> [V; $d0] {
                [V::ZERO; $d0]
            }

            fn initialize(&mut self, random_weight: &mut dyn FnMut() -> V) {
                $(self.$idx.initialize(random_weight);)+
            }

            fn feed_forward(&mut self, inputs: &[V; $d0]) {
                self.$first.forward(inputs);
                $(self.$next.forward(&self.$prev.outputs);)*
            }

            fn outputs(&self) -> [V; $dn] {
                self.$last.outputs
            }

            fn calculate_error(&self, targets: &[V; $dn]) -> V {
                mean_squared_error(&self.$last.outputs, targets)
            }

            fn train(&mut self, inputs: &[V; $d0], targets: &[V; $dn], rates: (V, V, V)) {
                // Deltas from the output back (C++ ChainNetworkDeltasCalculator),
                // all computed before any weight changes.
                self.$last.output_deltas(targets);
                $(
                    let signal = self.$hi.back_signal();
                    self.$lo.hidden_deltas(&signal);
                )*
                // Gradients (ChainNetworkGradientsCalculator), then updates.
                self.$first.gradients(inputs);
                $(self.$next.gradients(&self.$prev.outputs);)*
                $(self.$idx.update(rates);)+
            }
        }
    };
}

impl_layers!([0: A0, D0, D1] first 0, D0; last 0, D1; fwd []; back []; consts [D0, D1]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2] first 0, D0; last 1, D2;
    fwd [0 > 1]; back [1 > 0]; consts [D0, D1, D2]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2; 2: A2, D2, D3] first 0, D0; last 2, D3;
    fwd [0 > 1, 1 > 2]; back [2 > 1, 1 > 0]; consts [D0, D1, D2, D3]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2; 2: A2, D2, D3; 3: A3, D3, D4] first 0, D0; last 3, D4;
    fwd [0 > 1, 1 > 2, 2 > 3]; back [3 > 2, 2 > 1, 1 > 0]; consts [D0, D1, D2, D3, D4]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2; 2: A2, D2, D3; 3: A3, D3, D4; 4: A4, D4, D5]
    first 0, D0; last 4, D5;
    fwd [0 > 1, 1 > 2, 2 > 3, 3 > 4]; back [4 > 3, 3 > 2, 2 > 1, 1 > 0];
    consts [D0, D1, D2, D3, D4, D5]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2; 2: A2, D2, D3; 3: A3, D3, D4; 4: A4, D4, D5; 5: A5, D5, D6]
    first 0, D0; last 5, D6;
    fwd [0 > 1, 1 > 2, 2 > 3, 3 > 4, 4 > 5]; back [5 > 4, 4 > 3, 3 > 2, 2 > 1, 1 > 0];
    consts [D0, D1, D2, D3, D4, D5, D6]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2; 2: A2, D2, D3; 3: A3, D3, D4; 4: A4, D4, D5; 5: A5, D5, D6;
    6: A6, D6, D7]
    first 0, D0; last 6, D7;
    fwd [0 > 1, 1 > 2, 2 > 3, 3 > 4, 4 > 5, 5 > 6]; back [6 > 5, 5 > 4, 4 > 3, 3 > 2, 2 > 1, 1 > 0];
    consts [D0, D1, D2, D3, D4, D5, D6, D7]);
impl_layers!([0: A0, D0, D1; 1: A1, D1, D2; 2: A2, D2, D3; 3: A3, D3, D4; 4: A4, D4, D5; 5: A5, D5, D6;
    6: A6, D6, D7; 7: A7, D7, D8]
    first 0, D0; last 7, D8;
    fwd [0 > 1, 1 > 2, 2 > 3, 3 > 4, 4 > 5, 5 > 6, 6 > 7];
    back [7 > 6, 6 > 5, 5 > 4, 4 > 3, 3 > 2, 2 > 1, 1 > 0];
    consts [D0, D1, D2, D3, D4, D5, D6, D7, D8]);

/// A trainable feed-forward network over a tuple of [`Dense`] layers (C++
/// `NeuralNetwork<..., HiddenLayers<...>, ...>`). See the [module
/// documentation](self) for an example.
pub struct NeuralNetwork<L: Layers> {
    layers: L,
    inputs: L::Inputs,
    learning_rate: L::Value,
    momentum_rate: L::Value,
    acceleration_rate: L::Value,
}

impl<V: Fixed, L: Layers<Value = V> + Default> NeuralNetwork<L> {
    /// Build the network and initialise its weights from `random_weight`, in
    /// the C++ order: each layer's inputs and then its bias, from the input
    /// side, then one draw per output neuron (the C++ output neurons own an
    /// unused connection whose initialisation consumes a draw).
    pub fn new(mut random_weight: impl FnMut() -> V) -> Self {
        let mut layers = L::default();
        layers.initialize(&mut random_weight);
        for _ in 0..L::OUTPUTS {
            random_weight();
        }
        Self {
            layers,
            inputs: L::zero_inputs(),
            // DefaultNetworkInitializer: 1/4, 1/2, 1/16.
            learning_rate: V::pow2_neg(2),
            momentum_rate: V::pow2_neg(1),
            acceleration_rate: V::pow2_neg(4),
        }
    }
}

impl<V: Fixed, L: Layers<Value = V>> NeuralNetwork<L> {
    /// Replace the learning rate (default `2^-2`).
    pub fn set_learning_rate(&mut self, rate: V) {
        self.learning_rate = rate;
    }

    /// Replace the momentum rate (default `2^-1`).
    pub fn set_momentum_rate(&mut self, rate: V) {
        self.momentum_rate = rate;
    }

    /// Run one forward pass. Read the result with
    /// [`learned_values`](Self::learned_values).
    pub fn feed_forward(&mut self, inputs: &L::Inputs) {
        self.inputs = *inputs;
        self.layers.feed_forward(inputs);
    }

    /// The output layer's activations from the latest forward pass.
    pub fn learned_values(&self) -> L::Outputs {
        self.layers.outputs()
    }

    /// Mean-squared error of the latest forward pass against `targets` (C++
    /// `MeanSquaredErrorCalculator`).
    pub fn calculate_error(&self, targets: &L::Outputs) -> V {
        self.layers.calculate_error(targets)
    }

    /// One back-propagation step towards `targets` for the latest forward pass
    /// (C++ `ChainBackPropagationPolicy`).
    pub fn train(&mut self, targets: &L::Outputs) {
        let rates = (
            self.learning_rate,
            self.momentum_rate,
            self.acceleration_rate,
        );
        self.layers.train(&self.inputs, targets, rates);
    }

    /// The layers, to read their weights (`nn.layers().0.weights()`).
    pub fn layers(&self) -> &L {
        &self.layers
    }

    /// The layers, to load weights (`nn.layers_mut().0.set_weights(..)`).
    pub fn layers_mut(&mut self) -> &mut L {
        &mut self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Relu, Sigmoid, Tanh};
    use crate::rand::GlibcRand;
    use crate::{MultilayerPerceptron, Q8_8, Q16_16};

    /// A two-layer network is the one-hidden-layer MultilayerPerceptron (which
    /// is bit-exact with the C++): the same random draws, outputs, errors and
    /// weights after every training step.
    fn same_as_perceptron<V, H, O, const IN: usize, const HID: usize, const OUT: usize>(
        raw: impl Fn(i32) -> V,
        seed: u32,
    ) where
        V: Fixed + PartialEq + core::fmt::Debug,
        H: Activation<V>,
        O: Activation<V>,
    {
        let mut a = GlibcRand::new(seed);
        let mut b = GlibcRand::new(seed);
        let mut mlp = MultilayerPerceptron::<V, H, O, IN, HID, OUT>::new(|| raw(a.next_i32()));
        let mut net = NeuralNetwork::<(Dense<V, H, IN, HID>, Dense<V, O, HID, OUT>)>::new(|| {
            raw(b.next_i32())
        });
        assert_eq!(a.next_i32(), b.next_i32(), "same number of draws");

        let mut data = GlibcRand::new(seed + 1000);
        for step in 0..500 {
            let input: [V; IN] = core::array::from_fn(|_| raw(data.next_i32()));
            let target: [V; OUT] = core::array::from_fn(|_| raw(data.next_i32()));
            mlp.feed_forward(&input);
            net.feed_forward(&input);
            assert_eq!(net.learned_values(), mlp.learned_values(), "step {step}");
            assert_eq!(net.calculate_error(&target), mlp.calculate_error(&target));
            mlp.train(&target);
            net.train(&target);
            let (l0, l1) = net.layers();
            assert_eq!(l0.weights(), mlp.input_weights(), "step {step}");
            assert_eq!(l0.bias_weights(), mlp.input_bias_weights());
            assert_eq!(l1.weights(), mlp.hidden_weights());
            assert_eq!(l1.bias_weights(), mlp.hidden_bias_weights());
        }
    }

    /// Within +-1.0 (`one` is the format's 1.0 in raw units).
    fn unit(r: i32, one: i32) -> i32 {
        r % (2 * one + 1) - one
    }

    #[test]
    fn two_layers_match_multilayer_perceptron() {
        let q16 = |r: i32| Q16_16::from_raw(unit(r, 1 << 16));
        let q8 = |r: i32| Q8_8::from_raw(unit(r, 1 << 8) as i16);
        same_as_perceptron::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(q16, 1);
        same_as_perceptron::<Q16_16, Relu, Sigmoid, 4, 8, 3>(q16, 2);
        same_as_perceptron::<Q8_8, Tanh, Sigmoid, 2, 4, 1>(q8, 3);
        same_as_perceptron::<Q8_8, Relu, Sigmoid, 4, 8, 3>(q8, 4);
    }

    /// Setters and getters round-trip; the training state is untouched.
    #[test]
    fn weights_round_trip() {
        let mut net =
            NeuralNetwork::<(Dense<Q16_16, Relu, 2, 3>, Dense<Q16_16, Sigmoid, 3, 1>)>::new(|| {
                Q16_16::ZERO
            });
        let w = [[Q16_16::ONE; 3], [Q16_16::from_int(-2); 3]];
        net.layers_mut().0.set_weights(&w);
        net.layers_mut().1.set_bias_weights(&[Q16_16::from_int(3)]);
        assert_eq!(net.layers().0.weights(), w);
        assert_eq!(net.layers().1.bias_weights(), [Q16_16::from_int(3)]);
    }
}
