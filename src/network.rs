//! A trainable feed-forward network with any number of hidden layers, ported
//! from tinymind (C++)'s `NeuralNetwork<ValueType, Inputs, HiddenLayers<...>,
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
//! which the compiler checks. Each layer has its own activation; tinymind (C++)'s
//! network uses one activation for every hidden layer, which is the special
//! case where they are all the same. Tuples of 1 to 8 layers are supported.
//!
//! Training is SGD with momentum by default, or Adam or RMSprop ([`crate::optimizer`],
//! through [`NeuralNetwork::with_optimizer`]), with optional gradient clipping
//! ([`NeuralNetwork::set_gradient_clip`]), and every result is bit-identical to the C++
//! network with the same layer sizes, activations and initial weights (and to
//! [`MultilayerPerceptron`](crate::MultilayerPerceptron) for one hidden layer).

use core::marker::PhantomData;

use crate::activation::Activation;
use crate::neuralnet::Connection;
use crate::optimizer::{Optimizer, Sgd};
use crate::qformat::Scalar;

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

impl<V: Scalar, A, const IN: usize, const OUT: usize> Default for Dense<V, A, IN, OUT> {
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

impl<V: Scalar, A: Activation<V>, const IN: usize, const OUT: usize> Dense<V, A, IN, OUT> {
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

    /// One optimizer step on every connection (C++ `updateConnectionWeights`
    /// for this layer): the optimizer's step count advances first, then each
    /// connection is updated. Each update reads only its own connection, so the
    /// order within the layer does not matter. `clip` applies to the weights
    /// only; the bias weights use the raw gradient, as in the C++.
    fn update<Opt: Optimizer<V>>(
        &mut self,
        rates: (V, V, V),
        clip: Option<(V, V)>,
        optimizer: &mut Opt,
    ) {
        optimizer.begin_layer();
        for row in self.weights.iter_mut() {
            for c in row.iter_mut() {
                optimizer.update_weight(c, rates, clip);
            }
        }
        for c in self.bias.iter_mut() {
            optimizer.update_bias(c, rates);
        }
    }
}

/// A fully connected layer holding only its weights: the inference-only form
/// of [`Dense`] (C++ `IsTrainable = false`), for [`InferenceNetwork`].
///
/// One value per connection, no training state, and [`new`](Self::new) is a
/// `const fn`, so trained weights can be a `static` in flash.
pub struct InferenceDense<V, A, const IN: usize, const OUT: usize> {
    /// `weights[i][o]`: from input `i` to neuron `o`, as [`Dense::weights`]
    /// returns them.
    pub weights: [[V; OUT]; IN],
    /// `bias_weights[o]`: the bias input's weight to neuron `o`.
    pub bias_weights: [V; OUT],
    activation: PhantomData<fn() -> A>,
}

impl<V: Copy, A, const IN: usize, const OUT: usize> Clone for InferenceDense<V, A, IN, OUT> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Copy, A, const IN: usize, const OUT: usize> Copy for InferenceDense<V, A, IN, OUT> {}

impl<V, A, const IN: usize, const OUT: usize> InferenceDense<V, A, IN, OUT> {
    /// A layer with these weights, laid out as [`Dense::weights`] and
    /// [`Dense::bias_weights`] return them.
    pub const fn new(weights: [[V; OUT]; IN], bias_weights: [V; OUT]) -> Self {
        Self {
            weights,
            bias_weights,
            activation: PhantomData,
        }
    }
}

impl<V: Scalar, A: Activation<V>, const IN: usize, const OUT: usize> InferenceDense<V, A, IN, OUT> {
    /// `out[o] = A(1 * bias[o] + (0 + in[0] * w[0][o] + in[1] * w[1][o] + ...))`,
    /// the C++ sum order, as [`Dense`] computes it.
    pub fn forward(&self, input: &[V; IN]) -> [V; OUT] {
        let mut out = [V::ZERO; OUT];
        for (o, y) in out.iter_mut().enumerate() {
            let mut sum = V::ONE * self.bias_weights[o];
            let mut layer_sum = V::ZERO;
            for (&x, row) in input.iter().zip(&self.weights) {
                layer_sum += x * row[o];
            }
            sum += layer_sum;
            *y = A::activate(sum);
        }
        out
    }
}

impl<V: Copy, A, const IN: usize, const OUT: usize> Dense<V, A, IN, OUT> {
    /// This layer's weights as an [`InferenceDense`].
    pub fn to_inference(&self) -> InferenceDense<V, A, IN, OUT> {
        InferenceDense::new(self.weights(), self.bias_weights())
    }

    /// Load the weights of an [`InferenceDense`] of the same shape (the
    /// training state is left as it is).
    pub fn set_from(&mut self, from: &InferenceDense<V, A, IN, OUT>) {
        self.set_weights(&from.weights);
        self.set_bias_weights(&from.bias_weights);
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
        fn train<Opt: crate::optimizer::Optimizer<Self::Value>>(
            &mut self,
            inputs: &Self::Inputs,
            targets: &Self::Outputs,
            rates: (Self::Value, Self::Value, Self::Value),
            clip: Option<(Self::Value, Self::Value)>,
            optimizer: &mut Opt,
        );
        /// Number of output neurons: each owns one unused C++ connection whose
        /// initialisation consumes a random draw.
        const OUTPUTS: usize;
        /// The same stack as [`InferenceDense`](super::InferenceDense) layers.
        type Inference: InferenceLayersImpl<Inputs = Self::Inputs, Outputs = Self::Outputs>;
        fn to_inference(&self) -> Self::Inference;
        fn set_from(&mut self, from: &Self::Inference);
    }

    /// The forward pass over a tuple of
    /// [`InferenceDense`](super::InferenceDense) layers whose sizes chain.
    pub trait InferenceLayersImpl {
        type Inputs: Copy;
        type Outputs: Copy;
        fn feed_forward(&self, inputs: &Self::Inputs) -> Self::Outputs;
    }
}

/// A stack of layers [`NeuralNetwork`] can run: a tuple of 1 to 8 [`Dense`]
/// layers in which each layer's `OUT` is the next layer's `IN`. Implemented
/// by the crate only.
pub trait Layers: sealed::LayersImpl {}

impl<T: sealed::LayersImpl> Layers for T {}

/// A stack of layers [`InferenceNetwork`] can run: a tuple of 1 to 8
/// [`InferenceDense`] layers in which each layer's `OUT` is the next layer's
/// `IN`. Implemented by the crate only.
pub trait InferenceLayers: sealed::InferenceLayersImpl {}

impl<T: sealed::InferenceLayersImpl> InferenceLayers for T {}

/// Mean-squared error against `targets` (C++ `MeanSquaredErrorCalculator`):
/// the sum of squared differences, divided by the output count when it is more
/// than one.
fn mean_squared_error<V: Scalar, const OUT: usize>(outputs: &[V; OUT], targets: &[V; OUT]) -> V {
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
        impl<V: Scalar, $($act: Activation<V>,)+ $(const $d: usize,)+> sealed::LayersImpl
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

            fn train<Opt: Optimizer<V>>(
                &mut self,
                inputs: &[V; $d0],
                targets: &[V; $dn],
                rates: (V, V, V),
                clip: Option<(V, V)>,
                optimizer: &mut Opt,
            ) {
                // Deltas from the output back (C++ ChainNetworkDeltasCalculator),
                // all computed before any weight changes.
                self.$last.output_deltas(targets);
                $(
                    let signal = self.$hi.back_signal();
                    self.$lo.hidden_deltas(&signal);
                )*
                // Gradients (ChainNetworkGradientsCalculator), then updates from
                // the output layer back (ChainBackPropConnectionWeightUpdater):
                // the order sets each layer's step count for Adam.
                self.$first.gradients(inputs);
                $(self.$next.gradients(&self.$prev.outputs);)*
                self.$last.update(rates, clip, optimizer);
                $(self.$lo.update(rates, clip, optimizer);)*
            }

            type Inference = ($(InferenceDense<V, $act, $din, $dout>,)+);

            fn to_inference(&self) -> Self::Inference {
                ($(self.$idx.to_inference(),)+)
            }

            fn set_from(&mut self, from: &Self::Inference) {
                $(self.$idx.set_from(&from.$idx);)+
            }
        }

        impl<V: Scalar, $($act: Activation<V>,)+ $(const $d: usize,)+> sealed::InferenceLayersImpl
            for ($(InferenceDense<V, $act, $din, $dout>,)+)
        {
            type Inputs = [V; $d0];
            type Outputs = [V; $dn];

            fn feed_forward(&self, inputs: &[V; $d0]) -> [V; $dn] {
                let x = self.$first.forward(inputs);
                $(let x = self.$next.forward(&x);)*
                x
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
///
/// `Opt` is the optimizer: [`Sgd`] (the default) or an adaptive one from
/// [`optimizer`](crate::optimizer), passed to [`with_optimizer`](Self::with_optimizer).
pub struct NeuralNetwork<L: Layers, Opt = Sgd> {
    layers: L,
    inputs: L::Inputs,
    learning_rate: L::Value,
    momentum_rate: L::Value,
    acceleration_rate: L::Value,
    gradient_clip: Option<(L::Value, L::Value)>,
    optimizer: Opt,
}

impl<V: Scalar, L: Layers<Value = V> + Default> NeuralNetwork<L, Sgd> {
    /// Build the network, trained with [`Sgd`], and initialise its weights from
    /// `random_weight`, in the C++ order: each layer's inputs and then its bias,
    /// from the input side, then one draw per output neuron (the C++ output
    /// neurons own an unused connection whose initialisation consumes a draw).
    pub fn new(random_weight: impl FnMut() -> V) -> Self {
        Self::with_optimizer(random_weight, Sgd)
    }
}

impl<V: Scalar, L: Layers<Value = V> + Default, Opt: Optimizer<V>> NeuralNetwork<L, Opt> {
    /// As [`new`](NeuralNetwork::new), trained with `optimizer`, for example
    /// [`Adam`](crate::Adam).
    pub fn with_optimizer(mut random_weight: impl FnMut() -> V, optimizer: Opt) -> Self {
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
            gradient_clip: None,
            optimizer,
        }
    }
}

impl<V: Scalar, L: Layers<Value = V>, Opt: Optimizer<V>> NeuralNetwork<L, Opt> {
    /// The optimizer, for example to read [`Adam::timestep`](crate::Adam::timestep).
    pub fn optimizer(&self) -> &Opt {
        &self.optimizer
    }

    /// Replace the learning rate (default `2^-2`).
    pub fn set_learning_rate(&mut self, rate: V) {
        self.learning_rate = rate;
    }

    /// Replace the momentum rate (default `2^-1`).
    pub fn set_momentum_rate(&mut self, rate: V) {
        self.momentum_rate = rate;
    }

    /// Replace the acceleration rate, the weight on the delta weight from two
    /// steps back (default `2^-4`).
    pub fn set_acceleration_rate(&mut self, rate: V) {
        self.acceleration_rate = rate;
    }

    /// Clip each weight's gradient to `[min, max]` before the update, or
    /// `None` (the default). C++ `GradientClipByValue<ValueType, I, F>` is
    /// `Some((QValue(-I, F), QValue(I, F)))`; the default
    /// `GradientClipByValue<ValueType>` is `Some((-1.0, 1.0))`. Bias weights are
    /// never clipped, as in the C++.
    pub fn set_gradient_clip(&mut self, range: Option<(V, V)>) {
        self.gradient_clip = range;
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
        self.layers.train(
            &self.inputs,
            targets,
            rates,
            self.gradient_clip,
            &mut self.optimizer,
        );
    }

    /// The layers, to read their weights (`nn.layers().0.weights()`).
    pub fn layers(&self) -> &L {
        &self.layers
    }

    /// The layers, to load weights (`nn.layers_mut().0.set_weights(..)`).
    pub fn layers_mut(&mut self) -> &mut L {
        &mut self.layers
    }

    /// The trained network as an [`InferenceNetwork`]: the same weights and the
    /// same outputs, bit for bit, without the training state.
    pub fn to_inference(&self) -> InferenceNetwork<L::Inference> {
        InferenceNetwork::new(self.layers.to_inference())
    }

    /// Load every weight from an inference network of the same shape, for
    /// example to keep training shipped weights on the device. The optimizer
    /// state and delta weights are left as they are.
    pub fn set_weights(&mut self, from: &InferenceNetwork<L::Inference>) {
        self.layers.set_from(&from.layers);
    }
}

/// An inference-only feed-forward network over a tuple of [`InferenceDense`]
/// layers: tinymind (C++)'s `NeuralNetwork<..., HiddenLayers<...>, ...>` with
/// `IsTrainable = false`, and the any-depth form of
/// [`InferencePerceptron`](crate::InferencePerceptron).
///
/// Train with [`NeuralNetwork`] (on a host or on the device), then ship this:
/// one value per connection, no per-neuron state and no training code. Its
/// outputs are bit-identical to the trainable network's with the same
/// weights. [`feed_forward`](Self::feed_forward) takes `&self`, and
/// [`new`](Self::new) and [`InferenceDense::new`] are `const fn`, so a trained
/// network can be a `static`:
///
/// ```
/// use tinymind::{InferenceDense, InferenceNetwork, Q16_16, Relu, Sigmoid};
///
/// const fn q(x: i32) -> Q16_16 {
///     Q16_16::from_int(x)
/// }
///
/// // 2 inputs -> 2 ReLU -> 2 ReLU -> 1 sigmoid output: XOR by hand (OR minus
/// // twice AND, then a margin). Weights from `NeuralNetwork::to_inference`
/// // go here the same way.
/// static XOR: InferenceNetwork<(
///     InferenceDense<Q16_16, Relu, 2, 2>,
///     InferenceDense<Q16_16, Relu, 2, 2>,
///     InferenceDense<Q16_16, Sigmoid, 2, 1>,
/// )> = InferenceNetwork::new((
///     InferenceDense::new([[q(1), q(1)], [q(1), q(1)]], [q(0), q(-1)]), // [OR-ish, AND]
///     InferenceDense::new([[q(1), q(0)], [q(-2), q(0)]], [q(0), q(0)]), // OR - 2 AND
///     InferenceDense::new([[q(8)], [q(0)]], [q(-4)]),
/// ));
///
/// let half = Q16_16::from_raw(1 << 15);
/// for (a, b, want) in [(0, 0, false), (0, 1, true), (1, 0, true), (1, 1, false)] {
///     let [y] = XOR.feed_forward(&[q(a), q(b)]);
///     assert_eq!(y > half, want);
/// }
/// ```
pub struct InferenceNetwork<L> {
    /// The layers, first to last.
    pub layers: L,
}

impl<L> InferenceNetwork<L> {
    /// A network over these layers.
    pub const fn new(layers: L) -> Self {
        Self { layers }
    }
}

impl<L: InferenceLayers> InferenceNetwork<L> {
    /// One forward pass: the output layer's activations. Nothing is kept
    /// between calls.
    pub fn feed_forward(&self, inputs: &L::Inputs) -> L::Outputs {
        self.layers.feed_forward(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Linear, Relu, Sigmoid, Tanh};
    use crate::rand::GlibcRand;
    use crate::{MultilayerPerceptron, Q8_8, Q16_16};

    /// A two-layer network is the one-hidden-layer MultilayerPerceptron (which
    /// is bit-exact with the C++): the same random draws, outputs, errors and
    /// weights after every training step.
    fn same_as_perceptron<V, H, O, const IN: usize, const HID: usize, const OUT: usize>(
        raw: impl Fn(i32) -> V,
        seed: u32,
    ) where
        V: Scalar + PartialEq + core::fmt::Debug,
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

    /// The same with a linear output, gradient clipping and every rate set:
    /// the settings tinymind (C++)'s energy_efficiency example uses, on random data
    /// large enough that the clip and the unclipped biases both matter.
    #[test]
    fn two_layers_match_perceptron_with_clipping() {
        let mut a = GlibcRand::new(11);
        let mut b = GlibcRand::new(11);
        let w = |r: i32| Q16_16::from_raw(unit(r, 1 << 15));
        let mut mlp =
            MultilayerPerceptron::<Q16_16, Relu, Linear, 3, 5, 2>::new(|| w(a.next_i32()));
        let mut net =
            NeuralNetwork::<(Dense<Q16_16, Relu, 3, 5>, Dense<Q16_16, Linear, 5, 2>)>::new(|| {
                w(b.next_i32())
            });
        let range = (Q16_16::from_raw(-(1 << 15)), Q16_16::ONE);
        let lr = Q16_16::from_raw(1 << 10);
        mlp.set_learning_rate(lr);
        net.set_learning_rate(lr);
        mlp.set_acceleration_rate(Q16_16::from_raw(1 << 12));
        net.set_acceleration_rate(Q16_16::from_raw(1 << 12));
        mlp.set_gradient_clip(Some(range));
        net.set_gradient_clip(Some(range));
        let mut data = GlibcRand::new(12);
        for step in 0..1000 {
            let input: [Q16_16; 3] =
                core::array::from_fn(|_| Q16_16::from_raw(unit(data.next_i32(), 1 << 16)));
            let target: [Q16_16; 2] =
                core::array::from_fn(|_| Q16_16::from_raw(unit(data.next_i32(), 8 << 16)));
            mlp.feed_forward(&input);
            net.feed_forward(&input);
            assert_eq!(net.learned_values(), mlp.learned_values(), "step {step}");
            mlp.train(&target);
            net.train(&target);
            assert_eq!(net.layers().0.weights(), mlp.input_weights(), "step {step}");
            assert_eq!(
                net.layers().1.bias_weights(),
                mlp.hidden_bias_weights(),
                "step {step}"
            );
        }
    }

    #[test]
    fn linear_and_clip() {
        use crate::activation::Linear;
        let x = Q8_8::from_raw(-300);
        assert_eq!(<Linear as Activation<Q8_8>>::activate(x), x);
        assert_eq!(<Linear as Activation<Q8_8>>::derivative(x), Q8_8::ONE);
        let range = (Q16_16::from_int(-1), Q16_16::ONE);
        let clip = |g: i32| crate::neuralnet::clip_gradient(Q16_16::from_int(g), range);
        assert_eq!(clip(5), Q16_16::ONE);
        assert_eq!(clip(-5), Q16_16::from_int(-1));
        assert_eq!(clip(0), Q16_16::ZERO);
        // An inverted range does not panic: max wins above it, min below.
        let inverted = (Q16_16::ONE, Q16_16::ZERO);
        assert_eq!(
            crate::neuralnet::clip_gradient(Q16_16::from_int(3), inverted),
            Q16_16::ZERO
        );
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
