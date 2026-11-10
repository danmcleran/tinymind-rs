//! A multilayer perceptron with one hidden layer, ported from the
//! `NeuralNetwork` / `MultilayerPerceptron` path of tinymind (C++) `cpp/neuralnet.hpp`,
//! in two forms: [`MultilayerPerceptron`] trains on the device, and
//! [`InferencePerceptron`] only runs a trained network (C++ `IsTrainable = false`).
//!
//! The C++ network is assembled from layer, neuron, connection, gradient-manager
//! and training-policy templates, stored in placement-new byte buffers. This port
//! keeps only what the default MLP configuration actually executes
//! (`FixedPointTransferFunctions` with default policies, or tinymind (C++)'s
//! floating-point bundle for `f32`/`f64`, plus optional
//! `GradientClipByValue` and an optional Adam or RMSprop optimizer; batch size 1, no recurrence, one hidden layer; for more
//! hidden layers see [`NeuralNetwork`](crate::NeuralNetwork)) and stores everything
//! in plain fixed-size arrays:
//! no heap, so the crate stays `#![no_std]`.
//!
//! Every arithmetic step keeps the C++ operation order, because fixed-point
//! multiplication truncates and a reordered expression gives different bits.

use core::marker::PhantomData;

use crate::activation::Activation;
use crate::optimizer::{Optimizer, Sgd};
use crate::qformat::Scalar;

/// One weighted edge to a neuron in the next layer (C++ `TrainableConnection`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Connection<V> {
    pub weight: V,
    pub delta_weight: V,
    pub previous_delta_weight: V,
    pub gradient: V,
}

impl<V: Copy> Connection<V> {
    /// C++ `setDeltaWeight`: the old delta becomes the previous delta.
    pub(crate) fn set_delta_weight(&mut self, value: V) {
        self.previous_delta_weight = self.delta_weight;
        self.delta_weight = value;
    }
}

/// A neuron and its outgoing connections (C++ `TrainableNeuron`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neuron<V, const N: usize> {
    pub output: V,
    pub node_delta: V,
    pub connections: [Connection<V>; N],
}

impl<V: Scalar, const N: usize> Default for Neuron<V, N> {
    fn default() -> Self {
        Self {
            output: V::ZERO,
            node_delta: V::ZERO,
            connections: [Connection::default(); N],
        }
    }
}

impl<V: Scalar, const N: usize> Neuron<V, N> {
    /// C++ `TrainableNeuron::initializeWeights`: one random weight per connection,
    /// in connection order, then zero the delta weights.
    pub(crate) fn initialize_weights(&mut self, random_weight: &mut impl FnMut() -> V) {
        for c in &mut self.connections {
            c.weight = random_weight();
        }
        for c in &mut self.connections {
            c.set_delta_weight(V::ZERO);
        }
    }
}

/// `IN` inputs -> `HID` hidden neurons (activation `H`) -> `OUT` outputs (activation `O`).
///
/// C++: `MultilayerPerceptron<ValueType, IN, 1, HID, OUT, TransferFunctions>`.
///
/// `Opt` is the optimizer: [`Sgd`] (the default) or an adaptive one from
/// [`optimizer`](crate::optimizer), passed to [`with_optimizer`](Self::with_optimizer).
pub struct MultilayerPerceptron<
    V,
    H,
    O,
    const IN: usize,
    const HID: usize,
    const OUT: usize,
    Opt = Sgd,
> {
    input: [Neuron<V, HID>; IN],
    input_bias: Neuron<V, HID>,
    hidden: [Neuron<V, OUT>; HID],
    hidden_bias: Neuron<V, OUT>,
    /// Output neurons own one connection that is never used. It is kept because
    /// the C++ output neuron has one, and initialising it consumes a random draw;
    /// dropping it would shift every later random number.
    output: [Neuron<V, 1>; OUT],
    learning_rate: V,
    momentum_rate: V,
    acceleration_rate: V,
    gradient_clip: Option<(V, V)>,
    optimizer: Opt,
    /// `H` and `O` are only used in static calls; `PhantomData` records that the
    /// type "uses" them. `fn() -> _` keeps the struct `Send`/`Sync` regardless.
    activations: PhantomData<fn() -> (H, O)>,
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize>
    MultilayerPerceptron<V, H, O, IN, HID, OUT, Sgd>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
{
    /// Build the network, trained with [`Sgd`], and initialise its weights from
    /// `random_weight`, in the same order as the C++ constructor: input neurons,
    /// input bias, hidden neurons, hidden bias, output neurons.
    ///
    /// The C++ code reaches a static `generateRandomWeight()` policy that reads
    /// global `rand()` state; passing a closure makes that dependency explicit.
    pub fn new(random_weight: impl FnMut() -> V) -> Self {
        Self::with_optimizer(random_weight, Sgd)
    }
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>
    MultilayerPerceptron<V, H, O, IN, HID, OUT, Opt>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
    Opt: Optimizer<V>,
{
    /// As [`new`](MultilayerPerceptron::new), trained with `optimizer`, for
    /// example [`Adam`](crate::Adam).
    pub fn with_optimizer(mut random_weight: impl FnMut() -> V, optimizer: Opt) -> Self {
        let mut nn = Self {
            input: [Neuron::default(); IN],
            input_bias: Neuron::default(),
            hidden: [Neuron::default(); HID],
            hidden_bias: Neuron::default(),
            output: [Neuron::default(); OUT],
            // DefaultNetworkInitializer: 1/4, 1/2, 1/16.
            learning_rate: V::pow2_neg(2),
            momentum_rate: V::pow2_neg(1),
            acceleration_rate: V::pow2_neg(4),
            gradient_clip: None,
            optimizer,
            activations: PhantomData,
        };
        nn.input_bias.output = V::ONE;
        nn.hidden_bias.output = V::ONE;

        for n in &mut nn.input {
            n.initialize_weights(&mut random_weight);
        }
        nn.input_bias.initialize_weights(&mut random_weight);
        for n in &mut nn.hidden {
            n.initialize_weights(&mut random_weight);
        }
        nn.hidden_bias.initialize_weights(&mut random_weight);
        for n in &mut nn.output {
            n.initialize_weights(&mut random_weight);
        }
        nn
    }

    /// Replace the learning rate (default `2^-2`, tinymind (C++)'s
    /// `DefaultNetworkInitializer`).
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

    /// Clip each weight's and bias weight's gradient to `[min, max]` before
    /// the update, or `None` (the default) to leave gradients alone. C++
    /// `GradientClipByValue<ValueType, I, F>` is `Some((-b, b))` with `b =
    /// QValue(I, F)`, so the default `GradientClipByValue<ValueType>` is
    /// `Some((-1.0, 1.0))`.
    pub fn set_gradient_clip(&mut self, range: Option<(V, V)>) {
        self.gradient_clip = range;
    }

    /// Run one forward pass: load `inputs`, then compute the hidden and output
    /// activations. Read the result with [`learned_values`](Self::learned_values).
    pub fn feed_forward(&mut self, inputs: &[V; IN]) {
        for (neuron, &value) in self.input.iter_mut().zip(inputs) {
            neuron.output = value;
        }

        for h in 0..HID {
            let sum = weighted_sum(&self.input_bias, &self.input, h);
            self.hidden[h].output = H::activate(sum);
        }

        for o in 0..OUT {
            let sum = weighted_sum(&self.hidden_bias, &self.hidden, o);
            self.output[o].output = O::activate(sum);
        }
    }

    /// The output-layer activations from the latest
    /// [`feed_forward`](Self::feed_forward).
    pub fn learned_values(&self) -> [V; OUT] {
        self.output.map(|n| n.output)
    }

    /// Mean-squared error against `targets` (C++ `MeanSquaredErrorCalculator`).
    pub fn calculate_error(&self, targets: &[V; OUT]) -> V {
        let mut error = V::ZERO;
        for (target, neuron) in targets.iter().zip(&self.output) {
            let delta = *target - neuron.output;
            error += delta * delta;
        }
        if OUT > 1 {
            error = error.div_int(OUT as i32);
        }
        error
    }

    /// One back-propagation step with the network's optimizer (C++
    /// `ChainBackPropagationPolicy`).
    pub fn train(&mut self, targets: &[V; OUT]) {
        self.calculate_node_deltas(targets);
        self.calculate_gradients();
        self.update_weights();
        // FixedLearningRatePolicy::step is the identity, so there is no LR step.
    }

    fn calculate_node_deltas(&mut self, targets: &[V; OUT]) {
        for (neuron, &target) in self.output.iter_mut().zip(targets) {
            let error = target - neuron.output;
            neuron.node_delta = error * O::derivative(neuron.output);
        }

        for h in 0..HID {
            let mut sum = V::ZERO;
            for o in 0..OUT {
                sum += self.hidden[h].connections[o].weight * self.output[o].node_delta;
            }
            self.hidden[h].node_delta = sum * H::derivative(self.hidden[h].output);
        }
        // The C++ code also computes node deltas for the input layer. Nothing
        // reads them (there is no layer before the input), so they are skipped.
    }

    fn calculate_gradients(&mut self) {
        // Hidden -> output connections, then the hidden bias.
        for h in 0..HID {
            let out = self.hidden[h].output;
            for o in 0..OUT {
                self.hidden[h].connections[o].gradient = out * self.output[o].node_delta;
            }
        }
        for o in 0..OUT {
            self.hidden_bias.connections[o].gradient =
                self.hidden_bias.output * self.output[o].node_delta;
        }

        // Input -> hidden connections, then the input bias.
        for i in 0..IN {
            let out = self.input[i].output;
            for h in 0..HID {
                self.input[i].connections[h].gradient = out * self.hidden[h].node_delta;
            }
        }
        for h in 0..HID {
            self.input_bias.connections[h].gradient =
                self.input_bias.output * self.hidden[h].node_delta;
        }
    }

    fn update_weights(&mut self) {
        let rates = (
            self.learning_rate,
            self.momentum_rate,
            self.acceleration_rate,
        );

        let clip = self.gradient_clip;

        // One optimizer step (C++ `beginTrainingStep`), then the output layer
        // and the hidden layer, each neuron's incoming connections followed by
        // its bias connection, all clipped alike.
        self.optimizer.begin_step();
        for o in 0..OUT {
            for h in 0..HID {
                self.optimizer
                    .update_weight(&mut self.hidden[h].connections[o], rates, clip);
            }
            self.optimizer
                .update_weight(&mut self.hidden_bias.connections[o], rates, clip);
        }
        for h in 0..HID {
            for i in 0..IN {
                self.optimizer
                    .update_weight(&mut self.input[i].connections[h], rates, clip);
            }
            self.optimizer
                .update_weight(&mut self.input_bias.connections[h], rates, clip);
        }
    }
}

/// Weight getters. These need no activation bounds, so they live in their own
/// `impl` block. Each returns a whole matrix rather than taking an index, so
/// none of them can panic; the layout matches the order tinymind (C++)'s
/// `NetworkPropertiesFileManager` writes the weights in.
impl<V: Copy, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>
    MultilayerPerceptron<V, H, O, IN, HID, OUT, Opt>
{
    /// The optimizer, for example to read [`Adam::timestep`](crate::Adam::timestep).
    pub fn optimizer(&self) -> &Opt {
        &self.optimizer
    }

    /// Input-to-hidden weights: `input_weights()[input][hidden]`.
    pub fn input_weights(&self) -> [[V; HID]; IN] {
        self.input.map(|n| n.connections.map(|c| c.weight))
    }

    /// Input-layer bias to hidden neuron weights: `input_bias_weights()[hidden]`.
    pub fn input_bias_weights(&self) -> [V; HID] {
        self.input_bias.connections.map(|c| c.weight)
    }

    /// Hidden-to-output weights: `hidden_weights()[hidden][output]`.
    pub fn hidden_weights(&self) -> [[V; OUT]; HID] {
        self.hidden.map(|n| n.connections.map(|c| c.weight))
    }

    /// Hidden-layer bias to output neuron weights: `hidden_bias_weights()[output]`.
    pub fn hidden_bias_weights(&self) -> [V; OUT] {
        self.hidden_bias.connections.map(|c| c.weight)
    }

    /// Replace the input-to-hidden weights, laid out as
    /// [`input_weights`](Self::input_weights) returns them. Only the weights
    /// change; the training state (delta weights, gradients) is kept, as with
    /// the C++ `setInputLayerWeightForNeuronAndConnection`.
    pub fn set_input_weights(&mut self, weights: &[[V; HID]; IN]) {
        for (neuron, row) in self.input.iter_mut().zip(weights) {
            for (c, &w) in neuron.connections.iter_mut().zip(row) {
                c.weight = w;
            }
        }
    }

    /// Replace the input-layer bias weights (C++
    /// `setInputLayerBiasWeightForConnection`).
    pub fn set_input_bias_weights(&mut self, weights: &[V; HID]) {
        for (c, &w) in self.input_bias.connections.iter_mut().zip(weights) {
            c.weight = w;
        }
    }

    /// Replace the hidden-to-output weights (C++
    /// `setHiddenLayerWeightForNeuronAndConnection`).
    pub fn set_hidden_weights(&mut self, weights: &[[V; OUT]; HID]) {
        for (neuron, row) in self.hidden.iter_mut().zip(weights) {
            for (c, &w) in neuron.connections.iter_mut().zip(row) {
                c.weight = w;
            }
        }
    }

    /// Replace the hidden-layer bias weights (C++
    /// `setHiddenLayerBiasNeuronWeightForConnection`).
    pub fn set_hidden_bias_weights(&mut self, weights: &[V; OUT]) {
        for (c, &w) in self.hidden_bias.connections.iter_mut().zip(weights) {
            c.weight = w;
        }
    }

    /// Load every weight from an inference network of the same shape, for
    /// example to fine-tune shipped weights on the device.
    pub fn set_weights(&mut self, from: &InferencePerceptron<V, H, O, IN, HID, OUT>) {
        self.set_input_weights(&from.input_weights);
        self.set_input_bias_weights(&from.input_bias_weights);
        self.set_hidden_weights(&from.hidden_weights);
        self.set_hidden_bias_weights(&from.hidden_bias_weights);
    }

    /// The trained network as an [`InferencePerceptron`]: the same weights and
    /// the same outputs, bit for bit, without the training state.
    pub fn to_inference(&self) -> InferencePerceptron<V, H, O, IN, HID, OUT> {
        InferencePerceptron::new(
            self.input_weights(),
            self.input_bias_weights(),
            self.hidden_weights(),
            self.hidden_bias_weights(),
        )
    }
}

/// An inference-only multilayer perceptron: `IN` inputs -> `HID` hidden neurons
/// (activation `H`) -> `OUT` outputs (activation `O`), holding only its weights.
///
/// C++: `MultilayerPerceptron<ValueType, IN, 1, HID, OUT, TransferFunctions,
/// false>`, the untrainable network. Train offline, or on a host with
/// [`MultilayerPerceptron`], then ship this: it stores one value per connection
/// instead of four, has no per-neuron state, and has no training code. Its
/// outputs are bit-identical to the trainable network's with the same weights.
///
/// [`feed_forward`](Self::feed_forward) takes `&self` and keeps nothing between
/// calls, and [`new`](Self::new) is a `const fn`, so a trained network can be a
/// `static` (in flash on a microcontroller):
///
/// ```
/// use tinymind::{InferencePerceptron, Q8_8, Sigmoid, Tanh};
///
/// const fn q(x: i16) -> Q8_8 {
///     Q8_8::from_int(x as i32)
/// }
///
/// // XOR by hand: hidden neuron 0 is OR, hidden neuron 1 is AND, and the output
/// // is "OR and not AND". Trained weights, e.g. from
/// // `MultilayerPerceptron::to_inference` on a host, go here the same way.
/// static XOR: InferencePerceptron<Q8_8, Tanh, Sigmoid, 2, 2, 1> = InferencePerceptron::new(
///     [[q(4), q(4)], [q(4), q(4)]], // input -> hidden
///     [q(-2), q(-6)],               // input bias -> hidden
///     [[q(6)], [q(-6)]],            // hidden -> output
///     [q(-3)],                      // hidden bias -> output
/// );
///
/// let half = Q8_8::from_raw(128);
/// for (a, b, want) in [(0, 0, false), (0, 1, true), (1, 0, true), (1, 1, false)] {
///     let [y] = XOR.feed_forward(&[q(a), q(b)]);
///     assert_eq!(y > half, want);
/// }
/// ```
pub struct InferencePerceptron<V, H, O, const IN: usize, const HID: usize, const OUT: usize> {
    /// Input-to-hidden weights: `input_weights[input][hidden]`.
    pub input_weights: [[V; HID]; IN],
    /// Input-layer bias to hidden neuron weights: `input_bias_weights[hidden]`.
    pub input_bias_weights: [V; HID],
    /// Hidden-to-output weights: `hidden_weights[hidden][output]`.
    pub hidden_weights: [[V; OUT]; HID],
    /// Hidden-layer bias to output neuron weights: `hidden_bias_weights[output]`.
    pub hidden_bias_weights: [V; OUT],
    activations: PhantomData<fn() -> (H, O)>,
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize>
    InferencePerceptron<V, H, O, IN, HID, OUT>
{
    /// A network with these weights, laid out as the
    /// [`MultilayerPerceptron`] weight getters return them.
    pub const fn new(
        input_weights: [[V; HID]; IN],
        input_bias_weights: [V; HID],
        hidden_weights: [[V; OUT]; HID],
        hidden_bias_weights: [V; OUT],
    ) -> Self {
        Self {
            input_weights,
            input_bias_weights,
            hidden_weights,
            hidden_bias_weights,
            activations: PhantomData,
        }
    }
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize>
    InferencePerceptron<V, H, O, IN, HID, OUT>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
{
    /// One forward pass: the output-layer activations for `inputs`.
    pub fn feed_forward(&self, inputs: &[V; IN]) -> [V; OUT] {
        // The same sums, in the same order, as the trainable network: the bias
        // neuron's output (1) times its weight, plus the layer sum from zero.
        let mut hidden = [V::ZERO; HID];
        for (h, out) in hidden.iter_mut().enumerate() {
            let mut sum = V::ONE * self.input_bias_weights[h];
            let mut layer_sum = V::ZERO;
            for (&x, row) in inputs.iter().zip(&self.input_weights) {
                layer_sum += x * row[h];
            }
            sum += layer_sum;
            *out = H::activate(sum);
        }

        let mut output = [V::ZERO; OUT];
        for (o, out) in output.iter_mut().enumerate() {
            let mut sum = V::ONE * self.hidden_bias_weights[o];
            let mut layer_sum = V::ZERO;
            for (&x, row) in hidden.iter().zip(&self.hidden_weights) {
                layer_sum += x * row[o];
            }
            sum += layer_sum;
            *out = O::activate(sum);
        }
        output
    }
}

// Written out rather than derived: a derive would also require the activation
// marker types `H` and `O` to implement each trait.
impl<V: Copy, H, O, const IN: usize, const HID: usize, const OUT: usize> Clone
    for InferencePerceptron<V, H, O, IN, HID, OUT>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Copy, H, O, const IN: usize, const HID: usize, const OUT: usize> Copy
    for InferencePerceptron<V, H, O, IN, HID, OUT>
{
}

impl<V: PartialEq, H, O, const IN: usize, const HID: usize, const OUT: usize> PartialEq
    for InferencePerceptron<V, H, O, IN, HID, OUT>
{
    fn eq(&self, other: &Self) -> bool {
        self.input_weights == other.input_weights
            && self.input_bias_weights == other.input_bias_weights
            && self.hidden_weights == other.hidden_weights
            && self.hidden_bias_weights == other.hidden_bias_weights
    }
}

impl<V: Eq, H, O, const IN: usize, const HID: usize, const OUT: usize> Eq
    for InferencePerceptron<V, H, O, IN, HID, OUT>
{
}

impl<V: core::fmt::Debug, H, O, const IN: usize, const HID: usize, const OUT: usize>
    core::fmt::Debug for InferencePerceptron<V, H, O, IN, HID, OUT>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InferencePerceptron")
            .field("input_weights", &self.input_weights)
            .field("input_bias_weights", &self.input_bias_weights)
            .field("hidden_weights", &self.hidden_weights)
            .field("hidden_bias_weights", &self.hidden_bias_weights)
            .finish()
    }
}

/// `bias.output * bias.w[target] + sum(layer[n].output * layer[n].w[target])`,
/// summed in the C++ order: the layer sum starts from zero, then is added to the
/// bias term.
fn weighted_sum<V: Scalar, const N: usize, const C: usize>(
    bias: &Neuron<V, C>,
    layer: &[Neuron<V, C>; N],
    target: usize,
) -> V {
    // Callers pass target < C. The lookup is total anyway (an out-of-range
    // target would read 0), because when this function is not inlined, as at
    // opt-level s and z, the compiler cannot see the callers' bound and would
    // keep a bounds-check panic.
    let weight = |n: &Neuron<V, C>| n.connections.get(target).map_or(V::ZERO, |c| c.weight);
    let mut sum = bias.output * weight(bias);
    let mut layer_sum = V::ZERO;
    for neuron in layer {
        layer_sum += neuron.output * weight(neuron);
    }
    sum += layer_sum;
    sum
}

/// SGD with momentum and "acceleration" (C++ `updateConnectionWeightsImpl`, SGD path):
/// `dw = lr * g + momentum * dw[t-1] + acceleration * dw[t-2]`, `w += dw`.
///
/// `clip` is the gradient clipping range, `(min, max)` (C++
/// `GradientClipByValue`), applied to weights and bias weights alike.
pub(crate) fn update_connection<V: Scalar>(
    c: &mut Connection<V>,
    (lr, momentum, acceleration): (V, V, V),
    clip: Option<(V, V)>,
) {
    let gradient = match clip {
        Some(range) => clip_gradient(c.gradient, range),
        None => c.gradient,
    };
    let new_delta_weight =
        (lr * gradient) + (momentum * c.delta_weight) + (acceleration * c.previous_delta_weight);
    let current_weight = c.weight;
    c.set_delta_weight(new_delta_weight);
    c.weight = current_weight + new_delta_weight;
}

/// C++ `GradientClipByValue::clip`: two tests, max first, so a range with
/// `min > max` gives `max` for large gradients and `min` for small ones rather
/// than panicking as `clamp` would.
pub(crate) fn clip_gradient<V: Scalar>(gradient: V, (min, max): (V, V)) -> V {
    if gradient > max {
        max
    } else if gradient < min {
        min
    } else {
        gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Relu, Sigmoid, Tanh};
    use crate::rand::GlibcRand;
    use crate::{Q8_8, Q16_16};

    /// The inference network built from a trainable one gives the trainable
    /// network's outputs bit for bit, for random weights and inputs. `raw` maps
    /// a random `i32` to a value; `wide` draws from the whole storage range, so
    /// the sums wrap too.
    fn same_outputs<V, H, O, const IN: usize, const HID: usize, const OUT: usize>(
        raw: impl Fn(i32) -> V,
        seed: u32,
    ) where
        V: Scalar + PartialEq + core::fmt::Debug,
        H: Activation<V>,
        O: Activation<V>,
    {
        let mut rng = GlibcRand::new(seed);
        for trial in 0..50 {
            let mut nn = MultilayerPerceptron::<V, H, O, IN, HID, OUT>::new(|| raw(rng.next_i32()));
            let frozen = nn.to_inference();
            for _ in 0..20 {
                let input: [V; IN] = core::array::from_fn(|_| raw(rng.next_i32()));
                nn.feed_forward(&input);
                assert_eq!(
                    frozen.feed_forward(&input),
                    nn.learned_values(),
                    "trial {trial}"
                );
            }
        }
    }

    /// A value within +-2.0 (`one` is the format's 1.0 in raw units).
    fn small(r: i32, one: i32) -> i32 {
        r % (4 * one + 1) - 2 * one
    }

    #[test]
    fn inference_matches_trainable_q16_16() {
        let small_q = |r: i32| Q16_16::from_raw(small(r, 1 << 16));
        let wide_q = |r: i32| Q16_16::from_raw(r.wrapping_mul(2_654_435_761u32 as i32));
        same_outputs::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(small_q, 1);
        same_outputs::<Q16_16, Relu, Sigmoid, 4, 8, 3>(small_q, 2);
        same_outputs::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(wide_q, 3);
        same_outputs::<Q16_16, Relu, Sigmoid, 4, 8, 3>(wide_q, 4);
    }

    #[test]
    fn inference_matches_trainable_q8_8() {
        let small_q = |r: i32| Q8_8::from_raw(small(r, 1 << 8) as i16);
        let wide_q = |r: i32| Q8_8::from_raw(r as i16);
        same_outputs::<Q8_8, Tanh, Sigmoid, 2, 4, 1>(small_q, 5);
        same_outputs::<Q8_8, Relu, Sigmoid, 4, 8, 3>(small_q, 6);
        same_outputs::<Q8_8, Tanh, Sigmoid, 2, 4, 1>(wide_q, 7);
        same_outputs::<Q8_8, Relu, Sigmoid, 4, 8, 3>(wide_q, 8);
    }

    /// Mid-training weights convert too, and `set_weights` loads them back.
    #[test]
    fn train_freeze_and_reload() {
        type Net = MultilayerPerceptron<Q16_16, Tanh, Sigmoid, 2, 4, 1>;
        let mut rng = GlibcRand::new(9);
        let mut nn = Net::new(|| Q16_16::from_raw(small(rng.next_i32(), 1 << 16)));
        let samples = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];
        for step in 0..2000 {
            let (a, b, y) = samples[step % 4];
            nn.feed_forward(&[Q16_16::from_int(a), Q16_16::from_int(b)]);
            nn.train(&[Q16_16::from_int(y)]);
        }
        let frozen = nn.to_inference();
        for (a, b, _) in samples {
            let input = [Q16_16::from_int(a), Q16_16::from_int(b)];
            nn.feed_forward(&input);
            assert_eq!(frozen.feed_forward(&input), nn.learned_values());
        }

        let mut other = Net::new(|| Q16_16::ZERO);
        other.set_weights(&frozen);
        assert_eq!(other.to_inference(), frozen);
        assert_eq!(other.input_weights(), frozen.input_weights);
        assert_eq!(other.hidden_bias_weights(), frozen.hidden_bias_weights);
    }

    /// One value per connection and no per-neuron state.
    #[test]
    fn inference_network_is_smaller() {
        use core::mem::size_of;
        type Trainable = MultilayerPerceptron<Q16_16, Tanh, Sigmoid, 2, 4, 1>;
        type Inference = InferencePerceptron<Q16_16, Tanh, Sigmoid, 2, 4, 1>;
        // 2*4 + 4 + 4*1 + 1 = 17 weights of 4 bytes.
        assert_eq!(size_of::<Inference>(), 17 * 4);
        assert!(size_of::<Trainable>() > 4 * size_of::<Inference>());
    }
}
