//! Recurrent networks, ported from tinymind (C++)'s `cpp/neuralnet.hpp`:
//! [`LstmNetwork`] is tinymind (C++)'s `LstmNeuralNetwork` with one LSTM hidden
//! layer.
//!
//! An LSTM network reads a sequence one step at a time: each
//! [`feed_forward`](LstmNetwork::feed_forward) takes one input vector, updates
//! the hidden layer's state from it and from the previous step's outputs, and
//! produces the outputs for that step. [`reset_state`](LstmNetwork::reset_state)
//! clears the state between independent sequences.
//! [`train`](LstmNetwork::train) back-propagates the current step's error
//! one step back in time (truncated back-propagation through time, as the C++
//! does), so it is called after each step's forward pass. The gradient stops at
//! h[t-1]: the network learns to use what its state already carries, but no
//! step is trained to store something only a later step needs, so a task like
//! "repeat the previous input" is learned poorly unless the current step's
//! target also depends on it.
//!
//! Each of the `HID` hidden cells has four gates, each with its own weights
//! from the inputs, the input bias and the previous step's hidden outputs:
//!
//! ```text
//! candidate = H(W_c x + b_c + U_c h[t-1])       (H: the cell activation, usually Tanh)
//! input     = sigmoid(W_i x + b_i + U_i h[t-1])
//! forget    = sigmoid(W_f x + b_f + U_f h[t-1])
//! output    = sigmoid(W_o x + b_o + U_o h[t-1])
//! c[t] = forget * c[t-1] + candidate * input
//! h[t] = tanh(c[t]) * output
//! ```
//!
//! The weight getters and setters index gates in that order
//! ([`GATE_CANDIDATE`], [`GATE_INPUT`], [`GATE_FORGET`], [`GATE_OUTPUT`]); the
//! C++ stores gate `g` of hidden cell `n` as connection `n * 4 + g`. The
//! hidden outputs feed an output layer with activation `O`.
//!
//! ```
//! use tinymind::recurrent::LstmNetwork;
//! use tinymind::{Q16_16, Sigmoid, Tanh};
//!
//! // 1 input -> 4 LSTM cells -> 1 sigmoid output. Each sequence is 8 values
//! // of +0.5 or -0.5; the output should be 1 once a positive value has
//! // appeared in the sequence, 0 before.
//! let mut seed = 3u32;
//! let mut nn = LstmNetwork::<Q16_16, Tanh, Sigmoid, 1, 4, 1>::new(|| {
//!     seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
//!     Q16_16::from_raw((seed >> 16) as i32 % 16_385 - 8_192)
//! });
//! let half = Q16_16::ONE / Q16_16::from_int(2);
//! let neg = Q16_16::ZERO - half;
//! let mut bits = 1u32;
//! for _ in 0..1000 {
//!     nn.reset_state();
//!     let mut seen = false;
//!     for _ in 0..8 {
//!         bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
//!         let x = if bits >> 30 == 3 { half } else { neg }; // positive 1 time in 4
//!         seen |= x > Q16_16::ZERO;
//!         nn.feed_forward(&[x]);
//!         nn.train(&[if seen { Q16_16::ONE } else { Q16_16::ZERO }]);
//!     }
//! }
//! nn.reset_state();
//! let outputs = [neg, neg, half, neg, neg].map(|x| {
//!     nn.feed_forward(&[x]);
//!     nn.learned_values()[0]
//! });
//! assert!(outputs[..2].iter().all(|&y| y < half));
//! assert!(outputs[2..].iter().all(|&y| y > half));
//! ```
//!
//! The value type is any [`Scalar`] for which [`Tanh`] and [`Sigmoid`] are
//! implemented: `Q16_16`, `Q8_8`, and `f32`/`f64` with the `std` or `libm`
//! feature. Training is SGD with momentum by default, or Adam or RMSprop
//! ([`crate::optimizer`]), with optional gradient clipping; every result is
//! bit-identical to the C++ network with the same initial weights.
//!
//! tinymind (C++)'s GRU and Elman networks are not ported yet.

use core::marker::PhantomData;

use crate::activation::{Activation, Sigmoid, Tanh};
use crate::neuralnet::{Connection, Neuron};
use crate::optimizer::{Optimizer, Sgd, update_all};
use crate::qformat::Scalar;
use crate::training::decay_weights;

/// The number of gates of an LSTM cell.
pub const LSTM_GATES: usize = 4;
/// Gate index of the cell candidate (activation `H`).
pub const GATE_CANDIDATE: usize = 0;
/// Gate index of the input gate (sigmoid).
pub const GATE_INPUT: usize = 1;
/// Gate index of the forget gate (sigmoid).
pub const GATE_FORGET: usize = 2;
/// Gate index of the output gate (sigmoid).
pub const GATE_OUTPUT: usize = 3;

/// A neuron feeding every gate of every LSTM cell: an input, the input bias, or
/// one of the previous step's hidden outputs (C++ `TrainableNeuron` with
/// `HID * 4` connections, connection `n * 4 + g` = `connections[n][g]`).
#[derive(Clone, Copy, Debug)]
struct GatedNeuron<V, const HID: usize> {
    output: V,
    connections: [[Connection<V>; LSTM_GATES]; HID],
}

impl<V: Scalar, const HID: usize> GatedNeuron<V, HID> {
    fn new() -> Self {
        Self {
            output: V::ZERO,
            connections: [[Connection::default(); LSTM_GATES]; HID],
        }
    }

    /// C++ `TrainableNeuron::initializeWeights`: one draw per connection in
    /// connection order (cell, then gate), then zero delta weights.
    fn initialize_weights(&mut self, random_weight: &mut impl FnMut() -> V) {
        for c in self.connections.iter_mut().flatten() {
            c.weight = random_weight();
        }
        for c in self.connections.iter_mut().flatten() {
            c.set_delta_weight(V::ZERO);
        }
    }

    /// Adds this neuron's contribution to every gate's sum, `output * weight`,
    /// in the C++ order (each target's sum runs over the source neurons in
    /// order, so iterating sources on the outside keeps it).
    fn accumulate(&self, sums: &mut [[V; LSTM_GATES]; HID]) {
        for (cell_sums, cell_connections) in sums.iter_mut().zip(&self.connections) {
            for (sum, c) in cell_sums.iter_mut().zip(cell_connections) {
                *sum += self.output * c.weight;
            }
        }
    }

    /// Sets each gate connection's gradient to `output * delta` for that gate's
    /// delta (C++ `calculateAndUpdateGradientsGated`).
    fn set_gradients<const OUT: usize>(&mut self, cells: &[LstmCell<V, OUT>; HID]) {
        let output = self.output;
        for (cell_connections, cell) in self.connections.iter_mut().zip(cells) {
            for (c, &delta) in cell_connections.iter_mut().zip(&cell.gate_deltas) {
                c.gradient = output * delta;
            }
        }
    }

    fn weights(&self) -> [[V; LSTM_GATES]; HID] {
        self.connections.map(|cell| cell.map(|c| c.weight))
    }

    fn set_weights(&mut self, weights: &[[V; LSTM_GATES]; HID]) {
        for (cell_connections, cell_weights) in self.connections.iter_mut().zip(weights) {
            for (c, &w) in cell_connections.iter_mut().zip(cell_weights) {
                c.weight = w;
            }
        }
    }
}

/// One LSTM hidden cell (C++ `TrainableLstmHiddenLayerNeuron`): its output
/// h, cell state c, the gate activations of the latest forward pass, its
/// deltas, and its connections to the output layer.
#[derive(Clone, Copy, Debug)]
struct LstmCell<V, const OUT: usize> {
    output: V,
    state: V,
    previous_state: V,
    candidate: V,
    input_gate: V,
    forget_gate: V,
    output_gate: V,
    node_delta: V,
    gate_deltas: [V; LSTM_GATES],
    connections: [Connection<V>; OUT],
}

impl<V: Scalar, const OUT: usize> LstmCell<V, OUT> {
    fn new() -> Self {
        Self {
            output: V::ZERO,
            state: V::ZERO,
            previous_state: V::ZERO,
            candidate: V::ZERO,
            input_gate: V::ZERO,
            forget_gate: V::ZERO,
            output_gate: V::ZERO,
            node_delta: V::ZERO,
            gate_deltas: [V::ZERO; LSTM_GATES],
            connections: [Connection::default(); OUT],
        }
    }
}

/// An LSTM network: `IN` inputs -> `HID` LSTM cells (cell candidate
/// activation `H`) -> `OUT` outputs (activation `O`).
///
/// C++: `LstmNeuralNetwork<ValueType, IN, HiddenLayers<HID>, OUT,
/// TransferFunctions>` with `H` and `O` the transfer functions' hidden and
/// output activations. See the [module documentation](self) for the cell
/// equations and an example.
///
/// `Opt` is the optimizer: [`Sgd`] (the default) or an adaptive one from
/// [`optimizer`](crate::optimizer), passed to
/// [`with_optimizer`](Self::with_optimizer).
pub struct LstmNetwork<V, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt = Sgd> {
    input: [GatedNeuron<V, HID>; IN],
    input_bias: GatedNeuron<V, HID>,
    hidden: [LstmCell<V, OUT>; HID],
    hidden_bias: Neuron<V, OUT>,
    /// Output neurons own one connection that is never used, as in the C++,
    /// whose initialisation consumes a random draw.
    output: [Neuron<V, 1>; OUT],
    /// The previous step's hidden outputs, h[t-1] (C++ recurrent layer), and
    /// their connections to every gate.
    recurrent: [GatedNeuron<V, HID>; HID],
    learning_rate: V,
    momentum_rate: V,
    acceleration_rate: V,
    gradient_clip: Option<(V, V)>,
    optimizer: Opt,
    activations: PhantomData<fn() -> (H, O)>,
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize>
    LstmNetwork<V, H, O, IN, HID, OUT, Sgd>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
    Sigmoid: Activation<V>,
    Tanh: Activation<V>,
{
    /// Build the network, trained with [`Sgd`], and initialise its weights from
    /// `random_weight`, in the C++ constructor's order: each input's
    /// connections, the input bias's, each hidden cell's connections to the
    /// outputs, the hidden bias's, the output neurons' (one unused each), and
    /// last each recurrent connection.
    pub fn new(random_weight: impl FnMut() -> V) -> Self {
        Self::with_optimizer(random_weight, Sgd)
    }
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>
    LstmNetwork<V, H, O, IN, HID, OUT, Opt>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
    Sigmoid: Activation<V>,
    Tanh: Activation<V>,
    Opt: Optimizer<V>,
{
    /// As [`new`](LstmNetwork::new), trained with `optimizer`, for example
    /// [`Adam`](crate::Adam).
    pub fn with_optimizer(mut random_weight: impl FnMut() -> V, optimizer: Opt) -> Self {
        let mut nn = Self {
            input: [GatedNeuron::new(); IN],
            input_bias: GatedNeuron::new(),
            hidden: [LstmCell::new(); HID],
            hidden_bias: Neuron::default(),
            output: [Neuron::default(); OUT],
            recurrent: [GatedNeuron::new(); HID],
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
        for cell in &mut nn.hidden {
            for c in &mut cell.connections {
                c.weight = random_weight();
            }
            for c in &mut cell.connections {
                c.set_delta_weight(V::ZERO);
            }
        }
        nn.hidden_bias.initialize_weights(&mut random_weight);
        for n in &mut nn.output {
            n.initialize_weights(&mut random_weight);
        }
        for n in &mut nn.recurrent {
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
    /// `GradientClipByValue<ValueType>` is `Some((-1.0, 1.0))`.
    pub fn set_gradient_clip(&mut self, range: Option<(V, V)>) {
        self.gradient_clip = range;
    }

    /// The learning rate the next training step will use: the one set with
    /// [`set_learning_rate`](Self::set_learning_rate), as a
    /// [`Scheduled`](crate::training::Scheduled) optimizer has decayed it so far.
    pub fn learning_rate(&self) -> V {
        self.learning_rate
    }

    /// Clear the recurrent state: every cell's output h and state c become 0,
    /// so the next [`feed_forward`](Self::feed_forward) starts a new sequence
    /// (C++ `resetState`). Weights and optimizer state are kept.
    pub fn reset_state(&mut self) {
        for cell in &mut self.hidden {
            cell.state = V::ZERO;
            cell.output = V::ZERO;
        }
        for r in &mut self.recurrent {
            r.output = V::ZERO;
        }
    }

    /// One time step: load `inputs`, latch the previous step's hidden outputs
    /// as h[t-1], update every cell and compute the outputs. Read them with
    /// [`learned_values`](Self::learned_values).
    pub fn feed_forward(&mut self, inputs: &[V; IN]) {
        for (neuron, &value) in self.input.iter_mut().zip(inputs) {
            neuron.output = value;
        }
        // C++ latchPreviousOutputs: every cell reads the same h[t-1].
        for (r, cell) in self.recurrent.iter_mut().zip(&self.hidden) {
            r.output = cell.output;
        }

        let mut input_sums = [[V::ZERO; LSTM_GATES]; HID];
        for neuron in &self.input {
            neuron.accumulate(&mut input_sums);
        }
        let mut recurrent_sums = [[V::ZERO; LSTM_GATES]; HID];
        for neuron in &self.recurrent {
            neuron.accumulate(&mut recurrent_sums);
        }

        let bias = self.input_bias.output;
        for (((cell, bias_connections), input_sum), recurrent_sum) in self
            .hidden
            .iter_mut()
            .zip(&self.input_bias.connections)
            .zip(&input_sums)
            .zip(&recurrent_sums)
        {
            // Each gate's sum: bias term, then the inputs, then h[t-1].
            let mut sums = [V::ZERO; LSTM_GATES];
            for (((sum, c), &x), &r) in sums
                .iter_mut()
                .zip(bias_connections)
                .zip(input_sum)
                .zip(recurrent_sum)
            {
                let mut s = bias * c.weight;
                s += x;
                s += r;
                *sum = s;
            }
            let [
                candidate_sum,
                input_gate_sum,
                forget_gate_sum,
                output_gate_sum,
            ] = sums;
            let candidate = H::activate(candidate_sum);
            let input_gate = Sigmoid::activate(input_gate_sum);
            let forget_gate = Sigmoid::activate(forget_gate_sum);
            let output_gate = Sigmoid::activate(output_gate_sum);

            let previous_state = cell.state;
            cell.candidate = candidate;
            cell.input_gate = input_gate;
            cell.forget_gate = forget_gate;
            cell.output_gate = output_gate;
            cell.previous_state = previous_state;

            let state = (forget_gate * previous_state) + (candidate * input_gate);
            cell.state = state;
            cell.output = Tanh::activate(state) * output_gate;
        }

        // Output layer: bias term + (0 + sum of h * w).
        let mut layer_sums = [V::ZERO; OUT];
        for cell in &self.hidden {
            for (sum, c) in layer_sums.iter_mut().zip(&cell.connections) {
                *sum += cell.output * c.weight;
            }
        }
        let bias = self.hidden_bias.output;
        for ((neuron, c), &layer_sum) in self
            .output
            .iter_mut()
            .zip(&self.hidden_bias.connections)
            .zip(&layer_sums)
        {
            let mut sum = bias * c.weight;
            sum += layer_sum;
            neuron.output = O::activate(sum);
        }
    }

    /// The output-layer activations from the latest
    /// [`feed_forward`](Self::feed_forward).
    pub fn learned_values(&self) -> [V; OUT] {
        self.output.map(|n| n.output)
    }

    /// The hidden cells' outputs h from the latest
    /// [`feed_forward`](Self::feed_forward).
    pub fn hidden_outputs(&self) -> [V; HID] {
        self.hidden.map(|cell| cell.output)
    }

    /// The hidden cells' states c from the latest
    /// [`feed_forward`](Self::feed_forward).
    pub fn cell_states(&self) -> [V; HID] {
        self.hidden.map(|cell| cell.state)
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

    /// One training step for the latest forward pass (C++
    /// `ChainBackPropagationThruTimePolicy`): the error is propagated to the
    /// gates of the current step and to the weights from the inputs, the
    /// biases and h[t-1], not further back in time.
    pub fn train(&mut self, targets: &[V; OUT]) {
        self.calculate_deltas(targets);
        self.calculate_gradients();
        self.update_weights();
        // C++ stepLearningRate, after the update.
        self.learning_rate = self.optimizer.next_learning_rate(self.learning_rate);
    }

    fn calculate_deltas(&mut self, targets: &[V; OUT]) {
        for (neuron, &target) in self.output.iter_mut().zip(targets) {
            let error = target - neuron.output;
            neuron.node_delta = error * O::derivative(neuron.output);
        }

        for cell in &mut self.hidden {
            // dL/dh: the raw weighted sum of the output deltas
            // (calculateAndSetRawNodeDeltas).
            let mut sum = V::ZERO;
            for (c, o) in cell.connections.iter().zip(&self.output) {
                sum += c.weight * o.node_delta;
            }
            cell.node_delta = sum;

            // computeGateDeltas.
            let dl_dh = sum;
            let tanh_c = Tanh::activate(cell.state);
            let dl_do = dl_dh * tanh_c;
            let dl_dc = dl_dh * cell.output_gate * Tanh::derivative(tanh_c);
            cell.gate_deltas = [
                dl_dc * cell.input_gate * H::derivative(cell.candidate),
                dl_dc * cell.candidate * Sigmoid::derivative(cell.input_gate),
                dl_dc * cell.previous_state * Sigmoid::derivative(cell.forget_gate),
                dl_do * Sigmoid::derivative(cell.output_gate),
            ];
        }
        // The C++ also computes node deltas for the recurrent layer and the
        // input layer. Nothing reads them, so they are skipped.
    }

    fn calculate_gradients(&mut self) {
        // Hidden -> output, then the hidden bias.
        for cell in &mut self.hidden {
            let out = cell.output;
            for (c, o) in cell.connections.iter_mut().zip(&self.output) {
                c.gradient = out * o.node_delta;
            }
        }
        let bias = self.hidden_bias.output;
        for (c, o) in self.hidden_bias.connections.iter_mut().zip(&self.output) {
            c.gradient = bias * o.node_delta;
        }

        // h[t-1] -> gates (an LSTM gate adds U h[t-1] unscaled, so its
        // recurrent weights take the gate delta itself), then inputs and the
        // input bias -> gates.
        for r in &mut self.recurrent {
            r.set_gradients(&self.hidden);
        }
        for n in &mut self.input {
            n.set_gradients(&self.hidden);
        }
        self.input_bias.set_gradients(&self.hidden);
    }

    fn update_weights(&mut self) {
        let rates = (
            self.learning_rate,
            self.momentum_rate,
            self.acceleration_rate,
        );
        let clip = self.gradient_clip;

        // Weight decay on every weight but the bias weights (input, recurrent
        // and hidden->output alike, as the C++ gated and plain updaters do),
        // before the steps: no update reads the weight.
        if let Some(lambda) = self.optimizer.weight_decay() {
            let factor = self.learning_rate * lambda;
            for cell in &mut self.hidden {
                decay_weights(&mut cell.connections, factor);
            }
            for r in &mut self.recurrent {
                decay_weights(r.connections.as_flattened_mut(), factor);
            }
            for n in &mut self.input {
                decay_weights(n.connections.as_flattened_mut(), factor);
            }
        }

        // One optimizer step (C++ `beginTrainingStep`), then layer by layer
        // as the C++ does (output, recurrent, input). Each connection's update
        // reads only itself, so the order does not matter. Bias connections
        // are clipped like the weights; the recurrent layer has no bias.
        self.optimizer.begin_step();
        let opt = &self.optimizer;
        for cell in &mut self.hidden {
            update_all(opt, &mut cell.connections, rates, clip);
        }
        update_all(opt, &mut self.hidden_bias.connections, rates, clip);
        for r in &mut self.recurrent {
            update_all(opt, r.connections.as_flattened_mut(), rates, clip);
        }
        for n in &mut self.input {
            update_all(opt, n.connections.as_flattened_mut(), rates, clip);
        }
        update_all(
            opt,
            self.input_bias.connections.as_flattened_mut(),
            rates,
            clip,
        );
    }
}

/// Weight getters and setters. Each takes or returns a whole matrix, so none
/// can panic. Gates are indexed as [`GATE_CANDIDATE`], [`GATE_INPUT`],
/// [`GATE_FORGET`] and [`GATE_OUTPUT`].
impl<V: Scalar, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>
    LstmNetwork<V, H, O, IN, HID, OUT, Opt>
{
    /// The optimizer, with its state.
    pub fn optimizer(&self) -> &Opt {
        &self.optimizer
    }

    /// `[input][cell][gate]`: the weights from each input to each gate.
    pub fn input_weights(&self) -> [[[V; LSTM_GATES]; HID]; IN] {
        self.input.map(|n| n.weights())
    }

    /// `[cell][gate]`: the input bias weights.
    pub fn input_bias_weights(&self) -> [[V; LSTM_GATES]; HID] {
        self.input_bias.weights()
    }

    /// `[from cell][to cell][gate]`: the weights from each cell's previous
    /// output h[t-1] to each gate.
    pub fn recurrent_weights(&self) -> [[[V; LSTM_GATES]; HID]; HID] {
        self.recurrent.map(|n| n.weights())
    }

    /// `[cell][output]`: the weights from the cells to the outputs.
    pub fn hidden_weights(&self) -> [[V; OUT]; HID] {
        self.hidden.map(|cell| cell.connections.map(|c| c.weight))
    }

    /// `[output]`: the hidden bias weights.
    pub fn hidden_bias_weights(&self) -> [V; OUT] {
        self.hidden_bias.connections.map(|c| c.weight)
    }

    /// Replace the input weights (see [`input_weights`](Self::input_weights)).
    pub fn set_input_weights(&mut self, weights: &[[[V; LSTM_GATES]; HID]; IN]) {
        for (n, w) in self.input.iter_mut().zip(weights) {
            n.set_weights(w);
        }
    }

    /// Replace the input bias weights.
    pub fn set_input_bias_weights(&mut self, weights: &[[V; LSTM_GATES]; HID]) {
        self.input_bias.set_weights(weights);
    }

    /// Replace the recurrent weights (see
    /// [`recurrent_weights`](Self::recurrent_weights)).
    pub fn set_recurrent_weights(&mut self, weights: &[[[V; LSTM_GATES]; HID]; HID]) {
        for (n, w) in self.recurrent.iter_mut().zip(weights) {
            n.set_weights(w);
        }
    }

    /// Replace the weights from the cells to the outputs.
    pub fn set_hidden_weights(&mut self, weights: &[[V; OUT]; HID]) {
        for (cell, w) in self.hidden.iter_mut().zip(weights) {
            for (c, &w) in cell.connections.iter_mut().zip(w) {
                c.weight = w;
            }
        }
    }

    /// Replace the hidden bias weights.
    pub fn set_hidden_bias_weights(&mut self, weights: &[V; OUT]) {
        for (c, &w) in self.hidden_bias.connections.iter_mut().zip(weights) {
            c.weight = w;
        }
    }
}
