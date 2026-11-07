//! A trainable multilayer perceptron with one hidden layer, ported from the
//! `NeuralNetwork` / `MultilayerPerceptron` path of tinymind `cpp/neuralnet.hpp`.
//!
//! The C++ network is assembled from layer, neuron, connection, gradient-manager
//! and training-policy templates, stored in placement-new byte buffers. This port
//! keeps only what the default MLP configuration actually executes
//! (`FixedPointTransferFunctions` with default policies, batch size 1, no
//! recurrence, one hidden layer) and stores everything in plain fixed-size arrays:
//! no heap, so the crate stays `#![no_std]`.
//!
//! Every arithmetic step keeps the C++ operation order, because fixed-point
//! multiplication truncates and a reordered expression gives different bits.

use core::marker::PhantomData;

use crate::activation::Activation;
use crate::qformat::Fixed;

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
    fn set_delta_weight(&mut self, value: V) {
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

impl<V: Fixed, const N: usize> Default for Neuron<V, N> {
    fn default() -> Self {
        Self {
            output: V::ZERO,
            node_delta: V::ZERO,
            connections: [Connection::default(); N],
        }
    }
}

impl<V: Fixed, const N: usize> Neuron<V, N> {
    /// C++ `TrainableNeuron::initializeWeights`: one random weight per connection,
    /// in connection order, then zero the delta weights.
    fn initialize_weights(&mut self, random_weight: &mut impl FnMut() -> V) {
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
pub struct MultilayerPerceptron<V, H, O, const IN: usize, const HID: usize, const OUT: usize> {
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
    /// `H` and `O` are only used in static calls; `PhantomData` records that the
    /// type "uses" them. `fn() -> _` keeps the struct `Send`/`Sync` regardless.
    activations: PhantomData<fn() -> (H, O)>,
}

impl<V, H, O, const IN: usize, const HID: usize, const OUT: usize>
    MultilayerPerceptron<V, H, O, IN, HID, OUT>
where
    V: Fixed,
    H: Activation<V>,
    O: Activation<V>,
{
    /// Build the network and initialise its weights from `random_weight`, in the
    /// same order as the C++ constructor: input neurons, input bias, hidden
    /// neurons, hidden bias, output neurons.
    ///
    /// The C++ code reaches a static `generateRandomWeight()` policy that reads
    /// global `rand()` state; passing a closure makes that dependency explicit.
    pub fn new(mut random_weight: impl FnMut() -> V) -> Self {
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

    /// Replace the learning rate (default `2^-2`, tinymind's
    /// `DefaultNetworkInitializer`).
    pub fn set_learning_rate(&mut self, rate: V) {
        self.learning_rate = rate;
    }

    /// Replace the momentum rate (default `2^-1`).
    pub fn set_momentum_rate(&mut self, rate: V) {
        self.momentum_rate = rate;
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

    /// One back-propagation step with momentum (C++ `ChainBackPropagationPolicy`).
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

        // Output layer first, then the hidden layer, each neuron's incoming
        // connections followed by its bias connection.
        for o in 0..OUT {
            for h in 0..HID {
                update_connection(&mut self.hidden[h].connections[o], rates);
            }
            update_connection(&mut self.hidden_bias.connections[o], rates);
        }
        for h in 0..HID {
            for i in 0..IN {
                update_connection(&mut self.input[i].connections[h], rates);
            }
            update_connection(&mut self.input_bias.connections[h], rates);
        }
    }

    // Weight accessors, in the order tinymind's NetworkPropertiesFileManager writes them.

    /// Weight from input `input` to hidden neuron `hidden`.
    ///
    /// # Panics
    ///
    /// If `input >= IN` or `hidden >= HID`.
    pub fn input_weight(&self, input: usize, hidden: usize) -> V {
        self.input[input].connections[hidden].weight
    }

    /// Weight from the input-layer bias to hidden neuron `hidden`.
    ///
    /// # Panics
    ///
    /// If `hidden >= HID`.
    pub fn input_bias_weight(&self, hidden: usize) -> V {
        self.input_bias.connections[hidden].weight
    }

    /// Weight from hidden neuron `hidden` to output `output`.
    ///
    /// # Panics
    ///
    /// If `hidden >= HID` or `output >= OUT`.
    pub fn hidden_weight(&self, hidden: usize, output: usize) -> V {
        self.hidden[hidden].connections[output].weight
    }

    /// Weight from the hidden-layer bias to output `output`.
    ///
    /// # Panics
    ///
    /// If `output >= OUT`.
    pub fn hidden_bias_weight(&self, output: usize) -> V {
        self.hidden_bias.connections[output].weight
    }
}

/// `bias.output * bias.w[target] + sum(layer[n].output * layer[n].w[target])`,
/// summed in the C++ order: the layer sum starts from zero, then is added to the
/// bias term.
fn weighted_sum<V: Fixed, const N: usize, const C: usize>(
    bias: &Neuron<V, C>,
    layer: &[Neuron<V, C>; N],
    target: usize,
) -> V {
    let mut sum = bias.output * bias.connections[target].weight;
    let mut layer_sum = V::ZERO;
    for neuron in layer {
        layer_sum += neuron.output * neuron.connections[target].weight;
    }
    sum += layer_sum;
    sum
}

/// SGD with momentum and "acceleration" (C++ `updateConnectionWeightsImpl`, SGD path):
/// `dw = lr * g + momentum * dw[t-1] + acceleration * dw[t-2]`, `w += dw`.
fn update_connection<V: Fixed>(c: &mut Connection<V>, (lr, momentum, acceleration): (V, V, V)) {
    let new_delta_weight =
        (lr * c.gradient) + (momentum * c.delta_weight) + (acceleration * c.previous_delta_weight);
    let current_weight = c.weight;
    c.set_delta_weight(new_delta_weight);
    c.weight = current_weight + new_delta_weight;
}
