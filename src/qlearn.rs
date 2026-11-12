//! Tabular Q-learning, ported from tinymind (C++)'s `qlearn.hpp`: [`QLearner`]
//! (`QLearner` with `DefaultLearningPolicy` and `ArgMaxPolicy`), [`QTable`]
//! (`QValueTablePolicy`) and the [`Environment`] trait (what the C++ example
//! environments derive from `QLearningEnvironment` and supply as policies).
//!
//! The learner keeps a table of Q values, one per (state, action) pair, in
//! the value type of your choice (`Q16_16`, `Q8_8`, `f32`, `f64`): a 6-state,
//! 6-action Q16.16 table is 144 bytes. You describe the world through
//! [`Environment`]: which actions are valid in a state, and a source of random
//! integers for exploration. The loop that moves through the world is yours:
//! ask the learner for an action, apply it, and report the reward and the new
//! state.
//!
//! The update is tinymind (C++)'s, operation for operation:
//!
//! ```text
//! Q(s, a) = (1 - lr) * Q(s, a) + lr * (reward + discount * max_a' Q(s', a'))
//! ```
//!
//! for the experience's own state `s`, where the maximum runs over the actions
//! valid in the next state `s'` (0 if there are none): the greedy value,
//! whatever the learner explores, so the table converges to the optimal Q
//! values. Action selection explores with probability
//! `random_action_decision_point` percent; otherwise it takes the action with
//! the largest Q value, as tinymind (C++)'s `ArgMaxPolicy` does (the first of
//! equal maxima), unless every valid action has the same Q value (an unvisited
//! state), in which case a random valid action is taken.
//!
//! ```
//! use tinymind::qlearn::{Environment, Experience, QLearner};
//! use tinymind::rand::GlibcRand;
//! use tinymind::Q16_16;
//!
//! // A corridor of 4 cells; action 0 moves left, 1 moves right; cell 3 is the goal.
//! struct Corridor(GlibcRand);
//!
//! impl Environment<2> for Corridor {
//!     fn valid_actions(&self, _state: usize, actions: &mut [usize; 2]) -> usize {
//!         *actions = [0, 1];
//!         2
//!     }
//!     fn random_int(&mut self, min: usize, max: usize) -> usize {
//!         self.0.next_i32() as usize % (max + 1 - min) + min
//!     }
//! }
//!
//! let rate = Q16_16::from_int(1) / Q16_16::from_int(5);
//! let discount = Q16_16::from_int(8) / Q16_16::from_int(10);
//! // Explore 100% of the time while learning.
//! let mut learner =
//!     QLearner::<Q16_16, Corridor, 4, 2>::new(Corridor(GlibcRand::new(7)), rate, discount, 100);
//! for _ in 0..200 {
//!     learner.start_new_episode();
//!     let mut state = 0;
//!     while state != 3 {
//!         let action = learner.take_action(state).unwrap();
//!         let next = if action == 0 { state.saturating_sub(1) } else { state + 1 };
//!         let reward = if next == 3 { Q16_16::ONE } else { Q16_16::ZERO };
//!         learner.update_from_experience(&Experience { state, action, reward, new_state: next });
//!         state = next;
//!     }
//! }
//! // Now exploit: from every cell the learner heads right.
//! learner.set_random_action_decision_point(0);
//! for cell in 0..3 {
//!     assert_eq!(learner.take_action(cell), Some(1));
//! }
//! ```
//!
//! # Deep Q-network
//!
//! [`QNetwork`] (tinymind (C++)'s `QValueNeuralNetworkPolicy`) keeps the Q
//! values in a [`MultilayerPerceptron`] instead of a table: the network maps a
//! state, encoded as its input vector by a function you supply, to one Q
//! value per action, so states that look alike share what they learn. Each
//! update trains the network on the taken action's target. The update's
//! lookahead reads a second, target network, a copy of the online one
//! refreshed every so many training steps, which keeps the target from
//! chasing its own updates. It is a drop-in replacement for the table: pass
//! it to [`QLearner::with_q_values`].

use crate::activation::Activation;
use crate::neuralnet::{InferencePerceptron, MultilayerPerceptron};
use crate::optimizer::{Optimizer, Sgd};
use crate::qformat::Scalar;

/// The world a [`QLearner`] acts in: tinymind (C++)'s environment (the
/// `getValidActionsForState` an example environment defines, and its random
/// number policy's `randInt`).
pub trait Environment<const ACTIONS: usize> {
    /// Writes the actions valid in `state` to the front of `actions` and
    /// returns how many there are (0 if none; values above `ACTIONS` count as
    /// `ACTIONS`).
    fn valid_actions(&self, state: usize, actions: &mut [usize; ACTIONS]) -> usize;

    /// A uniformly distributed integer in `min..=max`, for exploration. The
    /// learner asks for `random_int(0, 99)` to decide whether to explore and
    /// `random_int(0, n - 1)` to pick one of `n` valid actions.
    fn random_int(&mut self, min: usize, max: usize) -> usize;
}

/// Where a [`QLearner`] keeps its Q values: tinymind (C++)'s Q value policy.
/// [`QTable`] is the tabular one.
pub trait QValues<V> {
    /// The Q value of taking `action` in `state`.
    fn q_value(&self, state: usize, action: usize) -> V;
    /// The Q value of the best action in the next state, used for the
    /// update's target. A table returns [`q_value`](QValues::q_value).
    fn future_q_value(&mut self, state: usize, action: usize) -> V;
    /// Stores a new Q value for `action` in `state`.
    fn set_q_value(&mut self, state: usize, action: usize, value: V);
}

/// A Q table, `values[state][action]` (tinymind (C++)'s `QValueTablePolicy`).
/// Reading a state or action out of range gives 0 and writing one does nothing,
/// as in the C++.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QTable<V, const STATES: usize, const ACTIONS: usize> {
    /// The Q values, by state then action.
    pub values: [[V; ACTIONS]; STATES],
}

impl<V: Scalar, const STATES: usize, const ACTIONS: usize> QTable<V, STATES, ACTIONS> {
    /// A table of zeros, the C++ initial Q value.
    pub fn new() -> Self {
        Self::from_values([[V::ZERO; ACTIONS]; STATES])
    }
}

impl<V, const STATES: usize, const ACTIONS: usize> QTable<V, STATES, ACTIONS> {
    /// A table holding `values`, for example ones learned earlier.
    pub const fn from_values(values: [[V; ACTIONS]; STATES]) -> Self {
        Self { values }
    }
}

impl<V: Scalar, const STATES: usize, const ACTIONS: usize> Default for QTable<V, STATES, ACTIONS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Scalar, const STATES: usize, const ACTIONS: usize> QValues<V>
    for QTable<V, STATES, ACTIONS>
{
    fn q_value(&self, state: usize, action: usize) -> V {
        self.values
            .get(state)
            .and_then(|row| row.get(action))
            .copied()
            .unwrap_or(V::ZERO)
    }

    fn future_q_value(&mut self, state: usize, action: usize) -> V {
        self.q_value(state, action)
    }

    fn set_q_value(&mut self, state: usize, action: usize, value: V) {
        if let Some(q) = self
            .values
            .get_mut(state)
            .and_then(|row| row.get_mut(action))
        {
            *q = value;
        }
    }
}

/// Q values from a neural network: a deep Q-network (tinymind (C++)'s
/// `QValueNeuralNetworkPolicy`), for [`QLearner::with_q_values`].
///
/// The online network is a [`MultilayerPerceptron`] with `IN` inputs, `HID`
/// hidden neurons and one output per action. `inputs` encodes a state as the
/// network's input vector (C++ `getInputValues`; a one-hot vector for a small
/// set of states). Reading a Q value runs the online network forward without
/// changing it. Storing one feeds the state forward, sets the taken action's
/// output to the new value, keeps the others as they are, and trains once,
/// unless the value is already there. Every `target_update_interval` training
/// steps the online weights are copied into the target network, which the
/// learner's lookahead reads ([`QValues::future_q_value`]); it starts as a
/// copy of the online network.
///
/// ```
/// use tinymind::qlearn::{QNetwork, QValues};
/// use tinymind::{Linear, MultilayerPerceptron, Q16_16, Tanh};
///
/// // Three states, one-hot encoded; two actions; four hidden neurons.
/// fn one_hot(state: usize) -> [Q16_16; 3] {
///     core::array::from_fn(|i| Q16_16::from_int(i32::from(i == state)))
/// }
/// let mut seed = 1;
/// let net = MultilayerPerceptron::<Q16_16, Tanh, Linear, 3, 4, 2>::new(|| {
///     seed = (seed * 7 + 3) % 17;
///     Q16_16::from_raw((seed - 8) * 4096)
/// });
/// let mut q = QNetwork::new(net, one_hot, 10);
/// let target = Q16_16::from_int(1) / Q16_16::from_int(2);
/// let before = q.q_value(1, 0);
/// q.set_q_value(1, 0, target);
/// // One training step moves Q(1, 0) towards the target.
/// let after = q.q_value(1, 0);
/// assert!((after - target).raw().abs() < (before - target).raw().abs());
/// ```
pub struct QNetwork<V, H, O, const IN: usize, const HID: usize, const ACTIONS: usize, Opt = Sgd> {
    network: MultilayerPerceptron<V, H, O, IN, HID, ACTIONS, Opt>,
    target: InferencePerceptron<V, H, O, IN, HID, ACTIONS>,
    inputs: fn(usize) -> [V; IN],
    target_update_interval: usize,
    iterations: usize,
}

impl<V, H, O, const IN: usize, const HID: usize, const ACTIONS: usize, Opt>
    QNetwork<V, H, O, IN, HID, ACTIONS, Opt>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
    Opt: Optimizer<V>,
{
    /// Q values from `network`, with `inputs` encoding each state as its
    /// input vector, and the target network refreshed every
    /// `target_update_interval` training steps (0 refreshes it after every
    /// step, as 1 does, like the C++ `>=` test).
    pub fn new(
        network: MultilayerPerceptron<V, H, O, IN, HID, ACTIONS, Opt>,
        inputs: fn(usize) -> [V; IN],
        target_update_interval: usize,
    ) -> Self {
        let target = network.to_inference();
        Self {
            network,
            target,
            inputs,
            target_update_interval,
            iterations: 0,
        }
    }

    /// The online network.
    pub fn network(&self) -> &MultilayerPerceptron<V, H, O, IN, HID, ACTIONS, Opt> {
        &self.network
    }

    /// The online network, to change its rates or weights.
    pub fn network_mut(&mut self) -> &mut MultilayerPerceptron<V, H, O, IN, HID, ACTIONS, Opt> {
        &mut self.network
    }

    /// The target network: the online weights as of the last refresh.
    pub fn target(&self) -> &InferencePerceptron<V, H, O, IN, HID, ACTIONS> {
        &self.target
    }
}

impl<V, H, O, const IN: usize, const HID: usize, const ACTIONS: usize, Opt> QValues<V>
    for QNetwork<V, H, O, IN, HID, ACTIONS, Opt>
where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
    Opt: Optimizer<V>,
{
    /// The online network's output for `action` (0 for an action out of
    /// range).
    fn q_value(&self, state: usize, action: usize) -> V {
        let outputs = self.network.evaluate(&(self.inputs)(state));
        outputs.get(action).copied().unwrap_or(V::ZERO)
    }

    /// The target network's output for `action` (0 for an action out of
    /// range).
    fn future_q_value(&mut self, state: usize, action: usize) -> V {
        let outputs = self.target.feed_forward(&(self.inputs)(state));
        outputs.get(action).copied().unwrap_or(V::ZERO)
    }

    /// Trains the online network once towards `value` for `action`, unless
    /// its output already equals it (an action out of range is ignored).
    fn set_q_value(&mut self, state: usize, action: usize, value: V) {
        self.network.feed_forward(&(self.inputs)(state));
        let mut targets = self.network.learned_values();
        let Some(target) = targets.get_mut(action) else {
            return;
        };
        if *target != value {
            *target = value;
            self.network.train(&targets);
            self.iterations += 1;
            if self.iterations >= self.target_update_interval {
                self.iterations = 0;
                self.target = self.network.to_inference();
            }
        }
    }
}

/// One step's outcome, reported to [`QLearner::update_from_experience`]
/// (tinymind (C++)'s `experience_t`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experience<V> {
    /// The state the step started in; the update changes its Q value. A state
    /// of `STATES` or more changes nothing (tinymind (C++)'s invalid state).
    pub state: usize,
    /// The action taken.
    pub action: usize,
    /// The reward it earned.
    pub reward: V,
    /// The state it led to.
    pub new_state: usize,
}

/// A Q-learning agent over `STATES` states and `ACTIONS` actions
/// (tinymind (C++)'s `QLearner`). It owns its [`Environment`] and its Q values (a
/// [`QTable`] unless built with [`with_q_values`](Self::with_q_values)).
#[derive(Clone, Debug)]
pub struct QLearner<V, E, const STATES: usize, const ACTIONS: usize, Q = QTable<V, STATES, ACTIONS>>
{
    environment: E,
    q_values: Q,
    state: Option<usize>,
    learning_rate: V,
    discount_factor: V,
    random_action_decision_point: usize,
}

impl<V: Scalar, E: Environment<ACTIONS>, const STATES: usize, const ACTIONS: usize>
    QLearner<V, E, STATES, ACTIONS>
{
    /// A learner with a zeroed [`QTable`]. `learning_rate` and
    /// `discount_factor` must be in [0, 1] and `random_action_decision_point`
    /// (the exploration percentage) at most 100; a value outside that range
    /// is ignored, leaving 0, as in the C++.
    pub fn new(
        environment: E,
        learning_rate: V,
        discount_factor: V,
        random_action_decision_point: usize,
    ) -> Self {
        Self::with_q_values(
            environment,
            QTable::new(),
            learning_rate,
            discount_factor,
            random_action_decision_point,
        )
    }
}

impl<V: Scalar, E: Environment<ACTIONS>, const STATES: usize, const ACTIONS: usize, Q: QValues<V>>
    QLearner<V, E, STATES, ACTIONS, Q>
{
    /// A learner over the given Q values; otherwise as [`new`](QLearner::new).
    pub fn with_q_values(
        environment: E,
        q_values: Q,
        learning_rate: V,
        discount_factor: V,
        random_action_decision_point: usize,
    ) -> Self {
        // The C++ static_asserts.
        const {
            assert!(STATES > 1, "a QLearner needs at least two states");
            assert!(ACTIONS > 1, "a QLearner needs at least two actions");
        }
        let mut learner = Self {
            environment,
            q_values,
            state: None,
            learning_rate: V::ZERO,
            discount_factor: V::ZERO,
            random_action_decision_point: 0,
        };
        learner.set_learning_rate(learning_rate);
        learner.set_discount_factor(discount_factor);
        learner.set_random_action_decision_point(random_action_decision_point);
        learner
    }

    /// The environment.
    pub fn environment(&self) -> &E {
        &self.environment
    }

    /// The environment, mutably (for example to draw from its generator).
    pub fn environment_mut(&mut self) -> &mut E {
        &mut self.environment
    }

    /// The Q values.
    pub fn q_values(&self) -> &Q {
        &self.q_values
    }

    /// The Q values, mutably.
    pub fn q_values_mut(&mut self) -> &mut Q {
        &mut self.q_values
    }

    /// The Q value of taking `action` in `state`.
    pub fn q_value(&self, state: usize, action: usize) -> V {
        self.q_values.q_value(state, action)
    }

    /// Sets the Q value of taking `action` in `state`.
    pub fn set_q_value(&mut self, state: usize, action: usize, value: V) {
        self.q_values.set_q_value(state, action, value);
    }

    /// The learning rate.
    pub fn learning_rate(&self) -> V {
        self.learning_rate
    }

    /// Sets the learning rate if it is in [0, 1]; otherwise keeps the old one.
    pub fn set_learning_rate(&mut self, rate: V) {
        if is_in_unit_interval(rate) {
            self.learning_rate = rate;
        }
    }

    /// The discount factor.
    pub fn discount_factor(&self) -> V {
        self.discount_factor
    }

    /// Sets the discount factor if it is in [0, 1]; otherwise keeps the old
    /// one.
    pub fn set_discount_factor(&mut self, factor: V) {
        if is_in_unit_interval(factor) {
            self.discount_factor = factor;
        }
    }

    /// How often the learner explores, in percent: 0 never, 100 always.
    pub fn random_action_decision_point(&self) -> usize {
        self.random_action_decision_point
    }

    /// Sets the exploration percentage if it is at most 100; otherwise keeps
    /// the old one.
    pub fn set_random_action_decision_point(&mut self, percent: usize) {
        if percent <= 100 {
            self.random_action_decision_point = percent;
        }
    }

    /// The current state, `None` at the start of an episode.
    pub fn state(&self) -> Option<usize> {
        self.state
    }

    /// Sets the current state if it is below `STATES`.
    pub fn set_state(&mut self, state: usize) {
        if state < STATES {
            self.state = Some(state);
        }
    }

    /// Forgets the current state (the C++ invalid state), as at the start of
    /// an episode.
    pub fn start_new_episode(&mut self) {
        self.state = None;
    }

    /// Chooses an action in `state` (exploring or exploiting), makes `state`
    /// the current state and returns the action, or `None` if `state` has no
    /// valid action.
    pub fn take_action(&mut self, state: usize) -> Option<usize> {
        let action = self.choose_action(state);
        self.set_state(state);
        action
    }

    /// Learns from a step: updates the Q value of `experience.action` in
    /// `experience.state` (unless the learning rate is 0 or the state is out of
    /// range), whatever the learner's current state, and moves to
    /// `experience.new_state`.
    pub fn update_from_experience(&mut self, experience: &Experience<V>) {
        let learning_rate = self.learning_rate;
        if learning_rate > V::ZERO && experience.state < STATES {
            let one_minus_learning_rate = V::ONE - learning_rate;
            let discount_factor = self.discount_factor;
            let old_q_value = self.q_values.q_value(experience.state, experience.action);
            let future_q_value = self.future_q_value(experience.new_state);
            let q_value = (one_minus_learning_rate * old_q_value)
                + (learning_rate * (experience.reward + (discount_factor * future_q_value)));
            self.q_values
                .set_q_value(experience.state, experience.action, q_value);
        }
        self.state = Some(experience.new_state);
    }

    /// max over the actions valid in `state` of their Q value, from the action
    /// [`best_action`](Self::best_action) picks (the first valid one when all
    /// tie, since each is then the maximum); 0 when none is valid. Never draws
    /// from the generator.
    fn future_q_value(&mut self, state: usize) -> V {
        let mut buffer = [0; ACTIONS];
        let count = self.environment.valid_actions(state, &mut buffer);
        let valid = buffer.get(..count).unwrap_or(&buffer);
        let Some(&first) = valid.first() else {
            return V::ZERO;
        };
        let action = self.best_action(state, valid).unwrap_or(first);
        self.q_values.future_q_value(state, action)
    }

    fn choose_action(&mut self, state: usize) -> Option<usize> {
        let mut buffer = [0; ACTIONS];
        let count = self.environment.valid_actions(state, &mut buffer);
        let valid = buffer.get(..count).unwrap_or(&buffer);
        if self.should_choose_random_action() {
            self.random_action(valid)
        } else {
            match self.best_action(state, valid) {
                Some(action) => Some(action),
                None => self.random_action(valid),
            }
        }
    }

    fn should_choose_random_action(&mut self) -> bool {
        match self.random_action_decision_point {
            0 => false,
            100 => true,
            // 100 equally likely values, `point` of them below it: a random
            // action with probability exactly point / 100.
            point => self.environment.random_int(0, 99) < point,
        }
    }

    fn random_action(&mut self, valid: &[usize]) -> Option<usize> {
        match valid {
            [] => None,
            [action] => Some(*action),
            _ => {
                let index = self.environment.random_int(0, valid.len() - 1);
                valid.get(index).copied()
            }
        }
    }

    /// tinymind (C++)'s `ArgMaxPolicy::selectBestActionForState`.
    fn best_action(&self, state: usize, valid: &[usize]) -> Option<usize> {
        match valid {
            [] => None,
            [action] => Some(*action),
            [a0, a1] => {
                let (q0, q1) = (
                    self.q_values.q_value(state, *a0),
                    self.q_values.q_value(state, *a1),
                );
                if q0 > q1 {
                    Some(*a0)
                } else if q1 > q0 {
                    Some(*a1)
                } else {
                    // A tie is no preference, as below.
                    None
                }
            }
            [first, rest @ ..] => {
                // The largest Q value, whatever its sign (the first of equal
                // maxima). Only when every valid action ties, as in an
                // unvisited state of a zeroed table, is there no preference:
                // `None` makes the caller take a random action (and the
                // update's bootstrap any of them, since each is the maximum).
                let mut best_action = *first;
                let mut best_q_value = self.q_values.q_value(state, best_action);
                let mut all_tied = true;
                for &action in rest {
                    let q_value = self.q_values.q_value(state, action);
                    if q_value != best_q_value {
                        all_tied = false;
                    }
                    if q_value > best_q_value {
                        best_q_value = q_value;
                        best_action = action;
                    }
                }
                if all_tied { None } else { Some(best_action) }
            }
        }
    }
}

/// tinymind (C++)'s `isInUnitInterval`: `!(v < 0) && !(1 < v)`, so a float NaN
/// passes, as in the C++.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn is_in_unit_interval<V: Scalar>(value: V) -> bool {
    !(value < V::ZERO) && !(V::ONE < value)
}
