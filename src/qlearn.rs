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
//! Q(s, a) = (1 - lr) * Q(s, a) + lr * (reward + discount * Q(s', a'))
//! ```
//!
//! where `a'` is the action the learner would choose in `s'`, exploration
//! included (so a random `a'` draws from the environment's generator, as in the
//! C++). Action selection explores with probability
//! `random_action_decision_point` percent; otherwise it takes the action with
//! the largest Q value, as tinymind (C++)'s `ArgMaxPolicy` does: with two valid
//! actions the first wins only if strictly larger; with three or more the
//! search starts from 0, so an action is chosen only if its Q value is
//! positive, and if none is, a random valid action is taken instead.
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
//!         learner.update_from_experience(&Experience { action, reward, new_state: next });
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
//! tinymind (C++)'s deep Q-network policy (`QValueNeuralNetworkPolicy`) is not
//! ported yet.

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
    /// learner asks for `random_int(0, 100)` to decide whether to explore and
    /// `random_int(0, n - 1)` to pick one of `n` valid actions.
    fn random_int(&mut self, min: usize, max: usize) -> usize;
}

/// Where a [`QLearner`] keeps its Q values: tinymind (C++)'s Q value policy.
/// [`QTable`] is the tabular one.
pub trait QValues<V> {
    /// The Q value of taking `action` in `state`.
    fn q_value(&self, state: usize, action: usize) -> V;
    /// The Q value of the action chosen in the next state, used for the
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

/// One step's outcome, reported to [`QLearner::update_from_experience`]
/// (tinymind (C++)'s `experience_t` without its `state`, which the C++ learner
/// does not read either: the step started in the learner's current state).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experience<V> {
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
    /// `discount_factor` must be below 1 and `random_action_decision_point`
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

    /// Sets the learning rate if it is below 1; otherwise keeps the old one.
    pub fn set_learning_rate(&mut self, rate: V) {
        if rate < V::ONE {
            self.learning_rate = rate;
        }
    }

    /// The discount factor.
    pub fn discount_factor(&self) -> V {
        self.discount_factor
    }

    /// Sets the discount factor if it is below 1; otherwise keeps the old one.
    pub fn set_discount_factor(&mut self, factor: V) {
        if factor < V::ONE {
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

    /// Forgets the current state: the next update only moves the learner to
    /// its new state and changes no Q value.
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

    /// Learns from the step taken in the current state: updates its Q value
    /// (unless the learning rate is 0 or there is no current state) and moves
    /// to `experience.new_state`.
    pub fn update_from_experience(&mut self, experience: &Experience<V>) {
        let learning_rate = self.learning_rate;
        if learning_rate > V::ZERO {
            let one_minus_learning_rate = V::ONE - learning_rate;
            let discount_factor = self.discount_factor;
            let old_q_value = match self.state {
                Some(state) => self.q_values.q_value(state, experience.action),
                None => V::ZERO,
            };
            let future_q_value = self.future_q_value(experience.new_state);
            let q_value = (one_minus_learning_rate * old_q_value)
                + (learning_rate * (experience.reward + (discount_factor * future_q_value)));
            if let Some(state) = self.state {
                self.q_values.set_q_value(state, experience.action, q_value);
            }
        }
        self.state = Some(experience.new_state);
    }

    fn future_q_value(&mut self, state: usize) -> V {
        match self.choose_action(state) {
            Some(action) => self.q_values.future_q_value(state, action),
            None => V::ZERO,
        }
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
            point => self.environment.random_int(0, 100) <= point,
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
                if self.q_values.q_value(state, *a0) > self.q_values.q_value(state, *a1) {
                    Some(*a0)
                } else {
                    Some(*a1)
                }
            }
            _ => {
                let mut best_q_value = V::ZERO;
                let mut best_action = None;
                for &action in valid {
                    let q_value = self.q_values.q_value(state, action);
                    if q_value > best_q_value {
                        best_q_value = q_value;
                        best_action = Some(action);
                    }
                }
                best_action
            }
        }
    }
}
