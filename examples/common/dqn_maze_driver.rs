// The deep Q-network maze example's program: tinymind (C++)'s
// examples/dqn_maze (dqn_maze.cpp, dqn_mazelearner.h, dqn_mazelearner.cpp),
// shared by examples/dqn_maze.rs and tests/dqn_maze_golden.rs.
//
// The maze of examples/maze (six rooms; room 5 is outside, the goal; action
// `a` moves to room `a`, valid only through a door), learned by a deep
// Q-network instead of a table: a 6-8-6 Q16.16 perceptron (tanh hidden layer,
// linear output, since Q values reach 1 / (1 - 0.8) = 5 at the goal) maps the
// one-hot room to a Q value per action, with a target network refreshed every
// 10 training steps. 10000 training episodes start in a random room and run
// until they reach the goal or 100 steps: all but the last 100 explore every
// move, then exploration drops by one percentage point per episode. The
// network's Q values are then compared with Q* from value iteration
// (dqn_maze_qtable.csv, and a line on stdout), and 100 greedy test episodes
// follow that learn nothing (learning rate 0).
//
// The weights come from the C++'s `std::default_random_engine` (libstdc++'s
// `minstd_rand0`) through `std::uniform_real_distribution<double>(-1, 1)`;
// exploration and the start rooms from glibc's rand(), seeded with 7.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::fmt_g;

use tinymind::qlearn::{Environment, Experience, QLearner, QNetwork};
use tinymind::rand::GlibcRand;
use tinymind::{Linear, MultilayerPerceptron, Q16_16, Tanh};

pub const STATES: usize = 6;
pub const ACTIONS: usize = 6;
const HIDDEN: usize = ACTIONS + 2;
const GOAL: usize = 5;
const RANDOM_SEED: u32 = 7;
const TARGET_UPDATE_INTERVAL: usize = 10;
const TOTAL_NUMBER_OF_EPISODES: usize = 10000;
const NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES: usize = TOTAL_NUMBER_OF_EPISODES - 100;
const NUMBER_OF_TEST_EPISODES: usize = 100;
const MAX_NUMBER_OF_STEPS_PER_EPISODE: usize = 100;

/// libstdc++'s `std::minstd_rand0` (`std::default_random_engine`):
/// x = 16807 x mod (2^31 - 1).
struct MinstdRand0(u64);

impl MinstdRand0 {
    const MODULUS: u64 = 2_147_483_647;

    fn new(seed: u64) -> Self {
        let state = seed % Self::MODULUS;
        MinstdRand0(if state == 0 { 1 } else { state })
    }

    /// The next value, in `1..=2^31 - 2`.
    fn next(&mut self) -> u64 {
        self.0 = (self.0 * 16807) % Self::MODULUS;
        self.0
    }

    /// libstdc++'s `std::uniform_real_distribution<double>(a, b)(*this)`:
    /// `generate_canonical<double, 53>` takes two draws (it needs 53 bits and
    /// counts 30 per draw, the truncated log2 of the range 2^31 - 2), then
    /// `u * (b - a) + a`.
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        const RANGE: f64 = 2_147_483_646.0; // max - min + 1
        let mut sum = 0.0f64;
        let mut scale = 1.0f64;
        for _ in 0..2 {
            sum += (self.next() - 1) as f64 * scale;
            scale *= RANGE;
        }
        let mut u = sum / scale;
        if u >= 1.0 {
            // std::nextafter(1.0, 0.0)
            u = f64::from_bits(1.0f64.to_bits() - 1);
        }
        u * (b - a) + a
    }
}

/// dqn_mazelearner.h's DQNMazeEnvironment: a reward per (room, action), -1
/// for an action with no door, and the C++ program's single rand() stream.
pub struct Maze {
    pub rng: GlibcRand,
    rewards: [[Q16_16; ACTIONS]; STATES],
}

const INVALID: Q16_16 = Q16_16::from_int(-1);

impl Maze {
    fn new() -> Self {
        // dqn_maze.cpp's reward table: 1 for the doors out (action 5), 0 for
        // the other doors.
        let mut rewards = [[INVALID; ACTIONS]; STATES];
        for (state, row) in rewards.iter_mut().enumerate() {
            for (action, reward) in row.iter_mut().enumerate() {
                if is_action_valid_for_state(state, action) {
                    *reward = Q16_16::from_int(i32::from(action == 5));
                }
            }
        }
        Self {
            rng: GlibcRand::new(RANDOM_SEED),
            rewards,
        }
    }

    pub fn reward(&self, state: usize, action: usize) -> Q16_16 {
        self.rewards
            .get(state)
            .and_then(|r| r.get(action))
            .copied()
            .unwrap_or(Q16_16::ZERO)
    }
}

/// The doors: dqn_mazelearner.h's isActionValidForState.
fn is_action_valid_for_state(state: usize, action: usize) -> bool {
    match state {
        0 => action == 4,
        1 => action == 3 || action == 5,
        2 => action == 3,
        3 => action == 1 || action == 4 || action == 2,
        4 => action == 0 || action == 5 || action == 3,
        5 => action == 1 || action == 4 || action == 5,
        _ => false,
    }
}

impl Environment<ACTIONS> for Maze {
    fn valid_actions(&self, state: usize, actions: &mut [usize; ACTIONS]) -> usize {
        let mut count = 0;
        for action in 0..ACTIONS {
            if self.reward(state, action) != INVALID {
                actions[count] = action;
                count += 1;
            }
        }
        count
    }

    fn random_int(&mut self, min: usize, max: usize) -> usize {
        // (rand() % (max + 1 - min)) + min, in int as in the C++.
        (self.rng.next_i32() % (max + 1 - min) as i32) as usize + min
    }
}

/// dqn_mazelearner.h's getInputValues: the room, one-hot.
fn one_hot(state: usize) -> [Q16_16; STATES] {
    core::array::from_fn(|i| Q16_16::from_int(i32::from(i == state)))
}

type Network = MultilayerPerceptron<Q16_16, Tanh, Linear, STATES, HIDDEN, ACTIONS>;
pub type Learner = QLearner<
    Q16_16,
    Maze,
    STATES,
    ACTIONS,
    QNetwork<Q16_16, Tanh, Linear, STATES, HIDDEN, ACTIONS>,
>;

pub fn new_learner() -> Learner {
    // UniformRealRandomNumberGenerator: a uniform double in [-1, 1) from
    // std::default_random_engine(7), truncated to Q16.16 (ValueConverter).
    // The C++ policy also builds its target network from the same generator,
    // then overwrites it with the online weights; nothing else draws from that
    // generator afterwards, so building only the online network here gives
    // the same weights.
    let mut generator = MinstdRand0::new(u64::from(RANDOM_SEED));
    let network = Network::new(|| Q16_16::from_f64_trunc(generator.uniform(-1.0, 1.0)));
    let q_values = QNetwork::new(network, one_hot, TARGET_UPDATE_INTERVAL);
    // dqn_mazelearner.cpp: LEARNING_RATE 1/5, DISCOUNT_FACTOR 8/10, explore 100%.
    let learning_rate = Q16_16::from_int(1) / Q16_16::from_int(5);
    let discount_factor = Q16_16::from_int(8) / Q16_16::from_int(10);
    QLearner::with_q_values(Maze::new(), q_values, learning_rate, discount_factor, 100)
}

/// One episode from a random room until the goal or 100 more steps; returns
/// its CSV line.
fn episode(
    learner: &mut Learner,
    index: usize,
    out: &mut impl Write,
    err: &mut impl Write,
) -> io::Result<String> {
    learner.start_new_episode();
    let mut state = (learner.environment_mut().rng.next_i32() % STATES as i32) as usize;
    writeln!(out, "*** starting in state {state} ***")?;
    let mut line = format!("{state},");
    let mut step = |learner: &mut Learner, state: usize, line: &mut String| -> io::Result<()> {
        // The maze has a door out of every room, so there is always an action.
        let action = learner.take_action(state).expect("every room has a door");
        writeln!(out, "take action {action}")?;
        *line += &format!("{action},");
        let reward = learner.environment().reward(state, action);
        learner.update_from_experience(&Experience {
            state,
            action,
            reward,
            new_state: action,
        });
        Ok(())
    };
    step(learner, state, &mut line)?;
    let mut step_count = 0;
    while learner.state() != Some(GOAL) && step_count < MAX_NUMBER_OF_STEPS_PER_EPISODE {
        state = learner.state().unwrap_or(GOAL);
        step(learner, state, &mut line)?;
        step_count += 1;
    }
    if step_count >= MAX_NUMBER_OF_STEPS_PER_EPISODE {
        writeln!(
            err,
            "Episode {index} reached max step count without finding goal state."
        )?;
    }
    Ok(line)
}

fn to_double(value: Q16_16) -> f64 {
    f64::from(value.raw()) / 65536.0
}

/// dqn_maze.cpp's `reportDistanceFromOptimal`: Q* by 5000 rounds of value
/// iteration in `f64` (Q*(s, a) = r(s, a) + gamma * max_a' Q*(a, a'), since
/// action `a` leads to room `a`), the network's and the optimal values of
/// every valid (state, action) to `path`, and the largest and RMS error to
/// `out`.
// Index loops, as the C++ writes them: the order of the f64 operations is the
// C++'s, which bit-exact output needs.
#[allow(clippy::needless_range_loop)]
fn report_distance_from_optimal(
    learner: &Learner,
    path: &Path,
    out: &mut impl Write,
) -> io::Result<()> {
    let gamma = to_double(learner.discount_factor());
    let mut q_star = [[0.0f64; ACTIONS]; STATES];
    for _ in 0..5000 {
        let mut next = [[0.0f64; ACTIONS]; STATES];
        for s in 0..STATES {
            for a in 0..ACTIONS {
                if !is_action_valid_for_state(s, a) {
                    continue;
                }
                let mut best = 0.0;
                let mut any = false;
                for next_action in 0..ACTIONS {
                    if is_action_valid_for_state(a, next_action)
                        && (!any || q_star[a][next_action] > best)
                    {
                        best = q_star[a][next_action];
                        any = true;
                    }
                }
                next[s][a] = to_double(learner.environment().reward(s, a)) + (gamma * best);
            }
        }
        q_star = next;
    }

    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "state,action,q,q_star")?;
    let mut max_error = 0.0f64;
    let mut sum_squares = 0.0;
    let mut count = 0u32;
    for s in 0..STATES {
        for a in 0..ACTIONS {
            if !is_action_valid_for_state(s, a) {
                continue;
            }
            let q = to_double(learner.q_value(s, a));
            let error = q - q_star[s][a];
            writeln!(file, "{s},{a},{},{}", fmt_g(q), fmt_g(q_star[s][a]))?;
            // std::max(maxError, fabs(error)): keeps maxError unless it is
            // less than |error|.
            if max_error < error.abs() {
                max_error = error.abs();
            }
            sum_squares += error * error;
            count += 1;
        }
    }
    file.flush()?;
    writeln!(
        out,
        "Q values vs Q*: max |error| {}, RMS {}",
        fmt_g(max_error),
        fmt_g((sum_squares / f64::from(count)).sqrt())
    )
}

/// Runs the example, writing dqn_maze_training.csv, dqn_maze_qtable.csv and
/// dqn_maze_test.csv to `out_dir`, the log to `out` and the C++'s stderr
/// lines to `err`.
pub fn run(out_dir: &Path, out: &mut impl Write, err: &mut impl Write) -> io::Result<()> {
    let mut learner = new_learner();

    let mut training = BufWriter::new(File::create(out_dir.join("dqn_maze_training.csv"))?);
    for i in 0..TOTAL_NUMBER_OF_EPISODES {
        // After the pure random episodes, explore one percentage point less
        // each episode.
        if i >= NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES {
            let point = learner.random_action_decision_point();
            if point > 0 {
                learner.set_random_action_decision_point(point - 1);
            }
        }
        let line = episode(&mut learner, i, out, err)?;
        writeln!(training, "{line}")?;
    }
    training.flush()?;

    report_distance_from_optimal(&learner, &out_dir.join("dqn_maze_qtable.csv"), out)?;

    // The test runs are greedy and learn nothing.
    learner.set_random_action_decision_point(0);
    learner.set_learning_rate(Q16_16::ZERO);
    let mut test = BufWriter::new(File::create(out_dir.join("dqn_maze_test.csv"))?);
    for i in 0..NUMBER_OF_TEST_EPISODES {
        let line = episode(&mut learner, i, out, err)?;
        writeln!(test, "{line}")?;
    }
    test.flush()?;

    if learner.state() == Some(GOAL) {
        writeln!(out, "Reached the goal state!")?;
    } else {
        writeln!(err, "Did not reach the goal state!")?;
    }
    out.flush()?;
    err.flush()
}
