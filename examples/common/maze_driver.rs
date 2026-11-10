// The maze example's program: tinymind (C++)'s examples/maze (maze.cpp,
// mazelearner.h, mazelearner.cpp), shared by examples/maze.rs and
// tests/maze_golden.rs.
//
// Six rooms; room 5 is outside, the goal. Action `a` moves to room `a`, valid
// only through a door. 5000 training episodes start in a random room and run
// until they reach the goal: the first 4900 always explore, then exploration
// drops by one percentage point per episode. The learned table is then
// compared with Q* from value iteration (maze_qtable.csv, and a line on
// stdout), and 100 greedy test episodes follow that learn nothing (learning
// rate 0). Each episode's path goes to a CSV line and to stdout.
//
// The C++ stores Q values in unsigned Q16.16 (`QValue<16, 16, false>`); this
// port uses the crate's signed Q16.16. Every Q value, reward, rate and
// discount here is non-negative, where the two formats have the same bits and
// the same arithmetic; the C++ marks invalid actions with a reward of -1, which
// is only compared for equality. tests/qlearn_sweep.rs also checks the Q table
// after every update against the unsigned C++.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::fmt_g;

use tinymind::Q16_16;
use tinymind::qlearn::{Environment, Experience, QLearner};
use tinymind::rand::GlibcRand;

pub const STATES: usize = 6;
pub const ACTIONS: usize = 6;
const GOAL: usize = 5;
const RANDOM_SEED: u32 = 7;
const TOTAL_NUMBER_OF_EPISODES: usize = 5000;
const NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES: usize = TOTAL_NUMBER_OF_EPISODES - 100;
const NUMBER_OF_TEST_EPISODES: usize = 100;

/// mazelearner.h's MazeEnvironment: a reward per (room, action), -1 for an
/// action with no door, and the C++ program's single rand() stream.
pub struct Maze {
    pub rng: GlibcRand,
    rewards: [[Q16_16; ACTIONS]; STATES],
}

const INVALID: Q16_16 = Q16_16::from_int(-1);

impl Maze {
    fn new() -> Self {
        let mut rewards = [[INVALID; ACTIONS]; STATES];
        for (state, row) in rewards.iter_mut().enumerate() {
            for (action, reward) in row.iter_mut().enumerate() {
                if is_action_valid_for_state(state, action) {
                    *reward = if action == 5 {
                        Q16_16::from_int(1)
                    } else {
                        Q16_16::from_int(0)
                    };
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

/// The doors: mazelearner.h's isActionValidForState.
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

pub type Learner = QLearner<Q16_16, Maze, STATES, ACTIONS>;

pub fn new_learner() -> Learner {
    // mazelearner.cpp: LEARNING_RATE 1/5, DISCOUNT_FACTOR 8/10, explore 100%.
    let learning_rate = Q16_16::from_int(1) / Q16_16::from_int(5);
    let discount_factor = Q16_16::from_int(8) / Q16_16::from_int(10);
    QLearner::new(Maze::new(), learning_rate, discount_factor, 100)
}

/// One episode from a random room to the goal; returns its CSV line.
fn episode(learner: &mut Learner, out: &mut impl Write) -> io::Result<String> {
    learner.start_new_episode();
    let mut state = (learner.environment_mut().rng.next_i32() % STATES as i32) as usize;
    writeln!(out, "*** starting in state {state} ***")?;
    let mut line = format!("{state},");
    loop {
        // The maze has a door out of every room, so there is always an action.
        let action = learner.take_action(state).expect("every room has a door");
        writeln!(out, "take action {action}")?;
        line += &format!("{action},");
        let reward = learner.environment().reward(state, action);
        learner.update_from_experience(&Experience {
            state,
            action,
            reward,
            new_state: action,
        });
        state = action;
        if learner.state() == Some(GOAL) {
            return Ok(line);
        }
    }
}

fn to_double(value: Q16_16) -> f64 {
    f64::from(value.raw()) / 65536.0
}

/// maze.cpp's `reportDistanceFromOptimal`: Q* by 5000 rounds of value
/// iteration in `f64` (Q*(s, a) = r(s, a) + gamma * max_a' Q*(a, a'), since
/// action `a` leads to room `a`), the learned and optimal values of every
/// valid (state, action) to `path`, and the largest and RMS error to `out`.
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
        "Q table vs Q*: max |error| {}, RMS {}",
        fmt_g(max_error),
        fmt_g((sum_squares / f64::from(count)).sqrt())
    )
}

/// Runs the example, writing maze_training.csv, maze_qtable.csv and
/// maze_test.csv to `out_dir` and the log to `out`.
pub fn run(out_dir: &Path, out: &mut impl Write) -> io::Result<()> {
    let mut learner = new_learner();

    let mut training = BufWriter::new(File::create(out_dir.join("maze_training.csv"))?);
    for i in 0..TOTAL_NUMBER_OF_EPISODES {
        // After the pure random episodes, explore one percentage point less
        // each episode.
        if i >= NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES {
            let point = learner.random_action_decision_point();
            if point > 0 {
                learner.set_random_action_decision_point(point - 1);
            }
        }
        let line = episode(&mut learner, out)?;
        writeln!(training, "{line}")?;
    }
    training.flush()?;

    report_distance_from_optimal(&learner, &out_dir.join("maze_qtable.csv"), out)?;

    // The test runs are greedy and learn nothing.
    learner.set_random_action_decision_point(0);
    learner.set_learning_rate(Q16_16::ZERO);
    let mut test = BufWriter::new(File::create(out_dir.join("maze_test.csv"))?);
    for _ in 0..NUMBER_OF_TEST_EPISODES {
        let line = episode(&mut learner, out)?;
        writeln!(test, "{line}")?;
    }
    test.flush()?;

    if learner.state() == Some(GOAL) {
        writeln!(out, "Reached the goal state!")?;
    } else {
        writeln!(out, "Did not reach the goal state!")?;
    }
    out.flush()
}
