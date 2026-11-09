// The maze example's program: tinymind (C++)'s examples/maze (maze.cpp,
// mazelearner.h, mazelearner.cpp), shared by examples/maze.rs and
// tests/maze_golden.rs.
//
// Six rooms; room 5 is outside, the goal. Action `a` moves to room `a`, valid
// only through a door. 500 training episodes start in a random room and run
// until they reach the goal: the first 400 always explore, then exploration
// drops by one percentage point per episode. 100 test episodes follow. Each
// episode's path goes to a CSV line and to stdout.
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

use tinymind::Q16_16;
use tinymind::qlearn::{Environment, Experience, QLearner};
use tinymind::rand::GlibcRand;

pub const STATES: usize = 6;
pub const ACTIONS: usize = 6;
const GOAL: usize = 5;
const RANDOM_SEED: u32 = 7;

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

/// Runs the example, writing maze_training.csv and maze_test.csv to `out_dir`
/// and the log to `out`.
pub fn run(out_dir: &Path, out: &mut impl Write) -> io::Result<()> {
    let mut learner = new_learner();

    let mut training = BufWriter::new(File::create(out_dir.join("maze_training.csv"))?);
    for i in 0..500 {
        // After 400 random episodes, explore one percentage point less each episode.
        if i >= 400 {
            let point = learner.random_action_decision_point();
            if point > 0 {
                learner.set_random_action_decision_point(point - 1);
            }
        }
        let line = episode(&mut learner, out)?;
        writeln!(training, "{line}")?;
    }
    training.flush()?;

    let mut test = BufWriter::new(File::create(out_dir.join("maze_test.csv"))?);
    for _ in 0..100 {
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
