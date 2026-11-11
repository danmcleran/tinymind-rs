---
paths:
  - "src/qlearn.rs"
  - "examples/maze.rs"
  - "examples/common/maze_driver.rs"
  - "tests/maze_golden.rs"
  - "tests/qlearn_sweep.rs"
  - "scripts/cpp/qlearn_sweep.cpp"
---

# Q-learning and the maze

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.

## `src/qlearn.rs`, `examples/maze.rs`, `examples/common/maze_driver.rs`

C++ source: `cpp/qlearn.hpp` (`QLearner`, `QValueTablePolicy`, `ArgMaxPolicy`, `DefaultLearningPolicy`, `QLearningEnvironment`), `examples/maze`

2026-11-09. `QLearner<V, E, STATES, ACTIONS, Q = QTable>` owns its `Environment<ACTIONS>` (user trait: `valid_actions` fills a buffer, `random_int(min, max)` = the C++ RandomNumberPolicy's `randInt`) and its `QValues<V>` (open trait; `QTable { pub values: [[V; A]; S] }`, out-of-range get 0 / set ignored). States and actions are `usize`, invalid = `None` (C++ uses the type's max). Learning rate, discount (`< ONE`) and exploration percent (`<= 100`, the C++ example policy's check) live in the learner; initial 0 (the C++ example policy leaves the percent uninitialized). The C++ `environment.takeAction(experience)` hook is dropped (the maze only stores it). Since tinymind #261 (`1b310c0`, ported 2026-11-10): the update is for `experience.state` (Experience gained `state`; >= STATES updates nothing, whatever the learner's own state), the future Q is `Q(s', a*)` with a* = ArgMax over s''s valid actions (first valid action when all tie, 0 when none; never draws RNG), exploration draws `random_int(0, 99) < point` (exactly p/100), ArgMax with 2 actions treats a tie as no preference (random), rates are accepted in [0, 1] (`isInUnitInterval` = `!(v < 0) && !(1 < v)`, so NaN passes; negatives now rejected, 1 accepted); nothing computed when lr = 0. ArgMax with 3+ (since #258): first of the largest Q whatever its sign, random only when every valid action ties. The C++ maze uses unsigned Q16.16 (`QValue<16,16,false>`); the port uses signed `Q16_16` (all values non-negative, -1 reward only compared), proved by the sweep hashing the unsigned C++ table. C++ `Constants` has no float specialization, so its QLearner needs a Q format; ours also runs on f32/f64. maze since #261: 5000 episodes (decay over the last 100), Q* by value iteration in f64 (`maze_qtable.csv`, max |err| 0.00069), greedy test runs at lr 0; C++ built with -ffp-contract=off; stdout (571 KB) and training CSV pinned by FNV-1a in `tests/maze_golden.rs`. Checked by `tests/maze_golden.rs` and `tests/qlearn_sweep.rs` (golden from `scripts/cpp/qlearn_sweep.cpp`, UBSan clean: maze in UQ16.16 on the example's schedule + 60 random envs in Q16.16/Q8.8 with 0/1/2/3+ valid actions, negative rewards, lr/df 0, exactly 1, > 1, < 0, exploration 0/100/150/random with decay, every 4th episode starting with an update for a random or invalid (S) state, every 5th step followed by a replayed experience for a random state). Plants that fail it (2026-11-10): exploring bootstrap, learner's state instead of experience.state, (p+1)/101 draw, 2-action tie to a1, rate 1 rejected, negative rate accepted, all-tied bootstrap drawing a random action. Equivalent mutant: dropping the `state < STATES` guard (the table ignores out-of-range writes and the bootstrap draws nothing). Kani `qlearner_never_panics` (304 checks, ~4 s; planted `valid[index]` fails it and panic-never). **Not ported yet: `QValueNeuralNetworkPolicy` (DQN)**. Until tinymind #258 its `getQValue(state, action)` ignored `state` and returned the online net's last outputs (shown by a probe: identical values for all six states); #258 feeds the state forward (cached until `setQValue` trains) and changed `examples/dqn_maze` to a one-hot state and a linear output. `QValues::q_value` takes `&self`; the DQN's needs a forward pass, so porting it changes that trait: 0.3.0 shipped it, so that is a breaking change (0.4.0), or a new trait beside it
