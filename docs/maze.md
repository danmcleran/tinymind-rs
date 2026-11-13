---
title: Maze (Q-learning)
layout: default
nav_order: 7
---

# Maze (Q-learning)

The `maze` example is a port of tinymind (C++)'s [maze example](https://danmcleran.github.io/tinymind/examples/maze.html): a Q-learner finds the way out of a six-room maze by trial and error, with its Q table in Q16.16 fixed point.

```text
________________________________________________
|                       |                       |
|                       |                       |
|           0           |          1             / 5
|                       |                       |
|____________/  ________|__/  __________________|_______________________
|                       |                       |                       |
|                       |                        /                      |
|           4           |          3            |           2           |
|                        /                      |                       |
|__/  __________________|_______________________|_______________________|
    5
```

Rooms 0 to 4 are inside; 5 is outside, the goal. Action `a` means "go to room `a`" and is valid only through a door. Reaching room 5 earns a reward of 1; every other move earns 0.

## How it works

- `tinymind::qlearn::QLearner` keeps one Q value per (room, action) pair in a `QTable`: 36 Q16.16 values, 144 bytes. The maze tells it which actions each room allows and supplies random numbers, through the `Environment` trait.
- Each step updates the Q value of the move just made, as tinymind (C++) does: `Q(s, a) = (1 - 0.2) * Q(s, a) + 0.2 * (reward + 0.8 * max Q(s', a'))`, the maximum over the moves possible from the next room. Q-learning learns the best moves' values whatever the learner explores, so it can explore at random throughout.
- 5,000 training episodes each start in a random room and run until the learner is outside. The first 4,900 explore at random; over the last 100 the chance of a random move falls by one percentage point per episode. The example then compares the learned table with the optimal Q values, Q\*, computed by value iteration, and runs 100 test episodes that always take the best move and learn nothing.
- Random numbers come from the crate's re-implementation of glibc's `rand()`, seeded with 7 as in the C++, so every episode is the same as the C++ example's.

The C++ example stores Q values in unsigned Q16.16; the crate's Q formats are signed. Every Q value, reward, rate and discount in this example is non-negative, where the two have the same bits and the same arithmetic, and the Q-learning sweep (below) checks the Q table after every update against the unsigned C++.

## Run it

```bash
cargo run --release --example maze -- output
```

The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| stdout | Each episode's start room and every action taken, and the learned table's largest and RMS distance from Q\* |
| `maze_training.csv` | The rooms visited in each of the 5,000 training episodes |
| `maze_qtable.csv` | Each possible move's learned Q value and Q\* |
| `maze_test.csv` | The rooms visited in each of the 100 test episodes |

## Output

![Moves per training episode, and moves per test episode against the shortest path]({{ site.baseurl }}/assets/plots/maze_paths.png)

While it explores at random, the learner takes 6.2 moves per episode on average, and up to 56. Over the last 50 training episodes the average falls to 1.9. Every learned Q value is within 0.0007 of Q\* (RMS 0.00067): 5.0 for the moves outside, 4.0 one move further away, and so on down by the discount of 0.8. In all 100 test episodes the learner leaves by the shortest path from its starting room: 4-5 and 1-5 in one move, 0-4-5 and 3-1-5 in two, 2-3-1-5 in three, and from outside, 5-5.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

## How it is checked

- `tests/maze_golden.rs` runs the example and compares its stdout and all three CSV files with the C++ example's.
- `tests/qlearn_sweep.rs` hashes the whole Q table after every update and compares it with tinymind (C++)'s `QLearner`: the maze as above, and 60 random environments in Q16.16 and Q8.8 with negative rewards, rooms with zero, one, two or more valid actions, learning rates and discount factors of exactly 1 and ones the setters must reject (negative or above 1), exploration rates above 100, and updates for a state other than the learner's, including an invalid one.
- A Kani proof shows the learner never panics, whatever the Q values, rates, states and actions, and whatever the environment returns.

## The deep Q-network version

The `dqn_maze` example is a port of tinymind (C++)'s `dqn_maze`: the same maze, with the Q values in a neural network instead of a table. `tinymind::qlearn::QNetwork` (tinymind (C++)'s `QValueNeuralNetworkPolicy`) holds a 6-8-6 Q16.16 `MultilayerPerceptron`, tanh hidden layer and linear output (Q values reach 5, beyond tanh's range), whose input is the room as a one-hot vector and whose outputs are the six actions' Q values. Each update trains the network once on the taken action's target, leaving the other outputs where they are. The update's lookahead reads a target network, a copy of the online one refreshed every 10 training steps. Exploration stays at 100% for 9,900 of the 10,000 episodes: with no replay buffer, a long greedy phase lets the network drift on the moves it stops taking.

```bash
cargo run --release --example dqn_maze -- output
```

It writes `dqn_maze_training.csv`, `dqn_maze_qtable.csv` and `dqn_maze_test.csv`, as the maze example does, and the same files as the C++ example, byte for byte. The initial weights come from the C++'s `std::default_random_engine` through `uniform_real_distribution<double>(-1, 1)`, reproduced in the example.

![Moves per training episode, and moves per test episode against the shortest path, for the deep Q-network]({{ site.baseurl }}/assets/plots/dqn_maze_paths.png)

The network's Q values end within 0.00064 of Q\* (RMS 0.00047), and all 100 test episodes take the shortest path. `tests/dqn_maze_golden.rs` compares every output with the C++ example's; a Kani proof shows the network's reads and training steps never panic, whatever the weights, rates, states and actions.

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/maze.rs), and for the deep Q-network version, [dqn_maze](https://github.com/danmcleran/tinymind-rs/blob/master/examples/common/dqn_maze_driver.rs)
