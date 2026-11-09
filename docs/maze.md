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
- Each step updates the Q value of the move just made, as tinymind (C++) does: `Q(s, a) = (1 - 0.2) * Q(s, a) + 0.2 * (reward + 0.8 * Q(s', a'))`, where `a'` is the action the learner would choose next.
- 500 training episodes each start in a random room and run until the learner is outside. The first 400 explore at random; from then on the chance of a random move falls by one percentage point per episode, so the learner increasingly follows its Q table. 100 test episodes follow.
- Random numbers come from the crate's re-implementation of glibc's `rand()`, seeded with 7 as in the C++, so every episode is the same as the C++ example's.

The C++ example stores Q values in unsigned Q16.16; the crate's Q formats are signed. Every Q value, reward, rate and discount in this example is non-negative, where the two have the same bits and the same arithmetic, and the Q-learning sweep (below) checks the Q table after every update against the unsigned C++.

## Run it

```bash
cargo run --release --example maze -- output
```

The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| stdout | Each episode's start room and every action taken |
| `maze_training.csv` | The rooms visited in each of the 500 training episodes |
| `maze_test.csv` | The rooms visited in each of the 100 test episodes |

## Output

![Moves per training episode, and moves per test episode against the shortest path]({{ site.baseurl }}/assets/plots/maze_paths.png)

While it explores at random, the learner takes 6.0 moves per episode on average, and up to 38. Over the last 50 training episodes the average falls to 2.5. In all 100 test episodes it leaves by the shortest path from its starting room: 4-5 and 1-5 in one move, 0-4-5 and 3-1-5 in two, 2-3-1-5 in three, and from outside, 5-5.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

## How it is checked

- `tests/maze_golden.rs` runs the example and compares its stdout and both CSV files with the C++ example's.
- `tests/qlearn_sweep.rs` hashes the whole Q table after every update and compares it with tinymind (C++)'s `QLearner`: the maze as above, and 60 random environments in Q16.16 and Q8.8 with negative rewards, rooms with zero, one, two or more valid actions, and learning rates, discount factors and exploration rates the setters must reject.
- A Kani proof shows the learner never panics, whatever the Q values, rates, states and actions, and whatever the environment returns.

tinymind (C++)'s deep Q-network (`QValueNeuralNetworkPolicy`, its `dqn_maze` example) is not ported yet.

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/maze.rs)
