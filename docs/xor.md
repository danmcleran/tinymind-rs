---
title: XOR example
layout: default
nav_order: 2
---

# XOR example

The `xor` binary is a port of tinymind's [XOR example](https://danmcleran.github.io/tinymind/examples/xor.html): a multilayer perceptron with 2 inputs, 4 tanh hidden neurons and 1 sigmoid output learns the XOR function, trained entirely in fixed-point arithmetic.

## How it works

- Training samples are random XOR cases drawn with a re-implementation of glibc's `rand()`, seeded as the C++ example seeds it, so both programs see the same samples in the same order.
- Each step runs a forward pass, computes the mean-squared error, and back-propagates with SGD and momentum unless the error is already within tinymind's zero tolerance.
- Every 100 steps the program logs the mean absolute error over the four XOR cases. At the end it sweeps the trained network over a 41×41 input grid.

## Run it

```bash
cargo run --release --bin xor -- output                  # Q16.16 (the C++ example's format)
cargo run --release --bin xor -- --format q8.8 output    # Q8.8
cargo run --release --bin xor -- output 0.25 0.5         # optional learning rate and momentum
```

It writes three files into the output directory, the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `nn_fixed_xor.txt` | Every weight, the target, the output and the error after each of the 20,000 steps, in raw fixed-point units |
| `xor_training.csv` | The four-case error every 100 steps |
| `xor_decision_surface.csv` | The trained network's output over the input grid |

## Q16.16

![XOR learning curve, Q16.16]({{ site.baseurl }}/assets/plots/xor_q16_16_learning_curve.png)

The error falls smoothly from the symmetric-guess level (about 0.5) and settles within about 1,000 steps. All four cases are then classified correctly, with predictions of about 0.08 to 0.09 and 0.91. The remaining error is the sigmoid output not saturating fully to 0 or 1 under the squared-error objective.

![XOR decision surface, Q16.16]({{ site.baseurl }}/assets/plots/xor_q16_16_decision_surface.png)

The heatmap is the network's output over the input square, the black line is the 0.5 decision boundary, and the corner markers are the XOR targets (red = 1, blue = 0).

## Q8.8

tinymind ships this example in Q16.16 only. The Q8.8 run is the same program with 8 integer and 8 fractional bits, compared against the C++ example built the same way.

![XOR learning curve, Q8.8]({{ site.baseurl }}/assets/plots/xor_q8_8_learning_curve.png)

With 1/256 resolution, training lurches for about 3,500 steps before it snaps to a solution.

![XOR decision surface, Q8.8]({{ site.baseurl }}/assets/plots/xor_q8_8_decision_surface.png)

All four cases are classified correctly, but the surface shows how: the network outputs 0.914 at 99.6% of the grid points, and only the exact corners (0, 0) and (1, 1) fall below 0.5. It has fitted the four training inputs rather than learned an XOR boundary. tinymind's notes on this example say that Q8.8 with three hidden neurons lurched through plateaus in the same way, which is why its example uses Q16.16 with four.

The plots are drawn from this port's output with tinymind's plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/src/bin/xor.rs)
