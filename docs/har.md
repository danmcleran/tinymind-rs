---
title: Activity recognition (LSTM)
layout: default
nav_order: 8
---

# Activity recognition (LSTM)

The `har_activity` example is a port of tinymind (C++)'s [human activity recognition example](https://danmcleran.github.io/tinymind/examples/har_activity.html): an LSTM network reads a window of 32 accelerometer samples, one sample at a time, and at the window's last sample says which of four activities it shows: walking, walking upstairs, walking downstairs or sitting. Upstairs and downstairs are exact time reversals of each other, so no single sample, and no per-axis average or spread, tells them apart: only the order of the samples does, which the network can see only by remembering earlier samples. The network is 3 inputs (the x, y and z accelerations), 16 LSTM cells and 4 sigmoid outputs, in Q16.16 fixed point.

## How it works

- `tinymind::recurrent::LstmNetwork` is tinymind (C++)'s `LstmNeuralNetwork` with one hidden layer. Each cell has four gates (cell candidate, input, forget, output), each with its own weights from the inputs, a bias and the previous step's 16 cell outputs: 1,348 trainable weights in all, 1,024 of them recurrent.
- Each `feed_forward` takes one sample and updates every cell from it and from the previous step's outputs. `reset_state` clears the cells before each window, so windows do not leak into each other.
- The example trains toward the window's activity at every sample, not only the last. Training goes one step back in time, as in the C++, with plain SGD (learning rate 0.1, no momentum) and gradients clipped to [-1, 1].
- The windows are synthesized. The three gait classes share a step rate, a sideways sway and gravity on the x axis, and differ only in the vertical waveform, scaled to the same mean and spread: a sine for walking, a sawtooth for the stairs (a sharp rise then a slow decline going up, the reverse going down). Sitting is constant gravity on the y axis. Every window has noise from the crate's re-implementation of glibc's `rand()`, seeded with 7 as in the C++, so the 800 training and 200 test windows are the C++ example's, bit for bit. A `har.csv` file in the C++ example's format (`window,t,ax,ay,az,label`) can be given instead.

## Run it

```bash
cargo run --release --example har_activity -- output
cargo run --release --example har_activity -- --readout-only output
```

`--readout-only`, as in the C++, keeps the LSTM's input, gate and recurrent weights at their random initial values and trains only the 16-to-4 readout: the comparison that shows what the recurrent weights learn.

The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| stdout | The mean error per training epoch, the test accuracy and two baselines: nearest centroid on per-axis mean and standard deviation (which ignores sample order) and chance |
| `har_samples.csv` | One synthesized window per activity |
| `har_loss.csv` | The mean error at each window's last sample, per epoch |
| `har_confusion.csv` | The test confusion matrix |

## Output

![Training loss, test confusion matrix and one window per activity]({{ site.baseurl }}/assets/plots/har_activity.png)

The mean error at a window's last sample stays near 0.09 for 30 epochs, then falls within about seven epochs and stays between about 0.001 and 0.006. The network classifies 153 of the 200 test windows correctly (76.5%): every upstairs, downstairs and sitting window, but only 3 of the 50 walking windows, 45 of which it takes for upstairs. Recurrent training in Q16.16 is noisy on this task and this data seed (7) is a weak one: with the data seed set to 1 through 10 (initial weights unchanged), the network scores 89.6% on average, from 73.5% to 100%, never at chance, while `--readout-only` stays at 50.0% on every seed. The nearest-centroid baseline gets 52.5% at the default seed and chance is 25%: what separates the stairs classes is learned by the recurrent weights.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

## How it is checked

- `tests/har_golden.rs` runs the example and compares its stdout and all three CSV files with the C++ example's; `scripts/compare-with-cpp.sh` also compares a `--readout-only` run with the C++ one.
- `tests/lstm_sweep.rs` hashes the outputs and every weight after every step and compares them with tinymind (C++)'s `LstmNeuralNetwork` in seven configurations: this example's network, other shapes, Q8.8, ReLU cells, linear outputs, gradient clipping and the Adam optimizer, on sequences of 1 to 12 steps with the state reset between them and some steps not trained. `tests/float_mlp.rs` does the same for four `f32` and `f64` LSTM networks.
- A Kani proof shows the network never panics, whatever its weights, rates, clipping range, inputs and targets.

tinymind (C++)'s GRU and Elman networks are not ported yet.

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/har_activity.rs)
