---
title: Energy example
layout: default
nav_order: 4
---

# Energy example

The `energy_efficiency` example is a port of tinymind (C++)'s [energy efficiency example](https://danmcleran.github.io/tinymind/examples/energy_efficiency.html): a regression that predicts a building's heating and cooling load from eight design features (relative compactness, surface, wall and roof areas, height, orientation, glazing area and its distribution), using the [UCI Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency), trained entirely in Q16.16 fixed point.

## How it works

- A multilayer perceptron with 8 inputs, 16 ReLU hidden neurons and 2 **linear** outputs: this is regression, so the outputs are the predicted loads themselves rather than class scores.
- The 768 samples are shuffled with a fixed seed and split 80/20 into training and test sets. The inputs are z-score normalised with training-set statistics and divided by 3; the targets are z-scored too and divided by 6, so a linear output's errors stay well inside Q16.16. Predictions are converted back to load units for the report.
- 40,000 training steps, each one training (however small its error), with a learning rate of 0.02, no momentum, and gradients **clipped** to [-1, 1]. A linear output's gradient is unbounded; without clipping it saturates the fixed-point multiply. As in tinymind (C++), the clip applies to the weights and not to the bias weights.

The linear activation is `tinymind::Linear`, and the clip is `set_gradient_clip`, available on both `MultilayerPerceptron` and the multi-layer `NeuralNetwork`.

## Run it

```bash
cargo run --release --example energy_efficiency -- output
```

The dataset ships in the repository as `data/ENB2012_data.csv` (licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `NOTICE`), byte for byte the file tinymind (C++) ships with its example. A different file can be passed as a second argument. The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `energy_loss.csv` | Mean absolute error over each block of 2,000 training steps |
| `energy_pred.csv` | True and predicted heating and cooling load for each test sample |

## Output

![Energy efficiency training loss and predicted-vs-actual loads]({{ site.baseurl }}/assets/plots/energy_regression.png)

The error falls from about 0.0096 over the first 2,000 steps to about 0.0020. On the 154 test buildings the mean absolute error is 1.74 for the heating load (R² 0.954) and 1.84 for the cooling load (R² 0.926). The scatter shows where a 16-neuron fixed-point network falls short: it under-predicts the largest cooling loads, predicting at most about 40 where the true values reach about 48.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/energy_efficiency.rs)
