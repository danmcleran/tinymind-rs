---
title: Iris example
layout: default
nav_order: 3
---

# Iris example

The `iris` example is a port of tinymind (C++)'s [iris example](https://danmcleran.github.io/tinymind/examples/iris.html): a three-way species classifier for the [UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) (sepal and petal measurements of setosa, versicolor and virginica flowers), trained entirely in Q16.16 fixed point.

## How it works

- A multilayer perceptron with 4 inputs, 8 ReLU hidden neurons and 3 sigmoid outputs, one per species. The prediction is the output with the largest value.
- The 150 samples are shuffled with a fixed seed and split 80/20 into training and test sets. Each feature is z-score normalised with training-set statistics, then divided by 3 to stay well inside Q16.16's range.
- 30,000 training steps, each on a training sample drawn at random, with SGD and momentum.

ReLU (`max(0, x)`) needs no lookup table, so in this port it works for every value type; its derivative is 1 where the neuron is active and 0 elsewhere.

## Run it

```bash
cargo run --release --example iris -- output
```

The dataset ships in the repository as `data/iris.data` (licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `NOTICE`). A different file can be passed as a second argument. The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `iris_loss.csv` | Mean absolute error over each block of 1,000 training steps |
| `iris_scatter.csv` | Petal length and width, true and predicted species, for each test sample |
| `iris_confusion.csv` | The test-set confusion matrix |

## Output

![Iris training loss, test confusion matrix and petal scatter]({{ site.baseurl }}/assets/plots/iris_classifier.png)

The error falls from about 0.066 after the first 1,000 steps to between about 0.01 and 0.02, and all 30 test samples are classified correctly. The petal scatter shows why the problem is easy for a small network: the three species form separate clusters.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/iris.rs)
