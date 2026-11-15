---
title: Handwritten digits
layout: default
nav_order: 4
---

# Handwritten digits example

The `optical_digits` example is a port of tinymind (C++)'s [optical digits example](https://danmcleran.github.io/tinymind/examples/optical_digits.html): a ten-way classifier for the [UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), trained entirely in Q16.16 fixed point. Each sample is an 8x8 bitmap (64 pixels, intensity 0 to 16) downsampled from a 32x32 scan of a handwritten digit.

## How it works

- A multilayer perceptron with 64 inputs (one per pixel, row by row), 32 ReLU hidden neurons and 10 sigmoid outputs, one per digit. The prediction is the output with the largest value.
- The dataset comes already split: 3,823 training images and 1,797 test images, written by different people. Each pixel is z-score normalised with training-set statistics, then divided by 3 to stay well inside Q16.16's range. Border pixels that are blank in every training image map to a constant 0.
- 60,000 training steps, each on a training image drawn at random, with SGD and momentum.

It is the [iris example]({{ site.baseurl }}/iris) scaled up: the same network type and training, with 16 times the inputs and more than three times the classes. The network has 2,410 weights and biases, 9,640 bytes as Q16.16 values.

## Run it

```bash
cargo run --release --example optical_digits -- output
```

The data ships in the repository as `data/optdigits.tra` and `data/optdigits.tes` (licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `NOTICE`). They are left out of the published crate, so run the example from a clone of the repository. A different data directory can be passed as a second argument. The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `digits_loss.csv` | Mean absolute error over each block of 2,000 training steps |
| `digits_confusion.csv` | The 10x10 test-set confusion matrix |
| `digits_samples.csv` | 40 test images (64 pixels, true and predicted digit): the first 20 misclassified ones, then correctly classified ones |

## Output

![Digit classifier training loss and test confusion matrix]({{ site.baseurl }}/assets/plots/optical_digits.png)

The error falls from about 0.020 over the first 2,000 steps to about 0.001 by the end, and 1,729 of the 1,797 test images (96.2%) are classified correctly. Zeros and sixes are the easiest (98.9% each) and eights the hardest (90.8%); the most common mistakes are an 8 or a 7 read as a 9.

![Grid of sampled test digits with true and predicted labels]({{ site.baseurl }}/assets/plots/optical_digits_grid.png)

The grid shows the images themselves, `t` the true digit and `p` the prediction. Many of the misclassified ones are hard to read at 8x8 even for a person.

The figures are drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/optical_digits.rs)
