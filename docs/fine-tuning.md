---
title: Fine-tuning on the device
layout: default
nav_order: 9
---

# Fine-tuning on the device

A model trained on the host meets data on the device that it never saw: a sensor ages, a user moves differently, a new class appears. Retraining the whole network on a microcontroller needs the training state of every layer and backpropagation through all of them. **Head-only fine-tuning** keeps the shipped model frozen and adapts only a small classifier on top of its embedding, from a few labelled samples, on the device. `tinymind::head` is a port of tinymind (C++)'s two heads for this ([its page](https://danmcleran.github.io/tinymind/training/on-device-fine-tuning.html)), and the `gas_sensor_drift` example is a port of [its drift example](https://danmcleran.github.io/tinymind/examples/gas_sensor_drift.html), which uses both.

## The heads

| Head | How it learns | RAM (32-value embedding, 6 classes, Q16.16) |
|---|---|---|
| `DenseHead` | SGD on a dense layer, with a softmax cross-entropy loss (the default) or a squared-error loss | 792 bytes |
| `PrototypeHead` | One running-mean embedding per class; the nearest mean wins | 784 bytes |

Both sit after anything that produces an embedding: an inference-only network, an int8 pipeline whose output the application converts to a Q format (tinymind (C++)'s converters for this are not ported), or the layers in front of the last one of a trained network. Nothing is backpropagated through the frozen model, so there are no stored activations and no gradients for it, and neither head allocates. Both run on Q16.16, Q8.8, `f32` and `f64`.

- **`DenseHead`.** Each parameter moves by learning rate &times; (delta &times; input), rounded to nearest as in every trainable layer of the crate; there is no momentum. The update is fused into the backward pass, so the head keeps no gradient buffer: its RAM is (inputs&nbsp;+&nbsp;1)&nbsp;&times;&nbsp;outputs values. The softmax subtracts the largest logit before the exponential, so no format overflows however large the logits grow; in fixed point the exponential is a 96-entry lookup table, and a class more than 6 behind the largest logit gets probability 0, so a well separated head stops moving instead of growing its logits until the format wraps. The squared-error loss needs no table. `backward` gives the gradient with respect to the embedding, for adapting something in front of the head too; the example uses it to train the frozen layer's biases. To undo an adaptation, reload the shipped weights, kept in flash.
- **`PrototypeHead`.** No learning rate and no epochs: one pass over the labelled samples is the whole training. A class can be added at any time, and a class never learned is never predicted. A cap on each class's count turns its prototype into an exponential moving average, so it keeps tracking a slowly drifting input. Squared Euclidean distance is the default; Manhattan distance has no multiplies and far more headroom before it overflows.

The [API reference]({{ site.baseurl }}/api/tinymind/head/) has an example of each.

## The gas sensor drift example

Metal-oxide gas sensors age: their baseline and gain shift over weeks and months, and a classifier trained on fresh data loses accuracy on later measurements of the same gases. The example synthesizes a 16-sensor, 8-feature (128-value) array response for six gases over 10 batches (months), each later batch with a larger per-feature gain and offset, inspired by the UCI Gas Sensor Array Drift Dataset. A 128-32-6 Q16.16 network (ReLU hidden layer, sigmoid outputs) is trained on batch 1 only, then scored on every batch:

![Training loss, the drift curve and the batch-1 confusion matrix]({{ site.baseurl }}/assets/plots/gas_drift.png)

The network classifies all 180 batch-1 validation samples correctly, and its accuracy falls from batch 4 on, to 79.0% on batch 10.

The example then splits the trained network into a frozen backbone (the 128-to-32 layer) and a head (the 32-to-6 layer); the split reproduces the network's hidden values exactly on all 9,000 samples. On each drifted batch it draws 1, 2, 5, 10 or 20 labelled samples per class and tests on the rest of the batch, averaging over 10 draws, with seven methods, each starting from the shipped model:

| Method | What adapts | Batch 10, 1 sample per class | 5 per class | 20 per class |
|---|---|---|---|---|
| None | nothing | 0.794 | 0.794 | 0.791 |
| Re-normalize | the input statistics, from the samples (labels unused) | 0.889 | 0.923 | 0.938 |
| Re-normalize, whole batch | the input statistics, from the whole batch | 0.937 | 0.936 | 0.937 |
| `PrototypeHead` | one mean embedding per class | 0.987 | 0.999 | 1.000 |
| `DenseHead` | the head, 30 epochs of SGD | 0.953 | 0.995 | 0.997 |
| Backbone biases | the frozen layer's 32 biases, through `backward` | 0.848 | 0.967 | 0.990 |
| Biases and `DenseHead` | both | 0.961 | 0.995 | 0.998 |

![Accuracy per batch, batch 10 against the number of samples, and batch 1 after adapting]({{ site.baseurl }}/assets/plots/gas_adapt.png)

The `PrototypeHead` is the most accurate here from a single sample per class. The `DenseHead` keeps what it knew: after adapting to batch 10 it still classifies batch 1 perfectly, while the `PrototypeHead`, which moves its class means, gets 76.8% of batch 1 right after one sample per class and 97.5% after twenty. Re-normalizing to batch 10's statistics leaves batch 1 at chance. The data is synthetic; tinymind (C++)'s page reports the same comparison averaged over 10 data seeds.

The heads' RAM against training the whole network, for this example's network in Q16.16 (Rust sizes):

| Configuration | Bytes |
|---|---|
| Trainable `MultilayerPerceptron`, 128-32-6 | 70,680 |
| Inference-only `InferencePerceptron`, 128-32-6 | 17,304 |
| `DenseHead`, 32 inputs, 6 classes | 792 |
| `PrototypeHead`, 32 inputs, 6 classes | 784 |

An inference-only network plus a head adapts in about a quarter of the RAM that training the whole network takes.

## On the boards

The heads run on an STM32F411 (Cortex-M4F, 100 MHz) and an STM32L031 (Cortex-M0+, 32 MHz) with the same results as tinymind (C++) on the same data. A `DenseHead<Q16_16, 32, 6>` training step takes 101 µs on the M4F and 1.64 ms on the M0+; a whole adaptation, 30 epochs over 30 labelled samples, 89 ms and 1.44 s. A `PrototypeHead` learns a sample in 5 µs and 243 µs. On the M0+ the Rust is level with tinymind (C++); on the M4F its dense step takes 37% more cycles. Details: [On-board results]({{ site.baseurl }}/benchmarks#fine-tuning-heads).

## Run it

```bash
cargo run --release --example gas_sensor_drift -- output
```

The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| stdout | The training log, the drift curve, the check that backbone plus head reproduces the network, and the batch 10 table above |
| `gas_loss.csv` | Mean \|error\| every 2,000 training steps |
| `gas_confusion.csv` | The batch-1 validation confusion matrix |
| `gas_drift.csv` | Accuracy on each batch |
| `gas_adapt.csv` | Accuracy per method, batch and number of samples (mean and spread over 10 draws), and on batch 1 after adapting |

The figures are drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

## How it is checked

- `tests/gas_golden.rs` runs the example and compares its stdout and all four CSV files with the C++ example's.
- `tests/head_sweep.rs` compares both heads with tinymind (C++)'s `DenseHead` and `PrototypeHead` after every step: nine dense heads (Q16.16, Q8.8, `f64` and `f32`; both losses; logits far beyond the exponential's table; sums that wrap in Q8.8) and seven prototype heads (both distances; squared distances that wrap; count caps, loaded and forgotten classes, ties), plus the softmax on its own.
- Kani proofs show the heads never panic, whatever their parameters, inputs, classes (out of range too), learning rates and caps, that no class count exceeds its cap, and that the softmax's exponential stays in [0, 1] and is 0 at or below -6.

What the application still owns: saving the adapted head to flash (and how often), forgetting (keep a few samples of the original classes and train on them alongside the new ones, or reload the shipped head), and the quality of the labels, which every method here trusts.

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/common/gas_driver.rs)
