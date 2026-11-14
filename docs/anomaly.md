---
title: Anomaly detection
layout: default
nav_order: 5
---

# Anomaly detection, trained on the device

A microcontroller watching a machine usually runs a model someone trained elsewhere. This example trains it on the microcontroller itself: an autoencoder learns what *normal* vibration looks like from the sensor stream, then flags anything it cannot reconstruct. No labelled faults are needed, and the model adapts to the machine it is attached to. It runs unchanged on a Cortex-M4F and on a Cortex-M0+ with 32 KB of flash and no FPU.

tinymind (C++) has no example like this one; it is built from parts of the port that are each checked bit for bit against tinymind (C++) (multi-layer training, the linear output, gradient clipping).

## How it works

- **The signal.** A machine's vibration is simulated as a sine tone with random phase and amplitude (0.4 to 0.6) plus noise, cut into windows of 8 samples. The data is synthetic, generated on the device from an integer sine table and a deterministic random-number generator, so the example needs no dataset.
- **The model.** An autoencoder, `NeuralNetwork<(Dense<Q16_16, Tanh, 8, 4>, Dense<Q16_16, Linear, 4, 8>)>`: 8 inputs squeezed through 4 tanh neurons and expanded back to 8 linear outputs, in Q16.16. It is trained to reproduce its input on 3,000 normal windows (SGD with momentum, gradients clipped to [-1, 1]).
- **The threshold.** After training, the device scores 256 more normal windows. A window's score is its reconstruction error (the mean squared difference between the network's output and its input), and the threshold is twice the largest normal score.
- **The test.** 256 fresh normal windows and 64 windows of each of four faults: the tone at a different frequency, an added third harmonic, a single large impulse, and a stalled sensor that holds one value.

## Run it

```bash
cargo run --release --example anomaly -- output   # also writes output/anomaly_scores.csv
```

On the boards, the same code is the `anomaly` workload of the on-board checks (`WORKLOADS=anomaly bench/run.sh`, see [On-board results]({{ site.baseurl }}/benchmarks)).

## Results

![Reconstruction error of every test window, by kind, against the threshold]({{ site.baseurl }}/assets/plots/anomaly_scores.png)

| Windows | Flagged |
|---|---|
| Normal | 0 of 256 (no false alarms) |
| Wrong frequency | 64 of 64 |
| Added harmonic | 64 of 64 |
| Impulse | 64 of 64 |
| Stalled sensor | 46 of 64 |

Normal windows reconstruct with errors two to three orders of magnitude below the frequency, harmonic and impulse faults. A stalled sensor is the hard case: when it sticks near zero, its flat window looks like a quiet stretch of normal signal, and 18 of the 64 go unflagged. A real deployment would add a rule for it (for example, no change across several windows).

## On the boards

The boards run the same program and print the same report. Its hash covers every trained weight and every score, and it matches the host's on both boards, so the microcontrollers trained the network and made every decision bit for bit as the desktop did.

| | NUCLEO-F411RE (Cortex-M4F, 100 MHz) | NUCLEO-L031K6 (Cortex-M0+, 32 MHz) |
|---|---:|---:|
| Training, 3,000 steps (signal generation included) | 0.27 s | 4.5 s |
| One training step | 8,489 cycles (85 µs) | 38,557 cycles (1.2 ms) |
| One detection (forward pass and error) | 1,611 cycles (16 µs) | 8,084 cycles (253 µs) |
| Stack, over the empty harness | 3.8 KB | 3.6 KB |
| Flash, over the empty harness | 18.6 KB | 18.3 KB |

The F411 figures are built with `-C target-cpu=cortex-m4`, like the other on-board results. Without it a training step takes 6,279 cycles and a detection 1,332: with that setting, LLVM's instruction scheduler moves the demo's small layers' products to the stack ([Settings for your own build]({{ site.baseurl }}/benchmarks#settings-for-your-own-build) has the options and what they cost other networks).

The flash figures include the demo's signal generator and its text report, not only the network. The L031 image leaves about 2.0 KB of its 32 KB of flash free.

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/common/anomaly_demo.rs)
