---
title: Digits from ONNX (int8, int4, 16x8)
layout: default
nav_order: 14
---

# Digits from ONNX: one network, three precisions

The `onnx_digits_int8` example is a port of tinymind (C++)'s [ONNX digits example](https://danmcleran.github.io/tinymind/examples/onnx_digits_int8.html): a small residual CNN trained in PyTorch on the UCI optical digits (8x8 bitmaps), quantized by ONNX Runtime, and run from generated code with no float and no allocation.

## How it works

- The network: Conv 3x3 (8 filters) + ReLU, Conv 3x3 + ReLU added to its input (a residual connection), MaxPool 2x2, depthwise Conv 3x3 + ReLU, pointwise Conv (16) + ReLU, and a dense layer to 10 logits.
- The models are the `.onnx` files tinymind (C++)'s example commits. `tools/import_onnx/example_digits.py` turns them into three Rust modules with the [ONNX importer]({{ site.baseurl }}/int8#importing-an-onnx-model):
  - **int8**: the model as ONNX Runtime quantized it (3,408 weight bytes);
  - **int4**: the model fine-tuned with int4 quantization-aware training, every weight layer on packed per-channel int4 weights (1,704 weight bytes);
  - **16x8**: the int8 model with int16 activations (TFLite's 16x8 scheme), same weights.
- Every module runs in one activation arena that the caller passes to `forward`. tinymind (C++)'s planner places each layer's output in it: the residual addition writes its sum over the first convolution's output, which it is the last layer to read, so the seven layers fit in 1,024 bytes (2,048 for 16x8), half of what a buffer per output would take. The arena form of the importer needs Rust 1.88.
- For each of the 1,797 test digits, the program maps the pixels onto the input grid, runs the three models, and compares their classes with the labels, with ONNX Runtime's int8 model and with the float model. For the first 256 digits it compares the int8 and 16x8 logits with the same weights run with float activations.

## Run it

```bash
cargo run --release --example onnx_digits_int8 -- output
```

It writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `onnx_digits_int8.csv` | Per digit: label, the int8, int4 and 16x8 classes, ONNX Runtime's and the float model's, and the 64 pixels |
| `onnx_digits_int8_logits.csv` | The int8 and 16x8 logits in real units beside the float-activation reference, first 256 digits |
| `onnx_digits_int8_plan.csv` | The arena plan: each buffer's offset, size, live layers and outputs |
| `onnx_digits_int8_summary.csv` | Accuracies, agreement, arena and weight sizes, logit errors |

## Output

![Arena plan, test confusion matrix, and the logit error of int8 and 16x8 activations]({{ site.baseurl }}/assets/plots/onnx_digits_int8.png)

The int8 model reads 96.99% of the test digits correctly, as the float model and ONNX Runtime do, and gives ONNX Runtime's class on 1,795 of 1,797 digits; on the other two, ONNX Runtime's two highest logits are one step apart. The int4 model reads 97.05% with half the weight bytes. With int16 activations the logits are 126 times closer to the float-activation ones (mean error 0.0013 against 0.16) for twice the arena, and the 16x8 model agrees with the float model on 1,795 digits (int8: 1,791).

On the boards (tinymind (C++)'s tier1 benchmark digits, [On-board results]({{ site.baseurl }}/benchmarks)), the int8 model takes 287,802 cycles on the Cortex-M4F and 719,369 on the Cortex-M0+, the 16x8 model 408,880 and 854,511.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/onnx_digits_int8.rs)
