# PyTorch importer

Turns a trained PyTorch model into a Rust module for the `tinymind` crate: int8
weights and biases, integer requantizers, baked activation tables and a
`forward` function. The generated module is integer-only and `no_std`; it needs
no float math and no setup on the device.

It is a port of tinymind's [`apps/import_pytorch`](https://github.com/danmcleran/tinymind/tree/master/apps/import_pytorch).
Calibration and quantization are tinymind's code, unchanged, so a model imported
here has the same int8 numbers as the same model imported with tinymind; only the
output is Rust instead of a C++ header.

## Use

Needs Python 3.10+ and numpy. PyTorch is only needed to train the model and for
the `from_torch` helper.

```python
import numpy as np
import tinymind_import as ti   # tools/import_pytorch/tinymind_import.py

# A trained torch.nn.Sequential of Linear / ReLU / Sigmoid / Tanh:
layers = ti.from_torch(net)    # names: fc1, relu1, fc2, ..., sigmoid1

calibration = [x.numpy() for x in representative_inputs]   # float32 arrays
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration,
                        "my_model.rs", "my_model")
```

Then in the Rust crate:

```rust
mod my_model;

let y = my_model::forward(&input); // [i8; NUM_INPUTS] -> [i8; NUM_OUTPUTS]
```

Inputs are int8 in the `INPUT` grid and outputs int8 in the `OUTPUT` grid: a
real value is `SCALE * (q - ZERO_POINT)`, with the constants in the module. On a
host, `tinymind::int8::calibration::quantize` (feature `std`) converts float
inputs.

`torch_xor.py` is the whole flow on a real model: train XOR in PyTorch, import,
write the module. Without `from_torch`, describe the layers by hand with
`Dense` / `ReLU` / `Sigmoid` / `Tanh` and numpy arrays from `state_dict()`, as
`example_mlp.py` does.

## Observers

Each tensor's range comes from an observer, chosen per layer (`from_torch` takes
a `{name: observer}` map; the default is min/max):

| Observer | For |
|---|---|
| `MinMaxObserver()` | Bounded tensors (after ReLU, sigmoid, tanh) |
| `PercentileObserver(0.05, 99.95)` | Heavy tails: clips the most extreme ~0.1% |
| `KLDivergenceObserver()` | TensorRT-style entropy calibration; slowest, often the most accurate |

Calibrate on inputs that look like real ones; the ranges they produce become the
int8 grids.

## What it supports

A single chain of `Dense` layers, each optionally followed by `ReLU`, `Sigmoid`
or `Tanh`, with per-tensor symmetric int8 weights: what the crate runs today.
Anything else (convolution, batch norm, softmax, branches) is rejected with an
error naming the layer. tinymind's importer covers more; those layers will come
as the crate gains them.

## How it is checked

- `test_parity.py`: on several models (layer sizes, observers, activations,
  input ranges), every scale, zero point, weight and bias equals what tinymind's
  importer computes.
- `example_mlp.py` defines a 3-8-4-2 model that uses all three observers. CI
  imports it with both importers; the C++ build runs tinymind's output on its
  runtime (`scripts/cpp/imported_mlp.cpp`) and the Rust example
  (`examples/imported_mlp.rs`) runs this importer's output. The two print the
  same requantizers, table hash and outputs, byte for byte
  (`scripts/compare-with-cpp.sh`). CI also regenerates the committed module and
  checks it has not changed.
- The requantizers equal tinymind's `buildRequantizer` on the same float32
  scales. The sigmoid and tanh tables evaluate `expf`/`tanhf` from the system C
  library through `ctypes`, as tinymind's `buildQSigmoidLUT` and the crate's
  `int8::calibration` builders do, so the three agree (checked on 400 tables);
  without a C library the importer falls back to numpy and warns.
- The generated module is part of the crate's Cortex-M panic check: it links with
  no reachable panic.

If `rustfmt` is installed, the module is formatted as `cargo fmt` would.
