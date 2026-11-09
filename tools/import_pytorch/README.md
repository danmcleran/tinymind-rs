# PyTorch importer

Turns a trained PyTorch model into a Rust module for the `tinymind` crate: int8
weights and biases, integer requantizers, baked activation tables and a
`forward` function. The generated module is integer-only and `no_std`; it needs
no float math and no setup on the device.

It is a port of tinymind (C++)'s [`apps/import_pytorch`](https://github.com/danmcleran/tinymind/tree/master/apps/import_pytorch).
Calibration and quantization are tinymind (C++)'s code, unchanged, so a model imported
here has the same int8 numbers as the same model imported with tinymind (C++); only the
output is Rust instead of a C++ header. tinymind (C++)'s importer leaves the layer types to
hand-written C++; this one also writes the network itself, so it tracks each
layer's shape, stride and padding.

## Use

Needs Python 3.10+ and numpy. PyTorch is only needed to train the model and for
the `from_torch` helper.

```python
import numpy as np
import tinymind_import as ti   # tools/import_pytorch/tinymind_import.py

# A trained torch.nn.Sequential of Linear / ReLU / Sigmoid / Tanh / Softmax:
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

A convolutional model needs its input shape, in PyTorch's order, and its
calibration inputs as NHWC arrays (the crate's layout):

```python
layers = ti.from_torch(net, input_shape=(1, 28, 28))   # (channels, height, width)
calibration = [x.permute(1, 2, 0).numpy() for x in representative_images]
ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration,
                        "my_cnn.rs", "my_cnn", input_shape=(28, 28, 1))  # (h, w, c)
```

The module's `forward` then takes an `&Input`, a `[[[i8; C]; W]; H]` feature
map. `from_torch` turns PyTorch's NCHW tensors and OIHW weights into NHWC and
OHWI, and reorders the columns of a `Linear` that follows a `Flatten` to match.

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

A single chain, with per-tensor symmetric int8 weights: what the crate runs
today.

| Layer | `from_torch` | Becomes |
|---|---|---|
| `Dense` | `nn.Linear` | `QDense` |
| `Conv2D` | `nn.Conv2d`: stride, zero padding (including `"same"`), groups 1 or depthwise (groups = in = out channels), no dilation | `QConv2D`, `QDepthwiseConv2D`, or `QPointwiseConv2D` for a 1x1 stride-1 unpadded one |
| `ReLU` (directly after a `Dense` or `Conv2D`) | `nn.ReLU` | `qrelu` after a dense layer; folded into a convolution's requantizer bound |
| `MaxPool2D`, `AvgPool2D` (no padding) | `nn.MaxPool2d`, `nn.AvgPool2d` | `QMaxPool2D`, `QAvgPool2D`, in the input's grid |
| `GlobalAvgPool2D` | `nn.AdaptiveAvgPool2d(1)` | `QGlobalAvgPool2D`, in the input's grid |
| `Flatten` | `nn.Flatten` | a copy of the feature map in NHWC order |
| `Sigmoid`, `Tanh` (on vectors) | `nn.Sigmoid`, `nn.Tanh` | a 256-entry table |
| `Softmax` (last axis, on vectors) | `nn.Softmax` | `QSoftmax`, bit-exact with TFLite's int8 softmax; output on the probability grid (scale 1/256, zero point -128) |

Pooling and flattening keep their input's scale and zero point, as the crate's
pooling layers have no requantizer. Anything else is rejected with an error
naming the layer. **Batch normalization is not supported yet**: fold it into the
preceding convolution before importing. tinymind (C++)'s importer covers more
(branches, mixture of experts); those layers will come as the crate gains them.

## How it is checked

- `test_parity.py`: on five dense models (layer sizes, observers, activations
  including softmax, input ranges) and the two convolutional examples, every
  scale, zero point, weight and bias equals what tinymind (C++)'s importer
  computes. tinymind (C++)'s importer has no pooling or flatten layers, so for it
  each one runs inside the next layer's float forward; since they keep their
  input's grid here, the calibrated ranges are the same.
- `example_mlp.py` defines two models: a 3-8-4-2 MLP that uses all three
  observers and ends in sigmoid, and a 4-10-3 classifier that ends in softmax.
  CI imports both with both importers; the C++ build runs tinymind (C++)'s output on
  its runtime (`scripts/cpp/imported_mlp.cpp`, with tinymind (C++)'s TFLite-exact
  softmax) and the Rust example (`examples/imported_mlp.rs`) runs this
  importer's output. The two print the same requantizers, table hash, softmax
  parameters and outputs, byte for byte (`scripts/compare-with-cpp.sh`). CI also
  regenerates the committed modules and checks they have not changed.
- `example_cnn.py` defines two convolutional models: a 12x12x1 network with a
  padded convolution, max pooling, depthwise and pointwise convolutions, global
  average pooling and softmax, and a 9x11x2 network with a strided convolution,
  average pooling, flatten, two dense layers and tanh. As for the MLPs, CI runs
  tinymind (C++)'s import on its runtime (`scripts/cpp/imported_cnn.cpp`) and
  this importer's in the Rust example (`examples/imported_cnn.rs`); they print
  the same requantizers, a hash of every intermediate tensor and the outputs for
  16 inputs, byte for byte.
- `torch_cnn.py` (needs PyTorch; not run in CI) checks `from_torch` against
  PyTorch itself: on three models covering every supported module, the float
  model built from the descriptors on NHWC inputs equals the PyTorch module on
  NCHW inputs to within float rounding. Planted layout mistakes (no `Flatten`
  reorder, rows and columns of the kernel swapped, `"same"` padded on the wrong
  side, padding axes swapped) each fail it.
- The softmax parameters equal tinymind (C++)'s `buildQSoftmaxFixedPointParams`
  on 504 calibrations (`test_parity.py`).
- The requantizers equal tinymind (C++)'s `buildRequantizer` on the same float32
  scales. The sigmoid and tanh tables evaluate `expf`/`tanhf` from the system C
  library through `ctypes`, as tinymind (C++)'s `buildQSigmoidLUT` and the crate's
  `int8::calibration` builders do, so the three agree (checked on 400 tables);
  without a C library the importer falls back to numpy and warns.
- The generated modules (one dense, two convolutional) are part of the crate's
  Cortex-M panic check: they link with no reachable panic.

If `rustfmt` is installed, the module is formatted as `cargo fmt` would.
