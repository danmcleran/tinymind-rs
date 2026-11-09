"""
Copyright (c) 2026 Dan McLeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

"""
PyTorch -> tinymind (Rust) int8 importer.

A port of tinymind's apps/import_pytorch/tinymind_import.py that writes a Rust
module for the `tinymind` crate instead of a C++ weights.hpp. Calibration and
quantization are tinymind's code, unchanged, so the int8 numbers are the same;
only the output differs:

  * tinymind emits float scales and leaves the C++ to build requantizers and
    sigmoid/tanh tables at startup (which needs float and std);
  * this emits the finished integer requantizers, baked activation tables and a
    `forward` function, so the generated module is no_std and float-free.

Supported layers (what the crate can run today), in a single chain: Dense,
ReLU, Sigmoid, Tanh and Softmax (over the last axis) on vectors; Conv2D
(tinymind's descriptor, plus the stride, padding and depthwise flag the
generated Rust needs), MaxPool2D, AvgPool2D, GlobalAvgPool2D and Flatten on
NHWC feature maps. A ReLU follows a Dense or Conv2D directly. BatchNorm2D is not
supported yet. Observers: MinMaxObserver, PercentileObserver,
KLDivergenceObserver (per tensor).

numpy only; PyTorch is needed only to train the model, and `from_torch` imports
it lazily.

Rounding follows std::lround (ties away from zero). The requantizers equal what
tinymind's buildRequantizer computes from the same float32 scales, and the
tables equal buildQSigmoidLUT / buildQTanhLUT (and the crate's
int8::calibration builders), because they evaluate expf / tanhf from the system
C library through ctypes, as those do; without it they fall back to numpy and
say so. Softmax parameters equal buildQSoftmaxFixedPointParams (TFLite Micro's),
for tinymind's TFLite-exact QSoftmaxFixedPoint1D and the crate's QSoftmax.
"""

import ctypes
import ctypes.util
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Rounding / quantization primitives (tinymind's, unchanged).
# ----------------------------------------------------------------------------


def lround(x: float) -> int:
    """C++ std::lround semantics: ties round away from zero."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def affine_asymmetric(fmin: float, fmax: float,
                      qmin: int = -128, qmax: int = 127) -> tuple[float, int]:
    """Mirror cpp/include/qcalibration.hpp::computeAffineParamsAsymmetric."""
    if fmin > 0.0:
        fmin = 0.0
    if fmax < 0.0:
        fmax = 0.0
    qrange = float(qmax - qmin)
    if fmax == fmin or qrange == 0.0:
        return 1.0, qmin
    scale = (fmax - fmin) / qrange
    zp = lround(qmin - fmin / scale)
    zp = max(qmin, min(qmax, zp))
    return scale, int(zp)


def symmetric_weight_scale(w: np.ndarray, qmax_signed: int = 127) -> float:
    absmax = float(np.max(np.abs(w))) if w.size else 0.0
    if absmax == 0.0:
        return 1.0
    return absmax / float(qmax_signed)


def quantize_symmetric_weights(w: np.ndarray, scale: float) -> np.ndarray:
    flat = w.reshape(-1).tolist()
    q = [max(-127, min(127, lround(v / scale))) for v in flat]
    return np.array(q, dtype=np.int8).reshape(w.shape)


def quantize_bias_int32(b: np.ndarray, bias_scale: float) -> np.ndarray:
    qmin = -(1 << 31)
    qmax = (1 << 31) - 1
    q = [max(qmin, min(qmax, lround(v / bias_scale))) for v in b.tolist()]
    return np.array(q, dtype=np.int32)


def quantize_multiplier(ratio: float) -> tuple[int, int]:
    """Mirror cpp/qaffine.hpp::quantizeMultiplier (and the crate's
    int8::calibration::quantize_multiplier): (multiplier, shift)."""
    if ratio == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(ratio)
    q = lround(mantissa * float(1 << 31))
    if q == (1 << 31):
        q >>= 1
        exponent += 1
    shift = -int(exponent)
    # Keep the shift in -31..31, Requantizer's contract: a smaller ratio
    # rounds every int32 accumulator to 0, a larger left shift saturates.
    if shift > 31:
        return 0, 0
    return int(q), max(shift, -31)


# ----------------------------------------------------------------------------
# Observers (tinymind's, unchanged).
# ----------------------------------------------------------------------------


class MinMaxObserver:
    """Streaming min/max range observer."""

    def __init__(self) -> None:
        self.min = math.inf
        self.max = -math.inf
        self.has_data = False

    def observe(self, x: np.ndarray) -> None:
        flat = x.reshape(-1)
        if flat.size == 0:
            return
        lo = float(flat.min())
        hi = float(flat.max())
        if lo < self.min:
            self.min = lo
        if hi > self.max:
            self.max = hi
        self.has_data = True

    def range(self) -> tuple[float, float]:
        if not self.has_data:
            return 0.0, 0.0
        return self.min, self.max


class PercentileObserver:
    """Percentile clipping observer. Stores every sample."""

    def __init__(self, lower: float = 0.05, upper: float = 99.95) -> None:
        self.lower = lower
        self.upper = upper
        self._samples: list[np.ndarray] = []
        self.has_data = False

    def observe(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._samples.append(x.reshape(-1).astype(np.float32))
        self.has_data = True

    def range(self) -> tuple[float, float]:
        if not self.has_data:
            return 0.0, 0.0
        cat = np.concatenate(self._samples)
        cat.sort()
        n = cat.shape[0]
        lo_idx = max(0, min(n - 1, int(round((self.lower / 100.0) * (n - 1)))))
        hi_idx = max(0, min(n - 1, int(round((self.upper / 100.0) * (n - 1)))))
        return float(cat[lo_idx]), float(cat[hi_idx])


class KLDivergenceObserver:
    """TensorRT-style symmetric calibration via KL divergence."""

    NUM_BINS = 2048
    TARGET_BINS = 128

    def __init__(self) -> None:
        self.absmax = 0.0
        self._samples: list[np.ndarray] = []
        self.has_data = False

    def observe(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._samples.append(x.reshape(-1).astype(np.float32))
        am = float(np.max(np.abs(x)))
        if am > self.absmax:
            self.absmax = am
        self.has_data = True

    def range(self) -> tuple[float, float]:
        if not self.has_data or self.absmax <= 0.0:
            return 0.0, 0.0
        cat = np.abs(np.concatenate(self._samples))
        bins = self.NUM_BINS
        target = self.TARGET_BINS
        hist, _ = np.histogram(cat, bins=bins, range=(0.0, self.absmax))
        best_kl = math.inf
        best_T = target
        eps = 1e-4
        for T in range(target, bins + 1):
            P = hist[:T].astype(np.float64).copy()
            outliers = float(hist[T:].sum())
            P[-1] += outliers
            Q = np.zeros(T, dtype=np.float64)
            for k in range(target):
                start = int((k * T) / target)
                end = int(((k + 1) * T) / target)
                end = min(end, T)
                if end <= start:
                    continue
                seg = hist[start:end].astype(np.float64)
                nz = (P[start:end] > 0)
                if not nz.any():
                    continue
                avg = seg.sum() / float(nz.sum())
                Q[start:end] = np.where(nz, avg, 0.0)
            if not (P > 0.0).any() or not (Q > 0.0).any():
                continue
            P = np.where(P > 0.0, P, eps)
            Q = np.where(Q > 0.0, Q, eps)
            p_norm = P / P.sum()
            q_norm = Q / Q.sum()
            kl = float(np.sum(p_norm * np.log(p_norm / q_norm)))
            if kl < best_kl:
                best_kl = kl
                best_T = T
        threshold = self.absmax * (best_T / bins)
        return -threshold, threshold


# ----------------------------------------------------------------------------
# Layer descriptors (the subset the crate supports).
# ----------------------------------------------------------------------------


@dataclass
class Layer:
    """name: the tensor flowing out of this layer. observer: its range observer."""

    name: str
    observer: object = field(default_factory=MinMaxObserver)


@dataclass
class Dense(Layer):
    weight: np.ndarray = None       # [out, in]
    bias: Optional[np.ndarray] = None
    input_name: str = ""
    forward: Optional[Callable[[np.ndarray], np.ndarray]] = None
    weight_scale: float = 0.0
    q_weight: np.ndarray = None
    q_bias: np.ndarray = None


@dataclass
class ReLU(Layer):
    input_name: str = ""


@dataclass
class Sigmoid(Layer):
    input_name: str = ""


@dataclass
class Tanh(Layer):
    input_name: str = ""


@dataclass
class Softmax(Layer):
    input_name: str = ""
    axis: int = -1


@dataclass
class Conv2D(Layer):
    """tinymind's Conv2D descriptor (OHWI weights `[filters][kh][kw][in_channels]`,
    `forward` the float reference), with what the generated Rust also needs:
    `stride` (rows, columns), `padding` (top, bottom, left, right; padded with
    real 0, which is the input zero point) and `depthwise` (one filter per
    channel, weights `[channels][kh][kw][1]`). tinymind's importer leaves the
    layer types to hand-written C++, so its descriptor has none of these."""
    weight: np.ndarray = None
    bias: Optional[np.ndarray] = None
    input_name: str = ""
    forward: Optional[Callable[[np.ndarray], np.ndarray]] = None
    weight_scale: float = 0.0
    q_weight: np.ndarray = None
    q_bias: np.ndarray = None
    stride: tuple = (1, 1)
    padding: tuple = (0, 0, 0, 0)
    depthwise: bool = False


@dataclass
class MaxPool2D(Layer):
    """Max pooling over `kernel` windows at `stride` (default: the kernel), no
    padding. Its output keeps its input's scale and zero point."""
    input_name: str = ""
    kernel: tuple = (2, 2)
    stride: Optional[tuple] = None


@dataclass
class AvgPool2D(Layer):
    """Average pooling, as MaxPool2D; the integer mean is rounded."""
    input_name: str = ""
    kernel: tuple = (2, 2)
    stride: Optional[tuple] = None


@dataclass
class GlobalAvgPool2D(Layer):
    """The mean of each channel over the whole map: `[h][w][c]` to `[c]`."""
    input_name: str = ""


@dataclass
class Flatten(Layer):
    """A feature map to a vector, in NHWC order (`(row * W + column) * C + channel`).
    `from_torch` reorders the next Linear's columns from PyTorch's NCHW order."""
    input_name: str = ""


SUPPORTED = (Dense, ReLU, Sigmoid, Tanh, Softmax, Conv2D, MaxPool2D, AvgPool2D,
             GlobalAvgPool2D, Flatten)
PASS_THROUGH = (MaxPool2D, AvgPool2D, GlobalAvgPool2D, Flatten)


def _windows(x: np.ndarray, kernel: tuple, stride: tuple) -> np.ndarray:
    """`[oh][ow][kh][kw][c]` windows of an `[h][w][c]` map."""
    kh, kw = kernel
    sh, sw = stride
    h, w, _ = x.shape
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    rows = [[x[i * sh:i * sh + kh, j * sw:j * sw + kw, :] for j in range(ow)] for i in range(oh)]
    return np.array(rows, dtype=x.dtype)


def conv2d_nhwc(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray],
                stride: tuple = (1, 1), padding: tuple = (0, 0, 0, 0),
                depthwise: bool = False) -> np.ndarray:
    """Float reference of the Conv2D descriptor on an `[h][w][c]` map.

    Accumulates in float64 without BLAS (`optimize=False`), then rounds to
    float32: a float32 BLAS kernel's summation order depends on the CPU, and a
    last-bit difference in an observed range changes a scale, so the generated
    module would differ between machines. A float32 product is exact in
    float64."""
    pt, pb, pl, pr = padding
    xp = np.pad(x.astype(np.float64), ((pt, pb), (pl, pr), (0, 0)))
    _, kh, kw, _ = weight.shape
    win = _windows(xp, (kh, kw), stride)
    w64 = weight.astype(np.float64)
    if depthwise:
        y = np.einsum("ijklc,ckl->ijc", win, w64[:, :, :, 0], optimize=False)
    else:
        y = np.einsum("ijklc,fklc->ijf", win, w64, optimize=False)
    if bias is not None:
        y = y + bias.astype(np.float64)
    return y.astype(np.float32)


def dense_ref(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """Float reference of a Dense layer, `x @ weight.T + bias`, accumulated in
    float64 without BLAS and rounded to float32 (see conv2d_nhwc)."""
    y = np.einsum("i,oi->o", x.astype(np.float64), weight.astype(np.float64), optimize=False)
    if bias is not None:
        y = y + bias.astype(np.float64)
    return y.astype(np.float32)


def pool2d_nhwc(x: np.ndarray, layer) -> np.ndarray:
    """Float reference of MaxPool2D / AvgPool2D."""
    stride = layer.stride or layer.kernel
    win = _windows(x, layer.kernel, stride)
    if isinstance(layer, MaxPool2D):
        return win.max(axis=(2, 3))
    # In float64, as conv2d_nhwc, so the result does not depend on the CPU.
    return win.astype(np.float64).mean(axis=(2, 3)).astype(np.float32)


# ----------------------------------------------------------------------------
# Calibration and weight quantization (tinymind's, for the supported layers).
# ----------------------------------------------------------------------------


def _float_step(layer: Layer, parent: np.ndarray) -> np.ndarray:
    """One layer of the float reference model."""
    if isinstance(layer, (Dense, Conv2D)):
        return layer.forward(parent)
    if isinstance(layer, (MaxPool2D, AvgPool2D)):
        return pool2d_nhwc(parent, layer)
    if isinstance(layer, GlobalAvgPool2D):
        return parent.astype(np.float64).mean(axis=(0, 1)).astype(np.float32)
    if isinstance(layer, Flatten):
        return parent.reshape(-1)
    if isinstance(layer, ReLU):
        return np.maximum(parent, 0.0)
    if isinstance(layer, Sigmoid):
        return 1.0 / (1.0 + np.exp(-parent))
    if isinstance(layer, Tanh):
        return np.tanh(parent)
    if isinstance(layer, Softmax):
        shifted = parent - np.max(parent, axis=layer.axis, keepdims=True)
        ex = np.exp(shifted)
        return ex / np.sum(ex, axis=layer.axis, keepdims=True)
    raise NotImplementedError(type(layer).__name__)


def run_float(layers: Sequence[Layer], x: np.ndarray) -> np.ndarray:
    """The float reference model on one input (NHWC for feature maps): what
    calibration observes, layer by layer. Returns the chain's output."""
    y = x.astype(np.float32)
    for layer in layers:
        y = _float_step(layer, y)
    return y


def calibrate(layers: Sequence[Layer], input_name: str, input_observer: object,
              dataset: Iterable[np.ndarray]) -> dict[str, tuple[float, int]]:
    """Stream `dataset` through the float model, observe ranges. Returns
    tensor_name -> (scale, zero_point)."""
    for x in dataset:
        x_f = x.astype(np.float32)
        input_observer.observe(x_f)
        per_sample: dict[str, np.ndarray] = {input_name: x_f}
        for layer in layers:
            y = _float_step(layer, per_sample[layer.input_name])
            if not isinstance(layer, PASS_THROUGH):
                layer.observer.observe(y)
            per_sample[layer.name] = y

    ranges: dict[str, tuple[float, int]] = {}
    ranges[input_name] = affine_asymmetric(*input_observer.range())
    for layer in layers:
        if isinstance(layer, Softmax):
            # TFLite-fixed: scale 1/256, zero_point -128.
            ranges[layer.name] = (1.0 / 256.0, -128)
            continue
        if isinstance(layer, Sigmoid):
            ranges[layer.name] = (1.0 / 256.0, -128)
            continue
        if isinstance(layer, Tanh):
            ranges[layer.name] = (1.0 / 128.0, 0)
            continue
        if isinstance(layer, PASS_THROUGH):
            # Pooling and flattening run in their input's grid (the int8
            # layers have no requantizer); tinymind's importer has no such
            # layers.
            ranges[layer.name] = ranges[layer.input_name]
            continue
        ranges[layer.name] = affine_asymmetric(*layer.observer.range())
    return ranges


def quantize_weights(layers: Sequence[Layer], ranges: dict[str, tuple[float, int]]) -> None:
    """Symmetric per-tensor weight quantization. Mutates each Dense / Conv2D."""
    for layer in layers:
        if isinstance(layer, (Dense, Conv2D)):
            layer.weight_scale = symmetric_weight_scale(layer.weight)
            layer.q_weight = quantize_symmetric_weights(layer.weight, layer.weight_scale)
            if layer.bias is not None:
                input_scale = ranges[layer.input_name][0]
                layer.q_bias = quantize_bias_int32(layer.bias, input_scale * layer.weight_scale)


# ----------------------------------------------------------------------------
# What the generated Rust computes, in integers.
# ----------------------------------------------------------------------------


def f32(x: float) -> float:
    """Round to float32, as a C++ `float` constant or a Rust `f32` stores it."""
    return float(np.float32(x))


def build_requantizer(input_scale: float, weight_scale: float, output_scale: float,
                      output_zero_point: int) -> tuple[int, int, int, int, int]:
    """tinymind buildRequantizer<int8_t>(..., -128, 127) on float32 scales:
    (multiplier, shift, zero_point, qmin, qmax)."""
    ratio = (f32(input_scale) * f32(weight_scale)) / f32(output_scale)
    m, s = quantize_multiplier(ratio)
    return m, s, int(output_zero_point), -128, 127


def build_softmax(input_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    """tinymind buildQSoftmaxFixedPointParams (TFLite Micro's int8 softmax
    parameters) on a float32 input scale: (input_multiplier, input_left_shift,
    diff_min)."""
    input_integer_bits = 5
    max_real_multiplier = float((1 << 31) - 1)
    real_multiplier = beta * f32(input_scale) * float(1 << (31 - input_integer_bits))
    if real_multiplier > max_real_multiplier:
        real_multiplier = max_real_multiplier
    multiplier, shift = quantize_multiplier(real_multiplier)
    left_shift = -shift
    max_input_rescaled = (float((1 << input_integer_bits) - 1)
                          * float(1 << (31 - input_integer_bits))
                          / math.ldexp(1.0, left_shift))
    return multiplier, left_shift, -int(math.floor(max_input_rescaled))


def _libm_function(name: str):
    path = ctypes.util.find_library("m")
    if path is None:
        return None
    try:
        fn = getattr(ctypes.CDLL(path), name)
    except (OSError, AttributeError):
        return None
    fn.restype = ctypes.c_float
    fn.argtypes = [ctypes.c_float]
    return fn


_EXPF = _libm_function("expf")
_TANHF = _libm_function("tanhf")
if _EXPF is None or _TANHF is None:
    print("tinymind_import: no C expf/tanhf found; activation tables use numpy's "
          "float32 math and may differ from tables built on the device in rare "
          "entries", file=sys.stderr)


def _expf(x: np.float32) -> np.float32:
    return np.float32(_EXPF(float(x))) if _EXPF else np.exp(np.float32(x))


def _tanhf(x: np.float32) -> np.float32:
    return np.float32(_TANHF(float(x))) if _TANHF else np.tanh(np.float32(x))


def build_lut(kind: str, input_scale: float, input_zero_point: int,
              output_scale: float, output_zero_point: int) -> list[int]:
    """tinymind buildQSigmoidLUT / buildQTanhLUT, step for step in float32."""
    one = np.float32(1.0)
    in_s, out_s = np.float32(input_scale), np.float32(output_scale)
    lut = []
    for idx in range(256):
        x_f = in_s * np.float32((idx - 128) - input_zero_point)
        if kind == "sigmoid":
            y_f = one / (one + _expf(-x_f))
        else:
            y_f = _tanhf(x_f)
        y_q = np.float32(output_zero_point) + y_f / out_s
        lut.append(max(-128, min(127, lround(float(y_q)))))
    return lut


# ----------------------------------------------------------------------------
# Rust emitter.
# ----------------------------------------------------------------------------


def _ident(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in name).upper()
    if not out or out[0].isdigit():
        out = "T_" + out
    return out


def _f32_literal(x: float) -> str:
    """The shortest decimal that round-trips the float32 value."""
    v = np.float32(x)
    s = np.format_float_positional(v, unique=True, trim="0")
    if s.endswith("."):
        s += "0"
    return s


def _rust_array(values: Sequence[int], per_line: int = 16, indent: str = "    ") -> str:
    vs = list(values)
    lines = [indent + ", ".join(str(v) for v in vs[i:i + per_line]) + ","
             for i in range(0, len(vs), per_line)]
    return "[\n" + "\n".join(lines) + "\n" + indent[:-4] + "]"


def _rust_nested(a: np.ndarray, indent: str = "") -> str:
    """A Rust nested-array literal of an integer array, in memory order."""
    if a.ndim == 1:
        return "[" + ", ".join(str(int(v)) for v in a.tolist()) + "]"
    return "[" + ", ".join(_rust_nested(sub) for sub in a) + "]"


def _rust_type(shape: tuple, elem: str = "i8") -> str:
    """`[[[i8; C]; W]; H]` for shape (H, W, C)."""
    t = elem
    for d in reversed(shape):
        t = f"[{t}; {d}]"
    return t


def _zeros(shape: tuple) -> str:
    """A zeroed Rust array expression of that shape."""
    t = "0i8"
    for d in reversed(shape):
        t = f"[{t}; {d}]"
    return t


def _plan(layers: Sequence[Layer], input_name: str,
          input_shape: Optional[tuple] = None) -> tuple[list[dict], tuple]:
    """Group the chain into steps and follow the tensor's shape through it:
    ("vec", n) for a vector, ("map", h, w, c) for an NHWC feature map. A ReLU
    joins the Dense or Conv2D before it. Returns (steps, input shape). Rejects
    what the crate cannot run."""
    if not layers:
        raise ValueError("the chain is empty")
    first = layers[0]
    if isinstance(first, Conv2D):
        if input_shape is None:
            raise ValueError("a model that starts with a Conv2D needs "
                             "input_shape=(height, width, channels)")
        shape = ("map",) + tuple(int(d) for d in input_shape)
    elif isinstance(first, Dense):
        shape = ("vec", first.weight.shape[1])
    else:
        raise ValueError("the chain must start with a Dense or Conv2D layer")
    in_shape = shape
    current = input_name
    steps: list[dict] = []
    i = 0
    while i < len(layers):
        layer = layers[i]
        if not isinstance(layer, SUPPORTED):
            raise NotImplementedError(
                f"{type(layer).__name__} is not supported by the tinymind crate yet "
                f"(supported: Dense, Conv2D, ReLU, Sigmoid, Tanh, Softmax, MaxPool2D, "
                f"AvgPool2D, GlobalAvgPool2D, Flatten)")
        if layer.input_name != current:
            raise ValueError(f"layer {layer.name!r} reads {layer.input_name!r}, but the "
                             f"chain is at {current!r}: only a single chain is supported")
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        relu = nxt if isinstance(nxt, ReLU) and nxt.input_name == layer.name else None
        step = {"layer": layer, "relu": None, "in": shape}
        if isinstance(layer, Dense):
            if shape[0] != "vec" or shape[1] != layer.weight.shape[1]:
                raise ValueError(f"Dense {layer.name!r} expects {layer.weight.shape[1]} inputs, "
                                 f"the chain carries {shape}")
            step.update(kind="dense", relu=relu, out=("vec", layer.weight.shape[0]))
        elif isinstance(layer, Conv2D):
            if shape[0] != "map":
                raise ValueError(f"Conv2D {layer.name!r} needs a feature map, the chain "
                                 f"carries {shape}")
            _, h, w, c = shape
            f, kh, kw, ci = layer.weight.shape
            if layer.depthwise:
                if ci != 1 or f != c:
                    raise ValueError(f"depthwise Conv2D {layer.name!r}: weights {layer.weight.shape} "
                                     f"for {c} channels")
                form = "depthwise"
            else:
                if ci != c:
                    raise ValueError(f"Conv2D {layer.name!r} expects {ci} channels, the chain "
                                     f"carries {c}")
                plain = (kh, kw) == (1, 1) and tuple(layer.stride) == (1, 1) \
                    and tuple(layer.padding) == (0, 0, 0, 0)
                form = "pointwise" if plain else "conv"
            sh, sw = layer.stride
            pt, pb, pl, pr = layer.padding
            if h + pt + pb < kh or w + pl + pr < kw or sh < 1 or sw < 1:
                raise ValueError(f"Conv2D {layer.name!r}: the kernel does not fit the padded input")
            oh, ow = (h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1
            step.update(kind="conv", form=form, relu=relu, out=("map", oh, ow, f))
        elif isinstance(layer, (MaxPool2D, AvgPool2D)):
            if shape[0] != "map":
                raise ValueError(f"{type(layer).__name__} {layer.name!r} needs a feature map")
            _, h, w, c = shape
            kh, kw = layer.kernel
            sh, sw = layer.stride or layer.kernel
            if h < kh or w < kw or sh < 1 or sw < 1:
                raise ValueError(f"{type(layer).__name__} {layer.name!r}: the window does not fit")
            step.update(kind="pool", out=("map", (h - kh) // sh + 1, (w - kw) // sw + 1, c))
        elif isinstance(layer, GlobalAvgPool2D):
            if shape[0] != "map":
                raise ValueError(f"GlobalAvgPool2D {layer.name!r} needs a feature map")
            step.update(kind="gap", out=("vec", shape[3]))
        elif isinstance(layer, Flatten):
            if shape[0] != "map":
                raise ValueError(f"Flatten {layer.name!r} needs a feature map")
            step.update(kind="flatten", out=("vec", shape[1] * shape[2] * shape[3]))
        elif isinstance(layer, (Sigmoid, Tanh)):
            if shape[0] != "vec":
                raise NotImplementedError(f"{type(layer).__name__} {layer.name!r}: only on "
                                          f"vectors (after a Dense layer)")
            step.update(kind="lut", lut="sigmoid" if isinstance(layer, Sigmoid) else "tanh",
                        out=shape)
        elif isinstance(layer, Softmax):
            if layer.axis not in (-1, 1):
                raise NotImplementedError(f"Softmax {layer.name!r}: only the last axis is supported")
            if shape[0] != "vec":
                raise NotImplementedError(f"Softmax {layer.name!r}: only on vectors")
            step.update(kind="softmax", out=shape)
        else:
            raise ValueError(f"ReLU {layer.name!r} must directly follow a Dense or Conv2D layer")
        steps.append(step)
        current = relu.name if relu is not None else layer.name
        shape = step["out"]
        i += 2 if relu is not None else 1
    return steps, in_shape


def emit_rust_module(path: str, model_name: str, layers: Sequence[Layer],
                     ranges: dict[str, tuple[float, int]], input_name: str,
                     input_shape: Optional[tuple] = None) -> None:
    """Write a Rust module for the `tinymind` crate: constants, layers and
    `forward`."""
    steps, in_shape = _plan(layers, input_name, input_shape)
    tensors: list[str] = [input_name]
    for st in steps:
        tensors.append(st["relu"].name if st["relu"] is not None else st["layer"].name)
    out_shape = steps[-1]["out"]
    output_name = tensors[-1]
    vector_model = in_shape[0] == "vec"

    out: list[str] = []
    out.append(f"//! `{model_name}`: generated by tinymind-rs tools/import_pytorch/tinymind_import.py.")
    out.append("//! Do not edit; regenerate from the model instead.")
    out.append("//!")
    out.append("//! Integer-only and `no_std`: int8 inputs in the `INPUT` grid, int8 outputs in the")
    out.append("//! `OUTPUT` grid (real value = `SCALE * (q - ZERO_POINT)`).")
    if not vector_model:
        out.append("//! Feature maps are NHWC without the batch axis: `[row][column][channel]`.")
    out.append("")
    out.append("// Not every program uses every constant.")
    out.append("#![allow(dead_code)]")
    out.append("")
    kinds = {st["kind"] for st in steps}
    uses = ["Requantizer"]
    if "dense" in kinds:
        uses.append("QDense")
    if "lut" in kinds:
        uses += ["LUT_SIZE", "apply_lut"]
    if any(st["kind"] == "dense" and st["relu"] is not None for st in steps):
        uses.append("qrelu")
    if "softmax" in kinds:
        uses.append("QSoftmax")
    forms = {st.get("form") for st in steps if st["kind"] == "conv"}
    if "conv" in forms:
        uses.append("QConv2D")
    if "depthwise" in forms:
        uses.append("QDepthwiseConv2D")
    if "pointwise" in forms:
        uses.append("QPointwiseConv2D")
    for st in steps:
        if st["kind"] == "pool":
            uses.append("QMaxPool2D" if isinstance(st["layer"], MaxPool2D) else "QAvgPool2D")
    if "gap" in kinds:
        uses.append("QGlobalAvgPool2D")
    out.append(f"use tinymind::int8::{{{', '.join(sorted(set(uses)))}}};")
    out.append("")
    if vector_model:
        out.append(f"/// Number of inputs.\npub const NUM_INPUTS: usize = {in_shape[1]};")
    else:
        h, w, c = in_shape[1:]
        plural = lambda n, word: f"{n} {word}" + ("" if n == 1 else "s")
        out.append(f"/// The input feature map: {plural(h, 'row')}, {plural(w, 'column')}, "
                   f"{plural(c, 'channel')}.")
        out.append(f"pub type Input = {_rust_type(in_shape[1:])};")
    if out_shape[0] == "vec":
        out.append(f"/// Number of outputs.\npub const NUM_OUTPUTS: usize = {out_shape[1]};")
        ret_type = "[i8; NUM_OUTPUTS]"
    else:
        out.append(f"/// The output feature map.\npub type Output = {_rust_type(out_shape[1:])};")
        ret_type = "Output"
    out.append("")
    for name in tensors:
        scale, zp = ranges[name]
        ident = _ident(name)
        out.append(f"/// Scale of the `{name}` tensor.")
        out.append(f"pub const {ident}_SCALE: f32 = {_f32_literal(scale)};")
        out.append(f"/// Zero point of the `{name}` tensor.")
        out.append(f"pub const {ident}_ZERO_POINT: i8 = {int(zp)};")
    # INPUT_* / OUTPUT_* aliases, unless the tensors already carry those names.
    for alias, name in (("INPUT", input_name), ("OUTPUT", output_name)):
        if _ident(name) != alias:
            out.append(f"/// Scale of the {alias.lower()} (`{name}`).")
            out.append(f"pub const {alias}_SCALE: f32 = {_ident(name)}_SCALE;")
            out.append(f"/// Zero point of the {alias.lower()} (`{name}`).")
            out.append(f"pub const {alias}_ZERO_POINT: i8 = {_ident(name)}_ZERO_POINT;")
    out.append("")

    def describe(shape: tuple) -> str:
        return f"{shape[1]}" if shape[0] == "vec" else "x".join(str(d) for d in shape[1:])

    body: list[str] = []
    current = input_name
    var = "input"
    for st in steps:
        kind, layer, extra = st["kind"], st["layer"], st["relu"]
        ident = _ident(layer.name)
        arg = var if var == "input" else f"&{var}"
        nvar = f"x{len(body)}"
        if kind == "dense":
            o, i_ = layer.weight.shape
            target = extra.name if extra is not None else layer.name
            t_scale, t_zp = ranges[target]
            in_scale, in_zp = ranges[current]
            m, s, zp, qmin, qmax = build_requantizer(in_scale, layer.weight_scale, t_scale, t_zp)
            rows = ",\n".join("    " + _rust_array(r.tolist(), indent="        ")
                              for r in layer.q_weight)
            out.append(f"/// `{layer.name}` int8 weights, `[output][input]` (weight scale "
                       f"{_f32_literal(layer.weight_scale)}).")
            out.append(f"pub static {ident}_WEIGHTS: [[i8; {i_}]; {o}] = [\n{rows},\n];")
            biases = "None"
            if layer.q_bias is not None:
                out.append(f"/// `{layer.name}` int32 biases (scale = input scale * weight scale).")
                out.append(f"pub static {ident}_BIASES: [i32; {o}] = {_rust_array(layer.q_bias.tolist(), 8)};")
                biases = f"Some(&{ident}_BIASES)"
            relu_note = f", then ReLU into `{extra.name}`" if extra is not None else ""
            out.append(f"/// `{layer.name}`: {i_} -> {o}, from `{current}`{relu_note}.")
            out.append(f"pub static {ident}: QDense<'static, {i_}, {o}> = QDense {{")
            out.append(f"    weights: &{ident}_WEIGHTS,")
            out.append(f"    biases: {biases},")
            out.append(f"    input_zero_point: {int(in_zp)},")
            out.append(f"    requantizer: Requantizer {{ multiplier: {m}, shift: {s}, zero_point: {zp}, qmin: {qmin}, qmax: {qmax} }},")
            out.append("};")
            out.append("")
            if extra is not None:
                body.append(f"    let {nvar} = {ident}.forward({arg}).map(|v| qrelu(v, {_ident(extra.name)}_ZERO_POINT));")
            else:
                body.append(f"    let {nvar} = {ident}.forward({arg});")
            current = target
        elif kind == "conv":
            target = extra.name if extra is not None else layer.name
            t_scale, t_zp = ranges[target]
            in_scale, in_zp = ranges[current]
            m, s, zp, _, qmax = build_requantizer(in_scale, layer.weight_scale, t_scale, t_zp)
            # A following ReLU clamps at the output zero point: the same as
            # qrelu after the requantizer, folded into its lower bound.
            qmin = int(t_zp) if extra is not None else -128
            rq = (f"Requantizer {{ multiplier: {m}, shift: {s}, zero_point: {zp}, "
                  f"qmin: {qmin}, qmax: {qmax} }}")
            f, kh, kw, ci = layer.weight.shape
            _, h, w, c = st["in"]
            sh, sw = layer.stride
            pt, pb, pl, pr = layer.padding
            form = st["form"]
            qw = layer.q_weight
            if form == "depthwise":
                wtype, wvals, layout = _rust_type((c, kh, kw)), qw[:, :, :, 0], "[channel][row][column]"
            elif form == "pointwise":
                wtype, wvals, layout = _rust_type((f, ci)), qw[:, 0, 0, :], "[filter][channel]"
            else:
                wtype, wvals, layout = _rust_type((f, kh, kw, ci)), qw, "[filter][row][column][channel]"
            out.append(f"/// `{layer.name}` int8 weights, `{layout}` (weight scale "
                       f"{_f32_literal(layer.weight_scale)}).")
            out.append(f"pub static {ident}_WEIGHTS: {wtype} = {_rust_nested(wvals)};")
            biases = "None"
            if layer.q_bias is not None:
                out.append(f"/// `{layer.name}` int32 biases (scale = input scale * weight scale).")
                out.append(f"pub static {ident}_BIASES: [i32; {f}] = {_rust_array(layer.q_bias.tolist(), 8)};")
                biases = f"Some(&{ident}_BIASES)"
            if form == "depthwise":
                out.append(f"/// `{layer.name}` requantizers, one per channel (the weights share one scale).")
                out.append(f"pub static {ident}_REQUANTIZERS: [Requantizer; {c}] = [{rq}; {c}];")
            relu_note = f", then ReLU into `{extra.name}`" if extra is not None else ""
            what = {"conv": f"{kh}x{kw} convolution, {f} filters, stride {sh}x{sw}, "
                            f"padding {pt}/{pb}/{pl}/{pr} (top/bottom/left/right)",
                    "depthwise": f"{kh}x{kw} depthwise convolution, stride {sh}x{sw}, "
                                 f"padding {pt}/{pb}/{pl}/{pr} (top/bottom/left/right)",
                    "pointwise": f"1x1 (pointwise) convolution, {ci} -> {f} channels"}[form]
            out.append(f"/// `{layer.name}`: {what}, from `{current}` ({describe(st['in'])}) to "
                       f"{describe(st['out'])}{relu_note}.")
            if form == "conv":
                out.append(f"pub static {ident}: QConv2D<'static, {ci}, {kh}, {kw}, {f}, {sh}, {sw}, "
                           f"{pt}, {pb}, {pl}, {pr}> = QConv2D {{")
                out.append(f"    weights: &{ident}_WEIGHTS,")
                out.append(f"    biases: {biases},")
                out.append(f"    input_zero_point: {int(in_zp)},")
                out.append(f"    requantizer: {rq},")
                out.append("};")
            elif form == "pointwise":
                out.append(f"pub static {ident}: QPointwiseConv2D<'static, {ci}, {f}> = QPointwiseConv2D {{")
                out.append(f"    weights: &{ident}_WEIGHTS,")
                out.append(f"    biases: {biases},")
                out.append(f"    input_zero_point: {int(in_zp)},")
                out.append(f"    requantizer: {rq},")
                out.append("};")
            else:
                out.append(f"pub static {ident}: QDepthwiseConv2D<'static, {c}, {kh}, {kw}, {sh}, {sw}, "
                           f"{pt}, {pb}, {pl}, {pr}> = QDepthwiseConv2D {{")
                out.append(f"    weights: &{ident}_WEIGHTS,")
                out.append(f"    biases: {biases},")
                out.append(f"    input_zero_point: {int(in_zp)},")
                out.append(f"    requantizers: &{ident}_REQUANTIZERS,")
                out.append("};")
            out.append("")
            body.append(f"    let mut {nvar} = {_zeros(st['out'][1:])};")
            body.append(f"    {ident}.forward({arg}, &mut {nvar});")
            current = target
        elif kind == "pool":
            kh, kw = layer.kernel
            sh, sw = layer.stride or layer.kernel
            ty = "QMaxPool2D" if isinstance(layer, MaxPool2D) else "QAvgPool2D"
            init = "QMaxPool2D" if ty == "QMaxPool2D" else "QAvgPool2D { qmin: -128, qmax: 127 }"
            verb = "max" if ty == "QMaxPool2D" else "average"
            out.append(f"/// `{layer.name}`: {kh}x{kw} {verb} pooling, stride {sh}x{sw}, from `{current}` "
                       f"({describe(st['in'])}) to {describe(st['out'])}, in the same grid.")
            out.append(f"pub static {ident}: {ty}<{kh}, {kw}, {sh}, {sw}> = {init};")
            out.append("")
            body.append(f"    let mut {nvar} = {_zeros(st['out'][1:])};")
            body.append(f"    {ident}.forward({arg}, &mut {nvar});")
            current = layer.name
        elif kind == "gap":
            out.append(f"/// `{layer.name}`: global average pooling of `{current}` ({describe(st['in'])}), "
                       f"in the same grid.")
            out.append(f"pub static {ident}: QGlobalAvgPool2D = QGlobalAvgPool2D {{ qmin: -128, qmax: 127 }};")
            out.append("")
            body.append(f"    let {nvar} = {ident}.forward({arg});")
            current = layer.name
        elif kind == "flatten":
            body.append(f"    // `{layer.name}`: flatten `{current}` ({describe(st['in'])}) in NHWC order.")
            body.append(f"    let mut {nvar} = [0i8; {st['out'][1]}];")
            body.append(f"    for (out, &v) in {nvar}.iter_mut().zip({var}.iter().flatten().flatten()) {{")
            body.append("        *out = v;")
            body.append("    }")
            current = layer.name
        elif kind == "softmax":
            in_scale, _ = ranges[current]
            m, ls, dmin = build_softmax(in_scale)
            out.append(f"/// `{layer.name}`: softmax over the `{current}` logits, onto the `{layer.name}` grid "
                       f"(scale 1/256, zero point -128).")
            out.append(f"pub static {ident}: QSoftmax = QSoftmax {{ input_multiplier: {m}, "
                       f"input_left_shift: {ls}, diff_min: {dmin} }};")
            out.append("")
            body.append(f"    let {nvar} = {ident}.forward(&{var});")
            current = layer.name
        else:
            in_scale, in_zp = ranges[current]
            o_scale, o_zp = ranges[layer.name]
            lut = build_lut(st["lut"], in_scale, in_zp, o_scale, o_zp)
            out.append(f"/// `{layer.name}`: {st['lut']} table from the `{current}` grid to the `{layer.name}` grid.")
            out.append(f"pub static {ident}_LUT: [i8; LUT_SIZE] = {_rust_array(lut)};")
            out.append("")
            body.append(f"    let {nvar} = {var}.map(|v| apply_lut(v, &{ident}_LUT));")
            current = layer.name
        var = nvar

    # The last step is the function's value, not a `let` (clippy::let_and_return).
    last = body[-1]
    prefix = f"    let {var} = "
    if last.startswith(prefix) and last.endswith(";"):
        body[-1] = "    " + last[len(prefix):-1]
    else:
        body.append(f"    {var}")
    out.append("/// Run the network on int8 inputs in the `INPUT` grid.")
    in_type = "[i8; NUM_INPUTS]" if vector_model else "Input"
    out.append(f"pub fn forward(input: &{in_type}) -> {ret_type} {{")
    out.extend(body)
    out.append("}")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    # Format as `cargo fmt` would, so the file is stable in a formatted crate.
    rustfmt = shutil.which("rustfmt")
    if rustfmt:
        subprocess.run([rustfmt, "--edition", "2024", path], check=True)
    else:
        print("tinymind_import: rustfmt not found; the module is valid but unformatted",
              file=sys.stderr)


# ----------------------------------------------------------------------------
# Top level.
# ----------------------------------------------------------------------------


def import_pytorch_model(layers: Sequence[Layer], input_name: str, input_observer: object,
                         dataset: Iterable[np.ndarray], output_path: str,
                         model_name: str,
                         input_shape: Optional[tuple] = None) -> dict[str, tuple[float, int]]:
    """Calibrate -> quantize -> write the Rust module. Returns the activation
    ranges (tensor name -> (scale, zero_point)). A model that starts with a
    Conv2D needs `input_shape=(height, width, channels)`; its calibration inputs
    are NHWC arrays of that shape."""
    _plan(layers, input_name, input_shape)  # reject unsupported models before calibrating
    ranges = calibrate(layers, input_name, input_observer, dataset)
    quantize_weights(layers, ranges)
    emit_rust_module(output_path, model_name, layers, ranges, input_name, input_shape)
    return ranges


def _pair(v) -> tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def from_torch(model, input_name: str = "input", observers: Optional[dict] = None,
               input_shape: Optional[tuple] = None) -> list[Layer]:
    """Describe a torch.nn.Sequential as layer descriptors (names: fc1, conv1,
    relu1, ...). `observers` maps a name to an observer; the default is
    MinMaxObserver.

    Vectors: Linear, ReLU, Sigmoid, Tanh, Softmax. Feature maps (pass
    `input_shape=(channels, height, width)`, PyTorch's order): Conv2d (groups 1,
    or depthwise with groups = in = out channels; no dilation; zero padding),
    MaxPool2d and AvgPool2d (no padding), AdaptiveAvgPool2d(1) and Flatten.
    PyTorch's NCHW tensors and OIHW weights become NHWC and OHWI, and a Linear
    after a Flatten has its columns reordered to match, so the descriptors and
    the generated module take NHWC inputs: calibrate with
    `x.permute(1, 2, 0).numpy()` per sample. BatchNorm2d is not supported yet."""
    import torch  # lazily: only this helper needs PyTorch

    observers = observers or {}
    layers: list[Layer] = []
    current = input_name
    counts: dict[str, int] = {}
    # ("vec", n) or ("map", h, w, c), followed through the chain.
    shape: tuple = ("vec", None)
    if input_shape is not None:
        c, h, w = (int(d) for d in input_shape)
        shape = ("map", h, w, c)
    column_order: Optional[np.ndarray] = None  # a Flatten's NCHW -> NHWC reorder

    def name_for(kind: str) -> str:
        counts[kind] = counts.get(kind, 0) + 1
        return f"{kind}{counts[kind]}"

    def observer(name: str):
        return observers.get(name, MinMaxObserver())

    for module in model:
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().cpu().numpy().astype(np.float32)
            b = (module.bias.detach().cpu().numpy().astype(np.float32)
                 if module.bias is not None else None)
            if column_order is not None:
                w = np.ascontiguousarray(w[:, column_order])
                column_order = None
            name = name_for("fc")
            layers.append(Dense(name=name, weight=w, bias=b, input_name=current,
                                forward=(lambda x, w=w, b=b: dense_ref(x, w, b)),
                                observer=observer(name)))
            shape = ("vec", w.shape[0])
        elif isinstance(module, torch.nn.Conv2d):
            if shape[0] != "map":
                raise ValueError("a Conv2d needs from_torch(..., input_shape=(channels, height, width))")
            if _pair(module.dilation) != (1, 1) or module.padding_mode != "zeros":
                raise NotImplementedError("Conv2d: only dilation 1 and zero padding are supported")
            _, h, w_, c = shape
            depthwise = module.groups != 1
            if depthwise and not (module.groups == module.in_channels == module.out_channels):
                raise NotImplementedError("Conv2d: groups must be 1, or equal the input and "
                                          "output channels (depthwise)")
            kh, kw = _pair(module.kernel_size)
            stride = _pair(module.stride)
            if isinstance(module.padding, str):
                if module.padding == "valid":
                    padding = (0, 0, 0, 0)
                else:  # "same", which PyTorch allows only at stride 1
                    th, tw = kh - 1, kw - 1
                    padding = (th // 2, th - th // 2, tw // 2, tw - tw // 2)
            else:
                ph, pw = _pair(module.padding)
                padding = (ph, ph, pw, pw)
            # OIHW -> OHWI; a depthwise [C][1][KH][KW] becomes [C][KH][KW][1].
            wt = module.weight.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
            wt = np.ascontiguousarray(wt)
            b = (module.bias.detach().cpu().numpy().astype(np.float32)
                 if module.bias is not None else None)
            name = name_for("conv")
            layers.append(Conv2D(
                name=name, weight=wt, bias=b, input_name=current, stride=stride,
                padding=padding, depthwise=depthwise,
                forward=(lambda x, wt=wt, b=b, st=stride, p=padding, d=depthwise:
                         conv2d_nhwc(x, wt, b, st, p, d)),
                observer=observer(name)))
            pt, pb, pl, pr = padding
            shape = ("map", (h + pt + pb - kh) // stride[0] + 1,
                     (w_ + pl + pr - kw) // stride[1] + 1, module.out_channels)
        elif isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            if shape[0] != "map":
                raise ValueError(f"{type(module).__name__} needs a feature map")
            if _pair(module.padding) != (0, 0) or module.ceil_mode:
                raise NotImplementedError(f"{type(module).__name__}: only no padding and "
                                          f"ceil_mode=False are supported")
            if isinstance(module, torch.nn.MaxPool2d) and _pair(module.dilation) != (1, 1):
                raise NotImplementedError("MaxPool2d: only dilation 1 is supported")
            kernel = _pair(module.kernel_size)
            stride = _pair(module.stride if module.stride is not None else module.kernel_size)
            is_max = isinstance(module, torch.nn.MaxPool2d)
            name = name_for("maxpool" if is_max else "avgpool")
            cls = MaxPool2D if is_max else AvgPool2D
            layers.append(cls(name=name, input_name=current, kernel=kernel, stride=stride))
            _, h, w_, c = shape
            shape = ("map", (h - kernel[0]) // stride[0] + 1, (w_ - kernel[1]) // stride[1] + 1, c)
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            if _pair(module.output_size) != (1, 1) or shape[0] != "map":
                raise NotImplementedError("AdaptiveAvgPool2d: only output size 1 (global "
                                          "average pooling) on a feature map is supported")
            name = name_for("gap")
            layers.append(GlobalAvgPool2D(name=name, input_name=current))
            shape = ("vec", shape[3])
        elif isinstance(module, torch.nn.Flatten):
            if shape[0] == "vec":
                continue  # already a vector, e.g. after AdaptiveAvgPool2d(1)
            _, h, w_, c = shape
            # PyTorch flattens NCHW: column (c * H + y) * W + x. The NHWC
            # vector's element (y * W + x) * C + c reads that column.
            column_order = np.arange(c * h * w_).reshape(c, h, w_).transpose(1, 2, 0).reshape(-1)
            name = name_for("flatten")
            layers.append(Flatten(name=name, input_name=current))
            shape = ("vec", c * h * w_)
        elif isinstance(module, torch.nn.ReLU):
            name = name_for("relu")
            layers.append(ReLU(name=name, input_name=current, observer=observer(name)))
        elif isinstance(module, torch.nn.Sigmoid):
            name = name_for("sigmoid")
            layers.append(Sigmoid(name=name, input_name=current))
        elif isinstance(module, torch.nn.Tanh):
            name = name_for("tanh")
            layers.append(Tanh(name=name, input_name=current))
        elif isinstance(module, torch.nn.Softmax):
            # The chain carries one sample at a time, so dim 1 of a batch is its
            # last axis.
            if module.dim not in (-1, 1):
                raise NotImplementedError(f"Softmax over dim {module.dim} is not supported")
            name = name_for("softmax")
            layers.append(Softmax(name=name, input_name=current))
        elif isinstance(module, torch.nn.BatchNorm2d):
            raise NotImplementedError("BatchNorm2d is not supported yet: fold it into the "
                                      "preceding Conv2d before importing")
        else:
            raise NotImplementedError(f"{type(module).__name__} is not supported yet")
        current = layers[-1].name
    return layers
