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

Supported layers (what the crate can run today): Dense, ReLU, Sigmoid, Tanh and
Softmax (over the last axis), in a single chain. Observers:
MinMaxObserver, PercentileObserver, KLDivergenceObserver (per tensor).

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
    return int(q), -int(exponent)


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


SUPPORTED = (Dense, ReLU, Sigmoid, Tanh, Softmax)


# ----------------------------------------------------------------------------
# Calibration and weight quantization (tinymind's, for the supported layers).
# ----------------------------------------------------------------------------


def calibrate(layers: Sequence[Layer], input_name: str, input_observer: object,
              dataset: Iterable[np.ndarray]) -> dict[str, tuple[float, int]]:
    """Stream `dataset` through the float model, observe ranges. Returns
    tensor_name -> (scale, zero_point)."""
    for x in dataset:
        x_f = x.astype(np.float32)
        input_observer.observe(x_f)
        per_sample: dict[str, np.ndarray] = {input_name: x_f}
        for layer in layers:
            parent = per_sample[layer.input_name]
            if isinstance(layer, Dense):
                y = layer.forward(parent)
            elif isinstance(layer, ReLU):
                y = np.maximum(parent, 0.0)
            elif isinstance(layer, Sigmoid):
                y = 1.0 / (1.0 + np.exp(-parent))
            elif isinstance(layer, Tanh):
                y = np.tanh(parent)
            elif isinstance(layer, Softmax):
                shifted = parent - np.max(parent, axis=layer.axis, keepdims=True)
                ex = np.exp(shifted)
                y = ex / np.sum(ex, axis=layer.axis, keepdims=True)
            else:
                raise NotImplementedError(type(layer).__name__)
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
        ranges[layer.name] = affine_asymmetric(*layer.observer.range())
    return ranges


def quantize_weights(layers: Sequence[Layer], ranges: dict[str, tuple[float, int]]) -> None:
    """Symmetric per-tensor weight quantization. Mutates each Dense."""
    for layer in layers:
        if isinstance(layer, Dense):
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


def _plan(layers: Sequence[Layer], input_name: str) -> list[tuple]:
    """Group the chain into steps: ("dense", layer, relu_or_None),
    ("lut", activation_layer, kind) and ("softmax", layer, None). Rejects what
    the crate cannot run."""
    current = input_name
    steps: list[tuple] = []
    i = 0
    while i < len(layers):
        layer = layers[i]
        if not isinstance(layer, SUPPORTED):
            raise NotImplementedError(
                f"{type(layer).__name__} is not supported by the tinymind crate yet "
                f"(supported: Dense, ReLU, Sigmoid, Tanh, Softmax)")
        if layer.input_name != current:
            raise ValueError(f"layer {layer.name!r} reads {layer.input_name!r}, but the "
                             f"chain is at {current!r}: only a single chain is supported")
        if isinstance(layer, Dense):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if isinstance(nxt, ReLU) and nxt.input_name == layer.name:
                steps.append(("dense", layer, nxt))
                current = nxt.name
                i += 2
                continue
            steps.append(("dense", layer, None))
            current = layer.name
        elif isinstance(layer, (Sigmoid, Tanh)):
            steps.append(("lut", layer, "sigmoid" if isinstance(layer, Sigmoid) else "tanh"))
            current = layer.name
        elif isinstance(layer, Softmax):
            if layer.axis not in (-1, 1):
                raise NotImplementedError(f"Softmax {layer.name!r}: only the last axis is supported")
            steps.append(("softmax", layer, None))
            current = layer.name
        else:
            raise ValueError(f"ReLU {layer.name!r} must directly follow a Dense layer")
        i += 1
    if not steps or steps[0][0] != "dense":
        raise ValueError("the chain must start with a Dense layer")
    return steps


def emit_rust_module(path: str, model_name: str, layers: Sequence[Layer],
                     ranges: dict[str, tuple[float, int]], input_name: str) -> None:
    """Write a Rust module for the `tinymind` crate: constants, layers and
    `forward`."""
    steps = _plan(layers, input_name)
    n_in = steps[0][1].weight.shape[1]
    tensors: list[str] = [input_name]
    width = n_in
    for kind, layer, extra in steps:
        if kind == "dense":
            width = layer.weight.shape[0]
            tensors.append(extra.name if extra is not None else layer.name)
        else:
            tensors.append(layer.name)
    n_out = width
    output_name = tensors[-1]

    out: list[str] = []
    out.append(f"//! `{model_name}`: generated by tinymind-rs tools/import_pytorch/tinymind_import.py.")
    out.append("//! Do not edit; regenerate from the model instead.")
    out.append("//!")
    out.append("//! Integer-only and `no_std`: int8 inputs in the `INPUT` grid, int8 outputs in the")
    out.append("//! `OUTPUT` grid (real value = `SCALE * (q - ZERO_POINT)`).")
    out.append("")
    out.append("// Not every program uses every constant.")
    out.append("#![allow(dead_code)]")
    out.append("")
    uses = ["QDense", "Requantizer"]
    if any(s[0] == "lut" for s in steps):
        uses += ["LUT_SIZE", "apply_lut"]
    if any(s[0] == "dense" and s[2] is not None for s in steps):
        uses.append("qrelu")
    if any(s[0] == "softmax" for s in steps):
        uses.append("QSoftmax")
    out.append(f"use tinymind::int8::{{{', '.join(sorted(uses))}}};")
    out.append("")
    out.append(f"/// Number of inputs.\npub const NUM_INPUTS: usize = {n_in};")
    out.append(f"/// Number of outputs.\npub const NUM_OUTPUTS: usize = {n_out};")
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

    body: list[str] = []
    current = input_name
    width = n_in
    var = "input"
    for kind, layer, extra in steps:
        if kind == "dense":
            ident = _ident(layer.name)
            o, i_ = layer.weight.shape
            assert i_ == width, f"{layer.name}: expects {i_} inputs, chain has {width}"
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
            nvar = f"x{len(body)}"
            if extra is not None:
                body.append(f"    let {nvar} = {ident}.forward({'&' if var != 'input' else ''}{var}).map(|v| qrelu(v, {_ident(extra.name)}_ZERO_POINT));")
            else:
                body.append(f"    let {nvar} = {ident}.forward({'&' if var != 'input' else ''}{var});")
            var = nvar
            current = target
            width = o
        elif kind == "softmax":
            ident = _ident(layer.name)
            in_scale, _ = ranges[current]
            m, ls, dmin = build_softmax(in_scale)
            out.append(f"/// `{layer.name}`: softmax over the `{current}` logits, onto the `{layer.name}` grid "
                       f"(scale 1/256, zero point -128).")
            out.append(f"pub static {ident}: QSoftmax = QSoftmax {{ input_multiplier: {m}, "
                       f"input_left_shift: {ls}, diff_min: {dmin} }};")
            out.append("")
            nvar = f"x{len(body)}"
            body.append(f"    let {nvar} = {ident}.forward(&{var});")
            var = nvar
            current = layer.name
        else:
            ident = _ident(layer.name)
            in_scale, in_zp = ranges[current]
            o_scale, o_zp = ranges[layer.name]
            lut = build_lut(extra, in_scale, in_zp, o_scale, o_zp)
            out.append(f"/// `{layer.name}`: {extra} table from the `{current}` grid to the `{layer.name}` grid.")
            out.append(f"pub static {ident}_LUT: [i8; LUT_SIZE] = {_rust_array(lut)};")
            out.append("")
            nvar = f"x{len(body)}"
            body.append(f"    let {nvar} = {var}.map(|v| apply_lut(v, &{ident}_LUT));")
            var = nvar
            current = layer.name

    # The last step is the function's value, not a `let` (clippy::let_and_return).
    last = body[-1]
    prefix = f"    let {var} = "
    assert last.startswith(prefix) and last.endswith(";")
    body[-1] = "    " + last[len(prefix):-1]
    out.append("/// Run the network on int8 inputs in the `INPUT` grid.")
    out.append("pub fn forward(input: &[i8; NUM_INPUTS]) -> [i8; NUM_OUTPUTS] {")
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
                         model_name: str) -> dict[str, tuple[float, int]]:
    """Calibrate -> quantize -> write the Rust module. Returns the activation
    ranges (tensor name -> (scale, zero_point))."""
    _plan(layers, input_name)  # reject unsupported models before calibrating
    ranges = calibrate(layers, input_name, input_observer, dataset)
    quantize_weights(layers, ranges)
    emit_rust_module(output_path, model_name, layers, ranges, input_name)
    return ranges


def from_torch(model, input_name: str = "input", observers: Optional[dict] = None) -> list[Layer]:
    """Describe a torch.nn.Sequential of Linear / ReLU / Sigmoid / Tanh /
    Softmax as layer descriptors (names: fc1, relu1, ...). `observers` maps a name to an
    observer; the default is MinMaxObserver."""
    import torch  # lazily: only this helper needs PyTorch

    observers = observers or {}
    layers: list[Layer] = []
    current = input_name
    counts: dict[str, int] = {}

    def name_for(kind: str) -> str:
        counts[kind] = counts.get(kind, 0) + 1
        return f"{kind}{counts[kind]}"

    for module in model:
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().cpu().numpy().astype(np.float32)
            b = (module.bias.detach().cpu().numpy().astype(np.float32)
                 if module.bias is not None else None)
            name = name_for("fc")
            layers.append(Dense(name=name, weight=w, bias=b, input_name=current,
                                forward=(lambda x, w=w, b=b: x @ w.T + (b if b is not None else 0.0)),
                                observer=observers.get(name, MinMaxObserver())))
        elif isinstance(module, torch.nn.ReLU):
            name = name_for("relu")
            layers.append(ReLU(name=name, input_name=current,
                               observer=observers.get(name, MinMaxObserver())))
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
        else:
            raise NotImplementedError(f"{type(module).__name__} is not supported yet")
        current = layers[-1].name
    return layers
