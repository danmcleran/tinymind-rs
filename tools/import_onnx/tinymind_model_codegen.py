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

tinymind (C++)'s ONNX code generator helpers, for this importer's int4 and
16x8 back ends.

Everything from `def _f32` to the end of `pack_int4` is copied verbatim from
tinymind (C++)'s apps/import_onnx/tinymind_model_codegen.py (scripts/
compare-with-cpp.sh extracts that range from both files and compares them):
the int4 requantization of a record's weights (requantize_int4, pack_int4),
the move of a record to int16 activations (int16_grid, requantize_16x8), the
int16 tables (activation_lut16) and the int16 softmax and add parameters
(softmax16_params, qadd_params). tinymind (C++)'s generator writes a C++
model.hpp from these; tinymind_import_onnx.py writes Rust from the same
numbers. It names this importer `ti`, as tinymind (C++)'s does its own:
lround and quantize_multiplier are the parser's, verbatim in both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import tinymind_import_onnx as ti

def _f32(x: float) -> float:
    """Round a double to the nearest float, as a C++ float variable holds it."""
    import numpy as np
    return float(np.float32(x))


def qadd_params(a_scale: float, b_scale: float, out_scale: float, left_shift: int = 20) -> dict:
    """Mirror qcalibration.hpp::buildQAddParams (float scales in)."""
    a, b, o = _f32(a_scale), _f32(b_scale), _f32(out_scale)
    twice_max = 2.0 * (a if a > b else b)
    real_out = twice_max / (float(1 << left_shift) * o)
    am, ash = ti.quantize_multiplier(a / twice_max)
    bm, bsh = ti.quantize_multiplier(b / twice_max)
    om, osh = ti.quantize_multiplier(real_out)
    return dict(left_shift=left_shift, input_a_multiplier=am, input_a_shift=ash,
                input_b_multiplier=bm, input_b_shift=bsh, output_multiplier=om, output_shift=osh)


def softmax_params(input_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    """Mirror qcalibration.hpp::buildQSoftmaxFixedPointParams for a float
    input scale; returns (input_multiplier, input_left_shift, diff_min)."""
    integer_bits = 5
    real = beta * _f32(input_scale) * float(1 << (31 - integer_bits))
    real = min(real, float((1 << 31) - 1))
    mult, shift = ti.quantize_multiplier(real)
    left = -shift
    max_rescaled = float((1 << integer_bits) - 1) * float(1 << (31 - integer_bits)) / float(1 << left)
    return mult, left, -int(math.floor(max_rescaled))


def softmax16_params(input_scale: float, beta: float = 1.0) -> tuple[int, int]:
    """Mirror qsoftmax_fixedpoint.hpp::buildQSoftmaxInt16Params: the int16
    softmax's (input_multiplier, input_shift) for an int16 input of scale
    input_scale, as TFLite computes them (input_scale * beta / (10 / 65535)).
    input_shift has tinymind's sign (negative shifts left)."""
    return ti.quantize_multiplier(input_scale * beta / (10.0 / 65535.0))


SOFTMAX16_OUTPUT_SCALE = 1.0 / 32768.0


def activation_lut(kind: str, in_scale: float, in_zp: int, out_scale: float, out_zp: int) -> list:
    """Mirror qactivations.hpp::buildQSigmoidLUT / buildQTanhLUT: entry idx
    holds the output for the int8 input idx - 128. The builders compute in
    float; the products and quotients are rounded to float here in the same
    order, and exp / tanh are taken in double and rounded to float."""
    import numpy as np
    f = np.float32
    s_in, s_out, zp_out = f(in_scale), f(out_scale), f(out_zp)
    lut = []
    for idx in range(256):
        x = f(s_in * f(idx - 128 - in_zp))
        if kind == "Sigmoid":
            e = f(math.exp(-float(x)))
            y = f(f(1.0) / f(f(1.0) + e))
        elif kind == "Tanh":
            y = f(math.tanh(float(x)))
        else:
            raise ValueError(kind)
        q = f(zp_out + f(y / s_out))
        lut.append(max(-128, min(127, ti.lround(float(q)))))
    return lut


def round_half_away(x: float) -> int:
    """std::round semantics on a double (ties away from zero), exactly."""
    if x < 0.0:
        return -round_half_away(-x)
    r = math.floor(x)
    return int(r) + 1 if x - r >= 0.5 else int(r)


def activation_lut16(kind: str, in_scale: float, in_zp: int, out_scale: float) -> list:
    """Mirror qactivations.hpp::buildQLUT16 (buildQSigmoidLUT16 /
    buildQTanhLUT16): 513 int16 entries, entry i the function at int16 input
    -32768 + 128 i less half the interval's midpoint interpolation error,
    on the symmetric output grid of out_scale. All in double, as there."""
    if kind == "Sigmoid":
        def fn(x):
            return 1.0 / (1.0 + math.exp(-x))
    elif kind == "Tanh":
        fn = math.tanh
    else:
        raise ValueError(kind)
    inv = 1.0 / out_scale
    lut = []
    for i in range(513):
        x = in_scale * float(-32768 + 128 * i - in_zp)
        sample = float(round_half_away(fn(x) * inv))
        v = sample
        if i < 512:
            x_next = in_scale * float(-32768 + 128 * (i + 1) - in_zp)
            mid_interp = float(round_half_away((fn(x_next) * inv + sample) / 2))
            mid = float(round_half_away(fn((x + x_next) / 2) * inv))
            v = sample - float(round_half_away((mid_interp - mid) / 2))
        lut.append(int(max(-32768.0, min(32767.0, v))))
    return lut


@dataclass
class Grid16:
    """An int8 activation grid widened to int16: symmetric (zero point 0)
    over the same real range, with the int8 range's ends as clamp bounds."""
    scale: float
    qmin: int
    qmax: int


def int16_grid(scale: float, zp: int) -> Grid16:
    """The int16 grid for an int8 tensor of (scale, zp). The int8 tensor
    holds reals in [lo, hi] = [(-128 - zp) * scale, (127 - zp) * scale]; the
    int16 grid is symmetric with scale max(|lo|, |hi|) / 32767, and a layer
    writing the tensor clamps to [lo, hi] on it. The clamp keeps whatever
    the int8 grid implied: a ReLU fused into the requantizer (zero point
    -128, so lo = 0) still clamps at zero."""
    lo = (-128 - zp) * scale
    hi = (127 - zp) * scale
    s16 = max(abs(lo), abs(hi)) / 32767.0
    return Grid16(scale=s16, qmin=max(-32768, round_half_away(lo / s16)),
                  qmax=min(32767, round_half_away(hi / s16)))


@dataclass
class Wide16x8:
    """A weight layer moved to int16 activations: int64 biases at the int16
    input scale, and one 64-bit requantizer per output channel."""
    biases: Optional[list]
    multipliers: list
    shifts: list
    qmin: int
    qmax: int


def requantize_16x8(layer) -> Wide16x8:
    """Move a weight record (Conv, DepthwiseConv, AveragePool, Dense) to
    int16 activations, keeping its int8 weights: the bias b (at input_scale *
    weight_scale) becomes round(b * input_scale / s_in16) at s_in16 *
    weight_scale, and each requantizer's ratio becomes s_in16 *
    weight_scale / s_out16. Raises ValueError for a ratio of 2^7 or more,
    beyond the 64-bit Requantizer's (TFLite's) shift contract."""
    gin = int16_grid(layer.input_scale, layer.input_zp)
    gout = int16_grid(layer.output_scale, layer.output_zp)
    multipliers, shifts = [], []
    for ws in layer.weight_scales:
        m, sh = ti.quantize_multiplier(gin.scale * float(ws) / gout.scale)
        if sh < -7:
            raise ValueError(f"{layer.name}: scale ratio {gin.scale * float(ws) / gout.scale:g} is 2^7 or more, "
                             "beyond the int64 requantizer's range")
        multipliers.append(m)
        shifts.append(sh)
    biases = None
    if layer.bias is not None:
        ratio = layer.input_scale / gin.scale
        biases = [round_half_away(float(b) * ratio) for b in layer.bias.tolist()]
    return Wide16x8(biases=biases, multipliers=multipliers, shifts=shifts, qmin=gout.qmin, qmax=gout.qmax)


def float_reference(layers, x):
    """Run parsed records in float: dequantized int8 weights and biases,
    real activations, every layer's output clipped to the real range its
    int8 grid holds, and no other rounding. Both the int8 and the 16x8 model
    approximate exactly this, so their distance from it measures what their
    activation grids cost. x is the model input (reals, flattened NHWC);
    returns the output, flattened. A softmax's output is not clipped: its
    int8 and int16 grids hold different ranges, and only the model's output
    may be a softmax."""
    import numpy as np
    vals = {-1: np.asarray(x, dtype=np.float64)}
    for i, l in enumerate(layers):
        g = l.geometry
        src = vals[l.sources[0]]
        op = l.op_type
        if l.weight is not None:
            out_ch = l.weight.shape[0]
            ws = np.array(l.weight_scales, dtype=np.float64)
            ws = np.repeat(ws, out_ch) if ws.size == 1 else ws
            w = l.weight.astype(np.float64) * ws.reshape((-1,) + (1,) * (l.weight.ndim - 1))
            b = np.zeros(out_ch) if l.bias is None else l.bias.astype(np.float64) * l.input_scale * ws
        if op == "Dense":
            y = w @ src + b
        elif op in ("Conv", "DepthwiseConv", "AveragePool", "MaxPool"):
            H, W, C = g["InputHeight"], g["InputWidth"], g["InputChannels"]
            KH, KW, SH, SW = g["KernelHeight"], g["KernelWidth"], g["StrideH"], g["StrideW"]
            pads = ((g.get("PadTop", 0), g.get("PadBottom", 0)), (g.get("PadLeft", 0), g.get("PadRight", 0)), (0, 0))
            xp = np.pad(src.reshape(H, W, C), pads)
            OH, OW = g["OutputHeight"], g["OutputWidth"]
            y = np.zeros((OH, OW, g.get("NumFilters", C)))
            for oh in range(OH):
                for ow in range(OW):
                    patch = xp[oh * SH:oh * SH + KH, ow * SW:ow * SW + KW, :]
                    if op == "Conv":
                        y[oh, ow] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
                    elif op == "MaxPool":
                        y[oh, ow] = patch.max(axis=(0, 1))
                    else:
                        y[oh, ow] = np.einsum("hwc,chw->c", patch, w.reshape(C, KH, KW)) + b
        elif op == "Add":
            y = src + vals[l.sources[1]]
        elif op == "Relu":
            y = np.maximum(src, 0.0)
        elif op == "Sigmoid":
            y = 1.0 / (1.0 + np.exp(-src))
        elif op == "Tanh":
            y = np.tanh(src)
        elif op == "Softmax":
            e = np.exp(src - src.max())
            vals[i] = e / e.sum()
            continue
        else:
            raise ValueError(op)
        lo = (-128 - l.output_zp) * l.output_scale
        hi = (127 - l.output_zp) * l.output_scale
        vals[i] = np.clip(np.asarray(y, dtype=np.float64).reshape(-1), lo, hi)
    return vals[len(layers) - 1]


# ----------------------------------------------------------------------------
# Emitter.
# ----------------------------------------------------------------------------


SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128


def output_size(layer, sizes: dict) -> int:
    g = layer.geometry
    if layer.op_type in ("Conv", "DepthwiseConv", "AveragePool"):
        return g["OutputHeight"] * g["OutputWidth"] * g["NumFilters"]
    if layer.op_type == "MaxPool":
        return g["OutputHeight"] * g["OutputWidth"] * g["InputChannels"]
    if layer.op_type == "Dense":
        return g["NumOutputs"]
    return sizes[layer.sources[0]]


def input_size(layer) -> Optional[int]:
    g = layer.geometry
    if layer.op_type == "Dense":
        return g["NumInputs"]
    if "InputHeight" in g:
        return g["InputHeight"] * g["InputWidth"] * g["InputChannels"]
    return None


def _rq(m: int, s: int, zp: int) -> str:
    return f"{{{m}, {s}, {zp}, -128, 127}}"


@dataclass
class Int4Dense:
    """A weight layer's weights requantized to per-channel int4 (a dense
    layer's, or a convolution's with each filter flattened)."""
    codes: object          # int8 numpy array [out][in], values -7..7
    scales: list           # per output channel
    biases: object         # int32 numpy array, or None
    multipliers: list
    shifts: list


def requantize_int4(layer) -> Int4Dense:
    """Requantize a Dense, Conv or DepthwiseConv record's int8 weights to
    symmetric per-channel int4: the float weights w = q8 * s8 of each output
    channel (a Dense row, a Conv filter [KH][KW][C], a depthwise channel's
    [KH][KW]) get scale max |w| / 7 and codes round(w / s4) in -7..7 (ties
    away from zero, as quantizeInt4PerChannel). The bias moves to the new
    accumulator scale input_scale * s4, and each requantizer to
    input_scale * s4 / output_scale. codes keeps the record's weight shape."""
    import numpy as np
    w8 = layer.weight.astype(np.float64).reshape(layer.weight.shape[0], -1)
    out_ch = w8.shape[0]
    s8 = layer.weight_scales if layer.per_channel else list(layer.weight_scales) * out_ch
    codes = np.zeros(w8.shape, dtype=np.int8)
    scales, multipliers, shifts = [], [], []
    biases = None if layer.bias is None else np.zeros(out_ch, dtype=np.int64)
    for o in range(out_ch):
        wf = w8[o] * float(s8[o])
        largest = float(np.max(np.abs(wf)))
        s4 = largest / 7.0 if largest > 0.0 else float(s8[o])
        scales.append(s4)
        codes[o] = [max(-7, min(7, ti.lround(v / s4))) for v in wf.tolist()]
        if biases is not None:
            biases[o] = max(-(1 << 31), min((1 << 31) - 1, ti.lround(float(layer.bias[o]) * float(s8[o]) / s4)))
        m, sh = ti.quantize_multiplier(layer.input_scale * s4 / layer.output_scale)
        multipliers.append(m)
        shifts.append(sh)
    return Int4Dense(codes=codes.reshape(layer.weight.shape), scales=scales, biases=biases,
                     multipliers=multipliers, shifts=shifts)


# The name #299 shipped, for dense layers.
requantize_dense_int4 = requantize_int4


def pack_int4(values) -> list:
    """TFLite / ONNX int4 packing: element k in the low nibble of byte k / 2
    when k is even, the high nibble when odd (as packInt4)."""
    flat = [int(v) & 15 for v in values]
    out = []
    for b in range((len(flat) + 1) // 2):
        low = flat[2 * b]
        high = flat[2 * b + 1] if 2 * b + 1 < len(flat) else 0
        out.append(low | (high << 4))
    return out
