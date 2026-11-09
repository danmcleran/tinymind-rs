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

"""
TFLite int8 model -> tinymind (Rust) module.

A port of tinymind's benchmarks/framework_comparison_f411/gen/make_mlperf.py,
which translates an int8 .tflite graph into tinymind's C++ layers, with a Rust
back end. The arithmetic is tinymind's, unchanged: each layer reuses the model's
own int8 weights and int32 biases, and the requantizer multipliers are derived
from its float scales as TFLite Micro derives them (quantize_multiplier,
act_range, rq and the softmax parameters below are copied), so the generated
module computes the same integers as tinymind's translation and as TFLM.

Differences from tinymind's script:
  * the model is read with the `tflite` package (flatbuffer bindings of the
    TFLite schema, numpy only) instead of TensorFlow's interpreter;
  * the output is a `no_std` Rust module with a `forward` function. Feature
    maps live in typed buffers, one per shape and as many as are live at once
    (tinymind plans one byte arena instead; typed arrays of different shapes
    cannot share storage without `unsafe`); SAME padding goes into the layers'
    padding parameters, which have no band limit in the crate (tinymind adds a
    QPad2D pass for wide layers: same values).

Supported operators: CONV_2D, DEPTHWISE_CONV_2D (depth multiplier 1),
AVERAGE_POOL_2D (global only, as tinymind), FULLY_CONNECTED, SOFTMAX (one row),
RESHAPE and ADD (same shapes, no broadcasting); fused NONE / RELU / RELU6
activations; int8 activations and weights, per-tensor or per-channel. The graph
may branch (residual connections): each operator reads any tensor written
before it.
"""

import math
import os
import shutil
import subprocess
import sys

import numpy as np
import tflite


# ---------------------------------------------------------------- tinymind's
# helpers, unchanged (gen/make_mlperf.py)


def quantize_multiplier(m):
    """TFLite QuantizeMultiplier: m = q * 2^(shift - 31), shift > 0 = left."""
    if m == 0.0:
        return 0, 0
    frac, exp = math.frexp(m)
    q = int(round(frac * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exp += 1
    if exp < -31:
        return 0, 0
    return q, exp


def f64(x):
    return float(np.float64(np.float32(x)))


def act_range(act, scale, zp):
    """TFLite CalculateActivationRangeQuantized for int8."""
    qmin, qmax = -128, 127
    if act == tflite.ActivationFunctionType.RELU:
        qmin = max(qmin, zp)
    elif act == tflite.ActivationFunctionType.RELU6:
        qmin = max(qmin, zp)
        qmax = min(qmax, zp + int(round(6.0 / f64(scale))))
    elif act != tflite.ActivationFunctionType.NONE:
        raise ValueError(f"unsupported fused activation {act}")
    return qmin, qmax


def rq(in_scale, w_scale, out_scale, out_zp, qmin, qmax):
    """The requantizer (multiplier, shift, zero point, qmin, qmax). Conv/FC
    effective scale as TFLM computes it: in double from float scales. The
    crate's shift > 0 means right shift (opposite of TFLite), as tinymind's."""
    mult, shift = quantize_multiplier(f64(in_scale) * f64(w_scale) / f64(out_scale))
    return (mult, -shift, out_zp, qmin, qmax)


def add_params(sa, sb, so):
    """ADD's ((mult_a, shift_a), (mult_b, shift_b), (mult_out, shift_out)) in
    the crate's shift convention, as make_mlperf.py computes them."""
    # TFLM add.cc: left shift 20, twice the larger input scale (in float).
    twice_max = f64(np.float32(2) * max(np.float32(sa), np.float32(sb)))
    ma, ea = quantize_multiplier(f64(sa) / twice_max)
    mb, eb = quantize_multiplier(f64(sb) / twice_max)
    mo, eo = quantize_multiplier(twice_max / f64(np.float32((1 << 20) * np.float32(so))))
    return (ma, -ea), (mb, -eb), (mo, -eo)


def same_pads(in_size, k, stride):
    out = (in_size + stride - 1) // stride
    total = max((out - 1) * stride + k - in_size, 0)
    return total // 2, total - total // 2, out


def softmax_params(beta, in_scale):
    """TFLM PreprocessSoftmaxScaling (5 integer bits) and CalculateInputRadius:
    (input_multiplier, input_left_shift, diff_min)."""
    real = min(f64(beta) * f64(in_scale) * (1 << 26), (1 << 31) - 1.0)
    mult, left = quantize_multiplier(real)
    assert left >= 0
    diff_min = -int(math.floor(31.0 * (1 << 26) / (1 << left)))
    return mult, left, diff_min


def fnv1a(data):
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------- reading


_OP_NAMES = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
_INT8 = tflite.TensorType.INT8
_INT32 = tflite.TensorType.INT32


class Tensor:
    def __init__(self, model, sub, idx):
        t = sub.Tensors(idx)
        self.index = idx
        self.name = t.Name().decode() if t.Name() else f"t{idx}"
        self.shape = [int(v) for v in t.ShapeAsNumpy()] if t.ShapeLength() else []
        self.type = t.Type()
        q = t.Quantization()
        self.scales = (np.asarray(q.ScaleAsNumpy(), np.float32)
                       if q is not None and q.ScaleLength() else np.zeros(0, np.float32))
        self.zero_points = (np.asarray(q.ZeroPointAsNumpy(), np.int64)
                            if q is not None and q.ZeroPointLength() else np.zeros(0, np.int64))
        buf = model.Buffers(t.Buffer())
        self.data = bytes(buf.DataAsNumpy().tobytes()) if buf is not None and buf.DataLength() else None

    def array(self):
        """The constant data as a numpy array of the tensor's shape."""
        if self.data is None:
            raise ValueError(f"tensor {self.name!r} has no constant data")
        dtype = {_INT8: np.int8, _INT32: np.int32}.get(self.type)
        if dtype is None:
            raise ValueError(f"tensor {self.name!r}: type {self.type} is not supported")
        return np.frombuffer(self.data, dtype=np.dtype(dtype).newbyteorder("<")).reshape(self.shape)

    def scale(self):
        return float(self.scales[0])

    def zp(self):
        return int(self.zero_points[0]) if self.zero_points.size else 0


def _options(op, cls):
    table = op.BuiltinOptions()
    x = cls()
    x.Init(table.Bytes, table.Pos)
    return x


# ---------------------------------------------------------------- translation


def _rust_nested(a: np.ndarray) -> str:
    if a.ndim == 1:
        return "[" + ", ".join(str(int(v)) for v in a.tolist()) + "]"
    return "[" + ", ".join(_rust_nested(sub) for sub in a) + "]"


def _rust_type(shape, elem="i8") -> str:
    t = elem
    for d in reversed(shape):
        t = f"[{t}; {d}]"
    return t


def _zeros(shape) -> str:
    t = "0i8"
    for d in reversed(shape):
        t = f"[{t}; {d}]"
    return t


def _f32_literal(x: float) -> str:
    """The shortest decimal that round-trips the float32 value."""
    s = np.format_float_positional(np.float32(x), unique=True, trim="0")
    return s + "0" if s.endswith(".") else s


def add_literal(a, b, y, act) -> str:
    """A `QAdd` literal for inputs and output `(scale, zero point)` `a`, `b`, `y`
    and fused activation `act`."""
    (ma, sa), (mb, sb), (mo, so) = add_params(a[0], b[0], y[0])
    qmin, qmax = act_range(act, y[0], y[1])
    return (f"QAdd {{ input_a_zero_point: {a[1]}, input_b_zero_point: {b[1]}, left_shift: 20, "
            f"input_a_multiplier: {ma}, input_a_shift: {sa}, input_b_multiplier: {mb}, "
            f"input_b_shift: {sb}, output_requantizer: {_rq_literal((mo, so, y[1], qmin, qmax))} }}")


def _rq_literal(r) -> str:
    return ("Requantizer { multiplier: %d, shift: %d, zero_point: %d, qmin: %d, qmax: %d }"
            % tuple(int(v) for v in r))


class Translation:
    """The model's operators as crate layers: `decls` (statics) and `steps`,
    each reading tensors written by earlier steps (or the input)."""

    def __init__(self, blob: bytes, name: str):
        self.name = name
        self.model = tflite.Model.GetRootAsModel(blob, 0)
        if self.model.SubgraphsLength() != 1:
            raise ValueError("only single-subgraph models are supported")
        self.sub = self.model.Subgraphs(0)
        if self.sub.InputsLength() != 1 or self.sub.OutputsLength() != 1:
            raise ValueError("only models with one input and one output are supported")
        self.input = self.tensor(self.sub.Inputs(0))
        self.output = self.tensor(self.sub.Outputs(0))
        for t in (self.input, self.output):
            if t.type != _INT8:
                raise ValueError(f"tensor {t.name!r}: only int8 inputs and outputs are supported")
        self.decls: list[str] = []
        # (kind, code, input tensors, output tensor, output shape)
        self.steps: list[tuple] = []
        self.ops: list[str] = []
        self.n = 0
        # The shape of every tensor written so far, ("map", h, w, c) or ("vec", n),
        # and the tensors that are another's value under a new name (a RESHAPE
        # that moves nothing).
        self.shapes = {self.input.index: self._shape_of(self.input)}
        self.alias: dict[int, int] = {}

    def tensor(self, idx):
        return Tensor(self.model, self.sub, idx)

    def _shape_of(self, t):
        s = t.shape
        if len(s) == 4 and s[0] == 1:
            return ("map", s[1], s[2], s[3])
        if len(s) == 2 and s[0] == 1:
            return ("vec", s[1])
        if len(s) == 1:
            return ("vec", s[0])
        raise ValueError(f"tensor {t.name!r}: shape {s} is not supported (batch 1, NHWC or a vector)")

    def _src(self, idx: int) -> int:
        return self.alias.get(idx, idx)

    def _reads(self, t):
        """The shape of `t`, which an earlier operator (or the input) wrote."""
        i = self._src(t.index)
        if i not in self.shapes:
            raise ValueError(f"operator reads {t.name!r} before any operator writes it")
        return self.shapes[i]

    def _step(self, kind: str, code: str, ins, out_tensor, out_shape):
        """kind "buffer": `code` writes a feature map through `{y}`, a
        `&mut` view of its place in the arena; kind "value": `code` is an expression, or, with `{y}`,
        statements that fill the new value `{y}`. In both, `{x0}`, `{x1}` are
        references to the inputs and `{n0}` the first input's variable."""
        self.steps.append((kind, code, [self._src(t.index) for t in ins], out_tensor.index, out_shape))
        self.shapes[out_tensor.index] = out_shape

    def translate(self):
        for k in range(self.sub.OperatorsLength()):
            op = self.sub.Operators(k)
            oc = self.model.OperatorCodes(op.OpcodeIndex())
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            name = _OP_NAMES.get(code, str(code))
            self.ops.append(name)
            ins = [self.tensor(i) if i >= 0 else None for i in op.InputsAsNumpy()]
            out = self.tensor(int(op.OutputsAsNumpy()[0]))
            if name == "CONV_2D":
                self.conv(op, ins, out, depthwise=False)
            elif name == "DEPTHWISE_CONV_2D":
                self.conv(op, ins, out, depthwise=True)
            elif name == "AVERAGE_POOL_2D":
                self.avgpool(op, ins, out)
            elif name == "FULLY_CONNECTED":
                self.fully_connected(op, ins, out)
            elif name == "SOFTMAX":
                self.softmax(op, ins, out)
            elif name == "RESHAPE":
                self.reshape(ins, out)
            elif name == "ADD":
                self.add(op, ins, out)
            else:
                raise ValueError(f"unsupported operator {name}")
        if self._src(self.output.index) not in self.shapes:
            raise ValueError("no operator writes the model's output")
        return self

    def conv(self, op, ins, y, depthwise):
        x, w, b = ins[0], ins[1], ins[2] if len(ins) > 2 else None
        if self._reads(x)[0] != "map":
            raise ValueError("convolution of a vector")
        o = _options(op, tflite.DepthwiseConv2DOptions if depthwise else tflite.Conv2DOptions)
        if o.DilationHFactor() != 1 or o.DilationWFactor() != 1:
            raise ValueError("dilated convolution is not supported")
        _, H, W, C = x.shape
        _, OH, OW, F = y.shape
        W8 = w.array()
        B32 = b.array() if b is not None else None
        z_in, z_out = x.zp(), y.zp()
        qmin, qmax = act_range(o.FusedActivationFunction(), y.scale(), z_out)
        sh, sw = o.StrideH(), o.StrideW()
        if depthwise:
            if o.DepthMultiplier() != 1 or W8.shape[0] != 1:
                raise ValueError("depthwise convolution: only depth multiplier 1 is supported")
        KH, KW = W8.shape[1], W8.shape[2]
        s_w = w.scales
        if len(s_w) == 1:
            s_w = np.repeat(s_w, F)
        rqs = [rq(x.scale(), s, y.scale(), z_out, qmin, qmax) for s in s_w]
        if o.Padding() == tflite.Padding.SAME:
            pt, pb, oh = same_pads(H, KH, sh)
            pl, pr, ow = same_pads(W, KW, sw)
            assert (oh, ow) == (OH, OW)
        else:
            pt = pb = pl = pr = 0
            assert ((H - KH) // sh + 1, (W - KW) // sw + 1) == (OH, OW)

        i = self.n
        self.n += 1
        pointwise = not depthwise and KH == KW == 1 and sh == sw == 1 and pt == pb == pl == pr == 0
        tag = ("DW" if depthwise else ("PW" if pointwise else "CONV")) + str(i)
        what = "depthwise convolution" if depthwise else ("pointwise convolution" if pointwise else "convolution")
        self.decls.append(f"/// `{y.name}`: {KH}x{KW} {what}, stride {sh}x{sw}, padding "
                          f"{pt}/{pb}/{pl}/{pr} (top/bottom/left/right), {H}x{W}x{C} -> {OH}x{OW}x{F}.")
        if depthwise:
            wt = np.transpose(W8[0], (2, 0, 1))  # [KH][KW][C] -> [C][KH][KW]
            self.decls.append(f"static {tag}_WEIGHTS: {_rust_type((C, KH, KW))} = {_rust_nested(wt)};")
        elif pointwise:
            self.decls.append(f"static {tag}_WEIGHTS: {_rust_type((F, C))} = {_rust_nested(W8[:, 0, 0, :])};")
        else:
            self.decls.append(f"static {tag}_WEIGHTS: {_rust_type((F, KH, KW, C))} = {_rust_nested(W8)};")
        biases = "None"
        if B32 is not None:
            self.decls.append(f"static {tag}_BIASES: [i32; {F}] = {_rust_nested(B32)};")
            biases = f"Some(&{tag}_BIASES)"
        self.decls.append(f"static {tag}_REQUANTIZERS: [Requantizer; {F}] = "
                          f"[{', '.join(_rq_literal(r) for r in rqs)}];")
        if depthwise:
            ty = f"QDepthwiseConv2D<'static, {C}, {KH}, {KW}, {sh}, {sw}, {pt}, {pb}, {pl}, {pr}>"
            init = "QDepthwiseConv2D"
        elif pointwise:
            ty = f"QPointwiseConv2DPerChannel<'static, {C}, {F}>"
            init = "QPointwiseConv2DPerChannel"
        else:
            ty = f"QConv2DPerChannel<'static, {C}, {KH}, {KW}, {F}, {sh}, {sw}, {pt}, {pb}, {pl}, {pr}>"
            init = "QConv2DPerChannel"
        self.decls.append(f"static {tag}: {ty} = {init} {{ weights: &{tag}_WEIGHTS, biases: {biases}, "
                          f"input_zero_point: {z_in}, requantizers: &{tag}_REQUANTIZERS }};\n")
        self._step("buffer", f"{tag}.forward({{x0}}, {{y}})", [x], y, ("map", OH, OW, F))

    def avgpool(self, op, ins, y):
        x = ins[0]
        if self._reads(x)[0] != "map":
            raise ValueError("average pool of a vector")
        o = _options(op, tflite.Pool2DOptions)
        _, H, W, C = x.shape
        if not (o.FilterHeight() == H and o.FilterWidth() == W and o.Padding() == tflite.Padding.VALID):
            raise ValueError("only a global average pool is supported")
        assert np.isclose(x.scale(), y.scale()) and x.zp() == y.zp(), "avgpool requantizes"
        qmin, qmax = act_range(o.FusedActivationFunction(), y.scale(), y.zp())
        i = self.n
        self.n += 1
        tag = f"AVGPOOL{i}"
        self.decls.append(f"/// `{y.name}`: global average pool, {H}x{W}x{C} -> {C}.")
        self.decls.append(f"static {tag}: QGlobalAvgPool2D = QGlobalAvgPool2D {{ qmin: {qmin}, qmax: {qmax} }};\n")
        self._step("value", f"{tag}.forward({{x0}})", [x], y, ("vec", C))

    def fully_connected(self, op, ins, y):
        x, w, b = ins[0], ins[1], ins[2] if len(ins) > 2 else None
        if self._reads(x)[0] != "vec":
            raise ValueError("FULLY_CONNECTED on a feature map: RESHAPE it to a vector first")
        o = _options(op, tflite.FullyConnectedOptions)
        W8 = w.array()
        n_out, n_in = W8.shape
        if len(w.scales) != 1:
            raise ValueError("per-channel FULLY_CONNECTED is not supported (tinymind's translator "
                             "asserts per-tensor)")
        qmin, qmax = act_range(o.FusedActivationFunction(), y.scale(), y.zp())
        i = self.n
        self.n += 1
        tag = f"FC{i}"
        self.decls.append(f"/// `{y.name}`: fully connected, {n_in} -> {n_out}.")
        self.decls.append(f"static {tag}_WEIGHTS: {_rust_type((n_out, n_in))} = {_rust_nested(W8)};")
        biases = "None"
        if b is not None:
            self.decls.append(f"static {tag}_BIASES: [i32; {n_out}] = {_rust_nested(b.array())};")
            biases = f"Some(&{tag}_BIASES)"
        r = rq(x.scale(), w.scale(), y.scale(), y.zp(), qmin, qmax)
        self.decls.append(f"static {tag}: QDense<'static, {n_in}, {n_out}> = QDense {{ weights: &{tag}_WEIGHTS, "
                          f"biases: {biases}, input_zero_point: {x.zp()}, requantizer: {_rq_literal(r)} }};\n")
        self._step("value", f"{tag}.forward({{x0}})", [x], y, ("vec", n_out))

    def softmax(self, op, ins, y):
        x = ins[0]
        shape = self._reads(x)
        if shape[0] != "vec":
            raise ValueError("only a softmax over one row is supported")
        beta = _options(op, tflite.SoftmaxOptions).Beta()
        assert np.float32(y.scale()) == np.float32(1.0 / 256) and y.zp() == -128
        mult, left, diff_min = softmax_params(beta, x.scale())
        i = self.n
        self.n += 1
        tag = f"SOFTMAX{i}"
        self.decls.append(f"/// `{y.name}`: softmax over {shape[1]} logits.")
        self.decls.append(f"static {tag}: QSoftmax = QSoftmax {{ input_multiplier: {mult}, "
                          f"input_left_shift: {left}, diff_min: {diff_min} }};\n")
        self._step("value", f"{tag}.forward({{x0}})", [x], y, shape)

    def reshape(self, ins, y):
        x = ins[0]
        shape = self._reads(x)
        new = self._shape_of(y)
        if new == shape or (new[0] == "vec" and shape[0] == "vec"):
            self.alias[y.index] = self._src(x.index)  # same memory, nothing to do
            return
        if shape[0] == "map" and new[0] == "vec":
            # NHWC is the vector's order already: copy the elements.
            n = new[1]
            self._step("value", f"let mut {{y}} = [0i8; {n}]; for (o, &v) in {{y}}.iter_mut()"
                       f".zip({{n0}}.iter().flatten().flatten()) {{ *o = v; }}", [x], y, new)
            return
        raise ValueError(f"RESHAPE from {shape} to {new} is not supported")

    def add(self, op, ins, y):
        a, b = ins[0], ins[1]
        shape = self._reads(a)
        if self._reads(b) != shape or self._shape_of(y) != shape:
            raise ValueError("ADD: only inputs and output of one shape are supported (no broadcasting)")
        for t in (a, b, y):
            if t.scales.size != 1:
                raise ValueError(f"ADD: tensor {t.name!r} is not quantized per tensor")
        act = _options(op, tflite.AddOptions).FusedActivationFunction()
        i = self.n
        self.n += 1
        tag = f"ADD{i}"
        dims = " x ".join(str(d) for d in shape[1:])
        self.decls.append(f"/// `{y.name}`: elementwise addition, {dims}.")
        self.decls.append(f"static {tag}: QAdd = "
                          f"{add_literal((a.scale(), a.zp()), (b.scale(), b.zp()), (y.scale(), y.zp()), act)};\n")
        if shape[0] == "map":
            self._step("buffer", f"{tag}.forward({{x0}}, {{x1}}, {{y}})", [a, b], y, shape)
        else:
            self._step("value", f"let mut {{y}} = [0i8; {shape[1]}]; "
                       f"{tag}.forward({{x0}}, {{x1}}, &mut {{y}});", [a, b], y, shape)

    def emit(self, path: str, model_note: str = ""):
        """Write the Rust module: constants, layers and `forward`."""
        in_t, out_t = self.input, self.output
        in_shape = self._shape_of(in_t)
        out_src = self._src(out_t.index)
        out_shape = self.shapes[out_src]
        lines = [f"//! `{self.name}`: generated by tinymind-rs tools/import_tflite/tinymind_import_tflite.py",
                 "//! from an int8 TFLite model. Do not edit; regenerate from the model instead."]
        if model_note:
            lines += [f"//! {line}" for line in model_note.splitlines()]
        lines += ["//!",
                  "//! Integer-only and `no_std`: int8 input in the `INPUT` grid, int8 output in the",
                  "//! `OUTPUT` grid (real value = `SCALE * (q - ZERO_POINT)`). Feature maps are NHWC",
                  "//! without the batch axis: `[row][column][channel]`.",
                  "",
                  "// Not every program uses every constant.",
                  "#![allow(dead_code)]",
                  "",
                  "use tinymind::int8::*;",
                  ""]
        if in_shape[0] == "map":
            lines.append(f"/// The input: {in_shape[1]} x {in_shape[2]} x {in_shape[3]} (NHWC).")
            lines.append(f"pub type Input = {_rust_type(in_shape[1:])};")
        else:
            lines.append(f"/// The input vector.\npub type Input = [i8; {in_shape[1]}];")
        if out_shape[0] == "vec":
            lines.append(f"/// The output vector.\npub type Output = [i8; {out_shape[1]}];")
        else:
            lines.append(f"/// The output feature map.\npub type Output = {_rust_type(out_shape[1:])};")
        lines += [f"/// Scale of the input tensor.\npub const INPUT_SCALE: f32 = {_f32_literal(in_t.scale())};",
                  f"/// Zero point of the input tensor.\npub const INPUT_ZERO_POINT: i8 = {in_t.zp()};",
                  f"/// Scale of the output tensor.\npub const OUTPUT_SCALE: f32 = {_f32_literal(out_t.scale())};",
                  f"/// Zero point of the output tensor.\npub const OUTPUT_ZERO_POINT: i8 = {out_t.zp()};",
                  ""]
        lines += self.decls
        # Feature maps live in one byte arena, planned as tinymind's translator
        # plans its arena: each map is placed first-fit at the lowest offset
        # that overlaps no map still to be read, so maps of every shape share
        # the same bytes (ResNet-8: 49,152 B). Layers take typed tensors: a
        # step splits the arena around its output (`split_at_mut`) and views
        # each part as an NHWC array (`view`/`view_mut`, safe: `as_chunks` and
        # `first_chunk`). Vectors are values.
        last_use = {}
        for k, (_, _, ins, _, _) in enumerate(self.steps):
            for t in ins:
                last_use[t] = k
        last_use[out_src] = len(self.steps)
        place: dict[int, tuple] = {}  # map tensor -> (offset, length, dims)
        live: list[int] = []
        arena = 0
        for k, (kind, _, _, out, shape) in enumerate(self.steps):
            if kind != "buffer":
                continue
            live = [t for t in live if last_use.get(t, -1) >= k]
            dims = tuple(shape[1:])
            need = int(np.prod(dims))
            off = 0
            for t in sorted(live, key=lambda t: place[t][0]):
                o, n, _ = place[t]
                if off + need <= o:
                    break
                off = max(off, o + n)
            place[out] = (off, need, dims)
            live.append(out)
            arena = max(arena, off + need)

        def dims_args(dims):
            return ", ".join(str(d) for d in dims)

        def read(t, part=None):
            """Expression for tensor t: a reference (`input`, a view, `&v`)."""
            if t == in_t.index:
                return "input"
            if t in place:
                o, n, dims = place[t]
                if part is None:
                    return f"view::<{dims_args(dims)}>(&arena[{o}..{o + n}])"
                name, base = part(o, n)
                return f"view::<{dims_args(dims)}>(&{name}[{o - base}..{o - base + n}])"
            return f"&{var[t]}"

        body: list[str] = []
        if place:
            body.append(f"    let mut arena = [0i8; {arena}];")
        var = {in_t.index: "input"}  # tensor -> the variable holding it (vectors)
        for k, (kind, code, ins, out, shape) in enumerate(self.steps):
            if kind == "buffer":
                o, n, dims = place[out]
                used = set()

                def part(to, tn, o=o, n=n, used=used):
                    if to + tn <= o:
                        used.add("lo")
                        return "lo", 0
                    used.add("hi")
                    return "hi", o + n

                refs = [read(t, part) for t in ins]
                for j, r in enumerate(refs):
                    code = code.replace("{x%d}" % j, r)
                code = code.replace("{y}", f"view_mut::<{dims_args(dims)}>(y)")
                lo = "lo" if "lo" in used else "_"
                hi = "hi" if "hi" in used else "_"
                body += ["    {",
                         f"        let ({lo}, rest) = arena.split_at_mut({o});",
                         f"        let (y, {hi}) = rest.split_at_mut({n});",
                         f"        {code};",
                         "    }"]
            else:
                refs = [read(t) for t in ins]
                bare = refs[0][1:] if refs[0].startswith("&") else refs[0]
                code = code.replace("{n0}", bare)
                for j, r in enumerate(refs):
                    code = code.replace("{x%d}" % j, r)
                y = f"v{k}"
                if "{y}" in code:
                    body.append("    " + code.replace("{y}", y))
                else:
                    body.append(f"    let {y} = {code};")
                var[out] = y
        if out_src in place:
            body.append(f"    *{read(out_src)}")
        else:
            cur = var[out_src]
            # The last value is the function's value (clippy::let_and_return).
            last = body[-1]
            if last.startswith(f"    let {cur} = ") and last.endswith(";"):
                body[-1] = "    " + last[len(f"    let {cur} = "):-1]
            else:
                body.append(f"    {cur}")
        if place:
            lines += ["/// Feature map `[..; H]` of `[..; W]` of `[i8; C]` stored at `s` (its first",
                      "/// `H * W * C` bytes), as an NHWC tensor.",
                      "fn view<const H: usize, const W: usize, const C: usize>(s: &[i8]) -> &[[[i8; C]; W]; H] {",
                      "    let (pixels, _) = s.as_chunks::<C>();",
                      "    let (rows, _) = pixels.as_chunks::<W>();",
                      "    rows.first_chunk::<H>().expect(\"the arena plan gives every map its size\")",
                      "}",
                      "",
                      "/// [`view`], writable.",
                      "fn view_mut<const H: usize, const W: usize, const C: usize>(s: &mut [i8]) -> &mut [[[i8; C]; W]; H] {",
                      "    let (pixels, _) = s.as_chunks_mut::<C>();",
                      "    let (rows, _) = pixels.as_chunks_mut::<W>();",
                      "    rows.first_chunk_mut::<H>().expect(\"the arena plan gives every map its size\")",
                      "}",
                      ""]
        lines += ["/// Run the network on an int8 input in the `INPUT` grid.",
                  "pub fn forward(input: &Input) -> Output {"] + body + ["}"]
        text = "\n".join(lines) + "\n"
        with open(path, "w") as f:
            f.write(text)
        rustfmt = shutil.which("rustfmt")
        if rustfmt:
            subprocess.run([rustfmt, "--edition", "2024", path], check=True)
        else:
            print("tinymind_import_tflite: rustfmt not found; the module is valid but unformatted",
                  file=sys.stderr)


def translate(tflite_path: str, output_path: str, name: str, model_note: str = "") -> Translation:
    """Translate the int8 model at `tflite_path` into a Rust module at
    `output_path`. Returns the translation (operators, input and output
    tensors)."""
    with open(tflite_path, "rb") as f:
        blob = f.read()
    t = Translation(blob, name).translate()
    t.emit(output_path, model_note)
    return t


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: tinymind_import_tflite.py <model.tflite> <out.rs> <name>")
    tr = translate(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"wrote {sys.argv[2]}: {len(tr.ops)} operators ({', '.join(sorted(set(tr.ops)))})")
