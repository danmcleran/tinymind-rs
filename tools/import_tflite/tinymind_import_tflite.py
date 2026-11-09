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
  * the output is a `no_std` Rust module with a `forward` function. Each layer's
    output is a typed array in its own block, so a tensor's stack space ends
    with the layer that reads it (tinymind plans one arena instead); SAME
    padding goes into the layers' padding parameters, which have no band limit
    in the crate (tinymind adds a QPad2D pass for wide layers: same values).

Supported operators: CONV_2D, DEPTHWISE_CONV_2D (depth multiplier 1),
AVERAGE_POOL_2D (global only, as tinymind), FULLY_CONNECTED, SOFTMAX (one row)
and RESHAPE; fused NONE / RELU / RELU6 activations; int8 activations and
weights, per-tensor or per-channel. ADD (residual networks) is not supported
yet.
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


def _rq_literal(r) -> str:
    return ("Requantizer { multiplier: %d, shift: %d, zero_point: %d, qmin: %d, qmax: %d }"
            % tuple(int(v) for v in r))


class Translation:
    """The model's operators as crate layers: `decls` (statics) and `steps`,
    each a function of the previous step's value, `x`."""

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
        self.steps: list[str] = []
        self.ops: list[str] = []
        self.n = 0
        # The value flowing through the chain: its tensor index and shape,
        # ("map", h, w, c) or ("vec", n).
        self.current = self.input.index
        self.shape = self._shape_of(self.input)

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

    def _reads(self, t):
        if t.index != self.current:
            raise ValueError(f"operator reads {t.name!r}, but the chain is at tensor {self.current}: "
                             f"only a single chain is supported")

    def _step(self, kind: str, code: str, out_tensor, out_shape):
        """kind "buffer": `code` is a layer that writes a feature map into a
        caller-owned buffer; kind "value": `code` is an expression of `{x}`
        (or, with `{y}`, statements that fill the new value `{y}`)."""
        self.steps.append((kind, code, out_shape))
        self.current = out_tensor.index
        self.shape = out_shape

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
            else:
                raise ValueError(f"unsupported operator {name}")
        if self.current != self.output.index:
            raise ValueError("the chain does not end at the model's output")
        return self

    def conv(self, op, ins, y, depthwise):
        x, w, b = ins[0], ins[1], ins[2] if len(ins) > 2 else None
        self._reads(x)
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
        self._step("buffer", tag, y, ("map", OH, OW, F))

    def avgpool(self, op, ins, y):
        x = ins[0]
        self._reads(x)
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
        self._step("value", f"{tag}.forward({{x}})", y, ("vec", C))

    def fully_connected(self, op, ins, y):
        x, w, b = ins[0], ins[1], ins[2] if len(ins) > 2 else None
        self._reads(x)
        if self.shape[0] != "vec":
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
        self._step("value", f"{tag}.forward({{x}})", y, ("vec", n_out))

    def softmax(self, op, ins, y):
        x = ins[0]
        self._reads(x)
        if self.shape[0] != "vec":
            raise ValueError("only a softmax over one row is supported")
        beta = _options(op, tflite.SoftmaxOptions).Beta()
        assert np.float32(y.scale()) == np.float32(1.0 / 256) and y.zp() == -128
        mult, left, diff_min = softmax_params(beta, x.scale())
        i = self.n
        self.n += 1
        tag = f"SOFTMAX{i}"
        self.decls.append(f"/// `{y.name}`: softmax over {self.shape[1]} logits.")
        self.decls.append(f"static {tag}: QSoftmax = QSoftmax {{ input_multiplier: {mult}, "
                          f"input_left_shift: {left}, diff_min: {diff_min} }};\n")
        self._step("value", f"{tag}.forward({{x}})", y, self.shape)

    def reshape(self, ins, y):
        x = ins[0]
        self._reads(x)
        new = self._shape_of(y)
        if new == self.shape or (new[0] == "vec" and self.shape[0] == "vec"):
            self.current = y.index  # same memory, nothing to do
            return
        if self.shape[0] == "map" and new[0] == "vec":
            # NHWC is the vector's order already: copy the elements.
            n = new[1]
            self._step("value", f"let mut {{y}} = [0i8; {n}]; for (o, &v) in {{y}}.iter_mut()"
                       f".zip({{x}}.iter().flatten().flatten()) {{{{ *o = v; }}}}", y, new)
            return
        raise ValueError(f"RESHAPE from {self.shape} to {new} is not supported")

    def emit(self, path: str, model_note: str = ""):
        """Write the Rust module: constants, layers and `forward`."""
        in_t, out_t = self.input, self.output
        in_shape = self._shape_of(in_t)
        out_shape = self.shape
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
        # Feature maps go into buffers, one per shape and as many as are live
        # at once: a layer writes into a buffer of its output's shape that does
        # not hold its input, so a chain of same-shaped layers ping-pongs
        # between two (typed arrays of different shapes cannot share storage
        # without `unsafe`). Vectors are values.
        body: list[str] = []
        buffers: dict[tuple, list[str]] = {}
        cur = None  # the variable holding the chain's value; None: `input`
        for k, (kind, code, shape) in enumerate(self.steps):
            x = "input" if cur is None else f"&{cur}"
            if kind == "buffer":
                dims = tuple(shape[1:])
                free = [b for b in buffers.get(dims, []) if b != cur]
                if free:
                    out = free[0]
                else:
                    out = f"b{sum(len(v) for v in buffers.values())}"
                    buffers.setdefault(dims, []).append(out)
                    body.append(f"    let mut {out} = {_zeros(dims)};")
                body.append(f"    {code}.forward({x}, &mut {out});")
                cur = out
            else:
                v = f"v{k}"
                if "{y}" in code:
                    body.append("    " + code.replace("{y}", v).replace("{x}", x.lstrip("&")))
                else:
                    body.append(f"    let {v} = {code.replace('{x}', x)};")
                cur = v
        # The last value is the function's value (clippy::let_and_return).
        last = body[-1]
        if last.startswith(f"    let {cur} = ") and last.endswith(";"):
            body[-1] = "    " + last[len(f"    let {cur} = "):-1]
        else:
            body.append(f"    {cur}")
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
