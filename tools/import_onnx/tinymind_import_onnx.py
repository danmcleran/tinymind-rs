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

ONNX Runtime int8 model -> tinymind (Rust) module.

A port of tinymind (C++)'s apps/import_onnx/tinymind_import_onnx.py with a
Rust back end. It reads a model quantized by ONNX Runtime's
`onnxruntime.quantization.quantize_static` (QDQ, the default, or QOperator
format; int8 weights and activations, weights per tensor or per output
channel) and writes a `no_std`, integer-only Rust module with the model's
int8 weights, int32 biases, requantizers, activation tables and a `forward`
function.

Everything up to and including `parse_onnx_model` is tinymind (C++)'s code,
unchanged: the same layers, scales, zero points, int8 weights (OHWI for
convolutions, [out][in] for dense layers, columns permuted to NHWC after a
Flatten), int32 biases and requantizer (multiplier, shift) pairs, and the same
refusals (see `parse_onnx_model`). New here, after it:

  * the layers are checked to form one chain from the model's input to its
    output, each reading the tensor the previous one wrote in the grid it was
    written in (tinymind (C++)'s importer emits per-layer constants for a
    hand-written network and leaves that to the caller);
  * the Rust back end: QConv2D / QConv2DPerChannel (QPointwiseConv2D /
    ...PerChannel for a 1x1, stride 1, unpadded convolution), QDense /
    QDensePerChannel, 256-entry tables for Sigmoid and Tanh (tinymind (C++)'s
    buildQSigmoidLUT / buildQTanhLUT, step for step in float32), qrelu for a
    standalone Relu, and QSoftmax (TFLite Micro's int8 softmax, parameters
    from tinymind (C++)'s buildQSoftmaxFixedPointParams) for a Softmax.

A Softmax's output is on TFLite's probability grid (scale 1/256, zero point
-128), which is what QSoftmax computes, not on the 1/255 grid ONNX Runtime
gives it; the module's OUTPUT_SCALE says which. Feature maps are NHWC: feed
the module the transpose of the NCHW tensor the ONNX model takes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# Shared helpers (mirror tinymind_import.py / cpp/qaffine.hpp).
# ----------------------------------------------------------------------------


def lround(x: float) -> int:
    """std::lround semantics: ties round away from zero."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def quantize_multiplier(ratio: float) -> tuple[int, int]:
    """Mirror cpp/qaffine.hpp::quantizeMultiplier (and tinymind_import.py).

    Decomposes a positive ratio into the Q0.31 (multiplier, shift) pair a
    Requantizer takes. shift > 0 right-shifts after the multiply; shift < 0
    left-shifts before it. The pair always satisfies -31 <= shift <= 31.
    """
    if ratio == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(ratio)
    q = lround(mantissa * float(1 << 31))
    if q == (1 << 31):
        q >>= 1
        exponent += 1
    shift = -int(exponent)
    if shift > 31:
        return 0, 0
    return int(q), max(shift, -31)


COPYRIGHT_BLOCK = """/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import_onnx.
*/"""


# ----------------------------------------------------------------------------
# Parsed model representation.
# ----------------------------------------------------------------------------


@dataclass
class OnnxQuantizedLayer:
    """One tinymind layer recovered from a quantized ONNX graph.

    op_type      : "Conv" / "Dense" for weight layers; "Relu" / "Sigmoid" /
                   "Tanh" / "Softmax" for activation-only records.
    name         : C++-safe layer name (from the ONNX node name).
    input_name / output_name : ONNX tensor names of the quantized input and
                   output.
    input_scale / input_zp / output_scale / output_zp : per-tensor
                   activation quantization.
    weight       : int8 numpy array in tinymind layout -- OHWI
                   [F][KH][KW][C] for Conv, [out][in] for Dense.
    weight_scales: one scale per output channel when per_channel, else a
                   single-element list. weight_scale is weight_scales[0].
    bias         : int32 numpy array at scale input_scale * weight_scales[o],
                   or None.
    multipliers / shifts : the Requantizer (multiplier, shift) per output
                   channel (one entry when per-tensor), for ratio
                   input_scale * weight_scale / output_scale.
    geometry     : Conv: H, W, C, KH, KW, StrideH, StrideW, F, PadTop,
                   PadBottom, PadLeft, PadRight, OutH, OutW.
                   Dense: NumInputs, NumOutputs.
    """

    op_type: str
    name: str
    input_name: str
    input_scale: float
    input_zp: int
    output_scale: float
    output_zp: int
    output_name: str = ""
    weight: object = None
    weight_scale: float = 0.0
    weight_scales: list = field(default_factory=list)
    per_channel: bool = False
    bias: object = None
    bias_scale: float = 0.0
    multipliers: list = field(default_factory=list)
    shifts: list = field(default_factory=list)
    geometry: dict = field(default_factory=dict)


def sanitize(name: str) -> str:
    """Map an ONNX name to a C++ identifier fragment."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    text = "".join(out).strip("_")
    return text if text else "layer"


def _finish_weight_layer(op_type, name, input_name, input_scale, input_zp,
                         output_name, output_scale, output_zp,
                         weight_q, weight_scales, bias_float, geometry):
    """Common tail: quantize the bias, build requantizers, fill the record."""
    import numpy as np

    out_channels = weight_q.shape[0]
    scales = [float(s) for s in weight_scales]
    per_channel = len(scales) > 1
    if per_channel and len(scales) != out_channels:
        raise ValueError(f"{name}: {len(scales)} weight scales for {out_channels} output channels")
    channel_scales = scales if per_channel else scales * out_channels

    bias_q = None
    if bias_float is not None:
        bias_float = np.asarray(bias_float, dtype=np.float64).reshape(-1)
        if bias_float.shape[0] != out_channels:
            raise ValueError(f"{name}: bias has {bias_float.shape[0]} values for {out_channels} outputs")
        qmin, qmax = -(1 << 31), (1 << 31) - 1
        bias_q = np.array([max(qmin, min(qmax, lround(b / (input_scale * s))))
                           for b, s in zip(bias_float.tolist(), channel_scales)], dtype=np.int32)

    multipliers, shifts = [], []
    for s in (scales if per_channel else scales[:1]):
        m, sh = quantize_multiplier(input_scale * s / output_scale)
        multipliers.append(m)
        shifts.append(sh)

    return OnnxQuantizedLayer(
        op_type=op_type, name=name, input_name=input_name,
        input_scale=input_scale, input_zp=input_zp,
        output_scale=output_scale, output_zp=output_zp, output_name=output_name,
        weight=weight_q.astype(np.int8), weight_scale=scales[0], weight_scales=scales,
        per_channel=per_channel, bias=bias_q, bias_scale=input_scale * scales[0],
        multipliers=multipliers, shifts=shifts, geometry=geometry)


def parse_onnx_model(model_path: str) -> list[OnnxQuantizedLayer]:
    """Parse a QDQ- or QOperator-format quantized ONNX model.

    Returns the tinymind layers in execution order. Raises ValueError for
    anything that cannot be mapped exactly (see the module docstring); a
    layer is never dropped silently.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    graph = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    inits[node.output[0]] = numpy_helper.to_array(a.t)

    producer = {}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    def attrs(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def scalar_scale(tensor, what):
        arr = np.asarray(inits[tensor], dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"{what}: per-channel activation scale is not supported")
        return float(arr[0])

    def zero_point(tensor, what, allow_int32=False):
        if tensor is None or tensor == "":
            return np.zeros(1, dtype=np.int64)
        arr = np.asarray(inits[tensor])
        if arr.dtype == np.uint8:
            raise ValueError(f"{what}: uint8 tensors are not supported; quantize with "
                             "activation_type=QuantType.QInt8 and weight_type=QuantType.QInt8")
        if arr.dtype != np.int8 and not (allow_int32 and arr.dtype == np.int32):
            raise ValueError(f"{what}: unsupported zero-point type {arr.dtype}")
        return arr.astype(np.int64).reshape(-1)

    def node_label(node):
        return sanitize(node.name if node.name else node.output[0])

    # Shapes, tracked in ONNX's NCHW convention. flat_chw[t] = (C, H, W) when
    # tensor t is a flattened 4D tensor, so the dense layer it feeds can
    # permute its weight columns to tinymind's NHWC flattening order.
    shapes: dict[str, list] = {}
    flat_chw: dict[str, tuple] = {}
    for vi in graph.input:
        if vi.name in inits:
            continue
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1)
        shapes[vi.name] = dims

    def same_shape(dst, src):
        if src in shapes:
            shapes[dst] = list(shapes[src])
        if src in flat_chw:
            flat_chw[dst] = flat_chw[src]

    def flatten_to_2d(dst, src):
        shape = shapes.get(src)
        if shape is None:
            raise ValueError(f"cannot flatten {src}: its shape is unknown")
        if len(shape) == 4:
            flat_chw[dst] = (shape[1], shape[2], shape[3])
        elif src in flat_chw:
            flat_chw[dst] = flat_chw[src]
        shapes[dst] = [shape[0], int(np.prod(shape[1:]))]

    def conv_geometry(node, x_name, weight_oihw):
        a = attrs(node)
        if a.get("group", 1) != 1:
            raise ValueError(f"{node.name}: grouped/depthwise Conv (group={a['group']}) is not supported")
        if any(d != 1 for d in a.get("dilations", [1, 1])):
            raise ValueError(f"{node.name}: dilated Conv is not supported")
        auto_pad = a.get("auto_pad", b"NOTSET")
        auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
        if auto_pad not in ("NOTSET", "VALID"):
            raise ValueError(f"{node.name}: auto_pad={auto_pad} is not supported; export explicit pads")
        pads = a.get("pads", [0, 0, 0, 0]) if auto_pad == "NOTSET" else [0, 0, 0, 0]
        strides = a.get("strides", [1, 1])
        x_shape = shapes.get(x_name)
        if x_shape is None or len(x_shape) != 4:
            raise ValueError(f"{node.name}: Conv input shape is unknown or not 4D")
        _, c, h, w = x_shape
        f, wc, kh, kw = weight_oihw.shape
        if wc != c:
            raise ValueError(f"{node.name}: weight has {wc} input channels, input has {c}")
        oh = (h + pads[0] + pads[2] - kh) // strides[0] + 1
        ow = (w + pads[1] + pads[3] - kw) // strides[1] + 1
        return {"InputHeight": h, "InputWidth": w, "InputChannels": c,
                "KernelHeight": kh, "KernelWidth": kw,
                "StrideH": strides[0], "StrideW": strides[1], "NumFilters": f,
                "PadTop": pads[0], "PadLeft": pads[1], "PadBottom": pads[2], "PadRight": pads[3],
                "OutputHeight": oh, "OutputWidth": ow}, [x_shape[0], f, oh, ow]

    def oihw_to_ohwi(weight):
        return np.ascontiguousarray(np.transpose(weight, (0, 2, 3, 1)))

    def dense_from_out_in(weight_out_in, x_name, label):
        """Permute the input columns of a [out][in] dense weight from NCHW
        flattening order to NHWC when the input is a flattened 4D tensor."""
        x_shape = shapes.get(x_name)
        n_in = weight_out_in.shape[1]
        if x_shape is not None and int(np.prod(x_shape[1:])) != n_in:
            raise ValueError(f"{label}: dense layer expects {n_in} inputs, its input has shape {x_shape}")
        if x_name in flat_chw:
            c, h, w = flat_chw[x_name]
            if c * h * w != n_in:
                raise ValueError(f"{label}: flattened input {c}x{h}x{w} does not match {n_in} weights")
            # tinymind index (y*W + x)*C + ch  <-  ONNX index ch*H*W + y*W + x
            perm = np.arange(n_in).reshape(c, h, w).transpose(1, 2, 0).reshape(-1)
            weight_out_in = weight_out_in[:, perm]
        return np.ascontiguousarray(weight_out_in)

    def check_weight_zero_points(zp, label):
        if np.any(zp != 0):
            raise ValueError(f"{label}: weight zero point {zp.tolist()} is not 0; tinymind's int8 "
                             "layers use symmetric weights (quantize with weight_type=QuantType.QInt8, "
                             "which is symmetric)")

    def weight_scales_for(scales, axis, n_out, out_axis, label):
        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        if scales.size == 1:
            return [float(scales[0])]
        if axis != out_axis or scales.size != n_out:
            raise ValueError(f"{label}: per-channel weight scales must run along the output channels")
        return [float(s) for s in scales]

    layers: list[OnnxQuantizedLayer] = []

    # ---------------------------------------------------------------- QDQ
    def dq_source(tensor):
        """(quantized tensor, scale tensor, zp tensor, axis) for a DQ output."""
        node = producer.get(tensor)
        if node is None or node.op_type != "DequantizeLinear":
            return None
        zp = node.input[2] if len(node.input) > 2 else ""
        return node.input[0], node.input[1], zp, attrs(node).get("axis", 1)

    def dq_activation(tensor, label):
        src = dq_source(tensor)
        if src is None or src[0] in inits:
            raise ValueError(f"{label}: input {tensor} is not a dequantized activation")
        return src[0], scalar_scale(src[1], label), int(zero_point(src[2], label)[0])

    def dq_constant(tensor, label, allow_int32=False):
        """Float value of a constant input: an initializer or the DQ of one.
        Returns (float array, raw quantized array or None, scales, zp, axis)."""
        if tensor in inits:
            return np.asarray(inits[tensor], dtype=np.float64), None, None, None, None
        src = dq_source(tensor)
        if src is None or src[0] not in inits:
            raise ValueError(f"{label}: {tensor} is not a constant")
        q = np.asarray(inits[src[0]])
        scales = np.asarray(inits[src[1]], dtype=np.float64)
        zp = zero_point(src[2], label, allow_int32=allow_int32)
        axis = src[3]
        if scales.size == 1:
            real = (q.astype(np.float64) - zp[0]) * scales.reshape(-1)[0]
        else:
            shape = [1] * q.ndim
            shape[axis % q.ndim] = -1
            real = (q.astype(np.float64) - zp.reshape(shape)) * scales.reshape(shape)
        return real, q, scales, zp, axis

    def quantized_output(tensor, label):
        """(scale, zp, quantized tensor name) of the QuantizeLinear consuming tensor."""
        users = consumers.get(tensor, [])
        qs = [n for n in users if n.op_type == "QuantizeLinear"]
        if len(users) != 1 or len(qs) != 1:
            return None
        q = qs[0]
        return scalar_scale(q.input[1], label), int(zero_point(q.input[2], label)[0]), q.output[0]

    skip_nodes: set[int] = set()

    def is_constant(tensor):
        if tensor in inits:
            return True
        src = dq_source(tensor)
        return src is not None and src[0] in inits

    def bias_add_after(tensor):
        """(Add node, constant operand) when tensor feeds a bias Add, directly
        or through a QuantizeLinear -> DequantizeLinear pair; else (None, None)."""
        hop = tensor
        for _ in range(2):
            users = consumers.get(hop, [])
            if len(users) != 1:
                return None, None
            user = users[0]
            if user.op_type == "Add":
                other = user.input[1] if user.input[0] == hop else user.input[0]
                if is_constant(other):
                    return user, other
                return None, None
            if user.op_type != "QuantizeLinear":
                return None, None
            dq_users = consumers.get(user.output[0], [])
            if len(dq_users) != 1 or dq_users[0].op_type != "DequantizeLinear":
                return None, None
            hop = dq_users[0].output[0]
        return None, None

    def parse_qdq_weight_node(node):
        label = node_label(node)
        x_q, x_scale, x_zp = dq_activation(node.input[0], label)
        _, w_q, w_scales, w_zp, w_axis = dq_constant(node.input[1], label)
        if w_q is None:
            raise ValueError(f"{label}: weights are float, not quantized")
        if w_q.dtype != np.int8:
            raise ValueError(f"{label}: weights must be int8, got {w_q.dtype}")
        check_weight_zero_points(w_zp, label)
        bias = None
        if node.op_type in ("Conv", "Gemm") and len(node.input) > 2 and node.input[2]:
            bias = dq_constant(node.input[2], label, allow_int32=True)[0].reshape(-1)
        a = attrs(node)
        out_tensor = node.output[0]
        if node.op_type == "Conv":
            geometry, out_shape = conv_geometry(node, x_q, w_q)
            scales = weight_scales_for(w_scales, w_axis, w_q.shape[0], 0, label)
            weight = oihw_to_ohwi(w_q)
            kind = "Conv"
        else:
            if node.op_type == "Gemm":
                if a.get("transA", 0) != 0 or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                    raise ValueError(f"{label}: Gemm with transA/alpha/beta other than 0/1/1 is not supported")
                trans_b = a.get("transB", 0)
            else:
                trans_b = 0
            if trans_b:
                w_out_in, out_axis = w_q, 0
            else:
                w_out_in, out_axis = w_q.T, 1
            scales = weight_scales_for(w_scales, w_axis, w_out_in.shape[0], out_axis, label)
            weight = dense_from_out_in(w_out_in, x_q, label)
            if node.op_type == "MatMul":
                # A bias Add after the MatMul (Linear on a 3D input, or a
                # hand-built graph) folds into this layer's bias. ONNX Runtime
                # may quantize the MatMul output before the Add (Q -> DQ ->
                # Add); the folded layer then rounds once where ONNX Runtime
                # rounds twice, so it can differ from it by one more LSB.
                add, other = bias_add_after(out_tensor)
                if add is not None:
                    bias = dq_constant(other, label, allow_int32=True)[0].reshape(-1)
                    out_tensor = add.output[0]
                    skip_nodes.add(id(add))
            geometry = {"NumInputs": weight.shape[1], "NumOutputs": weight.shape[0]}
            out_shape = [shapes.get(x_q, [1])[0], weight.shape[0]]
            kind = "Dense"
        q_out = quantized_output(out_tensor, label)
        if q_out is None:
            raise ValueError(f"{label}: output {out_tensor} is not quantized by a single QuantizeLinear")
        y_scale, y_zp, y_name = q_out
        shapes[y_name] = out_shape
        layers.append(_finish_weight_layer(kind, label, x_q, x_scale, x_zp, y_name, y_scale, y_zp,
                                           weight, scales, bias, geometry))

    def parse_qdq_activation(node):
        label = node_label(node)
        x_q, x_scale, x_zp = dq_activation(node.input[0], label)
        q_out = quantized_output(node.output[0], label)
        if q_out is None:
            raise ValueError(f"{label}: {node.op_type} output is not quantized by a single QuantizeLinear")
        y_scale, y_zp, y_name = q_out
        same_shape(y_name, x_q)
        layers.append(OnnxQuantizedLayer(op_type=node.op_type, name=label, input_name=x_q,
                                         input_scale=x_scale, input_zp=x_zp, output_name=y_name,
                                         output_scale=y_scale, output_zp=y_zp))

    # ---------------------------------------------------------- QOperator
    def qop_activation_params(node, s_idx, z_idx, label):
        return scalar_scale(node.input[s_idx], label), int(zero_point(node.input[z_idx], label)[0])

    def parse_qop_weight_node(node):
        label = node_label(node)
        if node.op_type == "QLinearConv":
            x, xs, xz, w, ws, wz, ys, yz = node.input[:8]
            b = node.input[8] if len(node.input) > 8 and node.input[8] else None
        elif node.op_type == "QGemm":
            x, xs, xz, w, ws, wz = node.input[:6]
            b = node.input[6] if len(node.input) > 6 and node.input[6] else None
            if len(node.input) < 9 or not node.input[7]:
                raise ValueError(f"{label}: QGemm with a float output is not supported")
            ys, yz = node.input[7], node.input[8]
        else:  # QLinearMatMul
            x, xs, xz, w, ws, wz, ys, yz = node.input[:8]
            b = None
        if w not in inits:
            raise ValueError(f"{label}: weight input {w} is not a constant")
        x_scale = scalar_scale(xs, label)
        x_zp = int(zero_point(xz, label)[0])
        w_q = np.asarray(inits[w])
        if w_q.dtype != np.int8:
            raise ValueError(f"{label}: weights must be int8, got {w_q.dtype}")
        check_weight_zero_points(zero_point(wz, label), label)
        w_scales = np.asarray(inits[ws], dtype=np.float64).reshape(-1)
        y_scale = scalar_scale(ys, label)
        y_zp = int(zero_point(yz, label)[0])
        y_name = node.output[0]
        a = attrs(node)
        if node.op_type == "QLinearConv":
            geometry, out_shape = conv_geometry(node, x, w_q)
            scales = weight_scales_for(w_scales, 0, w_q.shape[0], 0, label)
            bias = None
            if b is not None:
                bq = np.asarray(inits[b], dtype=np.float64).reshape(-1)
                bias = bq * x_scale * np.asarray(scales if len(scales) > 1 else scales * len(bq))
            weight = oihw_to_ohwi(w_q)
            kind = "Conv"
        else:
            if node.op_type == "QGemm":
                if a.get("transA", 0) != 0 or a.get("alpha", 1.0) != 1.0:
                    raise ValueError(f"{label}: QGemm with transA/alpha other than 0/1 is not supported")
                trans_b = a.get("transB", 0)
            else:
                trans_b = 0
            if trans_b:
                w_out_in, out_axis = w_q, 0
            else:
                w_out_in, out_axis = w_q.T, 1
            n_out = w_out_in.shape[0]
            # QOperator weight scales carry no axis attribute; per-channel
            # scales always run along the output channels.
            scales = weight_scales_for(w_scales, out_axis, n_out, out_axis, label)
            weight = dense_from_out_in(w_out_in, x, label)
            channel_scales = np.asarray(scales if len(scales) > 1 else scales * n_out)
            bias = None
            if b is not None:
                bias = np.asarray(inits[b], dtype=np.float64).reshape(-1) * x_scale * channel_scales
            if node.op_type == "QLinearMatMul":
                users = consumers.get(y_name, [])
                if len(users) == 1 and users[0].op_type == "QLinearAdd":
                    add = users[0]
                    # QLinearAdd(A, As, Az, B, Bs, Bz, Cs, Cz)
                    if add.input[0] == y_name:
                        c_q, c_s, c_z = add.input[3], add.input[4], add.input[5]
                    else:
                        c_q, c_s, c_z = add.input[0], add.input[1], add.input[2]
                    if c_q not in inits:
                        raise ValueError(f"{label}: QLinearAdd after QLinearMatMul is not a constant bias")
                    bias = ((np.asarray(inits[c_q], dtype=np.float64).reshape(-1)
                             - zero_point(c_z, label)[0]) * scalar_scale(c_s, label))
                    y_scale = scalar_scale(add.input[6], label)
                    y_zp = int(zero_point(add.input[7], label)[0])
                    y_name = add.output[0]
            geometry = {"NumInputs": weight.shape[1], "NumOutputs": weight.shape[0]}
            out_shape = [shapes.get(x, [1])[0], weight.shape[0]]
            kind = "Dense"
        shapes[y_name] = out_shape
        layers.append(_finish_weight_layer(kind, label, x, x_scale, x_zp, y_name, y_scale, y_zp,
                                           weight, scales, bias, geometry))
        return y_name

    # ------------------------------------------------------------- walk
    for node in graph.node:
        if id(node) in skip_nodes:
            continue
        op = node.op_type
        if op in ("QuantizeLinear", "DequantizeLinear"):
            same_shape(node.output[0], node.input[0])
            continue
        if op == "Constant":
            continue
        if op in ("Identity",):
            same_shape(node.output[0], node.input[0])
            continue
        if op == "Flatten":
            if attrs(node).get("axis", 1) != 1:
                raise ValueError(f"{node_label(node)}: Flatten with axis != 1 is not supported")
            flatten_to_2d(node.output[0], node.input[0])
            continue
        if op == "Reshape":
            target = np.asarray(inits.get(node.input[1], [])).reshape(-1)
            if target.size != 2:
                raise ValueError(f"{node_label(node)}: only a Reshape to 2D (a flatten) is supported")
            flatten_to_2d(node.output[0], node.input[0])
            continue
        if op in ("Conv", "Gemm", "MatMul"):
            parse_qdq_weight_node(node)
            continue
        if op in ("QLinearConv", "QGemm", "QLinearMatMul"):
            parse_qop_weight_node(node)
            if op == "QLinearMatMul":
                users = consumers.get(node.output[0], [])
                if len(users) == 1 and users[0].op_type == "QLinearAdd":
                    skip_nodes.add(id(users[0]))
            continue
        if op in ("Relu", "Sigmoid", "Tanh", "Softmax"):
            parse_qdq_activation(node)
            continue
        if op == "QLinearSigmoid":
            label = node_label(node)
            xs, xz = qop_activation_params(node, 1, 2, label)
            ys, yz = qop_activation_params(node, 3, 4, label)
            same_shape(node.output[0], node.input[0])
            layers.append(OnnxQuantizedLayer(op_type="Sigmoid", name=label, input_name=node.input[0],
                                             input_scale=xs, input_zp=xz, output_name=node.output[0],
                                             output_scale=ys, output_zp=yz))
            continue
        raise ValueError(f"unsupported ONNX op {op} (node {node.name or node.output[0]}); "
                         "tinymind_import_onnx maps Conv, Gemm, MatMul(+Add), Relu, Sigmoid, Tanh, "
                         "Softmax, Flatten, Reshape-to-2D and their QOperator forms")
    return layers


# ----------------------------------------------------------------------------
# Everything below is new in tinymind-rs: the chain check and the Rust back end.
# ----------------------------------------------------------------------------

import ctypes  # noqa: E402
import ctypes.util  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

# ONNX ops that move a tensor without changing its values: a layer reading
# their output reads the tensor they were given.
_PASS_THROUGH = ("QuantizeLinear", "DequantizeLinear", "Identity", "Flatten", "Reshape")


def _f32(x: float) -> float:
    """Round to float32, as a C++ `float` constant or a Rust `f32` stores it."""
    import numpy as np

    return float(np.float32(x))


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


def build_lut(kind: str, input_scale: float, input_zero_point: int,
              output_scale: float, output_zero_point: int) -> list[int]:
    """tinymind (C++)'s buildQSigmoidLUT / buildQTanhLUT, step for step in
    float32 with the C library's expf / tanhf (as tools/import_pytorch does)."""
    import numpy as np

    if _EXPF is None or _TANHF is None:
        print("tinymind_import_onnx: no C expf/tanhf found; activation tables use numpy's "
              "float32 math and may differ from tables built on the device in rare entries",
              file=sys.stderr)
    expf = (lambda v: np.float32(_EXPF(float(v)))) if _EXPF else (lambda v: np.exp(np.float32(v)))
    tanhf = (lambda v: np.float32(_TANHF(float(v)))) if _TANHF else (lambda v: np.tanh(np.float32(v)))
    one = np.float32(1.0)
    in_s, out_s = np.float32(input_scale), np.float32(output_scale)
    lut = []
    for idx in range(256):
        x_f = in_s * np.float32((idx - 128) - input_zero_point)
        y_f = one / (one + expf(-x_f)) if kind == "Sigmoid" else tanhf(x_f)
        y_q = np.float32(output_zero_point) + y_f / out_s
        lut.append(max(-128, min(127, lround(float(y_q)))))
    return lut


def build_softmax(input_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    """tinymind (C++)'s buildQSoftmaxFixedPointParams on a float32 input scale:
    (input_multiplier, input_left_shift, diff_min)."""
    input_integer_bits = 5
    real_multiplier = min(beta * _f32(input_scale) * float(1 << (31 - input_integer_bits)),
                          float((1 << 31) - 1))
    multiplier, shift = quantize_multiplier(real_multiplier)
    left_shift = -shift
    max_input_rescaled = (float((1 << input_integer_bits) - 1)
                          * float(1 << (31 - input_integer_bits))
                          / math.ldexp(1.0, left_shift))
    return multiplier, left_shift, -int(math.floor(max_input_rescaled))


def check_chain(model_path: str, layers: list[OnnxQuantizedLayer]) -> None:
    """Raise ValueError unless the layers form one chain from the model's only
    input to its only output, each reading the tensor the layer before it
    wrote (through Flatten / Reshape / Identity / Q / DQ, which move no values)
    in the grid that layer wrote it in."""
    import numpy as np
    import onnx

    graph = onnx.load(model_path).graph
    inits = {i.name for i in graph.initializer}
    inputs = [i.name for i in graph.input if i.name not in inits]
    outputs = [o.name for o in graph.output]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"only models with one input and one output are supported "
                         f"(inputs {inputs}, outputs {outputs})")
    source = {node.output[0]: node.input[0] for node in graph.node if node.op_type in _PASS_THROUGH}

    def origin(tensor):
        while tensor in source:
            tensor = source[tensor]
        return tensor

    if not layers:
        raise ValueError(f"{model_path}: no quantized layers found")
    if origin(layers[0].input_name) != inputs[0]:
        raise ValueError(f"{layers[0].name}: the first layer does not read the model's input")
    for prev, layer in zip(layers, layers[1:]):
        if origin(layer.input_name) != origin(prev.output_name):
            raise ValueError(f"{layer.name}: reads {layer.input_name}, not the output of {prev.name}; "
                             "only a chain of layers is supported")
        if (np.float32(layer.input_scale) != np.float32(prev.output_scale)
                or layer.input_zp != prev.output_zp):
            raise ValueError(f"{layer.name}: reads {prev.name}'s output in another grid "
                             f"(scale {layer.input_scale} zero point {layer.input_zp}, written as "
                             f"{prev.output_scale} / {prev.output_zp}); the requantization between "
                             "them is not supported")
    if origin(outputs[0]) != origin(layers[-1].output_name):
        raise ValueError(f"the model's output is not the last layer's ({layers[-1].name})")


def _rust_nested(a) -> str:
    if a.ndim == 1:
        return "[" + ", ".join(str(int(v)) for v in a.tolist()) + "]"
    return "[" + ", ".join(_rust_nested(sub) for sub in a) + "]"


def _rust_type(shape, elem: str = "i8") -> str:
    t = elem
    for d in reversed(shape):
        t = f"[{t}; {d}]"
    return t


def _f32_literal(x: float) -> str:
    """The shortest decimal that round-trips the float32 value."""
    import numpy as np

    s = np.format_float_positional(np.float32(x), unique=True, trim="0")
    return s + "0" if s.endswith(".") else s


def _rq_literal(multiplier: int, shift: int, zero_point: int) -> str:
    return (f"Requantizer {{ multiplier: {multiplier}, shift: {shift}, zero_point: {zero_point}, "
            "qmin: -128, qmax: 127 }")


def _value(ref: str) -> str:
    """The value behind a reference expression `&v` (arrays are Copy), or a
    reference that auto-derefs (`input`)."""
    return ref[1:] if ref.startswith("&") else ref


def _shape_text(shape) -> str:
    return " x ".join(str(d) for d in shape[1:])


def emit_rust_module(path: str, name: str, layers: list[OnnxQuantizedLayer],
                     model_note: str = "") -> None:
    """Write the Rust module for a chain of parsed layers (see check_chain):
    statics for every layer and `forward(&Input) -> Output`."""
    import numpy as np

    decls: list[str] = []
    body: list[str] = []
    shape = None  # ("map", h, w, c) or ("vec", n): what the last step wrote
    cur = "input"  # a reference to it
    out_scale, out_zp = layers[-1].output_scale, layers[-1].output_zp
    in_shape = None
    for i, layer in enumerate(layers):
        g = layer.geometry
        what = f"`{layer.name}` ({layer.op_type})"
        if layer.op_type == "Conv":
            h, w, c = g["InputHeight"], g["InputWidth"], g["InputChannels"]
            kh, kw, sh, sw = g["KernelHeight"], g["KernelWidth"], g["StrideH"], g["StrideW"]
            f, oh, ow = g["NumFilters"], g["OutputHeight"], g["OutputWidth"]
            pt, pb, pl, pr = g["PadTop"], g["PadBottom"], g["PadLeft"], g["PadRight"]
            if shape is None:
                shape = in_shape = ("map", h, w, c)
            if shape != ("map", h, w, c):
                raise ValueError(f"{layer.name}: a convolution over {shape}, expected {h}x{w}x{c}")
            pointwise = kh == kw == 1 and sh == sw == 1 and pt == pb == pl == pr == 0
            tag = ("PW" if pointwise else "CONV") + str(i)
            decls.append(f"/// {what}: {kh}x{kw} {'pointwise ' if pointwise else ''}convolution, stride "
                         f"{sh}x{sw}, padding {pt}/{pb}/{pl}/{pr} (top/bottom/left/right), "
                         f"{h}x{w}x{c} -> {oh}x{ow}x{f}.")
            if pointwise:
                decls.append(f"static {tag}_WEIGHTS: {_rust_type((f, c))} = "
                             f"{_rust_nested(layer.weight[:, 0, 0, :])};")
            else:
                decls.append(f"static {tag}_WEIGHTS: {_rust_type((f, kh, kw, c))} = "
                             f"{_rust_nested(layer.weight)};")
            n_out = f
            dims = (f"{c}, {f}" if pointwise else
                    f"{c}, {kh}, {kw}, {f}, {sh}, {sw}, {pt}, {pb}, {pl}, {pr}")
            ty = "QPointwiseConv2D" if pointwise else "QConv2D"
            new_shape = ("map", oh, ow, f)
        elif layer.op_type == "Dense":
            n_in, n_out = g["NumInputs"], g["NumOutputs"]
            if shape is None:
                shape = in_shape = ("vec", n_in)
            if shape[0] == "map" and shape[1] * shape[2] * shape[3] == n_in:
                # Weight columns are in NHWC order already (parse_onnx_model).
                cur = (f"{_value(cur)}.as_flattened().as_flattened()"
                       f".first_chunk::<{n_in}>().expect(\"{n_in} values\")")
            elif shape != ("vec", n_in):
                raise ValueError(f"{layer.name}: a dense layer with {n_in} inputs reads {shape}")
            tag = f"FC{i}"
            decls.append(f"/// {what}: fully connected, {n_in} -> {n_out}.")
            decls.append(f"static {tag}_WEIGHTS: {_rust_type((n_out, n_in))} = {_rust_nested(layer.weight)};")
            dims = f"{n_in}, {n_out}"
            ty = "QDense"
            new_shape = ("vec", n_out)
        else:
            if shape is None:
                raise ValueError(f"{layer.name}: the model must start with a convolution or a dense layer")
            new_shape = shape
            if layer.op_type in ("Sigmoid", "Tanh"):
                tag = f"{layer.op_type.upper()}{i}_LUT"
                lut = build_lut(layer.op_type, layer.input_scale, layer.input_zp,
                                layer.output_scale, layer.output_zp)
                decls.append(f"/// {what}: {layer.op_type.lower()} as a table from the input grid "
                             f"(scale {_f32_literal(layer.input_scale)}, zero point {layer.input_zp}) to "
                             f"the output's ({_f32_literal(layer.output_scale)}, {layer.output_zp}).")
                decls.append(f"static {tag}: [i8; LUT_SIZE] = {_rust_nested(np.array(lut))};\n")
                fn = f"apply_lut(x, &{tag})"
            elif layer.op_type == "Relu":
                if (np.float32(layer.input_scale) != np.float32(layer.output_scale)
                        or layer.input_zp != layer.output_zp):
                    raise ValueError(f"{layer.name}: a Relu that requantizes is not supported")
                fn = f"qrelu(x, {layer.input_zp})"
            elif layer.op_type == "Softmax":
                if shape[0] != "vec":
                    raise ValueError(f"{layer.name}: only a softmax over a vector is supported")
                if shape[1] >= 512:
                    raise ValueError(f"{layer.name}: QSoftmax takes at most 511 logits")
                tag = f"SOFTMAX{i}"
                mult, left, diff_min = build_softmax(layer.input_scale)
                decls.append(f"/// {what}: softmax over {shape[1]} logits, output on the 1/256 grid.")
                decls.append(f"static {tag}: QSoftmax = QSoftmax {{ input_multiplier: {mult}, "
                             f"input_left_shift: {left}, diff_min: {diff_min} }};\n")
                if i != len(layers) - 1:
                    raise ValueError(f"{layer.name}: a softmax is supported only as the last layer")
                out_scale, out_zp = 1.0 / 256.0, -128
                body.append(f"    let v{i} = {tag}.forward({cur});")
                cur, shape = f"&v{i}", new_shape
                continue
            else:
                raise ValueError(f"{layer.name}: no Rust back end for {layer.op_type}")
            if shape[0] == "vec":
                body.append(f"    let v{i} = {_value(cur)}.map(|x| {fn});")
            else:
                body.append(f"    let v{i} = {_value(cur)}.map(|row| row.map(|pixel| pixel.map(|x| {fn})));")
            cur, shape = f"&v{i}", new_shape
            continue

        # A weight layer (convolution or dense).
        biases = "None"
        if layer.bias is not None:
            decls.append(f"static {tag}_BIASES: [i32; {n_out}] = {_rust_nested(layer.bias)};")
            biases = f"Some(&{tag}_BIASES)"
        if layer.per_channel:
            ty += "PerChannel"
            rqs = ", ".join(_rq_literal(m, s, layer.output_zp)
                            for m, s in zip(layer.multipliers, layer.shifts))
            decls.append(f"static {tag}_REQUANTIZERS: [Requantizer; {n_out}] = [{rqs}];")
            rq_field = f"requantizers: &{tag}_REQUANTIZERS"
        else:
            rq_field = f"requantizer: {_rq_literal(layer.multipliers[0], layer.shifts[0], layer.output_zp)}"
        decls.append(f"static {tag}: {ty}<'static, {dims}> = {ty} {{ weights: &{tag}_WEIGHTS, "
                     f"biases: {biases}, input_zero_point: {layer.input_zp}, {rq_field} }};\n")
        if new_shape[0] == "map":
            body.append(f"    let mut v{i} = {_rust_type(new_shape[1:]).replace('[i8;', '[0i8;')};")
            body.append(f"    {tag}.forward({cur}, &mut v{i});")
        else:
            body.append(f"    let v{i} = {tag}.forward({cur});")
        cur, shape = f"&v{i}", new_shape

    # The last value is the function's value (clippy::let_and_return).
    last = cur[1:]
    if body[-1].startswith(f"    let {last} = ") and body[-1].endswith(";"):
        body[-1] = "    " + body[-1][len(f"    let {last} = "):-1]
    else:
        body.append(f"    {last}")

    lines = [f"//! `{name}`: generated by tinymind-rs tools/import_onnx/tinymind_import_onnx.py",
             "//! from an int8 ONNX model quantized by ONNX Runtime. Do not edit; regenerate",
             "//! from the model instead."]
    if model_note:
        lines += [f"//! {line}".rstrip() for line in model_note.splitlines()]
    lines += ["//!",
              "//! Integer-only and `no_std`: int8 input in the `INPUT` grid, int8 output in the",
              "//! `OUTPUT` grid (real value = `SCALE * (q - ZERO_POINT)`). Feature maps are NHWC",
              "//! without the batch axis, `[row][column][channel]`: the transpose of the NCHW",
              "//! tensor the ONNX model takes.",
              "",
              "// Not every program uses every constant.",
              "#![allow(dead_code)]",
              "",
              "use tinymind::int8::*;",
              ""]
    if in_shape[0] == "map":
        lines.append(f"/// The input: {_shape_text(in_shape)} (height x width x channels, NHWC).")
    else:
        lines.append("/// The input vector.")
    lines.append(f"pub type Input = {_rust_type(in_shape[1:])};")
    lines.append("/// The output vector." if shape[0] == "vec" else
                 f"/// The output: {_shape_text(shape)} (NHWC).")
    lines.append(f"pub type Output = {_rust_type(shape[1:])};")
    softmax_note = (" The last layer is a softmax: its output is on TFLite's\n/// probability grid, "
                    "1/256, not on the 1/255 grid ONNX Runtime gives it."
                    if layers[-1].op_type == "Softmax" else "")
    lines += [f"/// Scale of the input tensor.\npub const INPUT_SCALE: f32 = {_f32_literal(layers[0].input_scale)};",
              f"/// Zero point of the input tensor.\npub const INPUT_ZERO_POINT: i8 = {layers[0].input_zp};",
              f"/// Scale of the output tensor.{softmax_note}\npub const OUTPUT_SCALE: f32 = {_f32_literal(out_scale)};",
              f"/// Zero point of the output tensor.\npub const OUTPUT_ZERO_POINT: i8 = {out_zp};",
              ""]
    lines += decls
    lines += ["/// Run the network on an int8 input in the `INPUT` grid.",
              "pub fn forward(input: &Input) -> Output {"] + body + ["}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    rustfmt = shutil.which("rustfmt")
    if rustfmt:
        subprocess.run([rustfmt, "--edition", "2024", path], check=True)
    else:
        print("tinymind_import_onnx: rustfmt not found; the module is valid but unformatted",
              file=sys.stderr)


def import_onnx_model(model_path: str, output_path: str, name: str,
                      model_note: str = "") -> list[OnnxQuantizedLayer]:
    """Top level: parse the quantized ONNX model, check that its layers form a
    chain, and write the Rust module. Returns the parsed layers."""
    layers = parse_onnx_model(model_path)
    check_chain(model_path, layers)
    emit_rust_module(output_path, name, layers, model_note)
    return layers


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: tinymind_import_onnx.py <model.onnx> <out.rs> <name>")
    parsed = import_onnx_model(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"wrote {sys.argv[2]}: {len(parsed)} layers "
          f"({', '.join(layer.op_type for layer in parsed)}) from {os.path.basename(sys.argv[1])}")
