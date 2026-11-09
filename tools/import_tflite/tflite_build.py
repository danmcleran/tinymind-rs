"""Build small int8 TFLite models for testing the translator, without
TensorFlow: the flatbuffer is written with the `tflite` package's schema
builders.

A `Graph` holds tensors and operators; `conv`, `mul`, `pad` and `concat` add a
layer with deterministic random int8 weights and int32 biases, and calibrate
each new activation's scale from a float forward pass over sample inputs, so
outputs spread over the int8 range rather than saturating; a fused ReLU's
output covers the values before the ReLU, so the ReLU clips. `build()` returns
the `.tflite` bytes. Only what the tests need is supported.
"""

import flatbuffers
import numpy as np
import tflite

_INT8 = tflite.TensorType.INT8
_INT32 = tflite.TensorType.INT32
_ACT = {"none": tflite.ActivationFunctionType.NONE, "relu": tflite.ActivationFunctionType.RELU}


class _Tensor:
    def __init__(self, shape, dtype, scales=None, zps=None, data=None, qdim=0, name=""):
        self.shape, self.dtype, self.scales, self.zps = list(shape), dtype, scales, zps
        self.data, self.qdim, self.name = data, qdim, name


class Graph:
    def __init__(self, seed: int, input_shape, input_scale: float, input_zp: int, samples: int = 16):
        self.rng = np.random.default_rng(seed)
        self.tensors: list[_Tensor] = []
        self.ops: list[tuple] = []
        self.opcodes: list[int] = []
        # Float values of every activation over the sample inputs, for calibration.
        self.values: dict[int, np.ndarray] = {}
        x = self.rng.integers(-128, 128, (samples,) + tuple(input_shape[1:])).astype(np.int8)
        self.samples = x
        self.input = self._add(_Tensor(input_shape, _INT8, [input_scale], [input_zp], name="input"))
        self.values[self.input] = (x.astype(np.float64) - input_zp) * np.float32(input_scale)

    def _add(self, t: _Tensor) -> int:
        self.tensors.append(t)
        return len(self.tensors) - 1

    def _activation(self, value: np.ndarray, name: str, cover=None) -> int:
        """A new int8 activation tensor whose scale covers `cover` (default
        `value`; float, with the batch of samples first)."""
        cover = value if cover is None else cover
        lo, hi = min(float(cover.min()), 0.0), max(float(cover.max()), 0.0)
        scale = np.float32(max(hi - lo, 1e-6) / 255.0)
        zp = int(np.clip(round(-128 - lo / float(scale)), -128, 127))
        t = self._add(_Tensor((1,) + value.shape[1:], _INT8, [float(scale)], [zp], name=name))
        self.values[t] = value
        return t

    def _q(self, t: int) -> tuple:
        return np.float32(self.tensors[t].scales[0]), self.tensors[t].zps[0]

    def _real(self, t: int) -> np.ndarray:
        """`values[t]` rounded onto t's int8 grid and back: what the next
        layer sees."""
        s, z = self._q(t)
        q = np.clip(np.round(self.values[t] / float(s)) + z, -128, 127)
        return (q - z) * float(s)

    def _op(self, code: int, inputs, outputs, options_type: int, options: dict):
        if code not in self.opcodes:
            self.opcodes.append(code)
        self.ops.append((self.opcodes.index(code), list(inputs), list(outputs), options_type, options))

    def conv(self, x: int, filters: int, k: int, padding: str = "same", act: str = "none") -> int:
        """CONV_2D, stride 1, per-channel weights."""
        _, H, W, C = self.tensors[x].shape
        w = self.rng.integers(-127, 128, (filters, k, k, C)).astype(np.int8)
        ws = (self.rng.uniform(0.005, 0.03, filters)).astype(np.float32)
        s_in, _ = self._q(x)
        b = self.rng.integers(-2000, 2000, filters).astype(np.int32)
        bs = (np.float64(s_in) * ws.astype(np.float64)).astype(np.float32)
        wt = self._add(_Tensor(w.shape, _INT8, [float(v) for v in ws], [0] * filters, w, 0, "w"))
        bt = self._add(_Tensor(b.shape, _INT32, [float(v) for v in bs], [0] * filters, b, 0, "b"))
        xr = self._real(x)
        if padding == "same":
            p = (k - 1) // 2
            xr = np.pad(xr, ((0, 0), (p, k - 1 - p), (p, k - 1 - p), (0, 0)))
        OH, OW = xr.shape[1] - k + 1, xr.shape[2] - k + 1
        wf = w.astype(np.float64) * ws[:, None, None, None]
        y = np.zeros((xr.shape[0], OH, OW, filters))
        for i in range(OH):
            for j in range(OW):
                y[:, i, j, :] = np.einsum("nhwc,fhwc->nf", xr[:, i:i + k, j:j + k, :], wf)
        y += b * bs.astype(np.float64)
        # A fused ReLU's output is calibrated over the values before it, so its
        # zero point lies above -128 and the ReLU clips something (a real
        # converter calibrates after it, where a fused ReLU changes nothing).
        out = self._activation(np.maximum(y, 0) if act == "relu" else y, "conv", cover=y)
        opts = {"padding": tflite.Padding.SAME if padding == "same" else tflite.Padding.VALID,
                "act": _ACT[act]}
        self._op(tflite.BuiltinOperator.CONV_2D, [x, wt, bt], [out], tflite.BuiltinOptions.Conv2DOptions, opts)
        return out

    def mul(self, a: int, b: int, act: str = "none") -> int:
        y = self._real(a) * self._real(b)
        out = self._activation(np.maximum(y, 0) if act == "relu" else y, "mul", cover=y)
        self._op(tflite.BuiltinOperator.MUL, [a, b], [out], tflite.BuiltinOptions.MulOptions, {"act": _ACT[act]})
        return out

    def pad(self, x: int, top: int, bottom: int, left: int, right: int) -> int:
        p = np.array([[0, 0], [top, bottom], [left, right], [0, 0]], np.int32)
        pt = self._add(_Tensor(p.shape, _INT32, None, None, p, name="paddings"))
        s, z = self._q(x)
        y = np.pad(self._real(x), ((0, 0), (top, bottom), (left, right), (0, 0)))
        out = self._add(_Tensor((1,) + y.shape[1:], _INT8, [float(s)], [z], name="pad"))
        self.values[out] = y
        self._op(tflite.BuiltinOperator.PAD, [x, pt], [out], tflite.BuiltinOptions.PadOptions, {})
        return out

    def concat(self, a: int, b: int) -> int:
        """CONCATENATION along channels; `a` and `b` must share one
        quantization, which the output takes."""
        s, z = self._q(a)
        assert self._q(b) == (s, z), "concat inputs must share their quantization"
        y = np.concatenate([self._real(a), self._real(b)], axis=3)
        out = self._add(_Tensor((1,) + y.shape[1:], _INT8, [float(s)], [z], name="concat"))
        self.values[out] = y
        self._op(tflite.BuiltinOperator.CONCATENATION, [a, b], [out],
                 tflite.BuiltinOptions.ConcatenationOptions, {"axis": 3})
        return out

    def share(self, a: int, b: int):
        """Give activations `a` and `b` one quantization covering both (as a
        converter does before concatenating them)."""
        lo = min(float(self.values[a].min()), float(self.values[b].min()), 0.0)
        hi = max(float(self.values[a].max()), float(self.values[b].max()), 0.0)
        scale = float(np.float32(max(hi - lo, 1e-6) / 255.0))
        zp = int(np.clip(round(-128 - lo / scale), -128, 127))
        for t in (a, b):
            self.tensors[t].scales, self.tensors[t].zps = [scale], [zp]

    def build(self, output: int) -> bytes:
        b = flatbuffers.Builder(1024)

        def ivec(values, dtype=np.int32):
            return b.CreateNumpyVector(np.asarray(values, dtype))

        # Buffers: 0 is the empty sentinel; constant tensors get their own.
        buffer_data = [None]
        tensor_buffer = []
        for t in self.tensors:
            if t.data is not None:
                buffer_data.append(np.ascontiguousarray(t.data).astype(t.data.dtype.newbyteorder("<")).tobytes())
                tensor_buffer.append(len(buffer_data) - 1)
            else:
                tensor_buffer.append(0)
        buffers = []
        for data in buffer_data:
            vec = b.CreateNumpyVector(np.frombuffer(data, np.uint8)) if data else None
            tflite.BufferStart(b)
            if vec is not None:
                tflite.BufferAddData(b, vec)
            buffers.append(tflite.BufferEnd(b))

        tensors = []
        for t, buf in zip(self.tensors, tensor_buffer):
            name = b.CreateString(t.name)
            shape = ivec(t.shape)
            quant = None
            if t.scales is not None:
                sv = b.CreateNumpyVector(np.asarray(t.scales, np.float32))
                zv = b.CreateNumpyVector(np.asarray(t.zps, np.int64))
                tflite.QuantizationParametersStart(b)
                tflite.QuantizationParametersAddScale(b, sv)
                tflite.QuantizationParametersAddZeroPoint(b, zv)
                tflite.QuantizationParametersAddQuantizedDimension(b, t.qdim)
                quant = tflite.QuantizationParametersEnd(b)
            tflite.TensorStart(b)
            tflite.TensorAddShape(b, shape)
            tflite.TensorAddType(b, t.dtype)
            tflite.TensorAddBuffer(b, buf)
            tflite.TensorAddName(b, name)
            if quant is not None:
                tflite.TensorAddQuantization(b, quant)
            tensors.append(tflite.TensorEnd(b))

        ops = []
        for opcode, inputs, outputs, otype, o in self.ops:
            if otype == tflite.BuiltinOptions.Conv2DOptions:
                tflite.Conv2DOptionsStart(b)
                tflite.Conv2DOptionsAddPadding(b, o["padding"])
                tflite.Conv2DOptionsAddStrideW(b, 1)
                tflite.Conv2DOptionsAddStrideH(b, 1)
                tflite.Conv2DOptionsAddFusedActivationFunction(b, o["act"])
                tflite.Conv2DOptionsAddDilationWFactor(b, 1)
                tflite.Conv2DOptionsAddDilationHFactor(b, 1)
                opt = tflite.Conv2DOptionsEnd(b)
            elif otype == tflite.BuiltinOptions.MulOptions:
                tflite.MulOptionsStart(b)
                tflite.MulOptionsAddFusedActivationFunction(b, o["act"])
                opt = tflite.MulOptionsEnd(b)
            elif otype == tflite.BuiltinOptions.PadOptions:
                tflite.PadOptionsStart(b)
                opt = tflite.PadOptionsEnd(b)
            else:
                tflite.ConcatenationOptionsStart(b)
                tflite.ConcatenationOptionsAddAxis(b, o["axis"])
                tflite.ConcatenationOptionsAddFusedActivationFunction(b, tflite.ActivationFunctionType.NONE)
                opt = tflite.ConcatenationOptionsEnd(b)
            iv, ov = ivec(inputs), ivec(outputs)
            tflite.OperatorStart(b)
            tflite.OperatorAddOpcodeIndex(b, opcode)
            tflite.OperatorAddInputs(b, iv)
            tflite.OperatorAddOutputs(b, ov)
            tflite.OperatorAddBuiltinOptionsType(b, otype)
            tflite.OperatorAddBuiltinOptions(b, opt)
            ops.append(tflite.OperatorEnd(b))

        def table_vector(items):
            b.StartVector(4, len(items), 4)
            for item in reversed(items):
                b.PrependUOffsetTRelative(item)
            return b.EndVector()

        tv, opv = table_vector(tensors), table_vector(ops)
        inv, outv = ivec([self.input]), ivec([output])
        tflite.SubGraphStart(b)
        tflite.SubGraphAddTensors(b, tv)
        tflite.SubGraphAddInputs(b, inv)
        tflite.SubGraphAddOutputs(b, outv)
        tflite.SubGraphAddOperators(b, opv)
        sub = tflite.SubGraphEnd(b)

        codes = []
        for code in self.opcodes:
            tflite.OperatorCodeStart(b)
            tflite.OperatorCodeAddDeprecatedBuiltinCode(b, min(code, 127))
            tflite.OperatorCodeAddBuiltinCode(b, code)
            tflite.OperatorCodeAddVersion(b, 1)
            codes.append(tflite.OperatorCodeEnd(b))
        cv, sv, bv = table_vector(codes), table_vector([sub]), table_vector(buffers)
        tflite.ModelStart(b)
        tflite.ModelAddVersion(b, 3)
        tflite.ModelAddOperatorCodes(b, cv)
        tflite.ModelAddSubgraphs(b, sv)
        tflite.ModelAddBuffers(b, bv)
        model = tflite.ModelEnd(b)
        b.Finish(model, file_identifier=b"TFL3")
        return bytes(b.Output())
