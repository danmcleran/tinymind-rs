#!/usr/bin/env python3
"""The TFLite translator's PAD, MUL and CONCATENATION against TFLite itself.

  python3 -I -B tools/import_tflite/test_ops.py      (needs numpy, tflite, ai-edge-litert, cargo)

The MLPerf models use none of these operators, so small int8 models are built
here (tflite_build.py: deterministic weights, scales calibrated from a float
pass) that do, alongside convolutions with SAME and VALID padding and fused
ReLUs. Each model runs in LiteRT with its reference kernels (TFLite's
`BUILTIN_REF` resolver, not the optimized or XNNPACK ones) on 16 random int8
inputs, and the translated Rust module, built with cargo against this
repository's crate, must give the same bytes for every output.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlperf  # noqa: E402
import tflite_build as tb  # noqa: E402
import tinymind_import_tflite as tt  # noqa: E402
from ai_edge_litert.interpreter import Interpreter, OpResolverType  # noqa: E402


def pad_conv():
    """PAD on three sides, then a VALID convolution with a fused ReLU."""
    g = tb.Graph(11, (1, 5, 6, 3), 0.05, -9)
    x = g.pad(g.input, 1, 2, 2, 0)
    return g, g.conv(x, 4, 3, padding="valid", act="relu")


def gated():
    """Squeeze-and-excite style gating: two convolutions of one input,
    multiplied, then multiplied again with a fused ReLU."""
    g = tb.Graph(12, (1, 4, 4, 6), 0.04, 13)
    a = g.conv(g.input, 6, 1)
    b = g.conv(g.input, 6, 3)
    m = g.mul(a, b)
    return g, g.mul(m, a, act="relu")


def branches():
    """Two branches joined along channels (Inception style), padded, then
    convolved."""
    g = tb.Graph(13, (1, 4, 5, 3), 0.06, 3)
    a = g.conv(g.input, 2, 1, act="relu")
    b = g.conv(g.input, 5, 3)
    g.share(a, b)
    c = g.concat(a, b)
    p = g.pad(c, 1, 1, 1, 1)
    return g, g.conv(p, 4, 3, padding="valid")


MODELS = [("pad_conv", pad_conv), ("gated", gated), ("branches", branches)]


def reference(model: bytes, xs: np.ndarray) -> bytes:
    it = Interpreter(model_content=model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    out = []
    for x in xs:
        it.set_tensor(i, x[None])
        it.invoke()
        out.append(it.get_tensor(o).copy())
    return np.array(out, np.int8).tobytes()


def main():
    target = os.path.join(mlperf.ROOT, "target", "tflite-ops")
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, build in MODELS:
            g, out = build()
            model = g.build(out)
            path = os.path.join(tmp, name + ".tflite")
            with open(path, "wb") as f:
                f.write(model)
            want = reference(model, g.samples)
            module = os.path.join(tmp, name + ".rs")
            tr = tt.translate(path, module, name)
            got = mlperf.run_module(module, g.samples, tr.input.shape[1:], target,
                                    output_map=len(g.tensors[out].shape) == 4)
            spread = len(set(np.frombuffer(want, np.int8).tolist()))
            ok = got == want
            bad += not ok
            print(f"{name}: {len(tr.ops)} operators ({', '.join(sorted(set(tr.ops)))}), "
                  f"{len(want)} outputs, {spread} distinct values: "
                  f"{'IDENTICAL' if ok else 'DIFFERENT'} to TFLite's reference kernels")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
