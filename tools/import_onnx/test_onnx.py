#!/usr/bin/env python3
"""Check the ONNX importer's modules against ONNX Runtime, and its refusals.

  python3 -I tools/import_onnx/test_onnx.py      (needs numpy, onnx, onnxruntime, cargo)

scripts/compare-with-cpp.sh shows the generated modules compute what
tinymind (C++) computes from its own importer's output. This checks what both
are meant to compute: for each example model, the Rust module's int8 outputs
(examples/imported_onnx.rs on its 16 test inputs) against ONNX Runtime running
the quantized model on the same inputs. ONNX Runtime rounds a few results
differently (it requantizes in float), so a layer's output may differ by one
step; the outputs must be within one step of ONNX Runtime's, and most equal.
A softmax output is compared as a probability: the module's is on TFLite's
1/256 grid and ONNX Runtime's on 1/255, and the softmax kernels differ, so
within 2/256.

It also checks that the importer refuses what it cannot translate exactly:
a model whose layers do not form a chain, and one that requantizes a tensor
between two layers.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tinymind_import_onnx as ti  # noqa: E402

SAMPLES = 16


class Lcg:
    """examples/common/imported_onnx_driver.rs's generator."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state >> 24


def inputs(rng, n, zp):
    out = []
    for s in range(SAMPLES):
        v = []
        for _ in range(n):
            r = rng.next()
            v.append(r - 128 if s % 2 == 0 else max(-128, min(127, zp + r % 97 - 48)))
        out.append(np.array(v, dtype=np.int64))
    return out


def rust_outputs():
    out = subprocess.run(["cargo", "run", "-q", "--release", "--example", "imported_onnx"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    got = {}
    for line in out.splitlines():
        name, rest = line.split(" ", 1)
        if rest.startswith("input"):
            continue
        got.setdefault(name, []).append(np.array([int(v) for v in rest.split(":")[1].split()]))
    return got


def check_against_ort():
    got = rust_outputs()
    rng = Lcg(1)
    worst = 0
    for model in ("cnn", "cnn_qop", "mlp"):
        path = os.path.join(HERE, "models", f"{model}.onnx")
        layers = ti.parse_onnx_model(path)
        first, last = layers[0], layers[-1]
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
        n = int(np.prod(shape))
        exact = total = 0
        for s, q_nhwc in enumerate(inputs(rng, n, first.input_zp)):
            if len(shape) == 4:
                _, c, h, w = shape
                q = q_nhwc.reshape(1, h, w, c).transpose(0, 3, 1, 2)
            else:
                q = q_nhwc.reshape(shape)
            x = ((q - first.input_zp) * np.float32(first.input_scale)).astype(np.float32)
            y = sess.run(None, {meta.name: x})[0].reshape(-1)
            mine = got[f"onnx_{model}"][s]
            if last.op_type == "Softmax":
                diff = np.abs((mine + 128) / 256.0 - y)
                assert diff.max() <= 2 / 256, f"{model} sample {s}: {diff.max()} from ONNX Runtime"
                worst = max(worst, diff.max() * 256)
            else:
                ref = np.rint(y / np.float32(last.output_scale)).astype(np.int64) + last.output_zp
                diff = np.abs(mine - ref)
                assert diff.max() <= 1, f"{model} sample {s}: {mine} vs ONNX Runtime {ref}"
                exact += int((diff == 0).sum())
                total += diff.size
        assert len({tuple(v) for v in got[f"onnx_{model}"]}) > SAMPLES // 2, f"{model}: outputs barely vary"
        if total:
            assert exact / total > 0.9, f"{model}: only {exact}/{total} outputs equal ONNX Runtime's"
            print(f"  {model}: within 1 step of ONNX Runtime, {exact}/{total} equal")
        else:
            print(f"  {model}: probabilities within 2/256 of ONNX Runtime's")
    print(f"  softmax outputs: worst {worst:.2f}/256 from ONNX Runtime's")


def refuses(path, needle):
    try:
        ti.import_onnx_model(path, os.devnull, "x")
    except ValueError as e:
        assert needle in str(e), f"refused for another reason: {e}"
        return
    raise AssertionError(f"{path}: not refused (expected {needle!r})")


def check_refusals(tmp):
    model = onnx.load(os.path.join(HERE, "models", "mlp.onnx"))
    g = model.graph

    # Requantize the tanh's output (Q -> DQ -> Q at another scale -> DQ)
    # before the dense layer that reads it.
    gemm = next(n for n in g.node if n.op_type == "Gemm")
    dq = next(n for n in g.node if n.output[0] == gemm.input[0])
    scale = numpy_helper.to_array(next(i for i in g.initializer if i.name == dq.input[1]))
    g.initializer.append(numpy_helper.from_array(np.array(scale * 1.5, dtype=np.float32), "requant_scale"))
    g.node.extend([
        helper.make_node("QuantizeLinear", [dq.output[0], "requant_scale", dq.input[2]], ["requant_q"]),
        helper.make_node("DequantizeLinear", ["requant_q", "requant_scale", dq.input[2]], ["requant_dq"]),
    ])
    gemm.input[0] = "requant_dq"
    sorted_nodes = _toposort(g)
    del g.node[:]
    g.node.extend(sorted_nodes)
    path = os.path.join(tmp, "requant.onnx")
    onnx.save(model, path)
    refuses(path, "another grid")

    # A second output: the dense layer's input leaves the graph too.
    model = onnx.load(os.path.join(HERE, "models", "mlp.onnx"))
    model.graph.output.append(helper.make_tensor_value_info(
        next(n for n in model.graph.node if n.op_type == "Tanh").output[0], onnx.TensorProto.FLOAT, None))
    path = os.path.join(tmp, "two_outputs.onnx")
    onnx.save(model, path)
    refuses(path, "one input and one output")
    print("  refusals: a requantized tensor between layers, a second output")


def _toposort(g):
    ready = {i.name for i in g.input} | {i.name for i in g.initializer} | {""}
    nodes, out = list(g.node), []
    while nodes:
        for n in nodes:
            if all(i in ready for i in n.input):
                out.append(n)
                ready.update(n.output)
                nodes.remove(n)
                break
        else:
            raise ValueError("graph has a cycle")
    return out


def main():
    print("ONNX importer vs ONNX Runtime:")
    check_against_ort()
    with tempfile.TemporaryDirectory() as tmp:
        check_refusals(tmp)
    print("ok")


if __name__ == "__main__":
    main()
