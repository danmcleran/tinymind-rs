#!/usr/bin/env python3
"""The importer's convolutional examples:
  cnn       12x12x1 -> 3x3 conv (4 filters, padding 1) + batch norm + ReLU ->
            2x2 max pool -> 3x3 depthwise (padding 1) + batch norm + ReLU ->
            1x1 conv 4 -> 8 + ReLU -> global average pool -> dense 8 -> 3 ->
            softmax (each batch norm folded into its convolution);
  cnn_flat  9x11x2 -> 3x2 conv (5 filters, stride 2x1) + ReLU -> 2x2 average pool
            -> flatten -> dense 50 -> 6 + ReLU -> dense 6 -> 2 -> tanh.

  python3 -I tools/import_pytorch/example_cnn.py --rust examples/common/imported_cnn_model.rs
  python3 -I tools/import_pytorch/example_cnn.py --model cnn_flat \\
      --rust examples/common/imported_cnn_flat_model.rs
  python3 -I tools/import_pytorch/example_cnn.py --cpp cnn.hpp      # tinymind's importer

The weights and the calibration set are integer formulas, so they are exact
floats on every platform and the generated file is reproducible. The models
stand in for ones trained in PyTorch; with a real model, use
`tinymind_import.from_torch(model, input_shape=(channels, height, width))`.

--cpp runs tinymind (C++)'s own importer (apps/import_pytorch, from
TINYMIND_DIR, default ../tinymind) on the same models. It has no pooling or
flatten layers, so there each one is folded into the next layer's float
forward: those layers keep their input's scale in this importer, so the
calibrated ranges and the quantized weights are the same.
scripts/compare-with-cpp.sh feeds that header to scripts/cpp/imported_cnn.cpp
on tinymind (C++)'s runtime and byte-compares its output with the Rust example's.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def formula(shape: tuple, layer: int, scale: float = 1.0) -> np.ndarray:
    """Exact float weights in [-scale, scale] from an integer formula."""
    n = int(np.prod(shape))
    v = [((k * 7 + (k // 3) * 11 + layer * 5) % 17 - 8) / 8.0 * scale for k in range(n)]
    return np.array(v, dtype=np.float32).reshape(shape)


def bias(n: int, layer: int) -> np.ndarray:
    return np.array([((o * 3 + layer) % 5 - 2) / 8.0 for o in range(n)], dtype=np.float32)


def dense(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The importer's deterministic dense reference (float64 without BLAS, so
    the calibrated ranges do not depend on the CPU)."""
    return load(os.path.join(HERE, "tinymind_import.py"), "tinymind_import_rs").dense_ref(x, w, b)


def batch_norm(mod, name: str, input_name: str, n: int, layer: int):
    """A BatchNorm2D with exact-float, non-trivial statistics."""
    k = np.arange(n)
    f = lambda v: np.array(v, dtype=np.float32)  # noqa: E731
    return mod.BatchNorm2D(name=name, input_name=input_name,
                           gamma=f(1.0 + ((k * 3 + layer) % 5 - 2) / 4.0),
                           beta=f(((k * 5 + layer) % 7 - 3) / 8.0),
                           mean=f(((k * 2 + layer) % 5 - 2) / 8.0),
                           variance=f(0.5 + ((k + layer) % 4) / 4.0),
                           observer=mod.MinMaxObserver())


def calibration_set(shape: tuple) -> list:
    """32 NHWC inputs in [-1, 1]."""
    h, w, c = shape
    return [np.array([[[((s * 31 + (y * w + x) * 17 + ch * 5) % 41) / 20.0 - 1.0
                        for ch in range(c)] for x in range(w)] for y in range(h)],
                     dtype=np.float32) for s in range(32)]


def test_inputs(shape: tuple) -> list:
    """16 NHWC inputs in [-1, 1], different from the calibration set: what the
    example and scripts/cpp/imported_cnn.cpp run."""
    h, w, c = shape
    return [np.array([[[((s * 23 + (y * w + x) * 29 + ch * 13) % 37) / 18.0 - 1.0
                        for ch in range(c)] for x in range(w)] for y in range(h)],
                     dtype=np.float32) for s in range(16)]


def cnn(mod, tinymind_cpp: bool) -> list:
    conv, pool, gap = mod_ops(mod)
    w1, b1 = formula((4, 3, 3, 1), 1, 0.5), bias(4, 1)
    w2, b2 = formula((4, 3, 3, 1), 2, 0.5), bias(4, 2)
    w3, b3 = formula((8, 1, 1, 4), 3, 0.5), bias(8, 3)
    w4, b4 = formula((3, 8), 4), bias(3, 4)
    f1 = lambda x: conv(x, w1, b1, (1, 1), (1, 1, 1, 1), False)  # noqa: E731
    f2 = lambda x: conv(x, w2, b2, (1, 1), (1, 1, 1, 1), True)  # noqa: E731
    f3 = lambda x: conv(x, w3, b3, (1, 1), (0, 0, 0, 0), False)  # noqa: E731
    f4 = lambda x: dense(x, w4, b4)  # noqa: E731
    obs = mod.MinMaxObserver
    if tinymind_cpp:
        # No pooling layers: the max pool runs inside conv2's float forward,
        # the global pool inside fc1's.
        mp = lambda x: pool(x, (2, 2), (2, 2), np.max)  # noqa: E731
        return [
            mod.Conv2D(name="conv1", weight=w1, bias=b1, input_name="input", forward=f1, observer=obs()),
            batch_norm(mod, "bn1", "conv1", 4, 1),
            mod.ReLU(name="relu1", input_name="bn1", observer=obs()),
            mod.Conv2D(name="conv2", weight=w2, bias=b2, input_name="relu1",
                       forward=lambda x: f2(mp(x)), observer=obs()),
            batch_norm(mod, "bn2", "conv2", 4, 2),
            mod.ReLU(name="relu2", input_name="bn2", observer=mod.PercentileObserver(0.05, 99.95)),
            mod.Conv2D(name="conv3", weight=w3, bias=b3, input_name="relu2", forward=f3, observer=obs()),
            mod.ReLU(name="relu3", input_name="conv3", observer=mod.KLDivergenceObserver()),
            mod.Dense(name="fc1", weight=w4, bias=b4, input_name="relu3",
                      forward=lambda x: f4(gap(x)), observer=obs()),
            mod.Softmax(name="probs", input_name="fc1"),
        ]
    return [
        mod.Conv2D(name="conv1", weight=w1, bias=b1, input_name="input", forward=f1,
                   padding=(1, 1, 1, 1), observer=obs()),
        batch_norm(mod, "bn1", "conv1", 4, 1),
        mod.ReLU(name="relu1", input_name="bn1", observer=obs()),
        mod.MaxPool2D(name="pool1", input_name="relu1", kernel=(2, 2)),
        mod.Conv2D(name="conv2", weight=w2, bias=b2, input_name="pool1", forward=f2,
                   padding=(1, 1, 1, 1), depthwise=True, observer=obs()),
        batch_norm(mod, "bn2", "conv2", 4, 2),
        mod.ReLU(name="relu2", input_name="bn2", observer=mod.PercentileObserver(0.05, 99.95)),
        mod.Conv2D(name="conv3", weight=w3, bias=b3, input_name="relu2", forward=f3, observer=obs()),
        mod.ReLU(name="relu3", input_name="conv3", observer=mod.KLDivergenceObserver()),
        mod.GlobalAvgPool2D(name="gap", input_name="relu3"),
        mod.Dense(name="fc1", weight=w4, bias=b4, input_name="gap", forward=f4, observer=obs()),
        mod.Softmax(name="probs", input_name="fc1"),
    ]


def cnn_flat(mod, tinymind_cpp: bool) -> list:
    conv, pool, _ = mod_ops(mod)
    w1, b1 = formula((5, 3, 2, 2), 5, 0.5), bias(5, 5)
    w2, b2 = formula((6, 50), 6, 0.25), bias(6, 6)
    w3, b3 = formula((2, 6), 7), bias(2, 7)
    f1 = lambda x: conv(x, w1, b1, (2, 1), (0, 0, 0, 0), False)  # noqa: E731
    f2 = lambda x: dense(x, w2, b2)  # noqa: E731
    f3 = lambda x: dense(x, w3, b3)  # noqa: E731
    obs = mod.MinMaxObserver
    if tinymind_cpp:
        flat = lambda x: pool(x, (2, 2), (2, 2), np.mean).reshape(-1)  # noqa: E731
        return [
            mod.Conv2D(name="conv1", weight=w1, bias=b1, input_name="input", forward=f1, observer=obs()),
            mod.ReLU(name="relu1", input_name="conv1", observer=obs()),
            mod.Dense(name="fc1", weight=w2, bias=b2, input_name="relu1",
                      forward=lambda x: f2(flat(x)), observer=obs()),
            mod.ReLU(name="relu2", input_name="fc1", observer=mod.KLDivergenceObserver()),
            mod.Dense(name="fc2", weight=w3, bias=b3, input_name="relu2", forward=f3, observer=obs()),
            mod.Tanh(name="out", input_name="fc2"),
        ]
    return [
        mod.Conv2D(name="conv1", weight=w1, bias=b1, input_name="input", forward=f1,
                   stride=(2, 1), observer=obs()),
        mod.ReLU(name="relu1", input_name="conv1", observer=obs()),
        mod.AvgPool2D(name="pool1", input_name="relu1", kernel=(2, 2)),
        mod.Flatten(name="flat", input_name="pool1"),
        mod.Dense(name="fc1", weight=w2, bias=b2, input_name="flat", forward=f2, observer=obs()),
        mod.ReLU(name="relu2", input_name="fc1", observer=mod.KLDivergenceObserver()),
        mod.Dense(name="fc2", weight=w3, bias=b3, input_name="relu2", forward=f3, observer=obs()),
        mod.Tanh(name="out", input_name="fc2"),
    ]


_RS = None


def mod_ops(_mod):
    """The float conv, pool and global-pool references (this repo's importer's,
    used for both importers so they calibrate the same model)."""
    rs = _RS or load(os.path.join(HERE, "tinymind_import.py"), "tinymind_import_rs")

    def pool(x, kernel, stride, op):
        layer = (rs.MaxPool2D if op is np.max else rs.AvgPool2D)(
            name="p", kernel=kernel, stride=stride)
        return rs.pool2d_nhwc(x, layer)

    return rs.conv2d_nhwc, pool, (lambda x: x.astype(np.float64).mean(axis=(0, 1)).astype(np.float32))


# name -> (layer builder, input shape (h, w, c), outputs)
MODELS = {
    "cnn": (cnn, (12, 12, 1), 3),
    "cnn_flat": (cnn_flat, (9, 11, 2), 2),
}


def load(path: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[name] = mod  # dataclasses look their module up while it executes
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="cnn")
    ap.add_argument("--rust", help="write the Rust module (this repo's importer)")
    ap.add_argument("--cpp", help="write a header with tinymind's importer")
    args = ap.parse_args()
    build, shape, n_out = MODELS[args.model]
    model_name = f"example_{args.model}"
    if args.rust:
        mod = load(os.path.join(HERE, "tinymind_import.py"), "tinymind_import_rs")
        ranges = mod.import_pytorch_model(build(mod, False), "input", mod.MinMaxObserver(),
                                          calibration_set(shape), args.rust, model_name,
                                          input_shape=shape)
        print("wrote", args.rust, {k: (round(s, 6), z) for k, (s, z) in ranges.items()})
    if args.cpp:
        tm = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
        mod = load(os.path.join(tm, "apps", "import_pytorch", "tinymind_import.py"), "tinymind_import_cpp")
        mod.import_pytorch_model(build(mod, True), input_name="input",
                                 input_observer=mod.MinMaxObserver(),
                                 dataset=calibration_set(shape), output_path=args.cpp,
                                 namespace=model_name, meta={"NumOutputs": n_out})
        print("wrote", args.cpp)


if __name__ == "__main__":
    main()
