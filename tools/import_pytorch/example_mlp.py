#!/usr/bin/env python3
"""The importer's worked example: a 3-8-4-2 MLP (ReLU, ReLU, sigmoid) whose
hidden layers use all three observers.

  python3 -I tools/import_pytorch/example_mlp.py --rust examples/common/imported_mlp_model.rs
  python3 -I tools/import_pytorch/example_mlp.py --cpp weights.hpp   # tinymind's importer

The weights and the calibration set are integer formulas, so they are exact
floats on every platform and the generated file is reproducible. The model
stands in for one trained in PyTorch: with a real model, take the arrays from
`model.state_dict()` or use `tinymind_import.from_torch(model)`.

--cpp runs tinymind's own importer (apps/import_pytorch, from TINYMIND_DIR,
default ../tinymind) on the same model; scripts/compare-with-cpp.sh feeds that
header to a C++ program on tinymind's runtime and byte-compares its output with
the Rust example's.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SIZES = [3, 8, 4, 2]


def weights(layer: int, n_out: int, n_in: int) -> np.ndarray:
    """Exact float weights in [-1, 1] from an integer formula."""
    w = [[((o * 7 + i * 13 + layer * 5) % 17 - 8) / 8.0 for i in range(n_in)]
         for o in range(n_out)]
    return np.array(w, dtype=np.float32)


def biases(layer: int, n_out: int) -> np.ndarray:
    return np.array([((o * 3 + layer) % 5 - 2) / 4.0 for o in range(n_out)], dtype=np.float32)


def calibration_set() -> list:
    """64 inputs in [-1, 1]^3."""
    return [np.array([((s * 31 + k * 17) % 41) / 20.0 - 1.0 for k in range(3)], dtype=np.float32)
            for s in range(64)]


def layers(mod) -> list:
    w1, b1 = weights(1, 8, 3), biases(1, 8)
    w2, b2 = weights(2, 4, 8), biases(2, 4)
    w3, b3 = weights(3, 2, 4), biases(3, 2)
    return [
        mod.Dense(name="fc1", weight=w1, bias=b1, input_name="input",
                  forward=lambda x: x @ w1.T + b1, observer=mod.MinMaxObserver()),
        mod.ReLU(name="h1", input_name="fc1", observer=mod.MinMaxObserver()),
        mod.Dense(name="fc2", weight=w2, bias=b2, input_name="h1",
                  forward=lambda x: x @ w2.T + b2, observer=mod.MinMaxObserver()),
        mod.ReLU(name="h2", input_name="fc2", observer=mod.PercentileObserver(0.05, 99.95)),
        mod.Dense(name="fc3", weight=w3, bias=b3, input_name="h2",
                  forward=lambda x: x @ w3.T + b3, observer=mod.KLDivergenceObserver()),
        mod.Sigmoid(name="out", input_name="fc3"),
    ]


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[name] = mod  # dataclasses look their module up while it executes
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", help="write the Rust module (this repo's importer)")
    ap.add_argument("--cpp", help="write a weights.hpp with tinymind's importer")
    args = ap.parse_args()
    if args.rust:
        mod = load(os.path.join(HERE, "tinymind_import.py"), "tinymind_import_rs")
        ranges = mod.import_pytorch_model(layers(mod), "input", mod.MinMaxObserver(),
                                          calibration_set(), args.rust, "example_mlp")
        print("wrote", args.rust, {k: (round(s, 6), z) for k, (s, z) in ranges.items()})
    if args.cpp:
        tm = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
        mod = load(os.path.join(tm, "apps", "import_pytorch", "tinymind_import.py"), "tinymind_import_cpp")
        mod.import_pytorch_model(layers(mod), input_name="input", input_observer=mod.MinMaxObserver(),
                                 dataset=calibration_set(), output_path=args.cpp,
                                 namespace="example_mlp",
                                 meta={"NumInputs": SIZES[0], "NumOutputs": SIZES[-1]})
        print("wrote", args.cpp)


if __name__ == "__main__":
    main()
