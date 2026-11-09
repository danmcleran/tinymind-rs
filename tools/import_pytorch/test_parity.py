#!/usr/bin/env python3
"""Parity of this importer's calibration and quantization with tinymind's
(apps/import_pytorch/tinymind_import.py, from TINYMIND_DIR, default
../tinymind): on several models, every tensor's (scale, zero point), every weight
scale, and every int8 weight and int32 bias must be identical. The models are
dense chains, and the convolutional examples of example_cnn.py (tinymind's
importer has no pooling or flatten layers, so there each one runs inside the
next layer's float forward, as example_cnn.py --cpp does). Also: the softmax
parameters it bakes equal tinymind's C++ buildQSoftmaxFixedPointParams on the
504 calibrations of tests/golden/softmax/stdout.txt.

  python3 -I -B tools/import_pytorch/test_parity.py      (needs numpy)
"""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def model(mod, sizes, observers, final, seed):
    """A chain Dense(+ReLU) ... Dense (+final activation) with formula weights."""
    layers = []
    current = "input"
    n = len(sizes) - 1
    for l in range(n):
        n_in, n_out = sizes[l], sizes[l + 1]
        w = np.array([[((o * 7 + i * 13 + l * 5 + seed) % 23 - 11) / (9.0 + seed)
                       for i in range(n_in)] for o in range(n_out)], dtype=np.float32)
        b = np.array([((o * 3 + l + seed) % 7 - 3) / 5.0 for o in range(n_out)], dtype=np.float32)
        name = f"fc{l + 1}"
        layers.append(mod.Dense(name=name, weight=w, bias=b, input_name=current,
                                forward=(lambda x, w=w, b=b: x @ w.T + b),
                                observer=observers[l](mod)))
        current = name
        if l + 1 < n:
            layers.append(mod.ReLU(name=f"h{l + 1}", input_name=current, observer=observers[l](mod)))
            current = f"h{l + 1}"
    if final == "sigmoid":
        layers.append(mod.Sigmoid(name="out", input_name=current))
    elif final == "tanh":
        layers.append(mod.Tanh(name="out", input_name=current))
    elif final == "softmax":
        layers.append(mod.Softmax(name="out", input_name=current))
    return layers


def dataset(n_in, seed, lo=-1.0):
    return [np.array([((s * 31 + k * 17 + seed) % 41) / 20.0 * (1.0 - lo) / 2.0 + lo
                      for k in range(n_in)], dtype=np.float32) for s in range(48)]


MINMAX = lambda m: m.MinMaxObserver()
PCT = lambda m: m.PercentileObserver(0.05, 99.95)
KL = lambda m: m.KLDivergenceObserver()

CASES = [
    ([3, 8, 4, 2], [MINMAX, PCT, KL], "sigmoid", 0, -1.0),
    ([5, 6, 3], [KL, MINMAX], "tanh", 3, -2.0),
    ([2, 4, 1], [MINMAX, MINMAX], "sigmoid", 1, 0.0),
    ([4, 16, 8, 8, 3], [PCT, KL, MINMAX, PCT], None, 7, -0.5),
    ([6, 12, 5], [KL, PCT], "softmax", 5, -1.0),
]


def run(mod, case):
    sizes, observers, final, seed, lo = case
    layers = model(mod, sizes, observers, final, seed)
    ranges = mod.calibrate(layers, "input", mod.MinMaxObserver(), dataset(sizes[0], seed, lo))
    mod.quantize_weights(layers, ranges)
    dense = [(l.name, l.weight_scale, l.q_weight.tolist(), l.q_bias.tolist())
             for l in layers if type(l).__name__ == "Dense"]
    return ranges, dense


def main():
    tm = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
    ours = load(os.path.join(HERE, "tinymind_import.py"), "tm_import_rs")
    theirs = load(os.path.join(tm, "apps", "import_pytorch", "tinymind_import.py"), "tm_import_cpp")
    failures = 0
    for i, case in enumerate(CASES):
        a, b = run(ours, case), run(theirs, case)
        same = a == b
        failures += not same
        print(f"case {i} {case[0]} final={case[2]}: {'IDENTICAL' if same else 'DIFFERENT'}"
              f" ({len(a[0])} tensors, {sum(len(d[2]) * len(d[2][0]) for d in a[1])} weights)")
    failures += not conv_models_match(ours, theirs)
    failures += not softmax_params_match(ours)
    sys.exit(1 if failures else 0)


def conv_models_match(ours, theirs):
    """example_cnn.py's models: every tensor tinymind's importer calibrates has
    the same (scale, zero point) in ours, and every Conv2D and Dense the same
    weight scale, int8 weights and int32 biases."""
    ex = load(os.path.join(HERE, "example_cnn.py"), "example_cnn")
    ok = True
    for name, (build, shape, _) in sorted(ex.MODELS.items()):
        data = ex.calibration_set(shape)
        a_layers = build(ours, False)
        a_ranges = ours.calibrate(a_layers, "input", ours.MinMaxObserver(), data)
        ours.quantize_weights(a_layers, a_ranges)
        b_layers = theirs.fuse_layers(build(theirs, True))
        b_ranges = theirs.calibrate(b_layers, "input", theirs.MinMaxObserver(), data)
        theirs.quantize_weights(b_layers, b_ranges)

        def weights(layers):
            return {l.name: (l.weight_scale, l.q_weight.tolist(),
                             None if l.q_bias is None else l.q_bias.tolist())
                    for l in layers if type(l).__name__ in ("Conv2D", "Dense")}

        same_ranges = all(a_ranges.get(k) == v for k, v in b_ranges.items())
        a_w, b_w = weights(a_layers), weights(b_layers)
        same = same_ranges and a_w == b_w
        ok &= same
        n = sum(len(np.array(w[1]).reshape(-1)) for w in a_w.values())
        print(f"case {name}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(b_ranges)} tensors, "
              f"{n} weights, {len(a_w)} conv/dense layers)")
    return ok


def softmax_params_match(mod):
    """build_softmax vs the C++ parameters in the softmax sweep golden, whose
    input scales (1 + j/8) * 2^e are exact in float32."""
    betas = [1.0, 0.5, 2.5]
    n = bad = 0
    with open(os.path.join(ROOT, "tests", "golden", "softmax", "stdout.txt")) as f:
        for line in f:
            fields = dict(kv.split("=") for kv in line.split())
            e, j, b = int(fields["e"]), int(fields["j"]), int(fields["beta"])
            want = tuple(int(v) for v in fields["params"].split(","))
            got = mod.build_softmax((1.0 + j / 8.0) * 2.0 ** e, betas[b])
            n += 1
            bad += got != want
    print(f"softmax params: {n - bad} of {n} IDENTICAL")
    return bad == 0


if __name__ == "__main__":
    main()
