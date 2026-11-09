#!/usr/bin/env python3
"""Checks of the TFLite translator that need no model file.

  python3 -I -B tools/import_tflite/test_translator.py      (needs numpy, tflite)

* The softmax parameters it bakes equal tinymind (C++)'s C++
  buildQSoftmaxFixedPointParams (TFLite Micro's) on the 504 calibrations of
  tests/golden/softmax/stdout.txt.
* The ADD parameters it bakes (TFLite Micro's computation, partly in float32,
  as tinymind (C++)'s translator does it) equal tinymind (C++)'s
  buildQAddParams (in double) on the 48 calibrations of
  tests/golden/add/stdout.txt: the scaling by two and by 2^20 is exact in both.
* A fused ReLU raises the requantizer's lower bound to the output zero point,
  and ReLU6 also lowers the upper bound to the zero point plus 6 / scale, on
  convolutions and on ADD (the MLPerf models cannot show either: their outputs'
  zero point is -128).

The translation itself is checked end to end by mlperf.py, against tinymind
(C++)'s output hashes for the MLPerf Tiny keyword-spotting model (DS-CNN) and
image-classification model (ResNet-8).
"""

import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tflite  # noqa: E402
import tinymind_import_tflite as tt  # noqa: E402


def softmax_params_match() -> bool:
    betas = [1.0, 0.5, 2.5]
    n = bad = 0
    with open(os.path.join(ROOT, "tests", "golden", "softmax", "stdout.txt")) as f:
        for line in f:
            fields = dict(kv.split("=") for kv in line.split())
            e, j, b = int(fields["e"]), int(fields["j"]), int(fields["beta"])
            want = tuple(int(v) for v in fields["params"].split(","))
            got = tt.softmax_params(betas[b], (1.0 + j / 8.0) * 2.0 ** e)
            n += 1
            bad += got != want
    print(f"softmax params: {n - bad} of {n} IDENTICAL")
    return bad == 0


def add_params_match() -> bool:
    n = bad = 0
    scales = None
    with open(os.path.join(ROOT, "tests", "golden", "add", "stdout.txt")) as f:
        for line in f:
            if line.startswith("scales "):
                scales = [struct.unpack("<f", struct.pack("<I", int(v, 16)))[0] for v in line.split()[1:]]
            elif line.startswith("calibrated "):
                f_ = dict(kv.split("=") for kv in line.split()[2:])
                a, b, rq = (tuple(int(v) for v in f_[k].split(",")) for k in ("a", "b", "rq"))
                got = tt.add_params(*scales)
                n += 1
                bad += got != (a, b, rq[:2]) or int(f_["ls"]) != 20
    print(f"add params: {n - bad} of {n} IDENTICAL")
    return n == 48 and bad == 0


def add_relu_bounds() -> bool:
    A = tflite.ActivationFunctionType
    lit = tt.add_literal((0.05, -3), (0.07, 10), (0.1, -20), A.RELU)
    m = re.search(r"zero_point: (-?\d+), qmin: (-?\d+), qmax: (-?\d+)", lit)
    ok = m is not None and m.groups() == ("-20", "-20", "127")
    print(f"add fused ReLU: {'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok


def activation_ranges_match() -> bool:
    A = tflite.ActivationFunctionType
    cases = [
        ((A.NONE, 0.1, 5), (-128, 127)),
        ((A.RELU, 0.1, 5), (5, 127)),
        ((A.RELU, 0.1, -128), (-128, 127)),
        ((A.RELU6, 0.1, -20), (-20, 40)),        # 6 / 0.1 = 60
        ((A.RELU6, 0.01, 0), (0, 127)),          # 600 above the zero point: saturates
    ]
    ok = all(tt.act_range(*args) == want for args, want in cases)
    print(f"activation ranges: {'IDENTICAL' if ok else 'DIFFERENT'} ({len(cases)} cases)")
    return ok


if __name__ == "__main__":
    good = softmax_params_match()
    good &= add_params_match()
    good &= activation_ranges_match()
    good &= add_relu_bounds()
    sys.exit(0 if good else 1)
