#!/usr/bin/env python3
"""Build the ONNX importer's example models: tools/import_onnx/models/*.onnx.

  python3 -I tools/import_onnx/make_models.py      (needs torch, onnx, onnxruntime)

Each model is a small PyTorch module with seeded random weights, exported to
ONNX (opset 13) and quantized by ONNX Runtime's
`quantize_static` (int8 weights and activations) from 32 seeded calibration
inputs, the way a user quantizes a trained model:

  cnn      QDQ format, per-channel weights. Conv 3x2 (stride 1x2, padding 1x0)
           + ReLU -> pointwise Conv + ReLU -> Conv 3x3 (stride 2) + ReLU ->
           Flatten -> Linear + Tanh -> Linear + Sigmoid -> Linear -> Softmax,
           on a 2 x 8 x 10 input.
  cnn_qop  QOperator format, per-tensor weights. Conv 3x3 (padding 1) + ReLU
           -> pointwise Conv + ReLU -> Flatten -> Linear + ReLU -> Linear
           without bias (QLinearMatMul) -> Sigmoid (QLinearSigmoid), on a
           1 x 6 x 6 input.
  mlp      QDQ format, per-tensor weights. A MatMul with a separate bias Add
           (which the importer folds into one dense layer) -> Tanh -> Linear
           -> Softmax, on 7 inputs.

The quantized models are committed, so regenerating the Rust modules and the
C++ check needs only the `onnx` package. Rerun this only to change the
models; ONNX Runtime's calibration may differ between its versions.
"""

import os

import numpy as np
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

HERE = os.path.dirname(os.path.abspath(__file__))
nn = torch.nn


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(2, 6, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
        self.c2 = nn.Conv2d(6, 4, kernel_size=1)
        self.c3 = nn.Conv2d(4, 5, kernel_size=3, stride=2)
        self.f1 = nn.Linear(5 * 3 * 2, 12)
        self.f2 = nn.Linear(12, 8)
        self.f3 = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.c2(h))
        h = torch.relu(self.c3(h))
        h = torch.tanh(self.f1(h.flatten(1)))
        h = torch.sigmoid(self.f2(h))
        return torch.softmax(self.f3(h), dim=1)


class CnnQop(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(4, 3, kernel_size=1)
        self.f1 = nn.Linear(3 * 6 * 6, 10)
        self.f2 = nn.Linear(10, 3, bias=False)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.c2(h))
        h = torch.relu(self.f1(h.flatten(1)))
        return torch.sigmoid(self.f2(h))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(7, 9) * 0.4)
        self.b = nn.Parameter(torch.randn(9) * 0.2)
        self.f = nn.Linear(9, 5)

    def forward(self, x):
        h = torch.tanh(torch.matmul(x, self.w) + self.b)
        return torch.softmax(self.f(h), dim=1)


# name -> (module, input shape, ONNX Runtime format, per-channel weights,
# gain). PyTorch's default initialization shrinks the signal layer by layer,
# so the weights are scaled by the gain: otherwise the cnn's softmax gives
# nearly the same output for every input, and a comparison of outputs would
# check little.
MODELS = {
    "cnn": (Cnn, (1, 2, 8, 10), QuantFormat.QDQ, True, 3.0),
    "cnn_qop": (CnnQop, (1, 1, 6, 6), QuantFormat.QOperator, False, 2.0),
    "mlp": (Mlp, (1, 7), QuantFormat.QDQ, False, 1.0),
}


class Reader(CalibrationDataReader):
    def __init__(self, shape, seed):
        rng = np.random.default_rng(seed)
        self._it = iter([{"input": rng.standard_normal(shape).astype(np.float32)} for _ in range(32)])

    def get_next(self):
        return next(self._it, None)


def build(name: str, out_dir: str) -> str:
    module, shape, fmt, per_channel, gain = MODELS[name]
    torch.manual_seed(sorted(MODELS).index(name) + 1)
    model = module().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(gain)
    fp32 = os.path.join(out_dir, f"{name}_fp32.onnx")
    torch.onnx.export(model, torch.zeros(shape), fp32, input_names=["input"],
                      output_names=["output"], opset_version=13, dynamo=False)
    out = os.path.join(out_dir, f"{name}.onnx")
    quantize_static(fp32, out, Reader(shape, 7), quant_format=fmt,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                    per_channel=per_channel)
    os.remove(fp32)
    return out


def main():
    out_dir = os.path.join(HERE, "models")
    os.makedirs(out_dir, exist_ok=True)
    for name in MODELS:
        path = build(name, out_dir)
        print(f"wrote {os.path.relpath(path)} ({os.path.getsize(path)} B)")


if __name__ == "__main__":
    main()
