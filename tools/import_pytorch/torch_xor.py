#!/usr/bin/env python3
"""Train a 2-4-1 XOR network in PyTorch and import it as a Rust module.

  python3 -I tools/import_pytorch/torch_xor.py torch_xor_model.rs   (needs torch)

The whole flow in a few lines: train, describe with `from_torch`, calibrate on
representative inputs, write the module. In a Rust crate, `mod torch_xor_model;`
then `torch_xor_model::forward(&input)`.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinymind_import as ti  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "torch_xor_model.rs"
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                              torch.nn.Linear(4, 1), torch.nn.Sigmoid())
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(net(x), y)
        loss.backward()
        opt.step()
    print("float outputs:", [round(float(v), 3) for v in net(x).detach().view(-1)])

    layers = ti.from_torch(net)
    calibration = [row.numpy().astype(np.float32) for row in x]
    ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration, out, "torch_xor")
    print("wrote", out)


if __name__ == "__main__":
    main()
