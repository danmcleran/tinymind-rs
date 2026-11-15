#!/usr/bin/env python3
"""Check `from_torch` on LSTM models against PyTorch itself.

  python3 -I tools/import_pytorch/torch_lstm.py [out.rs]      (needs torch)

`from_torch` reads an nn.LSTM's weights (gates i, f, g, o in PyTorch's order)
and sums its two bias vectors. A mistake there, or in the importer's float
LSTM, leaves the importer self-consistent (calibration and the integer model
both use the translated descriptor) while the model computes something else,
so it is checked here against PyTorch's own output: for several models, the
descriptors' float forward must equal PyTorch's LSTM followed by the rest of
the Sequential, to within float rounding. It also checks that what the crate
cannot run is rejected (nn.GRU, two layers, bidirectional, an LSTM after
another layer). With an argument, it writes the first model as a Rust module.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinymind_import as ti  # noqa: E402

nn = torch.nn

# (name, steps, features, model): the first module is the LSTM, the rest reads
# its last hidden state.
MODELS = [
    ("classifier: LSTM 3 -> 8, linear, softmax", 12, 3, lambda: nn.Sequential(
        nn.LSTM(3, 8, batch_first=True), nn.Linear(8, 4), nn.Softmax(dim=1))),
    ("regressor: LSTM 2 -> 5 without bias, two linear, tanh", 10, 2, lambda: nn.Sequential(
        nn.LSTM(2, 5, bias=False), nn.Linear(5, 6), nn.ReLU(), nn.Linear(6, 1), nn.Tanh())),
    ("LSTM alone, 1 feature, 16 units, long sequence", 40, 1, lambda: nn.Sequential(
        nn.LSTM(1, 16))),
]

REJECTED = [
    ("nn.GRU", lambda: nn.Sequential(nn.GRU(2, 4), nn.Linear(4, 1))),
    ("two layers", lambda: nn.Sequential(nn.LSTM(2, 4, num_layers=2), nn.Linear(4, 1))),
    ("bidirectional", lambda: nn.Sequential(nn.LSTM(2, 4, bidirectional=True), nn.Linear(8, 1))),
    ("LSTM after a Linear", lambda: nn.Sequential(nn.Linear(2, 2), nn.LSTM(2, 4))),
]


def torch_forward(model, seq: np.ndarray) -> np.ndarray:
    """The LSTM over one sequence from zero state, then the rest of the
    Sequential on its last hidden state."""
    lstm, head = model[0], nn.Sequential(*list(model)[1:])
    x = torch.from_numpy(seq)[None] if lstm.batch_first else torch.from_numpy(seq)[:, None]
    with torch.no_grad():
        _, (h, _) = lstm(x)
        return head(h[-1])[0].numpy()


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for idx, (name, steps, features, build) in enumerate(MODELS):
        model = build().eval()
        with torch.no_grad():
            # Larger than PyTorch's default init, so the gates leave their
            # linear region and a wrong gate order shows.
            for p in model[0].parameters():
                p.mul_(3.0)
        layers = ti.from_torch(model, sequence_length=steps)
        for _ in range(8):
            seq = rng.uniform(-1.0, 1.0, (steps, features)).astype(np.float32)
            ref = torch_forward(model, seq)
            ours = ti.run_float(layers, seq)
            err = float(np.max(np.abs(ours - ref)))
            tol = 1e-5 * max(1.0, float(np.max(np.abs(ref))))
            worst = max(worst, err)
            if ours.shape != ref.shape or err > tol:
                print(f"DIFFERENT  {name}: shape {ours.shape} vs {ref.shape}, max error {err:.3g}")
                sys.exit(1)
        print(f"model {idx}: {name}: matches PyTorch ({len(layers)} layers)")
        if idx == 0 and len(sys.argv) > 1:
            calibration = [rng.uniform(-1.0, 1.0, (steps, features)).astype(np.float32)
                           for _ in range(32)]
            ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration,
                                    sys.argv[1], "torch_lstm")
            print("wrote", sys.argv[1])
    for name, build in REJECTED:
        try:
            ti.from_torch(build(), sequence_length=4)
        except NotImplementedError as e:
            print(f"rejected {name}: {e}")
            continue
        print(f"ACCEPTED  {name}, which the crate cannot run")
        sys.exit(1)
    print(f"all {len(MODELS)} models match PyTorch (worst error {worst:.2g}), "
          f"{len(REJECTED)} unsupported models rejected")


if __name__ == "__main__":
    main()
