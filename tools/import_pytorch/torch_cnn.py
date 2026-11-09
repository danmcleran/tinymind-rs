#!/usr/bin/env python3
"""Check `from_torch` on convolutional models against PyTorch itself.

  python3 -I tools/import_pytorch/torch_cnn.py [out.rs]      (needs torch)

`from_torch` turns PyTorch's NCHW tensors and OIHW weights into the NHWC
tensors and OHWI weights the crate uses, reorders a Linear after a Flatten, and
translates padding and depthwise groups, and folds batch norm into the
convolution before it. A mistake in any of that leaves the
importer self-consistent (calibration and the integer model both use the
translated descriptors) while the model computes something else, so it is
checked here against PyTorch's own output: for several models, the descriptors'
float forward on NHWC inputs, and the same chain run from each convolution's
folded weights, must equal the module's on NCHW inputs to within float
rounding. With an argument, it also writes the first model as a Rust
module.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinymind_import as ti  # noqa: E402

nn = torch.nn

# (name, input (channels, height, width), model)
MODELS = [
    ("padded conv, max pool, depthwise, pointwise, global pool", (1, 12, 12), lambda: nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(4, 4, 3, padding=1, groups=4), nn.ReLU(),
        nn.Conv2d(4, 8, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3), nn.Softmax(dim=1))),
    ("strided conv, average pool, flatten into linear", (2, 11, 9), lambda: nn.Sequential(
        nn.Conv2d(2, 5, (3, 2), stride=(2, 1)), nn.ReLU(), nn.AvgPool2d(2),
        nn.Flatten(), nn.Linear(5 * 2 * 4, 6), nn.ReLU(), nn.Linear(6, 2), nn.Tanh())),
    ("asymmetric padding, 'same', overlapping pool, no bias", (3, 8, 10), lambda: nn.Sequential(
        nn.Conv2d(3, 6, (5, 3), padding=(2, 1), bias=False), nn.ReLU(),
        nn.Conv2d(6, 6, 4, padding="same", groups=6), nn.ReLU(),
        nn.MaxPool2d(3, stride=2), nn.Flatten(), nn.Linear(6 * 3 * 4, 4))),
    ("batch norm after a conv and after a depthwise conv", (2, 10, 10), lambda: with_bn_stats(nn.Sequential(
        nn.Conv2d(2, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2d(4, 4, 3, groups=4, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)))),
]


def with_bn_stats(model):
    """Give every BatchNorm2d non-trivial statistics and affine parameters (a
    fresh one is the identity), so a mistake in the folding shows."""
    g = torch.Generator().manual_seed(1)
    for m in model:
        if isinstance(m, nn.BatchNorm2d):
            c = m.num_features
            m.running_mean.copy_(torch.randn(c, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(c, generator=g) * 2.0 + 0.1)
            m.weight.data.copy_(torch.randn(c, generator=g) * 2.0)
            m.bias.data.copy_(torch.randn(c, generator=g))
    return model


def from_folded_weights(layers):
    """The chain with BatchNorm folded, every Conv2D running from its own
    (folded) weights and bias rather than its float forward: what the int8
    layer is quantized from."""
    fused = ti.fuse_layers(layers)
    for layer in fused:
        if isinstance(layer, ti.Conv2D):
            layer.forward = (lambda x, l=layer:
                             ti.conv2d_nhwc(x, l.weight, l.bias, l.stride, l.padding, l.depthwise))
    return fused


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for idx, (name, (c, h, w), build) in enumerate(MODELS):
        model = build().eval()
        layers = ti.from_torch(model, input_shape=(c, h, w))
        for _ in range(8):
            x = rng.uniform(-1.0, 1.0, (c, h, w)).astype(np.float32)
            with torch.no_grad():
                ref = model(torch.from_numpy(x)[None])[0].numpy()
            ours = ti.run_float(layers, x.transpose(1, 2, 0))
            folded = ti.run_float(from_folded_weights(layers), x.transpose(1, 2, 0))
            err = max(float(np.max(np.abs(ours - ref))), float(np.max(np.abs(folded - ref))))
            tol = 1e-5 * max(1.0, float(np.max(np.abs(ref))))
            worst = max(worst, err)
            if ours.shape != ref.shape or err > tol:
                print(f"DIFFERENT  {name}: shape {ours.shape} vs {ref.shape}, max error {err:.3g}")
                sys.exit(1)
        print(f"model {idx}: {name}: matches PyTorch ({len(layers)} layers)")
        if idx == 0 and len(sys.argv) > 1:
            calibration = [rng.uniform(-1.0, 1.0, (h, w, c)).astype(np.float32) for _ in range(32)]
            ti.import_pytorch_model(layers, "input", ti.MinMaxObserver(), calibration,
                                    sys.argv[1], "torch_cnn", input_shape=(h, w, c))
            print("wrote", sys.argv[1])
    print(f"all {len(MODELS)} models match PyTorch (worst error {worst:.2g})")


if __name__ == "__main__":
    main()
