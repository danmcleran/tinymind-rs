#!/usr/bin/env python3
"""Render the xor example's plots for the docs site, in both formats.

Runs the Rust `xor` binary (which writes byte-identical output to the C++
example), then draws the learning curve and decision surface with tinymind's
own plotting style (`examples/plotting/tinymind_plot.py`, imported read-only
from TINYMIND_DIR, default ../tinymind). Writes PNGs to docs/assets/plots/.

Needs matplotlib. Use a virtual environment, for example:

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install matplotlib
    cargo build --release
    /tmp/venv/bin/python -I -B scripts/plot-xor.py

-B keeps Python from writing __pycache__ into the tinymind checkout.
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINYMIND_DIR = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
sys.path.insert(0, os.path.join(TINYMIND_DIR, "examples", "plotting"))
import tinymind_plot as tp  # noqa: E402

XOR = os.path.join(ROOT, "target", "release", "xor")
OUT = os.path.join(ROOT, "docs", "assets", "plots")
FORMATS = [("q16.16", "Q16.16", "q16_16"), ("q8.8", "Q8.8", "q8_8")]


def learning_curve(csv, label, png):
    cols, _ = tp.read_csv(csv)
    fig, ax = tp.new_fig(
        "XOR network learning curve (%s)" % label,
        "tinymind-rs: %s fixed-point MLP (2->4->1), tanh hidden + sigmoid output" % label)
    tp.line(ax, cols["iteration"], {"average error": cols["avg_error"]},
            xlabel="training iteration", ylabel="average |error|")
    tp.finish(fig, png, show=False)


def decision_surface(csv, label, png):
    cols, _ = tp.read_csv(csv)
    prob = cols["prob"]
    g = int(round(len(prob) ** 0.5))
    grid = [[0.0] * g for _ in range(g)]
    for k in range(len(prob)):
        grid[k // g][k % g] = prob[k]
    fig, ax = tp.new_fig("XOR decision surface (%s)" % label,
                         "tinymind-rs: fixed-point MLP (2->4->1)")
    im = ax.imshow(grid, origin="lower", extent=(0, 1, 0, 1),
                   cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    axis = [i / (g - 1) for i in range(g)]
    ax.contour(axis, axis, grid, levels=[0.5], colors=tp.FG, linewidths=1.5)
    for (cx, cy, lbl) in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        ax.scatter([cx], [cy], s=160, edgecolor="white", linewidth=1.5,
                   color=("#C44E52" if lbl else "#4C72B0"), zorder=5)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(XOR=1)")
    tp.finish(fig, png, show=False)


def main():
    if not os.path.exists(XOR):
        sys.exit("build first: cargo build --release")
    os.makedirs(OUT, exist_ok=True)
    for fmt, label, slug in FORMATS:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([XOR, "--format", fmt, tmp], check=True, stdout=subprocess.DEVNULL)
            for kind, fn in (("learning_curve", learning_curve),
                             ("decision_surface", decision_surface)):
                csv = os.path.join(tmp, "xor_training.csv" if kind == "learning_curve"
                                   else "xor_decision_surface.csv")
                png = os.path.join(OUT, "xor_%s_%s.png" % (slug, kind))
                fn(csv, label, png)
                print("wrote", os.path.relpath(png, ROOT))


if __name__ == "__main__":
    main()
