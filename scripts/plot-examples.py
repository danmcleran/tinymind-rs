#!/usr/bin/env python3
"""Render the example plots for the docs site: xor in both formats, iris and
energy efficiency.

Runs the Rust `xor`, `iris` and `energy_efficiency` examples (which write byte-identical output to
the C++ examples), then draws the learning curve and decision surface with tinymind's
own plotting style (`examples/plotting/tinymind_plot.py`, imported read-only
from TINYMIND_DIR, default ../tinymind). Writes PNGs to docs/assets/plots/.

Needs matplotlib. Use a virtual environment, for example:

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install matplotlib
    cargo build --release --examples
    /tmp/venv/bin/python -I -B scripts/plot-examples.py

-B keeps Python from writing __pycache__ into the tinymind checkout.
"""

import csv
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINYMIND_DIR = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
sys.path.insert(0, os.path.join(TINYMIND_DIR, "examples", "plotting"))
import tinymind_plot as tp  # noqa: E402

XOR = os.path.join(ROOT, "target", "release", "examples", "xor")
IRIS = os.path.join(ROOT, "target", "release", "examples", "iris")
ENERGY = os.path.join(ROOT, "target", "release", "examples", "energy_efficiency")
IRIS_NAMES = ["setosa", "versicolor", "virginica"]
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


def iris(out_dir, png):
    """The figure tinymind's examples/iris/plot.py draws: loss, confusion, scatter."""
    loss, _ = tp.read_csv(os.path.join(out_dir, "iris_loss.csv"))
    with open(os.path.join(out_dir, "iris_confusion.csv"), newline="") as f:
        rows = list(csv.reader(f))
    mat = [[int(rows[r + 1][c + 1]) for c in range(3)] for r in range(3)]
    total = sum(sum(r) for r in mat)
    acc = sum(mat[i][i] for i in range(3)) / total if total else 0.0
    scat, _ = tp.read_csv(os.path.join(out_dir, "iris_scatter.csv"))

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("tinymind-rs: Iris species classifier (Q16.16 MLP 4-8-3)",
                 fontsize=14, fontweight="bold")

    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)

    ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(IRIS_NAMES, rotation=20, ha="right")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(IRIS_NAMES)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("actual")
    vmax = max(max(r) for r in mat)
    for r in range(3):
        for c in range(3):
            ax2.text(c, r, str(mat[r][c]), ha="center", va="center",
                     color="white" if mat[r][c] > vmax / 2 else "#222",
                     fontsize=12, fontweight="bold")
    ax2.set_title("test confusion  (acc=%.3f)" % acc, fontsize=10, color=tp.MUTED)
    ax2.grid(False)

    pl, pw = scat["petal_len"], scat["petal_width"]
    tcl, pcl = scat["true_class"], scat["pred_class"]
    for k in range(3):
        xs = [pl[i] for i in range(len(pl)) if int(pcl[i]) == k]
        ys = [pw[i] for i in range(len(pw)) if int(pcl[i]) == k]
        ax3.scatter(xs, ys, s=40, color=tp.PALETTE[k], label=IRIS_NAMES[k],
                    edgecolor="white", linewidth=0.5)
    miss = [i for i in range(len(pl)) if int(pcl[i]) != int(tcl[i])]
    if miss:
        ax3.scatter([pl[i] for i in miss], [pw[i] for i in miss], s=120, marker="x",
                    color="#C44E52", label="misclassified", linewidth=2)
    ax3.set_xlabel("petal length (cm)")
    ax3.set_ylabel("petal width (cm)")
    ax3.set_title("test predictions", fontsize=10, color=tp.MUTED)
    ax3.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png)
    tp.plt.close(fig)


def _regression_metrics(true, pred):
    n = len(true)
    mae = sum(abs(pred[i] - true[i]) for i in range(n)) / n
    mean_true = sum(true) / n
    ss_res = sum((pred[i] - true[i]) ** 2 for i in range(n))
    ss_tot = sum((true[i] - mean_true) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return mae, r2


def _predicted_vs_actual(ax, true, pred, name, color):
    mae, r2 = _regression_metrics(true, pred)
    lo = min(min(true), min(pred))
    hi = max(max(true), max(pred))
    ax.plot([lo, hi], [lo, hi], color=tp.MUTED, linestyle="--", linewidth=1.2, label="y = x")
    ax.scatter(true, pred, s=30, color=color, edgecolor="white", linewidth=0.5,
               label="test samples")
    ax.set_xlabel("actual %s" % name)
    ax.set_ylabel("predicted %s" % name)
    ax.set_title("%s  (MAE=%.2f, R²=%.3f)" % (name, mae, r2), fontsize=10, color=tp.MUTED)
    ax.legend()


def energy(out_dir, png):
    """The figure tinymind's examples/energy_efficiency/plot.py draws: loss and
    predicted-vs-actual for both loads."""
    loss, _ = tp.read_csv(os.path.join(out_dir, "energy_loss.csv"))
    pred, _ = tp.read_csv(os.path.join(out_dir, "energy_pred.csv"))

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("tinymind-rs: Energy Efficiency regression (Q16.16 MLP 8-16-2)",
                 fontsize=14, fontweight="bold")
    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)
    _predicted_vs_actual(ax2, pred["y1_true"], pred["y1_pred"], "heating load", tp.PALETTE[3])
    _predicted_vs_actual(ax3, pred["y2_true"], pred["y2_pred"], "cooling load", tp.PALETTE[0])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png)
    tp.plt.close(fig)


def main():
    if not (os.path.exists(XOR) and os.path.exists(IRIS) and os.path.exists(ENERGY)):
        sys.exit("build first: cargo build --release --examples")
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
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([IRIS, tmp, os.path.join(ROOT, "data", "iris.data")],
                       check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "iris_classifier.png")
        iris(tmp, png)
        print("wrote", os.path.relpath(png, ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([ENERGY, tmp, os.path.join(ROOT, "data", "ENB2012_data.csv")],
                       check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "energy_regression.png")
        energy(tmp, png)
        print("wrote", os.path.relpath(png, ROOT))


if __name__ == "__main__":
    main()
