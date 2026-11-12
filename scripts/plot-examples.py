#!/usr/bin/env python3
"""Render the example plots for the docs site: xor in both formats, iris,
energy efficiency, anomaly detection, the int8 GBDT, the Q-learning and deep
Q-network mazes and the LSTM activity classifier.

Runs the Rust `xor`, `iris`, `energy_efficiency`, `anomaly`, `gbdt_tabular_int8`, `maze`,
`dqn_maze` and `har_activity`
examples (which, except anomaly, write byte-identical output to the C++ examples), then draws the learning curve and decision surface with tinymind's
own plotting style (`examples/plotting/tinymind_plot.py`, imported read-only
from TINYMIND_DIR, default ../tinymind). Writes PNGs to docs/assets/plots/.

Needs matplotlib. Use a virtual environment, for example:

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install matplotlib
    cargo build --release --examples --features std
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
ANOMALY = os.path.join(ROOT, "target", "release", "examples", "anomaly")
GBDT = os.path.join(ROOT, "target", "release", "examples", "gbdt_tabular_int8")
MAZE = os.path.join(ROOT, "target", "release", "examples", "maze")
DQN_MAZE = os.path.join(ROOT, "target", "release", "examples", "dqn_maze")
HAR = os.path.join(ROOT, "target", "release", "examples", "har_activity")
HAR_NAMES = ["walking", "upstairs", "downstairs", "sitting"]
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


def anomaly(out_dir, png):
    """Reconstruction error of every test window by kind, with the threshold."""
    rows = list(csv.DictReader(open(os.path.join(out_dir, "anomaly_scores.csv"), newline="")))
    threshold = int(rows[0]["threshold"]) / 65536.0
    kinds = []
    for r in rows:
        if r["kind"] not in kinds:
            kinds.append(r["kind"])

    tp.apply_style()
    fig, ax = tp.plt.subplots(figsize=(10, 4.6))
    fig.suptitle("tinymind-rs: anomaly detection, autoencoder trained on the device (Q16.16 8-4-8)",
                 fontsize=13, fontweight="bold")
    for i, kind in enumerate(kinds):
        errs = [max(int(r["error"]), 1) / 65536.0 for r in rows if r["kind"] == kind]
        flagged = sum(1 for e in errs if e > threshold)
        # A deterministic spread within each column, so the points do not overlap.
        xs = [i + 0.3 * ((j * 0.618034) % 1.0 - 0.5) for j in range(len(errs))]
        color = tp.PALETTE[0] if kind == "normal" else tp.PALETTE[3]
        ax.scatter(xs, errs, s=16, color=color, edgecolor="white", linewidth=0.4)
        ax.text(i, 1.3, "%d/%d flagged" % (flagged, len(errs)), ha="center",
                fontsize=9, color=tp.MUTED)
    ax.axhline(threshold, color="#C44E52", linestyle="--", linewidth=1.4,
               label="threshold (2 x the largest normal error)")
    ax.set_yscale("log")
    ax.set_ylim(top=2.5)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("reconstruction error (mean squared)")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png)
    tp.plt.close(fig)


def gbdt(out_dir, png):
    """The int8 ensemble's class over the feature grid, with the split thresholds."""
    rows = list(csv.DictReader(open(os.path.join(out_dir, "gbdt_tabular_int8.csv"), newline="")))
    g = int(round(len(rows) ** 0.5))
    # The CSV runs f1 (rows) outer and f0 (columns) inner.
    grid = [[int(rows[iy * g + ix]["class"]) for ix in range(g)] for iy in range(g)]

    from matplotlib.colors import ListedColormap
    tp.apply_style()
    fig, ax = tp.plt.subplots(figsize=(7.2, 6.4))
    fig.suptitle("tinymind-rs: int8 gradient-boosted trees (5 trees, 3 classes)",
                 fontsize=13, fontweight="bold")
    cmap = ListedColormap([tp.PALETTE[0], tp.PALETTE[1], tp.PALETTE[3]])
    im = ax.imshow(grid, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=0, vmax=2,
                   interpolation="nearest", aspect="auto")
    ax.grid(False)
    for x in (-0.15, 0.25):
        ax.axvline(x, color="white", linestyle="--", linewidth=1.0)
    for y in (-0.10, 0.10):
        ax.axhline(y, color="white", linestyle=":", linewidth=1.0)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    ax.set_title("predicted class over a 48 x 48 grid; dashed and dotted lines: split thresholds",
                 fontsize=9, color=tp.MUTED)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["class 0", "class 1", "class 2"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png)
    tp.plt.close(fig)


# Fewest moves from each room to the goal (room 5 must step out through its own door).
MAZE_SHORTEST = {0: 2, 1: 1, 2: 3, 3: 2, 4: 1, 5: 1}


def maze(out_dir, png, prefix="maze",
         title="tinymind-rs: Q-learning, the way out of a six-room maze (Q16.16)"):
    """Moves per training episode, and each test episode against the shortest path."""
    def load(name):
        with open(os.path.join(out_dir, name)) as f:
            return [[int(x) for x in line.strip().rstrip(",").split(",")] for line in f if line.strip()]
    train, test = load(prefix + "_training.csv"), load(prefix + "_test.csv")

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")
    moves = [len(e) - 1 for e in train]
    ax1.scatter(range(len(moves)), moves, s=3, color=tp.PALETTE[0], alpha=0.35, label="moves")
    shortest = [MAZE_SHORTEST[e[0]] for e in train]
    ax1.scatter(range(len(shortest)), shortest, s=2, color=tp.PALETTE[3], alpha=0.5,
                label="shortest from the start room")
    decay = len(moves) - 100
    ax1.axvline(decay, color=tp.MUTED, linestyle="--", linewidth=1.0)
    ax1.text(decay - 60, max(moves) * 0.6, "exploration falls\n1 point per episode", fontsize=9,
             color=tp.MUTED, ha="right")
    ax1.set_yscale("log")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("moves to the goal")
    ax1.set_title("%d episodes of pure exploration, then 100 of decaying exploration" % decay,
                  fontsize=9, color=tp.MUTED)
    ax1.legend(loc="upper left")

    rooms = sorted(MAZE_SHORTEST)
    taken = [sum(len(e) - 1 for e in test if e[0] == r) / max(1, sum(1 for e in test if e[0] == r))
             for r in rooms]
    xs = range(len(rooms))
    ax2.bar([x - 0.2 for x in xs], [MAZE_SHORTEST[r] for r in rooms], width=0.4,
            color=tp.PALETTE[3], label="shortest")
    ax2.bar([x + 0.2 for x in xs], taken, width=0.4, color=tp.PALETTE[0], label="learned (mean)")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(["room %d" % r for r in rooms], fontsize=9)
    ax2.set_ylabel("moves to the goal")
    ax2.set_title("100 test episodes, no exploration", fontsize=9, color=tp.MUTED)
    ax2.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png)
    tp.plt.close(fig)


def har(out_dir, png):
    """Training loss, test confusion matrix, and one synthesized window per activity."""
    loss, _ = tp.read_csv(os.path.join(out_dir, "har_loss.csv"))
    with open(os.path.join(out_dir, "har_confusion.csv"), newline="") as f:
        rows = list(csv.reader(f))
    n = len(HAR_NAMES)
    mat = [[int(rows[r + 1][c + 1]) for c in range(n)] for r in range(n)]
    total = sum(sum(r) for r in mat)
    acc = sum(mat[i][i] for i in range(n)) / total
    traces = {}
    with open(os.path.join(out_dir, "har_samples.csv"), newline="") as f:
        for row in csv.DictReader(f):
            tr = traces.setdefault(row["class"], {"t": [], "ax": [], "ay": [], "az": []})
            for k in tr:
                tr[k].append(float(row[k]))

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("tinymind-rs: LSTM activity classifier (Q16.16, 3-16-4)",
                 fontsize=13, fontweight="bold")
    tp.line(ax1, loss["epoch"], {"mean squared error": loss["avg_err"]},
            xlabel="training epoch", ylabel="error at a window's last step")
    ax1.set_title("training loss", fontsize=9, color=tp.MUTED)

    ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(HAR_NAMES, rotation=20, ha="right")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(HAR_NAMES)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("actual")
    vmax = max(max(r) for r in mat)
    for r in range(n):
        for c in range(n):
            ax2.text(c, r, str(mat[r][c]), ha="center", va="center",
                     color="white" if mat[r][c] > vmax / 2 else "#222",
                     fontsize=12, fontweight="bold")
    ax2.set_title("200 test windows, accuracy %.1f%%" % (100 * acc), fontsize=9, color=tp.MUTED)
    ax2.grid(False)

    for k, name in enumerate(HAR_NAMES):
        tr = traces[name]
        for axis, style in (("ax", "-"), ("ay", "--"), ("az", ":")):
            ax3.plot(tr["t"], tr[axis], style, color=tp.PALETTE[k], linewidth=1.4, alpha=0.9,
                     label=name if axis == "ax" else None)
    ax3.set_xlabel("sample in the window")
    ax3.set_ylabel("acceleration / 1.5 g")
    ax3.set_title("one window per activity (solid ax, dashed ay, dotted az)",
                  fontsize=9, color=tp.MUTED)
    ax3.legend(fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png)
    tp.plt.close(fig)


def main():
    if not all(os.path.exists(p) for p in (XOR, IRIS, ENERGY, ANOMALY, GBDT, MAZE, DQN_MAZE, HAR)):
        sys.exit("build first: cargo build --release --examples --features std")
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
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([ANOMALY, tmp], check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "anomaly_scores.png")
        anomaly(tmp, png)
        print("wrote", os.path.relpath(png, ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([GBDT, tmp], check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "gbdt_regions.png")
        gbdt(tmp, png)
        print("wrote", os.path.relpath(png, ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([MAZE, tmp], check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "maze_paths.png")
        maze(tmp, png)
        print("wrote", os.path.relpath(png, ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([DQN_MAZE, tmp], check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "dqn_maze_paths.png")
        maze(tmp, png, "dqn_maze",
             "tinymind-rs: a deep Q-network finds the way out of a six-room maze (Q16.16)")
        print("wrote", os.path.relpath(png, ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([HAR, tmp, os.path.join(tmp, "har.csv")], check=True, stdout=subprocess.DEVNULL)
        png = os.path.join(OUT, "har_activity.png")
        har(tmp, png)
        print("wrote", os.path.relpath(png, ROOT))


if __name__ == "__main__":
    main()
