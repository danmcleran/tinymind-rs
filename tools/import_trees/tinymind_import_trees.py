"""
Copyright (c) 2026 Dan McLeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

"""
Tree-ensemble importer: scikit-learn / XGBoost / LightGBM -> tinymind (Rust).

A port of tinymind's apps/import_trees/tinymind_import_trees.py that writes a
Rust module for the `tinymind` crate instead of a C++ header. Reading the
model, quantizing features and thresholds, scaling the leaves and measuring
the base score are tinymind's code, unchanged, so a model imported here has
the same integer trees as the same model imported with tinymind; only the
output is Rust. The module is integer-only and `no_std`, and gives each model
two equivalent forms: a node pool walked by `tinymind::tree` (`forward`,
`predict`), and the same trees compiled to nested `if`s (`forward_inline`,
`predict_inline`, tinymind's QTreeInlineEval).

Supported models:

  scikit-learn  DecisionTree{Classifier,Regressor}, RandomForest{Classifier,
                Regressor}, ExtraTrees{Classifier,Regressor},
                GradientBoosting{Classifier,Regressor}
  XGBoost       XGBClassifier / XGBRegressor / Booster (gbtree)
  LightGBM      LGBMClassifier / LGBMRegressor / Booster

How the integer model is built (tinymind's rules):

  * Features are quantized per feature onto an integer grid,
    q = round(x / scale) + zero_point, in int8, uint8 or int16
    (FeatureQuant.integer for features that already are integers,
    FeatureQuant.from_data for a min/max calibration).
  * Each split threshold t becomes the integer threshold T for which
    q <= T reproduces the split on the grid: floor(t/scale + zp) for
    scikit-learn and LightGBM (x <= t), ceil(t/scale + zp) - 1 for XGBoost
    (x < t). A split that sends the whole grid one way is removed.
  * Classifier forests vote (one class per tree, ties to the lowest class),
    or with voting="soft" average class probabilities like scikit-learn's
    predict. Boosted models and regressors add int32 leaf values, scaled by a
    power of two chosen so the sum cannot overflow.
  * The model's base score is measured as the model's raw output minus the
    summed tree outputs on the calibration data, and must be the same for
    every sample. That also checks the tree parsing against the library.

Not supported: missing values (NaN), categorical splits, multi-output trees,
XGBoost dart / gblinear, LightGBM linear trees and rf mode.

Typical use:

    from tinymind_import_trees import FeatureQuant, export_trees
    model = export_trees(clf, X_train, "my_model.rs", "my_model",
                         quant=FeatureQuant.from_data(X_train, "int8"))
    print(model.report)
    model.verify(X_test)   # cargo-builds the module; checks it bit for bit

The module is formatted with rustfmt when it is on PATH.
"""

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


_DTYPES = {
    "int8": (-128, 127, "int8_t"),
    "uint8": (0, 255, "uint8_t"),
    "int16": (-32768, 32767, "int16_t"),
}

# The Rust type of each grid (this port; the C++ name above is kept unchanged).
_RUST_TYPES = {"int8": "i8", "uint8": "u8", "int16": "i16"}


# ----------------------------------------------------------------------------
# Feature quantization
# ----------------------------------------------------------------------------

@dataclass
class FeatureQuant:
    """Per-feature affine grid: q = round(x / scale) + zero_point, clamped.

    Rounding is ties-away-from-zero in float32, exactly as the generated
    quantize_features() computes it, so the host reference and the device agree
    bit for bit.
    """
    scale: np.ndarray
    zero_point: np.ndarray
    dtype: str = "int8"
    integer_valued: bool = False

    def __post_init__(self):
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be one of %s" % sorted(_DTYPES))
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.zero_point = np.asarray(self.zero_point, dtype=np.int64)
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.scale)):
            raise ValueError("every feature scale must be finite and > 0")

    @property
    def qmin(self) -> int:
        return _DTYPES[self.dtype][0]

    @property
    def qmax(self) -> int:
        return _DTYPES[self.dtype][1]

    @property
    def ctype(self) -> str:
        return _DTYPES[self.dtype][2]

    @property
    def num_features(self) -> int:
        return int(self.scale.shape[0])

    @classmethod
    def integer(cls, num_features: int, dtype: str = "int8") -> "FeatureQuant":
        """Features that already are integers in the dtype's range (scale 1,
        zero point 0). Splits are then reproduced exactly."""
        return cls(np.ones(num_features, np.float32), np.zeros(num_features, np.int64), dtype, True)

    @classmethod
    def from_data(cls, X, dtype: str = "int8") -> "FeatureQuant":
        """Asymmetric min/max calibration, one grid per feature."""
        X = np.asarray(X, dtype=np.float64)
        qmin, qmax, _ = _DTYPES[dtype]
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        span = hi - lo
        scale = np.where(span > 0, span / (qmax - qmin), 1.0).astype(np.float32)
        zp = np.clip(np.round(qmin - lo / scale.astype(np.float64)), qmin, qmax).astype(np.int64)
        return cls(scale, zp, dtype, False)

    def quantize(self, X) -> np.ndarray:
        """Quantize rows of X (float) the way the generated quantize_features() does."""
        X = np.asarray(X)
        if self.integer_valued:
            Xi = np.asarray(X, dtype=np.float64)
            if not np.all(Xi == np.round(Xi)):
                raise ValueError("FeatureQuant.integer: features must be integer-valued")
            if Xi.min() < self.qmin or Xi.max() > self.qmax:
                raise ValueError("FeatureQuant.integer: features outside the %s range" % self.dtype)
            return Xi.astype(np.int64)
        v = X.astype(np.float32) / self.scale
        half = np.float32(0.5)
        r = np.where(v >= 0, v + half, v - half).astype(np.float32)
        r = np.clip(r, np.float32(-1.0e9), np.float32(1.0e9))
        q = np.trunc(r).astype(np.int64) + self.zero_point
        return np.clip(q, self.qmin, self.qmax)

    def threshold(self, feature: int, t: float, strict: bool, calib=None):
        """Integer threshold for split `x < t` (strict) or `x <= t`.

        On the grid itself the split is q <= floor(v) (or ceil(v) - 1 when
        strict), v = t / scale + zero_point, which is exact for integer
        features. A rounding grid cell straddles t, though: the samples that
        quantize to cell round(v) can sit on both sides. When calib = (x_col,
        q_col, go_left) is given (the calibration column, its quantized values
        and the library's own routing of each sample), the threshold for that
        one cell follows the majority of the calibration samples in it, ties to
        the rule above. XGBoost takes its thresholds from data values, so this
        is what keeps samples lying on a threshold on the right side.

        Returns ("split", T), or ("left", None) / ("right", None) when the
        whole grid goes one way.
        """
        v = float(t) / float(self.scale[feature]) + float(self.zero_point[feature])
        T = (math.ceil(v) - 1) if strict else math.floor(v)
        if calib is not None and not self.integer_valued:
            _, q_col, go_left = calib
            c = int(np.clip(_round_half_away(v), self.qmin, self.qmax))
            in_cell = (q_col == c)
            if in_cell.any():
                lefts = int(np.count_nonzero(go_left[in_cell]))
                rights = int(np.count_nonzero(in_cell)) - lefts
                if lefts > rights:
                    T = c
                elif rights > lefts:
                    T = c - 1
        if T >= self.qmax:
            return ("left", None)
        if T < self.qmin:
            return ("right", None)
        return ("split", int(T))


# ----------------------------------------------------------------------------
# Intermediate tree form
# ----------------------------------------------------------------------------

@dataclass
class _FNode:
    """A float tree node: a split (feature, threshold, strict, left, right) or
    a leaf (value: float, or a vector of class scores)."""
    feature: int = -1
    threshold: float = 0.0
    strict: bool = False
    left: int = -1
    right: int = -1
    value: object = None


@dataclass
class _FTree:
    """precision mirrors how the library compares: "f32x" (scikit-learn: a
    float32 copy of x against a float64 threshold), "f32" (XGBoost: both
    float32) or "f64" (LightGBM)."""
    nodes: List[_FNode]
    root: int = 0
    precision: str = "f64"

    def leaf_for(self, x) -> object:
        if self.precision == "f32":
            x = np.asarray(x, dtype=np.float32)
        elif self.precision == "f32x":
            # Widen the float32 copy before comparing: under NumPy 2 promotion
            # a float32 compared with a Python float is compared in float32.
            x = np.asarray(x, dtype=np.float32).astype(np.float64)
        n = self.root
        while self.nodes[n].feature >= 0:
            node = self.nodes[n]
            t = np.float32(node.threshold) if self.precision == "f32" else node.threshold
            xv = x[node.feature]
            go_left = (xv < t) if node.strict else (xv <= t)
            n = node.left if go_left else node.right
        return self.nodes[n].value


@dataclass
class _QTree:
    """An integer tree in tinymind QTreeNode layout (pre-order, root 0)."""
    feature: List[int] = field(default_factory=list)
    threshold: List[int] = field(default_factory=list)
    left: List[int] = field(default_factory=list)
    right: List[int] = field(default_factory=list)
    leaf: List[int] = field(default_factory=list)

    def __len__(self):
        return len(self.feature)

    def eval(self, q) -> int:
        n = 0
        while self.feature[n] >= 0:
            n = self.left[n] if q[self.feature[n]] <= self.threshold[n] else self.right[n]
        return self.leaf[n]


def _routes_left(tree: _FTree, x_col, t: float, strict: bool) -> np.ndarray:
    """The library's own routing of a column of values at threshold t."""
    if tree.precision == "f64":
        xs, tt = np.asarray(x_col, dtype=np.float64), float(t)
    elif tree.precision == "f32":
        xs, tt = np.asarray(x_col, dtype=np.float32), np.float32(t)
    else:
        xs, tt = np.asarray(x_col, dtype=np.float32).astype(np.float64), float(t)
    return (xs < tt) if strict else (xs <= tt)


def _quantize_tree(tree: _FTree, quant: FeatureQuant, leaf_fn, X=None, Q=None) -> _QTree:
    """Map a float tree onto the integer grid, dropping splits that send the
    whole grid one way; leaf_fn turns a leaf value into its int32 leaf. X / Q
    (calibration samples, float and quantized) settle the straddling cell."""
    out = _QTree()

    def emit(n: int) -> int:
        while True:
            node = tree.nodes[n]
            if node.feature < 0:
                break
            calib = None
            if X is not None:
                col = X[:, node.feature]
                calib = (col, Q[:, node.feature], _routes_left(tree, col, node.threshold, node.strict))
            kind, T = quant.threshold(node.feature, node.threshold, node.strict, calib)
            if kind == "left":
                n = node.left
            elif kind == "right":
                n = node.right
            else:
                break
        idx = len(out.feature)
        node = tree.nodes[n]
        if node.feature < 0:
            out.feature.append(-1)
            out.threshold.append(0)
            out.left.append(-1)
            out.right.append(-1)
            out.leaf.append(int(leaf_fn(node.value)))
            return idx
        out.feature.append(node.feature)
        out.threshold.append(T)
        out.left.append(-1)
        out.right.append(-1)
        out.leaf.append(0)
        out.left[idx] = emit(node.left)
        out.right[idx] = emit(node.right)
        return idx

    emit(tree.root)
    return out


# ----------------------------------------------------------------------------
# Readers: library model -> float trees
# ----------------------------------------------------------------------------

def _sklearn_tree(est, value_fn) -> _FTree:
    t = est.tree_
    if t.n_outputs != 1:
        raise ValueError("multi-output trees are not supported")
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            nodes.append(_FNode(value=value_fn(t.value[i][0])))
        else:
            nodes.append(_FNode(int(t.feature[i]), float(t.threshold[i]), False,
                                int(t.children_left[i]), int(t.children_right[i])))
    return _FTree(nodes, precision="f32x")


def _xgb_trees(booster) -> List[_FTree]:
    import json
    cfg = json.loads(booster.save_config())
    name = cfg["learner"]["gradient_booster"]["name"]
    if name != "gbtree":
        raise ValueError("only XGBoost gbtree boosters are supported (got %s)" % name)
    names = booster.feature_names
    index = {n: i for i, n in enumerate(names)} if names else None
    trees = []
    for dump in booster.get_dump(dump_format="json"):
        root = json.loads(dump)
        nodes: List[_FNode] = []

        def walk(d) -> int:
            idx = len(nodes)
            nodes.append(_FNode())
            if "leaf" in d:
                nodes[idx] = _FNode(value=float(d["leaf"]))
                return idx
            if "split_condition" not in d or "categories" in d:
                raise ValueError("XGBoost categorical splits are not supported")
            split = d["split"]
            f = index[split] if index is not None else int(split.lstrip("f"))
            by_id = {c["nodeid"]: c for c in d["children"]}
            left = walk(by_id[d["yes"]])
            right = walk(by_id[d["no"]])
            nodes[idx] = _FNode(f, float(np.float32(d["split_condition"])), True, left, right)
            return idx

        walk(root)
        trees.append(_FTree(nodes, precision="f32"))
    return trees


def _lgb_trees(dump) -> List[_FTree]:
    trees = []
    for info in dump["tree_info"]:
        nodes: List[_FNode] = []

        def walk(d) -> int:
            idx = len(nodes)
            nodes.append(_FNode())
            if "leaf_value" in d:
                if "leaf_coeff" in d:
                    raise ValueError("LightGBM linear trees are not supported")
                nodes[idx] = _FNode(value=float(d["leaf_value"]))
                return idx
            if d.get("decision_type", "<=") != "<=":
                raise ValueError("LightGBM categorical splits are not supported")
            left = walk(d["left_child"])
            right = walk(d["right_child"])
            nodes[idx] = _FNode(int(d["split_feature"]), float(d["threshold"]), False, left, right)
            return idx

        walk(info["tree_structure"])
        trees.append(_FTree(nodes))
    return trees


@dataclass
class _Plan:
    """What a model becomes: a vote over trees with class leaves, or an
    additive ensemble of (class, float leaf) trees plus a measured base."""
    kind: str                       # "vote" or "additive"
    task: str                       # "classification", "binary", "multiclass", "regression"
    trees: List[_FTree]
    tree_class: List[int]
    num_outputs: int
    leaf_mult: float = 1.0          # float leaf value * leaf_mult = contribution
    output_divisor: float = 1.0     # final value = sum / output_divisor (forest averages)
    raw_fn: object = None           # X -> raw output (n,) or (n, K), for the base measurement
    classes: Optional[np.ndarray] = None
    original: object = None         # X -> the original model's class index or value


def _from_raw(raw_fn, task):
    """The library's own prediction, from its raw score: what predict() returns
    for the scikit-learn wrappers, and what a bare Booster's score means."""
    def f(X):
        raw = np.asarray(raw_fn(X), dtype=np.float64)
        if task == "multiclass":
            return np.argmax(raw.reshape(raw.shape[0], -1), axis=1)
        if task == "binary":
            return (raw.reshape(-1) > 0).astype(np.int64)
        return raw.reshape(-1)
    return f


def _plan(model, voting: str) -> _Plan:
    mod = type(model).__module__
    name = type(model).__name__

    if mod.startswith("sklearn"):
        if name in ("DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier"):
            ests = [model] if name == "DecisionTreeClassifier" else list(model.estimators_)
            K = len(model.classes_)
            if voting == "hard" or name == "DecisionTreeClassifier":
                trees = [_sklearn_tree(e, lambda v: int(np.argmax(v))) for e in ests]
                return _Plan("vote", "classification", trees, [0] * len(trees), K, classes=model.classes_)
            # Soft vote: one additive tree per (tree, class) with the class probability.
            trees, cls = [], []
            for e in ests:
                for k in range(K):
                    trees.append(_sklearn_tree(e, lambda v, k=k: float(v[k] / v.sum())))
                    cls.append(k)
            return _Plan("additive", "multiclass", trees, cls, K, classes=model.classes_,
                         raw_fn=lambda X: model.predict_proba(X) * len(ests))
        if name in ("DecisionTreeRegressor", "RandomForestRegressor", "ExtraTreesRegressor"):
            ests = [model] if name == "DecisionTreeRegressor" else list(model.estimators_)
            trees = [_sklearn_tree(e, lambda v: float(v[0])) for e in ests]
            return _Plan("additive", "regression", trees, [0] * len(trees), 1,
                         output_divisor=float(len(ests)),
                         raw_fn=lambda X: model.predict(X) * len(ests))
        if name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
            stages = model.estimators_
            K = stages.shape[1]
            trees, cls = [], []
            for stage in stages:
                for k in range(K):
                    trees.append(_sklearn_tree(stage[k], lambda v: float(v[0])))
                    cls.append(k)
            if name == "GradientBoostingRegressor":
                return _Plan("additive", "regression", trees, cls, 1, leaf_mult=model.learning_rate,
                             raw_fn=model.predict)
            task = "binary" if K == 1 else "multiclass"
            return _Plan("additive", task, trees, cls, K, leaf_mult=model.learning_rate,
                         raw_fn=model.decision_function, classes=model.classes_)
        raise ValueError("unsupported scikit-learn model: %s" % name)

    if mod.startswith("xgboost"):
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        import json
        cfg = json.loads(booster.save_config())
        lmp = cfg["learner"]["learner_model_param"]
        if int(lmp.get("num_target", "1")) > 1:
            raise ValueError("multi-target XGBoost models are not supported")
        K = max(1, int(lmp.get("num_class", "0")))
        obj = cfg["learner"]["objective"]["name"]
        trees = _xgb_trees(booster)
        cls = [i % K for i in range(len(trees))]
        raw = lambda X: booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)), output_margin=True)
        classes = getattr(model, "classes_", None)
        if K > 1:
            return _Plan("additive", "multiclass", trees, cls, K, raw_fn=raw, classes=classes,
                         original=_from_raw(raw, "multiclass"))
        task = "binary" if obj.startswith("binary:") else "regression"
        return _Plan("additive", task, trees, cls, 1, raw_fn=raw, classes=classes,
                     original=_from_raw(raw, task))

    if mod.startswith("lightgbm"):
        booster = model.booster_ if hasattr(model, "booster_") else model
        dump = booster.dump_model()
        if dump.get("average_output"):
            raise ValueError("LightGBM rf mode (average_output) is not supported")
        K = int(dump["num_tree_per_iteration"])
        trees = _lgb_trees(dump)
        cls = [i % K for i in range(len(trees))]
        raw = lambda X: booster.predict(np.asarray(X, dtype=np.float64), raw_score=True)
        classes = getattr(model, "classes_", None)
        obj = str(dump.get("objective", ""))
        if K > 1:
            return _Plan("additive", "multiclass", trees, cls, K, raw_fn=raw, classes=classes,
                         original=_from_raw(raw, "multiclass"))
        task = "binary" if obj.startswith(("binary", "cross_entropy")) else "regression"
        return _Plan("additive", task, trees, cls, 1, raw_fn=raw, classes=classes,
                     original=_from_raw(raw, task))

    raise ValueError("unsupported model type: %s.%s" % (mod, name))


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

@dataclass
class ExportedModel:
    """An exported model and its integer host reference."""
    path: str
    name: str
    quant: FeatureQuant
    kind: str
    task: str
    num_outputs: int
    qtrees: List[_QTree]
    tree_class: List[int]
    base: List[int]
    output_scale: float
    original_predict: object
    report: str = ""

    def predict_raw(self, X) -> np.ndarray:
        """Integer outputs of the exported model: class votes (vote models) or
        int32 logits / sums (additive models), shape (n, num_outputs)."""
        Q = self.quant.quantize(X)
        out = np.zeros((Q.shape[0], self.num_outputs), dtype=np.int64)
        for i, q in enumerate(Q):
            if self.kind == "vote":
                for t in self.qtrees:
                    out[i, t.eval(q)] += 1
            else:
                out[i] = self.base
                for t, c in zip(self.qtrees, self.tree_class):
                    out[i, c] += t.eval(q)
        return out

    def predict(self, X) -> np.ndarray:
        """Class index (classifiers) or float value (regressors), as the header computes it."""
        raw = self.predict_raw(X)
        if self.task == "regression":
            return raw[:, 0] / self.output_scale
        if self.task == "binary":
            return (raw[:, 0] > 0).astype(np.int64)
        return np.argmax(raw, axis=1)

    def agreement(self, X) -> float:
        """Fraction of samples where the exported model matches the original
        (classifiers), or the maximum absolute error (regressors)."""
        ours = self.predict(X)
        ref = np.asarray(self.original_predict(X))
        if self.task == "regression":
            return float(np.max(np.abs(ours - ref.reshape(-1)))) if len(ours) else 0.0
        return float(np.mean(ours == ref))

    def verify(self, X, crate_dir: Optional[str] = None, target_dir: Optional[str] = None) -> dict:
        """Build the module with cargo against the `tinymind` crate (default:
        this repository) and check that both forms, the pool walk and the
        nested ifs, are bit-exact with predict_raw / predict, and
        quantize_features with FeatureQuant.quantize. Raises AssertionError on
        any difference. target_dir (a cargo target directory) lets several
        verifications share one build of the crate."""
        return _verify(self, X, crate_dir, target_dir)


def _leaf_scale(plan: _Plan, base: np.ndarray) -> int:
    """Power-of-two leaf scale so no class's |base| + sum of |leaf| can exceed 2^30."""
    bound = np.abs(base).astype(np.float64).copy()
    for t, c in zip(plan.trees, plan.tree_class):
        leaves = [abs(float(n.value)) * plan.leaf_mult for n in t.nodes if n.feature < 0]
        bound[c] += max(leaves) if leaves else 0.0
    b = float(bound.max()) if bound.size else 0.0
    if b == 0.0:
        return 1
    k = int(math.floor(math.log2((1 << 30) / b)))
    return 2 ** max(min(k, 30), -30)


def _round_half_away(v: float) -> int:
    return int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)


def export_trees(model, X_calib, output_path: str, name: str,
                 quant: Optional[FeatureQuant] = None, voting: str = "hard") -> ExportedModel:
    """Convert `model` to a Rust module for the `tinymind` crate at
    `output_path`; `name` names the model in the module's docs.

    X_calib: representative inputs (float), used to measure the base score, to
    check the tree parsing, and for the agreement report. quant defaults to
    FeatureQuant.from_data(X_calib, "int8"). voting ("hard" / "soft") applies
    to random-forest and extra-trees classifiers.
    """
    if voting not in ("hard", "soft"):
        raise ValueError("voting must be 'hard' or 'soft'")
    X = np.asarray(X_calib, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X_calib must be a non-empty 2-D array")
    if np.isnan(X).any():
        raise ValueError("missing values (NaN) are not supported")
    quant = quant or FeatureQuant.from_data(X, "int8")
    if quant.num_features != X.shape[1]:
        raise ValueError("quant has %d features, X_calib has %d" % (quant.num_features, X.shape[1]))
    plan = _plan(model, voting)
    for t in plan.trees:
        for n in t.nodes:
            if n.feature >= X.shape[1]:
                raise ValueError("tree splits on feature %d, but X_calib has %d" % (n.feature, X.shape[1]))

    K = plan.num_outputs
    Qc = quant.quantize(X)
    if plan.kind == "vote":
        qtrees = [_quantize_tree(t, quant, int, X, Qc) for t in plan.trees]
        base = [0] * K
        output_scale = 1.0
    else:
        # Measure the base: raw output minus the float trees' sum, which must not vary.
        sums = np.zeros((X.shape[0], K))
        for t, c in zip(plan.trees, plan.tree_class):
            sums[:, c] += [float(t.leaf_for(x)) * plan.leaf_mult for x in X]
        raw = np.asarray(plan.raw_fn(X), dtype=np.float64).reshape(X.shape[0], -1)
        if raw.shape[1] != K:
            raise ValueError("model raw output has %d columns, expected %d" % (raw.shape[1], K))
        diff = raw - sums
        base_f = diff.mean(axis=0)
        spread = float(np.max(np.abs(diff - base_f))) if diff.size else 0.0
        tol = 1e-5 * max(1.0, float(np.max(np.abs(raw))))
        if spread > tol:
            raise ValueError("tree parsing does not reproduce the model's raw output "
                             "(base varies by %.3g); please report this model type" % spread)
        scale = _leaf_scale(plan, base_f)
        qtrees = [_quantize_tree(t, quant, lambda v: _round_half_away(float(v) * plan.leaf_mult * scale), X, Qc)
                  for t in plan.trees]
        base = [_round_half_away(b * scale) for b in base_f]
        output_scale = float(scale) * plan.output_divisor

    if plan.kind == "additive" and sum(len(t) for t in qtrees) > 32767:
        raise ValueError("ensemble has more than 32767 nodes; QTreeNode indices are int16")
    for t in qtrees:
        if len(t) > 32767:
            raise ValueError("a tree has more than 32767 nodes; QTreeNode indices are int16")

    if plan.original is not None:
        original = plan.original
    elif plan.task == "regression":
        original = model.predict
    elif plan.classes is not None:
        classes = list(plan.classes)
        original = lambda Xs: np.array([classes.index(c) for c in model.predict(Xs)])
    else:
        original = lambda Xs: np.asarray(model.predict(Xs)).astype(np.int64)

    exported = ExportedModel(output_path, name, quant, plan.kind, plan.task, K, qtrees,
                             plan.tree_class, base, output_scale, original)
    with open(output_path, "w") as fh:
        fh.write(_render(exported, type(model).__name__, plan.classes))
    _rustfmt(output_path)

    nodes = sum(len(t) for t in qtrees)
    agree = exported.agreement(X)
    what = ("max abs error %.6g" % agree) if plan.task == "regression" else ("%.2f%% agreement" % (100 * agree))
    exported.report = ("%s -> %s: %d trees, %d nodes (%d decision), %s %s inputs, %s with the original "
                       "on the %d calibration samples"
                       % (type(model).__name__, output_path, len(qtrees), nodes,
                          sum(f >= 0 for t in qtrees for f in t.feature), quant.num_features, quant.dtype,
                          what, X.shape[0]))
    return exported




# ----------------------------------------------------------------------------
# Rust rendering (this port; tinymind renders a C++ header here)
# ----------------------------------------------------------------------------

def _f32_literal(v) -> str:
    """The shortest decimal that reads back as the same float32."""
    s = str(np.float32(v))
    assert np.float32(s) == np.float32(v)
    return s if ("." in s or "e" in s) else s + ".0"


def _rust_array(values) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _pool_rows(t: _QTree, offset: int = 0) -> str:
    rows = []
    for f, th, l, r, v in zip(t.feature, t.threshold, t.left, t.right, t.leaf):
        if f < 0:
            rows.append("    leaf(%d)," % v)
        else:
            rows.append("    split(%d, %d, %d, %d)," % (f, th, l + offset, r + offset))
    return "\n".join(rows)


def _inline_body(t: _QTree, n: int = 0) -> str:
    """Tree t below node n as a nested if/else expression."""
    if t.feature[n] < 0:
        return str(t.leaf[n])
    return "if x[%d] <= %d { %s } else { %s }" % (
        t.feature[n], t.threshold[n], _inline_body(t, t.left[n]), _inline_body(t, t.right[n]))


def _render(m: ExportedModel, model_name: str, classes) -> str:
    q = m.quant
    rt = _RUST_TYPES[q.dtype]
    K, T, F = m.num_outputs, len(m.qtrees), q.num_features
    vote = m.kind == "vote"
    out = []
    w = out.append

    if vote:
        what = "a vote: each tree names a class, and `forward` counts the votes"
    elif m.task == "regression":
        what = "an int32 sum; `raw / OUTPUT_SCALE` is the model's prediction"
    else:
        what = "int32 raw scores (log-odds or margins times `OUTPUT_SCALE`)"
    w("//! `%s`: generated by tinymind-rs tools/import_trees/tinymind_import_trees.py\n" % m.name)
    w("//! from a %s. Do not edit; regenerate from the model instead.\n//!\n" % model_name)
    w("//! Integer-only and `no_std`. Inputs are `%s` on each feature's grid\n" % rt)
    w("//! (`quantize_features` converts a float sample).\n//!\n//! The output is %s.\n//!\n" % what)
    w("//! Two equivalent forms: `forward`/`predict` walk the node pool with\n"
      "//! `tinymind::tree` (smaller code), `forward_inline`/`predict_inline` run the\n"
      "//! same trees compiled to nested `if`s (faster, larger). Both give the same\n"
      "//! results.\n\n")
    w("// Not every program uses every item, and the trees are written as trained:\n"
      "// a split can have two equal leaves, and an `else` holds a nested `if`.\n"
      "#![allow(dead_code, clippy::if_same_then_else, clippy::collapsible_else_if)]\n\n")
    w("use tinymind::tree::{%s, QTreeNode};\n\n" % ("QDecisionTree" if vote else "QGbdt"))
    w("/// The input feature type.\npub type InputType = %s;\n" % rt)
    w("/// Number of input features.\npub const NUM_FEATURES: usize = %d;\n" % F)
    w("/// Number of outputs (%s).\npub const NUM_OUTPUTS: usize = %d;\n"
      % ("classes" if vote or m.task == "multiclass" else "one raw score", K))
    w("/// Number of trees.\npub const NUM_TREES: usize = %d;\n\n" % T)

    integer_note = " The features are integers already: pass them as they are." if q.integer_valued else ""
    w("/// Each feature's grid: `q = round(x / scale) + zero_point`, ties away from\n"
      "/// zero, clamped to `[%d, %d]`.%s\n" % (q.qmin, q.qmax, integer_note))
    w("pub const FEATURE_SCALE: [f32; NUM_FEATURES] = [%s];\n" % ", ".join(_f32_literal(s) for s in q.scale))
    w("/// Each feature's zero point.\npub const FEATURE_ZERO_POINT: [i32; NUM_FEATURES] = %s;\n\n"
      % _rust_array(q.zero_point))
    w("""/// Quantize one float sample onto the feature grids (tinymind's
/// `quantizeFeatures`), bit for bit as the importer's reference does.
pub fn quantize_features(x: &[f32; NUM_FEATURES]) -> [InputType; NUM_FEATURES] {
    let mut q = [0; NUM_FEATURES];
    let grid = FEATURE_SCALE.iter().zip(&FEATURE_ZERO_POINT);
    for ((q, &x), (&scale, &zero_point)) in q.iter_mut().zip(x).zip(grid) {
        let v = x / scale;
        let r = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
        let i = r.clamp(-1.0e9, 1.0e9) as i32 + zero_point;
        *q = i.clamp(%d, %d) as InputType;
    }
    q
}

const fn split(feature: i16, threshold: InputType, left: i16, right: i16) -> QTreeNode<InputType> {
    QTreeNode { feature, threshold, left, right, leaf_value: 0 }
}

const fn leaf(value: i32) -> QTreeNode<InputType> {
    QTreeNode { feature: -1, threshold: 0, left: -1, right: -1, leaf_value: value }
}

/// The index of the largest value, the lowest on a tie.
fn argmax(v: &[i32; NUM_OUTPUTS]) -> usize {
    let mut best = (0, v[0]);
    for (c, &s) in v.iter().enumerate().skip(1) {
        if s > best.1 {
            best = (c, s);
        }
    }
    best.0
}

""" % (q.qmin, q.qmax))

    if vote:
        for k, t in enumerate(m.qtrees):
            w("static TREE_%d: [QTreeNode<InputType>; %d] = [\n%s\n];\n" % (k, len(t), _pool_rows(t)))
            w("const _: () = assert!(QDecisionTree::<'static, InputType, %d, NUM_FEATURES>::new(&TREE_%d).is_valid());\n\n"
              % (len(t), k))
        w("""fn vote(votes: &mut [i32; NUM_OUTPUTS], class: i32) {
    if let Some(v) = usize::try_from(class).ok().and_then(|c| votes.get_mut(c)) {
        *v += 1;
    }
}

/// Votes per class, one per tree (walking the node pools).
pub fn forward(x: &[InputType; NUM_FEATURES]) -> [i32; NUM_OUTPUTS] {
    let mut votes = [0; NUM_OUTPUTS];
""")
        for k, t in enumerate(m.qtrees):
            w("    vote(&mut votes, QDecisionTree::<'_, InputType, %d, NUM_FEATURES>::new(&TREE_%d).predict(x));\n"
              % (len(t), k))
        w("    votes\n}\n\n")
        w("/// Votes per class, one per tree (nested `if`s).\n"
          "pub fn forward_inline(x: &[InputType; NUM_FEATURES]) -> [i32; NUM_OUTPUTS] {\n"
          "    let mut votes = [0; NUM_OUTPUTS];\n")
        for k in range(T):
            w("    vote(&mut votes, tree_%d(x));\n" % k)
        w("    votes\n}\n\n")
    else:
        pool, roots = [], []
        for t in m.qtrees:
            roots.append(sum(len(u) for u in m.qtrees[:len(roots)]))
        rows = "\n".join(_pool_rows(t, off) for t, off in zip(m.qtrees, roots))
        N = sum(len(t) for t in m.qtrees)
        w("static NODES: [QTreeNode<InputType>; %d] = [\n%s\n];\n" % (N, rows))
        w("static TREE_ROOT: [i16; NUM_TREES] = %s;\n" % _rust_array(roots))
        w("static TREE_CLASS: [i16; NUM_TREES] = %s;\n" % _rust_array(m.tree_class))
        w("/// Each output's starting value (the model's base score, scaled).\n"
          "pub static BASE_SCORE: [i32; NUM_OUTPUTS] = %s;\n\n" % _rust_array(m.base))
        w("""/// The ensemble over the shared node pool.
pub static MODEL: QGbdt<'static, InputType, NUM_TREES, NUM_OUTPUTS, %d, NUM_FEATURES> = QGbdt {
    nodes: &NODES,
    tree_root: &TREE_ROOT,
    tree_class: &TREE_CLASS,
    base_score: Some(&BASE_SCORE),
};
const _: () = assert!(MODEL.is_valid());

/// Raw int32 outputs (walking the node pool).
pub fn forward(x: &[InputType; NUM_FEATURES]) -> [i32; NUM_OUTPUTS] {
    MODEL.forward(x)
}

/// Raw int32 outputs (nested `if`s).
pub fn forward_inline(x: &[InputType; NUM_FEATURES]) -> [i32; NUM_OUTPUTS] {
    let mut raw = BASE_SCORE;
""" % N)
        for k, c in enumerate(m.tree_class):
            w("    raw[%d] = raw[%d].wrapping_add(tree_%d(x));\n" % (c, c, k))
        w("    raw\n}\n\n")

    for k, t in enumerate(m.qtrees):
        w("#[rustfmt::skip]\nfn tree_%d(x: &[InputType; NUM_FEATURES]) -> i32 {\n    %s\n}\n\n"
          % (k, _inline_body(t)))

    if vote or m.task == "multiclass":
        rule = "most votes" if vote else "largest raw score"
        for suffix in ("", "_inline"):
            w("/// Predicted class index: %s, ties to the lowest class.\n" % rule)
            w("pub fn predict%s(x: &[InputType; NUM_FEATURES]) -> usize {\n    argmax(&forward%s(x))\n}\n\n"
              % (suffix, suffix))
    elif m.task == "binary":
        for suffix in ("", "_inline"):
            w("/// Predicted class index: 1 when the raw score is positive.\n")
            w("pub fn predict%s(x: &[InputType; NUM_FEATURES]) -> usize {\n    usize::from(forward%s(x)[0] > 0)\n}\n\n"
              % (suffix, suffix))
    if not vote:
        w("/// `raw / OUTPUT_SCALE` is the model's %s.\npub const OUTPUT_SCALE: f64 = %r;\n"
          % ("prediction" if m.task == "regression" else "raw score (log-odds or margin)", m.output_scale))

    if classes is not None and m.task != "regression":
        labels = list(classes)
        w("\n/// Number of classes.\npub const NUM_CLASSES: usize = %d;\n" % len(labels))
        if all(isinstance(c, (int, np.integer)) for c in labels):
            w("/// The original class labels, by predicted index.\npub const CLASS_LABELS: [i32; NUM_CLASSES] = %s;\n"
              % _rust_array(labels))
        else:
            w("// The original class labels, by predicted index: %s\n" % ", ".join(repr(c) for c in labels))
    return "".join(out)


def _rustfmt(path: str) -> None:
    """Format as `cargo fmt` would, so the file is stable in a formatted crate."""
    rustfmt = shutil.which("rustfmt")
    if rustfmt:
        subprocess.run([rustfmt, "--edition", "2024", path], check=True)


# ----------------------------------------------------------------------------
# Verification with cargo (this port; tinymind compiles its header with g++)
# ----------------------------------------------------------------------------

def _repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _verify(m: ExportedModel, X, crate_dir, target_dir) -> dict:
    X = np.asarray(X, dtype=np.float64)
    Q = m.quant.quantize(X)
    n, F = Q.shape
    expect_raw = m.predict_raw(X)
    expect_cls = m.predict(X) if m.task != "regression" else None
    rt = _RUST_TYPES[m.quant.dtype]
    float_ok = not m.quant.integer_valued
    main = ["#[path = %s]\nmod model;\n" % _rust_str(os.path.abspath(m.path)),
            "static Q: [[%s; %d]; %d] = [" % (rt, F, n)]
    main += ["    [%s]," % ", ".join(str(int(v)) for v in row) for row in Q]
    main.append("];")
    if float_ok:
        main.append("static X: [[f32; %d]; %d] = [" % (F, n))
        main += ["    [%s]," % ", ".join(_f32_literal(v) for v in row) for row in X]
        main.append("];")
    main.append("fn main() {\n    for s in 0..%d {\n        let x = &Q[s];" % n)
    for form in ("", "_inline"):
        main.append('        print!("F%s");' % form)
        main.append('        for v in model::forward%s(x) { print!(" {v}"); }' % form)
        if m.task != "regression":
            main.append('        print!(" | {}", model::predict%s(x));' % form)
        main.append("        println!();")
    if float_ok:
        main.append('        print!("Q");')
        main.append('        for v in model::quantize_features(&X[s]) { print!(" {v}"); }')
        main.append("        println!();")
    main.append("    }\n}")

    crate = os.path.abspath(crate_dir or _repo_root())
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "src"))
        with open(os.path.join(tmp, "Cargo.toml"), "w") as fh:
            fh.write('[package]\nname = "verify"\nversion = "0.0.0"\nedition = "2024"\n\n'
                     '[dependencies]\ntinymind = { path = %s }\n\n[workspace]\n' % _rust_str(crate))
        with open(os.path.join(tmp, "src", "main.rs"), "w") as fh:
            fh.write("\n".join(main) + "\n")
        env = dict(os.environ, CARGO_TARGET_DIR=target_dir or os.path.join(tmp, "target"))
        built = subprocess.run(["cargo", "run", "--release", "--quiet", "--offline"], cwd=tmp, env=env,
                               capture_output=True, text=True)
        if built.returncode != 0:
            raise AssertionError("cargo failed:\n" + built.stderr)
        text = built.stdout

    rows = text.split("\n")
    per = 2 + (1 if float_ok else 0)
    for s in range(n):
        block = rows[s * per:(s + 1) * per]
        for j, form in enumerate(("forward", "forward_inline")):
            parts = block[j].split()
            got = [int(v) for v in parts[1:1 + m.num_outputs]]
            assert got == [int(v) for v in expect_raw[s]], (
                "%s: sample %d raw %s, host reference %s" % (form, s, got, list(expect_raw[s])))
            if expect_cls is not None:
                assert int(parts[-1]) == int(expect_cls[s]), (
                    "%s: sample %d class %s, host reference %d" % (form, s, parts[-1], expect_cls[s]))
        if float_ok:
            got_q = [int(v) for v in block[-1].split()[1:]]
            assert got_q == [int(v) for v in Q[s]], (
                "quantize_features: sample %d gave %s, host reference %s" % (s, got_q, list(Q[s])))
    return {"samples": n, "forms": ["forward", "forward_inline"], "agreement": m.agreement(X)}


def _rust_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
