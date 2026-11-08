#!/usr/bin/env python3
"""Parity of this importer with tinymind (C++)'s (apps/import_trees, from TINYMIND_DIR,
default ../tinymind), and of the generated Rust with the tinymind (C++) runtime.

For every model family (and features far from zero), both importers export the same trained model on the
same feature grid. The integer models must be identical: every tree's nodes,
thresholds and leaves, the tree classes, the base scores, the output scale and
the feature grids. Then each output is checked on its own runtime against the
shared integer reference: tinymind (C++)'s header with g++ under all three of its
evaluation policies, and the Rust module with cargo in both of its forms (pool
walk and nested ifs), including quantize_features. Equal references and two
runtimes that each match theirs make the Rust and the C++ bit-identical.

  python3 -I -B tools/import_trees/test_parity.py

Needs numpy and scikit-learn; XGBoost and LightGBM models are checked when
those packages are installed (CI installs them), and skipped otherwise.
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TINYMIND_DIR = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CPP = load(os.path.join(TINYMIND_DIR, "apps", "import_trees", "tinymind_import_trees.py"), "tinymind_trees_cpp")
RS = load(os.path.join(HERE, "tinymind_import_trees.py"), "tinymind_trees_rs")


def data(n, features, classes, seed, integer=False):
    """Deterministic samples around zero, labels from a fixed rule."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features))
    if integer:
        X = np.round(X * 30.0)
    score = X @ np.linspace(1.0, -0.5, features) + 0.5 * np.sin(3.0 * X[:, 0])
    if classes == 0:
        return X, score
    edges = np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1])
    return X, np.searchsorted(edges, score)


def models():
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                                  GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    yield "DecisionTreeClassifier", DecisionTreeClassifier(max_depth=5, random_state=0), 3, {}
    yield "DecisionTreeRegressor", DecisionTreeRegressor(max_depth=5, random_state=0), 0, {}
    yield "RandomForestClassifier hard", RandomForestClassifier(n_estimators=12, max_depth=4, random_state=0), 3, {}
    yield "RandomForestClassifier soft", RandomForestClassifier(n_estimators=6, max_depth=3, random_state=0), 3, {"voting": "soft"}
    yield "RandomForestRegressor", RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0), 0, {}
    yield "ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=10, max_depth=4, random_state=0), 4, {}
    yield "ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=8, max_depth=4, random_state=0), 0, {}
    yield "GradientBoostingClassifier multiclass", GradientBoostingClassifier(n_estimators=15, max_depth=3, random_state=0), 3, {}
    yield "GradientBoostingClassifier binary", GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0), 2, {}
    yield "GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0), 0, {}
    try:
        import xgboost as xgb
    except ImportError:
        print("skip       XGBoost (not installed)")
    else:
        kw = dict(n_estimators=15, max_depth=3, random_state=0, n_jobs=1)
        yield "XGBClassifier multiclass", xgb.XGBClassifier(**kw), 3, {}
        yield "XGBClassifier binary", xgb.XGBClassifier(**kw), 2, {}
        yield "XGBRegressor", xgb.XGBRegressor(**kw), 0, {}
    try:
        import lightgbm as lgb
    except ImportError:
        print("skip       LightGBM (not installed)")
    else:
        kw = dict(n_estimators=15, num_leaves=8, random_state=0, n_jobs=1, verbose=-1)
        yield "LGBMClassifier multiclass", lgb.LGBMClassifier(**kw), 3, {}
        yield "LGBMClassifier binary", lgb.LGBMClassifier(**kw), 2, {}
        yield "LGBMRegressor", lgb.LGBMRegressor(**kw), 0, {}
    # Features whose range excludes 0, as most sensor data does: the grid's zero
    # point lies outside the storage range. The second also has one feature so
    # far from 0 that its grid is widened to include 0.
    yield "RandomForestClassifier offset", RandomForestClassifier(n_estimators=8, max_depth=4, random_state=0), 3, {"offset": 50.0}
    yield "GradientBoostingRegressor far", GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=0), 0, {"offset": 1.0e6}


def same_integer_model(a, b):
    """Every integer the two importers produce, compared field by field."""
    assert (a.kind, a.task, a.num_outputs) == (b.kind, b.task, b.num_outputs)
    assert a.output_scale == b.output_scale
    assert list(a.tree_class) == list(b.tree_class)
    assert [int(v) for v in a.base] == [int(v) for v in b.base]
    assert a.quant.dtype == b.quant.dtype and a.quant.integer_valued == b.quant.integer_valued
    assert np.array_equal(a.quant.scale, b.quant.scale)
    assert np.array_equal(a.quant.zero_point, b.quant.zero_point)
    assert len(a.qtrees) == len(b.qtrees)
    for s, t in zip(a.qtrees, b.qtrees):
        for f in ("feature", "threshold", "left", "right", "leaf"):
            assert [int(v) for v in getattr(s, f)] == [int(v) for v in getattr(t, f)], f


def main():
    failures = 0
    cases = 0
    cxx_inc = [os.path.join(TINYMIND_DIR, d) for d in ("cpp", os.path.join("cpp", "include"), "include")]
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "target")
        for k, (name, model, classes, kw) in enumerate(models()):
            # Calibrated grids in each type, and integer features, by turns.
            dtype = ("int8", "uint8", "int16", "integer")[k % 4]
            kw = dict(kw)
            offset = kw.pop("offset", 0.0)
            if offset:
                dtype = "int8"
            X, y = data(240, 5, classes, seed=k, integer=(dtype == "integer"))
            if offset == 1.0e6:
                X[:, 0] = offset + X[:, 0] * 1.0e-4   # tiny span, far from 0
                X[:, 1:] += 50.0
            elif offset:
                X += offset
            model.fit(X, y)
            quant = (RS.FeatureQuant.integer(5, "int16") if dtype == "integer"
                     else RS.FeatureQuant.from_data(X, dtype))
            cquant = CPP.FeatureQuant(quant.scale, quant.zero_point, quant.dtype, quant.integer_valued)
            try:
                c = CPP.export_trees(model, X, os.path.join(tmp, "m%d.hpp" % k), "m%d" % k, quant=cquant, **kw)
                r = RS.export_trees(model, X, os.path.join(tmp, "m%d.rs" % k), "m%d" % k, quant=quant, **kw)
                same_integer_model(c, r)
                assert np.array_equal(c.predict_raw(X), r.predict_raw(X))
                c.verify(X, include_dirs=cxx_inc)
                r.verify(X, target_dir=target)
                cases += 1
                print("case       %-40s %-7s %3d trees  IDENTICAL" % (name, dtype, len(r.qtrees)))
            except Exception as e:  # report every case, then fail
                failures += 1
                print("case       %-40s %-7s DIFFERENT: %s" % (name, dtype, e))
    print("%d models identical, %d different" % (cases, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
