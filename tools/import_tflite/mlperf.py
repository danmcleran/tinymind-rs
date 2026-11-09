#!/usr/bin/env python3
"""Translate an MLPerf Tiny reference model with the TFLite translator and check
it against tinymind (C++).

  python3 -I tools/import_tflite/mlperf.py kws [--out mlperf_kws.rs] [--model path.tflite]

Fetches the model from github.com/mlcommons/tiny at the commit tinymind (C++)'s
benchmark pins (benchmarks/framework_comparison_f411/gen/make_mlperf.py) and
checks its SHA-256, unless --model gives a local copy. Translates it, draws the
same seeded random int8 test inputs as make_mlperf.py, runs the generated
module with cargo against this repository's crate, and compares the FNV-1a hash
of every output byte with tinymind (C++)'s expected hash (TM_MLPERF_*_HASH in
fw/backend.hpp, from TINYMIND_DIR, default ../tinymind). tinymind (C++) and
TFLite's reference kernels produce that hash, on the host and on the F411.

The models are Apache-2.0 (MLCommons); neither they nor the modules generated
from them are committed here. Downloads are cached in target/tflite-models.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tinymind_import_tflite as tt  # noqa: E402

# As tinymind's gen/make_mlperf.py pins them.
MLPERF_TINY_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
MODELS = {
    "kws": ("benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite",
            "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"),
    "ic": ("benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite",
           "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"),
}
N_TEST = 8
NOTE = ("Derived from an MLPerf Tiny reference model (github.com/mlcommons/tiny,\n"
        f"commit {MLPERF_TINY_COMMIT}), Apache License 2.0, Copyright MLCommons.\n"
        "Not committed: regenerate with tools/import_tflite/mlperf.py.")


def fetch(name: str, local: str = None) -> str:
    path, sha = MODELS[name]
    if local is None:
        cache = os.path.join(ROOT, "target", "tflite-models")
        os.makedirs(cache, exist_ok=True)
        local = os.path.join(cache, os.path.basename(path))
        if not os.path.exists(local):
            url = f"https://raw.githubusercontent.com/mlcommons/tiny/{MLPERF_TINY_COMMIT}/{path}"
            print("fetching", url)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with open(local + ".part", "wb") as f:
                f.write(data)
            os.replace(local + ".part", local)
    got = hashlib.sha256(open(local, "rb").read()).hexdigest()
    if got != sha:
        sys.exit(f"{local}: sha256 {got}, expected {sha}")
    return local


def test_inputs(name: str, n_in: int) -> np.ndarray:
    """make_mlperf.py's seeded random int8 inputs."""
    rng = np.random.default_rng(1234 + len(name))
    return rng.integers(-128, 128, (N_TEST, n_in), dtype=np.int64).astype(np.int8)


def expected_hash(name: str) -> int:
    tm = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
    backend = os.path.join(tm, "benchmarks", "framework_comparison_f411", "fw", "backend.hpp")
    m = re.search(r"#define TM_MLPERF_%s_HASH (0x[0-9A-Fa-f]+)u" % name.upper(), open(backend).read())
    if not m:
        sys.exit(f"TM_MLPERF_{name.upper()}_HASH not found in {backend}")
    return int(m.group(1), 16)


def _rust_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_module(module: str, xs: np.ndarray, input_shape: list, target_dir: str = None,
               output_map: bool = False) -> bytes:
    """Every output byte of the module's forward on each input, in order.
    `input_shape` is the model's input shape without the batch axis;
    `output_map` says the output is a feature map (NHWC) rather than a vector."""
    shaped = xs.reshape([len(xs)] + list(input_shape))
    main = [f"#[path = {_rust_str(os.path.abspath(module))}]\nmod model;\n",
            f"static X: {tt._rust_type([len(xs)] + list(input_shape))} = {tt._rust_nested(shaped)};",
            "fn main() {",
            "    for x in &X {",
            "        let y = model::forward(x);",
            ("        for v in y.as_flattened().as_flattened() {" if output_map else "        for v in y {"),
            "            print!(\"{v} \");",
            "        }",
            "        println!();",
            "    }",
            "}"]
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "src"))
        with open(os.path.join(tmp, "Cargo.toml"), "w") as fh:
            fh.write('[package]\nname = "verify"\nversion = "0.0.0"\nedition = "2024"\n\n'
                     f'[dependencies]\ntinymind = {{ path = {_rust_str(ROOT)} }}\n\n[workspace]\n')
        with open(os.path.join(tmp, "src", "main.rs"), "w") as fh:
            fh.write("\n".join(main) + "\n")
        env = dict(os.environ, CARGO_TARGET_DIR=target_dir or os.path.join(tmp, "target"))
        built = subprocess.run(["cargo", "run", "--release", "--quiet", "--offline"], cwd=tmp, env=env,
                               capture_output=True, text=True)
        if built.returncode != 0:
            sys.exit("cargo failed:\n" + built.stderr)
    rows = [[int(v) for v in row.split()] for row in built.stdout.strip().split("\n")]
    return np.array(rows, dtype=np.int8).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(MODELS))
    ap.add_argument("--model", help="a local copy of the model (checked against the pinned SHA-256)")
    ap.add_argument("--out", help="where to write the module (default: a temporary file)")
    ap.add_argument("--data", help="also write the test inputs and the expected hash as Rust "
                                   "(for the board benchmark, bench/src/gen)")
    ap.add_argument("--target-dir", help="a cargo target directory to share between runs")
    args = ap.parse_args()
    model = fetch(args.name, args.model)
    with tempfile.TemporaryDirectory() as tmp:
        out = args.out or os.path.join(tmp, f"mlperf_{args.name}.rs")
        tr = tt.translate(model, out, f"mlperf_{args.name}", NOTE)
        n_in = int(np.prod(tr.input.shape))
        xs = test_inputs(args.name, n_in)
        outputs = run_module(out, xs, tr.input.shape[1:], args.target_dir)
    got, want = tt.fnv1a(outputs), expected_hash(args.name)
    if args.data:
        shape = [N_TEST] + list(tr.input.shape[1:])
        with open(args.data, "w") as f:
            f.write("// Generated by tools/import_tflite/mlperf.py. Do not edit, do not commit.\n"
                    "// Test inputs: make_mlperf.py's seeded random int8 values, not dataset samples.\n"
                    f"pub const N_TEST: usize = {N_TEST};\n"
                    f"pub static X: {tt._rust_type(shape)} = {tt._rust_nested(xs.reshape(shape))};\n"
                    f"/// tinymind (C++)'s expected output hash (TM_MLPERF_{args.name.upper()}_HASH).\n"
                    f"pub const EXPECT_HASH: u32 = 0x{want:08X};\n")
    print(f"mlperf {args.name}: {len(tr.ops)} operators ({', '.join(sorted(set(tr.ops)))}), "
          f"{N_TEST} inputs, hash 0x{got:08X}, tinymind (C++) 0x{want:08X}: "
          f"{'IDENTICAL' if got == want else 'DIFFERENT'}")
    sys.exit(0 if got == want else 1)


if __name__ == "__main__":
    main()
