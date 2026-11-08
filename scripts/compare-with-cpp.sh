#!/usr/bin/env bash
# Build tinymind's C++ examples and their Rust ports (xor in Q16.16 and Q8.8,
# iris, energy_efficiency, the int8 pytorch_quant xor, gbdt_tabular_int8), plus
# the C++ programs in scripts/cpp/ (deep networks, inference-only networks, the
# int8 softmax sweep, the tree sweep, the PyTorch importer's models), run both sides, and cmp every output file. Set TINYMIND_DIR to the
# tinymind checkout (default ../tinymind). Set KEEP_CPP_OUTPUT=dir to keep the
# C++ outputs, the only valid source for tests/golden.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYMIND_DIR="$(cd "${TINYMIND_DIR:-$ROOT/../tinymind}" && pwd)"
REF="$(tr -d '[:space:]' < "$ROOT/TINYMIND_REF")"
HEAD="$(git -C "$TINYMIND_DIR" rev-parse HEAD)"
echo "tinymind: $TINYMIND_DIR at ${HEAD:0:7}; TINYMIND_REF ${REF:0:7}"
if [ "$HEAD" != "$REF" ]; then
  echo "note: comparing against tinymind ${HEAD:0:7}, not the pinned ${REF:0:7} (see scripts/upstream-changes.sh)"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CXXFLAGS=(-O2 -Wall -Wextra -Werror -Wpedantic -DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1)

(cd "$ROOT" && cargo build --release --examples --features std)

status=0
# Each format: the Rust --format value, the Q-format bit count for xornet.h, and
# the tinymind table macros it needs.
for spec in "q16.16 16 16_16" "q8.8 8 8_8"; do
  read -r format bits tables <<< "$spec"
  echo "== $format"
  mkdir -p "$TMP/$format/src" "$TMP/$format/cpp" "$TMP/$format/rust"

  # The C++ example is Q16.16. For other formats, build a copy whose xornet.h
  # sets both bit counts; everything else is tinymind's code unchanged.
  # Glob, not a file list: older tinymind revisions also have examples/xor/xornet.cpp.
  cp "$TINYMIND_DIR"/examples/xor/*.cpp "$TINYMIND_DIR"/examples/xor/*.h "$TMP/$format/src/"
  hdr="$TMP/$format/src/xornet.h"
  sed -i -E "s/^(static const size_t NUMBER_OF_(FIXED|FRACTIONAL)_BITS = )16;$/\1$bits;/" "$hdr"
  n="$(grep -cE "^static const size_t NUMBER_OF_(FIXED|FRACTIONAL)_BITS = $bits;$" "$hdr")"
  if [ "$n" != 2 ]; then
    echo "xornet.h: expected 2 bit-count lines set to $bits, found $n (did tinymind change the header?)" >&2
    exit 1
  fi

  g++ "${CXXFLAGS[@]}" "-DTINYMIND_USE_SIGMOID_$tables=1" "-DTINYMIND_USE_TANH_$tables=1" \
    -o "$TMP/$format/cpp/xor" "$TMP/$format/src/"*.cpp "$TINYMIND_DIR/cpp/lookupTables.cpp" \
    -I"$TMP/$format/src" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include/"
  (cd "$TMP/$format/cpp" && ./xor > stdout.txt)
  "$ROOT/target/release/examples/xor" --format "$format" "$TMP/$format/rust" > "$TMP/$format/rust/stdout.txt"

  for f in stdout.txt nn_fixed_xor.txt xor_training.csv xor_decision_surface.csv; do
    if cmp -s "$TMP/$format/cpp/$f" "$TMP/$format/rust/$f"; then
      echo "IDENTICAL  $format $f"
    else
      echo "DIFFERENT  $format $f"
      status=1
    fi
  done
done

echo "== iris"
# The data file must be tinymind's, byte for byte: both programs read it.
if cmp -s "$ROOT/data/iris.data" "$TINYMIND_DIR/examples/iris/iris.data"; then
  echo "IDENTICAL  iris data/iris.data vs tinymind examples/iris/iris.data"
else
  echo "DIFFERENT  iris data/iris.data vs tinymind examples/iris/iris.data"
  status=1
fi
mkdir -p "$TMP/iris/cpp" "$TMP/iris/rust"
# Flags from examples/iris/Makefile's release target.
g++ -g -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/iris/iris" \
  "$TINYMIND_DIR/examples/iris/iris.cpp" "$TINYMIND_DIR/cpp/lookupTables.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include/" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1
cp "$TINYMIND_DIR/examples/iris/iris.data" "$TMP/iris/cpp/"
(cd "$TMP/iris/cpp" && ../iris > stdout.txt)
rm "$TMP/iris/cpp/iris.data"
"$ROOT/target/release/examples/iris" "$TMP/iris/rust" "$ROOT/data/iris.data" > "$TMP/iris/rust/stdout.txt"
for f in stdout.txt iris_loss.csv iris_scatter.csv iris_confusion.csv; do
  if cmp -s "$TMP/iris/cpp/$f" "$TMP/iris/rust/$f"; then
    echo "IDENTICAL  iris $f"
  else
    echo "DIFFERENT  iris $f"
    status=1
  fi
done

echo "== energy_efficiency"
# The data file must be tinymind's, byte for byte (CRLF in both working trees).
if cmp -s "$ROOT/data/ENB2012_data.csv" "$TINYMIND_DIR/examples/energy_efficiency/ENB2012_data.csv"; then
  echo "IDENTICAL  energy data/ENB2012_data.csv vs tinymind examples/energy_efficiency/ENB2012_data.csv"
else
  echo "DIFFERENT  energy data/ENB2012_data.csv vs tinymind examples/energy_efficiency/ENB2012_data.csv"
  status=1
fi
mkdir -p "$TMP/energy/cpp" "$TMP/energy/rust"
# Flags from examples/energy_efficiency/Makefile's release target.
g++ -g -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/energy/energy_efficiency" \
  "$TINYMIND_DIR/examples/energy_efficiency/energy_efficiency.cpp" "$TINYMIND_DIR/cpp/lookupTables.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include/" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1
cp "$TINYMIND_DIR/examples/energy_efficiency/ENB2012_data.csv" "$TMP/energy/cpp/"
(cd "$TMP/energy/cpp" && ../energy_efficiency > stdout.txt)
rm "$TMP/energy/cpp/ENB2012_data.csv"
"$ROOT/target/release/examples/energy_efficiency" "$TMP/energy/rust" "$ROOT/data/ENB2012_data.csv" > "$TMP/energy/rust/stdout.txt"
for f in stdout.txt energy_loss.csv energy_pred.csv; do
  if cmp -s "$TMP/energy/cpp/$f" "$TMP/energy/rust/$f"; then
    echo "IDENTICAL  energy $f"
  else
    echo "DIFFERENT  energy $f"
    status=1
  fi
done

echo "== int8_xor"
mkdir -p "$TMP/int8_xor/cpp" "$TMP/int8_xor/rust"
# Flags from examples/pytorch_quant/xor/Makefile's release target.
g++ -ggdb -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/int8_xor/xor_quant" \
  "$TINYMIND_DIR/examples/pytorch_quant/xor/xor_quant.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
(cd "$TMP/int8_xor/cpp" && ../xor_quant > stdout.txt)
"$ROOT/target/release/examples/int8_xor" "$TMP/int8_xor/rust" > "$TMP/int8_xor/rust/stdout.txt"
for f in stdout.txt xor_decision_surface.csv; do
  if cmp -s "$TMP/int8_xor/cpp/$f" "$TMP/int8_xor/rust/$f"; then
    echo "IDENTICAL  int8_xor $f"
  else
    echo "DIFFERENT  int8_xor $f"
    status=1
  fi
done

echo "== deep_mlp"
# tinymind's NeuralNetwork with one, two and three hidden layers, trained; every weight
# hashed after every step. tests/deep_mlp.rs checks the Rust network against the
# committed golden, so the golden must still be what the C++ prints.
mkdir -p "$TMP/deep_mlp"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/deep_mlp/deep_mlp" "$ROOT/scripts/cpp/deep_mlp.cpp" \
  "$TINYMIND_DIR/cpp/lookupTables.cpp" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1 \
  -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_TANH_8_8=1
"$TMP/deep_mlp/deep_mlp" > "$TMP/deep_mlp/stdout.txt"
if cmp -s "$TMP/deep_mlp/stdout.txt" "$ROOT/tests/golden/deep_mlp/stdout.txt"; then
  echo "IDENTICAL  deep_mlp stdout.txt (C++ vs tests/golden/deep_mlp)"
else
  echo "DIFFERENT  deep_mlp stdout.txt (C++ vs tests/golden/deep_mlp)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test deep_mlp >/dev/null 2>&1 \
  && echo "IDENTICAL  deep_mlp Rust vs tests/golden/deep_mlp" \
  || { echo "DIFFERENT  deep_mlp Rust vs tests/golden/deep_mlp"; status=1; }

echo "== inference_mlp"
# tinymind's inference-only MultilayerPerceptron (IsTrainable = false) with loaded
# weights; tests/inference_mlp.rs checks InferencePerceptron against the committed
# golden, so the golden must still be what the C++ prints.
mkdir -p "$TMP/inference_mlp"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/inference_mlp/inference_mlp" "$ROOT/scripts/cpp/inference_mlp.cpp" \
  "$TINYMIND_DIR/cpp/lookupTables.cpp" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1 \
  -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_TANH_8_8=1
"$TMP/inference_mlp/inference_mlp" > "$TMP/inference_mlp/stdout.txt"
if cmp -s "$TMP/inference_mlp/stdout.txt" "$ROOT/tests/golden/inference_mlp/stdout.txt"; then
  echo "IDENTICAL  inference_mlp stdout.txt (C++ vs tests/golden/inference_mlp)"
else
  echo "DIFFERENT  inference_mlp stdout.txt (C++ vs tests/golden/inference_mlp)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test inference_mlp >/dev/null 2>&1 \
  && echo "IDENTICAL  inference_mlp Rust vs tests/golden/inference_mlp" \
  || { echo "DIFFERENT  inference_mlp Rust vs tests/golden/inference_mlp"; status=1; }

echo "== gbdt"
mkdir -p "$TMP/gbdt/cpp" "$TMP/gbdt/rust"
# Flags from examples/gbdt_tabular_int8/Makefile's release target. The C++
# writes its CSV to the working directory in both modes.
g++ -g -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/gbdt/gbdt_tabular_int8" \
  "$TINYMIND_DIR/examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1
(cd "$TMP/gbdt/cpp" && ../gbdt_tabular_int8 > stdout.txt && ../gbdt_tabular_int8 --golden > gbdt_tabular_int8.golden)
"$ROOT/target/release/examples/gbdt_tabular_int8" "$TMP/gbdt/rust" > "$TMP/gbdt/rust/stdout.txt"
"$ROOT/target/release/examples/gbdt_tabular_int8" --golden "$TMP/gbdt/rust" > "$TMP/gbdt/rust/gbdt_tabular_int8.golden"
for f in stdout.txt gbdt_tabular_int8.csv gbdt_tabular_int8.golden; do
  if cmp -s "$TMP/gbdt/cpp/$f" "$TMP/gbdt/rust/$f"; then
    echo "IDENTICAL  gbdt $f"
  else
    echo "DIFFERENT  gbdt $f"
    status=1
  fi
done

echo "== tree"
# tinymind's QDecisionTree and QGBDT over random ensembles; tests/tree_sweep.rs
# checks the Rust trees against the committed golden, so the golden must still
# be what the C++ prints.
mkdir -p "$TMP/tree"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/tree/tree_sweep" "$ROOT/scripts/cpp/tree_sweep.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include"
"$TMP/tree/tree_sweep" > "$TMP/tree/stdout.txt"
if cmp -s "$TMP/tree/stdout.txt" "$ROOT/tests/golden/tree/stdout.txt"; then
  echo "IDENTICAL  tree stdout.txt (C++ vs tests/golden/tree)"
else
  echo "DIFFERENT  tree stdout.txt (C++ vs tests/golden/tree)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test tree_sweep >/dev/null 2>&1 \
  && echo "IDENTICAL  tree Rust vs tests/golden/tree" \
  || { echo "DIFFERENT  tree Rust vs tests/golden/tree"; status=1; }

echo "== softmax"
# tinymind's QSoftmaxFixedPoint1D over a sweep of calibrations; tests/softmax_sweep.rs
# checks the Rust kernel against the committed golden, so the golden must still be
# what the C++ prints.
mkdir -p "$TMP/softmax"
g++ -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/softmax/softmax_sweep" "$ROOT/scripts/cpp/softmax_sweep.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
"$TMP/softmax/softmax_sweep" > "$TMP/softmax/stdout.txt"
if cmp -s "$TMP/softmax/stdout.txt" "$ROOT/tests/golden/softmax/stdout.txt"; then
  echo "IDENTICAL  softmax stdout.txt (C++ vs tests/golden/softmax)"
else
  echo "DIFFERENT  softmax stdout.txt (C++ vs tests/golden/softmax)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --features std --test softmax_sweep >/dev/null 2>&1 \
  && echo "IDENTICAL  softmax Rust vs tests/golden/softmax" \
  || { echo "DIFFERENT  softmax Rust vs tests/golden/softmax"; status=1; }

echo "== imported_mlp (PyTorch importer)"
# Needs a Python with numpy (PYTHON, default python3), e.g. a venv.
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import numpy" 2>/dev/null; then
  echo "the importer checks need numpy: set PYTHON to a Python that has it (e.g. a venv)" >&2
  exit 1
fi
export TINYMIND_DIR
mkdir -p "$TMP/imported"
# Same example model through tinymind's importer -> weights.hpp -> tinymind's C++ runtime.
"$PYTHON" -I -B "$ROOT/tools/import_pytorch/example_mlp.py" --cpp "$TMP/imported/weights.hpp" >/dev/null
"$PYTHON" -I -B "$ROOT/tools/import_pytorch/example_mlp.py" --model classifier --cpp "$TMP/imported/classifier.hpp" >/dev/null
g++ -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/imported/imported_mlp" "$ROOT/scripts/cpp/imported_mlp.cpp" \
  -I"$TMP/imported" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
"$TMP/imported/imported_mlp" > "$TMP/imported/cpp.txt"
"$ROOT/target/release/examples/imported_mlp" > "$TMP/imported/rust.txt"
if cmp -s "$TMP/imported/cpp.txt" "$TMP/imported/rust.txt"; then
  echo "IDENTICAL  imported_mlp stdout.txt"
else
  echo "DIFFERENT  imported_mlp stdout.txt"
  status=1
fi
# The committed generated modules are exactly what this repo's importer writes.
for model in mlp classifier; do
  "$PYTHON" -I -B "$ROOT/tools/import_pytorch/example_mlp.py" --model "$model" --rust "$TMP/imported/$model.rs" >/dev/null
  if cmp -s "$TMP/imported/$model.rs" "$ROOT/examples/common/imported_${model}_model.rs"; then
    echo "IDENTICAL  imported_mlp examples/common/imported_${model}_model.rs (regenerated)"
  else
    echo "DIFFERENT  imported_mlp examples/common/imported_${model}_model.rs (regenerated)"
    status=1
  fi
done
# Calibration and quantization parity with tinymind's importer on more models.
if "$PYTHON" -I -B "$ROOT/tools/import_pytorch/test_parity.py" > "$TMP/imported/parity.txt"; then
  echo "IDENTICAL  importer parity ($(grep -c "^case.*IDENTICAL" "$TMP/imported/parity.txt") models, $(grep "softmax params" "$TMP/imported/parity.txt" | cut -d" " -f3-5) softmax calibrations)"
else
  cat "$TMP/imported/parity.txt"
  echo "DIFFERENT  importer parity"
  status=1
fi

if [ -n "${KEEP_CPP_OUTPUT:-}" ]; then
  # For refreshing goldens: tests/golden must come from the C++ build.
  mkdir -p "$KEEP_CPP_OUTPUT"
  for format in q16.16 q8.8; do
    cp -r "$TMP/$format/cpp" "$KEEP_CPP_OUTPUT/$format"
    rm "$KEEP_CPP_OUTPUT/$format/xor"
  done
  cp -r "$TMP/iris/cpp" "$KEEP_CPP_OUTPUT/iris"
  cp -r "$TMP/int8_xor/cpp" "$KEEP_CPP_OUTPUT/int8_xor"
  cp -r "$TMP/energy/cpp" "$KEEP_CPP_OUTPUT/energy"
  mkdir -p "$KEEP_CPP_OUTPUT/deep_mlp"
  cp "$TMP/deep_mlp/stdout.txt" "$KEEP_CPP_OUTPUT/deep_mlp/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/inference_mlp"
  cp "$TMP/inference_mlp/stdout.txt" "$KEEP_CPP_OUTPUT/inference_mlp/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/softmax"
  cp "$TMP/softmax/stdout.txt" "$KEEP_CPP_OUTPUT/softmax/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/imported_mlp"
  cp "$TMP/imported/cpp.txt" "$KEEP_CPP_OUTPUT/imported_mlp/stdout.txt"
  cp -r "$TMP/gbdt/cpp" "$KEEP_CPP_OUTPUT/gbdt"
  mkdir -p "$KEEP_CPP_OUTPUT/tree"
  cp "$TMP/tree/stdout.txt" "$KEEP_CPP_OUTPUT/tree/stdout.txt"
  echo "C++ outputs copied to $KEEP_CPP_OUTPUT"
fi
exit $status
