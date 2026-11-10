#!/usr/bin/env bash
# Build tinymind's C++ examples and their Rust ports (xor in Q16.16 and Q8.8,
# iris, energy_efficiency, maze, the int8 pytorch_quant xor, gbdt_tabular_int8), plus
# the C++ programs in scripts/cpp/ (deep networks, inference-only networks, the
# int8 softmax sweep, the Q-learning and tree sweeps, the PyTorch and tree importers' models), run both sides, and cmp every output file. Set TINYMIND_DIR to the
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
identical=0
# One passed comparison: report it and count it (the docs quote the count).
same() {
  echo "IDENTICAL  $*"
  identical=$((identical + 1))
}
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
      same "$format $f"
    else
      echo "DIFFERENT  $format $f"
      status=1
    fi
  done
done

echo "== iris"
# The data file must be tinymind's, byte for byte: both programs read it.
if cmp -s "$ROOT/data/iris.data" "$TINYMIND_DIR/examples/iris/iris.data"; then
  same "iris data/iris.data vs tinymind examples/iris/iris.data"
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
    same "iris $f"
  else
    echo "DIFFERENT  iris $f"
    status=1
  fi
done

echo "== energy_efficiency"
# The data file must be tinymind's, byte for byte (CRLF in both working trees).
if cmp -s "$ROOT/data/ENB2012_data.csv" "$TINYMIND_DIR/examples/energy_efficiency/ENB2012_data.csv"; then
  same "energy data/ENB2012_data.csv vs tinymind examples/energy_efficiency/ENB2012_data.csv"
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
    same "energy $f"
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
    same "int8_xor $f"
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
  same "deep_mlp stdout.txt (C++ vs tests/golden/deep_mlp)"
else
  echo "DIFFERENT  deep_mlp stdout.txt (C++ vs tests/golden/deep_mlp)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test deep_mlp >/dev/null 2>&1 \
  && same "deep_mlp Rust vs tests/golden/deep_mlp" \
  || { echo "DIFFERENT  deep_mlp Rust vs tests/golden/deep_mlp"; status=1; }

echo "== float_mlp"
# tinymind's NeuralNetwork in double and float with its floating-point
# activations; tests/float_mlp.rs checks the Rust f64/f32 networks against the
# committed golden. -ffp-contract=off: fused multiply-adds would change the sums.
mkdir -p "$TMP/float_mlp"
g++ -O2 -ffp-contract=off -Wall -Wextra -Werror -Wpedantic -o "$TMP/float_mlp/float_mlp" "$ROOT/scripts/cpp/float_mlp.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_RAND=1
"$TMP/float_mlp/float_mlp" > "$TMP/float_mlp/stdout.txt"
# The golden holds the double runs only: the float ones depend on the C
# library's tanhf/expf (correctly rounded since glibc 2.41), so they are
# compared live below, C++ and Rust on this machine.
if cmp -s <(grep '^f64 ' "$TMP/float_mlp/stdout.txt") "$ROOT/tests/golden/float_mlp/stdout.txt"; then
  same "float_mlp double runs (C++ vs tests/golden/float_mlp)"
else
  echo "DIFFERENT  float_mlp double runs (C++ vs tests/golden/float_mlp)"
  status=1
fi
TINYMIND_FLOAT_MLP_CPP="$TMP/float_mlp/stdout.txt" \
  cargo test -q --manifest-path "$ROOT/Cargo.toml" --features std --test float_mlp >/dev/null 2>&1 \
  && same "float_mlp Rust vs C++ on this host (float and double, every step)" \
  || { echo "DIFFERENT  float_mlp Rust vs C++ on this host"; status=1; }

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
  same "inference_mlp stdout.txt (C++ vs tests/golden/inference_mlp)"
else
  echo "DIFFERENT  inference_mlp stdout.txt (C++ vs tests/golden/inference_mlp)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test inference_mlp >/dev/null 2>&1 \
  && same "inference_mlp Rust vs tests/golden/inference_mlp" \
  || { echo "DIFFERENT  inference_mlp Rust vs tests/golden/inference_mlp"; status=1; }

echo "== inference_deep"
# tinymind's inference-only NeuralNetwork (IsTrainable = false) with one to
# three hidden layers and loaded weights; tests/inference_deep.rs checks
# InferenceNetwork against the committed golden.
mkdir -p "$TMP/inference_deep"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/inference_deep/inference_deep" "$ROOT/scripts/cpp/inference_deep.cpp" \
  "$TINYMIND_DIR/cpp/lookupTables.cpp" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1 \
  -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_TANH_8_8=1
"$TMP/inference_deep/inference_deep" > "$TMP/inference_deep/stdout.txt"
if cmp -s "$TMP/inference_deep/stdout.txt" "$ROOT/tests/golden/inference_deep/stdout.txt"; then
  same "inference_deep stdout.txt (C++ vs tests/golden/inference_deep)"
else
  echo "DIFFERENT  inference_deep stdout.txt (C++ vs tests/golden/inference_deep)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test inference_deep >/dev/null 2>&1 \
  && same "inference_deep Rust vs tests/golden/inference_deep" \
  || { echo "DIFFERENT  inference_deep Rust vs tests/golden/inference_deep"; status=1; }

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
    same "gbdt $f"
  else
    echo "DIFFERENT  gbdt $f"
    status=1
  fi
done

echo "== maze"
mkdir -p "$TMP/maze/cpp" "$TMP/maze/rust"
# Flags from examples/maze/Makefile's release target, plus -ffp-contract=off:
# the example computes Q* by value iteration in double, and a compiler that
# fuses multiply-adds (g++ 15 on CI does) changes it.
g++ -g -O3 -ffp-contract=off -Wall -Wextra -Werror -Wpedantic -o "$TMP/maze/maze" \
  "$TINYMIND_DIR/examples/maze/maze.cpp" "$TINYMIND_DIR/examples/maze/mazelearner.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include"
(cd "$TMP/maze/cpp" && ../maze > stdout.txt)
"$ROOT/target/release/examples/maze" "$TMP/maze/rust" > "$TMP/maze/rust/stdout.txt"
for f in stdout.txt maze_training.csv maze_qtable.csv maze_test.csv; do
  if cmp -s "$TMP/maze/cpp/$f" "$TMP/maze/rust/$f"; then
    same "maze $f"
  else
    echo "DIFFERENT  maze $f"
    status=1
  fi
done

echo "== qlearn"
# tinymind's QLearner on the maze and random environments, Q table hashed after
# every update; tests/qlearn_sweep.rs checks the Rust learner against the
# committed golden, so the golden must still be what the C++ prints.
mkdir -p "$TMP/qlearn"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/qlearn/qlearn_sweep" "$ROOT/scripts/cpp/qlearn_sweep.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include"
"$TMP/qlearn/qlearn_sweep" > "$TMP/qlearn/stdout.txt"
if cmp -s "$TMP/qlearn/stdout.txt" "$ROOT/tests/golden/qlearn/stdout.txt"; then
  same "qlearn stdout.txt (C++ vs tests/golden/qlearn)"
else
  echo "DIFFERENT  qlearn stdout.txt (C++ vs tests/golden/qlearn)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test qlearn_sweep >/dev/null 2>&1 \
  && same "qlearn Rust vs tests/golden/qlearn" \
  || { echo "DIFFERENT  qlearn Rust vs tests/golden/qlearn"; status=1; }

echo "== lstm"
# tinymind's LstmNeuralNetwork on short sequences, every weight hashed after
# every step; tests/lstm_sweep.rs checks the Rust network against the committed
# golden, so the golden must still be what the C++ prints.
mkdir -p "$TMP/lstm"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/lstm/lstm_sweep" "$ROOT/scripts/cpp/lstm_sweep.cpp" \
  "$TINYMIND_DIR/cpp/lookupTables.cpp" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1 \
  -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_TANH_8_8=1
"$TMP/lstm/lstm_sweep" > "$TMP/lstm/stdout.txt"
if cmp -s "$TMP/lstm/stdout.txt" "$ROOT/tests/golden/lstm/stdout.txt"; then
  same "lstm stdout.txt (C++ vs tests/golden/lstm)"
else
  echo "DIFFERENT  lstm stdout.txt (C++ vs tests/golden/lstm)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test lstm_sweep >/dev/null 2>&1 \
  && same "lstm Rust vs tests/golden/lstm" \
  || { echo "DIFFERENT  lstm Rust vs tests/golden/lstm"; status=1; }

echo "== har_activity"
mkdir -p "$TMP/har/cpp" "$TMP/har/rust"
# Flags from examples/har_activity/Makefile's release target, plus
# -ffp-contract=off: the example synthesizes its windows in double arithmetic,
# and a compiler that fuses multiply-adds (g++ 15 on CI does) changes them.
g++ -g -O3 -ffp-contract=off -Wall -Wextra -Werror -Wpedantic -o "$TMP/har/har_activity" \
  "$TINYMIND_DIR/examples/har_activity/har_activity.cpp" "$TINYMIND_DIR/cpp/lookupTables.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include/" \
  -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 \
  -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1
(cd "$TMP/har/cpp" && ../har_activity > stdout.txt)
(cd "$TMP/har/rust" && "$ROOT/target/release/examples/har_activity" . > stdout.txt)
for f in stdout.txt har_samples.csv har_loss.csv har_confusion.csv; do
  if cmp -s "$TMP/har/cpp/$f" "$TMP/har/rust/$f"; then
    same "har_activity $f"
  else
    echo "DIFFERENT  har_activity $f"
    status=1
  fi
  if ! cmp -s "$TMP/har/cpp/$f" "$ROOT/tests/golden/har/$f"; then
    echo "DIFFERENT  har_activity $f (C++ vs tests/golden/har)"
    status=1
  fi
done
# --readout-only (only the hidden-to-output weights train): compared live, no
# golden. Only stdout and the files training changes.
mkdir -p "$TMP/har/cpp_ro" "$TMP/har/rust_ro"
(cd "$TMP/har/cpp_ro" && ../har_activity --readout-only > stdout.txt) &
(cd "$TMP/har/rust_ro" && "$ROOT/target/release/examples/har_activity" --readout-only . > stdout.txt) &
wait
for f in stdout.txt har_loss.csv har_confusion.csv; do
  if cmp -s "$TMP/har/cpp_ro/$f" "$TMP/har/rust_ro/$f"; then
    same "har_activity --readout-only $f"
  else
    echo "DIFFERENT  har_activity --readout-only $f"
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
  same "tree stdout.txt (C++ vs tests/golden/tree)"
else
  echo "DIFFERENT  tree stdout.txt (C++ vs tests/golden/tree)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test tree_sweep >/dev/null 2>&1 \
  && same "tree Rust vs tests/golden/tree" \
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
  same "softmax stdout.txt (C++ vs tests/golden/softmax)"
else
  echo "DIFFERENT  softmax stdout.txt (C++ vs tests/golden/softmax)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --features std --test softmax_sweep >/dev/null 2>&1 \
  && same "softmax Rust vs tests/golden/softmax" \
  || { echo "DIFFERENT  softmax Rust vs tests/golden/softmax"; status=1; }

echo "== conv"
# tinymind's int8 convolution and pooling layers over random tensors;
# tests/conv_sweep.rs checks the Rust layers against the committed golden, so
# the golden must still be what the C++ prints.
mkdir -p "$TMP/conv"
g++ -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/conv/conv_sweep" "$ROOT/scripts/cpp/conv_sweep.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1
"$TMP/conv/conv_sweep" > "$TMP/conv/stdout.txt"
if cmp -s "$TMP/conv/stdout.txt" "$ROOT/tests/golden/conv/stdout.txt"; then
  same "conv stdout.txt (C++ vs tests/golden/conv)"
else
  echo "DIFFERENT  conv stdout.txt (C++ vs tests/golden/conv)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --test conv_sweep >/dev/null 2>&1 \
  && same "conv Rust vs tests/golden/conv" \
  || { echo "DIFFERENT  conv Rust vs tests/golden/conv"; status=1; }

echo "== add"
# tinymind's int8 elementwise addition and its calibration;
# tests/add_sweep.rs checks the Rust layer against the committed golden.
mkdir -p "$TMP/add"
g++ -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/add/add_sweep" "$ROOT/scripts/cpp/add_sweep.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
"$TMP/add/add_sweep" > "$TMP/add/stdout.txt"
if cmp -s "$TMP/add/stdout.txt" "$ROOT/tests/golden/add/stdout.txt"; then
  same "add stdout.txt (C++ vs tests/golden/add)"
else
  echo "DIFFERENT  add stdout.txt (C++ vs tests/golden/add)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --features std --test add_sweep >/dev/null 2>&1 \
  && same "add Rust vs tests/golden/add" \
  || { echo "DIFFERENT  add Rust vs tests/golden/add"; status=1; }

echo "== compose"
# tinymind's int8 QMul, QConcat2_2D and QPad2D with their calibration;
# tests/compose_sweep.rs checks the Rust layers against the committed golden.
mkdir -p "$TMP/compose"
g++ -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/compose/compose_sweep" "$ROOT/scripts/cpp/compose_sweep.cpp" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
"$TMP/compose/compose_sweep" > "$TMP/compose/stdout.txt"
if cmp -s "$TMP/compose/stdout.txt" "$ROOT/tests/golden/compose/stdout.txt"; then
  same "compose stdout.txt (C++ vs tests/golden/compose)"
else
  echo "DIFFERENT  compose stdout.txt (C++ vs tests/golden/compose)"
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --features std --test compose_sweep >/dev/null 2>&1 \
  && same "compose Rust vs tests/golden/compose" \
  || { echo "DIFFERENT  compose Rust vs tests/golden/compose"; status=1; }

echo "== kws_int8"
# tinymind's examples/kws_cortex_m_int8, compiled unchanged inside
# scripts/cpp/kws_int8_dump.cpp, which prints every tensor (its own stdout
# carries host timings). tests/kws_int8_golden.rs checks the Rust port against
# the committed golden, so the golden must still be what the C++ prints.
# -ffp-contract=off: the example draws its float weights as u * (b - a) + a,
# and a compiler that targets FMA by default (g++ 15 on Ubuntu 26.04) fuses
# that into one rounding, which moves a per-channel scale by an ulp. Rust,
# like ISO C++, rounds each operation; so does g++ without FMA.
mkdir -p "$TMP/kws_int8"
g++ -O3 -ffp-contract=off -Wall -Wextra -Werror -Wpedantic -o "$TMP/kws_int8/kws_int8_dump" "$ROOT/scripts/cpp/kws_int8_dump.cpp" \
  -I"$TINYMIND_DIR/examples/kws_cortex_m_int8" \
  -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1
"$TMP/kws_int8/kws_int8_dump" > "$TMP/kws_int8/tensors.txt"
if cmp -s "$TMP/kws_int8/tensors.txt" "$ROOT/tests/golden/kws_int8/tensors.txt"; then
  same "kws_int8 tensors.txt (C++ vs tests/golden/kws_int8)"
else
  echo "DIFFERENT  kws_int8 tensors.txt (C++ vs tests/golden/kws_int8)"
  # Which tensors: one line each, so the names say where it starts.
  diff "$ROOT/tests/golden/kws_int8/tensors.txt" "$TMP/kws_int8/tensors.txt" | cut -c1-200 | head -20
  status=1
fi
cargo test -q --manifest-path "$ROOT/Cargo.toml" --features std --test kws_int8_golden >/dev/null 2>&1 \
  && same "kws_int8 Rust vs tests/golden/kws_int8" \
  || { echo "DIFFERENT  kws_int8 Rust vs tests/golden/kws_int8"; status=1; }

echo "== imported_mlp (PyTorch importer)"
# Needs a Python with numpy (PYTHON, default python3), e.g. a venv.
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import numpy, sklearn" 2>/dev/null; then
  echo "the importer checks need numpy and scikit-learn: set PYTHON to a Python that has them (e.g. a venv)" >&2
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
  same "imported_mlp stdout.txt"
else
  echo "DIFFERENT  imported_mlp stdout.txt"
  status=1
fi
# The committed generated modules are exactly what this repo's importer writes.
for model in mlp classifier; do
  "$PYTHON" -I -B "$ROOT/tools/import_pytorch/example_mlp.py" --model "$model" --rust "$TMP/imported/$model.rs" >/dev/null
  if cmp -s "$TMP/imported/$model.rs" "$ROOT/examples/common/imported_${model}_model.rs"; then
    same "imported_mlp examples/common/imported_${model}_model.rs (regenerated)"
  else
    echo "DIFFERENT  imported_mlp examples/common/imported_${model}_model.rs (regenerated)"
    status=1
  fi
done
# Calibration and quantization parity with tinymind's importer on more models.
if "$PYTHON" -I -B "$ROOT/tools/import_pytorch/test_parity.py" > "$TMP/imported/parity.txt"; then
  same "importer parity ($(grep -c "^case.*IDENTICAL" "$TMP/imported/parity.txt") models, $(grep "softmax params" "$TMP/imported/parity.txt" | cut -d" " -f3-5) softmax calibrations)"
else
  cat "$TMP/imported/parity.txt"
  echo "DIFFERENT  importer parity"
  status=1
fi

echo "== imported_cnn (PyTorch importer, convolutional models)"
mkdir -p "$TMP/cnn"
# The same models through tinymind's importer -> headers -> tinymind's C++ runtime.
for model in cnn cnn_flat; do
  "$PYTHON" -I -B "$ROOT/tools/import_pytorch/example_cnn.py" --model "$model" --cpp "$TMP/cnn/$model.hpp" >/dev/null
done
g++ -O3 -Wall -Wextra -Werror -Wpedantic -o "$TMP/cnn/imported_cnn" "$ROOT/scripts/cpp/imported_cnn.cpp" \
  -I"$TMP/cnn" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" \
  -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
"$TMP/cnn/imported_cnn" > "$TMP/cnn/cpp.txt"
"$ROOT/target/release/examples/imported_cnn" > "$TMP/cnn/rust.txt"
if cmp -s "$TMP/cnn/cpp.txt" "$TMP/cnn/rust.txt"; then
  same "imported_cnn stdout.txt"
else
  echo "DIFFERENT  imported_cnn stdout.txt"
  status=1
fi
# The committed generated modules are exactly what this repo's importer writes.
for model in cnn cnn_flat; do
  "$PYTHON" -I -B "$ROOT/tools/import_pytorch/example_cnn.py" --model "$model" --rust "$TMP/cnn/$model.rs" >/dev/null
  if cmp -s "$TMP/cnn/$model.rs" "$ROOT/examples/common/imported_${model}_model.rs"; then
    same "imported_cnn examples/common/imported_${model}_model.rs (regenerated)"
  else
    echo "DIFFERENT  imported_cnn examples/common/imported_${model}_model.rs (regenerated)"
    status=1
  fi
done

echo "== import_tflite (TFLite translator)"
# Needs tflite and flatbuffers in PYTHON. The MLPerf Tiny keyword-spotting and
# image-classification models are downloaded at tinymind's pinned commit
# (cached in target/tflite-models).
if "$PYTHON" -I -B "$ROOT/tools/import_tflite/test_translator.py" > "$TMP/tflite_unit.txt"; then
  same "TFLite translator softmax and ADD params, activation ranges"
else
  cat "$TMP/tflite_unit.txt"
  echo "DIFFERENT  TFLite translator softmax and ADD params, activation ranges"
  status=1
fi
if "$PYTHON" -I -B "$ROOT/tools/import_tflite/mlperf.py" kws --target-dir "$TMP/tflite_target" > "$TMP/tflite_kws.txt"; then
  same "MLPerf Tiny keyword spotting (DS-CNN): output hash = tinymind's"
else
  cat "$TMP/tflite_kws.txt"
  echo "DIFFERENT  MLPerf Tiny keyword spotting (DS-CNN)"
  status=1
fi
if "$PYTHON" -I -B "$ROOT/tools/import_tflite/mlperf.py" ic --target-dir "$TMP/tflite_target" > "$TMP/tflite_ic.txt"; then
  same "MLPerf Tiny image classification (ResNet-8): output hash = tinymind's"
else
  cat "$TMP/tflite_ic.txt"
  echo "DIFFERENT  MLPerf Tiny image classification (ResNet-8)"
  status=1
fi
# PAD, MUL and CONCATENATION, which the MLPerf models do not use: small models
# built by tools/import_tflite/tflite_build.py, against TFLite's reference
# kernels (LiteRT, ai-edge-litert in PYTHON).
if "$PYTHON" -I -B "$ROOT/tools/import_tflite/test_ops.py" > "$TMP/tflite_ops.txt" 2>&1; then
  same "TFLite translator PAD, MUL, CONCATENATION: output = TFLite reference kernels"
else
  cat "$TMP/tflite_ops.txt"
  echo "DIFFERENT  TFLite translator PAD, MUL, CONCATENATION"
  status=1
fi

echo "== imported_trees (tree importer)"
# Needs scikit-learn in PYTHON; XGBoost and LightGBM for all of test_parity.py.
mkdir -p "$TMP/trees"
# The same models through tinymind's importer -> headers -> tinymind's C++ runtime.
for model in gbdt forest; do
  "$PYTHON" -I -B "$ROOT/tools/import_trees/example_trees.py" --model "$model" --cpp "$TMP/trees/$model.hpp" >/dev/null
done
"$PYTHON" -I -B "$ROOT/tools/import_trees/example_trees.py" --data-cpp "$TMP/trees/trees_data.hpp"
g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/trees/imported_trees" "$ROOT/scripts/cpp/imported_trees.cpp" \
  -I"$TMP/trees" -I"$TINYMIND_DIR/cpp" -I"$TINYMIND_DIR/cpp/include" -I"$TINYMIND_DIR/include" -DTINYMIND_ENABLE_FLOAT=1
"$TMP/trees/imported_trees" > "$TMP/trees/cpp.txt"
"$ROOT/target/release/examples/imported_trees" > "$TMP/trees/rust.txt"
if cmp -s "$TMP/trees/cpp.txt" "$TMP/trees/rust.txt"; then
  same "imported_trees stdout.txt"
else
  echo "DIFFERENT  imported_trees stdout.txt"
  status=1
fi
if cmp -s "$TMP/trees/cpp.txt" "$ROOT/tests/golden/imported_trees/stdout.txt"; then
  same "imported_trees stdout.txt (C++ vs tests/golden/imported_trees)"
else
  echo "DIFFERENT  imported_trees stdout.txt (C++ vs tests/golden/imported_trees)"
  status=1
fi
# The committed generated modules are exactly what this repo's importer writes.
for model in gbdt forest; do
  "$PYTHON" -I -B "$ROOT/tools/import_trees/example_trees.py" --model "$model" --rust "$TMP/trees/$model.rs" >/dev/null
  if cmp -s "$TMP/trees/$model.rs" "$ROOT/examples/common/imported_trees_${model}_model.rs"; then
    same "imported_trees examples/common/imported_trees_${model}_model.rs (regenerated)"
  else
    echo "DIFFERENT  imported_trees examples/common/imported_trees_${model}_model.rs (regenerated)"
    status=1
  fi
done
"$PYTHON" -I -B "$ROOT/tools/import_trees/example_trees.py" --data-rust "$TMP/trees/data.rs"
if cmp -s "$TMP/trees/data.rs" "$ROOT/examples/common/imported_trees_data.rs"; then
  same "imported_trees examples/common/imported_trees_data.rs (regenerated)"
else
  echo "DIFFERENT  imported_trees examples/common/imported_trees_data.rs (regenerated)"
  status=1
fi
# Integer-model parity with tinymind's importer, and both runtimes, on more models.
if "$PYTHON" -I -B "$ROOT/tools/import_trees/test_parity.py" > "$TMP/trees/parity.txt"; then
  same "tree importer parity ($(grep -c "^case.*IDENTICAL" "$TMP/trees/parity.txt") models)"
else
  cat "$TMP/trees/parity.txt"
  echo "DIFFERENT  tree importer parity"
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
  mkdir -p "$KEEP_CPP_OUTPUT/float_mlp"
  grep '^f64 ' "$TMP/float_mlp/stdout.txt" > "$KEEP_CPP_OUTPUT/float_mlp/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/inference_deep"
  cp "$TMP/inference_deep/stdout.txt" "$KEEP_CPP_OUTPUT/inference_deep/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/softmax"
  cp "$TMP/softmax/stdout.txt" "$KEEP_CPP_OUTPUT/softmax/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/imported_mlp"
  cp "$TMP/imported/cpp.txt" "$KEEP_CPP_OUTPUT/imported_mlp/stdout.txt"
  cp -r "$TMP/gbdt/cpp" "$KEEP_CPP_OUTPUT/gbdt"
  mkdir -p "$KEEP_CPP_OUTPUT/imported_trees"
  cp "$TMP/trees/cpp.txt" "$KEEP_CPP_OUTPUT/imported_trees/stdout.txt"
  cp -r "$TMP/maze/cpp" "$KEEP_CPP_OUTPUT/maze"
  mkdir -p "$KEEP_CPP_OUTPUT/qlearn"
  cp "$TMP/qlearn/stdout.txt" "$KEEP_CPP_OUTPUT/qlearn/stdout.txt"
  cp -r "$TMP/har/cpp" "$KEEP_CPP_OUTPUT/har"
  mkdir -p "$KEEP_CPP_OUTPUT/lstm"
  cp "$TMP/lstm/stdout.txt" "$KEEP_CPP_OUTPUT/lstm/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/tree"
  cp "$TMP/tree/stdout.txt" "$KEEP_CPP_OUTPUT/tree/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/kws_int8"
  cp "$TMP/kws_int8/tensors.txt" "$KEEP_CPP_OUTPUT/kws_int8/tensors.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/imported_cnn"
  cp "$TMP/cnn/cpp.txt" "$KEEP_CPP_OUTPUT/imported_cnn/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/conv"
  cp "$TMP/conv/stdout.txt" "$KEEP_CPP_OUTPUT/conv/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/compose"
  cp "$TMP/compose/stdout.txt" "$KEEP_CPP_OUTPUT/compose/stdout.txt"
  mkdir -p "$KEEP_CPP_OUTPUT/add"
  cp "$TMP/add/stdout.txt" "$KEEP_CPP_OUTPUT/add/stdout.txt"
  echo "C++ outputs copied to $KEEP_CPP_OUTPUT"
fi
# Every comparison passed: the count the docs quote must be this one.
if [ "$status" -eq 0 ]; then
  "$ROOT/scripts/check-counts.sh" comparisons "$identical" || status=1
fi
exit $status
