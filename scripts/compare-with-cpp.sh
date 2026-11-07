#!/usr/bin/env bash
# Build tinymind's C++ examples and their Rust ports (xor in Q16.16 and Q8.8,
# iris), run both, and cmp every output file. Set TINYMIND_DIR to the
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

(cd "$ROOT" && cargo build --release --examples)

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

if [ -n "${KEEP_CPP_OUTPUT:-}" ]; then
  # For refreshing goldens: tests/golden must come from the C++ build.
  mkdir -p "$KEEP_CPP_OUTPUT"
  for format in q16.16 q8.8; do
    cp -r "$TMP/$format/cpp" "$KEEP_CPP_OUTPUT/$format"
    rm "$KEEP_CPP_OUTPUT/$format/xor"
  done
  cp -r "$TMP/iris/cpp" "$KEEP_CPP_OUTPUT/iris"
  echo "C++ outputs copied to $KEEP_CPP_OUTPUT"
fi
exit $status
