#!/usr/bin/env bash
# Build the tinymind C++ xor example and the Rust port in each supported format
# (Q16.16, Q8.8), run both, and cmp every output file. Set TINYMIND_DIR to the
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

(cd "$ROOT" && cargo build --release)

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
  "$ROOT/target/release/xor" --format "$format" "$TMP/$format/rust" > "$TMP/$format/rust/stdout.txt"

  for f in stdout.txt nn_fixed_xor.txt xor_training.csv xor_decision_surface.csv; do
    if cmp -s "$TMP/$format/cpp/$f" "$TMP/$format/rust/$f"; then
      echo "IDENTICAL  $format $f"
    else
      echo "DIFFERENT  $format $f"
      status=1
    fi
  done
done

if [ -n "${KEEP_CPP_OUTPUT:-}" ]; then
  # For refreshing goldens: tests/golden must come from the C++ build.
  mkdir -p "$KEEP_CPP_OUTPUT"
  for format in q16.16 q8.8; do
    cp -r "$TMP/$format/cpp" "$KEEP_CPP_OUTPUT/$format"
    rm "$KEEP_CPP_OUTPUT/$format/xor"
  done
  echo "C++ outputs copied to $KEEP_CPP_OUTPUT"
fi
exit $status
