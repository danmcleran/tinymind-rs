#!/usr/bin/env bash
# Build the tinymind C++ xor example and the Rust port, run both, and cmp
# every output file. Set TINYMIND_DIR to the tinymind checkout (default ../tinymind).
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
mkdir -p "$TMP/cpp" "$TMP/rust"

(
  cd "$TINYMIND_DIR"
  # Glob, not a file list: older tinymind revisions also have examples/xor/xornet.cpp.
  g++ -O2 -Wall -Wextra -Werror -Wpedantic -o "$TMP/cpp/xor" examples/xor/*.cpp cpp/lookupTables.cpp -Icpp -Icpp/include -Iinclude/ -DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1
)
(cd "$TMP/cpp" && ./xor > stdout.txt)

(cd "$ROOT" && cargo build --release)
"$ROOT/target/release/xor" "$TMP/rust" > "$TMP/rust/stdout.txt"

status=0
for f in stdout.txt nn_fixed_xor.txt xor_training.csv xor_decision_surface.csv; do
  if cmp -s "$TMP/cpp/$f" "$TMP/rust/$f"; then
    echo "IDENTICAL  $f"
  else
    echo "DIFFERENT  $f"
    status=1
  fi
done
exit $status
