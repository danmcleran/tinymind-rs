#!/usr/bin/env bash
# List tinymind commits since the pinned TINYMIND_REF that touch the C++ sources
# this crate ports. Extra arguments go to `git log` (e.g. --stat, -p).
#
# Keep PORTED in sync with the module table in CLAUDE.md: when a new tinymind
# file is ported (or a ported path starts depending on another header), add it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TINYMIND_DIR="${TINYMIND_DIR:-$ROOT/../tinymind}"
REF="$(tr -d '[:space:]' < "$ROOT/TINYMIND_REF")"

PORTED=(
  cpp/qformat.hpp
  cpp/include/tinymind_rounding.hpp
  cpp/typeChooser.hpp
  cpp/tanh.hpp
  cpp/sigmoid.hpp
  cpp/activation.hpp
  cpp/activationFunctions.hpp
  cpp/lookupTable.hpp
  cpp/interpolate.hpp
  cpp/lookupTables.cpp
  cpp/neuralnet.hpp
  cpp/fixedPointTransferFunctions.hpp
  cpp/nninit.hpp
  cpp/error.hpp
  cpp/zeroTolerance.hpp
  cpp/constants.hpp
  cpp/learningRateSchedule.hpp
  cpp/gradientClipping.hpp
  cpp/weightDecay.hpp
  cpp/adam.hpp
  cpp/rmsprop.hpp
  cpp/qlearn.hpp
  cpp/include/nnproperties.hpp
  examples/xor
  examples/iris
  examples/energy_efficiency
  examples/maze
  examples/har_activity
  examples/pytorch_quant/xor
  apps/import_pytorch
  apps/import_trees
  cpp/qaffine.hpp
  cpp/qdense.hpp
  cpp/qpointwiseconv2d.hpp
  cpp/qconv2d.hpp
  cpp/qdepthwiseconv2d.hpp
  cpp/qpool2d.hpp
  cpp/qadd.hpp
  cpp/qmul.hpp
  cpp/qpad.hpp
  cpp/qlstm.hpp
  cpp/qgru.hpp
  cpp/qconcat.hpp
  benchmarks/framework_comparison_f411/gen/out/mlp_tinymind.hpp
  cpp/qactivations.hpp
  cpp/qsoftmax_fixedpoint.hpp
  cpp/qtree.hpp
  examples/gbdt_tabular_int8
  examples/kws_cortex_m_int8
  cpp/include/qcalibration.hpp
  cpp/include/simd/simd_dispatch.hpp
  unit_test/qformat_oracle
)

if ! git -C "$TINYMIND_DIR" cat-file -e "$REF^{commit}" 2>/dev/null; then
  echo "TINYMIND_REF $REF is not in $TINYMIND_DIR (fetch it, or check TINYMIND_DIR)" >&2
  exit 1
fi

echo "tinymind: $TINYMIND_DIR at $(git -C "$TINYMIND_DIR" rev-parse --short HEAD) ($(git -C "$TINYMIND_DIR" rev-parse --abbrev-ref HEAD)); pinned TINYMIND_REF: ${REF:0:7}"
if [ -n "$(git -C "$TINYMIND_DIR" status --porcelain -- "${PORTED[@]}")" ]; then
  echo "warning: tinymind has uncommitted changes in ported sources" >&2
fi

changes="$(git -C "$TINYMIND_DIR" log --oneline "$@" "$REF..HEAD" -- "${PORTED[@]}")"
if [ -z "$changes" ]; then
  echo "No changes to ported sources since TINYMIND_REF."
else
  echo "Commits since TINYMIND_REF touching ported sources:"
  echo "$changes"
fi
