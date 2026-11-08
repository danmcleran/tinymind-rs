#!/usr/bin/env bash
# The docs quote two counts that only a full run can compute. The job that
# computes each one calls this with the real number, and every place the docs
# quote it must match:
#
#   scripts/check-counts.sh comparisons N   "N checks today"  (scripts/compare-with-cpp.sh)
#   scripts/check-counts.sh harnesses N     "N harnesses"     (the kani CI job)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
kind="${1:?usage: check-counts.sh comparisons|harnesses N}"
n="${2:?usage: check-counts.sh comparisons|harnesses N}"
case "$kind" in
  comparisons) pattern='[0-9]+ checks today' ;;
  harnesses) pattern='[0-9]+ (Kani )?harnesses' ;;
  *) echo "check-counts: unknown count '$kind'" >&2; exit 2 ;;
esac
status=0
found=0
while IFS=: read -r file line text; do
  found=1
  quoted="${text%% *}"
  if [ "$quoted" != "$n" ]; then
    echo "check-counts: $file:$line says '$text', but there are $n" >&2
    status=1
  fi
done < <(grep -noE "$pattern" README.md CLAUDE.md docs/*.md)
[ "$found" = 1 ] || { echo "check-counts: no doc quotes the $kind count" >&2; status=1; }
[ "$status" -eq 0 ] && echo "check-counts: $kind = $n everywhere the docs quote it"
exit "$status"
