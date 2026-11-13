#!/usr/bin/env bash
# Run the on-board checks (bench/) under QEMU, on the instruction sets of both
# boards, without the boards: every workload's hash must match, as on the
# chips. QEMU does not model cycles, so this checks correctness only; timings
# stay with bench/run.sh.
#
#   scripts/qemu-bench.sh                     every workload, both cores, opt 3/s/z
#   CORES=m4 OPTS=3 WORKLOADS="mlp kws" scripts/qemu-bench.sh
#
# Cores: m4 = QEMU mps2-an386 (Cortex-M4, thumbv7em-none-eabihf, the F411's
# instruction set), m0 = QEMU microbit (Cortex-M0, thumbv6m-none-eabi: the
# L031's Cortex-M0+ runs the same ARMv6-M instructions), given 2M of flash and
# 256K of RAM for the sweeps image (the chip has 256K and 16K). The `sweeps`
# workload runs the host test sweeps (tests/sweeps/) against their goldens.
# The MLPerf workloads run on m4 only, and only when their generated sources
# exist (see bench/README.md). Needs qemu-system-arm, the two Rust targets and a tinymind
# (C++) checkout (TINYMIND_DIR, default ../tinymind) for bench/gen.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/bench"
OUT="${QEMU_OUT:-$ROOT/target/qemu-bench}"
mkdir -p "$OUT"

python3 -I gen.py "${MLP_SAMPLES:-360}" >/dev/null

fail=0
runs=0
for core in ${CORES:-m4 m0}; do
  case "$core" in
    m4) target=thumbv7em-none-eabihf machine=(-machine mps2-an386 -cpu cortex-m4) ;;
    m0) target=thumbv6m-none-eabi machine=(-machine microbit -global nrf51-soc.flash-size=2097152 -global nrf51-soc.sram-size=262144) ;;
    *) echo "qemu-bench: unknown core '$core'" >&2; exit 2 ;;
  esac
  default="mlp profile xor-q16 xor-q8 anomaly fft head trees kws kws-profile sweeps"
  if [ "$core" = m4 ]; then
    for wl in mlperf-kws mlperf-ic; do
      [ -f "src/gen/${wl//-/_}.rs" ] && default="$default $wl"
    done
  fi
  for wl in ${WORKLOADS:-$default}; do
    for opt in ${OPTS:-3 s z}; do
      name="$core-$wl-O$opt"
      log="$OUT/$name.txt"
      # Not RUSTFLAGS: it would replace bench/.cargo's -Tlink.x.
      CARGO_PROFILE_RELEASE_OPT_LEVEL=$opt cargo build -q --release --target "$target" \
        --features "qemu-$core,$wl" --target-dir target/qemu
      status=0
      timeout "${QEMU_TIMEOUT:-600}" qemu-system-arm "${machine[@]}" -nographic -monitor none \
        -serial none -semihosting-config enable=on,target=native \
        -kernel "target/qemu/$target/release/tinymind-bench" > "$log" 2>&1 || status=$?
      runs=$((runs + 1))
      problem=""
      [ "$status" -eq 0 ] || problem="qemu exit $status"
      grep -q '^# PANIC' "$log" && problem="panic"
      grep -q '^# end bench' "$log" || problem="${problem:-no end marker}"
      grep -qE '^[a-z0-9_]*match(es)?=0' "$log" && problem="mismatch: $(grep -E '^[a-z0-9_]*match(es)?=0' "$log" | tr '\n' ' ')"
      grep -qE '^[a-z0-9_]*match(es)?=1' "$log" || problem="${problem:-no hash check in the report}"
      if [ -n "$problem" ]; then
        echo "FAIL  $name: $problem (report: $log)"
        fail=1
      else
        echo "ok    $name"
      fi
    done
  done
done
[ "$fail" -eq 0 ] || { echo "qemu-bench: failures above" >&2; exit 1; }
echo "qemu-bench: $runs runs, every hash matched"
