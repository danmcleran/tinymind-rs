#!/usr/bin/env bash
# Build, flash and run the on-board checks, and print each board's report.
#
#   bench/run.sh [board ...]       boards: f411 l031 (default: both)
#   WORKLOADS="anomaly" bench/run.sh f411    only some workloads
#   OPT=s bench/run.sh                       every image at opt-level s
#
# Needs: the thumbv7em-none-eabihf and thumbv6m-none-eabi Rust targets, openocd,
# arm-none-eabi-size, a tinymind checkout (TINYMIND_DIR, default ../tinymind; the
# MLP is generated from its benchmark) and the boards on ST-LINKs. The probes
# are picked by serial number; the VCP port follows the serial, so replugging
# (which renumbers /dev/ttyACM*) does not matter. Override with
# F411_STLINK=<serial> / L031_STLINK=<serial>.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${BENCH_OUT:-$HERE/results}"
mkdir -p "$OUT"

F411_STLINK="${F411_STLINK:-0667FF3833584B3043112826}"
L031_STLINK="${L031_STLINK:-066DFF485652867187112939}"

board_info() {
  case "$1" in
    f411) echo "thumbv7em-none-eabihf target/stm32f4x.cfg $F411_STLINK 360" ;;
    l031) echo "thumbv6m-none-eabi target/stm32l0.cfg $L031_STLINK 32" ;;
    *) echo "unknown board $1" >&2; exit 2 ;;
  esac
}

run_image() { # board target cfg serial features name
  local board=$1 target=$2 cfg=$3 serial=$4 features=$5 name=$6
  local port elf raw
  port="$(ls /dev/serial/by-id/*STLink_"$serial"-if02 2>/dev/null | head -1)"
  [ -n "$port" ] || { echo "no VCP for ST-LINK $serial" >&2; exit 1; }
  # The trees image fits the L031's 32 KB only at opt-level s: at 3 the
  # nested-if form of the 90-tree model is expanded further. OPT=s (or z, 2)
  # builds every image at that level instead.
  local opt=${OPT:-3}
  [ -z "${OPT:-}" ] && [ "$board-$name" = l031-trees ] && opt=s
  (cd "$HERE" && CARGO_PROFILE_RELEASE_OPT_LEVEL=$opt cargo build -q --release --target "$target" --features "$features")
  elf="$HERE/target/$target/release/tinymind-bench"
  cp "$elf" "$OUT/$board-$name.elf"
  arm-none-eabi-size "$elf" | tail -1 | awk -v n="$board-$name" '{print n": text="$1" data="$2" bss="$3}' >> "$OUT/sizes.txt"
  raw="$OUT/$board-$name.txt"
  openocd -f interface/stlink.cfg -c "adapter serial $serial" -f "$cfg" \
    -c "program $elf verify exit" >/dev/null 2>&1
  stty -F "$port" 115200 raw -echo
  ( timeout 300 awk '{ sub(/\r$/, ""); print } /^# end bench/ { exit }' < "$port" > "$raw" ) &
  local reader=$!
  sleep 0.5
  openocd -f interface/stlink.cfg -c "adapter serial $serial" -f "$cfg" \
    -c "init; reset run; exit" >/dev/null 2>&1
  wait "$reader" || true
  sed -n '/^# begin bench/,/^# end bench/p' "$raw"
}

: > "$OUT/sizes.txt"
for board in "${@:-f411 l031}"; do
  for b in $board; do
    read -r target cfg serial n <<< "$(board_info "$b")"
    python3 -I "$HERE/gen.py" "$n" >/dev/null
    for wl in ${WORKLOADS:-baseline mlp profile xor-q16 xor-q8 anomaly fft trees kws kws-profile}; do
      features="$b"; [ "$wl" = baseline ] || features="$b,$wl"
      echo "### $b $wl"
      run_image "$b" "$target" "$cfg" "$serial" "$features" "$wl"
    done
  done
done
echo "### sizes"
cat "$OUT/sizes.txt"
