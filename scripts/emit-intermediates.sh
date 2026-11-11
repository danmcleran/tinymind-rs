#!/usr/bin/env bash
# Build the examples (host) or the board images (Arm) and keep the compiler's
# intermediate files, to read:
#
#   <name>.mir   MIR: Rust after type checking and borrow checking, generics
#                not yet instantiated
#   <name>.ll    LLVM IR after optimisation
#   <name>.s     assembly
#   <name>.lst   Arm only: disassembly of the linked image (final addresses,
#                loop alignment, literal pools), from arm-none-eabi-objdump
#
#   scripts/emit-intermediates.sh [host|arm|all]      default: host
#
# host  every example in examples/, plus the library (tinymind.*), for this
#       machine, into target/intermediates/host/. Most library code is generic,
#       so it appears in the example that instantiates it, not in tinymind.*.
#       Built with one codegen unit so each crate gives one .s and one .ll; a
#       normal release build splits a crate into several, so inlining can
#       differ slightly from it.
# arm   the examples need std, so they cannot be built for a microcontroller.
#       The Arm code is the firmware in bench/: one image per board and
#       workload, built exactly as bench/run.sh builds it (LTO, one codegen
#       unit, the same flags), into target/intermediates/arm/<board>/. Each
#       image is one crate after LTO, so its files hold the library code too.
#       Boards: f411 (Cortex-M4F, thumbv7em-none-eabihf) and l031 (Cortex-M0+,
#       thumbv6m-none-eabi).
#
# Environment:
#   EMIT=asm,llvm-ir,mir     which forms (also: llvm-bc, obj)
#   OUT=target/intermediates where to put them
#   EMIT_RUSTFLAGS=...       host: extra rustc flags, e.g.
#                            "-C llvm-args=-x86-asm-syntax=intel" or
#                            "-C debuginfo=1" (source lines in the .s and .ll)
#   BOARDS="f411 l031"       arm: which boards
#   WORKLOADS="mlp kws"      arm: which workloads (default: every one that
#                            fits the board; the mlperf ones only if their
#                            generated sources exist, see bench/README.md)
#   OPT=s                    arm: opt-level for every image (default 3, and s
#                            for the L031 trees image, as bench/run.sh)
#
# Builds use their own target directories (target/emit, bench/target/emit),
# so they do not disturb the normal builds' caches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mode="${1:-host}"
emit="${EMIT:-asm,llvm-ir,mir}"
out="${OUT:-target/intermediates}"
case "$out" in /*) ;; *) out="$ROOT/$out" ;; esac
status=0

exts() { # the file extensions $emit produces
  local kind
  for kind in ${emit//,/ }; do
    case "$kind" in
      asm) echo s ;;
      llvm-ir) echo ll ;;
      mir) echo mir ;;
      llvm-bc) echo bc ;;
      obj) echo o ;;
      *) echo "emit-intermediates: unknown EMIT kind '$kind'" >&2; exit 2 ;;
    esac
  done
}

collect() { # collect <dir> <built file stem> <dest dir> <name>
  local ext src
  mkdir -p "$3"
  for ext in $(exts); do
    # Newest match: an older build with other flags leaves files under another hash.
    src="$(find "$1" -maxdepth 1 -name "$2-*.$ext" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2-)"
    if [ -z "$src" ]; then
      echo "emit-intermediates: no .$ext for $4" >&2
      status=1
      continue
    fi
    cp "$src" "$3/$4.$ext"
  done
}

host() {
  RUSTFLAGS="--emit=$emit -C codegen-units=1 ${EMIT_RUSTFLAGS:-}" \
    cargo build --release --features std --examples --target-dir target/emit
  local example name
  for example in examples/*.rs; do
    name="$(basename "$example" .rs)"
    collect target/emit/release/examples "$name" "$out/host" "$name"
  done
  collect target/emit/release/deps tinymind "$out/host" tinymind
}

arm() {
  local board target n workloads wl opt features deps
  for board in ${BOARDS:-f411 l031}; do
    case "$board" in
      f411) target=thumbv7em-none-eabihf n=360 workloads="mlp xor-q16 xor-q8 anomaly trees kws" ;;
      l031) target=thumbv6m-none-eabi n=32 workloads="mlp xor-q16 xor-q8 anomaly trees kws" ;;
      *) echo "emit-intermediates: unknown board '$board'" >&2; exit 2 ;;
    esac
    if [ "$board" = f411 ]; then
      for wl in mlperf-kws mlperf-ic; do
        [ -f "bench/src/gen/${wl//-/_}.rs" ] && workloads="$workloads $wl"
      done
    fi
    python3 -I bench/gen.py "$n" >/dev/null
    deps="bench/target/emit/$target/release/deps"
    for wl in ${WORKLOADS:-$workloads}; do
      opt="${OPT:-3}"
      [ -z "${OPT:-}" ] && [ "$board-$wl" = l031-trees ] && opt=s
      features="$board"; [ "$wl" = baseline ] || features="$board,$wl"
      echo "### $board $wl (opt-level $opt)"
      # Remove the last image's files: every image is the same crate, and a
      # build that fails must not leave the previous workload's files to copy.
      rm -f "$deps"/tinymind_bench-*
      (cd bench && CARGO_PROFILE_RELEASE_OPT_LEVEL=$opt cargo rustc -q --release \
        --target "$target" --features "$features" --target-dir target/emit -- --emit="$emit")
      collect "$deps" tinymind_bench "$out/arm/$board" "$wl"
      if command -v arm-none-eabi-objdump >/dev/null; then
        arm-none-eabi-objdump -d -C "bench/target/emit/$target/release/tinymind-bench" \
          > "$out/arm/$board/$wl.lst"
      fi
    done
  done
}

case "$mode" in
  host) host ;;
  arm) arm ;;
  all) host; arm ;;
  *) echo "usage: emit-intermediates.sh [host|arm|all]" >&2; exit 2 ;;
esac
[ "$status" -eq 0 ] || exit "$status"
echo "emit-intermediates: wrote $(find "$out" -type f | wc -l) files under $out"
