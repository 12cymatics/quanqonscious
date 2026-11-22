#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -n "${LEAN_TOOLCHAIN_ROOT:-}" ]; then
    lean_bin="$LEAN_TOOLCHAIN_ROOT/bin/lean"
    leanc_bin="$LEAN_TOOLCHAIN_ROOT/bin/leanc"
else
    lean_bin="$(command -v lean)"
    leanc_bin="$(command -v leanc)"
fi
version="$("$lean_bin" --version)"
case "$version" in
    "Lean (version 4.36.0,"*) ;;
    *) echo "Lean 4.36.0 is required" >&2; exit 1 ;;
esac
printf '%s\n' "$version" > proof-check.txt
"$lean_bin" -o Main.olean -c Main.c Main.lean >> proof-check.txt 2>&1
"$leanc_bin" -O2 -o gate-kernel Main.c
./gate-kernel version >> proof-check.txt
