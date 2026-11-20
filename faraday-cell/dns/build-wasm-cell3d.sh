#!/bin/sh
set -e
cd "$(dirname "$0")"
OUT="${1:-faraday_cell3d.wasm}"
clang++ --target=wasm32 -O3 -msimd128 -ffp-contract=off -fno-exceptions -fno-rtti \
-nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
-Wl,--initial-memory=134217728 \
-o "$OUT" faraday_cell3d.cpp
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
