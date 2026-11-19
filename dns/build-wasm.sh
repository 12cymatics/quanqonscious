#!/bin/sh
set -e
cd "$(dirname "$0")"
OUT="${1:-faraday_disc.wasm}"
clang++ --target=wasm32 -O3 -msimd128 -ffp-contract=off -fno-exceptions -fno-rtti \
-nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
-Wl,--initial-memory=100663296 \
-o "$OUT" faraday_disc.cpp
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
