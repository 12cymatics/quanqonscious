#!/bin/sh
set -e
cd "$(dirname "$0")"
OUT="${1:-faraday_disc_d128.wasm}"
clang++ --target=wasm32 -O3 -mbulk-memory -fno-exceptions -fno-rtti \
  -nostdlib -Wl,--no-entry -Wl,--export-all \
  -Wl,--initial-memory=67108864 \
  -o "$OUT" faraday_disc_d128.cpp
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
