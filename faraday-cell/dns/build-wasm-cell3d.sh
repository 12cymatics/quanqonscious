#!/bin/sh
# Build dns/faraday_cell3d.cpp to dns/faraday_cell3d.wasm.
#
# The same toolchain and the same flags as dns/build-wasm.sh, for the same
# reasons, and they are load bearing:
#
#   -msimd128           128-bit SIMD. clang vectorises the element-wise loops,
#                       which are order independent, and leaves the dot-product
#                       reductions alone -- speed with no change to the
#                       arithmetic, which is the only kind worth taking here.
#   -ffp-contract=off   no fused multiply-add. An FMA rounds once where
#                       JavaScript rounds twice, and the two would drift apart,
#                       which would destroy the parity the gate asserts. THIS ONE
#                       IS UNVERIFIED HERE, and it is recorded that way rather
#                       than asserted. Measured by injection: building with
#                       -ffp-contract=fast changes the module by ONE BYTE out of
#                       71 403 and dns/check-cell3d-wasm.mjs stays green -- so on
#                       this target, with this source, the flag's effect on the
#                       numbers could not be demonstrated. It stays because it is
#                       the correct intent; it is not load bearing on any
#                       evidence collected here. The same injection on the
#                       two-dimensional module leaves dns/check-wasm-build.mjs
#                       green as well, and that gate compares three
#                       implementations element for element over a full drive
#                       period -- so dns/build-wasm.sh now records the flag the
#                       same way rather than claiming it.
#   -fno-exceptions     nothing to unwind, and no libc to unwind with. The
#                       module reports a refusal as an error code and the loader
#                       raises the Error the JavaScript would have thrown.
#   -fno-rtti           no type info, nothing needs it.
#   -nostdlib           freestanding. No libc means no libm, which is the point:
#                       a transcendental computed here could not be bit-identical
#                       to JavaScript's, so none is computed here.
#   --no-entry          a library, not a program.
#   --export-all        JavaScript calls the exported functions by name.
#   --initial-memory    must cover the static arena; see ARENA in the source.
#                       128 MiB against ARENA's 96 MB of doubles.
#
# There is no -ffast-math and there never will be: it permits reassociation,
# which is exactly the licence to change the arithmetic that this repository
# refuses everywhere else.
#
# An output path may be given as the first argument; without one it writes
# dns/faraday_cell3d.wasm beside the source. dns/check-cell3d-wasm.mjs uses the
# argument to build a fresh module without overwriting the committed one, so it
# can hold the two to producing identical numbers.
set -e
cd "$(dirname "$0")"
OUT="${1:-faraday_cell3d.wasm}"
clang++ --target=wasm32 -O3 -msimd128 -ffp-contract=off -fno-exceptions -fno-rtti \
  -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
  -Wl,--initial-memory=134217728 \
  -o "$OUT" faraday_cell3d.cpp
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
