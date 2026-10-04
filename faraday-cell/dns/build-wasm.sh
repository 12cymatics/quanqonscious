#!/bin/sh
# Build dns/faraday_disc.cpp to dns/faraday_disc.wasm.
#
# clang targets wasm32 directly; Emscripten is not used and not needed, because
# this module has no libc surface at all -- no I/O, no allocation, no
# transcendentals. The flags are load bearing:
#
#   -msimd128           128-bit SIMD. Measured at 2.09x over the scalar build
#                       and BIT-IDENTICAL to it -- 3504 of 3504 values exact at
#                       48 x 24 -- because clang vectorises the element-wise
#                       loops, which are order independent, and leaves the dot
#                       product reductions alone. Speed with no change to the
#                       arithmetic, which is the only kind worth taking here.
#                       SIMD is baseline in every browser that also has the
#                       workers, Float64Array and canvas this page already needs;
#                       the loader refuses with a named reason if it is missing
#                       rather than substituting anything.
#   -ffp-contract=off   no fused multiply-add. An FMA rounds once where
#                       JavaScript rounds twice, and the two would drift apart,
#                       which would destroy the parity the tests assert. THAT IS
#                       THE INTENT AND IT IS UNVERIFIED, and it is recorded that
#                       way rather than asserted. Measured by injection: built
#                       with -ffp-contract=fast, dns/check-wasm-build.mjs is still
#                       green -- and that gate holds the fresh module, the
#                       committed module and the JavaScript solver to element-wise
#                       identical values over a FULL DRIVE PERIOD on every
#                       unknown. The same injection on dns/faraday_cell3d.cpp
#                       leaves its own 55-check parity gate green too, changing
#                       the module by one byte out of 71403. The flag stays
#                       because it is the correct intent; it is not load bearing
#                       on any evidence collected here.
#   -fno-exceptions     nothing to unwind, and no libc to unwind with.
#   -fno-rtti           no type info, nothing needs it.
#   -nostdlib           freestanding.
#   --no-entry          a library, not a program.
#   --export-all        JavaScript calls the exported functions by name.
#   --initial-memory    must cover the static arena; see ARENA in the source.
#
# There is no -ffast-math and there never will be: it permits reassociation,
# which is exactly the licence to change the arithmetic that this repository
# refuses everywhere else.
#
# An output path may be given as the first argument; without one it writes
# dns/faraday_disc.wasm beside the source. dns/check-wasm-build.mjs uses the
# argument to build a fresh module without overwriting the committed one, so it
# can hold the two to producing identical numbers.
set -e
cd "$(dirname "$0")"
OUT="${1:-faraday_disc.wasm}"
clang++ --target=wasm32 -O3 -msimd128 -ffp-contract=off -fno-exceptions -fno-rtti \
  -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
  -Wl,--initial-memory=100663296 \
  -o "$OUT" faraday_disc.cpp
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
