#!/bin/sh
# Compile the canonical fluid core for a native host without changing arithmetic.
# With no extra sources, produce an object file for a coupled solver to link.
# Usage: native/build.sh [output-object]
#        native/build.sh output-executable entry.cpp [additional-source.cpp ...]
set -eu
NATIVE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'USAGE'
Usage:
  native/build.sh
      Build native/build/fluid-runtime.o using CXX (default: c++).
  native/build.sh output-object
      Build only the canonical native fluid runtime object.
  native/build.sh output-executable entry.cpp [additional-source.cpp ...]
      Link the runtime with the explicitly selected sources. Paths may be absolute.

Examples, run from the repository root:
  node native/check-fluid-runtime.mjs
  native/build.sh native/build/faraday-native native/run.cpp \
    native/rigid-coupler.cpp native/rigid-interpolation.cpp native/rigid-contact.cpp

Requires a POSIX shell and a C++20 compiler. Arithmetic uses -ffp-contract=off
and -fno-fast-math. No generated fixture or executable is required by the checks.
USAGE
  exit 0
fi
OUTPUT=${1:-"$NATIVE_DIR/build/fluid-runtime.o"}
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$(dirname -- "$OUTPUT")"
if [ "$#" -eq 0 ]; then
  "${CXX:-c++}" -std=c++20 -O3 -ffp-contract=off -fno-fast-math \
    -Wall -Wextra -c "$NATIVE_DIR/fluid-runtime.cpp" -o "$OUTPUT"
else
  "${CXX:-c++}" -std=c++20 -O3 -ffp-contract=off -fno-fast-math \
    -Wall -Wextra -I"$NATIVE_DIR" "$NATIVE_DIR/fluid-runtime.cpp" "$@" -o "$OUTPUT"
fi
printf 'Built %s\n' "$OUTPUT"
