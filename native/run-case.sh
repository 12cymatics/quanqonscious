#!/bin/sh
# Explicit native experiment launcher. Keeps an incomplete recording on failure.
set -eu

usage() {
  printf '%s\n' \
    'Usage: sh native/run-case.sh --case input.json --out NEW_OUTPUT_DIRECTORY' \
    'Requires Node.js and a C++20 compiler. The output directory must not exist.' \
    'Builds, prepares the declared case, then runs it without changing its grid or physics.'
}
fail() { printf '%s\n' "$1" >&2; exit 1; }
CASE_PATH=
OUTPUT_DIR=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --case)
      [ -z "$CASE_PATH" ] || fail '--case was supplied more than once.'
      [ "$#" -ge 2 ] || fail '--case requires a path.'
      CASE_PATH=$2; shift 2 ;;
    --out)
      [ -z "$OUTPUT_DIR" ] || fail '--out was supplied more than once.'
      [ "$#" -ge 2 ] || fail '--out requires a new directory path.'
      OUTPUT_DIR=$2; shift 2 ;;
    *) usage >&2; fail "Unknown argument: $1" ;;
  esac
done
[ -n "$CASE_PATH" ] && [ -n "$OUTPUT_DIR" ] || { usage >&2; exit 1; }
case "$CASE_PATH" in /*) ;; *) CASE_PATH="$PWD/$CASE_PATH" ;; esac
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;; esac
[ -f "$CASE_PATH" ] && [ -r "$CASE_PATH" ] || fail "Cannot read case: $CASE_PATH"
command -v node >/dev/null 2>&1 || fail 'Node.js is required; see native/README.md.'
command -v "${CXX:-c++}" >/dev/null 2>&1 || fail 'A C++20 compiler is required; set CXX to its executable path.'
NATIVE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# mkdir is intentionally exclusive: an earlier result is never replaced.
mkdir -- "$OUTPUT_DIR" || fail "Cannot create new output directory: $OUTPUT_DIR"
printf 'Building the native executable. Log: %s/build.log\n' "$OUTPUT_DIR"
if sh "$NATIVE_DIR/build.sh" "$OUTPUT_DIR/faraday-native" \
    "$NATIVE_DIR/run.cpp" "$NATIVE_DIR/rigid-coupler.cpp" \
    "$NATIVE_DIR/rigid-contact.cpp" "$NATIVE_DIR/rigid-interpolation.cpp" \
    >"$OUTPUT_DIR/build.log" 2>&1; then :
else
  cat "$OUTPUT_DIR/build.log" >&2
  fail 'Native build failed; the output directory and log were retained.'
fi
printf 'Preparing the declared case. Log: %s/preparation.log\n' "$OUTPUT_DIR"
if node "$NATIVE_DIR/prepare-case.mjs" --case "$CASE_PATH" --out "$OUTPUT_DIR/prepared.fns" \
    >"$OUTPUT_DIR/preparation.json" 2>"$OUTPUT_DIR/preparation.log"; then :
else
  cat "$OUTPUT_DIR/preparation.log" >&2
  fail 'Case preparation refused; no native calculation was started.'
fi
printf 'Running the native calculation. Progress: %s/solver.log\n' "$OUTPUT_DIR"
if "$OUTPUT_DIR/faraday-native" "$OUTPUT_DIR/prepared.fns" \
    >"$OUTPUT_DIR/recording.partial.jsonl" 2>"$OUTPUT_DIR/solver.log"; then
  mv -- "$OUTPUT_DIR/recording.partial.jsonl" "$OUTPUT_DIR/recording.jsonl"
  printf 'Completed the declared interval. Import: %s/recording.jsonl\n' "$OUTPUT_DIR"
else
  NATIVE_STATUS=$?
  cat "$OUTPUT_DIR/solver.log" >&2
  printf 'Calculation stopped with status %s. Incomplete result retained: %s/recording.partial.jsonl\n' "$NATIVE_STATUS" "$OUTPUT_DIR" >&2
  exit "$NATIVE_STATUS"
fi
