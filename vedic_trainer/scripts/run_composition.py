"""Run a sutra queue through SERIES / PARALLEL / CONCURRENT (and friends).

Examples
--------
    # all 29 sutras, every mode, on a deterministic input
    python scripts/run_composition.py

    # one mode, an explicit queue, JSON out
    python scripts/run_composition.py --mode SERIES --queue 1,3,9,14 --json

    # show the CONCURRENT wave schedule without evaluating
    python scripts/run_composition.py --mode CONCURRENT --show-waves

Queue indices are 1-based on the command line (S1..S29) and converted to the
0-based internal index. Everything runs in exact ℚ; the report shows the
largest denominator reached, which is the real cost driver of exact
composition.

This is a REPORTER, not a gate: it exits 0 whether or not a mode returns the
zero map or raises, because both are expected, documented results for the
canonical queue. SERIES over all 29 *is* the zero map (S20 projects to one
Walsh row, S21 takes absolute values, S22 differences a constant), and
COMPOSITE *does* raise on S17's precondition. Printing those as ordinary
rows is the correct output, not a swallowed failure.

The checks that must fail loudly live elsewhere and do:
`composition.known_annihilating_runs` / `has_known_annihilating_run` match the
queue against a fixed list holding the one annihilating run we know about
(S20 -> S21 -> S22), and that match is asserted in `test_composition.py`;
`scripts/show_sutras.py --verify` exits non-zero on any structural violation.
Note the first pair is a lookup, not a search: it detects the known run and
nothing else, so it cannot certify an arbitrary queue as non-degenerate.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic.kernel import composition as C  # noqa: E402


def parse_queue(text: str) -> List[int]:
    """'1,3,9' -> [0, 2, 8]. 'all' -> every sutra. 1-based in, 0-based out."""
    if text.strip().lower() == "all":
        return list(C.ALL)
    out: List[int] = []
    fields = text.split(",")
    for pos, part in enumerate(fields):
        part = part.strip()
        # An empty field is a malformed queue, not an omission to skip over.
        # This used to `continue`, so `--queue "1,,3"` and `--queue "1,3,"`
        # silently ran a 2-sutra queue while the caller believed they had
        # asked for something else. Composition results depend on queue
        # length (W = ceil(sqrt(N)), the delta weights, the 1/N in COMPOSITE),
        # so a quietly shortened queue produces a different, plausible number.
        if not part:
            raise ValueError(
                f"malformed queue {text!r}: field {pos + 1} of "
                f"{len(fields)} is empty. Expected comma-separated sutra "
                f"numbers 1..{C.N_SUTRAS} with no empty fields, or 'all'.")
        n = int(part)
        if not 1 <= n <= C.N_SUTRAS:
            raise ValueError(f"sutra number out of range 1..{C.N_SUTRAS}: {n}")
        out.append(n - 1)
    if not out:
        raise ValueError("empty queue")
    return out


def default_psi() -> Sequence[Fraction]:
    """A fixed, non-degenerate input with Ψ_0 != 0 (S17's precondition)."""
    return tuple(Fraction(v * v + 1, 3) for v in range(16))


def describe(vec: Sequence[Fraction]) -> dict:
    nz = sum(1 for x in vec if x != 0)
    max_den = max(x.denominator for x in vec)
    max_num = max(abs(x.numerator) for x in vec)
    return {
        "nonzero": nz,
        "all_zero": nz == 0,
        "max_denominator_digits": len(str(max_den)),
        "max_numerator_digits": len(str(max_num)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="ALL",
                    help="SERIES | PARALLEL | CONCURRENT | CANONICAL | COMPOSITE | ALL")
    ap.add_argument("--queue", default="all",
                    help="comma-separated 1-based sutra numbers, or 'all'")
    ap.add_argument("--show-waves", action="store_true",
                    help="print the CONCURRENT wave schedule and exit")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    ks = parse_queue(args.queue)
    psi = default_psi()

    if args.show_waves:
        waves = C.concurrent_waves(ks)
        payload = {
            "queue_size": len(ks),
            "wave_count": C.wave_count(len(ks)),
            "waves": [[f"S{k+1}" for k in w] for w in waves],
        }
        print(json.dumps(payload, indent=2))
        return 0

    modes = (["SERIES", "PARALLEL", "CONCURRENT", "CANONICAL", "COMPOSITE"]
             if args.mode.upper() == "ALL" else [args.mode.upper()])

    results = {}
    for m in modes:
        try:
            out = C.compose(m, psi, ks)
            results[m] = {"ok": True, **describe(out),
                          "value": [f"{x.numerator}/{x.denominator}" for x in out]}
        except ValueError as e:
            # Preconditions raise; they are not silently absorbed.
            results[m] = {"ok": False, "error": str(e)}

    payload = {"queue": [f"S{k+1}" for k in ks], "queue_size": len(ks),
               "results": results}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"queue: {len(ks)} sutras -> {', '.join(payload['queue'])}\n")
    print(f"{'mode':11} {'status':8} {'nonzero':>8} {'max den digits':>15}")
    print("-" * 46)
    for m, r in results.items():
        if r["ok"]:
            flag = "ZERO MAP" if r["all_zero"] else "ok"
            print(f"{m:11} {flag:8} {r['nonzero']:>8} {r['max_denominator_digits']:>15}")
        else:
            print(f"{m:11} {'RAISED':8}  {r['error'][:40]}")

    # Exit 0 even when a mode shows RAISED. This driver's job is to report
    # what each mode does on the given queue and Psi, and "COMPOSITE is out
    # of domain here, because S17 needs Psi_0 != 0" is one of the answers it
    # exists to produce -- reported in the table and in --json, not absorbed.
    # The exit code says whether the report was produced, not whether the
    # subject passed. COMPOSITE raises on the default Psi, so returning
    # non-zero for it would make the documented invocation permanently red,
    # which trains a reader to ignore the status rather than read it.
    # An exception this driver does not expect is not caught at all and
    # leaves a traceback and a non-zero exit of its own.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
