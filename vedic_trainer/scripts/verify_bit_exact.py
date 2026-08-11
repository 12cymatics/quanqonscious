"""Bit-exact verification gate.

Runs the ℚ kernel against the committed fixtures. Exits 0 on success and
1 on any mismatch with a unified diff. Training scripts call this via
``subprocess.check_call`` before any optimizer step is taken.

No floats. The kernel is ℚ throughout; the fixtures store {num, den}
rationals; mismatches are bit-exact integer-pair mismatches.
"""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from vedic.kernel import conservation_exact as ce
from vedic.kernel import sutras_exact as se
from vedic.kernel.q import Q16


REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "fixtures"


def _ensure_fixtures() -> None:
    needed = [
        FIXTURE_DIR / "psi_inputs.json",
        FIXTURE_DIR / "sutra_outputs.json",
        FIXTURE_DIR / "conservation_residuals.json",
    ]
    if all(p.exists() for p in needed):
        return
    cmd = [sys.executable, str(REPO / "scripts" / "build_fixtures.py"), "--out", str(FIXTURE_DIR)]
    subprocess.check_call(cmd, cwd=REPO)


def _obj_to_frac(o: dict[str, int]) -> Fraction:
    return Fraction(int(o["num"]), int(o["den"]))


def _objs_to_q16(objs: list[dict[str, int]]) -> Q16:
    return tuple(_obj_to_frac(o) for o in objs)


def _load(name: str) -> dict[str, object]:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    _ensure_fixtures()

    psi_data = _load("psi_inputs.json")
    sutra_data = _load("sutra_outputs.json")
    cons_data = _load("conservation_residuals.json")

    inputs: list[Q16] = [_objs_to_q16(p) for p in psi_data["inputs"]]
    failures: list[str] = []

    for i, rec in enumerate(sutra_data["records"]):
        psi = _objs_to_q16(rec["input"])
        if psi != inputs[i]:
            failures.append(f"input mismatch at idx {i}")
        # Spot-check a representative subset (the simulator-match test
        # already checks every operator; here we re-verify the high-impact
        # operators that drive the auxiliary losses).
        if _objs_to_q16(rec["S5"]) != se.s5_shunyam_samya(psi):
            failures.append(f"S5 mismatch at idx {i}")
        if _objs_to_q16(rec["S9"]) != se.s9_chalana_kalanabhyam(psi):
            failures.append(f"S9 mismatch at idx {i}")
        if _objs_to_q16(rec["S11"]) != se.s11_vyasti_samasti(psi):
            failures.append(f"S11 mismatch at idx {i}")
        if _objs_to_q16(rec["S29"]) != se.s29_mean_drive(psi):
            failures.append(f"S29 mismatch at idx {i}")

    for i, rec in enumerate(cons_data["records"]):
        psi = inputs[i // 3]
        trace = _obj_to_frac(rec["trace_sum"])
        r1, r2, r3, r4 = ce.all_residuals(psi, trace)
        if _obj_to_frac(rec["R1"]) != r1:
            failures.append(f"R1 mismatch at idx {i}")
        if _obj_to_frac(rec["R2"]) != r2:
            failures.append(f"R2 mismatch at idx {i}")
        if _obj_to_frac(rec["R3"]) != r3:
            failures.append(f"R3 mismatch at idx {i}")
        if _obj_to_frac(rec["R4"]) != r4:
            failures.append(f"R4 mismatch at idx {i}")

    if failures:
        for f in failures[:20]:
            print(f"FAIL: {f}", file=sys.stderr)
        print(f"{len(failures)} bit-exact mismatches", file=sys.stderr)
        return 1
    print(
        f"OK — {len(inputs)} inputs, {len(sutra_data['records'])} sutra records, "
        f"{len(cons_data['records'])} conservation records bit-exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
