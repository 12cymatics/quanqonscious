"""Apply the pre-registered decision rules to the SCAN length records.

    python scripts/summarize_scan_length.py

Reads results/scan_length/ as ``run_scan_length_experiment.py`` writes it and
prints, then writes to summary.json, the verdicts of
``vedic/eval/decision.py``, every per-seed number they rest on, the base
model's reference values, and each run's psi_rms at its first and last logged
step (the pre-registration's limitation 1). Written before any result
existed; nothing in it is chosen after seeing one.

It refuses to summarise an incomplete experiment: every model needs both
records, each covering the whole split.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic.eval.decision import SEEDS, criterion1, primary  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "scan_length"
N_TEST = 3920


def _load(results: Path, kind: str, name: str) -> dict:
    path = results / f"{kind}_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the experiment is not complete")
    return json.loads(path.read_text(encoding="utf-8"))


def _heldout(results: Path, name: str) -> float:
    h = _load(results, "heldout", name)["heldout"]
    if h["n_examples"] != N_TEST:
        raise ValueError(f"heldout_{name}: {h['n_examples']} examples, not {N_TEST}")
    return h["ce_loss"]


def _em(results: Path, name: str) -> tuple[int, int, int]:
    r = _load(results, "em", name)["scan"]["length"]
    if r["n_total"] != N_TEST:
        raise ValueError(f"em_{name}: {r['n_total']} examples, not {N_TEST}")
    return r["n_correct"], r["n_total"], r["n_budget_exhausted"]


def _psi(results: Path, name: str) -> dict:
    hist = _load(results, "trainer_state", name)["log_history"]
    train = [e for e in hist if "loss" in e and "psi_rms" in e]
    if not train:
        raise ValueError(f"trainer_state_{name}: no training lines carry psi_rms")
    keys = ("step", "psi_rms", "L_CE", "L_chi", "L_cons", "L_curv", "L_dual", "L_total")
    return {"first": {k: train[0][k] for k in keys}, "last": {k: train[-1][k] for k in keys}}


def summarize(results: Path = RESULTS) -> dict:
    ce = {arm: {s: _heldout(results, f"{arm}_seed{s}") for s in SEEDS}
          for arm in ("full", "no_sutra")}
    em = {arm: {s: _em(results, f"{arm}_seed{s}") for s in SEEDS}
          for arm in ("full", "no_sutra")}
    p = primary(ce["full"], ce["no_sutra"])
    c = criterion1({s: em["full"][s][:2] for s in SEEDS},
                   {s: em["no_sutra"][s][:2] for s in SEEDS})
    base_correct, base_total, base_open = _em(results, "base")

    def frac(x: Fraction) -> dict:
        return {"exact": f"{x.numerator}/{x.denominator}", "value": float(x)}

    return {
        "primary": {"conclusion": p.conclusion,
                    "ce": {arm: ce[arm] for arm in ce},
                    "effect_by_seed": p.effects, "mean_effect": p.mean_effect,
                    "range_full": p.range_full, "range_no_sutra": p.range_control},
        "criterion1": {"triggered": c.triggered, "floor_limited": c.floor_limited,
                       "d_pp": frac(c.d_pp),
                       "em_pct": {"full": {s: frac(v) for s, v in c.em_full_pct.items()},
                                  "no_sutra": {s: frac(v) for s, v in c.em_control_pct.items()}},
                       "n_correct": {arm: {s: em[arm][s][0] for s in SEEDS} for arm in em},
                       "n_budget_exhausted": {arm: {s: em[arm][s][2] for s in SEEDS}
                                              for arm in em}},
        "base": {"ce": _heldout(results, "base"), "em_n_correct": base_correct,
                 "em_n_total": base_total, "em_n_budget_exhausted": base_open},
        "psi": {f"{arm}_seed{s}": _psi(results, f"{arm}_seed{s}")
                for arm in ("full", "no_sutra") for s in SEEDS},
    }


def main() -> int:
    out = summarize()
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
