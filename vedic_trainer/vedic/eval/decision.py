"""The decision rules of PREREGISTRATION_SCAN_LENGTH.md, as code.

Written and committed before any result of the experiment existed, so the
verdict is computed by rules fixed in advance rather than read off the
numbers. Each function states the rule it implements, word for word where
the pre-registration gives words.

Exact-match rates are kept as Fractions of integer counts, so the 2.0-point
threshold and the 1.0% floor are compared exactly. Cross-entropies are the
measured floats and are compared as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

SEEDS = (0, 1, 2)
#: Criterion 1 threshold, in percentage points.
EM_THRESHOLD_PP = Fraction(2)
#: Below this mean EM in both arms the comparison is floor-limited, in percent.
EM_FLOOR_PCT = Fraction(1)


def _check_seeds(*arms: Mapping[int, object]) -> None:
    for arm in arms:
        if tuple(sorted(arm)) != SEEDS:
            raise ValueError(f"need exactly seeds {SEEDS}; got {sorted(arm)}")


@dataclass(frozen=True)
class PrimaryVerdict:
    effects: dict[int, float]          # CE(full_s) - CE(no_sutra_s)
    mean_effect: float
    range_full: float
    range_control: float
    conclusion: str                    # "reduce", "increase" or "no detectable effect"


def primary(ce_full: Mapping[int, float], ce_control: Mapping[int, float]) -> PrimaryVerdict:
    """The effect for seed s is CE(full_s) − CE(no_sutra_s). "Reduce" is
    claimed only if the effect is negative for all three seeds and |mean
    effect| exceeds the larger of the two arms' across-seed ranges (max − min
    of the three CEs); "increase" the same way with the sign reversed;
    otherwise "no detectable effect at this scale"."""
    _check_seeds(ce_full, ce_control)
    effects = {s: ce_full[s] - ce_control[s] for s in SEEDS}
    mean = sum(effects.values()) / len(SEEDS)
    r_full = max(ce_full.values()) - min(ce_full.values())
    r_ctrl = max(ce_control.values()) - min(ce_control.values())
    spread = max(r_full, r_ctrl)
    if all(e < 0 for e in effects.values()) and abs(mean) > spread:
        conclusion = "reduce"
    elif all(e > 0 for e in effects.values()) and abs(mean) > spread:
        conclusion = "increase"
    else:
        conclusion = "no detectable effect"
    return PrimaryVerdict(effects, mean, r_full, r_ctrl, conclusion)


@dataclass(frozen=True)
class Criterion1Verdict:
    em_full_pct: dict[int, Fraction]
    em_control_pct: dict[int, Fraction]
    d_pp: Fraction                     # mean over seeds of EM(full) - EM(control)
    triggered: bool                    # the hypothesis is falsified on this experiment
    floor_limited: bool


def em_pct(n_correct: int, n_total: int) -> Fraction:
    if n_total <= 0:
        raise ValueError("an empty split has no exact-match rate")
    return Fraction(100 * n_correct, n_total)


def criterion1(em_full: Mapping[int, tuple[int, int]],
               em_control: Mapping[int, tuple[int, int]]) -> Criterion1Verdict:
    """d = the mean over seeds of EM(full_s) − EM(no_sutra_s), in percentage
    points. The criterion is triggered iff d < 2.0. If both arms' mean EM is
    below 1.0%, the outcome is floor-limited: the criterion still falls as
    computed, but the comparison is uninformative.

    Each value is (n_correct, n_total)."""
    _check_seeds(em_full, em_control)
    full = {s: em_pct(*em_full[s]) for s in SEEDS}
    ctrl = {s: em_pct(*em_control[s]) for s in SEEDS}
    d = sum((full[s] - ctrl[s] for s in SEEDS), Fraction(0)) / len(SEEDS)
    mean_full = sum(full.values(), Fraction(0)) / len(SEEDS)
    mean_ctrl = sum(ctrl.values(), Fraction(0)) / len(SEEDS)
    return Criterion1Verdict(full, ctrl, d, triggered=d < EM_THRESHOLD_PP,
                             floor_limited=mean_full < EM_FLOOR_PCT and mean_ctrl < EM_FLOOR_PCT)
