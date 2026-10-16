"""The pre-registered decision rules, at their boundaries.

Committed before the experiment produced a number. Every case is constructed
by hand, and the boundaries are tested on both sides: a mean effect equal to
the spread is not enough, a d of exactly 2.0 points does not trigger the
criterion, and a mean EM of exactly 1.0% is not floor-limited.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from vedic.eval.decision import criterion1, em_pct, primary


def test_reduce_needs_every_seed_negative_and_the_mean_past_the_spread():
    ctrl = {0: 1.00, 1: 1.02, 2: 1.04}            # range 0.04
    full = {0: 0.90, 1: 0.92, 2: 0.94}            # range 0.04, every effect -0.10
    v = primary(full, ctrl)
    assert v.conclusion == "reduce"
    assert v.effects == {0: 0.90 - 1.00, 1: 0.92 - 1.02, 2: 0.94 - 1.04}


def test_one_seed_the_other_way_is_no_claim():
    v = primary({0: 0.90, 1: 0.92, 2: 1.05}, {0: 1.00, 1: 1.02, 2: 1.04})
    assert v.conclusion == "no detectable effect"


def test_a_mean_effect_equal_to_the_spread_is_no_claim():
    """Exceeds is strict. Dyadic values keep the float arithmetic exact."""
    ctrl = {0: 1.0, 1: 1.25, 2: 1.5}              # range 0.5
    full = {0: 0.5, 1: 0.75, 2: 1.0}              # effects all -0.5, mean -0.5
    assert primary(full, ctrl).conclusion == "no detectable effect"
    full = {0: 0.375, 1: 0.625, 2: 0.875}         # range 0.5, effects -0.625
    assert primary(full, ctrl).conclusion == "reduce"


def test_increase_is_the_mirror_image():
    assert primary({0: 1.10, 1: 1.12, 2: 1.14}, {0: 1.00, 1: 1.02, 2: 1.04}).conclusion \
        == "increase"


def test_the_larger_arm_range_is_the_bar():
    ctrl = {0: 1.0, 1: 1.0, 2: 1.0}               # range 0
    full = {0: 0.5, 1: 0.75, 2: 0.0}              # range 0.75, effects -0.5,-0.25,-1
    v = primary(full, ctrl)
    assert v.range_full == 0.75 and v.range_control == 0.0
    assert v.mean_effect == -0.5833333333333334
    assert v.conclusion == "no detectable effect"


def test_all_three_seeds_are_required():
    with pytest.raises(ValueError):
        primary({0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0})
    with pytest.raises(ValueError):
        criterion1({0: (0, 10), 1: (0, 10), 3: (0, 10)}, {0: (0, 10), 1: (0, 10), 2: (0, 10)})


def test_em_is_an_exact_percentage():
    assert em_pct(98, 3920) == Fraction(5, 2)
    with pytest.raises(ValueError):
        em_pct(0, 0)


def test_criterion_one_at_exactly_two_points_is_not_triggered():
    n = 3920
    ctrl = {s: (196, n) for s in (0, 1, 2)}       # 5.0%
    full = {s: (196 + 78, n) for s in (0, 1, 2)}  # 6.98...% : d < 2
    assert criterion1(full, ctrl).triggered
    full = {s: (196 + 79, n) for s in (0, 1, 2)}  # d = 79/39.2 = 2.015... > 2
    assert not criterion1(full, ctrl).triggered
    n = 100
    full = {s: (7, n) for s in (0, 1, 2)}         # exactly 2.0 points above 5%
    ctrl = {s: (5, n) for s in (0, 1, 2)}
    v = criterion1(full, ctrl)
    assert v.d_pp == 2 and not v.triggered


def test_floor_limited_needs_both_arms_below_one_percent():
    n = 3920
    zero = {s: (0, n) for s in (0, 1, 2)}
    v = criterion1(zero, zero)
    assert v.triggered and v.floor_limited and v.d_pp == 0
    one_pct = {s: (1, 100) for s in (0, 1, 2)}    # exactly 1.0% (not reachable at n=3920)
    assert not criterion1(one_pct, zero).floor_limited
    just_under = {s: (99, 10000) for s in (0, 1, 2)}
    assert criterion1(just_under, zero).floor_limited


# ------------------------------------------------------- the summarizer


def _write_records(d, ce, em, *, n=3920, skip=None):
    import json
    d.mkdir(parents=True, exist_ok=True)
    names = ["base"] + [f"{a}_seed{s}" for a in ("full", "no_sutra") for s in (0, 1, 2)]
    for name in names:
        if name == skip:
            continue
        (d / f"heldout_{name}.json").write_text(json.dumps(
            {"heldout": {"ce_loss": ce[name], "n_examples": n}}))
        (d / f"em_{name}.json").write_text(json.dumps(
            {"scan": {"length": {"n_correct": em[name], "n_total": n,
                                 "n_budget_exhausted": 0}}}))
        if name != "base":
            line = {"loss": 1.0, "step": 10, "psi_rms": 2.0, "L_CE": 1.0, "L_chi": 0.5,
                    "L_cons": 0.25, "L_curv": 0.125, "L_dual": 0.5, "L_total": 1.5}
            (d / f"trainer_state_{name}.json").write_text(json.dumps(
                {"log_history": [line, {"eval_loss": 1.0, "step": 10},
                                 dict(line, step=1060, psi_rms=0.5)]}))


def _summarize():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.summarize_scan_length import summarize
    return summarize


def test_the_summary_applies_the_rules_to_the_records(tmp_path):
    ce = {"base": 2.0, "full_seed0": 0.5, "full_seed1": 0.625, "full_seed2": 0.75,
          "no_sutra_seed0": 1.0, "no_sutra_seed1": 1.0, "no_sutra_seed2": 1.0}
    em = {"base": 0, "full_seed0": 0, "full_seed1": 0, "full_seed2": 0,
          "no_sutra_seed0": 0, "no_sutra_seed1": 0, "no_sutra_seed2": 0}
    _write_records(tmp_path, ce, em)
    out = _summarize()(tmp_path)
    assert out["primary"]["conclusion"] == "reduce"
    assert out["primary"]["effect_by_seed"] == {0: -0.5, 1: -0.375, 2: -0.25}
    assert out["criterion1"]["triggered"] and out["criterion1"]["floor_limited"]
    assert out["psi"]["full_seed1"] == {
        "first": {"step": 10, "psi_rms": 2.0, "L_CE": 1.0, "L_chi": 0.5, "L_cons": 0.25,
                  "L_curv": 0.125, "L_dual": 0.5, "L_total": 1.5},
        "last": {"step": 1060, "psi_rms": 0.5, "L_CE": 1.0, "L_chi": 0.5, "L_cons": 0.25,
                 "L_curv": 0.125, "L_dual": 0.5, "L_total": 1.5}}


def test_the_summary_refuses_an_incomplete_experiment(tmp_path):
    ce = {k: 1.0 for k in ["base"] + [f"{a}_seed{s}" for a in ("full", "no_sutra")
                                      for s in (0, 1, 2)]}
    em = {k: 0 for k in ce}
    _write_records(tmp_path / "missing", ce, em, skip="full_seed2")
    with pytest.raises(FileNotFoundError, match="not complete"):
        _summarize()(tmp_path / "missing")
    _write_records(tmp_path / "short", ce, em, n=3919)
    with pytest.raises(ValueError, match="not 3920"):
        _summarize()(tmp_path / "short")
