"""The treatment configs are what the pre-registered rule makes of the controls.

The six configs under ``configs/scan_length/`` are part of the
pre-registration. The treatment arm differs from the control arm in its
auxiliary weights alone, and those weights come from a measurement, not from
a choice. This checks the committed files against each other: that each
``full_seed<k>.yaml`` is its control with only the weights and output_dir
changed, that the weights are the rule applied to the committed measurement,
and that the measurement was taken on this control config, this corpus and a
clean committed tree.

Re-measuring needs the 135M model; ``calibrate_aux_weights.py`` does that,
and it reproduced these measurements bit for bit in a fresh environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs" / "scan_length"
SEEDS = (0, 1, 2)
TERMS = {"L_chi": "alpha_chi", "L_cons": "beta_cons",
         "L_curv": "gamma_curv", "L_dual": "delta_dual"}
#: SHA-256 of the length-split train corpus, as test_scan_data.py pins it.
TRAIN_SHA256 = "0c6f7628058cc2207881373d7d20494a5005111cb255ab4f6678bcf82308a8cb"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def _calibration(seed: int) -> dict:
    return json.loads((CONFIGS / f"calibration_seed{seed}.json").read_text(encoding="utf-8"))


def _script():
    import sys
    sys.path.insert(0, str(REPO))
    from scripts import calibrate_aux_weights
    return calibrate_aux_weights


@pytest.mark.parametrize("seed", SEEDS)
def test_the_control_has_no_auxiliary_weight(seed):
    assert set(_load(f"no_sutra_seed{seed}.yaml")["loss_weights"].values()) == {0.0}


@pytest.mark.parametrize("seed", SEEDS)
def test_the_treatment_differs_only_in_weights_and_output_dir(seed):
    control = _load(f"no_sutra_seed{seed}.yaml")
    full = _load(f"full_seed{seed}.yaml")
    assert full["seed"] == control["seed"] == seed
    assert full["output_dir"] == f"checkpoints/scan_length/full_seed{seed}"
    assert control["output_dir"] == f"checkpoints/scan_length/no_sutra_seed{seed}"
    assert set(full) == set(control)
    for key in control:
        if key not in ("loss_weights", "output_dir"):
            assert full[key] == control[key], key


@pytest.mark.parametrize("seed", SEEDS)
def test_the_weights_are_the_rule_applied_to_the_measurement(seed):
    cal = _calibration(seed)
    s = _script()
    assert cal["rule"] == {"fraction": s.FRACTION, "records": s.CALIBRATION_RECORDS,
                           "batch": s.CALIBRATION_BATCH,
                           "gradients_of": "LoRA parameters",
                           "mode": "eval (no dropout)"}
    g_ce = cal["measured"]["L_CE"]["grad_norm_lora"]
    expected = {TERMS[k]: s.FRACTION * g_ce / cal["measured"][k]["grad_norm_lora"]
                for k in TERMS}
    assert cal["weights"] == expected
    assert _load(f"full_seed{seed}.yaml")["loss_weights"] == expected
    assert all(w > 0 for w in expected.values())


@pytest.mark.parametrize("seed", SEEDS)
def test_the_measurement_was_taken_on_this_control_corpus_and_a_clean_tree(seed):
    cal = _calibration(seed)
    control = (CONFIGS / f"no_sutra_seed{seed}.yaml").read_bytes()
    assert cal["control_config_sha256"] == hashlib.sha256(control).hexdigest()
    assert cal["train_sha256"] == TRAIN_SHA256
    code = cal["environment"]["code"]
    assert len(code["commit"]) == 40
    assert code["tracked_files_modified"] is False
    assert cal["environment"]["torch_num_threads"] == _load(f"no_sutra_seed{seed}.yaml")["num_threads"]


def test_cross_entropy_at_initialisation_does_not_depend_on_the_seed():
    """LoRA's B starts at zero, so the initial model is the base model for
    every seed and CE₀ must be identical; only the Ψ projection differs. A
    difference here would mean the seeds did not start from the same model."""
    ce = {_calibration(s)["measured"]["L_CE"]["value"] for s in SEEDS}
    assert len(ce) == 1
