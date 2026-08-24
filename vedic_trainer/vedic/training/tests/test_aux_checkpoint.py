"""The trained Ψ projection must survive a save/load round trip.

Why this exists
---------------
`VedicTrainer`'s docstring claimed the auxiliary modules were "registered as
`self.aux_modules` so they are saved with the checkpoint via the standard
PyTorch state-dict path". No such attribute existed, `_save` was not
overridden, and PEFT's `_save` writes adapter tensors only -- the committed
checkpoints hold 240 tensors, none of them TesseractWM or Hessian.

So the 9,216-parameter projection was trained (an earlier fix specifically
added it to the optimizer) and then thrown away at save time. Reloading gave
a fresh random orthogonal projection, so Ψ and every auxiliary loss derived
from it could not be reproduced from a saved run.

Nothing here imports `VedicTrainer`. An earlier version of this file did,
and failed collection in CI with `ModuleNotFoundError: No module named
'transformers'` -- the trainer inherits from `transformers.Trainer`, which
CI does not install. Skipping when transformers is absent would have made
the guard inert in the one environment that runs it automatically, which is
the condition that let the original defect through. The save contract lives
in `vedic/training/aux_state.py`, whose real dependencies are torch and the
modules themselves; the `_save` override is checked by parsing the source.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from vedic.memory import TesseractWM
from vedic.training.aux_state import (
    AUX_STATE_FILE,
    build_aux_modules,
    load_aux_state,
    save_aux_state,
)

D_MODEL = 32
TRAINER_SRC = Path(__file__).resolve().parents[1] / "trainer.py"


def test_the_projection_has_parameters_worth_saving():
    """Guards the rest: if it had none, saving it would prove nothing."""
    n = sum(p.numel() for p in TesseractWM(d_model=D_MODEL).parameters())
    assert n > 0, "TesseractWM has no parameters; the round trip is vacuous"


def test_saved_state_restores_the_same_projection(tmp_path: Path):
    original = build_aux_modules(D_MODEL)
    with torch.no_grad():                       # make it distinguishable
        original["tesseract_wm"].proj.weight.add_(1.5)
    save_aux_state(original, tmp_path)

    restored = load_aux_state(tmp_path, d_model=D_MODEL)
    assert torch.equal(original["tesseract_wm"].proj.weight,
                       restored["tesseract_wm"].proj.weight), \
        "the restored projection is not the saved one"


def test_a_fresh_module_does_not_match_a_trained_one():
    """The failure mode being prevented: a random re-init standing in."""
    trained = build_aux_modules(D_MODEL)
    with torch.no_grad():
        trained["tesseract_wm"].proj.weight.add_(1.5)
    fresh = build_aux_modules(D_MODEL)
    assert not torch.equal(trained["tesseract_wm"].proj.weight,
                           fresh["tesseract_wm"].proj.weight), (
        "a fresh projection equals a trained one, so this test cannot "
        "distinguish restoring from re-initialising")


def test_loading_without_the_file_raises_rather_than_reinitialising(tmp_path):
    """Checkpoints written before the fix have no aux state. Say so."""
    with pytest.raises(FileNotFoundError, match="cannot be restored"):
        load_aux_state(tmp_path, d_model=D_MODEL)


def test_save_writes_the_expected_filename(tmp_path: Path):
    path = save_aux_state(build_aux_modules(D_MODEL), tmp_path)
    assert path.name == AUX_STATE_FILE
    assert path.exists()


def test_every_auxiliary_module_is_covered_by_the_saved_state(tmp_path: Path):
    """A module added to the dict but missing from the reload is a leak."""
    original = build_aux_modules(D_MODEL)
    save_aux_state(original, tmp_path)
    restored = load_aux_state(tmp_path, d_model=D_MODEL)
    assert set(original.keys()) == set(restored.keys())
    assert (set(original.state_dict().keys())
            == set(restored.state_dict().keys()))


def test_the_trainer_overrides_save_and_calls_the_helper():
    """Parsed, not imported: this must hold where transformers is absent too.

    Without the override, PEFT's `_save` writes adapter tensors only and the
    projection is discarded again -- silently, exactly as before.
    """
    tree = ast.parse(TRAINER_SRC.read_text(encoding="utf-8"))
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "VedicTrainer"),
               None)
    assert cls is not None, "VedicTrainer not found in trainer.py"

    save = next((n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_save"), None)
    assert save is not None, (
        "VedicTrainer does not override _save, so the auxiliary modules are "
        "not written and the docstring's claim is false again")

    called = {n.func.id for n in ast.walk(save)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "save_aux_state" in called, (
        "_save is overridden but does not call save_aux_state, so nothing "
        "guarantees the auxiliary state is written")
