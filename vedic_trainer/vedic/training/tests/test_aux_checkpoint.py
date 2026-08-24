"""The trained Ψ projection must survive a save/load round trip.

Why this exists
---------------
`VedicTrainer`'s docstring claimed the auxiliary modules were "registered as
`self.aux_modules` so they are saved with the checkpoint via the standard
PyTorch state-dict path". No such attribute existed, `_save` was not
overridden, and PEFT's `_save` writes adapter tensors only. The committed
checkpoints hold 240 tensors, none of them TesseractWM or Hessian.

So the 9,216-parameter projection was trained -- an earlier fix specifically
added it to the optimizer -- and then thrown away at save time. Reloading
produced a fresh random orthogonal projection, which means Ψ and every
auxiliary loss derived from it could not be reproduced from a saved run.

These tests exercise the save and load directly, without a HF Trainer, so
they run in the suite rather than only in a training job.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.memory import TesseractWM
from vedic.training.trainer import VedicTrainer

D_MODEL = 32


def _modules() -> nn.ModuleDict:
    return nn.ModuleDict({
        "tesseract_wm": TesseractWM(d_model=D_MODEL),
        "hessian": HessianModule(),
        "s5": S5(), "s7": S7(), "s11": S11(),
    })


def test_the_projection_has_parameters_worth_saving():
    """Guards the rest: if it had none, saving it would prove nothing."""
    n = sum(p.numel() for p in TesseractWM(d_model=D_MODEL).parameters())
    assert n > 0, "TesseractWM has no parameters; the round trip is vacuous"


def test_saved_state_restores_the_same_projection(tmp_path: Path):
    original = _modules()
    with torch.no_grad():                       # make it distinguishable
        original["tesseract_wm"].proj.weight.add_(1.5)
    torch.save(original.state_dict(), tmp_path / VedicTrainer.AUX_STATE_FILE)

    restored = VedicTrainer.load_aux_state(tmp_path, d_model=D_MODEL)
    a = original["tesseract_wm"].proj.weight
    b = restored["tesseract_wm"].proj.weight
    assert torch.equal(a, b), "the restored projection is not the saved one"


def test_a_fresh_module_does_not_match_a_trained_one(tmp_path: Path):
    """The failure mode being prevented: a random re-init standing in."""
    trained = _modules()
    with torch.no_grad():
        trained["tesseract_wm"].proj.weight.add_(1.5)
    fresh = _modules()
    assert not torch.equal(trained["tesseract_wm"].proj.weight,
                           fresh["tesseract_wm"].proj.weight), (
        "a fresh projection equals a trained one, so this test cannot "
        "distinguish restoring from re-initialising")


def test_loading_without_the_file_raises_rather_than_reinitialising(tmp_path):
    """Checkpoints written before the fix have no aux state. Say so."""
    with pytest.raises(FileNotFoundError, match="cannot be restored"):
        VedicTrainer.load_aux_state(tmp_path, d_model=D_MODEL)


def test_the_trainer_declares_a_save_override():
    """_save must be overridden here, not inherited."""
    assert "_save" in vars(VedicTrainer), (
        "VedicTrainer does not override _save, so the auxiliary modules are "
        "not written and the docstring's claim is false again")


def test_every_auxiliary_module_is_covered_by_the_saved_state(tmp_path: Path):
    """A module added to the ModuleDict but missing from the reload is a leak."""
    original = _modules()
    torch.save(original.state_dict(), tmp_path / VedicTrainer.AUX_STATE_FILE)
    restored = VedicTrainer.load_aux_state(tmp_path, d_model=D_MODEL)
    assert set(original.keys()) == set(restored.keys())
    assert (set(original.state_dict().keys())
            == set(restored.state_dict().keys()))
