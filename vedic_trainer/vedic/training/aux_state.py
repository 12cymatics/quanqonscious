"""Persisting the auxiliary modules, independent of the training stack.

Why this is its own module
--------------------------
The save/load contract lives here rather than on ``VedicTrainer`` because
that class inherits from ``transformers.Trainer``, and importing it drags in
the whole training stack. CI installs the ℚ-only dependencies, so a test that
imported ``VedicTrainer`` to check a state dict round trip failed collection
with ``ModuleNotFoundError: No module named 'transformers'``.

Skipping the test when transformers is absent would have been the easy fix
and the wrong one: the defect being guarded here is that the trained Ψ
projection was silently discarded at save time, and a guard that does not run
in CI is the condition that let it happen. Saving a state dict needs torch
and the modules; it does not need a Trainer. So the contract moved to where
its real dependencies are.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.memory import TesseractWM

AUX_STATE_FILE = "vedic_aux_modules.pt"


def build_aux_modules(d_model: int) -> nn.ModuleDict:
    """The auxiliary modules, in the layout `_save` writes and `load` expects."""
    return nn.ModuleDict({
        "tesseract_wm": TesseractWM(d_model=d_model),
        "hessian": HessianModule(),
        "s5": S5(),
        "s7": S7(),
        "s11": S11(),
    })


def save_aux_state(modules: nn.ModuleDict, checkpoint_dir: str | Path) -> Path:
    """Write the auxiliary state beside the adapter. Returns the path."""
    target = Path(checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / AUX_STATE_FILE
    torch.save(modules.state_dict(), path)
    return path


def load_aux_state(checkpoint_dir: str | Path, d_model: int) -> nn.ModuleDict:
    """Rebuild the auxiliary modules from a checkpoint.

    Raises rather than returning a freshly initialised projection: a randomly
    re-initialised Ψ is not the Ψ that was trained, and silently substituting
    one is exactly how the missing save went unnoticed.
    """
    path = Path(checkpoint_dir) / AUX_STATE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing, so the trained Psi projection cannot be "
            f"restored. Checkpoints written before this was fixed do not "
            f"contain it and cannot be reconstructed -- retrain rather than "
            f"proceeding with a random projection.")
    modules = build_aux_modules(d_model)
    modules.load_state_dict(torch.load(path, map_location="cpu"))
    return modules
