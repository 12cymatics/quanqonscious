"""Verify the parallel executor returns identical ℚ outputs across modes."""
from __future__ import annotations

from fractions import Fraction
from typing import List

import pytest

from vedic.external import ExecutionMode, SutraExecutor
from vedic.kernel.q import Q16


def _make_inputs() -> List[Q16]:
    return [
        tuple(Fraction(v - 8, 16) for v in range(16)),
        tuple(Fraction(1, v + 2) for v in range(16)),
        tuple(Fraction(7, 100) if v % 2 == 0 else Fraction(-3, 100) for v in range(16)),
        tuple(Fraction(0) for _ in range(16)),
    ]


@pytest.mark.parametrize(
    "mode",
    [ExecutionMode.SERIAL, ExecutionMode.THREADS, ExecutionMode.PROCESSES],
)
def test_executor_modes_agree(mode: ExecutionMode) -> None:
    inputs = _make_inputs()
    serial = SutraExecutor(mode=ExecutionMode.SERIAL).execute(inputs)
    other = SutraExecutor(mode=mode, max_workers=2).execute(inputs)
    assert serial == other


def test_executor_output_is_q16() -> None:
    inputs = _make_inputs()
    outputs = SutraExecutor(mode=ExecutionMode.SERIAL).execute(inputs)
    for out in outputs:
        assert isinstance(out, tuple)
        assert len(out) == 16
        assert all(isinstance(x, Fraction) for x in out)


def test_executor_rejects_unknown_mode() -> None:
    exe = SutraExecutor()
    exe.mode = "no-such-mode"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        exe.execute([_make_inputs()[0]])
