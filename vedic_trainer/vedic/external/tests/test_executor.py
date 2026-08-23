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
    """The three modes must agree — and on outputs that can disagree.

    3 of the 4 fixture inputs run through `_PIPELINE` to the all-zero vector,
    and on the zero vector every implementation agrees no matter what it
    does. Asserting `serial == other` across all four therefore tested one
    real case and three vacuous ones. The degenerate rows are still compared
    (they must stay zero), but agreement is separately required on at least
    one row that is not.
    """
    inputs = _make_inputs()
    serial = SutraExecutor(mode=ExecutionMode.SERIAL).execute(inputs)
    other = SutraExecutor(mode=mode, max_workers=2).execute(inputs)
    assert serial == other

    live = [i for i, out in enumerate(serial) if any(v != 0 for v in out)]
    assert live, (
        "every fixture input collapsed to the zero vector, so this comparison "
        "cannot distinguish a correct executor from a broken one. Add an "
        "input that survives the pipeline.")
    for i in live:
        assert serial[i] == other[i]


def test_the_pipeline_degeneracy_is_known_and_pinned() -> None:
    """Which inputs collapse is a property of the operator set, so pin it.

    If a future change makes more inputs collapse, the mode-agreement test
    above quietly loses coverage. This fails instead.
    """
    outs = SutraExecutor(mode=ExecutionMode.SERIAL).execute(_make_inputs())
    collapsed = [i for i, out in enumerate(outs) if all(v == 0 for v in out)]
    assert collapsed == [0, 2, 3], (
        f"the set of collapsing inputs changed to {collapsed}; check whether "
        f"test_executor_modes_agree still has a live case")


def test_the_pipeline_declares_which_sutras_it_omits() -> None:
    """22 of 29, and the 7 excluded are named rather than merely missing."""
    from vedic.external.executor import _PIPELINE

    assert len(_PIPELINE) == 22
    names = {fn.__name__.split("_")[0] for fn in _PIPELINE}
    excluded = {f"s{n}" for n in (3, 17, 23, 18, 27, 7, 22)} - names
    assert excluded, "the exclusion list no longer matches the pipeline"


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
