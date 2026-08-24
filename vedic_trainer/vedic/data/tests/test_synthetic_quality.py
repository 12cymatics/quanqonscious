"""Synthetic-data invariants — exact ℚ throughout, no floats."""
from __future__ import annotations

from fractions import Fraction

from vedic.data.audit_filter import audit_psi
from vedic.data.synthetic_contradiction import generate_contradiction_pair
from vedic.data.synthetic_paraphrase import axis_difference, generate_paraphrase_pair
from vedic.data.tesseract_encode import decode_psi_to_axes, encode_text_to_psi


_BASE_TEXTS = (
    "I saw the cat sleeping on the mat",
    "We will meet tomorrow at noon",
    "It was likely raining when she arrived",
    "They never finished the project",
    "He observed the experiment carefully",
    "She might travel to Paris next month",
    "The proof appeared in the journal yesterday",
    "I have direct evidence of the result",
    "He cannot attend the meeting",
    "We are going to the conference",
)


def test_encode_decode_roundtrip() -> None:
    for text in _BASE_TEXTS:
        psi = encode_text_to_psi(text)
        f0, f1, f2, f3 = decode_psi_to_axes(psi)
        # Every axis must round-trip to ±1 (the encoder uses unit feature
        # values for the synthetic corpus).
        for f in (f0, f1, f2, f3):
            assert f in (Fraction(1), Fraction(-1)), f"axis feature out of range: {f}"


def test_contradiction_pair_polarity_inverted() -> None:
    """Every axis of the contradiction Ψ is the negation of the base's.

    This previously decoded ``pair.contradiction_psi`` into ``f_neg`` and
    then never used it, asserting instead that ``s2_nikhilam(base_psi)``
    flips axis 0 -- a property of S2, on an object the generator does not
    return. ``generate_contradiction_pair`` could have emitted anything at
    all as its contradiction and this test would still have passed.

    A comment justified the detour by saying the S5 centering step "kills
    the bit-0-uniform component" so the direct check was unavailable. It is
    not: ``_antipodal_psi`` is S5(S2(Ψ)), and decoding it gives exactly
    -f for all four axes on every base text. The check the name promises
    was available the whole time.
    """
    for text in _BASE_TEXTS:
        pair = generate_contradiction_pair(text)
        f_base = decode_psi_to_axes(pair.base_psi)
        f_neg = decode_psi_to_axes(pair.contradiction_psi)
        assert tuple(f_neg) == tuple(-f for f in f_base), (
            f"contradiction Ψ is not the antipode of the base for {text!r}: "
            f"base={tuple(str(f) for f in f_base)} "
            f"contradiction={tuple(str(f) for f in f_neg)}")


def test_paraphrase_pair_differs_on_targeted_axis_only() -> None:
    for text in _BASE_TEXTS:
        for axis in range(4):
            pair = generate_paraphrase_pair(text, axis)
            diffs = axis_difference(pair.psi_a, pair.psi_b)
            for k, d in enumerate(diffs):
                if k == axis:
                    assert d != Fraction(0), (
                        f"paraphrase pair text={text!r} axis={axis} produced "
                        f"zero differential on the targeted axis"
                    )
                else:
                    assert d == Fraction(0), (
                        f"paraphrase pair text={text!r} axis={axis} differs on "
                        f"non-targeted axis {k}: diff={d}"
                    )


def test_audit_closure_on_t29_multiple() -> None:
    # R2/R3/R4 vanish algebraically; R1 closes when trace is a multiple of 435.
    for text in _BASE_TEXTS:
        psi = encode_text_to_psi(text)
        result = audit_psi(psi, Fraction(435))
        assert result.closed, f"audit failed for trace=435 text={text!r}: {result.residuals}"


def test_audit_closure_off_grid_fails_on_r1() -> None:
    psi = encode_text_to_psi(_BASE_TEXTS[0])
    result = audit_psi(psi, Fraction(1))
    assert not result.closed
    assert result.residuals[0] == Fraction(1)
