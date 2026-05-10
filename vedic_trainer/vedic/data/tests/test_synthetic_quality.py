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
    for text in _BASE_TEXTS:
        pair = generate_contradiction_pair(text)
        f_base = decode_psi_to_axes(pair.base_psi)
        f_neg = decode_psi_to_axes(pair.contradiction_psi)
        # The contradiction Ψ has been centered (S5), which kills the
        # bit-0-uniform component of the encoded representation. We check
        # that the polarity axis itself flipped sign on the *raw* S2 result.
        # Specifically: axis 0 of S2(Ψ) equals −axis 0 of Ψ.
        from vedic.kernel.sutras_exact import s2_nikhilam
        s2_psi = s2_nikhilam(pair.base_psi)
        f_s2 = decode_psi_to_axes(s2_psi)
        assert f_s2[0] == -f_base[0], "polarity axis did not flip under S2"


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
