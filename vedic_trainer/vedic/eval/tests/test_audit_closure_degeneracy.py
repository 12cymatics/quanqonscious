"""The audit-closure metric does not depend on the text it is given.

This is not a bug report against the encoder; it is a property of the four
residuals. R2/R3/R4 are algebraic identities that vanish on every input, so
closure reduces to R1, which depends only on the trace counter. These tests
pin that down so the metric cannot quietly be reported as a model score.
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from vedic.data.audit_filter import audit_psi
from vedic.data.tesseract_encode import encode_text_to_psi
from vedic.eval.compositional_audit import (
    audit_closure_rate,
    score_audit_psi_batch,
)

ENGLISH = [f"the {w} is not moving north" for w in
           "cat dog tree river stone bird wind cloud".split()] * 60
NOISE = ["".join(random.Random(i).choices("qxzjkvw ", k=40)) for i in range(480)]


def test_two_unrelated_corpora_give_identical_closure_flags():
    assert score_audit_psi_batch(ENGLISH) == score_audit_psi_batch(NOISE)


def test_two_unrelated_corpora_give_the_identical_rate():
    assert audit_closure_rate(ENGLISH) == audit_closure_rate(NOISE)


def test_audit_closure_is_index_dependent_only():
    """At a fixed trace index, every text gives the same verdict."""
    verdicts = {audit_psi(encode_text_to_psi(t), Fraction(7)).closed
                for t in ENGLISH + NOISE}
    assert len(verdicts) == 1, (
        "closure now varies with the text — the metric has become "
        "discriminative, and compositional_audit's docstring and the "
        "README's falsification criterion 2 must be revisited")


def test_only_r1_is_ever_nonzero():
    """R2/R3/R4 are identities; only R1 can move."""
    nonzero = {"R1": 0, "R2": 0, "R3": 0, "R4": 0}
    # Both corpora in full. The claim is that only R1 can ever move; a
    # slice tests it on the first 200 English strings and says nothing about
    # the rest or about the noise corpus, which is where a text-dependent
    # residual would most plausibly show up.
    for i, text in enumerate(ENGLISH + NOISE):
        for key, val in zip(nonzero, audit_psi(encode_text_to_psi(text),
                                               Fraction(i)).residuals):
            if val != 0:
                nonzero[key] += 1
    assert nonzero["R2"] == nonzero["R3"] == nonzero["R4"] == 0
    assert nonzero["R1"] > 0, "R1 never moved either — nothing constrains anything"


def test_closure_tracks_multiples_of_435():
    """R1 closes exactly on multiples of T(29) = 435, and only there."""
    flags = score_audit_psi_batch(["x"] * 900)
    assert [i for i, f in enumerate(flags) if f] == [0, 435, 870]


def test_an_empty_corpus_has_no_rate():
    with pytest.raises(ValueError):
        audit_closure_rate([])
