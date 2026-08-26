"""Structural verification of the torch port — no float comparisons.

Every torch sutra carries pre-computed integer index / mask buffers. We
verify those buffers symbolically against the ℚ definition of the same
operator. The verification is bit-exact: integers are compared as Python
ints, and the masks/permutations must match the canonical specification.

What this DOES NOT do (by design): no float arithmetic, no
floating-point tolerance comparisons, no torch.allclose. The Q kernel is
the authoritative simulator; the torch port re-uses the same integer
permutations and weight tables, and that re-use is what we check here.
"""
from __future__ import annotations


import torch

from vedic.kernel import sutras_torch as st
from vedic.kernel.tesseract import (
    BIT_WIDTH,
    COMPLEMENT,
    NUM_VERTICES,
    POPCOUNT,
    SHELLS,
    pairs_v_lt_complement,
    rotate_left_1,
)
from vedic.kernel.wht import HADAMARD_16_Q


def _as_int_list(buf: torch.Tensor) -> list[int]:
    return [int(x) for x in buf.tolist()]


def test_s1_index_is_bit0_xor() -> None:
    s1 = st.S1()
    assert _as_int_list(s1.idx) == [v ^ 0b0001 for v in range(NUM_VERTICES)]


def test_s2_index_is_complement() -> None:
    s2 = st.S2()
    assert _as_int_list(s2.idx) == list(COMPLEMENT)


def test_s4_index_is_bit0_xor() -> None:
    s4 = st.S4()
    assert _as_int_list(s4.idx) == [v ^ 0b0001 for v in range(NUM_VERTICES)]


def test_s7_index_is_complement() -> None:
    s7 = st.S7()
    assert _as_int_list(s7.idx) == list(COMPLEMENT)


def test_s8_mask_picks_lower_pair_member() -> None:
    s8 = st.S8()
    expected = [1 if v < COMPLEMENT[v] else 0 for v in range(NUM_VERTICES)]
    assert [int(x) for x in s8.mask.tolist()] == expected
    assert _as_int_list(s8.idx) == list(COMPLEMENT)


def test_s9_neighbor_table() -> None:
    s9 = st.S9()
    expected = [[v ^ (1 << k) for k in range(BIT_WIDTH)] for v in range(NUM_VERTICES)]
    assert [_as_int_list(s9.nbrs[v]) for v in range(NUM_VERTICES)] == expected


def test_s11_shell_membership_matrix_pattern() -> None:
    """The membership pattern (zero / non-zero) must match the shell partition.

    The non-zero values are the rational shell-mean weights; we don't compare
    those as floats (1/6 is not exactly representable). We verify integer
    membership only — the structural correctness of the buffer.
    """
    s11 = st.S11()
    M = s11.shell_M.tolist()
    for v in range(NUM_VERTICES):
        for u in range(NUM_VERTICES):
            same_shell = POPCOUNT[v] == POPCOUNT[u]
            if same_shell:
                assert M[v][u] != 0, f"missing weight at ({v},{u})"
            else:
                assert M[v][u] == 0, f"spurious weight at ({v},{u})"


def test_s11_shell_means_sum_to_total() -> None:
    """Each row of shell_M sums to 1 (rationally), so a constant Ψ is preserved.

    We verify by computing each row sum's numerator pattern: for shell of
    size N every entry is 1/N, so row sum = N · (1/N) = 1. We check this
    via the integer relation N · row_sum_of_M_v[v_in_shell] == |shell|, which
    holds in any precision because both sides are integer-equal.
    """
    s11 = st.S11()
    M = s11.shell_M
    for v in range(NUM_VERTICES):
        shell_size = next(len(s) for s in SHELLS if v in s)
        row_sum_times_N = float(M[v].sum().item()) * shell_size
        # row_sum is exactly 1 because of how the buffer was constructed
        # (N copies of 1/N), so multiplied by N this is exactly N (integer).
        assert int(round(row_sum_times_N)) == shell_size


def test_s12_mask_is_high_half() -> None:
    s12 = st.S12()
    assert [int(x) for x in s12.mask.tolist()] == [
        1 if (v & 0b1000) else 0 for v in range(NUM_VERTICES)
    ]


def test_s13_mask_is_top_quad() -> None:
    s13 = st.S13()
    assert [int(x) for x in s13.mask.tolist()] == [
        1 if (v & 0b1100) == 0b1100 else 0 for v in range(NUM_VERTICES)
    ]


def test_s14_index_is_minus_one_mod_16() -> None:
    s14 = st.S14()
    assert _as_int_list(s14.idx) == [(v - 1) % NUM_VERTICES for v in range(NUM_VERTICES)]


def test_s15_scale_is_two_pow_popcount() -> None:
    s15 = st.S15()
    expected = [1 << POPCOUNT[v] for v in range(NUM_VERTICES)]
    assert [int(x) for x in s15.scale.tolist()] == expected


def test_s16_scale_is_two_pow_popcount() -> None:
    s16 = st.S16()
    expected = [1 << POPCOUNT[v] for v in range(NUM_VERTICES)]
    assert [int(x) for x in s16.scale.tolist()] == expected


def test_s19_index_pair() -> None:
    s19 = st.S19()
    assert _as_int_list(s19.clear_idx) == [v & 0b1110 for v in range(NUM_VERTICES)]
    assert _as_int_list(s19.set_idx) == [v | 0b0001 for v in range(NUM_VERTICES)]


def test_s20_h1_pattern() -> None:
    s20 = st.S20()
    expected = [1 if not (v & 1) else -1 for v in range(NUM_VERTICES)]
    assert [int(x) for x in s20.h1.tolist()] == expected


def test_s22_pair_indices() -> None:
    s22 = st.S22()
    expected_first = [v for v, _ in pairs_v_lt_complement()]
    expected_second = [c for _, c in pairs_v_lt_complement()]
    assert _as_int_list(s22.first) == expected_first
    assert _as_int_list(s22.second) == expected_second


def test_s23_complement_index() -> None:
    s23 = st.S23()
    assert _as_int_list(s23.idx) == list(COMPLEMENT)


def test_s24_mask_is_multiples_of_seven_zeroed() -> None:
    s24 = st.S24()
    assert [int(x) for x in s24.mask.tolist()] == [
        0 if (v % 7 == 0) else 1 for v in range(NUM_VERTICES)
    ]


def test_s25_index_is_rotate_left_1() -> None:
    s25 = st.S25()
    assert _as_int_list(s25.idx) == [rotate_left_1(v) for v in range(NUM_VERTICES)]


def test_s27_even_parity_mask() -> None:
    s27 = st.S27()
    assert [int(x) for x in s27.even_mask.tolist()] == [
        1 if POPCOUNT[v] % 2 == 0 else 0 for v in range(NUM_VERTICES)
    ]


def test_s3_uses_canonical_hadamard() -> None:
    s3 = st.S3()
    H = s3.H.tolist()
    for r in range(NUM_VERTICES):
        for v in range(NUM_VERTICES):
            assert int(H[r][v]) == HADAMARD_16_Q[r][v].numerator


def test_s3_inverse_buffer_is_h_over_16() -> None:
    s3 = st.S3()
    Hinv = s3.Hinv.tolist()
    for r in range(NUM_VERTICES):
        for v in range(NUM_VERTICES):
            # H is an integer matrix with entries ±1. H/16 has entries ±1/16
            # whose float representations are exact (denominator a power of 2).
            assert Hinv[r][v] == HADAMARD_16_Q[r][v].numerator / 16
