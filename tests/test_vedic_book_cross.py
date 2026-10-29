import os
import sys
from fractions import Fraction as F

import pytest
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import true_sutras as T  # noqa: E402
import vedic_book as b  # noqa: E402


def q(value):
    value = sp.Rational(value)
    return F(int(value.p), int(value.q))


@pytest.mark.parametrize("d", [9, 19, 29, 49, 59, 99, 109, 239, 1009])
def test_s1_repetends_agree(d):
    assert b.ekadhikena(d).value == T.s1_recurring_decimal_division(1, d)
    assert b.ekadhikena(d).value == T.s1_recurring_decimal_multiplication(1, d)


@pytest.mark.parametrize("a,c,k", [(98, 97, 2), (103, 107, 2), (88, 112, 2), (9998, 9993, 4), (996, 1004, 3)])
def test_s2_s10_us7_products_agree(a, c, k):
    assert q(b.nikhilam(a, c, 10 ** k).value) == T.s2_multiply(a, c, k)['product']
    assert q(b.yavadunam(a, 10 ** k).value) == T.s10_square(a, k)['value']
    assert q(b.extended_square(a, 10 ** k).value) == T.us7_square(a, k)['value']


@pytest.mark.parametrize("a,c", [([1, 2, 3], [4, 5, 6]), ([9, 9], [9, 9]), ([1, 0, 1, 0], [2, 0, 3, 7])])
def test_s3_crosswise_columns_agree(a, c):
    assert tuple(q(v) for v in b.urdhva(a, c).value) == T.s3_columns(a, c)


@pytest.mark.parametrize("dividend,divisor", [((1, 0, -7, 6), (1, -2)), ((2, 3, -5, 1, 4), (1, 1, -3)),
                                              ((1, 2, 1), (1, 1))])
def test_s4_polynomial_divisions_agree(dividend, divisor):
    quotient, remainder = b.paravartya(tuple(reversed(dividend)), tuple(reversed(divisor))).value
    mine_q, mine_r = T.s4_polynomial_divide(dividend, divisor)
    assert tuple(q(v) for v in reversed(quotient)) == mine_q
    width = max(len(mine_r), len(remainder))
    assert (0,) * (width - len(remainder)) + tuple(q(v) for v in reversed(remainder)) == \
        (0,) * (width - len(mine_r)) + mine_r


def test_s5_s6_s7_equation_rules_agree():
    assert b.shunyam(7, 9, 3, 21).value == sp.FiniteSet(T.s5_independent_terms(7, 9, 3, 21))
    for a, c, p, d, e, f in [(6, 7, 8, 19, 14, 16), (12, 8, 7, 16, 16, 14), (499, 172, 212, 9779, 387, 477)]:
        assert tuple(q(v) for v in b.anurupye(a, c, p, d, e, f).value) == T.s6_solve(a, c, p, d, e, f)
    for a, c, p, d in [(45, -23, 113, -91), (3, 5, 13, 11), (7, 2, 16, 11)]:
        assert tuple(q(v) for v in b.sankalana(a, c, p, d).value) == T.s7_solve(a, c, p, c, a, d)


@pytest.mark.parametrize("a,c,e", [(1, -5, 6), (2, 5, 2), (4, -17, 4), (3, 0, -12)])
def test_s8_s9_completion_and_differential_agree(a, c, e):
    _, h, k, _ = b.purana(a, c, e).value
    mine_h, mine_k = T.s8_complete_square(a, c, e)
    assert (q(h), q(k)) == (-mine_h, mine_k)
    assert tuple(q(v) for v in b.calana([e, c, a]).value) == tuple(reversed(T.s9_differential(a, c, e)['first_differential']))


def test_s11_real_roots_agree():
    roots = {r for r in b.vyashti(7, 5, 706).value if r.is_real}
    assert {q(r) for r in roots} == set(T.s11_sum_of_fourth_powers(7, 5, 706)['roots'])


@pytest.mark.parametrize("d", [3, 7, 11, 13, 17, 19, 21, 23, 27, 29, 31, 37, 41, 49, 97, 101])
def test_s12_remainder_digits_agree(d):
    assert b.sesanyankena(d).value == T.s12_recurring_decimal(1, d)[1]


@pytest.mark.parametrize("first,step", [((2, 1), (1, 0)), ((3, 1), (1, 0)), ((1, 2), (1, 1)), ((5, -3), (2, 4))])
def test_s13_roots_agree(first, step):
    theirs = b.sopantya(list(first), list(step)).value
    mine = T.s13_solve((first[1], first[0]), (step[1], step[0]))
    assert mine in {q(r) for r in theirs}


@pytest.mark.parametrize("n,k", [(123, 3), (7, 2), (37, 1), (20, 1), (777, 3), (123456789, 4)])
def test_s14_nines_agree(n, k):
    assert q(b.ekanyunena(n, k).value) == T.s14_multiply_by_nines(n, k)['value']


def test_s15_s16_us10_us13_coefficient_rules_agree():
    factors = ([7, 1], [9, 1], [2, 3])
    descending = [tuple(reversed(f)) for f in factors]
    assert q(b.gunitasamuccaya(factors).value) == T.s15_product_of_sums(descending)
    assert q(b.samuccayagunitah(factors[0], factors[1]).value) == T.us10_sum_of_product(descending[:2])[0]
    both = b.gunitasamuccaya_samuccayagunitah(factors).value
    mine = T.us13_verify(descending)
    assert (q(both[0]), q(both[1])) == (mine[0], mine[1]) and mine[2]
    roots = [1, 2, 3, 4]
    first = b.gunakasamuccaya([[r, 1] for r in roots]).value
    assert tuple(q(v) for v in reversed(first)) == T.s16_differentials(roots)[0]


@pytest.mark.parametrize("a,c", [(49, 49), (46, 43), (52, 47), (41, 59)])
def test_us1_working_bases_agree(a, c):
    assert q(b.anurupyena(a, c, 100, F(1, 2)).value) == T.us1_multiply(a, c, 2, 2, 'submultiple')['value']


def test_us2_us4_us5_us6_us8_agree():
    assert b.shishyate(65, [2, 4, 8, 16, 32, 64]).value == T.us2_constant_remainder(65, [2, 4, 8, 16, 32, 64])
    for n in range(1, 7):
        assert b.kevalaih(n).value == T.us4_sevenths(n)
    for d in (7, 13, 19, 21, 37):
        for n in (0, 343, d * 98765, d * 98765 + 2, 10 ** 12 + 3):
            assert b.vestanam(n, d).value[0] == T.us5_divisible(n, d)[0]
    for n, k in [(9989, 4), (104, 2), (96, 2), (997, 3)]:
        assert q(b.tavadunam_cube(n, 10 ** k).value) == T.us6_cube(n, k)['value']
    for a, c in [(43, 47), (27, 23), (884, 816), (793, 797)]:
        assert q(b.antyayordasakepi(a, c).value) == T.us8_multiply(a, c)['value']


def test_us3_us9_us11_us12_agree():
    for quadratic, factor in [((2, 5, 2), (2, 1)), ((-3, 5, 2), (3, 1)), ((5, 7, 2), (1, 1))]:
        c0, c1, c2 = quadratic
        second = b.adyam(quadratic, factor).value
        first_mine, second_mine = T.us3_factor_quadratic(c2, c1, c0)
        x = sp.Symbol('x')
        theirs = sp.expand((factor[0] + factor[1] * x) * (second[0] + second[1] * x))
        mine = sp.expand((first_mine[0] * x + first_mine[1]) *
                         (sp.Rational(second_mine[0].numerator, second_mine[0].denominator) * x +
                          sp.Rational(second_mine[1].numerator, second_mine[1].denominator)))
        assert theirs == mine == c2 * x ** 2 + c1 * x + c0
    assert b.antyayoreva([1, 1], [3, 1], [0, 1], 1, 3).value == \
        sp.FiniteSet(T.us9_last_terms((1, 1, 1), (1, 3, 3), (1, 1), (1, 3)))
    first, second = b.lopana(2, 6, 3, 7, 11, 7).value
    left, right = T.us11_factor_homogeneous(
        {('x', 'x'): 2, ('y', 'y'): 6, ('z', 'z'): 3, ('x', 'y'): 7, ('y', 'z'): 11, ('x', 'z'): 7}, ['x', 'y', 'z'])
    xs = sp.symbols('x y z')
    theirs = sp.expand(sum(c * v for c, v in zip(first, xs)) * sum(c * v for c, v in zip(second, xs)))
    mine = sp.expand(sum(sp.Rational(c.numerator, c.denominator) * v for c, v in zip(left, xs)) *
                     sum(sp.Rational(c.numerator, c.denominator) * v for c, v in zip(right, xs)))
    assert theirs == mine
    for total, split in [(F(17, 4), 4), (F(10, 3), 3), (F(26, 5), 5)]:
        assert {q(v) for v in b.vilokanam(total, split).value} == set(T.us12_reciprocal(total))
