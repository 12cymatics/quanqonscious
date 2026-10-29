from fractions import Fraction

import pytest
import sympy as sp

import vedic_book as b
from vedic_book_cases import EXAMPLE_PARAMETERS


@pytest.mark.parametrize("sid", range(1, 30))
def test_every_application_has_exact_output_and_primary_pagination(sid):
    result = b.evaluate(sid, **EXAMPLE_PARAMETERS[sid])
    assert result.sutra.id == sid
    assert result.sutra.pages
    assert result.spectral_coefficients
    assert all(isinstance(c, sp.Rational) for c in result.spectral_coefficients)
    assert not sp.sympify(result.value).atoms(sp.Float)


@pytest.mark.parametrize("d", [9, 19, 29, 49, 99, 109, 999, 1009])
def test_ekadhikena_reconstructs_the_complete_reciprocal(d):
    digits = b.ekadhikena(d).value
    numerator = 0
    for digit in digits:
        assert 0 <= digit <= 9
        numerator = 10 * numerator + digit
    assert Fraction(numerator, 10 ** len(digits) - 1) == Fraction(1, d)
    if d == 1009:
        assert len(digits) == 252


@pytest.mark.parametrize("a,bn,base", [(98, 97, 100), (103, 107, 100), (-2, 3, 10),
                                            (Fraction(2, 3), Fraction(9, 7), Fraction(5, 2))])
def test_deficiency_products_include_both_terms(a, bn, base):
    assert b.nikhilam(a, bn, base).value == a * bn
    assert b.anurupyena(a, bn, base, Fraction(3, 2)).value == a * bn
    assert b.yavadunam(a, base).value == a * a
    assert b.extended_square(a, base).value == a * a
    assert b.lessen(a, base).value == 2 * a - base


def test_speed_of_light_square_is_exact():
    result = b.yavadunam(299792458, 300000000)
    assert result.value == 89875517873681764
    assert result.spectral_coefficients == (207542, 89875474800000000, 43073681764)


def test_crosswise_keeps_all_diagonals_and_zero_coefficients():
    assert b.urdhva([3, 2, 1], [6, 5, 4]).value == (18, 27, 28, 13, 4)
    assert b.urdhva([1, 0, 0], [0, 2]).value == (0, 2, 0, 0)


@pytest.mark.parametrize("dividend,divisor", [([1, 2, 3, 4], [2, 1]), ([1, 0, 2], [3]),
                                             ([1], [1, 2]), ([1, 2, 1], [1, 1])])
def test_transposed_division_reconstructs_dividend(dividend, divisor):
    q, r = b.paravartya(dividend, divisor).value
    x = sp.Symbol("x")
    assert sp.expand(b.polynomial(divisor, x) * b.polynomial(q, x)
                     + b.polynomial(r, x) - b.polynomial(dividend, x)) == 0
    assert len(r) < len(divisor) or r == (0,)


def test_equal_product_equation_handles_identity_and_rejects_wrong_premise():
    assert b.shunyam(7, 9, 3, 21).value == sp.FiniteSet(0)
    assert b.shunyam(7, 9, 9, 7).value == sp.S.Complexes
    with pytest.raises(ValueError):
        b.shunyam(1, 2, 3, 5)


@pytest.mark.parametrize("args", [(6, 7, 8, 19, 14, 16), (7, 6, 8, 14, 19, 16)])
def test_proportional_system_solves_both_equations(args):
    a, c, e, d, f, g = args
    x, y = b.anurupye(*args).value
    assert a * x + c * y == e
    assert d * x + f * y == g


def test_symmetric_system():
    assert b.sankalana(45, -23, 113, -91).value == (2, -1)


@pytest.mark.parametrize("args", [(1, -5, 6), (3, 2, 7), (1, 2, 1)])
def test_completion_preserves_all_roots(args):
    a, c, d = args
    _, h, k, roots = b.purana(*args).value
    x = sp.Symbol("x")
    assert sp.expand(a * (x - h) ** 2 + k - (a * x ** 2 + c * x + d)) == 0
    for root in roots:
        assert sp.simplify(a * root ** 2 + c * root + d) == 0


def test_differentiation_and_full_factor_product_rule():
    factors = ([2], [1, 2, 3], [0, 0, 1], [3, 4])
    x = sp.Symbol("x")
    product = sp.prod(b.polynomial(f, x) for f in factors)
    result = b.gunakasamuccaya(factors)
    assert sp.expand(b.polynomial(result.value, x) - sp.diff(product, x)) == 0
    assert b.gunakasamuccaya([[2], [3]]).value == (0,)
    assert b.calana([5]).value == (0,)


def test_biquadratic_retains_complex_branches():
    roots = b.vyashti(7, 5, 706).value
    assert len(roots) == 4
    assert set(roots) == {-10, -2, -6 - sp.sqrt(22) * sp.I, -6 + sp.sqrt(22) * sp.I}
    for x in roots:
        assert sp.expand((x + 7) ** 4 + (x + 5) ** 4 - 706) == 0


@pytest.mark.parametrize("d", [3, 7, 11, 13, 17, 19, 27, 101, 1009])
def test_remainder_times_last_digit_matches_long_division(d):
    remainder, expected = 1, []
    while True:
        digit, remainder = divmod(10 * remainder, d)
        expected.append(digit)
        if remainder == 1:
            break
    assert b.sesanyankena(d).value == tuple(expected)


@pytest.mark.parametrize("first,difference", [([2, 1], [1, 0]), ([1, 2], [1, 1]),
                                              ([1, 1], [0, 0]), ([-7, -7], [3, 3]),
                                              ([0, 1], [0, 1]), ([1, 1], [1, 0])])
def test_sopantya_preserves_uncancelled_equation_domain(first, difference):
    x = sp.Symbol("x")
    a, d = b.polynomial(first, x), b.polynomial(difference, x)
    A, B, C, D = (sp.expand(a + k * d) for k in range(4))
    numerator = sp.expand(C * D + B * D - B * C - A * D)
    excluded = sp.Union(*(sp.solveset(f, x, sp.S.Complexes) for f in (A, B, C, D)))
    expected = sp.Complement(sp.solveset(numerator, x, sp.S.Complexes), excluded)
    assert b.sopantya(first, difference).value == expected


def test_sopantya_is_not_a_fibonacci_convergent():
    assert b.sopantya([2, 1], [1, 0]).value == sp.FiniteSet(sp.Rational(-13, 3))
    assert b.sopantya([1, 2], [1, 1]).value == sp.FiniteSet(-1, sp.Rational(-10, 13))


def test_pde_callers_keep_equation_solving_and_continued_fractions_distinct():
    from vedic_pde_solver import VedicExact, ExactConstants
    assert VedicExact.S13_sopantya([2, 1], [1, 0]).value == sp.FiniteSet(sp.Rational(-13, 3))
    assert VedicExact.continued_fraction_convergent([3, 7, 15, 1], 4) == (355, 113)
    assert ExactConstants.sqrt2(5) == Fraction(41, 29)
    assert ExactConstants.phi(5) == Fraction(8, 5)
    with pytest.raises(ValueError):
        VedicExact.continued_fraction_convergent([3, 7], 4)


def test_nines_and_coefficient_sum_identities():
    assert b.ekanyunena(123, 3).value == 122877
    factors = ([1, 2, 3], [4, -2], [0, 5])
    assert b.gunitasamuccaya(factors).value == 60
    assert b.samuccayagunitah(*factors[:2]).value == 12
    assert b.gunitasamuccaya_samuccayagunitah(factors).value == (60, 60)


def test_subsutra_book_cases():
    assert b.shishyate(65, [2, 4, 8, 16, 32, 64]).value == ((32, 1), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1))
    assert b.adyam([2, 5, 2], [2, 1]).value == (1, 2)
    for n in range(1, 7):
        digits = b.kevalaih(n).value
        block = sum(d * 10 ** (5 - i) for i, d in enumerate(digits))
        assert Fraction(block, 999999) == Fraction(n, 7)
    for d in (7, 11, 13, 19, 21, 23, 29, 31, 37):
        for n in (0, 1, 42, 343, 999999, d * 123457, d * 98765 + 1, 10 ** 15 + 37):
            for result in b.osculate(n, d)[2]:
                assert (10 * result - n) % d == 0
            assert b.vestanam(n, d).value[0] == (n % d == 0)
    assert b.antyayordasakepi(43, 47).value == 2021
    assert b.antyayordasakepi(884, 816).value == 721344
    assert b.antyayordasakepi(793, 797).value == 793 * 797
    assert b.tavadunam_cube(9989, 10000).value == 996703628669
    assert b.tavadunam_cube(9989, 10000).spectral_coefficients == (11, 9967, 363, -1331)
    for n in (91, 96, 104, 997, 1003):
        assert b.tavadunam_cube(n, 10 ** len(str(n - 1))).value == n ** 3
    assert b.vilokanam(Fraction(17, 4), 4).value == sp.FiniteSet(4, sp.Rational(1, 4))
    assert b.vilokanam(Fraction(10, 3), 3, [4, 1], [-4, 1]).value == sp.FiniteSet(8, -8)
    assert b.vilokanam(Fraction(3, 2), 2, sign=-1).value == sp.FiniteSet(2, sp.Rational(-1, 2))


def test_last_term_equation_excludes_denominator_roots():
    assert b.antyayoreva([1, 1], [3, 1], [0, 1], 1, 3).value == sp.FiniteSet(0)
    assert b.antyayoreva([0, 1], [0, 2], [1], 1, 3).value == sp.S.EmptySet
    assert b.antyayoreva([1, 1], [1, 1], [1], 1, 1).value == sp.Complement(sp.S.Complexes, sp.FiniteSet(-1, -2))


def test_elimination_and_retention_reconstructs_all_six_terms():
    x, y, z = sp.symbols("x y z")
    first, second = b.lopana(2, 6, 3, 7, 11, 7).value
    p = sum(c * v for c, v in zip(first, (x, y, z)))
    q = sum(c * v for c, v in zip(second, (x, y, z)))
    assert sp.expand(p * q) == 2 * x ** 2 + 6 * y ** 2 + 3 * z ** 2 + 7 * x * y + 11 * y * z + 7 * z * x


@pytest.mark.parametrize("value", [0.1, True, "1/3", complex(1)])
def test_no_inexact_or_coerced_kernel_operands(value):
    with pytest.raises(TypeError):
        b.rational(value)


@pytest.mark.parametrize("sid,parameters", [(1, {"denominator": 7}),
    (4, {"dividend": [1], "divisor": [0]}), (6, {"a": 1, "b": 2, "c": 3, "d": 2, "e": 4, "f": 6}),
    (12, {"denominator": 10}), (18, {"dividend": 65, "divisors": [2, 3]}),
    (19, {"quadratic": [2, 5, 2], "factor": [3, 1]}), (20, {"numerator": 7}),
    (24, {"a": 43, "b": 48}), (28, {"total": 3, "split": 2})])
def test_invalid_application_premises_are_visible(sid, parameters):
    with pytest.raises(ValueError):
        b.evaluate(sid, **parameters)
