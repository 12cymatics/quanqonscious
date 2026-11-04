import os
import sys
from fractions import Fraction as F
from itertools import product

import pytest
import sympy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import true_sutras as T  # noqa: E402

X = sympy.Symbol('x')


def repetend(n, d):
    period = 1
    while (n * pow(10, period, d)) % d != n % d:
        period += 1
    return tuple(int(ch) for ch in str(n * (10 ** period - 1) // d).zfill(period))


def poly(coeffs):
    return sum(sympy.Rational(c.numerator, c.denominator) * X ** (len(coeffs) - 1 - i)
               for i, c in enumerate(map(F, coeffs)))


def test_s1_book_example_one_nineteenth():
    assert T.s1_recurring_decimal_division(1, 19) == tuple(int(c) for c in '052631578947368421')
    assert T.s1_recurring_decimal_multiplication(1, 19) == tuple(int(c) for c in '052631578947368421')


@pytest.mark.parametrize('d', [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 199, 239, 999, 1009])
def test_s1_both_methods_match_long_division(d):
    for n in sorted({1, 2, 7, d // 2, d - 1}):
        if 0 < n < d:
            want = repetend(n, d)
            assert T.s1_recurring_decimal_division(n, d) == want
            assert T.s1_recurring_decimal_multiplication(n, d) == want


def test_s1_refuses_outside_its_domain():
    for args in [(1, 17), (19, 19), (0, 19), (20, 19)]:
        with pytest.raises(ValueError):
            T.s1_recurring_decimal_division(*args)


def test_s1_squares_ending_in_5():
    assert T.s1_square_ending_in_5(15) == (2, 25, 225)
    assert T.s1_square_ending_in_5(195) == (380, 25, 38025)
    for n in range(5, 100006, 10):
        assert T.s1_square_ending_in_5(n)[2] == n * n
    with pytest.raises(ValueError):
        T.s1_square_ending_in_5(14)


def test_s2_complement_is_all_from_nine_last_from_ten():
    assert T.s2_complement(3270, 4) == 6730
    assert T.s2_complement(1) == 9
    for k in (1, 2, 3, 4):
        for n in range(1, 10 ** k, max(1, 10 ** k // 97)):
            assert T.s2_complement(n, k) == 10 ** k - n
    for bad in [(0, 2), (100, 2)]:
        with pytest.raises(ValueError):
            T.s2_complement(*bad)


def test_s2_book_example_and_products_across_the_base():
    r = T.s2_multiply(98, 97)
    assert (r['left'], r['right'], r['product']) == (95, 6, 9506)
    for a, b in product(range(85, 116), range(85, 116)):
        assert T.s2_multiply(a, b)['product'] == a * b
    for a, b in [(9998, 9993), (10007, 10012), (996, 1004), (100, 100), (1, 1), (12345, 9)]:
        assert T.s2_multiply(a, b)['product'] == a * b


def test_s2_division_matches_divmod():
    for divisor in [7, 8, 9, 88, 89, 97, 98, 99, 897, 989, 9989, 11, 50]:
        for dividend in list(range(0, 400)) + [1234, 13579, 1011, 98765432, 10 ** 12 + 7]:
            assert T.s2_divide(dividend, divisor) == divmod(dividend, divisor)


def test_s3_book_columns_and_products():
    columns, value = T.s3_multiply(123, 456)
    assert columns == (4, 13, 28, 27, 18)
    assert value == 56088
    for a, b in [(0, 0), (9, 9), (99, 99), (123456789, 987654321), (10 ** 20 + 1, 37)]:
        assert T.s3_multiply(a, b)[1] == a * b


def test_s3_polynomials_and_duplex_squaring():
    p, q = (F(1), F(7)), (F(1), F(9))
    assert sympy.expand(poly(T.s3_polynomial_multiply(p, q)) - poly(p) * poly(q)) == 0
    assert T.s3_duplex([1, 2, 3]) == 2 * 1 * 3 + 2 * 2
    assert T.s3_duplex([1, 2, 3, 4]) == 2 * 1 * 4 + 2 * 2 * 3
    for n in [0, 7, 12, 99, 1234, 98765, 10 ** 15 + 3]:
        assert T.s3_square_by_duplex(n)[1] == n * n


def test_s4_division_matches_divmod():
    for divisor in [10, 11, 12, 13, 19, 112, 123, 1123, 1009, 1999]:
        for dividend in list(range(0, 300)) + [1234, 13579, 98765432, 10 ** 12 + 7]:
            assert T.s4_divide(dividend, divisor) == divmod(dividend, divisor)
    for bad in [(100, 23), (100, 9)]:
        with pytest.raises(ValueError):
            T.s4_divide(*bad)


def test_s4_polynomial_and_equation_rules():
    q, r = T.s4_polynomial_divide((1, 0, -7, 6), (1, -2))
    assert sympy.expand(poly(q) * (X - 2) + poly(r) - (X ** 3 - 7 * X + 6)) == 0
    assert T.s4_solve_linear(3, 7, 1, 13) == 3
    assert T.s4_solve_simultaneous(2, 3, 8, 4, 5, 14) == (1, 2)
    for a, b, p, c, d, q in [(5, 2, 42, 3, -4, 2), (11, 6, 28, 7, -4, 10), (3, 7, 1, 5, 2, 9)]:
        x, y = T.s4_solve_simultaneous(a, b, p, c, d, q)
        assert a * x + b * y == p and c * x + d * y == q
    x = T.s4_cyclic_fraction_root(1, 2, 3, -1, -2, -3)
    assert F(1) / ((x - 1) * (x - 2)) + F(2) / ((x - 2) * (x - 3)) + F(3) / ((x - 3) * (x - 1)) == 0


def test_s5_six_meanings_against_substitution():
    assert T.s5_common_factor(9, 7, 1) == -1
    assert T.s5_independent_terms(7, 9, 3, 21) == 0
    x = T.s5_equal_numerators(2, -1, 3, -1)
    assert 1 / (2 * x - 1) + 1 / (3 * x - 1) == 0
    assert T.s5_fraction_equation_roots((2, 9), (2, 7), (2, 7), (2, 9)) == (-4,)
    roots = T.s5_fraction_equation_roots((3, 4), (6, 7), (5, 6), (2, 3))
    assert set(roots) == {F(-5, 4), F(-1)}
    for x in roots:
        assert (3 * x + 4) / (6 * x + 7) == (5 * x + 6) / (2 * x + 3)
    x = T.s5_reciprocal_sums([(1, 1, -7), (1, 1, -9)], [(1, 1, -6), (1, 1, -10)])
    assert x == 8 and 1 / (x - 7) + 1 / (x - 9) == 1 / (x - 6) + 1 / (x - 10)
    assert T.s5_cubic_binomials(-3, -9) == 6
    x = T.s5_cubic_binomials(-140, -60)
    assert (x - 140) ** 3 + (x - 60) ** 3 == 2 * (x - 100) ** 3


def test_s5_refuses_when_the_samuccaya_differs():
    with pytest.raises(ValueError):
        T.s5_independent_terms(7, 9, 3, 20)
    with pytest.raises(ValueError):
        T.s5_reciprocal_sums([(1, 1, -7), (1, 1, -9)], [(1, 1, -6), (1, 1, -11)])
    with pytest.raises(ValueError):
        T.s5_fraction_equation_roots((1, 1), (1, 2), (1, 3), (1, 5))


def test_s6_ratio_rule():
    assert T.s6_solve(6, 7, 8, 19, 14, 16) == (0, F(8, 7))
    assert T.s6_solve(12, 8, 7, 16, 16, 14) == (0, F(7, 8))
    x, y = T.s6_solve(499, 172, 212, 9779, 387, 477)
    assert (x, y) == (0, F(212, 172)) and 499 * x + 172 * y == 212 and 9779 * x + 387 * y == 477
    with pytest.raises(ValueError):
        T.s6_solve(2, 3, 8, 4, 5, 14)
    assert T.s6_solve_system([[1, 2, 4], [2, 4, 1], [4, 1, 2]], [1, 2, 4]) == (1, 0, 0)
    assert T.s6_solve_system([[3, 5, 7], [1, 2, 2], [2, 9, 3]], [14, 4, 6]) == (0, 0, 2)
    with pytest.raises(ValueError):
        T.s6_solve_system([[1, 2, 3], [4, 4, 6], [7, 6, 9]], [6, 12, 18])
    with pytest.raises(ValueError):
        T.s6_solve(1, 2, 3, 2, 4, 6)


def test_s7_addition_and_subtraction():
    assert T.s7_solve(45, -23, 113, 23, -45, 91) == (2, -1)
    assert T.s7_solve(3, 5, 13, 5, 3, 11) == (1, 2)
    with pytest.raises(ValueError):
        T.s7_solve(3, 5, 13, 4, 3, 11)


def test_s8_completion():
    assert T.s8_solve_cubic(6, 11, 6) == (-3, -2, -1)
    h, k = T.s8_complete_square(2, 5, 2)
    assert sympy.expand(2 * (X + sympy.Rational(h.numerator, h.denominator)) ** 2
                        + sympy.Rational(k.numerator, k.denominator) - (2 * X ** 2 + 5 * X + 2)) == 0
    shift, big_p, big_q = T.s8_complete_cube(-6, 11, -6)
    y = sympy.Symbol('y')
    depressed = sympy.expand((y - sympy.Rational(shift.numerator, shift.denominator)) ** 3
                             - 6 * (y - sympy.Rational(shift.numerator, shift.denominator)) ** 2
                             + 11 * (y - sympy.Rational(shift.numerator, shift.denominator)) - 6)
    assert depressed == sympy.expand(y ** 3 + sympy.Rational(big_p.numerator, big_p.denominator) * y
                                     + sympy.Rational(big_q.numerator, big_q.denominator))
    with pytest.raises(ValueError):
        T.s8_solve_cubic(0, 0, -2)


def test_s9_first_differential_is_root_of_discriminant():
    r = T.s9_differential(4, -17, 4)
    assert r['roots'] == (F(1, 4), 4) and r['discriminant'] == 225
    for x in r['roots']:
        assert (2 * 4 * x - 17) ** 2 == r['discriminant']
    r = T.s9_differential(1, 0, -2)
    assert r['roots'] is None and r['centre'] == 0 and r['radicand'] == 2


def test_s10_squares_and_book_cubes():
    assert T.s10_square(97)['value'] == 9409
    assert T.s10_cube(104)['portions'] == (112, 48, 64) and T.s10_cube(104)['value'] == 1124864
    for n in list(range(80, 125)) + [989, 996, 1007, 9997, 10003, 99999]:
        assert T.s10_square(n)['value'] == n * n
        assert T.s10_cube(n)['value'] == n ** 3


def test_s11_average_binomial():
    r = T.s11_sum_of_fourth_powers(7, 5, 706)
    assert r['roots'] == (-10, -2) and r['nonreal_a_squared'] == (-22,)
    for x in r['roots']:
        assert (x + 7) ** 4 + (x + 5) ** 4 == 706


@pytest.mark.parametrize('d', [3, 7, 11, 13, 17, 19, 21, 23, 27, 29, 31, 37, 41, 43, 47, 49, 51, 97, 101, 103, 107])
def test_s12_remainders_by_the_last_digit(d):
    for n in sorted({1, 2, d - 1}):
        remainders, digits = T.s12_recurring_decimal(n, d)
        assert digits == repetend(n, d)
        assert remainders[-1] == n


def test_s12_book_examples():
    assert T.s12_recurring_decimal(1, 7) == ((3, 2, 6, 4, 5, 1), (1, 4, 2, 8, 5, 7))
    assert T.s12_recurring_decimal(1, 13)[1] == (0, 7, 6, 9, 2, 3)
    assert T.s12_last_digit_multiplier(7) == 7 and T.s12_last_digit_multiplier(13) == 3
    with pytest.raises(ValueError):
        T.s12_recurring_decimal(1, 14)


def test_s13_ultimate_and_twice_the_penultimate():
    x = T.s13_solve((1, 3), (0, 1))
    assert x == F(-16, 3)
    a, b, c, d = x + 3, x + 4, x + 5, x + 6
    assert 1 / (a * b) + 1 / (a * c) == 1 / (a * d) + 1 / (b * c)
    x = T.s13_solve_binomials((2, 1), (3, 2), (4, 3), (5, 4))
    assert x == F(-10, 13)
    a, b, c, d = 2 * x + 1, 3 * x + 2, 4 * x + 3, 5 * x + 4
    assert 1 / (a * b) + 1 / (a * c) == 1 / (a * d) + 1 / (b * c)
    with pytest.raises(ValueError):
        T.s13_solve_binomials((1, 1), (1, 2), (1, 4), (1, 5))


def test_s14_three_cases():
    assert T.s14_multiply_by_nines(777, 3)['value'] == 777 * 999
    assert T.s14_multiply_by_nines(7, 2)['value'] == 693
    assert T.s14_multiply_by_nines(37, 1)['value'] == 333
    for k in (1, 2, 3, 4):
        for n in list(range(1, 300)) + [10 ** k, 10 ** k - 1, 10 ** k + 1, 123456789]:
            r = T.s14_multiply_by_nines(n, k)
            assert r['value'] == n * (10 ** k - 1)


def test_s15_us10_us13_sum_of_coefficients():
    factors = [(1, 1), (1, 2), (1, 3)]
    assert T.s15_product_of_sums(factors) == 24
    assert T.us10_sum_of_product(factors)[0] == 24
    assert T.us13_verify(factors) == (24, 24, True)
    assert T.s15_check([(1, 7), (1, 9)], (1, 16, 63))
    assert not T.s15_check([(1, 7), (1, 9)], (1, 16, 64))


def test_s16_differentials_are_sums_of_factor_products():
    for roots in ([1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [F(1, 2), -3, 7]):
        e = sympy.prod([X + sympy.Rational(F(r).numerator, F(r).denominator) for r in roots])
        for j, d in enumerate(T.s16_differentials(roots), start=1):
            assert sympy.expand(poly(d) - sympy.diff(e, X, j)) == 0


def test_us1_proportionate_working_bases():
    assert T.us1_multiply(49, 49, 2, 2, 'submultiple')['value'] == 2401
    assert T.us1_multiply(29, 29, 1, 3, 'multiple')['value'] == 841
    for a, b in product(range(40, 61), range(40, 61)):
        assert T.us1_multiply(a, b, 2, 2, 'submultiple')['value'] == a * b
    for a, b in product(range(25, 36), range(25, 36)):
        assert T.us1_multiply(a, b, 1, 3, 'multiple')['value'] == a * b


def test_us3_first_by_first_last_by_last():
    book = [(2, 5, 2), (2, 5, -3), (2, 7, 5), (2, 9, 10), (2, -5, -3), (3, 13, 4), (3, -7, 2), (4, 12, 5),
            (6, 11, 3), (6, 11, -10), (6, 13, 6), (6, -13, -19), (6, 37, 6), (7, -6, -1), (8, -22, 5),
            (9, -15, 4), (12, 13, -4)]
    for a, b, c in book:
        first, second = T.us3_factor_quadratic(a, b, c)
        f = (first[0] * X + first[1]) * (sympy.Rational(second[0].numerator, second[0].denominator) * X
                                          + sympy.Rational(second[1].numerator, second[1].denominator))
        assert sympy.expand(f - (a * X ** 2 + b * X + c)) == 0
    with pytest.raises(ValueError):
        T.us3_factor_quadratic(1, 1, 1)


def test_us2_remainder_remains_constant():
    assert T.us2_constant_remainder(65, [2, 4, 8, 16, 32, 64]) == ((32, 1), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1))
    for n, ds in [(97, [3, 6, 12, 24, 48]), (1001, [10, 20, 40, 100, 200])]:
        assert T.us2_constant_remainder(n, ds) == tuple(divmod(n, d) for d in ds)
    with pytest.raises(ValueError):
        T.us2_constant_remainder(65, [2, 3])


def test_us4_for_seven_the_multiplicand_is_143():
    for n in range(1, 7):
        assert T.us4_sevenths(n) == repetend(n, 7)


@pytest.mark.parametrize('d', [7, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 49, 51, 53, 59])
def test_us5_osculation_decides_divisibility(d):
    pos = T.us5_positive_osculator(d)
    neg = T.us5_negative_osculator(d)
    assert (10 * pos) % d == 1 and (10 * neg) % d == d - 1
    for n in list(range(0, 600)) + [d * 123457, d * 98765 + 1, 10 ** 15 + 37]:
        assert T.us5_divisible(n, d, 'positive')[0] == (n % d == 0)
        assert T.us5_divisible(n, d, 'negative')[0] == (n % d == 0)


def test_us6_book_cube_by_deficiency():
    r = T.us6_cube(9989)
    assert r['portions'] == (9967, 363, -1331) and r['normalised'] == (9967, 362, 8669)
    assert r['value'] == 9989 ** 3 == 996703628669
    for n in list(range(80, 125)) + [989, 996, 1004, 9997, 10003, 99999]:
        assert T.us6_cube(n)['value'] == n ** 3


def test_us7_book_squares():
    book = {9: 81, 8: 64, 97: 9409, 92: 8464, 108: 11664, 103: 10609, 989: 978121, 988: 976144,
            993: 986049, 9989: 99780121, 9984: 99680256, 9993: 99860049}
    for n, sq in book.items():
        assert T.us7_square(n)['value'] == sq == n * n
    assert T.us7_square(775, 2, 8, 'multiple')['value'] == 775 ** 2


def test_us8_last_digits_total_ten():
    assert T.us8_multiply(27, 23)['value'] == 621
    assert T.us8_multiply(884, 816)['value'] == 721344
    assert T.us8_multiply(793, 797)['value'] == 793 * 797
    for prev in range(0, 30):
        for b in range(1, 10):
            a1, a2 = 10 * prev + b, 10 * prev + (10 - b)
            assert T.us8_multiply(a1, a2)['value'] == a1 * a2
    with pytest.raises(ValueError):
        T.us8_multiply(27, 24)


def test_us9_only_the_last_terms():
    x = T.us9_last_terms((1, 1, 1), (1, 3, 3), (1, 1), (1, 3))
    assert x == 0 and (x * x + x + 1) / (x * x + 3 * x + 3) == (x + 1) / (x + 3)
    x = T.us9_last_terms((1, 1, 2), (1, 3, 5), (1, 1), (1, 3))
    assert x == F(1, 3) and (x * x + x + 2) / (x * x + 3 * x + 5) == (x + 1) / (x + 3)
    x = T.us9_last_terms((2, 7, 3), (2, 11, 7), (2, 3), (2, 7))
    assert (2 * x * x + 7 * x + 3) / (2 * x * x + 11 * x + 7) == (2 * x + 3) / (2 * x + 7)
    num, den = T.us9_series_sum([(1, 1), (1, 2), (1, 3), (1, 4)])
    lhs = sum(1 / ((X + i) * (X + i + 1)) for i in range(1, 4))
    assert sympy.simplify(lhs - poly(num) / poly(den)) == 0
    num, den = T.us9_series_sum([(1, 1), (5, 1), (7, 1)], kind='difference')
    lhs = 4 * X / ((X + 1) * (5 * X + 1)) + 2 * X / ((5 * X + 1) * (7 * X + 1))
    assert sympy.simplify(lhs - poly(num) / poly(den)) == 0


def test_us11_alternate_elimination_and_retention():
    left, right = T.us11_factor_homogeneous(
        {('x', 'x'): 2, ('y', 'y'): 6, ('z', 'z'): 3, ('x', 'y'): 7, ('y', 'z'): 11, ('x', 'z'): 7}, ['x', 'y', 'z'])
    xs = sympy.symbols('x y z')
    e = 2 * xs[0] ** 2 + 6 * xs[1] ** 2 + 3 * xs[2] ** 2 + 7 * xs[0] * xs[1] + 11 * xs[1] * xs[2] + 7 * xs[0] * xs[2]
    f = sum(sympy.Rational(c.numerator, c.denominator) * v for c, v in zip(left, xs)) * \
        sum(sympy.Rational(c.numerator, c.denominator) * v for c, v in zip(right, xs))
    assert sympy.expand(f - e) == 0
    left, right = T.us11_factor_homogeneous(
        {('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): 5, ('x', 'w'): 2, ('y', 'w'): -5, ('w', 'w'): -12}, ['x', 'y', 'w'])
    xs = sympy.symbols('x y w')
    e = 2 * xs[0] ** 2 + 2 * xs[1] ** 2 + 5 * xs[0] * xs[1] + 2 * xs[0] * xs[2] - 5 * xs[1] * xs[2] - 12 * xs[2] ** 2
    f = sum(sympy.Rational(c.numerator, c.denominator) * v for c, v in zip(left, xs)) * \
        sum(sympy.Rational(c.numerator, c.denominator) * v for c, v in zip(right, xs))
    assert sympy.expand(f - e) == 0
    xs = sympy.symbols('x y z')
    e = 3 * xs[0] ** 2 - 4 * xs[0] * xs[1] - xs[0] * xs[2] + xs[1] ** 2 - xs[1] * xs[2] - 2 * xs[2] ** 2
    left, right = T.us11_factor_homogeneous(
        {('x', 'x'): 3, ('x', 'y'): -4, ('x', 'z'): -1, ('y', 'y'): 1, ('y', 'z'): -1, ('z', 'z'): -2},
        ['x', 'y', 'z'])
    f = sum(sympy.Rational(c.numerator, c.denominator) * v for c, v in zip(left, xs)) * \
        sum(sympy.Rational(c.numerator, c.denominator) * v for c, v in zip(right, xs))
    assert sympy.expand(f - e) == 0
    with pytest.raises(ValueError):
        T.us11_factor_homogeneous({('x', 'x'): 1, ('y', 'y'): -1, ('z', 'z'): -1}, ['x', 'y', 'z'])


def test_us12_mere_observation():
    assert T.us12_reciprocal(F(17, 4)) == (F(1, 4), 4)
    assert T.us12_reciprocal(F(-17, 4)) == (-4, F(-1, 4))
    for a in [F(2), F(3, 5), F(7, 2), F(11, 13)]:
        assert set(T.us12_reciprocal(a + 1 / a)) == {a, 1 / a}
        assert set(T.us12_reciprocal(a - 1 / a, sign=-1)) == {a, -1 / a}
    with pytest.raises(ValueError):
        T.us12_reciprocal(F(3))
