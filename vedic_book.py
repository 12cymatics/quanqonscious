from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Tuple

import sympy as sp


BOOK = {
    "author": "Bharati Krishna Tirthaji",
    "title": "Vedic Mathematics",
    "edition": "1992 revised edition, 2010/2012 reprint",
    "url": "https://archive.org/details/vedic-mathematics-bharati-krishna-tirth-ji-maharaj",
    "pagination": "printed",
}


@dataclass(frozen=True)
class Sutra:
    id: int
    name: str
    application: str
    pages: Tuple[int, ...]


SUTRAS = (
    Sutra(1, "Ekadhikena Purvena", "nine_ender_repetend", (2, 3, 4)),
    Sutra(2, "Nikhilam Navatascaramam Dasatah", "near_base_product", (12, 13)),
    Sutra(3, "Urdhva Tiryagbhyam", "crosswise_coefficients", (33, 34, 35, 36)),
    Sutra(4, "Paravartya Yojayet", "transposed_polynomial_division", (53, 54, 55, 56, 57, 58)),
    Sutra(5, "Sunyam Samyasamuccaye", "equal_product_equation", (98,)),
    Sutra(6, "Anurupye Sunyamanyat", "proportional_rhs_system", (130, 131, 132)),
    Sutra(7, "Sankalana Vyavakalanabhyam", "symmetric_system", (132,)),
    Sutra(8, "Puranapuranabhyam", "complete_quadratic", (155, 156)),
    Sutra(9, "Calana Kalanabhyam", "polynomial_derivative", (145, 175, 176, 177, 178)),
    Sutra(10, "Yavadunam", "deficiency_square", (24, 25, 26, 27)),
    Sutra(11, "Vyastisamastih", "symmetric_biquadratic", (160, 161)),
    Sutra(12, "Sesanyankena Caramena", "remainder_digits", (195, 196, 197)),
    Sutra(13, "Sopantyadvayamantyam", "arithmetic_progression_equation", (136, 137)),
    Sutra(14, "Ekanyunena Purvena", "nines_product", (29, 30, 31, 32)),
    Sutra(15, "Gunitasamuccayah", "product_coefficient_sum", (76, 81, 82, 83, 84, 85)),
    Sutra(16, "Gunakasamuccayah", "factor_product_derivative", (175, 176, 177, 178)),
    Sutra(17, "Anurupyena", "working_base_product", (17, 18, 19, 20, 21, 22)),
    Sutra(18, "Sisyate Sesasamjnah", "constant_remainders", (19,)),
    Sutra(19, "Adyamadyenantyamantyena", "complementary_binomial_factor", (74, 75)),
    Sutra(20, "Kevalaih Saptakam Gunyat", "sevenths_repetend", (202,)),
    Sutra(21, "Vestanam", "osculation", (259, 260, 261, 262, 265, 266)),
    Sutra(22, "Yavadunam Tavadunam", "deficiency_cube", (288, 289)),
    Sutra(23, "Yavadunam Tavadunikrtya Varganca Yojayet", "deficiency_square", (24, 25, 26, 27)),
    Sutra(24, "Antyayordasakepi", "complementary_last_digits", (28, 29)),
    Sutra(25, "Antyayoreva", "last_term_equation", (138, 139, 140, 141, 142, 143)),
    Sutra(26, "Samuccayagunitah", "sum_crosswise_products", (76,)),
    Sutra(27, "Lopanasthapanabhyam", "homogeneous_quadratic_factors", (77, 78, 79)),
    Sutra(28, "Vilokanam", "observed_reciprocal_pair", (147, 148)),
    Sutra(29, "Gunitasamuccayah Samuccayagunitah", "coefficient_sum_identity", (76,)),
)


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction, sp.Rational)):
        raise TypeError("An exact integer or rational is required")
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    return sp.Rational(value)


def integer(value):
    value = rational(value)
    if value.q != 1:
        raise ValueError("An integer is required")
    return int(value)


def coefficients(values):
    values = tuple(rational(v) for v in values)
    if not values:
        raise ValueError("A nonempty coefficient sequence is required")
    return values


def crosswise(a, b):
    a, b = coefficients(a), coefficients(b)
    return tuple(sum((a[i] * b[k - i] for i in range(len(a))
                      if 0 <= k - i < len(b)), sp.S.Zero)
                 for k in range(len(a) + len(b) - 1))


def polynomial(values, variable):
    return sp.Add(*(v * variable ** i for i, v in enumerate(values)), evaluate=False)


def nine_ender_repetend(denominator):
    denominator = integer(denominator)
    if denominator < 9 or denominator % 10 != 9:
        raise ValueError("The denominator must be positive and end in 9")
    multiplier = (denominator + 1) // 10
    digit, carry = 1, 0
    reversed_digits = []
    while True:
        reversed_digits.append(digit)
        carry, digit = divmod(multiplier * digit + carry, 10)
        if (digit, carry) == (1, 0):
            return tuple(reversed(reversed_digits))


@dataclass(frozen=True)
class Evaluation:
    sutra: Sutra
    value: object
    spectral_coefficients: Tuple[sp.Rational, ...]


def evaluation(sid, value, terms):
    return Evaluation(SUTRAS[sid - 1], value, coefficients(terms))


def ekadhikena(denominator):
    digits = nine_ender_repetend(denominator)
    return evaluation(1, digits, digits)


def nikhilam(a, b, base):
    a, b, base = map(rational, (a, b, base))
    if base <= 0:
        raise ValueError("The base must be positive")
    da, db = base - a, base - b
    cross, right = (a - db) * base, da * db
    return evaluation(2, cross + right, (da, db, cross, right))


def urdhva(a, b):
    result = crosswise(a, b)
    return evaluation(3, result, result)


def paravartya(dividend, divisor):
    dividend, divisor = coefficients(dividend), coefficients(divisor)
    if divisor[-1] == 0:
        raise ValueError("The leading divisor coefficient must be nonzero")
    if len(dividend) < len(divisor):
        return evaluation(4, ((sp.S.Zero,), dividend), (sp.S.Zero,) + dividend)
    remainder = list(dividend)
    quotient = [sp.S.Zero] * (len(dividend) - len(divisor) + 1)
    for k in range(len(quotient) - 1, -1, -1):
        quotient[k] = remainder[k + len(divisor) - 1] / divisor[-1]
        for j in range(len(divisor)):
            remainder[k + j] += -divisor[j] * quotient[k]
    if any(remainder[len(divisor) - 1:]):
        raise ArithmeticError("Polynomial division did not eliminate the leading terms")
    quotient = tuple(quotient)
    remainder = tuple(remainder[:len(divisor) - 1]) or (sp.S.Zero,)
    return evaluation(4, (quotient, remainder), quotient + remainder)


def shunyam(a, b, c, d):
    a, b, c, d = map(rational, (a, b, c, d))
    if a * b != c * d:
        raise ValueError("The independent-term products must be equal")
    difference = a + b - c - d
    result = sp.S.Complexes if difference == 0 else sp.FiniteSet(0)
    return evaluation(5, result, (a * b, c * d, difference))


def anurupye(a, b, c, d, e, f):
    a, b, c, d, e, f = map(rational, (a, b, c, d, e, f))
    determinant = a * e - b * d
    if determinant == 0:
        raise ValueError("A unique solution requires independent equations")
    if b * f == e * c and (b != 0 or e != 0):
        result = (sp.S.Zero, c / b if b != 0 else f / e)
    elif a * f == d * c and (a != 0 or d != 0):
        result = (c / a if a != 0 else f / d, sp.S.Zero)
    else:
        raise ValueError("Neither coefficient column is proportional to the right-hand side")
    return evaluation(6, result, (determinant,) + result)


def sankalana(a, b, c, d):
    a, b, c, d = map(rational, (a, b, c, d))
    if a + b == 0 or a - b == 0:
        raise ValueError("The symmetric system must have a unique solution")
    total = (c + d) / (a + b)
    difference = (c - d) / (a - b)
    result = ((total + difference) / 2, (total - difference) / 2)
    return evaluation(7, result, (total, difference) + result)


def purana(a, b, c):
    a, b, c = map(rational, (a, b, c))
    if a == 0:
        raise ValueError("The quadratic coefficient must be nonzero")
    h = -b / (2 * a)
    k = c - b * b / (4 * a)
    roots = (h - sp.sqrt(-k / a), h + sp.sqrt(-k / a))
    return evaluation(8, (a, h, k, roots), (a, h, k))


def calana(p):
    p = coefficients(p)
    result = tuple(i * p[i] for i in range(1, len(p))) or (sp.S.Zero,)
    return evaluation(9, result, result)


def yavadunam(n, base):
    n, base = map(rational, (n, base))
    if base <= 0:
        raise ValueError("The base must be positive")
    deficiency = base - n
    left, right = (n - deficiency) * base, deficiency * deficiency
    return evaluation(10, left + right, (deficiency, left, right))


def vyashti(a, b, total):
    a, b, total = map(rational, (a, b, total))
    middle, half_difference = (a + b) / 2, (a - b) / 2
    n = half_difference
    delta = sp.sqrt(8 * n ** 4 + total / 2)
    roots = tuple(-middle + sign * sp.sqrt(-3 * n ** 2 + branch * delta)
                  for branch in (-1, 1) for sign in (-1, 1))
    return evaluation(11, roots, (middle, n, 2, 12 * n ** 2, 2 * n ** 4 - total))


def sesanyankena(denominator):
    denominator = integer(denominator)
    if denominator <= 1 or gcd(denominator, 10) != 1:
        raise ValueError("The denominator must exceed one and be coprime to ten")
    last_digit = (-pow(denominator, -1, 10)) % 10
    remainder, digits = 1, []
    while True:
        remainder = (10 * remainder) % denominator
        digits.append((remainder * last_digit) % 10)
        if remainder == 1:
            break
    return evaluation(12, tuple(digits), digits)


def sopantya(first, difference):
    first, difference = coefficients(first), coefficients(difference)
    if len(first) != 2 or len(difference) != 2:
        raise ValueError("Supply ascending constant and linear coefficients")
    a0, a1 = first
    d0, d1 = difference
    if a1 == 0 and d1 == 0:
        raise ValueError("At least one denominator factor must depend on the variable")
    x = sp.Symbol("x")
    factors = tuple((a0 + k * d0) + (a1 + k * d1) * x for k in range(4))
    if any(factor == 0 for factor in factors):
        raise ValueError("A denominator factor is identically zero")
    constant, linear = 3 * a0 + 7 * d0, 3 * a1 + 7 * d1
    excluded = sp.FiniteSet(*(-factor.subs(x, 0) / sp.diff(factor, x)
                              for factor in factors if sp.diff(factor, x) != 0))
    if difference == (0, 0) or (constant == 0 and linear == 0):
        result = sp.Complement(sp.S.Complexes, excluded)
    else:
        candidates = []
        if linear != 0:
            candidates.append(-constant / linear)
        if d1 != 0:
            candidates.append(-d0 / d1)
        result = sp.FiniteSet(*(r for r in candidates if r not in excluded))
    return evaluation(13, result, (d0, d1, constant, linear))


def ekanyunena(n, places):
    n, places = integer(n), integer(places)
    if places < 1:
        raise ValueError("At least one nine is required")
    base = 10 ** places
    left, right = (n - 1) * base, base - n
    return evaluation(14, left + right, (left, right))


def gunitasamuccaya(factors):
    factors = tuple(coefficients(f) for f in factors)
    if not factors:
        raise ValueError("At least one polynomial factor is required")
    result, total = (sp.S.One,), sp.S.One
    for factor in factors:
        result = crosswise(result, factor)
        total *= sum(factor)
    if sum(result) != total:
        raise ArithmeticError("The coefficient-sum identity failed")
    return evaluation(15, total, result)


def gunakasamuccaya(factors):
    factors = tuple(coefficients(f) for f in factors)
    if not factors:
        raise ValueError("At least one polynomial factor is required")
    result = [sp.S.Zero] * max(1, sum(len(f) - 1 for f in factors))
    for i, factor in enumerate(factors):
        term = calana(factor).value
        for j, other in enumerate(factors):
            if i != j:
                term = crosswise(term, other)
        for k, coefficient in enumerate(term):
            if coefficient != 0:
                result[k] += coefficient
    return evaluation(16, tuple(result), result)


def anurupyena(a, b, reference_base, ratio):
    a, b, reference_base, ratio = map(rational, (a, b, reference_base, ratio))
    if reference_base <= 0 or ratio <= 0:
        raise ValueError("The reference base and ratio must be positive")
    base = reference_base * ratio
    da, db = base - a, base - b
    left, right = (a - db) * ratio, da * db
    return evaluation(17, left * reference_base + right, (ratio, da, db, left, right))


def shishyate(dividend, divisors):
    dividend = integer(dividend)
    divisors = tuple(integer(d) for d in divisors)
    if not divisors or any(d <= 0 for d in divisors):
        raise ValueError("Positive divisors are required")
    pairs = tuple(divmod(dividend, d) for d in divisors)
    if len({r for q, r in pairs}) != 1:
        raise ValueError("These divisions do not have a constant remainder")
    return evaluation(18, pairs, tuple(v for pair in pairs for v in pair))


def adyam(quadratic, factor):
    c, b, a = coefficients(quadratic)
    q, p = coefficients(factor)
    if a == 0 or p == 0 or q == 0:
        raise ValueError("First and last factor coefficients must be nonzero")
    result = (c / q, a / p)
    if crosswise((q, p), result) != (c, b, a):
        raise ValueError("The supplied binomial is not a factor")
    return evaluation(19, result, (q, p) + result)


def kevalaih(numerator):
    numerator = integer(numerator)
    if not 1 <= numerator <= 6:
        raise ValueError("A proper positive seventh requires a numerator from one to six")
    block = numerator * 143 * (1000 - 1)
    digits = tuple(int(d) for d in str(block).zfill(6))
    return evaluation(20, digits, digits)


def osculate(n, divisor):
    n, divisor = integer(n), integer(divisor)
    if n < 0 or divisor <= 1 or gcd(divisor, 10) != 1:
        raise ValueError("Use n >= 0 and a divisor > 1 coprime to ten")
    positive = pow(10, -1, divisor)
    negative = divisor - positive
    prefix, digit = divmod(n, 10)
    return positive, negative, (prefix + positive * digit, prefix - negative * digit)


def vestanam(n, divisor):
    positive, negative, _ = osculate(n, divisor)
    chains = []
    for step in (lambda v: v // 10 + positive * (v % 10), lambda v: abs(v // 10 - negative * (v % 10))):
        chain, value = [n], n
        while True:
            following = step(value)
            if following >= value:
                break
            chain.append(following)
            value = following
        chains.append(tuple(chain))
    divisible = chains[0][-1] % divisor == 0
    if divisible != (chains[1][-1] % divisor == 0):
        raise ArithmeticError("Positive and negative osculation disagree")
    return evaluation(21, (divisible,) + tuple(chains), (positive, negative) + chains[0] + chains[1])


def lessen(n, base):
    n, base = rational(n), rational(base)
    if base <= 0:
        raise ValueError("The base must be positive")
    deficiency = base - n
    return evaluation(22, n - deficiency, (n, deficiency, n - deficiency))


def tavadunam_cube(n, base):
    n, base = rational(n), rational(base)
    if base <= 0:
        raise ValueError("The base must be positive")
    deficiency = base - n
    left, middle, right = n - 2 * deficiency, 3 * deficiency * deficiency, -deficiency ** 3
    return evaluation(22, left * base * base + middle * base + right, (deficiency, left, middle, right))


def extended_square(n, base):
    result = yavadunam(n, base)
    return evaluation(23, result.value, result.spectral_coefficients)


def antyayordasakepi(a, b):
    a, b = integer(a), integer(b)
    if a >= 0 and b >= 0:
        for k in range(1, len(str(max(a, b))) + 1):
            base = 10 ** k
            pa, da = divmod(a, base)
            pb, db = divmod(b, base)
            if pa == pb and da + db == base:
                left, right = pa * (pa + 1), da * db
                return evaluation(24, base * base * left + right, (left, right))
    raise ValueError("Equal nonnegative prefixes and last parts totalling a power of ten are required")


def samuccayagunitah(a, b):
    result = crosswise(a, b)
    return evaluation(26, sum(result), result)


def lopana(a, b, c, xy, yz, zx):
    a, b, c, xy, yz, zx = map(rational, (a, b, c, xy, yz, zx))
    if a == 0:
        raise ValueError("This x-retaining application requires a nonzero x-squared coefficient")
    dy, dz = sp.sqrt(xy ** 2 - 4 * a * b), sp.sqrt(zx ** 2 - 4 * a * c)
    if dy.is_Rational is not True or dz.is_Rational is not True:
        raise ValueError("The retained binary quadratics must factor over the rationals")
    for sy in (-1, 1):
        for sz in (-1, 1):
            p, q = (xy + sy * dy) / (2 * a), (zx + sz * dz) / (2 * a)
            r, s = xy - a * p, zx - a * q
            if p * s + q * r == yz:
                result = ((sp.S.One, p, q), (a, r, s))
                return evaluation(27, result, result[0] + result[1])
    raise ValueError("The retained factors do not reconstruct the yz coefficient")


def vilokanam(total, split, numerator=(0, 1), denominator=(1, 0), sign=1):
    total, split = rational(total), rational(split)
    sign = integer(sign)
    numerator, denominator = coefficients(numerator), coefficients(denominator)
    if len(numerator) != 2 or len(denominator) != 2:
        raise ValueError("Supply ascending constant and linear coefficients")
    if sign not in (-1, 1) or split == 0:
        raise ValueError("A nonzero observed ratio and sign +1 or -1 are required")
    if split + sign / split != total:
        raise ValueError("The observed reciprocal pair does not equal the right-hand side")
    if numerator == (0, 0) or denominator == (0, 0):
        raise ValueError("Both reciprocal ratio factors must be nonzero polynomials")
    x = sp.Symbol("x")
    A, B = polynomial(numerator, x), polynomial(denominator, x)
    excluded = sp.Union(sp.solveset(A, x, sp.S.Complexes), sp.solveset(B, x, sp.S.Complexes))
    roots = sp.Union(*(sp.solveset(sp.expand(A - ratio * B), x, sp.S.Complexes)
                       for ratio in (split, sign / split)))
    return evaluation(28, sp.Complement(roots, excluded),
                      (total, split, sign / split) + numerator + denominator)


def antyayoreva(a, b, common, numerator_last, denominator_last):
    a, b = coefficients(a), coefficients(b)
    common = coefficients(common)
    if len(a) != 2 or len(b) != 2:
        raise ValueError("Linear ratio factors are required")
    d, e = rational(numerator_last), rational(denominator_last)
    x = sp.Symbol("x")
    A, B, C = polynomial(a, x), polynomial(b, x), polynomial(common, x)
    if sp.expand(B) == 0 or sp.expand(B * C + e) == 0:
        raise ValueError("A denominator is identically zero")
    constant, linear = d * b[0] - e * a[0], d * b[1] - e * a[1]
    if linear == 0:
        if constant != 0:
            result = sp.S.EmptySet
        else:
            result = sp.Complement(sp.S.Complexes,
                                   sp.Union(sp.solveset(B, x), sp.solveset(B * C + e, x)))
    else:
        root = -constant / linear
        result = sp.S.EmptySet if B.subs(x, root) == 0 or (B * C + e).subs(x, root) == 0 else sp.FiniteSet(root)
    return evaluation(25, result, (d, e, constant, linear))


def gunitasamuccaya_samuccayagunitah(factors):
    result = gunitasamuccaya(factors)
    return evaluation(29, (result.value, sum(result.spectral_coefficients)), result.spectral_coefficients)


OPERATIONS = (
    ekadhikena, nikhilam, urdhva, paravartya, shunyam, anurupye, sankalana,
    purana, calana, yavadunam, vyashti, sesanyankena, sopantya, ekanyunena,
    gunitasamuccaya, gunakasamuccaya, anurupyena, shishyate, adyam, kevalaih,
    vestanam, tavadunam_cube, extended_square, antyayordasakepi, antyayoreva,
    samuccayagunitah, lopana, vilokanam, gunitasamuccaya_samuccayagunitah,
)


def evaluate(sid, **parameters):
    sid = integer(sid)
    if not 1 <= sid <= 29:
        raise ValueError("The sutra id must be between one and twenty-nine")
    return OPERATIONS[sid - 1](**parameters)
