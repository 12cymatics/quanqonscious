from fractions import Fraction
from itertools import combinations
from math import factorial, gcd, isqrt


def _digits(n):
    if n < 0:
        raise ValueError(f"digit methods take non-negative integers, got {n}")
    return [int(ch) for ch in str(n)]


def _require_int(*values):
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"expected an integer, got {v!r}")


def _rational_sqrt(value):
    value = Fraction(value)
    if value < 0:
        return None
    num, den = value.numerator, value.denominator
    rn, rd = isqrt(num), isqrt(den)
    if rn * rn == num and rd * rd == den:
        return Fraction(rn, rd)
    return None


def _nearest_power_of_ten(*numbers):
    best = None
    for k in range(1, max(len(str(abs(n))) for n in numbers) + 2):
        cost = sum(abs(n - 10 ** k) for n in numbers)
        if best is None or cost < best[0]:
            best = (cost, k)
    return best[1]


def _poly_mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += Fraction(a) * Fraction(b)
    return tuple(out)


def _poly_divmod(n, d):
    n = [Fraction(c) for c in n]
    d = [Fraction(c) for c in d]
    if not d or d[0] == 0:
        raise ValueError("divisor polynomial needs a non-zero leading coefficient")
    if len(n) < len(d):
        return (Fraction(0),), tuple(n)
    q = []
    for i in range(len(n) - len(d) + 1):
        coef = n[i] / d[0]
        q.append(coef)
        for j, dc in enumerate(d):
            n[i + j] -= coef * dc
    r = n[len(n) - len(d) + 1:]
    return tuple(q), tuple(r) if r else (Fraction(0),)


def _sc(poly):
    return sum(Fraction(c) for c in poly)


def _linear_root(p, q):
    if p == 0:
        raise ValueError(f"{p}x + {q} = 0 has no unique root")
    return Fraction(-q) / Fraction(p)


def _proportional(u, v):
    return Fraction(u[0]) * Fraction(v[1]) == Fraction(u[1]) * Fraction(v[0])


def s1_ekadhika_of(denominator):
    _require_int(denominator)
    if denominator <= 0 or denominator % 10 != 9:
        raise ValueError(f"the ekadhika rule applies to denominators ending in 9, got {denominator}")
    return denominator // 10 + 1


def s1_recurring_decimal_division(numerator, denominator):
    _require_int(numerator, denominator)
    e = s1_ekadhika_of(denominator)
    if not 0 < numerator < denominator:
        raise ValueError(f"need 0 < numerator < denominator, got {numerator}/{denominator}")
    digits = []
    dividend = numerator
    while True:
        q, r = divmod(dividend, e)
        digits.append(q)
        dividend = 10 * r + q
        if dividend == numerator:
            return tuple(digits)


def s1_recurring_decimal_multiplication(numerator, denominator):
    _require_int(numerator, denominator)
    e = s1_ekadhika_of(denominator)
    if not 0 < numerator < denominator:
        raise ValueError(f"need 0 < numerator < denominator, got {numerator}/{denominator}")
    reversed_digits = []
    value = numerator
    while True:
        reversed_digits.append(value % 10)
        value = e * (value % 10) + value // 10
        if value == numerator:
            return tuple(reversed(reversed_digits))


def s1_square_ending_in_5(n):
    _require_int(n)
    if n < 0 or n % 10 != 5:
        raise ValueError(f"the ekadhika squaring rule needs a non-negative number ending in 5, got {n}")
    a = n // 10
    left = a * (a + 1)
    return left, 25, left * 100 + 25


def s2_complement(n, k=None):
    _require_int(n)
    if k is None:
        k = len(str(n))
    if not 0 < n < 10 ** k:
        raise ValueError(f"need 0 < n < 10^{k}, got {n}")
    digits = [int(ch) for ch in str(n).zfill(k)]
    last = max(i for i, d in enumerate(digits) if d != 0)
    out = [9 - d for d in digits[:last]] + [10 - digits[last]] + [0] * (k - last - 1)
    return int(''.join(str(d) for d in out))


def s2_multiply(a, b, k=None):
    _require_int(a, b)
    if a <= 0 or b <= 0:
        raise ValueError(f"Nikhilam multiplication takes positive integers, got {a}, {b}")
    if k is None:
        k = _nearest_power_of_ten(a, b)
    base = 10 ** k
    da, db = a - base, b - base
    left = a + db
    right = da * db
    carry, right_portion = divmod(right, base)
    left_portion = left + carry
    return {'base': base, 'deviations': (da, db), 'left': left, 'right': right,
            'left_portion': left_portion, 'right_portion': right_portion,
            'product': left_portion * base + right_portion}


def s2_divide(dividend, divisor):
    _require_int(dividend, divisor)
    if dividend < 0 or divisor <= 0:
        raise ValueError(f"Nikhilam division takes dividend >= 0 and divisor > 0, got {dividend}, {divisor}")
    k = len(str(divisor))
    complement = [int(ch) for ch in str(10 ** k - divisor).zfill(k)]
    digits = _digits(dividend)
    if len(digits) <= k:
        q, r = 0, dividend
    else:
        columns = list(digits)
        quotient_columns = len(digits) - k
        for i in range(quotient_columns):
            qi = columns[i]
            for j, c in enumerate(complement, start=1):
                columns[i + j] += qi * c
        q = 0
        for value in columns[:quotient_columns]:
            q = q * 10 + value
        r = 0
        for value in columns[quotient_columns:]:
            r = r * 10 + value
    extra, r = divmod(r, divisor)
    return q + extra, r


def s3_columns(a_digits, b_digits):
    width = max(len(a_digits), len(b_digits))
    a = [0] * (width - len(a_digits)) + list(a_digits)
    b = [0] * (width - len(b_digits)) + list(b_digits)
    return tuple(sum(a[i] * b[k - i] for i in range(width) if 0 <= k - i < width)
                 for k in range(2 * width - 1))


def _carry_columns(columns):
    out = []
    carry = 0
    for col in reversed(columns):
        carry, digit = divmod(col + carry, 10)
        out.append(digit)
    while carry:
        carry, digit = divmod(carry, 10)
        out.append(digit)
    value = 0
    for d in reversed(out):
        value = value * 10 + d
    return value


def s3_multiply(a, b):
    _require_int(a, b)
    columns = s3_columns(_digits(a), _digits(b))
    return columns, _carry_columns(columns)


def s3_polynomial_multiply(p, q):
    return _poly_mul(p, q)


def s3_duplex(digits):
    n = len(digits)
    total = 0
    for i in range(n // 2):
        total += 2 * digits[i] * digits[n - 1 - i]
    if n % 2:
        total += digits[n // 2] ** 2
    return total


def s3_square_by_duplex(n):
    _require_int(n)
    d = _digits(n)
    m = len(d)
    columns = [s3_duplex(d[max(0, k - m + 1):k + 1]) if k < m else s3_duplex(d[k - m + 1:])
               for k in range(2 * m - 1)]
    return tuple(columns), _carry_columns(columns)


def s4_divide(dividend, divisor):
    _require_int(dividend, divisor)
    if dividend < 0 or divisor < 10:
        raise ValueError(f"Paravartya division takes dividend >= 0 and divisor >= 10, got {dividend}, {divisor}")
    ds = _digits(divisor)
    if ds[0] != 1:
        raise ValueError(f"Paravartya transposes a divisor just above a power of ten (leading digit 1), got {divisor}")
    m = len(ds) - 1
    transposed = [-d for d in ds[1:]]
    digits = _digits(dividend)
    if len(digits) <= m:
        q, r = 0, dividend
    else:
        columns = list(digits)
        quotient_columns = len(digits) - m
        for i in range(quotient_columns):
            qi = columns[i]
            for j, t in enumerate(transposed, start=1):
                columns[i + j] += qi * t
        q = 0
        for value in columns[:quotient_columns]:
            q = q * 10 + value
        r = 0
        for value in columns[quotient_columns:]:
            r = r * 10 + value
    extra, r = divmod(r, divisor)
    return q + extra, r


def s4_polynomial_divide(dividend, divisor):
    dividend = [Fraction(c) for c in dividend]
    divisor = [Fraction(c) for c in divisor]
    if not divisor or divisor[0] != 1:
        raise ValueError("Paravartya polynomial division takes a monic divisor")
    m = len(divisor) - 1
    transposed = [-c for c in divisor[1:]]
    columns = list(dividend)
    quotient_columns = len(columns) - m
    if quotient_columns <= 0:
        return (Fraction(0),), tuple(columns)
    for i in range(quotient_columns):
        qi = columns[i]
        for j, t in enumerate(transposed, start=1):
            columns[i + j] += qi * t
    remainder = tuple(columns[quotient_columns:]) or (Fraction(0),)
    return tuple(columns[:quotient_columns]), remainder


def s4_solve_linear(a, b, c, d):
    if Fraction(a) == Fraction(c):
        raise ValueError("ax + b = cx + d with a = c has no unique root")
    return (Fraction(d) - Fraction(b)) / (Fraction(a) - Fraction(c))


def s4_solve_simultaneous(a, b, p, c, d, q):
    a, b, p, c, d, q = map(Fraction, (a, b, p, c, d, q))
    den = b * c - a * d
    if den == 0:
        raise ValueError("the cyclic rule needs bc - ad != 0")
    return (b * q - d * p) / den, (p * c - q * a) / den


def s4_cyclic_fraction_root(n1, n2, n3, a, b, c):
    n1, n2, n3, a, b, c = map(Fraction, (n1, n2, n3, a, b, c))
    total = n1 + n2 + n3
    if total == 0:
        raise ValueError("the numerators sum to zero, so the equation has no unique root")
    return -(n1 * c + n2 * a + n3 * b) / total


def s5_common_factor(a, b, k):
    if Fraction(a) == Fraction(b):
        raise ValueError("a(x + k) = b(x + k) with a = b is an identity")
    return Fraction(-k)


def s5_independent_terms(a, b, c, d):
    if Fraction(a) * Fraction(b) != Fraction(c) * Fraction(d):
        raise ValueError(f"the samuccaya differs: ab = {a * b} but cd = {c * d}")
    if Fraction(a) + Fraction(b) == Fraction(c) + Fraction(d):
        raise ValueError("a + b = c + d makes both sides identical")
    return Fraction(0)


def s5_equal_numerators(a, b, c, d):
    if Fraction(a) + Fraction(c) == 0:
        raise ValueError("the sum of the denominators has no x term")
    return -(Fraction(b) + Fraction(d)) / (Fraction(a) + Fraction(c))


def s5_fraction_equation_roots(n1, d1, n2, d2):
    total_n = (Fraction(n1[0]) + Fraction(n2[0]), Fraction(n1[1]) + Fraction(n2[1]))
    total_d = (Fraction(d1[0]) + Fraction(d2[0]), Fraction(d1[1]) + Fraction(d2[1]))
    if total_n == (0, 0) or not _proportional(total_n, total_d):
        raise ValueError("N1 + N2 and D1 + D2 are not the same samuccaya")
    roots = [_linear_root(*total_n)]
    x2_left = Fraction(n1[0]) * Fraction(d2[0])
    x2_right = Fraction(n2[0]) * Fraction(d1[0])
    if x2_left != x2_right:
        diff1 = (Fraction(n1[0]) - Fraction(d1[0]), Fraction(n1[1]) - Fraction(d1[1]))
        diff2 = (Fraction(n2[0]) - Fraction(d2[0]), Fraction(n2[1]) - Fraction(d2[1]))
        if diff1 == (0, 0) or not _proportional(diff1, diff2):
            raise ValueError("the quadratic case needs N1 - D1 and N2 - D2 to be the same samuccaya")
        roots.append(_linear_root(*diff1))
    return tuple(roots)


def s5_reciprocal_sums(lhs, rhs):
    fractions = list(lhs) + list(rhs)
    if len(lhs) != 2 or len(rhs) != 2:
        raise ValueError("this meaning takes two fractions on each side")
    lcm = 1
    for _, p, _ in fractions:
        p = Fraction(p)
        if p == 0 or p.denominator != 1:
            raise ValueError("denominators must be linear with integer x-coefficients")
        lcm = lcm * abs(int(p)) // gcd(lcm, abs(int(p)))
    scaled = [(Fraction(n) * lcm / Fraction(p), Fraction(lcm), Fraction(q) * lcm / Fraction(p))
              for n, p, q in fractions]
    if len({s[0] for s in scaled}) != 1:
        raise ValueError("after scaling the x-coefficients the numerators differ")
    left = (scaled[0][1] + scaled[1][1], scaled[0][2] + scaled[1][2])
    right = (scaled[2][1] + scaled[3][1], scaled[2][2] + scaled[3][2])
    if left != right:
        raise ValueError(f"D1 + D2 = {left} differs from D3 + D4 = {right}")
    return _linear_root(*left)


def s5_cubic_binomials(p, q):
    if Fraction(p) == Fraction(q):
        raise ValueError("(x+p)^3 + (x+p)^3 = 2(x+p)^3 is an identity")
    return -(Fraction(p) + Fraction(q)) / 2


def _determinant(rows):
    m = [list(r) for r in rows]
    n = len(m)
    det = Fraction(1)
    for col in range(n):
        pivot = next((i for i in range(col, n) if m[i][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            det = -det
        det *= m[col][col]
        for i in range(col + 1, n):
            factor = m[i][col] / m[col][col]
            for j in range(col, n):
                m[i][j] -= factor * m[col][j]
    return det


def s6_solve(a, b, p, c, d, q):
    a, b, p, c, d, q = map(Fraction, (a, b, p, c, d, q))
    if a * d == b * c:
        raise ValueError("the equations are dependent, so the ratio rule has no unique answer")
    if b != 0 and d != 0 and b * q == d * p:
        return Fraction(0), p / b
    if a != 0 and c != 0 and a * q == c * p:
        return p / a, Fraction(0)
    raise ValueError("no unknown's coefficients are in the ratio of the independent terms")


def s6_solve_system(rows, rhs):
    rows = [[Fraction(v) for v in row] for row in rows]
    rhs = [Fraction(v) for v in rhs]
    n = len(rows[0])
    if len(rows) != n or _determinant(rows) == 0:
        raise ValueError("the ratio rule needs a square system with a unique solution")
    for col in range(n):
        column = [row[col] for row in rows]
        pivot = next((i for i, v in enumerate(column) if v != 0), None)
        if pivot is None:
            continue
        ratio = rhs[pivot] / column[pivot]
        if all(rhs[i] == column[i] * ratio for i in range(len(rows))):
            out = [Fraction(0)] * n
            out[col] = ratio
            return tuple(out)
    raise ValueError("no coefficient column is proportional to the independent terms")


def s7_solve(a1, b1, p1, a2, b2, p2):
    a1, b1, p1, a2, b2, p2 = map(Fraction, (a1, b1, p1, a2, b2, p2))
    if (a2, b2) != (b1, a1) and (a2, b2) != (-b1, -a1):
        raise ValueError("Sankalana-vyavakalana needs the x and y coefficients interchanged")
    sum_x, sum_y, sum_p = a1 + a2, b1 + b2, p1 + p2
    dif_x, dif_y, dif_p = a1 - a2, b1 - b2, p1 - p2
    eqs = []
    for cx, cy, rhs in ((sum_x, sum_y, sum_p), (dif_x, dif_y, dif_p)):
        if cx == 0 and cy == 0:
            raise ValueError("addition or subtraction cancelled both unknowns")
        if cx == cy:
            eqs.append(('x+y', rhs / cx))
        elif cx == -cy:
            eqs.append(('x-y', rhs / cx))
        else:
            raise ValueError("addition and subtraction did not isolate x + y and x - y")
    values = dict(eqs)
    if set(values) != {'x+y', 'x-y'}:
        raise ValueError("addition and subtraction gave the same combination twice")
    x = (values['x+y'] + values['x-y']) / 2
    y = (values['x+y'] - values['x-y']) / 2
    return x, y


def s8_complete_square(a, b, c):
    a, b, c = map(Fraction, (a, b, c))
    if a == 0:
        raise ValueError("not a quadratic")
    h = b / (2 * a)
    return h, c - b * b / (4 * a)


def s8_complete_cube(p, q, r):
    p, q, r = map(Fraction, (p, q, r))
    shift = p / 3
    big_p = q - p * p / 3
    big_q = 2 * p ** 3 / 27 - p * q / 3 + r
    return shift, big_p, big_q


def s8_solve_cubic(p, q, r):
    shift, big_p, big_q = s8_complete_cube(p, q, r)
    if big_q != 0:
        raise ValueError(f"completing the cube leaves y^3 + ({big_p})y + ({big_q}) = 0 with a non-zero constant; "
                         "completion alone does not resolve it")
    roots = [Fraction(0)]
    root = _rational_sqrt(-big_p)
    if root is None:
        raise ValueError(f"y^2 = {-big_p} has no rational root")
    if root != 0:
        roots += [root, -root]
    return tuple(sorted({y - shift for y in roots}))


def s9_differential(a, b, c):
    a, b, c = map(Fraction, (a, b, c))
    if a == 0:
        raise ValueError("not a quadratic")
    discriminant = b * b - 4 * a * c
    root = _rational_sqrt(discriminant)
    first_differential = (2 * a, b)
    if root is None:
        return {'first_differential': first_differential, 'discriminant': discriminant, 'roots': None,
                'centre': -b / (2 * a), 'radicand': discriminant / (4 * a * a)}
    roots = tuple(sorted({(-b + root) / (2 * a), (-b - root) / (2 * a)}))
    return {'first_differential': first_differential, 'discriminant': discriminant, 'roots': roots,
            'centre': -b / (2 * a), 'radicand': discriminant / (4 * a * a)}


def s10_square(n, k=None):
    _require_int(n)
    if n <= 0:
        raise ValueError(f"Yavadunam squaring takes positive integers, got {n}")
    if k is None:
        k = _nearest_power_of_ten(n)
    base = 10 ** k
    d = n - base
    left, right = n + d, d * d
    carry, right_portion = divmod(right, base)
    return {'base': base, 'deviation': d, 'left': left, 'right': right,
            'value': (left + carry) * base + right_portion}


def s10_cube(n, k=None):
    _require_int(n)
    if n <= 0:
        raise ValueError(f"Yavadunam cubing takes positive integers, got {n}")
    if k is None:
        k = _nearest_power_of_ten(n)
    base = 10 ** k
    d = n - base
    portions = [n + 2 * d, 3 * d * d, d ** 3]
    carry = 0
    out = []
    for value in reversed(portions[1:]):
        carry, digits = divmod(value + carry, base)
        out.append(digits)
    head = portions[0] + carry
    value = head
    for digits in reversed(out):
        value = value * base + digits
    return {'base': base, 'deviation': d, 'portions': tuple(portions), 'value': value}


def s11_sum_of_fourth_powers(u, v, p):
    u, v, p = map(Fraction, (u, v, p))
    if u == v:
        raise ValueError("the two binomials must differ")
    m, n = (u + v) / 2, (u - v) / 2
    disc = 8 * n ** 4 + p / 2
    sq = _rational_sqrt(disc)
    if sq is None:
        raise ValueError(f"a^2 = -3n^2 ± sqrt({disc}) is not rational")
    real_roots = []
    nonreal = []
    for a2 in (-3 * n * n + sq, -3 * n * n - sq):
        if a2 < 0:
            nonreal.append(a2)
            continue
        a = _rational_sqrt(a2)
        if a is None:
            raise ValueError(f"a^2 = {a2} has no rational root")
        real_roots += [a - m, -a - m]
    return {'average': m, 'half_difference': n, 'a_squared': (-3 * n * n + sq, -3 * n * n - sq),
            'roots': tuple(sorted(set(real_roots))), 'nonreal_a_squared': tuple(nonreal)}


def s12_last_digit_multiplier(denominator):
    _require_int(denominator)
    if denominator <= 1 or gcd(denominator, 10) != 1:
        raise ValueError(f"needs a denominator > 1 coprime to 10, got {denominator}")
    return next(m for m in range(10) if (m * denominator) % 10 == 9)


def s12_recurring_decimal(numerator, denominator):
    _require_int(numerator, denominator)
    ell = s12_last_digit_multiplier(denominator)
    if not 0 < numerator < denominator:
        raise ValueError(f"need 0 < numerator < denominator, got {numerator}/{denominator}")
    ratio = 10 % denominator
    remainders = []
    r = numerator
    while True:
        r = (r * ratio) % denominator
        remainders.append(r)
        if r == numerator:
            break
    return tuple(remainders), tuple((x * ell) % 10 for x in remainders)


def s13_solve(first, step):
    p, q = map(Fraction, first)
    r, s = map(Fraction, step)
    if r == 0 and s == 0:
        raise ValueError("the binomials must be in arithmetic progression with a non-zero step")
    c = (p + 2 * r, q + 2 * s)
    d = (p + 3 * r, q + 3 * s)
    return _linear_root(2 * c[0] + d[0], 2 * c[1] + d[1])


def s13_solve_binomials(a, b, c, d):
    a, b, c, d = [(Fraction(x[0]), Fraction(x[1])) for x in (a, b, c, d)]
    step = (b[0] - a[0], b[1] - a[1])
    for left, right in ((b, c), (c, d)):
        if (right[0] - left[0], right[1] - left[1]) != step:
            raise ValueError("A, B, C, D are not in arithmetic progression")
    return _linear_root(2 * c[0] + d[0], 2 * c[1] + d[1])


def s14_multiply_by_nines(n, k):
    _require_int(n, k)
    if n <= 0 or k <= 0:
        raise ValueError(f"needs positive n and k, got {n}, {k}")
    base = 10 ** k
    if n < base:
        left = n - 1
        right = base - n
        return {'case': 1 if len(str(n)) == k else 2, 'left': left, 'right': right, 'value': left * base + right}
    excess, rest = divmod(n, base)
    left = n - (excess + 1)
    right = base - rest
    carry, right_portion = divmod(right, base)
    return {'case': 3, 'left': left, 'right': right, 'value': (left + carry) * base + right_portion}


def s15_product_of_sums(factors):
    total = Fraction(1)
    for f in factors:
        total *= _sc(f)
    return total


def s15_check(factors, product):
    return s15_product_of_sums(factors) == _sc(product)


def s16_differentials(roots):
    roots = [Fraction(a) for a in roots]
    n = len(roots)
    out = []
    for j in range(1, n + 1):
        poly = [Fraction(0)] * (n - j + 1)
        for subset in combinations(range(n), n - j):
            term = (Fraction(1),)
            for i in subset:
                term = _poly_mul(term, (Fraction(1), roots[i]))
            offset = len(poly) - len(term)
            for i, cf in enumerate(term):
                poly[offset + i] += cf
        out.append(tuple(factorial(j) * c for c in poly))
    return tuple(out)


def us1_multiply(a, b, k, factor, kind):
    _require_int(a, b, k, factor)
    if factor <= 0 or k < 0:
        raise ValueError("the working-base factor must be positive")
    power = 10 ** k
    if kind == 'multiple':
        working = Fraction(power * factor)
        scale = Fraction(factor)
    elif kind == 'submultiple':
        working = Fraction(power, factor)
        scale = Fraction(1, factor)
    else:
        raise ValueError("kind is 'multiple' or 'submultiple'")
    da, db = a - working, b - working
    left = (a + db) * scale
    right = da * db
    value = left * power + right
    if value.denominator != 1:
        raise ArithmeticError("Anurupyena recombination did not give an integer")
    return {'working_base': working, 'left': left, 'right': right, 'value': int(value)}


def us1_split_middle(a, b, c):
    _require_int(a, b, c)
    target = a * c
    for b1 in sorted({d for x in range(1, abs(target) + 1) if target % x == 0 for d in (x, -x)} | {0},
                     key=lambda v: (abs(v), v)):
        b2 = b - b1
        if b1 * b2 == target:
            return b1, b2
    raise ValueError(f"{a}x^2 + {b}x + {c} has no integer split of the middle coefficient")


def us3_factor_quadratic(a, b, c):
    _require_int(a, b, c)
    if a == 0:
        raise ValueError("not a quadratic")
    b1, b2 = us1_split_middle(a, b, c)
    if b1 == 0:
        if c != 0:
            raise ValueError("a zero middle part needs c = 0")
        g = a
        first = (1, 0)
    else:
        g = gcd(a, b1)
        first = (a // g, b1 // g)
    second_x = Fraction(a, first[0])
    second_c = Fraction(c, first[1]) if first[1] != 0 else Fraction(b2, 1)
    return first, (second_x, second_c)


def us2_constant_remainder(dividend, divisors):
    _require_int(dividend, *divisors)
    if not divisors or any(d <= 0 for d in divisors):
        raise ValueError("positive divisors are required")
    pairs = tuple(divmod(dividend, d) for d in divisors)
    if len({r for _, r in pairs}) != 1:
        raise ValueError(f"the remainders {[r for _, r in pairs]} do not remain constant")
    return pairs


def us4_sevenths(n):
    _require_int(n)
    if not 0 < n < 7:
        raise ValueError(f"needs 0 < n < 7, got {n}")
    product = s14_multiply_by_nines(143 * n, 3)
    return tuple(int(ch) for ch in str(product['value']).zfill(6))


def us5_positive_osculator(divisor):
    _require_int(divisor)
    if divisor <= 1 or gcd(divisor, 10) != 1:
        raise ValueError(f"osculation needs a divisor > 1 coprime to 10, got {divisor}")
    multiplier = {9: 1, 1: 9, 3: 3, 7: 7}[divisor % 10]
    return (divisor * multiplier) // 10 + 1


def us5_negative_osculator(divisor):
    _require_int(divisor)
    if divisor <= 1 or gcd(divisor, 10) != 1:
        raise ValueError(f"osculation needs a divisor > 1 coprime to 10, got {divisor}")
    multiplier = {1: 1, 9: 9, 3: 7, 7: 3}[divisor % 10]
    return (divisor * multiplier) // 10


def us5_divisible(n, divisor, kind='positive'):
    _require_int(n, divisor)
    if n < 0:
        raise ValueError("osculation takes a non-negative number")
    if kind == 'positive':
        osc = us5_positive_osculator(divisor)
        step = lambda v: v // 10 + osc * (v % 10)
    elif kind == 'negative':
        osc = us5_negative_osculator(divisor)
        step = lambda v: abs(v // 10 - osc * (v % 10))
    else:
        raise ValueError("kind is 'positive' or 'negative'")
    trace = [n]
    value = n
    while True:
        nxt = step(value)
        if nxt >= value:
            break
        trace.append(nxt)
        value = nxt
    return value % divisor == 0, tuple(trace), osc


def us6_cube(n, k=None):
    _require_int(n)
    if n <= 0:
        raise ValueError("needs a positive integer")
    if k is None:
        k = _nearest_power_of_ten(n)
    base = 10 ** k
    deficiency = base - n
    portions = (n - 2 * deficiency, 3 * deficiency * deficiency, -deficiency ** 3)
    carry, right = divmod(portions[2], base)
    carry, middle = divmod(portions[1] + carry, base)
    left = portions[0] + carry
    return {'base': base, 'deficiency': deficiency, 'portions': portions,
            'normalised': (left, middle, right), 'value': (left * base + middle) * base + right}


def us7_square(n, k=None, factor=1, kind='multiple'):
    _require_int(n, factor)
    if n <= 0:
        raise ValueError("needs a positive integer")
    if k is None:
        k = _nearest_power_of_ten(n)
    if factor == 1:
        result = s10_square(n, k)
        return {'working_base': Fraction(result['base']), 'left': Fraction(result['left']),
                'right': result['right'], 'value': result['value']}
    return us1_multiply(n, n, k, factor, kind)


def us8_multiply(a, b):
    _require_int(a, b)
    if a <= 0 or b <= 0:
        raise ValueError("needs positive integers")
    for k in range(1, len(str(max(a, b))) + 1):
        base = 10 ** k
        if a // base == b // base and (a % base) + (b % base) == base:
            prev = a // base
            left = prev * (prev + 1)
            right = (a % base) * (b % base)
            return {'k': k, 'previous': prev, 'left': left, 'right': right,
                    'value': left * base * base + right}
    raise ValueError(f"{a} and {b} do not share a previous part with last parts totalling a power of ten")


def us9_last_terms(numerator, denominator, b, d):
    a1, c = _poly_divmod(numerator, b)
    a2, e = _poly_divmod(denominator, d)
    if a1 != a2:
        raise ValueError("barring the independent terms, numerator and denominator are not in the ratio B : D")
    if len(c) != 1 or len(e) != 1:
        raise ValueError("the last terms must be constants")
    c, e = c[0], e[0]
    bx, bc = map(Fraction, b)
    dx, dc = map(Fraction, d)
    return _linear_root(e * bx - c * dx, e * bc - c * dc)


def us9_series_sum(binomials, kind='equal'):
    binomials = [tuple(Fraction(v) for v in x) for x in binomials]
    if len(binomials) < 2:
        raise ValueError("needs at least two binomials")
    first, last = binomials[0], binomials[-1]
    if kind == 'equal':
        step = tuple(binomials[1][i] - binomials[0][i] for i in range(2))
        for x, y in zip(binomials, binomials[1:]):
            if tuple(y[i] - x[i] for i in range(2)) != step:
                raise ValueError("the binomials are not in arithmetic progression")
        numerator = (Fraction(len(binomials) - 1),)
    elif kind == 'difference':
        numerator = tuple(last[i] - first[i] for i in range(2))
    else:
        raise ValueError("kind is 'equal' or 'difference'")
    return numerator, _poly_mul(first, last)


def us10_sum_of_product(factors):
    product = (Fraction(1),)
    for f in factors:
        product = s3_polynomial_multiply(product, f)
    return _sc(product), product


def _binary_factorisations(a, b, c):
    target = a * c
    splits = {b1 for x in range(1, abs(target) + 1) if target % x == 0 for b1 in (x, -x)} | {0}
    out = []
    for b1 in sorted(splits, key=lambda v: (abs(v), v)):
        if b1 * (b - b1) != target:
            continue
        if b1 == 0:
            if c != 0:
                continue
            first = (Fraction(1), Fraction(0))
            second = (Fraction(a), Fraction(b))
        else:
            g = gcd(a, b1)
            first = (Fraction(a // g), Fraction(b1 // g))
            second = (Fraction(a) / first[0], Fraction(c) / first[1])
        out += [(first, second), (second, first)]
    return out


def us11_factor_homogeneous(coefficients, variables):
    coefficients = {tuple(sorted(k)): Fraction(v) for k, v in coefficients.items()}
    if any(v.denominator != 1 for v in coefficients.values()):
        raise ValueError("integer coefficients are required")
    pivot = variables[0]
    others = variables[1:]
    a = int(coefficients.get((pivot, pivot), 0))
    if a == 0:
        raise ValueError("the retained quadratics need a non-zero coefficient on the first variable squared")
    options = []
    for v in others:
        b = int(coefficients.get(tuple(sorted((pivot, v))), 0))
        c = int(coefficients.get((v, v), 0))
        found = _binary_factorisations(a, b, c)
        if not found:
            raise ValueError(f"eliminating all but {pivot} and {v} leaves a quadratic with no integer factorisation")
        options.append((v, found))

    def search(idx, left, right):
        if idx == len(options):
            expanded = {}
            for x in variables:
                for y in variables:
                    key = tuple(sorted((x, y)))
                    expanded[key] = expanded.get(key, Fraction(0)) + left[x] * right[y]
            keys = set(expanded) | set(coefficients)
            if all(expanded.get(key, 0) == coefficients.get(key, 0) for key in keys):
                return dict(left), dict(right)
            return None
        v, found = options[idx]
        for first, second in found:
            if left:
                scale = left[pivot] / first[0]
                f1 = (first[0] * scale, first[1] * scale)
                f2 = (second[0] / scale, second[1] / scale)
                if f2[0] != right[pivot]:
                    continue
            else:
                f1, f2 = first, second
            new_left = dict(left)
            new_right = dict(right)
            new_left[pivot], new_right[pivot] = f1[0], f2[0]
            new_left[v], new_right[v] = f1[1], f2[1]
            result = search(idx + 1, new_left, new_right)
            if result:
                return result
        return None

    result = search(0, {}, {})
    if result is None:
        raise ValueError("alternate elimination and retention found no consistent pair of factors")
    left, right = result
    return tuple(left[v] for v in variables), tuple(right[v] for v in variables)


def us12_reciprocal(r, sign=1):
    r = Fraction(r)
    if sign not in (1, -1):
        raise ValueError("sign is 1 for x + 1/x or -1 for x - 1/x")
    flip = 1
    if sign == 1 and r < 0:
        r, flip = -r, -1
    p, q = r.numerator, r.denominator
    for u in range(1, q + 1):
        if q % u:
            continue
        v = q // u
        if gcd(u, v) != 1:
            continue
        if u * u + sign * v * v == p:
            a = Fraction(u, v)
            return tuple(sorted({flip * a, flip * sign / a}))
    raise ValueError(f"{r} is not of the form a {'+' if sign == 1 else '-'} 1/a by observation")


def us13_verify(factors):
    product_of_sums = s15_product_of_sums(factors)
    sum_of_product, product = us10_sum_of_product(factors)
    return product_of_sums, sum_of_product, product_of_sums == sum_of_product
