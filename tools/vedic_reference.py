#!/usr/bin/env python3
import json
import sys
from fractions import Fraction as F
from math import floor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.set_int_max_str_digits(0)

from vedic_book_reference import NEIGHBORS, operands_of  # noqa: E402
from vedic_field import FieldBinding  # noqa: E402

PHI = F(12586269025, 7778742049)
PHI_INV = F(12586269025 - 7778742049, 7778742049)


def store(x):
    return x


def mean_abs(xs):
    return sum((abs(x) for x in xs), F(0)) / len(xs)


def proportionate(num, mag):
    if mag != 0:
        return num / mag
    if num != 0:
        raise ZeroDivisionError("a nonzero term over a zero magnitude")
    return num


def popcount(i):
    return bin(i).count("1")


def nbrs(i):
    return [i ^ 1, i ^ 2, i ^ 4, i ^ 8]


def sgn_ge0(x):
    return 1 if x >= 0 else -1


def ifloor(x):
    return floor(x)


def iabs_floor(x):
    return abs(floor(x))


def mean(p):
    return sum(p, F(0)) / 16


def inverse(p):
    return [sum((p[i] if (i >> k) & 1 else -p[i]) for i in range(16)) / 16 for k in range(4)]


def inject(p, k, d):
    return [p[i] + d if (i >> k) & 1 else p[i] - d for i in range(16)]


def set_b(p, s):
    return [((p[0] + p[1]) - (p[14] + p[15])) * s,
            ((p[0] + p[2]) - (p[13] + p[15])) * s,
            ((p[0] + p[4]) - (p[11] + p[15])) * s]


def radix_power_above(m, rad):
    order = 1
    while not (m < F(rad) ** order):
        order += 1
    return F(rad) ** order


def walk(lam, x):
    prev, nxt = (x - 1) % 4, (x + 1) % 4
    mu = [F(0)] * 4
    mu[x] = F(1, 2)
    wp, wn = abs(lam[prev]), abs(lam[nxt])
    tot = wp + wn
    if tot == 0:
        mu[prev] += F(1, 2) * F(1, 2)
        mu[nxt] += F(1, 2) * F(1, 2)
    else:
        mu[prev] += F(1, 2) * wp / tot
        mu[nxt] += F(1, 2) * wn / tot
    return mu


def w1(mu, nu):
    acc, cum = F(0), []
    for i in range(4):
        acc += mu[i] - nu[i]
        cum.append(acc)
    c = sorted(cum)[1]
    return sum(abs(f - c) for f in cum)


def ricci(lam, k):
    mk = walk(lam, k)
    s = F(0)
    for y in ((k - 1) % 4, (k + 1) % 4):
        s += 1 - w1(mk, walk(lam, y))
    return s / 2


def grvq(lam, k, t):
    prev, curr, nxt = lam[(k - 1) % 4], lam[k], lam[(k + 1) % 4]
    s9 = ((prev + nxt) / 2 - curr) / 20
    s7 = ricci(lam, k) / 40
    av = abs(curr)
    B = max(2 * av + 1, F(10))
    d = (B - av) / B
    s10 = curr * d * d * F(3, 10)
    cyc = (k + floor(t / 1000)) % 4
    s6 = curr * [PHI, PHI_INV, -PHI, -PHI_INV][cyc] * F(7, 10)
    opp = lam[(k + 2) % 4]
    s14 = -curr * F(2, 5) / (opp * opp + 1)
    s15 = proportionate((prev * nxt - curr * curr) * F(4, 5), mean_abs(lam))
    s28 = (abs(prev) - abs(nxt)) / 1000
    s29 = (sum(lam) / 4 - curr) * F(9, 10) / 20
    return s9 + s7 + s10 + s6 + s14 + s15 + s28 + s29


def field(sid, psi, a, al):
    binding = FieldBinding.from_parameters(sid, operands_of(sid, psi))
    return list(binding.rotate_matched(tuple(psi), NEIGHBORS, al))



GRVQ_K = {1: 0, 5: 1, 9: 2, 14: 3}


def sutra(sid, psi, a, gain, t):
    al = F(sid, 435) * a[0] * gain
    out = field(sid, psi, a, al)
    if sid in GRVQ_K:
        k = GRVQ_K[sid]
        out = inject(out, k, grvq(inverse(psi), k, t) * al)
    return out


def q(x):
    return F(int(x["n"]), int(x["d"]))


def s(x):
    return {"n": str(x.numerator), "d": str(x.denominator)}


def main():
    req = json.load(sys.stdin)
    out = []
    for c in req["cases"]:
        psi = [q(x) for x in c["psi"]]
        try:
            if c["kind"] == "sutra":
                r = sutra(c["id"], psi, [q(x) for x in c["args"]], q(c["gain"]), q(c["t"]))
                out.append({"psi": [s(x) for x in r]})
            elif c["kind"] == "grvq":
                out.append({"delta": s(grvq([q(x) for x in c["lambda"]], c["k"], q(c["t"])))})
            elif c["kind"] == "setB":
                out.append({"B": [s(x) for x in set_b(psi, q(c["s"]))]})
        except (ZeroDivisionError, ValueError, ArithmeticError) as e:
            out.append({"refused": str(e)})
    json.dump({"results": out}, sys.stdout)


if __name__ == "__main__":
    main()
