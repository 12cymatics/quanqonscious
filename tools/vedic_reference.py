#!/usr/bin/env python3
import json
import sys
from fractions import Fraction as F
from math import floor

PHI = F(12586269025, 7778742049)
PHI_INV = F(12586269025 - 7778742049, 7778742049)


STORE_BITS = 128


def store(x):
    if x == 0:
        return F(0)
    neg = x < 0
    n, d = abs(x.numerator), x.denominator
    k = STORE_BITS - (n.bit_length() - d.bit_length())
    num = n << k if k >= 0 else n
    den = d if k >= 0 else d << -k
    if num < (den << (STORE_BITS - 1)):
        raise ArithmeticError("store hypothesis")
    m = (2 * num + den) // (2 * den)
    m = -m if neg else m
    return F(m, 1 << k) if k >= 0 else F(m << -k)


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
        raise ZeroDivisionError("walk measure")
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
    s15 = (prev * nxt - curr * curr) * F(4, 5)
    s28 = (prev * prev - nxt * nxt) / 1000
    s29 = (sum(lam) / 4 - curr) * F(9, 10) / 20
    return s9 + s7 + s10 + s6 + s14 + s15 + s28 + s29


def field(sid, psi, a, al):
    p = list(psi)
    if sid in (1, 14):
        ax = 1 << int(a[1])
        one = a[2]
        for i in range(16):
            nb = p[i ^ ax]
            f = nb / (nb * nb + one) if sid == 1 else -1 / (abs(nb) + one)
            p[i] = p[i] * (1 + al * f)
    elif sid == 2:
        m = max([F(1)] + [abs(x) for x in psi])
        B = radix_power_above(m, int(a[1]))
        p = [x + ((B - x) - x) * al for x in psi]
    elif sid == 3:
        conv = [sum(psi[j] * psi[i ^ j] for j in range(16)) / a[1] for i in range(16)]
        p = [x + (conv[i] - x) * al for i, x in enumerate(psi)]
    elif sid == 4:
        layers = [[x for i, x in enumerate(psi) if popcount(i) == h] for h in range(5)]
        c = [sum(L, F(0)) / len(L) for L in layers]
        for i in range(16):
            r = F(popcount(i)) / a[1]
            s = r / (1 + r)
            P = c[4]
            for k in (3, 2, 1, 0):
                P = c[k] + s * P
            p[i] = psi[i] + (P - psi[i]) * al
    elif sid == 5:
        for i in range(8):
            j = i ^ 15
            ti, tj = -psi[j], -psi[i]
            p[i] = psi[i] + (ti - psi[i]) * al
            p[j] = psi[j] + (tj - psi[j]) * al
    elif sid == 6:
        for axis in range(4):
            m = 1 << axis
            for base in range(16):
                if base & m:
                    continue
                v0, v1 = p[base], p[base | m]
                if v0 == 0:
                    raise ZeroDivisionError("S6")
                ratio = v1 / v0
                err = ratio - (a[1] if ratio >= 0 else -a[1])
                corr = store(-err * al * abs(v0 + v1))
                p[base | m] += corr
                p[base] -= corr / 2
    elif sid == 7:
        for axis in range(4):
            m = 1 << axis
            for base in range(16):
                if base & m:
                    continue
                S = (p[base] + p[base | m]) / 2
                A = (p[base] - p[base | m]) / 2
                dl = store((S * S - A * A) * A * al)
                p[base] = (S + dl) + (A - dl)
                p[base | m] = (S + dl) - (A - dl)
    elif sid == 8:
        for i in range(16):
            mini = sum(((p[i] + p[j]) / 2 for j in nbrs(i)), F(0)) / 4
            p[i] = p[i] + (mini - p[i]) * (al * a[1])
    elif sid == 9:
        lap = [sum(psi[j] for j in nbrs(i)) - 4 * psi[i] for i in range(16)]
        p = [psi[i] + lap[i] * al for i in range(16)]
    elif sid == 10:
        B = radix_power_above(max(F(1), abs(mean(psi))), int(a[1]))
        p = [x * (1 + al * ((B - abs(x)) / B) ** 2) for x in psi]
    elif sid == 11:
        layers = [[x for i, x in enumerate(psi) if popcount(i) == h] for h in range(5)]
        whole = mean(psi)
        p = [x + ((sum(layers[popcount(i)], F(0)) / len(layers[popcount(i)]) + whole) / 2 - x) * al for i, x in enumerate(psi)]
    elif sid == 12:
        m = int(a[1])
        for i, x in enumerate(psi):
            v = iabs_floor(x)
            dr = 0 if v == 0 else 1 + (v - 1) % m
            p[i] = x + (sgn_ge0(x) * dr - x) * al
    elif sid == 13:
        for i in range(8):
            j = i ^ 15
            num, den = abs(p[i].numerator), abs(p[j].numerator)
            if den == 0:
                raise ZeroDivisionError("S13")
            n, d, cf = num, den, []
            while len(cf) < int(a[1]) and d > 0:
                cf.append(n // d)
                n, d = d, n % d
            p0, p1, q0, q1 = 1, cf[0], 0, 1
            for ak in cf[1:]:
                p0, p1, q0, q1 = p1, ak * p1 + p0, q1, ak * q1 + q0
            diff = F(p1, q1) - p[i] / p[j]
            p[i] += diff * al / 2
            p[j] -= diff * al / 4
    elif sid == 15:
        for axis in range(4):
            m = 1 << axis
            bases = [b for b in range(16) if not b & m]
            sP = sum((p[b | m] for b in bases), F(0))
            sM = sum((p[b] for b in bases), F(0))
            sPr = sum((p[b] * p[b | m] for b in bases), F(0))
            corr = store(-(sP * sM - sPr))
            for b in bases:
                p[b | m] = p[b | m] * (1 + al * corr)
                p[b] = p[b] * (1 - al * corr / 2)
    elif sid == 16:
        for i in range(8):
            j = i ^ 15
            A, B = abs(p[i].numerator), abs(p[j].numerator)
            r0, r1, s0, s1, t0, t1 = A, B, 1, 0, 0, 1
            while r1 > 0:
                q = r0 // r1
                r0, r1 = r1, r0 - q * r1
                s0, s1 = s1, s0 - q * s1
                t0, t1 = t1, t0 - q * t1
            if r0 > 1:
                p[i] = p[i] + (p[i] / r0 - p[i]) * al
                p[j] = p[j] + (p[j] / r0 - p[j]) * al
            else:
                vi, vj = p[i], p[j]
                p[i] = vi + al * t0 * vj
                p[j] = vj + al * s0 * vi
    elif sid == 17:
        for i in range(16):
            g = sum((p[j] - p[i] for j in nbrs(i)), F(0))
            p[i] = p[i] + g * a[1] / 4 * al
    elif sid == 18:
        mA, mB = int(a[1]), int(a[2])
        for i, x in enumerate(psi):
            h = popcount(i)
            v = iabs_floor(x)
            if v % mA != h % mA or v % mB != h % mB:
                p[i] = x + ((h % mA - v % mA + mA) % mA) * al
    elif sid == 19:
        for i in range(8):
            j = i ^ 15
            vi, vj = p[i], p[j]
            disc = (vi + vj) ** 2 - 4 * vi * vj
            if disc < 0:
                corr = disc * (-al / 2)
                p[i] += corr
                p[j] -= corr
            else:
                ai, aj = abs(vi), abs(vj)
                ratio = ai / aj if ai > aj else aj / ai
                scale = al * (ratio - a[1]) * a[2]
                if ai > aj:
                    p[i] -= scale
                    p[j] += scale
                else:
                    p[j] -= scale
                    p[i] += scale
    elif sid == 20:
        b = int(a[1])
        for i, x in enumerate(psi):
            r = iabs_floor(x) % b
            if r != 0:
                p[i] = x + al * (-r if r <= b // 2 else b - r)
    elif sid == 21:
        o, m = int(a[1]), int(a[2])
        for i, x in enumerate(psi):
            v = iabs_floor(x)
            osc = v // 10 + (v % 10) * o
            target = v if osc % m == 0 else v + (m - osc % m)
            if target != v:
                p[i] = x + al * (target - v)
    elif sid == 22:
        for i in range(16):
            p[i] = p[i] + ((p[i] + p[i ^ 15]) / 2 - p[i]) * al
    elif sid == 23:
        T = int(a[1])
        for i in range(8):
            j = i ^ 15
            s = iabs_floor(psi[i]) % T + iabs_floor(psi[j]) % T
            if s != T and s != 0:
                adj = F(T - s, 2) * al
                p[i] += adj
                p[j] += adj
    elif sid == 24:
        M = 10 ** int(a[1])
        tails = [iabs_floor(x) % M for x in psi]
        g = tails[0]
        for t in tails[1:]:
            x, y = g, t
            while y > 0:
                x, y = y, x % y
            g = x
        if g > 1:
            p = [x + (tails[i] // g - tails[i]) * al for i, x in enumerate(psi)]
        else:
            for i, x in enumerate(psi):
                dr = tails[i]
                while dr > 9:
                    dr = sum(int(c) for c in str(dr))
                frac = x - floor(x)
                p[i] = x - al * (frac - F(dr, 9))
    elif sid == 25:
        prodS, prodT = F(1), F(1)
        for axis in range(4):
            lo = sum((psi[i] for i in range(16) if not i & (1 << axis)), F(0))
            hi = sum((psi[i] for i in range(16) if i & (1 << axis)), F(0))
            prodS *= lo
            prodT *= hi
        if prodS == 0 or prodT == 0:
            raise ZeroDivisionError("S25")
        if abs(prodS) != abs(prodT):
            sg = 1 if abs(prodS) > abs(prodT) else -1
            for i in range(16):
                h = popcount(i)
                if h == 2:
                    g = sum((p[j] - p[i] for j in nbrs(i)), F(0))
                    push = sg * g * a[2]
                else:
                    push = sg * (h - 2) * a[1]
                p[i] += al * push
    elif sid == 26:
        flag, mx = 0, F(0)
        for i, x in enumerate(psi):
            if abs(x) > mx:
                mx, flag = abs(x), i
        portion = psi[flag] / a[1]
        p = [x if i == flag else x + (portion - x) * al for i, x in enumerate(psi)]
    elif sid == 27:
        paths = [[0, 1, 3, 7, 15], [0, 2, 6, 14, 15], [0, 4, 12, 13, 15], [0, 8, 10, 11, 15]]
        mod = [False] * 16
        for path in paths:
            vals = [p[i] for i in path]
            md = sum((vals[k + 1] - vals[k] for k in range(4)), F(0)) / 4
            for k, idx in enumerate(path):
                if mod[idx]:
                    continue
                diff = vals[0] + k * md - vals[k]
                if diff != 0:
                    p[idx] += al * diff
                    mod[idx] = True
        for i in range(16):
            if mod[i]:
                continue
            nb = sum((p[j] for j in nbrs(i)), F(0)) / 4
            p[i] = p[i] + (nb - p[i]) * al / 2
    elif sid == 28:
        for i in range(16):
            for j in nbrs(i):
                if j <= i:
                    continue
                dl = store(al * (p[i] * p[i] - p[j] * p[j]) * a[1])
                p[i] -= dl
                p[j] += dl
    elif sid == 29:
        a29 = al * a[1]
        m = mean(psi)
        p = [x + (m - x) * a29 for x in psi]
    else:
        raise ValueError(sid)
    return p


GRVQ_K = {1: 0, 5: 1, 9: 2, 14: 3}


def sutra(sid, psi, a, gain, t):
    al = F(sid, 435) * a[0] * gain
    out = field(sid, psi, a, al)
    if sid in GRVQ_K:
        k = GRVQ_K[sid]
        out = inject(out, k, grvq(inverse(psi), k, t) * gain)
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
        except ZeroDivisionError as e:
            out.append({"refused": str(e)})
    json.dump({"results": out}, sys.stdout)


if __name__ == "__main__":
    main()
