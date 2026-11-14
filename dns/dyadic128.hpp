#pragma once

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;

struct D128 { u64 hi, lo; i32 e; i32 s; };

static inline D128 dZero(){ D128 z; z.hi = 0; z.lo = 0; z.e = 0; z.s = 0; return z; }
static inline bool dIsZero(const D128 &a){ return (a.hi | a.lo) == 0; }
static inline int dSign(const D128 &a){ return dIsZero(a) ? 0 : (a.s ? -1 : 1); }
static inline D128 dNeg(D128 a){ if (!dIsZero(a)) a.s ^= 1; return a; }
static inline D128 dAbs(D128 a){ a.s = 0; return a; }
static inline D128 dScale2(D128 a, i32 k){ if (!dIsZero(a)) a.e += k; return a; }

static inline void mul64(u64 a, u64 b, u64 &hi, u64 &lo){
  u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
  u64 p00 = a0*b0, p01 = a0*b1, p10 = a1*b0, p11 = a1*b1;
  u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
  lo = (mid << 32) | (u32)p00;
  hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

static inline u64 addc(u64 a, u64 b, u64 &carry){
  u64 s = a + b;
  u64 c1 = s < a;
  u64 t = s + carry;
  u64 c2 = t < s;
  carry = c1 | c2;
  return t;
}

static inline u64 subb(u64 a, u64 b, u64 &borrow){
  u64 d = a - b;
  u64 b1 = a < b;
  u64 t = d - borrow;
  u64 b2 = d < borrow;
  borrow = b1 | b2;
  return t;
}

static inline int clz64(u64 x){ return x ? __builtin_clzll(x) : 64; }

static inline void mul128(u64 ah, u64 al, u64 bh, u64 bl, u64 r[4]){
  u64 t1, x1, x0, y1, y0, z1, z0;
  mul64(al, bl, t1, r[0]);
  mul64(al, bh, x1, x0);
  mul64(ah, bl, y1, y0);
  mul64(ah, bh, z1, z0);
  u64 c = 0;
  u64 r1 = addc(t1, x0, c); u64 c1 = c; c = 0;
  r1 = addc(r1, y0, c); c1 += c; c = 0;
  u64 r2 = addc(x1, y1, c); u64 c2 = c; c = 0;
  r2 = addc(r2, z0, c); c2 += c; c = 0;
  r2 = addc(r2, c1, c); c2 += c;
  r[1] = r1; r[2] = r2; r[3] = z1 + c2;
}

static inline D128 roundLimbs(const u64 *w, int n, i32 e, i32 s){
  int top = n - 1;
  while (top >= 0 && w[top] == 0) top--;
  if (top < 0) return dZero();
  int L = top*64 + (64 - clz64(w[top]));
  D128 r; r.s = s;
  if (L <= 128){
    int sh = 128 - L;
    u64 lo = w[0], hi = n > 1 ? w[1] : 0;
    if (sh >= 64){ hi = lo << (sh - 64); lo = 0; }
    else if (sh > 0){ hi = (hi << sh) | (lo >> (64 - sh)); lo <<= sh; }
    r.hi = hi; r.lo = lo; r.e = e - sh;
    return r;
  }
  int sh = L - 128;
  int q = sh >> 6, b = sh & 63;
  u64 lo, hi;
  if (b == 0){ lo = w[q]; hi = w[q + 1]; }
  else {
    lo = (w[q] >> b) | (w[q + 1] << (64 - b));
    u64 nx = q + 2 < n ? w[q + 2] : 0;
    hi = (w[q + 1] >> b) | (nx << (64 - b));
  }
  int rb = sh - 1;
  u64 roundBit = (w[rb >> 6] >> (rb & 63)) & 1;
  u64 sticky = 0;
  for (int i = 0; i < (rb >> 6); i++) sticky |= w[i];
  if (rb & 63) sticky |= w[rb >> 6] & ((1ULL << (rb & 63)) - 1);
  r.e = e + sh;
  if (roundBit && (sticky || (lo & 1))){
    lo += 1;
    if (lo == 0){ hi += 1; if (hi == 0){ hi = 1ULL << 63; r.e += 1; } }
  }
  r.hi = hi; r.lo = lo;
  return r;
}

static inline D128 dFromI64(i64 v){
  if (v == 0) return dZero();
  u64 w[2] = { v < 0 ? (u64)0 - (u64)v : (u64)v, 0 };
  return roundLimbs(w, 2, 0, v < 0 ? 1 : 0);
}

static inline D128 dMul(const D128 &a, const D128 &b){
  if (dIsZero(a) || dIsZero(b)) return dZero();
  u64 r[4];
  mul128(a.hi, a.lo, b.hi, b.lo, r);
  D128 o; o.s = a.s ^ b.s;
  i32 e = a.e + b.e;
  u64 hi, lo, remHi, remLo, half;
  if (r[3] >> 63){
    hi = r[3]; lo = r[2]; remHi = r[1]; remLo = r[0]; half = 1ULL << 63;
    e += 128;
  } else {
    hi = (r[3] << 1) | (r[2] >> 63); lo = (r[2] << 1) | (r[1] >> 63);
    remHi = r[1] & 0x7FFFFFFFFFFFFFFFULL; remLo = r[0]; half = 1ULL << 62;
    e += 127;
  }
  bool up = remHi > half || (remHi == half && (remLo != 0 || (lo & 1)));
  if (up){
    lo += 1;
    if (lo == 0){ hi += 1; if (hi == 0){ hi = 1ULL << 63; e += 1; } }
  }
  o.hi = hi; o.lo = lo; o.e = e;
  return o;
}

static inline D128 dAdd(const D128 &a0, const D128 &b0){
  if (dIsZero(a0)) return b0;
  if (dIsZero(b0)) return a0;
  if (a0.e - b0.e > 129) return a0;
  if (b0.e - a0.e > 129) return b0;
  const D128 &a = a0.e >= b0.e ? a0 : b0;
  const D128 &b = a0.e >= b0.e ? b0 : a0;
  int d = a.e - b.e;
  u64 X[5] = { 0, 0, 0, 0, 0 }, Y[5] = { b.lo, b.hi, 0, 0, 0 };
  int q = d >> 6, s = d & 63;
  if (s == 0){ X[q] = a.lo; X[q + 1] = a.hi; }
  else {
    X[q] = a.lo << s;
    X[q + 1] = (a.hi << s) | (a.lo >> (64 - s));
    X[q + 2] = a.hi >> (64 - s);
  }
  u64 W[5];
  i32 sign;
  if (a.s == b.s){
    u64 c = 0;
    for (int i = 0; i < 5; i++) W[i] = addc(X[i], Y[i], c);
    sign = a.s;
  } else {
    bool xGe = true;
    for (int i = 4; i >= 0; i--){ if (X[i] != Y[i]){ xGe = X[i] > Y[i]; break; } }
    u64 br = 0;
    if (xGe){ for (int i = 0; i < 5; i++) W[i] = subb(X[i], Y[i], br); sign = a.s; }
    else { for (int i = 0; i < 5; i++) W[i] = subb(Y[i], X[i], br); sign = b.s; }
  }
  return roundLimbs(W, 5, b.e, sign);
}

static inline D128 dSub(const D128 &a, const D128 &b){ return dAdd(a, dNeg(b)); }

static inline int dCmp(const D128 &a, const D128 &b){
  int sa = dSign(a), sb = dSign(b);
  if (sa != sb) return sa < sb ? -1 : 1;
  if (sa == 0) return 0;
  int mag;
  if (a.e != b.e) mag = a.e > b.e ? 1 : -1;
  else if (a.hi != b.hi) mag = a.hi > b.hi ? 1 : -1;
  else if (a.lo != b.lo) mag = a.lo > b.lo ? 1 : -1;
  else mag = 0;
  return sa > 0 ? mag : -mag;
}

static inline void udivKnuth(const u64 n[4], u64 dh, u64 dl, u32 qd[5], u64 &rh, u64 &rl){
  u32 u[9];
  for (int i = 0; i < 4; i++){ u[2*i] = (u32)n[i]; u[2*i + 1] = (u32)(n[i] >> 32); }
  u[8] = 0;
  u32 v[4] = { (u32)dl, (u32)(dl >> 32), (u32)dh, (u32)(dh >> 32) };
  const u64 B = 1ULL << 32;
  for (int j = 4; j >= 0; j--){
    u64 num = ((u64)u[j + 4] << 32) | u[j + 3];
    u64 qhat = num / v[3], rhat = num % v[3];
    while (qhat >= B || qhat*v[2] > ((rhat << 32) | u[j + 2])){
      qhat -= 1; rhat += v[3];
      if (rhat >= B) break;
    }
    i64 borrow = 0;
    u64 carry = 0;
    for (int i = 0; i < 4; i++){
      u64 p = qhat*v[i] + carry;
      carry = p >> 32;
      i64 t = (i64)u[i + j] - (i64)(u32)p + borrow;
      u[i + j] = (u32)t;
      borrow = t >> 32;
    }
    i64 t = (i64)u[j + 4] - (i64)carry + borrow;
    u[j + 4] = (u32)t;
    if (t < 0){
      qhat -= 1;
      u64 c = 0;
      for (int i = 0; i < 4; i++){
        u64 s2 = (u64)u[i + j] + v[i] + c;
        u[i + j] = (u32)s2; c = s2 >> 32;
      }
      u[j + 4] = (u32)((u64)u[j + 4] + c);
    }
    qd[j] = (u32)qhat;
  }
  rl = ((u64)u[1] << 32) | u[0];
  rh = ((u64)u[3] << 32) | u[2];
}

static inline D128 dDiv(const D128 &a, const D128 &b){
  if (dIsZero(b)) __builtin_trap();
  if (dIsZero(a)) return dZero();
  bool ge = a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
  int k = ge ? 127 : 128;
  u64 n[4];
  if (k == 128){ n[3] = a.hi; n[2] = a.lo; n[1] = 0; n[0] = 0; }
  else { n[3] = a.hi >> 1; n[2] = (a.lo >> 1) | (a.hi << 63); n[1] = a.lo << 63; n[0] = 0; }
  u32 qd[5];
  u64 rh, rl;
  udivKnuth(n, b.hi, b.lo, qd, rh, rl);
  u64 qlo = ((u64)qd[1] << 32) | qd[0], qhi = ((u64)qd[3] << 32) | qd[2];
  u64 t2hi = (rh << 1) | (rl >> 63), t2lo = rl << 1, t2top = rh >> 63;
  bool gt, eq;
  if (t2top){ gt = true; eq = false; }
  else { gt = t2hi > b.hi || (t2hi == b.hi && t2lo > b.lo); eq = t2hi == b.hi && t2lo == b.lo; }
  D128 r; r.s = a.s ^ b.s; r.e = a.e - b.e - k;
  if (gt || (eq && (qlo & 1))){
    qlo += 1;
    if (qlo == 0){ qhi += 1; if (qhi == 0){ qhi = 1ULL << 63; r.e += 1; } }
  }
  r.hi = qhi; r.lo = qlo;
  return r;
}

static inline u64 isqrt64(u64 x){
  if (x < 2) return x;
  u64 g = 1ULL << ((64 - clz64(x) + 1) >> 1);
  while (true){
    u64 y = (g + x/g) >> 1;
    if (y >= g) return g;
    g = y;
  }
}

static inline void isqrt256(const u64 M[4], u64 &rh, u64 &rl){
  u64 s = isqrt64(M[3]) + 1;
  u64 xh, xl;
  if (s >= (1ULL << 32)){ xh = ~0ULL; xl = ~0ULL; }
  else { xh = s << 32; xl = 0; }
  while (true){
    u32 qd[5];
    u64 remh, reml;
    udivKnuth(M, xh, xl, qd, remh, reml);
    u64 ql = ((u64)qd[1] << 32) | qd[0], qh = ((u64)qd[3] << 32) | qd[2], qt = qd[4];
    u64 c = 0;
    u64 sl = addc(xl, ql, c);
    u64 sh = addc(xh, qh, c);
    u64 st = qt + c;
    u64 yl = (sl >> 1) | (sh << 63), yh = (sh >> 1) | (st << 63);
    if (yh > xh || (yh == xh && yl >= xl)){ rh = xh; rl = xl; return; }
    xh = yh; xl = yl;
  }
}

static inline D128 dSqrt(const D128 &a){
  if (a.s && !dIsZero(a)) __builtin_trap();
  if (dIsZero(a)) return dZero();
  int s = 128;
  if (((a.e - s) & 1) != 0) s = 127;
  u64 M[4];
  if (s == 128){ M[3] = a.hi; M[2] = a.lo; M[1] = 0; M[0] = 0; }
  else { M[3] = a.hi >> 1; M[2] = (a.lo >> 1) | (a.hi << 63); M[1] = a.lo << 63; M[0] = 0; }
  u64 rh, rl;
  isqrt256(M, rh, rl);
  u64 sq[4];
  mul128(rh, rl, rh, rl, sq);
  u64 br = 0, d[4];
  for (int i = 0; i < 4; i++) d[i] = subb(M[i], sq[i], br);
  bool up = d[3] != 0 || d[2] != 0 || d[1] > rh || (d[1] == rh && d[0] > rl);
  D128 r; r.s = 0; r.e = (a.e - s)/2;
  if (up){
    rl += 1;
    if (rl == 0){ rh += 1; if (rh == 0){ rh = 1ULL << 63; r.e += 1; } }
  }
  r.hi = rh; r.lo = rl;
  return r;
}
