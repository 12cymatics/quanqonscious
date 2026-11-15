'use strict';

const DB = (function(){
  if (typeof require === 'function') return require('./dyadic128.js');
  if (typeof globalThis !== 'undefined' && globalThis.DYADIC128) return globalThis.DYADIC128;
  throw new Error('faraday-bessel-d128: dns/dyadic128.js is not loaded.');
})();

const SERIES_BITS = 200;

function requireOrder(m){
  if (!Number.isSafeInteger(m) || m < 0) throw new RangeError(`m = ${m}: a non-negative whole order is required.`);
  return m;
}
function requireValue(x, name){
  if (x && typeof x.m === 'bigint' && Number.isSafeInteger(x.e)) return x;
  if (typeof x === 'string') return DB.fromDecimal(x);
  throw new TypeError(`${name}: a 128-bit value or decimal string is required; a float is not accepted.`);
}

function toFixed(x, bits){
  const sh = x.e + bits;
  return sh >= 0 ? x.m << BigInt(sh) : (x.m >= 0n ? x.m >> BigInt(-sh) : -((-x.m) >> BigInt(-sh)));
}

function besselJ(m, xIn){
  requireOrder(m);
  const x = requireValue(xIn, 'x');
  if (DB.isZero(x)) return m === 0 ? DB.ONE : DB.ZERO;
  if (DB.sign(x) < 0){ const v = besselJ(m, DB.neg(x)); return m % 2 ? DB.neg(v) : v; }
  const half = DB.scale2(x, -1);
  let prefactor = DB.ONE, fact = 1n;
  for (let i = 1; i <= m; i++){ prefactor = DB.mul(prefactor, half); fact *= BigInt(i); }
  prefactor = DB.div(prefactor, DB.fromBigInt(fact));
  const F = BigInt(SERIES_BITS), H = toFixed(half, SERIES_BITS), H2 = (H*H) >> F, M = BigInt(m);
  let term = 1n << F, sum = term, k = 1n;
  while (term !== 0n){ term = -(((term*H2) >> F)/(k*(k + M))); sum += term; k += 1n; }
  return DB.mul(prefactor, DB.store(sum, -SERIES_BITS, 1n));
}

function besselJp(m, xIn){
  requireOrder(m);
  const x = requireValue(xIn, 'x');
  if (DB.isZero(x)) return m === 1 ? DB.fromRational(1n, 2n) : DB.ZERO;
  return DB.sub(DB.div(DB.mul(DB.fromBigInt(BigInt(m)), besselJ(m, x)), x), besselJ(m + 1, x));
}

const STEP = DB.fromDecimal('0.3');
const FLOOR_MIN = DB.fromDecimal('0.02');

function zeroNear(f, m, xStarIn){
  const xStar = requireValue(xStarIn, 'xStar');
  const mm = DB.fromBigInt(BigInt(m) - 2n);
  const floorX = DB.cmp(mm, FLOOR_MIN) > 0 ? mm : FLOOR_MIN;
  const tol = DB.store(1n, -118, 1n);
  const seek = dir => {
    let x = DB.cmp(xStar, floorX) > 0 ? xStar : floorX, fx = f(m, x);
    if (DB.isZero(fx)) return x;
    for (let i = 0; i < 400; i++){
      const nx = dir > 0 ? DB.add(x, STEP) : DB.sub(x, STEP);
      if (DB.cmp(nx, floorX) <= 0) return null;
      const nf = f(m, nx);
      if (DB.isZero(nf)) return nx;
      if (DB.sign(fx) !== DB.sign(nf)){
        let p = DB.cmp(x, nx) < 0 ? x : nx, q = DB.cmp(x, nx) < 0 ? nx : x, fp = DB.cmp(x, nx) < 0 ? fx : nf;
        while (DB.cmp(DB.sub(q, p), DB.mul(tol, q)) > 0){
          const mid = DB.scale2(DB.add(p, q), -1), fm = f(m, mid);
          if (DB.isZero(fm)) return mid;
          if (DB.sign(fm) === DB.sign(fp)){ p = mid; fp = fm; } else q = mid;
        }
        return DB.scale2(DB.add(p, q), -1);
      }
      x = nx; fx = nf;
    }
    return null;
  };
  const up = seek(1), down = seek(-1);
  if (up === null) return down;
  if (down === null) return up;
  return DB.cmp(DB.abs(DB.sub(up, xStar)), DB.abs(DB.sub(down, xStar))) <= 0 ? up : down;
}
const jZeroNear = (m, xStar) => zeroNear(besselJ, requireOrder(m), xStar);
const jpZeroNear = (m, xStar) => zeroNear(besselJp, requireOrder(m), xStar);

function omegaOf(k, sigma, rho, h, g){
  const K = requireValue(k, 'k');
  const top = DB.add(DB.mul(requireValue(g, 'g'), K), DB.div(DB.mul(requireValue(sigma, 'sigma'), DB.mul(K, DB.mul(K, K))), requireValue(rho, 'rho')));
  return DB.sqrt(DB.mul(top, DB.tanh(DB.mul(K, requireValue(h, 'h')))));
}

function kOfOmega(w, sigma, rho, h, g){
  const W = requireValue(w, 'w');
  let lo = DB.fromDecimal('1e-9'), hi = DB.fromDecimal('1e8');
  const tol = DB.store(1n, -118, 1n);
  while (DB.cmp(DB.sub(hi, lo), DB.mul(tol, hi)) > 0){
    const mid = DB.sqrt(DB.mul(lo, hi));
    if (DB.cmp(omegaOf(mid, sigma, rho, h, g), W) < 0) lo = mid; else hi = mid;
  }
  return DB.sqrt(DB.mul(lo, hi));
}

const FARADAY_BESSEL_D128 = Object.freeze({ besselJ, besselJp, jZeroNear, jpZeroNear, omegaOf, kOfOmega, SERIES_BITS });
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_BESSEL_D128;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_BESSEL_D128 = FARADAY_BESSEL_D128;
