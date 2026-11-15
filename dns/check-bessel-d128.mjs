import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const DY = require(join(HERE, 'dyadic128.js'));
const B = require(join(HERE, 'faraday-bessel-d128.js'));
const E = require(join(HERE, 'faraday-disc-d128.js'));
const D = require(join(HERE, 'faraday-disc.js'));
const K = require(join(HERE, '..', 'faraday', 'kernel.js'));

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}
const N = DY.toNumberForDisplay;
const l2 = x => DY.isZero(x) ? -Infinity : x.e + 128;
const within = (x, ref, bits) => DY.cmp(DY.abs(DY.sub(x, DY.fromDecimal(ref))), DY.store(1n, -bits, 1n)) <= 0;

console.log('1. values and zeros against published digits');
{
  ok(within(B.besselJ(0, DY.ONE), '0.7651976865579665514497175261026632209093', 126), 'J0(1) to 2^-126 of its 40 published digits');
  ok(within(B.besselJ(1, DY.ONE), '0.4400505857449335159596822037189149131274', 126), 'J1(1) to 2^-126 of its 40 published digits');
  ok(within(B.jZeroNear(0, '2.4'), '2.404825557695772768621631879326454643124', 112), 'j(0,1) to 2^-112 of 2.4048255576957727686216318793264546431');
  ok(within(B.jZeroNear(1, '3.8'), '3.831705970207512315614435886308160766564', 112), 'j(1,1) to 2^-112 of 3.8317059702075123156144358863081607666');
  ok(within(B.jZeroNear(0, '5.5'), '5.520078110286310649596604112813027425221', 112), 'j(0,2) to 2^-112 of 5.5200781102863106495966041128130274252');
  const jp = B.jpZeroNear(1, '1.8');
  const d = B.besselJp(1, jp), eps = DY.store(1n, -100, 1n);
  ok(DY.cmp(DY.abs(d), DY.store(1n, -115, 1n)) <= 0 && DY.sign(B.besselJp(1, DY.sub(jp, eps))) !== DY.sign(B.besselJp(1, DY.add(jp, eps))),
     `J1' vanishes at j'(1,1) = ${N(jp)} to 2^${l2(DY.abs(d))} and changes sign across +-2^-100`);
  ok(Math.abs(N(jp) - K.jpZeroNear(1, 1.8)) < 1e-15 && Math.abs(N(B.jZeroNear(10, '14.5')) - K.jZeroNear(10, 14.5)) < 1e-14,
     'j\'(1,1) and j(10,1) agree with the float kernel to its precision');
}

console.log('2. the completeness sums and the recurrence hold to the last bits, out to x = 60');
{
  for (const xs of ['0.5', '3', '14.4', '37.5', '60']){
    const X = DY.fromDecimal(xs), top = Math.ceil(Number(xs)) + 90;
    let s1 = B.besselJ(0, X), s2 = DY.mul(s1, s1);
    for (let k = 1; k <= top; k++){ const j = B.besselJ(k, X); if (k % 2 === 0) s1 = DY.add(s1, DY.scale2(j, 1)); s2 = DY.add(s2, DY.scale2(DY.mul(j, j), 1)); }
    let worstRec = DY.ZERO;
    for (const m of [1, 5, 12, 40]){
      const e = DY.abs(DY.sub(DY.add(B.besselJ(m - 1, X), B.besselJ(m + 1, X)), DY.div(DY.mul(DY.fromBigInt(BigInt(2*m)), B.besselJ(m, X)), X)));
      if (DY.cmp(e, worstRec) > 0) worstRec = e;
    }
    const e1 = DY.abs(DY.sub(s1, DY.ONE)), e2 = DY.abs(DY.sub(s2, DY.ONE));
    ok(l2(e1) <= -118 && l2(e2) <= -118 && l2(worstRec) <= -118,
       `x = ${xs}, orders to ${top}: J0 + 2 sum J2k - 1 = 2^${l2(e1)}, J0^2 + 2 sum Jk^2 - 1 = 2^${l2(e2)}, J(m-1) + J(m+1) - (2m/x) Jm = 2^${l2(worstRec)}`);
  }
}

console.log('3. every order at x = 60 against a 600-bit series');
{
  const X = DY.fromDecimal('60');
  const wide = m => {
    const F = 600n, sh = X.e - 1 + 600, H = X.m << BigInt(sh), H2 = (H*H) >> F, M = BigInt(m);
    let term = 1n << F;
    for (let i = 1; i <= m; i++) term = ((term*H) >> F)/BigInt(i);
    let sum = term, k = 1n;
    while (term !== 0n){ term = -(((term*H2) >> F)/(k*(k + M))); sum += term; k += 1n; }
    return DY.store(sum, -600, 1n);
  };
  let worst = DY.ZERO;
  for (let m = 0; m <= 90; m += 3){ const e = DY.abs(DY.sub(B.besselJ(m, X), wide(m))); if (DY.cmp(e, worst) > 0) worst = e; }
  ok(l2(worst) <= -125, `J0 to J90 at 60, every third order, within 2^${l2(worst)}`);
  const X2 = DY.fromDecimal('33.3');
  const wide33 = (() => {
    const F = 600n, sh = X2.e - 1 + 600, H = sh >= 0 ? X2.m << BigInt(sh) : X2.m >> BigInt(-sh), H2 = (H*H) >> F, M = 20n;
    let term = 1n << F;
    for (let i = 1; i <= 20; i++) term = ((term*H) >> F)/BigInt(i);
    let sum = term, k = 1n;
    while (term !== 0n){ term = -(((term*H2) >> F)/(k*(k + M))); sum += term; k += 1n; }
    return DY.store(sum, -600, 1n);
  })();
  const e33 = DY.abs(DY.sub(B.besselJ(20, X2), wide33));
  ok(l2(e33) <= -125, `J20(33.3) within 2^${l2(e33)} of the 600-bit series as well`);
  const ulp = x => Math.pow(2, Math.floor(Math.log2(Math.abs(x))) - 52);
  let worstUlps = 0;
  for (const [m, x] of [[0, 0.7], [3, 9.1], [12, 14.4], [20, 33.3]]){
    const b = K.besselJ(m, x);
    worstUlps = Math.max(worstUlps, Math.abs(N(B.besselJ(m, DY.fromDecimal(String(x)))) - b)/ulp(b));
  }
  ok(worstUlps <= 32, `the float kernel lies within ${worstUlps.toFixed(0)} of its own ulps of these values; its Miller recurrence is what drifts, as J20(33.3) above shows`);
}

console.log('4. the dispersion relation and its inverse');
{
  const w = DY.mul(DY.scale2(DY.pi(), 1), DY.fromBigInt(40n));
  const k = B.kOfOmega(w, '0.0728', '1000', '0.002', '9.80665');
  const back = B.omegaOf(k, '0.0728', '1000', '0.002', '9.80665');
  ok(l2(DY.div(DY.abs(DY.sub(back, w)), w)) <= -115, `omega(k(omega)) = omega to 2^${l2(DY.div(DY.abs(DY.sub(back, w)), w))} relative`);
  ok(Math.abs(N(k) - K.kOfOmega(2*Math.PI*40, 0.0728, 1000, 0.002, 9.80665)) < 1e-12*N(k), `k = ${N(k).toPrecision(12)} /m, as the float kernel's`);
}

console.log('5. grid sizing without floats picks the float sizing\'s grid');
{
  const cases = [
    { m: 12, near: '14', R: '12.125e-3', h: '3e-3', rho: '998.2041322005837', nu: '1.0035510043427237e-6', gamma: '0.07273614042160757', f: 111n, contact: 'free', jp: true },
    { m: 10, near: '14.5', R: '0.015', h: '0.002', rho: '1000', nu: '1e-6', gamma: '0.0728', f: 80n, contact: 'pinned', jp: false },
    { m: 7, near: '18.29', R: '0.015', h: '0.001', rho: '1000', nu: '1e-6', gamma: '0.0728', f: 111n, contact: 'pinned', jp: false }];
  for (const c of cases){
    const od = DY.mul(DY.scale2(DY.pi(), 1), DY.fromBigInt(c.f));
    const root = c.jp ? B.jpZeroNear(c.m, c.near) : B.jZeroNear(c.m, c.near), k = DY.div(root, DY.fromDecimal(c.R));
    const gq = E.suggestGrid({ m: c.m, k, R: c.R, h: c.h, rho: c.rho, nu: c.nu, gamma: c.gamma, contact: c.contact,
                               omegaResponse: DY.scale2(od, -1), omegaDrive: od, profile: r => B.besselJ(c.m, DY.mul(k, r)) });
    const kf = (c.jp ? K.jpZeroNear(c.m, +c.near) : K.jZeroNear(c.m, +c.near))/(+c.R), odf = 2*Math.PI*Number(c.f);
    const gf = D.suggestGrid({ m: c.m, k: kf, R: +c.R, h: +c.h, rho: +c.rho, nu: +c.nu, gamma: +c.gamma, contact: c.contact,
                               omegaResponse: odf/2, omegaDrive: odf, profile: r => K.besselJ(c.m, kf*r) });
    const chosen = gq.unreachable ? 'no grid within the caps' : `${gq.nr}x${gq.nz}, zStretch ${gq.zStretchText}, ${gq.stepsPerPeriod} steps, surface operator off by ${N(gq.err).toExponential(3)}`;
    ok(!gq.unreachable && !gf.unreachable && gq.nr === gf.nr && gq.nz === gf.nz && Math.abs(N(gq.rStretch) - gf.rStretch) < 1e-15 && Number(gq.zStretchText) === gf.zStretch
       && gq.stepsPerPeriod === gf.stepsPerPeriod && Math.abs(N(gq.err) - gf.err) < 1e-12,
       `${c.m}-fold ${c.contact} at ${c.f} Hz: ${chosen}, as the float sizing chose`,
       gf.unreachable ? 'the float sizing found no grid' : `the float sizing chose ${gf.nr}x${gf.nz}, zStretch ${gf.zStretch}, ${gf.stepsPerPeriod} steps`);
    const key = t => `${t.nr}|${t.nz}|${Number(t.zStretchText === undefined ? t.zStretch : t.zStretchText)}`;
    const floatByKey = new Map(gf.tried.map(t => [key(t), t]));
    let differ = 0, stretches = new Set();
    for (const t of gq.tried){
      const u = floatByKey.get(key(t));
      if (t.refused){ differ++; continue; }
      stretches.add(N(t.rStretch));
      const verdict = DY.cmp(t.err, DY.fromRational(1n, 100n)) <= 0;
      if (!u || Math.abs(N(t.rStretch) - u.rStretch) > 1e-15 || t.stepsPerPeriod !== u.stepsPerPeriod || t.layersOk !== u.layersOk
          || verdict !== (u.err <= 1e-2) || Math.abs(N(t.err) - u.err) > 1e-7*u.err) differ++;
    }
    ok(gq.tried.length === gf.tried.length && differ === 0,
       `and all ${gq.tried.length} candidate grids carry the float sizing's radial stretch (${[...stretches].map(x => x.toFixed(1)).join(', ')}), step count, layer verdict and `
       + `surface-operator verdict, with the operator error within 1e-7 of the float value (${differ} differ)`);
  }
  const toD = x => { if (x === 0) return DY.ZERO; let mm = Math.abs(x), e2 = 0; while (!Number.isInteger(mm)){ mm *= 2; e2++; } return DY.store(BigInt(Math.sign(x)*mm), -e2, 1n); };
  const kf = K.jpZeroNear(12, 14)/12.125e-3;
  let worst = 0;
  for (const nr of [40, 512]){
    const S = new D.FaradayDisc({ m: 12, nr, nz: 12, R: 12.125e-3, h: 3e-3, rho: 998.2, nu: 1e-6, gamma: 0.0727, contact: 'free', rStretch: 2.2, zStretch: 2.2 });
    for (let i = 0; i < nr; i++) S.eta[i] = K.besselJ(12, kf*S.rc[i]);
    const rad = { rf: Array.from(S.rf, toD), rc: Array.from(S.rc, toD), drc: Array.from(S.drc, toD), drf: Array.from(S.drf, toD) };
    const q = E.surfaceOperatorError(rad, 12, 'free', Array.from(S.eta, toD), toD(kf)), f = S.surfaceOperatorError(kf);
    worst = Math.max(worst, Math.abs(N(q) - f)/f);
  }
  ok(worst < 1e-12, `on the float sizing's own grid and profile, at 40 and 512 cells, the operator error is the float one to ${worst.toExponential(2)}: `
     + `the spread above, up to 3e-8 at 512 cells, comes from the float grid and its 27-ulp Bessel profile, not from the formula`);
}

console.log('6. refusals');
{
  const refuses = (fn, re) => { try { fn(); return false; } catch (e){ return re.test(e.message); } };
  ok(refuses(() => B.besselJ(2, 1.5), /float is not accepted/), 'a float argument is refused');
  ok(refuses(() => B.besselJ(-1, DY.ONE), /non-negative whole order/), 'a negative order is refused');
  ok(refuses(() => E.suggestGrid({ m: 2, k: '100', R: '0.01', h: '0.003', rho: '1000', nu: '1e-6', gamma: '0.07', contact: 'free', omegaResponse: '300', omegaDrive: '600', profile: 3 }), /profile must be a function/),
     'sizing without a profile function is refused');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
