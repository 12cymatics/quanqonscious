/* Tests for the Faraday kernel embedded in cymatic.html.

   The kernel is not duplicated here. This file slices the region between the
   FARADAY KERNEL BEGIN and END markers straight out of cymatic.html and
   evaluates it, so there is exactly one copy of the physics and no second
   copy that can drift away from the page.

   Every expected value in faraday/reference.json was computed by mpmath,
   scipy.special or scipy.optimize.brentq -- never by running the code under
   test. Where a value is fixed by an identity rather than a table (a
   round-trip, a threshold crossing, a conservation), the identity is asserted
   directly.

   Run: node faraday/check-kernel.mjs
*/
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const HTML = readFileSync(join(ROOT, 'cymatic.html'), 'utf8');
const REF  = JSON.parse(readFileSync(join(here, 'reference.json'), 'utf8'));

/* ---- slice the kernel out of the page ---------------------------------- */
const BEGIN = 'FARADAY KERNEL BEGIN';
const END   = 'FARADAY KERNEL END';
const bi = HTML.indexOf(BEGIN), ei = HTML.indexOf(END);
if (bi < 0 || ei < 0 || ei <= bi)
  throw new Error(`kernel markers missing or out of order in cymatic.html (begin=${bi}, end=${ei})`);
// start after the comment that carries the BEGIN marker, end before the one
// that carries END
const src = HTML.slice(HTML.indexOf('*/', bi) + 2, HTML.lastIndexOf('/*', ei));
if (src.length < 5000)
  throw new Error(`sliced kernel is only ${src.length} chars — the markers are not bracketing the physics`);

const EXPORTS = ['millerAll','besselJ','besselJp','besselPair','jpZeroNear','radialIndexOf',
  'jpZerosNearN','radialProfile','simpsonR','angularQuartic','angularSquare','jpZeros',
  'surfaceTension','density','viscosity','omegaOf','kOfOmega','dampingRate','plateTransfer',
  'mathieu','mathieuKT','pinnedEdgeSpectrum','pinnedEdgeExtrapolated','ellipseE','reinforcedR4','boundaryImpedance','atlasAt','foldPrior',
  'transitionRisk','resolvePatternState','CELLS','ATLAS','TRANSITION','BOUNDARY',
  'EGG_I_P','EGG_II_P','G_ACC','QUAD_R','STONE','MIN_MV','OVERDRIVE_MV'];

const sandbox = { Math, Number, Map, Set, Array, Float64Array, Float32Array, Int32Array,
                  isFinite, isNaN, console, JSON, Object, String, Error, TypeError };
vm.createContext(sandbox);
vm.runInContext(src + '\n;globalThis.__K = {' + EXPORTS.join(',') + '};', sandbox,
                { filename: 'cymatic.html#kernel' });
const K = sandbox.__K;
for (const name of EXPORTS)
  if (K[name] === undefined) throw new Error(`kernel export ${name} is undefined after evaluation`);

/* ---- harness ----------------------------------------------------------- */
let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond) { pass++; }
  else failures.push(`${label}${detail ? '  — ' + detail : ''}`);
}
// relative agreement against an independently computed reference. This is not
// a tolerance standing in for a wrong answer: it is the width of binary64
// itself, and each call states the decades it demands.
function rel(got, want, decades, label){
  if (!isFinite(got)) return ok(false, label, `got ${got}`);
  const denom = Math.abs(want) > 0 ? Math.abs(want) : 1;
  const err = Math.abs(got - want)/denom;
  ok(err <= Math.pow(10, -decades), label,
     `got ${got}, want ${want}, relative error ${err.toExponential(3)} > 1e-${decades}`);
}
function eq(got, want, label){
  ok(Object.is(got, want), label, `got ${got}, want ${want}`);
}
function section(name){ console.log('\n' + name); }

/* ── 1. Bessel functions ────────────────────────────────────────────────── */
section('1. Bessel J_m and J\'_m against mpmath');
for (const { m, z, J } of REF.besselJ){
  // absolute floor for the deeply evanescent entries: J_170(100) = 2e-25 is
  // below binary64's relative reach next to the O(1) terms in the recurrence,
  // so those are held to absolute agreement instead, which is the real claim.
  const got = K.besselJ(m, z);
  if (Math.abs(J) < 1e-12)
    ok(Math.abs(got - J) < 1e-15, `J_${m}(${z}) = ${J.toExponential(3)}`,
       `got ${got.toExponential(3)}, absolute error ${Math.abs(got-J).toExponential(3)}`);
  else rel(got, J, 12, `J_${m}(${z})`);
}
for (const { m, z, Jp } of REF.besselJp) rel(K.besselJp(m, z), Jp, 10, `J'_${m}(${z})`);

/* besselPair returns [J, J'] from one recurrence. The value that matters is
   its agreement with mpmath, so that is what is asserted; the cross-check
   against the two scalar calls is held to 14 decades rather than to the bit,
   because besselJ starts Miller at order m and besselPair at m+1. Same
   method, one order apart, so the round-off accumulates differently -- at
   m=150, z=182 they part company at the sixth ulp. Demanding bit-equality
   there would be asserting something that is not true of the code and need
   not be. */
for (const { m, z, J } of REF.besselJ){
  const p = K.besselPair(m, z);
  ok(Array.isArray(p) && p.length === 2, `besselPair(${m},${z}) returns a pair`,
     `got ${JSON.stringify(p)}`);
  if (Math.abs(J) < 1e-12)
    ok(Math.abs(p[0] - J) < 1e-15, `besselPair J at m=${m}, z=${z} (evanescent)`,
       `got ${p[0].toExponential(3)}, want ${J.toExponential(3)}`);
  else rel(p[0], J, 12, `besselPair J against mpmath at m=${m}, z=${z}`);
}
for (const { m, z, Jp } of REF.besselJp)
  rel(K.besselPair(m, z)[1], Jp, 10, `besselPair J' against mpmath at m=${m}, z=${z}`);
for (const [m, z] of [[0,1],[5,25],[37,40],[150,182]]){
  const p = K.besselPair(m, z);
  rel(p[0], K.besselJ(m, z),  14, `besselPair J tracks besselJ at m=${m}`);
  rel(p[1], K.besselJp(m, z), 14, `besselPair J' tracks besselJp at m=${m}`);
}
// at z = 0 the pair is exact and known: J_m(0) = [m==0], J'_m(0) = 1/2 at m=1
eq(K.besselPair(0,0)[0], 1, 'J_0(0) = 1');
eq(K.besselPair(3,0)[0], 0, 'J_3(0) = 0');
eq(K.besselPair(1,0)[1], 0.5, "J'_1(0) = 1/2");
eq(K.besselPair(4,0)[1], 0,   "J'_4(0) = 0");

/* ── 2. zeros of J'_m ───────────────────────────────────────────────────── */
section("2. zeros of J'_m against scipy.special.jnp_zeros");
for (const { m, zeros } of REF.jpZeros){
  const got = K.jpZeros(m, zeros.length, zeros[zeros.length-1]);
  ok(got.length === zeros.length, `jpZeros(${m}) returns ${zeros.length}`,
     `got ${got.length}`);
  for (let i = 0; i < Math.min(got.length, zeros.length); i++)
    rel(got[i], zeros[i], 9, `j'_{${m},${i+1}}`);
  // and each returned zero really is one
  for (const z of got)
    ok(Math.abs(K.besselJp(m, z)) < 1e-9, `J'_${m} vanishes at returned zero ${z.toFixed(6)}`,
       `J' = ${K.besselJp(m, z).toExponential(3)}`);
}
/* jpZerosNearN is the root finder the renderer actually calls -- jpZeros is
   defined in the page but has no call site, so verifying only jpZeros would
   have left the live path covered nowhere but indirectly. Both are checked.
   jpZerosNearN returns the `count` roots nearest a target argument, so the
   expected set is the scipy roots sorted by distance from that target. */
for (const { m, zeros } of REF.jpZeros){
  for (const [target, count] of [[zeros[0], 3], [zeros[3], 3], [zeros[2], 1]]){
    const got = K.jpZerosNearN(m, target, count).slice().sort((a,b) => a-b);
    const want = zeros.slice()
      .sort((a,b) => Math.abs(a-target) - Math.abs(b-target))
      .slice(0, count).sort((a,b) => a-b);
    ok(got.length === count, `jpZerosNearN(${m}, ${target.toFixed(3)}, ${count}) returns ${count}`,
       `got ${got.length}`);
    for (let i = 0; i < Math.min(got.length, want.length); i++)
      rel(got[i], want[i], 9, `jpZerosNearN m=${m} near ${target.toFixed(2)}, root ${i+1}`);
    for (const z of got)
      ok(Math.abs(K.besselJp(m, z)) < 1e-9,
         `jpZerosNearN m=${m} returned a true zero at ${z.toFixed(6)}`,
         `J' = ${K.besselJp(m, z).toExponential(3)}`);
  }
  // it must never return a duplicate: three roots means three distinct roots
  const trio = K.jpZerosNearN(m, zeros[2], 3);
  ok(new Set(trio.map(z => z.toFixed(6))).size === trio.length,
     `jpZerosNearN(${m}) returns distinct roots`, `got ${trio.map(z=>z.toFixed(4)).join(',')}`);
}

// Whatever jpZeroNear returns must BE a zero. The old bare-Newton solver
// escaped at an inflection, hit its guard, and returned the FLOOR m/2 as
// though it were a root -- J'_142 has no zero at 71. Those phantom modes had
// small k, hence little damping, so they ranked highest by growth and
// dominated the set. The target below is that same 71.21; the bracketed
// finder walks out to the true first zero at 146.232174 instead.
{
  const z = K.jpZeroNear(142, 71.21);
  ok(z !== null, 'jpZeroNear(142, 71.21) returns a root', `got ${z}`);
  ok(Math.abs(z - 71) > 1, 'it is not the m/2 floor the old solver returned',
     `got ${z}`);
  rel(z, REF.jpZeros.find(x => x.m === 142).zeros[0], 9,
      "jpZeroNear(142, 71.21) walks out to j'_{142,1}");
  ok(Math.abs(K.besselJp(142, z)) < 1e-9, 'and it really is a zero',
     `J' = ${K.besselJp(142, z).toExponential(3)}`);
}
for (const [m, t] of [[37,40],[0,10],[150,183],[5,25]]){
  const z = K.jpZeroNear(m, t);
  ok(z !== null && Math.abs(K.besselJp(m, z)) < 1e-9,
     `jpZeroNear(${m}, ${t}) returns a true zero`, `got ${z}`);
}

/* ── 2b. radial index ───────────────────────────────────────────────────── */
section("2b. radial index against scipy.special.jnp_zeros");
for (const { m, n, z } of REF.radialIndex)
  eq(K.radialIndexOf(m, z), n, `radialIndexOf(${m}, j'_{${m},${n}}) = ${n}`);
// monotone: a larger argument can never have a smaller index
for (const m of [0, 12, 37, 150]){
  const zs = REF.radialIndex.filter(r => r.m === m).map(r => r.z);
  for (let i = 1; i < zs.length; i++)
    ok(K.radialIndexOf(m, zs[i]) > K.radialIndexOf(m, zs[i-1]),
       `radialIndexOf is strictly increasing at m=${m}, root ${i+1}`,
       `${K.radialIndexOf(m, zs[i-1])} then ${K.radialIndexOf(m, zs[i])}`);
}
// and strictly between two roots it holds the lower index
for (const m of [0, 12, 150]){
  const zs = REF.radialIndex.filter(r => r.m === m).map(r => r.z);
  const mid = (zs[2] + zs[3])/2;
  eq(K.radialIndexOf(m, mid), 3, `radialIndexOf(${m}, midway between roots 3 and 4) = 3`);
}

/* ── 3. quadrature ──────────────────────────────────────────────────────── */
section('3. angular and radial quadrature');
for (const { a, b, I } of REF.angularQuartic)
  rel(K.angularQuartic(a, b), I, 12, `INT cos^2(${a}t)cos^2(${b}t)`);
for (const { a, I } of REF.angularSquare)
  rel(K.angularSquare(a), I, 12, `INT cos^2(${a}t)`);
// the defect that motivated this file: the two nonzero branches must sit in
// the same normalisation as the branches with a zero order.  cos^2(0)=1, so
// angularQuartic(0,b) must equal angularSquare(b) identically.
for (const b of [1, 3, 12])
  rel(K.angularQuartic(0, b), K.angularSquare(b), 15,
      `angularQuartic(0,${b}) == angularSquare(${b})`);
// ...and a self-overlap must be strictly below the square, never above it:
// INT cos^4 < INT cos^2 for every nonzero order.
for (const a of [1, 3, 7, 12])
  ok(K.angularQuartic(a, a) < K.angularSquare(a),
     `angularQuartic(${a},${a}) < angularSquare(${a})`,
     `${K.angularQuartic(a,a)} vs ${K.angularSquare(a)}`);
// distinct orders: exactly two thirds of the equal-order value
for (const [a,b] of [[2,5],[1,2],[4,9]])
  rel(K.angularQuartic(a,b)/K.angularQuartic(a,a), 2/3, 14,
      `angularQuartic(${a},${b}) is 2/3 of angularQuartic(${a},${a})`);

{
  const N = K.QUAD_R;
  ok(N % 2 === 0, `QUAD_R = ${N} is even (Simpson requires it)`);
  const build = fn => { const v = new Float64Array(N+1);
    for (let i = 0; i <= N; i++) v[i] = fn(i/N); return v; };
  rel(K.simpsonR(build(() => 1)),      REF.simpsonR[0].exact, 13, 'INT_0^1 1 * r dr = 1/2');
  rel(K.simpsonR(build(r => r)),       REF.simpsonR[1].exact, 13, 'INT_0^1 r * r dr = 1/3');
  rel(K.simpsonR(build(r => r*r)),     REF.simpsonR[2].exact, 13, 'INT_0^1 r^2 * r dr = 1/4');
  // INT_0^1 J0(a r) r dr = J1(a)/a, and a = j'_{0,1} is a zero of J'_0 = -J1,
  // so this integral is EXACTLY zero. A relative comparison of two numbers
  // that are both ~1e-16 says nothing; the claim is absolute.
  const j01 = REF.jpZeros.find(x => x.m === 0).zeros[0];
  const zeroInt = K.simpsonR(build(r => K.besselJ(0, j01*r)));
  ok(Math.abs(zeroInt) < 1e-12, "INT_0^1 J0(j'_{0,1} r) r dr is zero",
     `got ${zeroInt.toExponential(3)}`);
  const j32 = REF.jpZeros.find(x => x.m === 3).zeros[1];
  rel(K.simpsonR(build(r => K.besselJ(3, j32*r)**2)), REF.simpsonR[4].exact, 8,
      "INT_0^1 J3(j'_{3,2} r)^2 r dr");
}

/* ── 4. water properties ────────────────────────────────────────────────── */
section('4. water properties');
for (const { T, sigma, rho, mu } of REF.water){
  rel(K.surfaceTension(T), sigma, 13, `sigma(${T} C)`);
  rel(K.density(T),        rho,   13, `rho(${T} C)`);
  rel(K.viscosity(T),      mu,    13, `mu(${T} C)`);
}
// density must peak near 4 C -- the property the Kell form exists to capture
ok(K.density(4) > K.density(0) && K.density(4) > K.density(8),
   'density has its maximum near 4 C',
   `rho(0)=${K.density(0)}, rho(4)=${K.density(4)}, rho(8)=${K.density(8)}`);

/* ── 5. dispersion ──────────────────────────────────────────────────────── */
section('5. gravity-capillary dispersion on finite depth');
for (const { f, T, h_mm, k, lambda } of REF.dispersion){
  const s = K.surfaceTension(T), r = K.density(T), h = h_mm/1000;
  const w = 2*Math.PI*(f/2);
  rel(K.kOfOmega(w, s, r, h), k, 9, `k at ${f} Hz, ${T} C, ${h_mm} mm`);
  rel(2*Math.PI/K.kOfOmega(w, s, r, h), lambda, 9, `wavelength at ${f} Hz`);
  // round trip: omega(k(omega)) must return omega
  rel(K.omegaOf(K.kOfOmega(w, s, r, h), s, r, h), w, 11, `omega round-trip at ${f} Hz`);
}
// the two asymptotic limits must emerge, not be assumed: deep-water gravity
// at long wave, capillary at short.
{
  const s = K.surfaceTension(20), r = K.density(20);
  const kLong = 5;      // 1.26 m wave, deep relative to 1 m
  rel(K.omegaOf(kLong, s, r, 1)**2/(K.G_ACC*kLong*Math.tanh(kLong*1)), 1, 3,
      'long-wave limit is gravity-dominated');
  const kShort = 2e4;   // 314 um
  const ratio = (s*kShort**3/r)/(K.G_ACC*kShort);
  rel(ratio, REF.capillaryRatio[0].ratio, 10,
      'capillary-to-gravity ratio at k = 2e4 matches the independent value');
  ok(ratio > 1e3, 'short-wave limit is capillary-dominated by three decades',
     `sigma k^2/(rho g) = ${ratio.toExponential(3)}`);
}

/* ── 6. damping ─────────────────────────────────────────────────────────── */
section('6. viscous damping');
{
  const nu = K.viscosity(20)/K.density(20);
  const k = 1156.359261, w = 2*Math.PI*55.5, h = 0.002;
  const d = K.dampingRate(k, w, nu, h);
  rel(d.bulk, 2*nu*k*k, 15, 'bulk term is 2 nu k^2');
  rel(d.stokesDepth, Math.sqrt(2*nu/w), 15, 'Stokes depth is sqrt(2 nu/omega)');
  rel(d.layer, w*(k*d.stokesDepth)/(2*Math.sinh(2*k*h)), 14, 'layer term matches its formula');
  rel(d.total, d.bulk + d.layer, 15, 'total is the sum of both terms');
  ok(d.layer > 0, 'the bottom layer term is kept, not dropped', `layer = ${d.layer}`);
  // the layer term must die as kh grows -- a result, not an assumption
  const deep = K.dampingRate(k, w, nu, 0.5);
  ok(deep.layer < d.layer*1e-6, 'layer term vanishes in deep water',
     `shallow ${d.layer.toExponential(3)} vs deep ${deep.layer.toExponential(3)}`);
  rel(deep.total, deep.bulk, 6, 'deep water is bulk-damped');
}

/* ── 7. Mathieu tongue ──────────────────────────────────────────────────── */
section('7. damped Mathieu, first tongue');
{
  const w0 = 2*Math.PI*55.5, gamma = 3.0, k = 1156.36, h = 0.002, wd = 2*w0;
  const at = 4*gamma*w0/(k*Math.tanh(k*h));
  const m = K.mathieu(w0, gamma, at, k, h, wd);
  rel(m.accelThreshold, at, 13, 'accelThreshold inverts epsThreshold');
  rel(m.eps, m.epsThreshold, 12, 'at threshold acceleration, eps = eps_c');
  ok(Math.abs(m.growth) < 1e-9, 'growth is exactly zero at threshold', `got ${m.growth}`);
  eq(m.detune, 0, 'zero detuning at omega_d = 2 omega_0');
  // and it must cross: below threshold negative, above positive
  ok(K.mathieu(w0, gamma, at*0.9, k, h, wd).growth < 0, 'sub-threshold drive decays');
  ok(K.mathieu(w0, gamma, at*1.5, k, h, wd).growth > 0, 'super-threshold drive grows');
  // supercritical growth must keep rising with drive -- not be clamped back to
  // the near-onset value
  const g2 = K.mathieu(w0, gamma, at*2,  k, h, wd).growth;
  const g8 = K.mathieu(w0, gamma, at*8,  k, h, wd).growth;
  ok(g8 > g2*3, 'growth keeps increasing far above onset', `g(2x)=${g2}, g(8x)=${g8}`);
  // detuning off the tongue must kill it
  ok(K.mathieu(w0, gamma, at*1.5, k, h, wd*1.5).growth <= 0,
     'a drive detuned off the tongue does not grow');
}

/* ── 7b. pinned contact line: the edge-constrained spectrum ─────────────── */
section('7b. pinned (edge-constrained) spectrum');
for (const cse of REF.pinnedEdge){
  const { m, diameterMm, tempC, depthMm, basisN } = cse;
  const R = diameterMm/2000, sg = K.surfaceTension(tempC), rh = K.density(tempC);
  const h = depthMm/1000;
  const tag = `m=${m}, D=${diameterMm}mm, ${tempC}C, ${depthMm}mm`;
  const raw = K.pinnedEdgeSpectrum(m, R, sg, rh, h, basisN, 6);

  // the free spectrum this is built on
  for (let i = 0; i < Math.min(6, cse.freeHz.length); i++)
    rel(Math.sqrt(raw.freeW2[i])/(2*Math.PI), cse.freeHz[i], 9, `${tag}: free mode ${i+1}`);
  // c_n = 2/(R^2(1 - (m/z_n)^2)) -- the Neumann norm cancels J_m(z_n)^2 outright
  for (let i = 0; i < Math.min(6, cse.c.length); i++){
    rel(raw.c[i], cse.c[i], 12, `${tag}: c_${i+1}`);
    ok(raw.c[i] > 0, `${tag}: c_${i+1} is positive (the interlacing depends on it)`);
  }

  const ex = K.pinnedEdgeExtrapolated(m, R, sg, rh, h, basisN, 6);
  ok(ex.extrapolated === true, `${tag}: result is marked extrapolated`);
  eq(ex.basisPair.join(','), `${basisN/2},${basisN}`, `${tag}: extrapolated from N/2 and N`);
  for (let i = 0; i < Math.min(cse.pinned.length, ex.pinned.length); i++){
    const got = ex.pinned[i], want = cse.pinned[i];
    rel(got.hz,          want.hz,          9, `${tag}: pinned mode ${i+1}`);
    rel(got.hzTruncated, want.hzTruncated, 9, `${tag}: pinned mode ${i+1} before extrapolation`);
    rel(got.kTanhEff,    want.kTanhEff,    9, `${tag}: pinned mode ${i+1} modal k tanh(kh)`);
    // extrapolation must MOVE the answer and move it the right way: the
    // truncated root descends to the limit, so the step is negative
    ok(got.richardsonStep < 0,
       `${tag}: mode ${i+1} converges from above`, `step ${got.richardsonStep}`);
    ok(got.hz < got.hzTruncated,
       `${tag}: extrapolation lowers mode ${i+1} toward the limit`);
  }
  /* The extrapolant must be near the converged value, not merely
     self-consistent, so it is compared against an N=1024 extrapolant computed
     in scipy. Two claims, both measured rather than picked:

       - the envelope. Across these eight cases the worst extrapolated error is
         6.55e-6 (m=10, mode 3) and the worst TRUNCATED error is 7.36e-5, so
         1e-5 is the envelope the method actually holds to at N=128. It is not
         a tolerance standing in for a wrong answer; raising the basis tightens
         it, on the N^-2 rate the extrapolation is built on.
       - the gain, which is the real claim. Extrapolation must beat the
         truncated root by at least a factor of ten in EVERY case. Measured:
         184x at m=1 falling to 11.2x at m=10, because the C/N^2 constant grows
         with the angular order. That fall-off is why the bound above is 1e-5
         and not 1e-7, and why high m wants a larger basis. */
  for (let i = 0; i < Math.min(cse.convergedHz.length, ex.pinned.length); i++){
    const lim = cse.convergedHz[i];
    const eEx  = Math.abs(ex.pinned[i].hz - lim)/lim;
    const eRaw = Math.abs(ex.pinned[i].hzTruncated - lim)/lim;
    ok(eEx < 1e-5, `${tag}: mode ${i+1} is within 1e-5 of the N=1024 limit`,
       `got ${ex.pinned[i].hz}, limit ${lim}, rel ${eEx.toExponential(2)}`);
    ok(eRaw/eEx > 10,
       `${tag}: extrapolation beats the truncated root tenfold on mode ${i+1}`,
       `truncated ${eRaw.toExponential(2)}, extrapolated ${eEx.toExponential(2)}, ` +
       `gain ${(eRaw/eEx).toFixed(1)}x`);
  }
  // Rayleigh: a constraint raises every eigenvalue and the result strictly
  // interlaces the unconstrained spectrum. This is the structural property the
  // whole construction stands on, so it is asserted, not assumed.
  for (let i = 0; i < Math.min(5, raw.pinned.length); i++){
    const f0 = Math.sqrt(raw.freeW2[i])/(2*Math.PI);
    const f1 = Math.sqrt(raw.freeW2[i+1])/(2*Math.PI);
    ok(raw.pinned[i].hz > f0 && raw.pinned[i].hz < f1,
       `${tag}: pinned mode ${i+1} lies strictly between free modes ${i+1} and ${i+2}`,
       `${f0} < ${raw.pinned[i].hz} < ${f1}`);
  }
  // eta(R) = 0 -- the constraint the whole thing exists to impose. It holds by
  // construction (the wall sum IS the secular function), so it must hold to
  // machine precision against the largest single term, not merely be small.
  for (const pm of raw.pinned.slice(0, 4))
    ok(Math.abs(pm.wallResidual) < 1e-12*pm.wallScale,
       `${tag}: eta(R)=0 at ${pm.hz.toFixed(4)} Hz`,
       `residual ${pm.wallResidual.toExponential(3)} against largest term ` +
       `${pm.wallScale.toExponential(3)} — ratio ` +
       `${(Math.abs(pm.wallResidual)/pm.wallScale).toExponential(2)}`);
  // the modal projection of k tanh(kh) must lie inside the basis range it
  // averages, or it is not a weighted mean of anything
  for (const pm of raw.pinned.slice(0, 3)){
    const kt = raw.freeK.map(kk => kk*Math.tanh(kk*h));
    ok(pm.kTanhEff > Math.min(...kt) && pm.kTanhEff < Math.max(...kt),
       `${tag}: modal k tanh(kh) at ${pm.hz.toFixed(3)} Hz is inside the basis range`,
       `${pm.kTanhEff}`);
  }
}
/* The truncation guard. What it protects against is real and is demonstrated
   here directly: jpZeros with a hint of 0 scans only to x=40 and returns about
   a dozen zeros of J'_0 rather than the 64 asked for, and a secular sum over
   that basis converges to the wrong roots instead of failing. The guard itself
   is defensive and NOT reachable through pinnedEdgeSpectrum, because that
   function computes its own hint (m + 3.2N) and the hint is adequate -- which
   is the property asserted below, for every basis the suite uses. Forcing the
   guard to fire needs a basis so large the scan runs to x ~ 27000, which is
   too slow to sit in this suite; it is kept as insurance and labelled as such
   rather than claimed to be a tested gate. */
{
  const starved = K.jpZeros(0, 64, 0);
  ok(starved.length < 64,
     'a zero hint really does starve the basis (what the guard exists for)',
     `asked 64, got ${starved.length}`);
  for (const cse of REF.pinnedEdge){
    const full = K.jpZeros(cse.m, cse.basisN, cse.m + 3.2*cse.basisN);
    eq(full.length, cse.basisN,
       `the computed hint supplies a full basis at m=${cse.m}, N=${cse.basisN}`);
    const half = K.jpZeros(cse.m, cse.basisN/2, cse.m + 3.2*cse.basisN/2);
    eq(half.length, cse.basisN/2,
       `and at the half basis used for extrapolation, m=${cse.m}`);
  }
}
// mathieuKT is the same analysis the free path runs, expressed in k tanh(kh)
for (const [w0, g, acc, k, h, wd] of [[300, 3, 8, 1156, 0.002, 600], [900, 5, 40, 400, 0.005, 1800]]){
  const a = K.mathieu(w0, g, acc, k, h, wd);
  const b = K.mathieuKT(w0, g, acc, k*Math.tanh(k*h), wd);
  rel(b.growth, a.growth, 15, 'mathieuKT reproduces mathieu exactly (growth)');
  rel(b.eps, a.eps, 15, 'mathieuKT reproduces mathieu exactly (eps)');
  rel(b.accelThreshold, a.accelThreshold, 15, 'mathieuKT reproduces mathieu exactly (threshold)');
}

/* ── 8. fluid-loaded plate ──────────────────────────────────────────────── */
section('8. fluid-loaded Kirchhoff-Love base');
{
  const k = 1156.36, w = 2*Math.PI*111, rhoW = K.density(20), h = 0.002;
  const p = K.plateTransfer(k, w, rhoW, h);
  const D = K.STONE.E*K.STONE.thickness**3/(12*(1 - K.STONE.poisson**2));
  rel(p.D, D, 15, 'D = E t^3 / (12(1-nu^2))');
  rel(p.added, rhoW/(k*Math.tanh(k*h)), 15, 'added mass is rho_w/(k tanh kh)');
  rel(p.stiff, D*k**4, 14, 'stiffness is D k^4');
  rel(p.inert, w*w*(K.STONE.density*K.STONE.thickness + p.added), 14,
      'inertia carries plate mass plus added mass');
  // The magnitude is fully determined, so it is pinned rather than merely
  // checked for being finite. The loss term was untested until now.
  const loss = 2*K.STONE.lossFactor*Math.sqrt(Math.abs(p.stiff*p.inert));
  rel(p.magnitude, 1/Math.hypot(p.stiff - p.inert, loss), 14,
      'magnitude is 1/hypot(stiff - inert, loss) with the hysteretic loss term');
  ok(p.magnitude > 0 && isFinite(p.magnitude), 'transfer magnitude is finite and positive');
  // and the loss term must be load-bearing: with zero damping the transfer at
  // resonance would be singular, so it cannot be quietly absent
  ok(loss > 0, 'the hysteretic loss term is nonzero', `loss = ${loss}`);
  // At this k the base is stiffness-controlled -- D k^4 exceeds the inertia by
  // nine decades -- so fluid loading is invisible HERE. That is a result, not
  // a reason to skip the check: it is tested at the wavenumber where the two
  // balance, which is where the transfer actually resonates.
  ok(p.stiff/p.inert > 1e6, 'at kR ~ 14 the base is stiffness-controlled',
     `stiff/inert = ${(p.stiff/p.inert).toExponential(2)}`);
  {
    // solve D k^4 = w^2 (rho_s t + rho_w/(k tanh kh)) for w at fixed k
    const kr = 200, rw = K.density(20), hh = 0.002;
    const add = rw/(kr*Math.tanh(kr*hh));
    const wRes = Math.sqrt(D*kr**4/(K.STONE.density*K.STONE.thickness + add));
    const at  = K.plateTransfer(kr, wRes,      rw, hh);
    const off = K.plateTransfer(kr, wRes*1.35, rw, hh);
    rel(at.stiff, at.inert, 12, 'at the plate resonance, stiffness equals inertia');
    ok(at.magnitude > off.magnitude*3,
       'the transfer peaks at the plate resonance',
       `on ${at.magnitude.toExponential(3)} vs off ${off.magnitude.toExponential(3)}`);
    // there, dropping the added mass moves the resonance measurably
    const wBare = Math.sqrt(D*kr**4/(K.STONE.density*K.STONE.thickness));
    ok(Math.abs(wBare - wRes)/wRes > 0.01,
       'fluid loading shifts the plate resonance by more than a percent',
       `${wRes.toFixed(2)} loaded vs ${wBare.toFixed(2)} dry`);
  }
}

/* ── 9. elliptic integral and boundary set ──────────────────────────────── */
section('9. elliptic integral and the ten walls');
for (const { m, E } of REF.ellipseE) rel(K.ellipseE(m), E, 12, `E(m=${m.toFixed(6)})`);
{
  ok(K.BOUNDARY.length === 10, 'ten walls are carried', `got ${K.BOUNDARY.length}`);
  const exact = K.BOUNDARY.filter(b => b.pd != null).length;
  const bounded = K.BOUNDARY.filter(b => b.pdRange).length;
  eq(exact, 8, 'eight walls have an exact P/D');
  eq(bounded, 2, 'two walls (the eggs) are carried as an interval');
  // the convexity bound that refuses Egg II's published divisor
  rel(K.EGG_II_P, 10*Math.PI + 10 - 8*Math.atan(5/4), 15, 'Egg II perimeter from its formula');
  ok(K.EGG_II_P/20 < 2, "Egg II's published divisor 20 violates P/D >= 2",
     `P/D = ${(K.EGG_II_P/20).toFixed(4)}`);
  ok(K.EGG_I_P/18 >= 2 && K.EGG_I_P/18 <= Math.PI,
     "Egg I's divisor 18 is inside [2, pi] but unsourced",
     `P/D = ${(K.EGG_I_P/18).toFixed(4)}`);
  const b = K.boundaryImpedance(0.37, 1.1);
  ok(b.impedanceLo <= b.impedance && b.impedance <= b.impedanceHi,
     'the impedance interval brackets its midpoint',
     `${b.impedanceLo} <= ${b.impedance} <= ${b.impedanceHi}`);
  ok(b.impedanceHi > b.impedanceLo, 'the eggs make it a genuine interval, not a point',
     `width ${(b.impedanceHi - b.impedanceLo).toExponential(3)}`);
  eq(b.walls, 10, 'all ten walls entered the product');
  eq(b.exact, 8, 'eight entered exactly');
  // R4 reinforcement is a product of Lorentzians, so strictly decreasing in r
  ok(K.reinforcedR4(0.5) > K.reinforcedR4(1.5) && K.reinforcedR4(1.5) > K.reinforcedR4(3.0),
     'R4 reinforcement decreases with source distance');
  rel(K.reinforcedR4(0), 1, 15, 'R4 reinforcement is 1 at the source');
}

/* ── 10. atlas folding ──────────────────────────────────────────────────── */
section('10. atlas priors');
{
  // 56 Hz medium is unanimous 6/6/6 -- confidence 1, no competitor
  const r56 = K.atlasAt('medium', 56);
  ok(r56 !== null, 'atlas has 56 Hz for the medium cell');
  const p56 = K.foldPrior(r56);
  eq(p56.fold, 6, '56 Hz medium folds to 6');
  rel(p56.confidence, 1, 15, '56 Hz confidence is exactly 1 (unanimous)');
  eq(p56.competing.length, 0, '56 Hz has no competing state');
  // 115 Hz medium is 10/16/10 -- 2 of 3, with 16 competing at 1/3
  const p115 = K.foldPrior(K.atlasAt('medium', 115));
  eq(p115.fold, 10, '115 Hz medium folds to 10');
  rel(p115.confidence, 2/3, 14, '115 Hz confidence is 2/3');
  eq(p115.competing.length, 1, '115 Hz has one competing state');
  eq(p115.competing[0].fold, 16, '115 Hz competitor is 16');
  // a split label 2/4 must split its weight, not count twice
  const p148 = K.foldPrior(K.atlasAt('small', 148));
  ok(p148.confidence <= 0.5 + 1e-12, 'a lone 2/4 split label gives at most half weight',
     `confidence ${p148.confidence}`);
  ok(K.transitionRisk(p148, K.atlasAt('small', 148), 148) >= 0.5,
     'a split label raises transition risk to at least 0.5');
  // unclassified must report as unclassified, not as a fold
  const p52 = K.foldPrior(K.atlasAt('medium', 52));
  ok(p52.unclassified === true, '52 Hz medium is unanimous-unclassified');
  eq(K.transitionRisk(p52, K.atlasAt('medium', 52), 52), 1, 'unclassified carries full risk');
  // confidence is over ALL replicates including unclassified ones
  const p50 = K.foldPrior(K.atlasAt('medium', 50));   // ["8","unclassified","unclassified"]
  rel(p50.confidence, 1/3, 14, '50 Hz confidence counts the unclassified replicates');
}

/* ── 11. the assembled state ────────────────────────────────────────────── */
section('11. resolvePatternState end to end');
function state(over = {}){
  return K.resolvePatternState({ cellKey:'medium', f:111, amplitudeMv:200, tempC:20,
    depthMm:2, rim:'free', tSec:0, tfeSec:0, driveR:0, ...over });
}
{
  const s = state();
  rel(s.wavenumber, REF.dispersion.find(d => d.f === 111).k, 8,
      'assembled wavenumber matches brentq');
  rel(s.wavelength, 2*Math.PI/s.wavenumber, 15, 'wavelength is 2pi/k');
  rel(s.responseHz, 111/2, 15, 'the response is subharmonic');
  // The lineage spec's 48 is sixteen angular orders by three radial roots --
  // a CAP, not a count that can always be reached. At 111 Hz the rim argument
  // is kR = 14.02, so m only runs 0..14: fifteen orders exist and forty-five
  // modes is the whole field, not a truncation of it.
  eq(s.radialRoots, 3, 'three radial roots per order');
  const mMax = Math.floor(s.wavenumber*(K.CELLS.medium.d/2000));
  eq(mMax, 14, 'kR = 14.02 at 111 Hz, so m runs 0..14');
  eq(s.angularOrders, 15, 'all fifteen available angular orders are kept at 111 Hz');
  ok(s.angularOrders <= 16, 'the cap of sixteen is respected');
  eq(s.ordersScanned, s.angularOrders*s.radialRoots,
     'the assembled field is orders x roots with no root dropped');
  ok(s.nearestTheoryModes.length === 4, 'four nearest theory modes reported');
  // sorted by growth, descending -- the defect that once made the "strongest"
  // modes simply the lowest angular orders present
  for (let i = 1; i < s.nearestTheoryModes.length; i++)
    ok(s.nearestTheoryModes[i-1].growth >= s.nearestTheoryModes[i].growth - 1e-12,
       `theory modes sorted by growth at index ${i}`,
       `${s.nearestTheoryModes[i-1].growth} then ${s.nearestTheoryModes[i].growth}`);
  // every reported mode must sit on a true zero of J', and its radial index
  // must be the honest count, never a McMahon inverse (which gives -16 at m=150)
  for (const t of s.nearestTheoryModes){
    ok(Math.abs(K.besselJp(t.m, t.jp)) < 1e-8,
       `mode m=${t.m} sits on a true J' zero`, `J' = ${K.besselJp(t.m, t.jp).toExponential(3)}`);
    ok(Number.isInteger(t.n) && t.n >= 1,
       `mode m=${t.m} has a positive integer radial index`, `n = ${t.n}`);
    eq(t.n, K.radialIndexOf(t.m, t.jp),
       `mode m=${t.m} reports the counted radial index`);
    rel(t.k, t.jp/(K.CELLS.medium.d/2000), 13, `mode m=${t.m}: k = j'/R`);
  }
  // competition covers the whole field and ranks what survives
  const c = s.competition;
  ok(c && c.size === s.ordersScanned, 'competition covers every assembled mode',
     `${c && c.size} vs ${s.ordersScanned}`);
  ok(c.survivors.length >= 1, 'at least one mode survives the competition');
  ok(c.survivors.length <= c.size, 'survivors do not exceed the field');
  eq(c.killed, c.size - c.survivors.length, 'killed count is consistent');
  for (let i = 1; i < c.survivors.length; i++)
    ok(c.survivors[i-1].amp >= c.survivors[i].amp, `survivors ranked by amplitude at ${i}`);
  for (const v of c.survivors) ok(v.amp > 0, `survivor m=${v.m} has positive amplitude`);
  ok(s.growingRetained === s.unstableCount,
     'every growing mode is retained, none collapsed back to onset');
  // the surface is one of two discrete phases -- never a slow envelope
  ok(s.phase === 1 || s.phase === -1 || Math.abs(Math.abs(s.phase) - 1) < 1e-9,
     'at t=0 the phase is a unit sign, not a fractional envelope', `phase = ${s.phase}`);
  /* The drawable state set at 111 Hz is fixed by the atlas, not by theory:
     the medium cell's 111 Hz record is ["10","10","2"], so 10-fold carries 2 of
     3 replicates and 2-fold carries 1. Both clear the 0.15 share floor, so two
     states are drawn, and the paper's convention makes fold = 2m. */
  eq(s.states.length, 2, '111 Hz draws exactly two states');
  eq(s.states.map(x => x.fold).join(','), '10,2', 'folds are 10 then 2');
  eq(s.states.map(x => x.m).join(','), '5,1', 'and m = fold/2 is 5 then 1');
  rel(s.states[0].weight, 2/3, 14, '10-fold carries 2 of 3 replicates');
  rel(s.states[1].weight, 1/3, 14, '2-fold carries 1 of 3');
  for (const st of s.states){
    ok(Math.abs(K.besselJp(st.m, st.radial.jp)) < 1e-8,
       `state fold=${st.fold} sits on a true J' zero`);
    eq(st.radial.n, K.radialIndexOf(st.m, st.radial.jp),
       `state fold=${st.fold} reports the counted radial index`);
  }
  // above the atlas the state set comes from the competition instead, and says so
  {
    const t = state({ f: 260, amplitudeMv: 430 });
    ok(t.states.every(x => x.theoryOnly === true),
       'above 199 Hz every state is flagged theory-only');
    ok(t.warnings.some(w => /[Tt]heory only|least-damped/.test(w.text)),
       'and the page says the atlas has run out');
  }
}
// 5000 Hz. 200 mV is 0.8 g and eps = 4.8e-4 against eps_c = 0.103 there, so
// that drive is far SUB-critical at a 2500 Hz response -- correctly, since eps
// falls as 1/omega_0^2. Crossing the tongue at 5 kHz takes about 250 g, which
// is 63000 mV at the declared 0.004 g/mV. Both regimes are checked.
{
  const quiet = state({ f: 5000, tempC: 25, depthMm: 1 });
  ok(quiet.onset.eps < quiet.onset.epsThreshold,
     '200 mV is sub-critical at 5 kHz', `eps ${quiet.onset.eps} vs ${quiet.onset.epsThreshold}`);
  eq(quiet.unstableCount, 0, 'a sub-critical drive grows nothing at 5 kHz');
}
{
  const s = state({ f: 5000, tempC: 25, depthMm: 1, amplitudeMv: 63000 });
  const xStar = s.wavenumber*(K.CELLS.medium.d/2000);
  ok(xStar > 150, `rim argument at 5 kHz is kR = ${xStar.toFixed(1)}`);
  const highest = Math.max(...s.nearestTheoryModes.map(t => t.m));
  ok(highest > 20, 'high angular orders are reached at 5 kHz, not clipped low',
     `highest reported m = ${highest}`);
  ok(s.unstableCount > 20,
     'a supercritical drive retains many growing modes, not just the near-onset one',
     `${s.unstableCount} growing of ${s.ordersScanned}`);
  eq(s.growingRetained, s.unstableCount,
     'every growing mode at 5 kHz is retained, none collapsed back to onset');
  eq(s.angularOrders, 16, 'at 5 kHz the sixteen-order cap is the binding limit');
  eq(s.ordersScanned, 48, 'forty-eight modes -- the lineage spec in full');
  ok(s.aboveAtlas === true, '5 kHz is flagged as beyond the empirical atlas');
  // and the onset must be far supercritical there
  ok(s.onset.eps > s.onset.epsThreshold,
     'the drive is above the Mathieu threshold at 5 kHz',
     `eps ${s.onset.eps} vs eps_c ${s.onset.epsThreshold}`);
}
// pinned rim must warn hard rather than silently pass off the free basis
{
  const s = state({ rim: 'pinned' });
  ok(s.warnings.some(w => w.hard && /pinned/i.test(w.text)),
     'a pinned contact line raises a hard warning');
}
// the large cell above 65 Hz has no atlas morphology and must say so
{
  const s = state({ cellKey: 'large', f: 111 });
  ok(s.warnings.some(w => w.hard && /unstable|indistinct/i.test(w.text)),
     'the large cell above 65 Hz warns hard');
}
// below the reported minimum drive, and above the overdrive limit
{
  ok(state({ amplitudeMv: K.MIN_MV - 1 }).warnings.some(w => /minimum drive/i.test(w.text)),
     'a sub-minimum drive is flagged');
  ok(state({ amplitudeMv: K.OVERDRIVE_MV + 1 }).warnings.some(w => /[Oo]verdriven/.test(w.text)),
     'an overdriven drive is flagged');
}
/* The competition outcome itself, pinned at the two drives where the angular
   quartic's factor of two changed it. These are the assertions that would have
   caught that defect in the only place it matters -- what the page draws.
   Doubling every m != 0 overlap leaves m = 0 under-penalised, so at 184 Hz an
   axisymmetric mode that the correct matrix kills survives, and at 70 Hz the
   m = 2 / m = 5 order inverts. Both are deterministic: the amplitudes come
   from a fixed-seed ODE integrated to steady state. */
{
  const s = state({ f: 184, amplitudeMv: 430 });
  const m = s.competition.survivors.map(v => v.m);
  eq(m.join(','), '13,7', '184 Hz at 430 mV: exactly m=13 then m=7 survive');
  ok(!m.includes(0),
     '184 Hz: the axisymmetric mode is killed, not kept by an under-penalised overlap',
     `survivors ${m.join(',')}`);
  rel(s.competition.survivors[0].amp, 2.21260, 4, '184 Hz: m=13 saturates at 2.2126');
  rel(s.competition.survivors[1].amp, 2.13145, 4, '184 Hz: m=7 saturates at 2.1315');
}
{
  const s = state({ f: 70, amplitudeMv: 430 });
  const m = s.competition.survivors.map(v => v.m);
  eq(m.join(','), '0,2,5', '70 Hz at 430 mV: m=0, then m=2, then m=5');
  rel(s.competition.survivors[0].amp, 2.92950, 4, '70 Hz: m=0 saturates at 2.9295');
  rel(s.competition.survivors[1].amp, 2.50716, 4, '70 Hz: m=2 saturates at 2.5072');
  rel(s.competition.survivors[2].amp, 2.49881, 4, '70 Hz: m=5 saturates at 2.4988');
  ok(s.competition.survivors[1].amp > s.competition.survivors[2].amp,
     '70 Hz: m=2 outranks m=5 (the doubled matrix inverts this)',
     `m=2 ${s.competition.survivors[1].amp}, m=5 ${s.competition.survivors[2].amp}`);
}

// determinism: same inputs, same state
{
  const a = state(), b = state();
  eq(a.ordersScanned, b.ordersScanned, 'mode count is deterministic');
  rel(a.wavenumber, b.wavenumber, 15, 'wavenumber is deterministic');
  eq(a.competition.survivors.length, b.competition.survivors.length,
     'survivor count is deterministic');
  for (let i = 0; i < a.nearestTheoryModes.length; i++)
    rel(a.nearestTheoryModes[i].growth, b.nearestTheoryModes[i].growth, 15,
        `growth at index ${i} is deterministic`);
}
// monotone in drive: more acceleration cannot reduce the growth of a fixed mode
{
  const lo = state({ amplitudeMv: 120 }), hi = state({ amplitudeMv: 360 });
  ok(hi.accelAssumed > lo.accelAssumed, 'more mV means more acceleration');
  ok(hi.onset.growth > lo.onset.growth,
     'a stronger drive grows the resonant mode faster',
     `${lo.onset.growth} then ${hi.onset.growth}`);
  ok(hi.unstableCount >= lo.unstableCount,
     'a stronger drive destabilises at least as many modes',
     `${lo.unstableCount} then ${hi.unstableCount}`);
}

/* ---- report ------------------------------------------------------------ */
console.log('\n' + '─'.repeat(66));
if (failures.length){
  console.log(`${pass} passed, ${failures.length} FAILED\n`);
  for (const f of failures) console.log('  FAIL  ' + f);
  process.exit(1);
}
console.log(`${pass} passed, 0 failed`);
