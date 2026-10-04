/* Checks for the linearised free-surface Navier-Stokes solver in a circular
   cell, dns/faraday-disc.js, and for the disc Floquet driver beside it.

   Every expected value comes from outside the solver: the analytic disc
   dispersion relation, Bessel's own differential equation, exact properties of
   the grid maps, or identities the equations must satisfy whatever the
   discretisation. Nothing here compares the code against its own opinion.

   The four defects this suite was written against, all found by measurement and
   all recorded in the solver's own comments:
     - the surface pressure applied as a predictor force, O(1/dz), which the
       projection then had to cancel: fields non-finite 0.57 periods in
     - a step limit that ignored the azimuthal direction: m = 12 blew up at step
       37 with amplification 2.15, while m = 0 and m = 2 survived 4000 steps
     - a radial grading that clustered at the axis instead of the rim, dropping
       the step limit to 3.5e-7 s for resolution where the mode is below
       round-off
     - uniform grids, which measured the damping inconsistently: Floquet growth
       drifted -5.82, -6.94, -8.47 across three refinements rather than settling,
       because each one resolved a little more of a Stokes layer that was never
       resolved

   Run: node dns/check-disc.mjs
*/
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ROOT = join(here, '..');
const K = require(join(ROOT, 'faraday', 'kernel.js'));
const D = require(join(here, 'faraday-disc.js'));
const F = require(join(here, 'faraday-floquet.js'));
const { FaradayDisc, gradeToEnd, gradeBothEnds, suggestGrid } = D;
const W = require(join(here, 'faraday-disc-wasm.js'));

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
function rel(got, want, decades, label){
  if (!Number.isFinite(got)) return ok(false, label, `got ${got}`);
  const denom = Math.abs(want) > 0 ? Math.abs(want) : 1;
  const err = Math.abs(got - want)/denom;
  ok(err <= Math.pow(10, -decades), label,
     `got ${got}, want ${want}, relative error ${err.toExponential(3)} > 1e-${decades}`);
}
function throws(label, fn, fragment){
  try { fn(); ok(false, label, 'did not throw'); }
  catch (e){ ok(!fragment || String(e.message).includes(fragment), label,
                `threw "${String(e.message).slice(0, 140)}"`); }
}
const section = t => console.log('\n' + t);

/* The apparatus the renderer shows: a 24.25 mm quartz cell, 3 mm of water. */
const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837,
               nu: 1.0035510043427237e-6, gamma: 0.07273614042160757,
               g: 9.80665 };

/* The analytic disc dispersion relation. k is set by the geometry -- the free
   contact line makes J'_m(kR) = 0 -- and the frequency by the layer. This is the
   reference, and it is computed from the kernel's Bessel zeros and Math.tanh,
   not from the solver. */
function analytic(m, n){
  const jp = K.jpZerosNearN(m, m + 1.9 + 3.2*(n - 1), n).sort((a, b) => a - b)[n-1];
  const k = jp/CELL.R;
  const om2 = (CELL.g*k + CELL.gamma*Math.pow(k, 3)/CELL.rho)*Math.tanh(k*CELL.h);
  return { jp, k, omega: Math.sqrt(om2), hz: Math.sqrt(om2)/(2*Math.PI) };
}

function seedMode(S, m, k, amp){
  for (let i = 0; i < S.nr; i++) S.eta[i] = amp*K.besselJ(m, k*S.rc[i]);
}

/* Free oscillation frequency, from the zero crossings of the elevation at the
   radius where it is largest. Phase-insensitive by construction: it counts
   crossings rather than reading an amplitude at whatever phase a run ends on --
   which is the mistake that first made this look 4x more dissipative than it is. */
function measureOmega(S, periods, omegaGuess){
  const dt = S.stableStep(0.4);
  const steps = Math.round(periods*(2*Math.PI/omegaGuess)/dt);
  let iT = 0, best = 0;
  for (let i = 0; i < S.nr; i++)
    if (Math.abs(S.eta[i]) > best){ best = Math.abs(S.eta[i]); iT = i; }
  const series = [];
  for (let s = 0; s < steps; s++){ S.step(dt); series.push(S.eta[iT]); }
  const cross = [];
  for (let s = 1; s < series.length; s++)
    if ((series[s-1] > 0) !== (series[s] > 0)){
      const f = series[s-1]/(series[s-1] - series[s]);
      cross.push((s - 1 + f)*dt);
    }
  if (cross.length < 3) throw new Error(
    `measureOmega saw ${cross.length} zero crossings in ${steps} steps; it needs `
    + `at least three to name a period. The mode did not oscillate.`);
  const halves = [];
  for (let c = 1; c < cross.length; c++) halves.push(cross[c] - cross[c-1]);
  const T = 2*halves.reduce((a, b) => a + b, 0)/halves.length;
  return { omega: 2*Math.PI/T, steps, dt, crossings: cross.length };
}

/* ── 1. the grid maps ───────────────────────────────────────────────────── */
section('1. the graded grid maps');
for (const [name, fn] of [['gradeToEnd', gradeToEnd], ['gradeBothEnds', gradeBothEnds]]){
  const n = 16, L = 3;
  const uniform = fn(n, L, 0);
  let uniformOk = true;
  for (let i = 0; i <= n; i++) if (Math.abs(uniform[i] - L*i/n) > 1e-15) uniformOk = false;
  ok(uniformOk, `${name} at stretch 0 is exactly uniform`);
  const gr = fn(n, L, 2.2);
  ok(gr[0] === 0 && gr[n] === L, `${name} hits both endpoints exactly`,
     `${gr[0]} .. ${gr[n]}`);
  let monotone = true;
  for (let i = 1; i <= n; i++) if (!(gr[i] > gr[i-1])) monotone = false;
  ok(monotone, `${name} is strictly monotone`);
}
{
  const g = gradeToEnd(16, 3, 2.2);
  ok(g[1] - g[0] > g[16] - g[15],
     'gradeToEnd puts the small cells at the far end, where the rim is',
     `first ${(g[1]-g[0]).toExponential(3)}, last ${(g[16]-g[15]).toExponential(3)}`);
  const b = gradeBothEnds(16, 3, 2.2);
  ok(b[1] - b[0] < b[8] - b[7] && b[16] - b[15] < b[8] - b[7],
     'gradeBothEnds puts the small cells at both ends and the large ones in the middle');
  rel(b[1] - b[0], b[16] - b[15], 12, 'gradeBothEnds is symmetric');
}

/* ── 2. the surface operator is exactly -k^2 on a Bessel mode ───────────── */
section('2. the surface Laplacian on the mode it is built for');
/* D_m J_m(kr) = -k^2 J_m(kr) is Bessel's equation, so this tests the graded
   metric against an identity rather than against another discretisation. A
   wrong face area, spacing or axis treatment moves it. */
/* Asserted as CONVERGENCE, not against a fixed tolerance. On a graded grid a
   flux-form second difference is second order only where the spacing varies
   smoothly, so its error at a given cell is a property of the grading, not a
   constant: measured at nr = 160, the worst relative error over the profile was
   1.1e-2 at m = 0, 1.1e-1 at m = 2, 2.3e-2 at m = 5 and 4.2e-3 at m = 12, worst
   in the coarse middle of the radius where the mode is small and where a
   rim-clustered grid deliberately puts its large cells. A fixed bound would be a
   statement about that grading. Halving the spacing and requiring the error to
   fall is a statement about the operator, which is what is being tested. */
/* Measured in a volume-weighted L2 norm over the profile, not at the worst
   single cell. A worst-point error is not a convergence measure on a graded
   grid: which cell is worst moves between resolutions, so the apparent order
   jumps -- measured 0.661 at m = 0 that way, against 1.0 or better for every
   other m, purely because J_0 peaks at the axis where a rim-clustered grid keeps
   its coarsest cells. The norm is what the discretisation's order is defined
   against. */
function laplacianErrorNorm(m, k, nr){
  const S = new FaradayDisc({ m, nr, nz: 8, ...CELL, contact: 'free' });
  seedMode(S, m, k, 1);
  let num = 0, den = 0;
  for (let i = 0; i < S.nr; i++){
    const wgt = S.rc[i]*S.drc[i];
    const got = S.surfaceLaplacian(i), want = -k*k*S.eta[i];
    num += (got - want)*(got - want)*wgt;
    den += want*want*wgt;
  }
  return Math.sqrt(num/den);
}
for (const m of [0, 2, 5, 12]){
  const a = analytic(m, 1);
  const coarse = laplacianErrorNorm(m, a.k, 80);
  const fine = laplacianErrorNorm(m, a.k, 160);
  const order = Math.log(coarse/fine)/Math.log(2);
  ok(fine < coarse, `m = ${m}: refining the radius reduces the error in `
     + `D_m J_m(kr) = -k^2 J_m(kr)`,
     `${coarse.toExponential(3)} at nr=80 then ${fine.toExponential(3)} at nr=160`);
  ok(order > 0.9, `m = ${m}: and it falls at first order or better `
     + `(order ${order.toFixed(2)})`,
     `observed order ${order.toFixed(3)}`);
}

/* ── 3. the free oscillation is the analytic disc frequency ─────────────── */
section('3. free oscillation against the analytic disc dispersion relation');
/* Viscosity is dropped to 1e-10 so the frequency is the inviscid one; the
   damping is checked separately below. Convergence is asserted rather than a
   single grid, because the surface conditions are first order and a single grid
   cannot tell a consistent scheme from a wrong one. */
/* 1.5 periods, not 2: the zero-crossing measure needs three crossings to name a
   period and 1.5 periods gives exactly three. The low-m modes cost the most time
   here -- m = 0 and m = 2 oscillate at 10 and 7.7 Hz against m = 12's 55 Hz, so
   their periods are five to seven times longer at a comparable step -- and they
   are the reason this section is timed in minutes rather than seconds. */
const OSC_PERIODS = 1.5;
for (const [m, n] of [[0, 1], [2, 1], [12, 1]]){
  const a = analytic(m, n);
  const errs = [];
  for (const [nr, nz] of [[20, 10], [30, 15]]){
    const S = new FaradayDisc({ m, nr, nz, ...CELL, nu: 1e-10, contact: 'free',
                                accel: 0, omegaD: 1 });
    seedMode(S, m, a.k, 1e-9);
    const r = measureOmega(S, OSC_PERIODS, a.omega);
    errs.push(Math.abs(r.omega/a.omega - 1));
    ok(Math.abs(r.omega/a.omega - 1) < 0.03,
       `m = ${m}, n = ${n}, ${nr}x${nz}: omega within 3% of ${a.omega.toFixed(2)} rad/s`,
       `got ${r.omega.toFixed(3)}, ${((r.omega/a.omega-1)*100).toFixed(3)}%`);
    ok(S.maxDivergence() < 1e-12,
       `m = ${m}, ${nr}x${nz}: the corrected field is divergence free`,
       `max |div u| / volume = ${S.maxDivergence().toExponential(2)}`);
  }
  ok(errs[1] < errs[0], `m = ${m}: refining the grid moves omega toward the analytic value`,
     `${(errs[0]*100).toFixed(3)}% then ${(errs[1]*100).toFixed(3)}%`);
}

/* ── 4. energy ──────────────────────────────────────────────────────────── */
section('4. energy: dissipated with viscosity, conserved without');
{
  const m = 5, a = analytic(m, 1);
  const S = new FaradayDisc({ m, nr: 24, nz: 12, ...CELL, contact: 'free',
                              accel: 0, omegaD: 1 });
  seedMode(S, m, a.k, 1e-9);
  const dt = S.stableStep(0.4);
  const steps = Math.round(OSC_PERIODS*(2*Math.PI/a.omega)/dt);
  let E = S.energy().total, monotone = true, worst = 0;
  const E0 = E;
  for (let s = 0; s < steps; s++){
    S.step(dt);
    const En = S.energy().total;
    if (En > E*(1 + 1e-9)){ monotone = false; worst = Math.max(worst, En/E - 1); }
    E = En;
  }
  ok(monotone, 'with no drive the perturbation energy never increases',
     `largest increase ${worst.toExponential(3)} of the running value`);
  ok(E < E0, `and it has decreased over ${OSC_PERIODS} periods`,
     `${E0.toExponential(4)} to ${E.toExponential(4)}`);

  const S2 = new FaradayDisc({ m, nr: 24, nz: 12, ...CELL, nu: 1e-10,
                               contact: 'free', accel: 0, omegaD: 1 });
  seedMode(S2, m, a.k, 1e-9);
  const dt2 = S2.stableStep(0.4);
  const steps2 = Math.round(OSC_PERIODS*(2*Math.PI/a.omega)/dt2);
  const F0 = S2.energy().total;
  for (let s = 0; s < steps2; s++) S2.step(dt2);
  const F1 = S2.energy().total;
  ok(Math.abs(F1/F0 - 1) < 0.05,
     `with viscosity removed the energy is conserved to within 5% over `
     + `${OSC_PERIODS} periods`,
     `${F0.toExponential(4)} to ${F1.toExponential(4)}, `
     + `${((F1/F0-1)*100).toFixed(3)}%`);
}

/* ── 5. the damping is viscous, not numerical ──────────────────────────── */
section('5. the damping scales with viscosity');
/* Numerical dissipation from a first-order boundary treatment does not care
   what nu is. Physical dissipation in a resolved layer is linear in it, and a
   Stokes layer that is NOT resolved goes as its square root. Measuring the
   exponent is what distinguishes the three. */
{
  const m = 5, a = analytic(m, 1);
  const rates = [];
  for (const mult of [1, 2, 4]){
    const S = new FaradayDisc({ m, nr: 24, nz: 12, ...CELL, nu: CELL.nu*mult,
                                contact: 'free', accel: 0, omegaD: 1 });
    seedMode(S, m, a.k, 1e-9);
    const dt = S.stableStep(0.4);
    const steps = Math.round(OSC_PERIODS*(2*Math.PI/a.omega)/dt);
    const E0 = S.energy().total;
    for (let s = 0; s < steps; s++) S.step(dt);
    rates.push(-Math.log(S.energy().total/E0)/(2*steps*dt));
  }
  const slope = Math.log(rates[2]/rates[0])/Math.log(4);
  const bulk = 2*CELL.nu*a.k*a.k;
  console.log(`       gamma = ${rates.map(r => r.toFixed(3)).join(', ')} s^-1 `
    + `at nu, 2nu, 4nu -- exponent ${slope.toFixed(3)}; bulk 2 nu k^2 alone is `
    + `${bulk.toFixed(3)}`);
  /* This expectation was wrong before it was measured. It demanded an exponent
     near 1, on the reasoning that resolved dissipation is linear in nu. Half of
     that is right: the bulk term 2 nu k^2 is linear. But the floor, rim and
     surface Stokes layers dissipate as sqrt(nu), and a real layer carries both,
     so the exponent belongs strictly BETWEEN the two. The arithmetic, at m = 5,
     n = 1: 2 nu k^2 is 0.562 of the measured 2.463 per second, leaving 1.90 in
     the layers, for a predicted exponent of (0.562*1 + 1.90*0.5)/2.463 = 0.61.
     Measured 0.755, on the layer side of the middle. What the test can and does
     establish is that the exponent is well away from 0 -- the scheme's own
     dissipation does not care what nu is -- and away from a pure square root. */
  ok(slope > 0.55 && slope < 1.05,
     'the decay rate sits between the bulk term\'s linear law and a Stokes '
     + 'layer\'s square root, which is what a resolved layer plus a bulk gives',
     `exponent ${slope.toFixed(4)}; 0 would be the scheme's own dissipation, `
     + `0.5 a pure boundary layer, 1 a pure bulk`);
  ok(slope > 0.3, 'and far enough from zero that the dissipation is not the '
     + 'scheme\'s own', `exponent ${slope.toFixed(4)}`);
  ok(rates[0] > 0, 'and it is a decay, not a growth', `${rates[0]} s^-1`);
  ok(rates[0] > bulk,
     'the disc damps harder than the bulk term alone, which is the sidewall and '
     + 'floor the renderer\'s gamma leaves out',
     `${rates[0].toFixed(4)} against 2 nu k^2 = ${bulk.toFixed(4)} s^-1`);
}

/* ── 6. the Stokes layer is resolved, and the solver says so ───────────── */
section('6. the grading puts cells in the layers that set the damping');
{
  const S = new FaradayDisc({ m: 12, nr: 48, nz: 24, ...CELL, contact: 'free',
                              accel: 7.060788, omegaD: 2*Math.PI*111 });
  const st = S.stokesResolution(2*Math.PI*55.5);
  console.log(`       delta = ${(st.delta*1e6).toFixed(1)} um; cells inside it: `
    + `floor ${st.cellsInFloorLayer.toFixed(2)}, surface `
    + `${st.cellsInSurfaceLayer.toFixed(2)}, rim ${st.cellsInRimLayer.toFixed(2)}`);
  rel(st.delta, Math.sqrt(2*CELL.nu/(2*Math.PI*55.5)), 12,
      'the reported Stokes depth is sqrt(2 nu / omega)');
  ok(st.cellsInFloorLayer > 2 && st.cellsInSurfaceLayer > 2 && st.cellsInRimLayer > 2,
     'at 48x24 every Stokes layer holds more than two cells',
     `floor ${st.cellsInFloorLayer.toFixed(2)}, surface `
     + `${st.cellsInSurfaceLayer.toFixed(2)}, rim ${st.cellsInRimLayer.toFixed(2)}`);
  const U = new FaradayDisc({ m: 12, nr: 48, nz: 24, ...CELL, contact: 'free',
                              rStretch: 0, zStretch: 0 });
  const su = U.stokesResolution(2*Math.PI*55.5);
  ok(su.cellsInFloorLayer < 1,
     'while an ungraded grid of the same size holds less than one, which is why '
     + 'the grading exists',
     `${su.cellsInFloorLayer.toFixed(3)} cells in the floor layer`);
}

/* The grid the Floquet sections run on, sized to the mode rather than chosen.
   The sections below used a fixed 24 x 12, which the resolution gate now refuses
   and was right to: at that grid this mode gets 0.69 cells into the floor Stokes
   layer, so its damping came out of the truncation error. */
const A8 = analytic(12, 1);
const K8 = A8.k;
const SIZED8 = suggestGrid({ m: 12, k: K8, ...CELL, contact: 'free',
                             omegaResponse: Math.PI*111, omegaDrive: 2*Math.PI*111,
                             profile: r => K.besselJ(12, K8*r) });
if (SIZED8.unreachable) throw new Error(
  'check-disc: no grid on the ladder carries m = 12, n = 1 at 111 Hz, which is '
  + 'the operating point every Floquet section below is written against.');
const GRID8 = { nr: SIZED8.nr, nz: SIZED8.nz,
                rStretch: SIZED8.rStretch, zStretch: SIZED8.zStretch };
const ETA8 = (() => {
  const S = new FaradayDisc({ m: 12, ...GRID8, ...CELL, contact: 'free' });
  const e = new Float64Array(GRID8.nr);
  for (let i = 0; i < GRID8.nr; i++) e[i] = K.besselJ(12, K8*S.rc[i]);
  return e;
})();
console.log(`\n  sized for the Floquet sections: ${GRID8.nr} x ${GRID8.nz}, `
  + `r stretch ${GRID8.rStretch}, z stretch ${GRID8.zStretch}; surface operator `
  + `error ${SIZED8.err.toExponential(2)}, ${SIZED8.layers.toFixed(2)} cells in the `
  + `tightest Stokes layer, ${SIZED8.stateSize} unknowns, ${SIZED8.stepsPerPeriod} `
  + `steps per drive period`);

/* ── 7. refusals ───────────────────────────────────────────────────────── */
section('7. what it refuses rather than answering');
throws('m = 1 is refused, naming the cancellation that makes it different',
       () => new FaradayDisc({ m: 1, nr: 16, nz: 8, ...CELL }), 'm = 1');
throws('a contact line that is neither free nor pinned is refused',
       () => new FaradayDisc({ m: 2, nr: 16, nz: 8, ...CELL, contact: 'slip' }),
       'contact');
throws('zero viscosity is refused', () => new FaradayDisc({ m: 2, nr: 16, nz: 8, ...CELL, nu: 0 }), 'nu');
throws('a negative radius is refused', () => new FaradayDisc({ m: 2, nr: 16, nz: 8, ...CELL, R: -1 }), 'R');
throws('a non-integer mode number is refused',
       () => new FaradayDisc({ m: 2.5, nr: 16, nz: 8, ...CELL }), 'm');
throws('a grid too coarse to carry the surface stencil is refused',
       () => new FaradayDisc({ m: 2, nr: 16, nz: 3, ...CELL }), 'nz');
throws('a negative grading is refused, since it would put cells where the '
       + 'gradients are not',
       () => new FaradayDisc({ m: 2, nr: 16, nz: 8, ...CELL, zStretch: -1 }), 'Stretch');
throws('a zero drive frequency is refused by the Floquet driver',
       () => F.floquetDisc({ nr: 16, nz: 8, ...CELL, m: 2, omegaD: 0 }), 'omegaD');
throws('a step above the explicit stability limit is refused',
       () => F.floquetDisc({ nr: 24, nz: 12, ...CELL, m: 12, accel: 7,
                             omegaD: 2*Math.PI*111, dt: 1e-3 }), 'stability limit');
/* A short Krylov space is now ESCALATED rather than refused -- the Arnoldi space
   is extended, so asking for more costs only the extra vectors -- so the refusal
   is reached by capping the escalation as well. With the cap at 4 the ladder has
   a single rung and there is no evidence the leading modulus has settled, which
   is the refusal. It runs on a grid that passes the resolution gate, because
   otherwise that gate fires first and this would be testing the wrong refusal. */
throws('a Krylov space too short to find the dominant mode is refused, not '
       + 'reported',
       () => F.floquetDisc({ ...GRID8, ...CELL, m: 12, accel: 7, eta0: ETA8,
                             omegaD: 2*Math.PI*111, contact: 'free',
                             engine: 'wasm', krylov: 4, krylovCap: 4 }),
       'not converged in the Krylov dimension');
throws('a grid that cannot carry the mode is refused before a step is taken, '
       + 'naming the cells it is short of',
       () => F.floquetDisc({ nr: 16, nz: 8, ...CELL, m: 12, accel: 7, k: K8,
                             omegaD: 2*Math.PI*111, contact: 'free' }),
       'cannot carry the mode');
throws('and the refusal names the Stokes layer, not just the surface operator',
       () => F.floquetDisc({ nr: 16, nz: 8, ...CELL, m: 12, accel: 7, k: K8,
                             omegaD: 2*Math.PI*111, contact: 'free' }),
       'cells at the floor');

/* ── 8. Floquet ────────────────────────────────────────────────────────── */
section('8. Floquet multipliers on the disc');
{
  const m = 12;
  const common = { ...GRID8, ...CELL, m, contact: 'free', eta0: ETA8, k: K8,
                   omegaD: 2*Math.PI*111, engine: 'wasm' };

  const quiet = F.floquetDisc({ ...common, accel: 0 });
  ok(quiet.muMax < 1, 'an undriven layer has every multiplier inside the unit circle',
     `|mu| = ${quiet.muMax}`);

  const driven = F.floquetDisc({ ...common, accel: 7.060788 });
  console.log(`       a = 0: |mu| = ${quiet.muMax.toFixed(8)} +/- `
    + `${quiet.muSpread.toExponential(2)}, growth ${quiet.growth.toFixed(4)} s^-1`);
  console.log(`       a = 7.0608 m/s^2: |mu| = ${driven.muMax.toFixed(8)} +/- `
    + `${driven.muSpread.toExponential(2)}, growth ${driven.growth.toFixed(4)} `
    + `+/- ${driven.growthWidth.toFixed(4)} s^-1`);

  /* No claim is made here that this drive raises the multiplier. It should, and
     at krylov 20 it measurably does -- 0.94444 driven against 0.94378 undriven
     at nr = 28, nz = 14 -- but that margin is 0.07% while the cluster's own
     width is 0.5%, so the ordering is BELOW the resolution of the method at
     this grid and asserting it would be asserting noise. The effect is asserted
     where it exceeds the width instead, at a = 60 below. What this drive
     establishes is the thing that matters and is far outside the width: the
     disc calls it stable by a wide margin where the renderer calls it unstable. */
  ok(driven.muMax + driven.muSpread < 1,
     'this drive leaves the mode stable even allowing the cluster its full width',
     `|mu| = ${driven.muMax} + ${driven.muSpread}`);
  ok(driven.unstable === false,
     'and the verdict is reported as decided rather than borderline',
     `unstable = ${driven.unstable}`);

  /* The drive has to be able to reach onset, or the solver is not measuring a
     Faraday problem at all. Bracketing the crossing is the check, not a single
     threshold: a bracket is falsifiable on both sides. */
  const strong = F.floquetDisc({ ...common, accel: 60 });
  ok(strong.muMax > 1, 'a strong enough drive puts a multiplier outside it',
     `a = 60 m/s^2 gives |mu| = ${strong.muMax}`);
  console.log(`       a = 60 m/s^2: |mu| = ${strong.muMax.toFixed(8)}, growth `
    + `${strong.growth.toFixed(4)} s^-1`);
  ok(driven.muMax < 1 && strong.muMax > 1,
     'so the Faraday threshold for this mode is bracketed between 7.06 and 60 m/s^2',
     `|mu| = ${driven.muMax.toFixed(6)} and ${strong.muMax.toFixed(6)}`);

  /* growth is ln|mu| over the drive period by definition, so this is an
     identity the driver must satisfy rather than a physical claim. */
  rel(driven.growth, Math.log(driven.muMax)/driven.Td, 10,
      'the reported growth is ln|mu| over one drive period');
  ok(driven.steps*driven.dt > 0 && Math.abs(driven.steps*driven.dt - driven.Td)
       < 1e-12*driven.Td,
     'the integration covers exactly one drive period',
     `${driven.steps} steps of ${driven.dt} against Td = ${driven.Td}`);
  ok(!driven.breakdown || driven.krylov > 1,
     'the Krylov space either filled or broke down on an invariant subspace',
     `krylov ${driven.krylov}, residual ${driven.residual}`);
  ok(driven.engine === 'wasm',
     'and it ran in the C++ engine, which is what was asked for',
     `engine ${driven.engine}`);

  /* The JavaScript reference, on the same problem. Everything above ran in C++
     because it is four times faster and the suite has to finish; this is the line
     that makes that legitimate. It is not a tolerance: the two implementations
     compute the same arithmetic in the same order and must agree to the last bit,
     so the assertion is equality, and any difference at all fails it. */
  const ref = F.floquetDisc({ ...common, accel: 7.060788, engine: 'js' });
  ok(ref.muMax === driven.muMax,
     'the JavaScript solver returns the identical multiplier on the identical '
     + 'problem, to the last bit, so running the physics in C++ above asserts '
     + 'nothing the JavaScript does not',
     `js ${ref.muMax} against wasm ${driven.muMax}`);
  ok(ref.krylov === driven.krylov && ref.steps === driven.steps,
     'and reaches it through the same Krylov dimension over the same steps',
     `js krylov ${ref.krylov}/${ref.steps} steps, wasm ${driven.krylov}/${driven.steps}`);
  ok(ref.engine === 'js', 'with the engine it reports being the one asked for',
     `engine ${ref.engine}`);
}

/* ── 9. the contact line changes the answer ────────────────────────────── */
section('9. the contact line is a physical choice, not a default');
/* Measured by direct integration of the undriven layer, not by comparing two
   Floquet multipliers. That comparison was tried first and was NOT resolved at a
   grid this suite can afford: at nr = 24, nz = 12 the free rim gave |mu| =
   0.94411 with a cluster width of 4.4e-3 and the pinned rim 0.96255 with a width
   of 2.4e-2, so the 1.8e-2 difference between them sat inside the pinned case's
   own uncertainty. Rather than widen the claim until it passed, the measurement
   was changed to one that is well conditioned: an energy decay rate from a
   deterministic time integration, with no Krylov space and no cluster to pick a
   member of. */
{
  const m = 12, a = analytic(m, 1);
  const rates = {};
  for (const contact of ['free', 'pinned']){
    const S = new FaradayDisc({ m, nr: 24, nz: 12, ...CELL, contact,
                                accel: 0, omegaD: 1 });
    seedMode(S, m, a.k, 1e-9);
    const dt = S.stableStep(0.4);
    const steps = Math.round(OSC_PERIODS*(2*Math.PI/a.omega)/dt);
    const E0 = S.energy().total;
    for (let q = 0; q < steps; q++) S.step(dt);
    rates[contact] = -Math.log(S.energy().total/E0)/(2*steps*dt);
  }
  console.log(`       free rim decays at ${rates.free.toFixed(4)} s^-1, `
    + `pinned at ${rates.pinned.toFixed(4)} s^-1`);
  ok(Math.abs(rates.pinned - rates.free) > 0.05*rates.free,
     'pinning the contact line changes the decay rate by more than 5%, so the '
     + 'option is load bearing rather than cosmetic',
     `${rates.free.toFixed(5)} against ${rates.pinned.toFixed(5)} s^-1`);
  ok(rates.free > 0 && rates.pinned > 0,
     'and both rims dissipate rather than produce',
     `${rates.free}, ${rates.pinned}`);
}

/* ── 10. the disc is not the periodic box ──────────────────────────────── */
section('10. the disc answers differently from the periodic box');
/* This is the reason the disc solver exists. The box solver next door takes the
   same layer, the same drive and the same dominant wavenumber and answers for a
   two-dimensional periodic strip; the disc adds the no-slip sidewall, which for
   a rim-concentrated mode is most of its damping. If the two agreed there would
   be nothing to build. */
{
  const m = 12, a = analytic(m, 1);
  const boxed = F.floquet({ nx: 32, ns: 32, L: 2*Math.PI/a.k, h0: CELL.h,
                            rho: CELL.rho, nu: CELL.nu, gamma: CELL.gamma,
                            accel: 7.060788, omegaD: 2*Math.PI*111, m: 4 });
  const disc = F.floquetDisc({ ...GRID8, ...CELL, m, contact: 'free', k: K8,
                               eta0: ETA8, accel: 7.060788,
                               omegaD: 2*Math.PI*111, engine: 'wasm' });
  console.log(`       periodic box: growth ${boxed.growth.toFixed(4)} s^-1 `
    + `(|mu| ${boxed.muMax.toFixed(6)})`);
  console.log(`       disc:         growth ${disc.growth.toFixed(4)} s^-1 `
    + `(|mu| ${disc.muMax.toFixed(6)})`);
  ok(boxed.muMax > 1, 'the periodic box calls this drive unstable', `|mu| = ${boxed.muMax}`);
  ok(disc.muMax < 1, 'the disc calls the same drive stable', `|mu| = ${disc.muMax}`);
  ok(disc.growth + disc.growthWidth < boxed.growth - 5,
     'and by a margin far larger than the cluster width, so the sidewall is not '
     + 'a rounding difference',
     `${disc.growth.toFixed(3)} +/- ${disc.growthWidth.toFixed(3)} against `
     + `${boxed.growth.toFixed(3)} s^-1`);

  /* The renderer's own figure, for the same mode and the same drive, from the
     damped Mathieu form. This is the number the deck prints, and the disc
     disagrees with it in SIGN -- which is the finding, and the reason the panel
     on the page had to stop being a second opinion printed beside it. */
  console.log(`       the deck's Mathieu growth for this state is +3.24 s^-1, so `
    + `the disc reverses the verdict rather than trimming it`);
}

/* ── 11. the C++ period map is the JavaScript one ───────────────────────── */
section('11. the C++ engine against the JavaScript solver, bit for bit');
/* Not "close". dns/faraday_disc.cpp is a transcription of dns/faraday-disc.js:
   the same discretisation in the same order, double precision, no fused
   multiply-add, no fast-math, and no transcendental at all -- the drive's cosine
   is evaluated in JavaScript into a table and passed in. The only floating point
   operations either side performs are +, -, *, / and sqrt, all of which IEEE-754
   specifies exactly, so the two must agree to the last bit and the assertion is
   equality. A tolerance here would hide exactly the kind of divergence that
   matters: one term reassociated, one loop bound off by a cell.

   This runs on a coarse grid on purpose. Parity is a property of the
   transcription, not of the resolution: whether the two implementations agree
   does not depend on whether the grid carries the mode, and a coarse grid
   exercises every term in seconds instead of minutes. */
{
  const m = 12, a = analytic(m, 1);
  for (const contact of ['free', 'pinned']){
    const nr = 20, nz = 10;
    const mk = () => {
      const S = new FaradayDisc({ m, nr, nz, ...CELL, contact,
                                  accel: 7.060788, omegaD: 2*Math.PI*111 });
      return S;
    };
    const S = mk();
    const n = 2*(nr - 1)*nz + nr*nz + nr;
    const dt = S.stableStep(0.4);
    const href = 1e-9, uref = href*Math.PI*111;

    /* A state with every block non-zero, so a dropped term anywhere shows. */
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.sin(0.7*i + 0.3) + 0.25*Math.cos(1.9*i);
    for (let i = 0; i < nr; i++)
      v[2*(nr-1)*nz + nr*nz + i] = K.besselJ(m, a.k*S.rc[i]);

    const E = W.createDiscEngine({}).load(S, 1, dt, 7.060788, 2*Math.PI*111, S.g);
    const gNow = S.g + 7.060788*Math.cos(0);

    /* One step first: a divergence here is still attributable to a single term. */
    const js1 = new Float64Array(n), wa1 = new Float64Array(n);
    F.applyDiscPeriodMap(mk(), v, 1, dt, js1, href, uref);
    E.stepOnce(v, wa1, dt, gNow, href, uref);
    let same = 0;
    for (let i = 0; i < n; i++) if (js1[i] === wa1[i]) same++;
    ok(same === n, `${contact} rim, one step: all ${n} values identical`,
       `${same} of ${n} identical`);

    /* Then a full drive period, where a difference of one bit in one cell has
       had every step to grow into the answer. */
    const steps = Math.max(1, Math.round((2*Math.PI/(2*Math.PI*111))/dt));
    const E2 = W.createDiscEngine({}).load(S, steps, dt, 7.060788, 2*Math.PI*111, S.g);
    const jsP = new Float64Array(n), waP = new Float64Array(n);
    F.applyDiscPeriodMap(mk(), v, steps, dt, jsP, href, uref);
    E2.applyPeriodMap(v, waP, dt, href, uref);
    same = 0;
    let worst = 0;
    for (let i = 0; i < n; i++){
      if (jsP[i] === waP[i]) same++;
      const d = Math.abs(jsP[i] - waP[i]);
      const sc = Math.max(Math.abs(jsP[i]), Math.abs(waP[i]));
      if (sc > 0) worst = Math.max(worst, d/sc);
    }
    ok(same === n, `${contact} rim, one drive period of ${steps} steps: all `
       + `${n} values identical`,
       `${same} of ${n} identical, worst relative difference ${worst.toExponential(3)}`);
    /* The map is not the identity, or the equality above would be vacuous. */
    let moved = 0;
    for (let i = 0; i < n; i++) if (waP[i] !== v[i]) moved++;
    ok(moved > 0.9*n, `${contact} rim: and the period map moved the state, so the `
       + `agreement is not two copies of the input`, `${moved} of ${n} moved`);
  }

  /* The loader refuses rather than substituting. Bytes that are not a module
     cannot compile, and what comes back must be a refusal that names the engine
     -- not a silent run of the JavaScript under the C++ name. */
  throws('bytes that are not a WebAssembly module are refused',
         () => W.createDiscEngine({ bytes: new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]) }),
         'would not compile');
  /* And a request the arena cannot hold is refused, not silently truncated.
     This asked for a 512 x 96 grid first, on the assumption that the largest grid
     the sizing ladder allows would overflow. Measured: it does not -- 512 x 96
     with 100000 drive samples fits in the 3000000-double arena with room to
     spare, and the load was accepted, which is why the test read "did not throw".
     What the arena is actually sized against is the TOTAL, and the drive table is
     the term that scales with the step count rather than with the grid: 512 x 96
     is refused at 2000000 samples, and a 20 x 10 grid is refused at 4000000. The
     small grid is used so the refusal is reached without building a large one. */
  throws('a drive table larger than the module arena is refused',
         () => { const S = new FaradayDisc({ m: 12, nr: 20, nz: 10, ...CELL });
                 W.createDiscEngine({}).load(S, 4000000, 1e-6, 0, 1, S.g); },
         'arena');
}

/* ── 11b. the preconditioner is the operator's own diagonal ─────────────── */
section('11b. the pressure solve is preconditioned by the operator\'s diagonal');
/* Two claims, and both are checkable exactly.

   The first is that the two-application red/black probe recovers the TRUE
   diagonal. That rests on the composite stencil reaching only cells of the
   opposite parity, which is an argument about the discretisation -- so it is
   checked against the definition instead: N applications of the operator to N
   unit vectors, reading the i-th entry of the i-th result. If the stencil ever
   grows a corner term the two will disagree and this fails.

   The second is that preconditioning is a speed-up and not a looser solve. The
   comparison is against a plain conjugate gradient written here, on the same
   operator, driven to the same tolerance, so what is being compared is iteration
   counts at equal accuracy and nothing else. */
for (const [m, nr, nz, zs] of [[12, 20, 10, 2.2], [0, 24, 12, 2.2],
                               [2, 16, 8, 0], [12, 40, 12, 3.0]]){
  const S = new FaradayDisc({ m, nr, nz, ...CELL, rStretch: 2.2, zStretch: zs });
  const n = nr*nz;
  const probe = new Float64Array(n), q = new Float64Array(n), byUnit = new Float64Array(n);
  for (let i = 0; i < n; i++){
    probe.fill(0); probe[i] = 1;
    S.applyL(probe, q);
    byUnit[i] = q[i];
  }
  let same = 0, negative = 0;
  for (let i = 0; i < n; i++){
    if (byUnit[i] === S._pdiag[i]) same++;
    if (S._pdiag[i] < 0) negative++;
  }
  ok(same === n,
     `m = ${m}, ${nr}x${nz}: the two-application diagonal equals the ${n}-`
     + `application one in every entry`,
     `${same} of ${n} identical`);
  ok(negative === n,
     `m = ${m}, ${nr}x${nz}: and every entry is negative, as a negative definite `
     + `operator requires`, `${negative} of ${n} negative`);

  /* Iterations at equal accuracy, against a plain CG written here. */
  const eta0 = new Float64Array(nr);
  const a = analytic(m, 1);
  for (let i = 0; i < nr; i++) S.eta[i] = 1e-9*K.besselJ(m, a.k*S.rc[i]);
  const dt = S.stableStep(0.4);
  S.step(dt);
  const rhs = Float64Array.from(S._div);
  const plainCG = () => {
    const x = new Float64Array(n), r = new Float64Array(n), d = new Float64Array(n),
          Q = new Float64Array(n);
    S.applyL(x, Q);
    let rr = 0;
    for (let i = 0; i < n; i++){ r[i] = rhs[i] - Q[i]; d[i] = r[i]; rr += r[i]*r[i]; }
    const rr0 = rr;
    let it = 0;
    for (; it < 40*(nr + nz); it++){
      S.applyL(d, Q);
      let dq = 0;
      for (let i = 0; i < n; i++) dq += d[i]*Q[i];
      if (dq === 0) break;
      const alpha = rr/dq;
      let rr2 = 0;
      for (let i = 0; i < n; i++){ x[i] += alpha*d[i]; r[i] -= alpha*Q[i]; rr2 += r[i]*r[i]; }
      if (Math.sqrt(rr2/rr0) < 1e-11){ rr = rr2; it++; break; }
      const beta = rr2/rr; rr = rr2;
      for (let i = 0; i < n; i++) d[i] = r[i] + beta*d[i];
    }
    return { it, res: Math.sqrt(rr/rr0) };
  };
  const plain = plainCG();
  S.p.fill(0);
  S.solveP(rhs, 1e-11, 40*(nr + nz));
  ok(S.cgResidual < 1e-11 && plain.res < 1e-11,
     `m = ${m}, ${nr}x${nz}: both solves reach the same 1e-11 residual, so the `
     + `comparison is iterations at equal accuracy`,
     `preconditioned ${S.cgResidual.toExponential(2)}, plain ${plain.res.toExponential(2)}`);
  ok(S.cgIters < plain.it,
     `m = ${m}, ${nr}x${nz}: and the preconditioned solve gets there in fewer `
     + `iterations`,
     `${S.cgIters} against ${plain.it}, a factor of ${(plain.it/S.cgIters).toFixed(2)}`);
  console.log(`       m = ${m}, ${nr}x${nz}: ${S.cgIters} preconditioned against `
    + `${plain.it} plain iterations to 1e-11`);
}

/* ── 12. sizing the grid, and extending the Krylov space ────────────────── */
section('12. the grid is sized to the mode, and the space is extended not rebuilt');
{
  /* The operating point the renderer defaults to is reachable; a mode with 37
     radial wavelengths across a 24 mm cell is not, and that is reported as
     unreachable rather than answered on a grid that cannot carry it. */
  const a = analytic(12, 3);
  const good = suggestGrid({ m: 12, k: a.k, ...CELL, contact: 'free',
                             omegaResponse: a.omega, omegaDrive: 2*a.omega,
                             profile: r => K.besselJ(12, a.k*r) });
  ok(!good.unreachable, 'a mode with a few radial wavelengths gets a grid',
     JSON.stringify(good.unreachable ? { tried: good.tried.length } : {
       nr: good.nr, nz: good.nz }));
  ok(good.err <= 1e-2, 'whose surface operator error meets the one per cent gate',
     `${good.err.toExponential(3)}`);
  ok(good.layers >= 2, 'and which holds two cells in every Stokes layer',
     `${good.layers.toFixed(3)} in the tightest`);
  /* The cheapest that holds, not the first found. */
  const cheaper = good.tried.filter(t => t.err <= 1e-2 && t.layersOk
                                         && t.work < good.work);
  ok(cheaper.length === 0,
     'and it is the cheapest of every grid tried that meets both gates',
     `${cheaper.length} cheaper passing grids, e.g. `
     + (cheaper[0] ? `${cheaper[0].nr}x${cheaper[0].nz}` : 'none'));

  const hard = analytic(17, 46);
  const no = suggestGrid({ m: 17, k: hard.k, ...CELL, contact: 'free',
                           omegaResponse: hard.omega, omegaDrive: 2*hard.omega,
                           profile: r => K.besselJ(17, hard.k*r) });
  ok(no.unreachable === true,
     `a mode with kR = ${hard.k*CELL.R < 1 ? '' : (hard.k*CELL.R).toFixed(0)} is `
     + `reported unreachable rather than answered`,
     no.unreachable ? `${no.tried.length} grids tried` : `${no.nr}x${no.nz} offered`);
  const best = no.tried.reduce((x, y) => y.err < x.err ? y : x);
  ok(best.err > 1e-2,
     'and the closest grid on the ladder really does miss the gate',
     `${best.err.toExponential(3)} at ${best.nr}x${best.nz}`);
  console.log(`       m = 17, n = 46: closest of ${no.tried.length} grids is `
    + `${best.nr}x${best.nz} at ${best.err.toExponential(2)}, against the 1e-2 gate`);

  /* A profile with no structure the operator can see is refused by the fit rather
     than given a wavenumber -- and which profiles those are is not what was first
     written here. This asked for a constant elevation at m = 12 to be refused,
     which it is not, and measurement says why: the azimuthal part of the operator
     is -m^2 eta/r^2, which does not vanish on a constant. At m = 12 a constant
     elevation gives D_m eta = -4.464e-1 at the axis cell and fits lambda =
     -8.491e+6 with a residual of 4.95 -- a legitimate answer, saying loudly that
     the profile is not a mode. The operator is blind to a constant only at m = 0,
     where D_0 of a constant is exactly zero, and that is the case refused. */
  throws('a constant elevation at m = 0 has no mode to fit, and says so',
         () => { const S = new FaradayDisc({ m: 0, nr: 24, nz: 12, ...CELL });
                 S.eta.fill(1e-9); return S.surfaceModeFit(); },
         'no radial structure');
  {
    const S = new FaradayDisc({ m: 12, nr: 24, nz: 12, ...CELL });
    S.eta.fill(1e-9);
    const fit = S.surfaceModeFit();
    ok(fit.err > 1,
       'while at m = 12 a constant elevation does fit -- the azimuthal term acts '
       + 'on it -- and the residual says it is not a mode',
       `residual ${fit.err.toExponential(3)}`);
  }
  throws('an identically zero surface has nothing to fit at any m',
         () => { const S = new FaradayDisc({ m: 12, nr: 24, nz: 12, ...CELL });
                 return S.surfaceModeFit(); },
         'identically zero');
}
{
  /* Extending an Arnoldi space must give exactly what building it from scratch
     would. If it does not, escalating the Krylov dimension is not free -- it is
     wrong -- and the escalation the Floquet driver now does on its own would be
     reporting the eigenvalues of a basis that is not orthonormal. */
  const n = 40;
  const A = (v, out) => {
    for (let i = 0; i < n; i++){
      let x = (1 + 0.05*i)*v[i];
      if (i > 0) x += 0.7*v[i-1];
      if (i + 1 < n) x += 0.3*v[i+1];
      out[i] = x + 0.02*v[n-1-i];
    }
  };
  const v0 = new Float64Array(n);
  for (let i = 0; i < n; i++) v0[i] = Math.cos(0.37*i) + 0.11;

  let calls = 0;
  const counted = (v, out) => { calls++; A(v, out); };
  const six = F.arnoldi(counted, v0, 6);
  const atSix = calls;
  const extended = F.arnoldi(counted, v0, 12, six);
  const toExtend = calls - atSix;
  calls = 0;
  const fresh = F.arnoldi(counted, v0, 12);

  ok(atSix === 6 && calls === 12 && toExtend === 6,
     'extending a 6-dimensional space to 12 costs six more operator '
     + 'applications, not twelve',
     `6 then ${toExtend} to extend, against ${calls} to rebuild`);
  let worst = 0;
  for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++)
    worst = Math.max(worst, Math.abs(extended.Hm[i][j] - fresh.Hm[i][j]));
  ok(worst === 0,
     'and the extended Hessenberg matrix is identical to the rebuilt one, bit '
     + 'for bit',
     `worst difference ${worst}`);
  ok(six.V.length === 7,
     'the previous result is left able to be extended, carrying its next basis '
     + 'vector', `${six.V.length} vectors for a 6-dimensional space`);
  ok(six.used === 6,
     'and unchanged by the extension that read it',
     `used ${six.used}, V ${six.V.length}`);

  /* Three hops must land where one does. */
  const hop = F.arnoldi(A, v0, 12, F.arnoldi(A, v0, 8, F.arnoldi(A, v0, 4)));
  let worstHop = 0;
  for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++)
    worstHop = Math.max(worstHop, Math.abs(hop.Hm[i][j] - fresh.Hm[i][j]));
  ok(worstHop === 0, 'and 4 to 8 to 12 lands on the same matrix as 12 outright',
     `worst difference ${worstHop}`);

  /* A space that closed cannot be extended, and says so rather than reading a
     basis vector that was never built. */
  const B = (v, out) => {
    out.fill(0);
    out[0] = 2*v[0] + v[1];
    out[1] = v[0] + 3*v[1] + v[2];
    out[2] = v[1] + 4*v[2];
  };
  const bv = new Float64Array(n); bv[0] = 1; bv[1] = 0.5; bv[2] = -0.25;
  const closed = F.arnoldi(B, bv, 8);
  ok(closed.breakdown && closed.used === 3,
     'an operator with a three-dimensional invariant subspace breaks down at '
     + 'three', `used ${closed.used}, breakdown ${closed.breakdown}`);
  throws('and extending that closed space is refused',
         () => F.arnoldi(B, bv, 12, closed), 'space that closed');
  throws('as is asking for a dimension that is not an extension',
         () => F.arnoldi(A, v0, 12, fresh), 'not an extension');
}
{
  /* The grading reaches the solver. It did not once: floquetDisc built its grid
     without forwarding rStretch and zStretch, so every run used the solver's
     default 2.2/2.2 and a grid sized to put 2.85 cells in the floor layer was
     integrated with 1.22 there. Nothing announced it. The refusal message names
     the cells it measured, and it is reached before any integration, so this
     costs nothing and fails if the forwarding is dropped again. */
  const cellsFromRefusal = zStretch => {
    try {
      F.floquetDisc({ nr: 16, nz: 8, ...CELL, m: 12, accel: 7, k: K8,
                      omegaD: 2*Math.PI*111, contact: 'free', zStretch });
    } catch (e){
      const mt = /([0-9.]+) cells at the floor/.exec(e.message);
      if (!mt) throw new Error(`the refusal did not report floor cells: ${e.message}`);
      return parseFloat(mt[1]);
    }
    throw new Error('a 16x8 grid was not refused at m = 12');
  };
  const mild = cellsFromRefusal(1.0), severe = cellsFromRefusal(3.8);
  ok(severe > 1.5*mild,
     'a harder vertical grading puts substantially more cells in the floor '
     + 'Stokes layer, so the grading the caller asks for is the grading solved',
     `${mild.toFixed(3)} cells at stretch 1.0 against ${severe.toFixed(3)} at 3.8`);
}

console.log('\n' + '-'.repeat(66));
if (failures.length){
  console.log(`${pass} passed, ${failures.length} FAILED\n`);
  for (const f of failures) console.log('  FAIL  ' + f);
  process.exit(1);
}
console.log(`${pass} checks passed, 0 failed`);
