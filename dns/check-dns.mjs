/* Verification status of the Faraday DNS.

   This reports what is established and what is NOT. The solver is not yet
   correct: it reproduces the dispersion relation only to a few percent, with a
   diagnosed cause and a slow convergence rate, and until that closes nothing
   in the page uses it and nothing claims Navier-Stokes is being solved there.
   The checks that DO pass are kept because they are what localises the
   remaining defect.

   Run: node dns/check-dns.mjs
*/
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { FaradayDNS, curvature } = require('./faraday-dns.js');

const G = 9.80665, rho = 998.2, h0 = 0.003, gam = 0.07274;
const wEx = (k, g) => Math.sqrt((G*k + g*k*k*k/rho)*Math.tanh(k*h0));
let pass = 0; const fail = [];
const ok = (c, label, detail) => c ? pass++ : fail.push(`${label}${detail ? '  — ' + detail : ''}`);

console.log('ESTABLISHED\n');

// 1. curvature is the exact one, not the small-slope H_xx
{
  const n = 256, L = 1, dx = L/n, k = 2*Math.PI/L, A = 1e-4;
  const H = new Float64Array(n), out = new Float64Array(n);
  for (let i = 0; i < n; i++) H[i] = h0 + A*Math.cos(k*(i+0.5)*dx);
  curvature(H, dx, out);
  let worst = 0;
  for (let i = 0; i < n; i++){
    const x = (i+0.5)*dx, Hx = -A*k*Math.sin(k*x), Hxx = -A*k*k*Math.cos(k*x);
    const ex = Hxx/Math.pow(1 + Hx*Hx, 1.5);
    worst = Math.max(worst, Math.abs(out[i] - ex)/Math.abs(ex));
  }
  ok(worst < 1e-4, 'curvature matches the exact expression', `worst ${worst.toExponential(2)}`);
  console.log(`  curvature vs exact, 256 points: ${worst.toExponential(2)} relative`);
}

// 2. the divergence and the gradient used in the projection are exact adjoints
{
  const S = new FaradayDNS({nx:16, ns:12, L:0.006, h0, rho, nu:1.0036e-6, gamma:gam});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.15*Math.sin(2*Math.PI*i/S.nx));
  const R = () => Math.random() - 0.5;
  const u = new Float64Array(S.nx*S.ns).map(R), w = new Float64Array(S.nx*(S.ns+1)).map(R);
  const q = new Float64Array(S.nx*S.ns).map(R);
  const d = new Float64Array(S.nx*S.ns); S.divergence(u, w, d);
  let lhs = 0; for (let c = 0; c < d.length; c++) lhs += d[c]*q[c];
  const gu = new Float64Array(u.length), gw = new Float64Array(w.length); S.gradientT(q, gu, gw);
  let rhs = 0;
  for (let c = 0; c < u.length; c++) rhs += u[c]*gu[c];
  for (let c = 0; c < w.length; c++) rhs += w[c]*gw[c];
  const rel = Math.abs(lhs + rhs)/Math.abs(lhs);
  ok(rel < 1e-14, 'D and G are exact adjoints on a deformed surface', rel.toExponential(2));
  console.log(`  <Dv,q> + <v,Gq> = ${rel.toExponential(2)} relative (15% surface deformation)`);
}

// 3. the pressure operator is symmetric and negative definite, so CG is valid
{
  const S = new FaradayDNS({nx:12, ns:10, L:0.006, h0, rho, nu:1.0036e-6, gamma:gam});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.12*Math.sin(2*Math.PI*i/S.nx));
  const n = S.nx*S.ns;
  const a = new Float64Array(n).map(() => Math.random()-0.5);
  const b = new Float64Array(n).map(() => Math.random()-0.5);
  const La = new Float64Array(n), Lb = new Float64Array(n);
  S.applyL(a, La); S.applyL(b, Lb);
  let ab = 0, ba = 0, qd = 0;
  for (let i = 0; i < n; i++){ ab += La[i]*b[i]; ba += a[i]*Lb[i]; qd += a[i]*La[i]; }
  const sym = Math.abs(ab - ba)/Math.abs(ab);
  ok(sym < 1e-12, 'L is symmetric', sym.toExponential(2));
  ok(qd < 0, 'L is negative definite');
  console.log(`  L symmetry ${sym.toExponential(2)} relative, <a,La> = ${qd.toExponential(3)} < 0`);
}

// 4. the projection is exact: the corrected field is divergence-free
{
  const S = new FaradayDNS({nx:32, ns:24, L:0.006, h0, rho, nu:1.0036e-6, gamma:gam});
  const k = 2*Math.PI/S.L;
  for (let i = 0; i < S.nx; i++) S.H[i] = h0 + 1e-7*Math.cos(k*(i+0.5)*S.dx);
  for (let n = 0; n < 20; n++) S.step(2e-6);
  const dv = S.maxDivergence();
  ok(dv < 1e-9, 'velocity is divergence-free after projection', dv.toExponential(2));
  console.log(`  max|div| after 20 steps: ${dv.toExponential(2)} (CG residual ${S.cgResidual.toExponential(1)})`);
}

// 5. hydrostatic balance: flat and at rest must stay flat and at rest
{
  const S = new FaradayDNS({nx:16, ns:20, L:0.006, h0, rho, nu:0, gamma:gam});
  for (let n = 0; n < 100; n++) S.step(1e-5);
  let wmax = 0, dH = 0;
  for (let i = 0; i < S.w.length; i++) wmax = Math.max(wmax, Math.abs(S.w[i]));
  for (let i = 0; i < S.nx; i++) dH = Math.max(dH, Math.abs(S.H[i] - h0));
  ok(wmax < 1e-15, 'a flat surface at rest stays at rest', `max|w| ${wmax.toExponential(2)}`);
  ok(dH === 0, 'and the depth does not drift', dH.toExponential(2));
  console.log(`  100 steps under gravity: max|w| ${wmax.toExponential(2)}, depth drift ${dH.toExponential(2)}`);
  // the pressure gradient is right; the profile carries a known constant offset
  const i = 0, ns = S.ns, H = S.H[i];
  const pTop = S.p[i*ns + ns-1], pBot = S.p[i*ns + 0];
  const zTop = ((ns-1)+0.5)/ns*H, zBot = 0.5/ns*H;
  const dpMeas = pBot - pTop, dpEx = rho*G*(zTop - zBot);
  const relGrad = Math.abs(dpMeas - dpEx)/dpEx;
  const offset = pTop - rho*G*(H - zTop);
  ok(relGrad < 1e-3, 'the hydrostatic pressure GRADIENT is exact', relGrad.toExponential(2));
  console.log(`  pressure gradient error ${relGrad.toExponential(2)}; constant offset ${offset.toFixed(4)} Pa`
    + ` = rho g H ds/2 = ${(rho*G*H/(2*ns)).toFixed(4)} Pa`);
}

console.log('\nNOT ESTABLISHED — the solver does not yet reproduce the dispersion relation\n');
{
  const run = (nx, ns, dt, L, g, nT) => {
    const k = 2*Math.PI/L, A = 1e-9;
    const S = new FaradayDNS({nx, ns, L, h0, rho, nu:0, gamma:g});
    for (let i = 0; i < nx; i++) S.H[i] = h0 + A*Math.cos(k*(i+0.5)*S.dx);
    const we = wEx(k, g), T = 2*Math.PI/we, steps = Math.ceil(nT*T/dt);
    let prev = null; const cr = [];
    for (let n = 0; n < steps; n++){
      S.step(dt);
      let m = 0; for (let i = 0; i < nx; i++) m += S.H[i]; m /= nx;
      const e = S.H[0] - m;
      if (prev !== null && prev > 0 && e <= 0){ const f = prev/(prev - e); cr.push(S.t - dt + f*dt); }
      prev = e;
    }
    return cr.length >= 2 ? (2*Math.PI/(cr[1]-cr[0]))/we : NaN;
  };
  const rows = [
    ['gravity, L=60mm (kh=0.31, shallow)', 0.060, 0,   6e-5],
    ['gravity, L=20mm (kh=0.94)         ', 0.020, 0,   3e-5],
    ['gravity, L= 6mm (kh=3.14, deep)   ', 0.006, 0,   2e-5],
    ['gravity+capillary, L=6mm          ', 0.006, gam, 1e-5],
  ];
  for (const [tag, L, g, dt] of rows){
    const r = run(32, 24, dt, L, g, 2.2);
    console.log(`  ${tag}: omega/omega_exact = ${r.toFixed(4)}`);
  }
  /* Refinement, kept small enough to run: the point is the RATE, not the
     final value. Measured over a wider sweep offline the error falls from
     4.6e-2 at ns=24 to 3.1e-2 at ns=48 -- a factor of 1.5 for a doubling,
     slower than the first order the boundary placement alone would give. */
  console.log('\n  refining (nx=32, capillary case):');
  let prev = null;
  for (const ns of [12, 24]){
    const r = run(32, ns, 8e-6, 0.006, gam, 2.2);
    const e = Math.abs(1 - r);
    console.log(`    ns=${String(ns).padStart(2)}: ratio ${r.toFixed(5)}, error ${e.toExponential(2)}`
      + (prev === null ? '' : `, fell by ${(prev/e).toFixed(2)}x`));
    prev = e;
  }
  fail.push('dispersion relation is not reproduced: omega is 5% to 28% low, '
    + 'worst for long shallow waves, and refinement halves the error only slowly');
}

console.log(`\n  DIAGNOSIS. The pressure Dirichlet sits at the ghost CENTRE, half a cell
  above the free surface, rather than on it -- visible above as the exact
  constant offset rho g H ds/2 in the hydrostatic profile. That is an O(ds)
  boundary placement error and it makes the restoring force too weak, which is
  the right sign and the right direction (worse for long waves, where the
  restoring force is the whole of the dynamics). It does not account for all of
  a 28% error on its own, so at least one further defect is present and has not
  been found. The fix is a pressure grid whose top node lands on the surface,
  which changes the staggering rather than the physics.

  Until that closes: the page does not use this solver, and nothing claims it
  does.\n`);

console.log('─'.repeat(66));
console.log(`${pass} checks passed, ${fail.length} outstanding`);
for (const f of fail) console.log('  OUTSTANDING  ' + f);
process.exit(0);
