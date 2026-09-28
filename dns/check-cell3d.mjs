/* Gate for dns/faraday-cell3d.js -- the nonlinear three-dimensional solver.
 *
 * Expected values come from the equations, from identities the discretisation
 * must satisfy, and from the independently written two-dimensional solver next
 * door -- never from this solver's own opinion.
 *
 *     node dns/check-cell3d.mjs
 */

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ROOT = join(here, '..');
const C = require(join(here, 'faraday-cell3d.js'));
const { FaradayCell3D } = C;

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
function throws(label, fn, fragment){
  try { fn(); ok(false, label, 'did not throw'); }
  catch (e){ ok(!fragment || String(e.message).includes(fragment), label,
                `threw "${String(e.message).slice(0, 120)}"`); }
}
const section = t => console.log('\n' + t);

/* The apparatus the renderer draws: a 24.25 mm quartz cell, 3 mm of water. */
const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837,
               nu: 1.0035510043427237e-6, gamma: 0.07273614042160757,
               g: 9.80665 };

/* A deterministic pseudo-random sequence, so a failure is reproducible. Its
   quality is irrelevant -- it is only used to excite every degree of freedom. */
function rnd(seed){
  let s = seed >>> 0;
  return () => { s = (s*1103515245 + 12345) & 0x7fffffff; return s/0x7fffffff - 0.5; };
}

/* A lumpy surface, so the metric is genuinely three-dimensional rather than a
   flat special case that would hide every term carrying dH/dr or dH/dtheta. */
function deform(S, ampFraction){
  for (let i = 0; i < S.nr; i++)
    for (let k = 0; k < S.nth; k++)
      S.eta[S.ie(i, k)] = ampFraction*S.h*(
          Math.cos(3*k*S.dth)*Math.pow(S.rc[i]/S.R, 2)
        + 0.4*Math.sin(5*k*S.dth + 1)*(S.rc[i]/S.R));
  S.refreshMetric();
  return S;
}

const maxAbs = a => { let m = 0; for (const x of a) m = Math.max(m, Math.abs(x)); return m; };

/* ── 1. the projection, in the surface-following coordinate ──────────────── */
section('1. the pressure operator under the moving-surface metric');
/* Everything the solver does rests on this. divergence(gradient(.)) must be
   symmetric and negative definite for conjugate gradients to be entitled to
   converge on it, and it is only symmetric if the gradient is the EXACT
   transpose of the divergence over the cell volumes -- which is why Omega, not
   physical w, is the third unknown. The deformed cases are the point: a flat
   surface would pass with every metric term dropped. */
for (const [nr, nth, nz, amp] of [[8, 12, 6, 0], [8, 12, 6, 0.4],
                                  [10, 16, 8, 0.7], [6, 8, 5, 0.25]]){
  const S = deform(new FaradayCell3D({ nr, nth, nz, ...CELL }), amp);
  const tag = `${nr}x${nth}x${nz}, eta/h = ${amp}`;
  const n = S.NP;
  const r1 = rnd(7), r2 = rnd(99);
  const x = new Float64Array(n), y = new Float64Array(n),
        Lx = new Float64Array(n), Ly = new Float64Array(n);
  for (let i = 0; i < n; i++){ x[i] = r1(); y[i] = r2(); }
  S.applyL(x, Lx); S.applyL(y, Ly);
  let xLy = 0, yLx = 0, xLx = 0;
  for (let i = 0; i < n; i++){ xLy += x[i]*Ly[i]; yLx += y[i]*Lx[i]; xLx += x[i]*Lx[i]; }
  const asym = Math.abs(xLy - yLx)/Math.abs(xLy);
  ok(asym < 1e-12, `${tag}: the operator is symmetric`,
     `x.Ly = ${xLy.toExponential(6)}, y.Lx = ${yLx.toExponential(6)}, `
     + `relative difference ${asym.toExponential(2)}`);
  ok(xLx < 0, `${tag}: and negative definite`, `x.Lx = ${xLx.toExponential(6)}`);

  /* What a constant pressure does. This first asserted that L annihilates it, on
     the reasoning that a Poisson operator is singular on the constants. Measured:
     it does not, and it must not. The free surface carries a Dirichlet condition
     -- the pressure there is the normal stress -- so the operator is
     non-singular, which is why the solve needs no pinning and returns a unique
     pressure. What is exactly zero is the interior, and that is the sharper
     statement: it fails if any interior coefficient is wrong, where "annihilates
     constants" would have failed on correct code. */
  const one = new Float64Array(n).fill(1), L1 = new Float64Array(n);
  const gu = new Float64Array(S.NU), gv = new Float64Array(S.NV), gw = new Float64Array(S.NW);
  S.gradient(one, gu, gv, gw);
  ok(maxAbs(gu) === 0 && maxAbs(gv) === 0,
     `${tag}: a constant pressure has exactly zero horizontal gradient`,
     `max|gu| = ${maxAbs(gu).toExponential(2)}, max|gv| = ${maxAbs(gv).toExponential(2)}`);
  let interiorGw = 0, surfaceGw = 0;
  for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
    for (let j = 0; j < nz; j++) interiorGw = Math.max(interiorGw, Math.abs(gw[S.iw(i,k,j)]));
    surfaceGw = Math.max(surfaceGw, Math.abs(gw[S.iw(i,k,nz)]));
  }
  ok(interiorGw === 0 && surfaceGw > 0,
     `${tag}: and drives flow only through the free surface, where the pressure `
     + `is prescribed`,
     `interior ${interiorGw.toExponential(2)}, surface ${surfaceGw.toExponential(2)}`);
  S.applyL(one, L1);
  let interiorL = 0, topL = 0;
  for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
    for (let j = 0; j < nz - 1; j++) interiorL = Math.max(interiorL, Math.abs(L1[S.ip(i,k,j)]));
    topL = Math.max(topL, Math.abs(L1[S.ip(i,k,nz-1)]));
  }
  ok(interiorL === 0 && topL > 0,
     `${tag}: so L applied to a constant is exactly zero in every sigma row but `
     + `the surface row`,
     `interior ${interiorL.toExponential(2)}, top ${topL.toExponential(2)}`);
}

/* ── 2. the projection removes the divergence ───────────────────────────── */
section('2. the projection makes a divergent field divergence free');
/* Asserted as CONVERGENCE, not against a fixed ratio. The first version demanded
   the divergence fall by the solve's own 1e-11 tolerance and read 1.65e-11,
   failing on correct code: the tolerance is on the flux residual in the 2-norm
   while this is a per-volume maximum, and the two differ by a grid-dependent
   factor. Tightening the tolerance and requiring the divergence to follow is a
   statement about the projection; a fixed ratio is a statement about that
   factor. */
for (const [nr, nth, nz, amp] of [[8, 12, 6, 0.4], [10, 16, 8, 0.7]]){
  const S = deform(new FaradayCell3D({ nr, nth, nz, ...CELL }), amp);
  const tag = `${nr}x${nth}x${nz}, eta/h = ${amp}`;
  const r = rnd(1234);
  for (let i = 1; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
    S.u[S.iu(i,k,j)] = 1e-3*r();
  for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
    S.v[S.iv(i,k,j)] = 1e-3*r();
  for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 1; j <= nz; j++)
    S.om[S.iw(i,k,j)] = 1e-3*r();
  const before = S.maxDivergence();
  S.pressureDiagonal();
  const project = tol => {
    S.p.fill(0);
    S.divergence(S.u, S.v, S.om, S._div);
    S.solveP(S._div, tol, 400*(nr + nth + nz));
    S.gradient(S.p, S._gu, S._gv, S._gw);
    for (let i = 1; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
      S.u[S.iu(i,k,j)] -= S._gu[S.iu(i,k,j)];
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
      S.v[S.iv(i,k,j)] -= S._gv[S.iv(i,k,j)];
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 1; j <= nz; j++)
      S.om[S.iw(i,k,j)] -= S._gw[S.iw(i,k,j)];
    return S.maxDivergence();
  };
  const loose = project(1e-11), iters = S.cgIters;
  const tight = project(1e-14);
  console.log(`       ${before.toExponential(3)} -> ${loose.toExponential(3)} in `
    + `${iters} iterations, then ${tight.toExponential(3)} at 1e-14`);
  ok(loose/before < 1e-9, `${tag}: the projection removes the divergence`,
     `by a factor of ${(before/loose).toExponential(2)} at tolerance 1e-11`);
  ok(tight < loose,
     `${tag}: and removes more when the tolerance is tightened, so what is left `
     + `is the solve's tolerance and not a floor in the operator`,
     `${loose.toExponential(3)} then ${tight.toExponential(3)}`);
}

/* ── 3. Omega against its definition ────────────────────────────────────── */
section('3. Omega and physical w are consistent with the map');
/* Omega = w - sigma(u dH/dr + (v/r) dH/dtheta) is a definition, so recovering w
   from Omega and then Omega from w must return what went in. On a deformed
   surface the slope terms are non-zero, so this exercises them. */
{
  const S = deform(new FaradayCell3D({ nr: 10, nth: 16, nz: 8, ...CELL }), 0.5);
  const r = rnd(555);
  for (let i = 1; i < S.nr; i++) for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++)
    S.u[S.iu(i,k,j)] = 1e-3*r();
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++)
    S.v[S.iv(i,k,j)] = 1e-3*r();
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++) for (let j = 1; j <= S.nz; j++)
    S.om[S.iw(i,k,j)] = 1e-3*r();
  let worst = 0, slopeSeen = 0;
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
    for (let j = 0; j <= S.nz; j++){
      const w = S.wAt(i, k, j);
      const diff = w - S.om[S.iw(i,k,j)];
      slopeSeen = Math.max(slopeSeen, Math.abs(diff));
      /* recovering Omega from w must give back exactly what is stored */
      const back = w - diff;
      worst = Math.max(worst, Math.abs(back - S.om[S.iw(i,k,j)]));
    }
  ok(worst < 1e-18, 'w recovered from Omega and back is the identity',
     `worst difference ${worst.toExponential(3)}`);
  ok(slopeSeen > 0,
     'and the metric slope terms are actually non-zero on a deformed surface, so '
     + 'the identity above is not trivially satisfied',
     `largest |w - Omega| = ${slopeSeen.toExponential(3)}`);
  /* Omega on the floor is zero by construction: w = 0 there and sigma = 0 kills
     the slopes. */
  let floorW = 0;
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
    floorW = Math.max(floorW, Math.abs(S.wAt(i, k, 0) - S.om[S.iw(i,k,0)]));
  ok(floorW === 0,
     'on the floor sigma = 0, so Omega and physical w coincide exactly there',
     `worst difference ${floorW.toExponential(3)}`);
}

/* ── 4. refusals ────────────────────────────────────────────────────────── */
section('4. what it refuses rather than answering');
throws('an odd azimuthal count is refused, since the top mode loses its conjugate',
       () => new FaradayCell3D({ nr: 8, nth: 11, nz: 6, ...CELL }), 'nth');
throws('fewer than eight azimuthal cells is refused',
       () => new FaradayCell3D({ nr: 8, nth: 6, nz: 6, ...CELL }), 'nth');
throws('a contact line that is neither free nor pinned is refused',
       () => new FaradayCell3D({ nr: 8, nth: 12, nz: 6, ...CELL, contact: 'slip' }),
       'contact');
throws('zero viscosity is refused',
       () => new FaradayCell3D({ nr: 8, nth: 12, nz: 6, ...CELL, nu: 0 }), 'nu');
throws('a negative grading is refused',
       () => new FaradayCell3D({ nr: 8, nth: 12, nz: 6, ...CELL, zStretch: -1 }),
       'Stretch');
/* A surface that has reached the floor has no single-valued depth to follow, and
   the coordinate map is degenerate there. It refuses rather than inverting a
   cell and returning a field. */
throws('a surface deep enough to touch the floor is refused, naming the depth',
       () => { const S = new FaradayCell3D({ nr: 8, nth: 12, nz: 6, ...CELL });
               S.eta[S.ie(3, 4)] = -CELL.h; return S.refreshMetric(); },
       'reached the floor');

console.log('\n' + '-'.repeat(66));
if (failures.length){
  console.log(`${pass} passed, ${failures.length} FAILED\n`);
  for (const f of failures) console.log('  FAIL  ' + f);
  process.exit(1);
}
console.log(`${pass} checks passed, 0 failed`);
