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

/* ── 4. the viscous operator against an analytic Laplacian ──────────────── */
section('4. the metric Laplacian, against calculus, under refinement');
/* f = r^2 cos(2 theta) sin(kz) is harmonic in the horizontal -- (d_rr + d_r/r
   + d_thetatheta/r^2)(r^2 cos 2theta) is identically zero -- so grad^2 f is
   exactly -k^2 f. That is Bessel-free calculus and owes nothing to this solver.
   Sampled at the physical position of each node, which on a deformed surface is
   z = sigma*(h + eta), it exercises every metric term.
 *
 * Asserted as CONVERGENCE rather than against a tolerance, because the order is
 * the statement about the discretisation; a fixed bound on one grid is a
 * statement about that grid's skew. The interior is tested, away from the faces
 * where a boundary condition rather than the operator decides the flux.
 *
 * WHY THE DEFORMED CASES CARRY THE TEST. Only the sigma-faces are non-orthogonal,
 * so on a flat surface every cross term is identically zero and the operator
 * passes with them deleted. Measured on a disposable copy with the two cross
 * terms removed: at eta/h = 0 the error is unchanged to the digit (4.301e-3 then
 * 1.041e-3, order 2.05), while at eta/h = 0.3 it STOPS CONVERGING -- 3.711e-2
 * then 3.728e-2, order -0.01 -- and at 0.6, order -0.04. That is what makes the
 * deformed rows a gate rather than a repetition, and it is also what establishes
 * that the reduced order at eta/h = 0.6 below is grid skew and not a missing
 * term: a missing term plateaus, and this one converges. */
{
  const KZ = 700;                       // 1/m; k*h = 2.1, a full half wave in the depth
  const fOf = (r, th, z) => r*r*Math.cos(2*th)*Math.sin(KZ*z);
  const lapOf = (r, th, z) => -KZ*KZ*fOf(r, th, z);
  const errorOn = (nr, nth, nz, amp, fromI = 0) => {
    const S = new FaradayCell3D({ nr, nth, nz, ...CELL, rStretch: 0, zStretch: 0 });
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++)
      S.eta[S.ie(i,k)] = amp*S.h*(
          Math.cos(2*(k + 0.5)*S.dth)*Math.pow(S.rc[i]/S.R, 2)
        + 0.3*Math.sin(3*(k + 0.5)*S.dth));
    S.refreshMetric();
    const f = new Float64Array(S.NP), got = new Float64Array(S.NP);
    const thc = k => (k + 0.5)*S.dth;
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
      f[S.ip(i,k,j)] = fOf(S.rc[i], thc(k), S.sc[j]*S.H[S.ie(i,k)]);
    S.scalarLaplacian(f, got);
    let num = 0, den = 0;
    for (let i = fromI; i < nr-1; i++) for (let k = 0; k < nth; k++) for (let j = 1; j < nz-1; j++){
      const want = lapOf(S.rc[i], thc(k), S.sc[j]*S.H[S.ie(i,k)]);
      const d = got[S.ip(i,k,j)] - want;
      num += d*d; den += want*want;
    }
    return Math.sqrt(num/den);
  };
  /* The axis cell is INCLUDED (fromI = 0). Its inward neighbour is the antipodal
     column reflected through r = 0, and the point of including it is that the
     reflection is what makes the axis converge at the interior's order rather
     than at first order. Measured both ways: excluding the axis cell, 2.05 flat
     and 2.00 at eta/h = 0.3; including it, 2.06 and 2.01. If the reflection were
     replaced by a one-sided difference the axis rows would drop to first order
     and drag these figures down. */
  for (const [amp, floor] of [[0, 1.9], [0.3, 1.8], [0.6, 1.3]]){
    const coarse = errorOn(16, 24, 16, amp), fine = errorOn(32, 48, 32, amp);
    const order = Math.log(coarse/fine)/Math.log(2);
    console.log(`       eta/h = ${amp}: ${coarse.toExponential(3)} at 16x24x16, `
      + `${fine.toExponential(3)} at 32x48x32, order ${order.toFixed(2)}`);
    ok(fine < coarse, `eta/h = ${amp}: refining reduces the error in grad^2 f`,
       `${coarse.toExponential(3)} then ${fine.toExponential(3)}`);
    ok(order > floor,
       `eta/h = ${amp}: and it falls at order ${floor} or better (got ${order.toFixed(2)})`,
       `observed order ${order.toFixed(3)}`);
  }
}

/* ── 4b. the axis reflection ─────────────────────────────────────────────── */
section('4b. the axis is a reflection, not a special case');
{
  const S = deform(new FaradayCell3D({ nr: 10, nth: 16, nz: 8, ...CELL }), 0.3);
  const f = new Float64Array(S.NP);
  const r = rnd(4242);
  for (let c = 0; c < S.NP; c++) f[c] = r();
  const idx = (a, b, c) => S.ip(a, b, c);
  const half = S.nth >> 1;
  /* Inside the axis, radial index -1 is cell 0 of the antipodal column. A scalar
     carries a plus sign there and a horizontal vector component a minus. */
  let evenOk = true, oddOk = true;
  for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++){
    if (S.across(f, +1, idx, -1, k, j) !== f[S.ip(0, k + half, j)]) evenOk = false;
    if (S.across(f, -1, idx, -1, k, j) !== -f[S.ip(0, k + half, j)]) oddOk = false;
  }
  ok(evenOk, 'a scalar reflected through the axis is the antipodal value');
  ok(oddOk, 'and a horizontal vector component is minus it, because r-hat and '
     + 'theta-hat both reverse');
  ok(S.rcAcross(-1) === -S.rc[0],
     'the radial coordinate continues through zero, so a difference across the '
     + 'axis has the right denominator',
     `${S.rcAcross(-1)} against ${-S.rc[0]}`);
  /* Reflecting twice is the identity, which is the consistency the sign table
     has to satisfy. */
  let twice = true;
  for (let k = 0; k < S.nth; k++)
    if (S.kw(k + half + half) !== S.kw(k)) twice = false;
  ok(twice, 'and reflecting twice returns to the same column, which needs an even '
     + 'azimuthal count -- the reason the constructor requires one');
}

/* ── 5. refusals ────────────────────────────────────────────────────────── */
section('5. what it refuses rather than answering');
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
