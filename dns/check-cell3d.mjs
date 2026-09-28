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

/* ── 4b. the axis reflection ─────────────────────────────────────────────── */
section('4b. the axis is a reflection, not a special case');
/* The reflection lives in colValueAtZ, which every cross-column comparison goes
   through. Radial index -1 is column 0 of the antipodal half, carrying the family's
   own sign: plus for pressure and the vertical component, minus for the two
   horizontal ones, because r-hat and theta-hat both reverse while z-hat does not.
   Exact for every azimuthal mode including m = 1 -- the one a single-mode solver
   cannot handle, and which dns/faraday-disc.js refuses outright.

   Both sides of the comparison go through the SAME reconstruction, so the only
   thing under test is which column is read and with what sign. Comparing against
   the stored column value instead was too strong an expectation and failed on
   correct code at 1e-15: a quadratic Lagrange reconstruction of three equal values
   returns that value only to round-off, because its weights sum to one exactly
   only in exact arithmetic. Routing both sides through the reconstruction removes
   that term and leaves an equality that is exact. */
{
  const S = deform(new FaradayCell3D({ nr: 10, nth: 16, nz: 8, ...CELL }), 0.3);
  const half = S.nth >> 1;
  const r = rnd(4242);
  for (const [name, fam, field, want] of [
        ['pressure',  S.FAM.p, new Float64Array(S.NP), +1],
        ['radial',    S.FAM.u, new Float64Array(S.NU), -1],
        ['azimuthal', S.FAM.v, new Float64Array(S.NV), -1],
        ['vertical',  S.FAM.w, new Float64Array(S.NW), +1]]){
    const col = new Float64Array(fam.rn.length*S.nth);
    for (let a = 0; a < fam.rn.length; a++)
      for (let k = 0; k < S.nth; k++){
        col[a*S.nth + k] = r();
        for (let b = 0; b < fam.sn.length; b++)
          field[fam.idx(a, k, b)] = col[a*S.nth + k];
      }
    let worst = 0, mag = 0;
    for (let k = 0; k < S.nth; k++){
      const z = 0.5*S.h;
      const got = S.colValueAtZ(field, fam, -1, k, z, 1);
      const expect = want*S.colValueAtZ(field, fam, 0, k + half, z, 1);
      worst = Math.max(worst, Math.abs(got - expect));
      mag = Math.max(mag, Math.abs(expect));
    }
    ok(worst === 0 && mag > 0,
       `the ${name} field reflects through the axis to the antipodal column with `
       + `sign ${want > 0 ? '+1' : '-1'}`,
       `worst difference ${worst.toExponential(3)} on a scale of ${mag.toExponential(3)}`);
  }
  ok(S.FAM.u.axisSign === -1 && S.FAM.v.axisSign === -1
     && S.FAM.p.axisSign === +1 && S.FAM.w.axisSign === +1,
     'and the sign table is odd for the horizontal components, even for the others');
  let twice = true;
  for (let k = 0; k < S.nth; k++)
    if (S.kw(k + half + half) !== S.kw(k)) twice = false;
  ok(twice,
     'reflecting twice returns to the same column, which needs an even azimuthal '
     + 'count -- the reason the constructor requires one');
}

/* ── 4c. node geometry and the axis value of the radial velocity ─────────── */
section('4c. the staggered node families, and u_r at the axis');
{
  const S = new FaradayCell3D({ nr: 10, nth: 16, nz: 6, ...CELL });
  /* Every family's control-volume boundary list must be exactly one longer than
     its node list, or some node has no cell and its flux balance is incomplete.
     Cheap, and it catches an off-by-one in the descriptor that would otherwise
     surface as a wrong answer at one boundary. */
  let shaped = true;
  const names = ['p', 'u', 'v', 'w'];
  for (const n of names){
    const f = S.FAM[n];
    if (f.rb.length !== f.rn.length + 1) shaped = false;
    if (f.sb.length !== f.sn.length + 1) shaped = false;
    if (!(f.rLo >= 0 && f.rHi < f.rn.length)) shaped = false;
    if (!(f.sLo >= 0 && f.sHi < f.sn.length)) shaped = false;
  }
  ok(shaped, 'each family has one more control-volume boundary than it has nodes, '
     + 'and its unknown range lies inside its node list',
     names.map(n => `${n}: rn ${S.FAM[n].rn.length}/rb ${S.FAM[n].rb.length}`).join(', '));
  ok(S.FAM.u.axisSign === -1 && S.FAM.v.axisSign === -1
     && S.FAM.p.axisSign === +1 && S.FAM.w.axisSign === +1,
     'and the reflection signs are odd for the two horizontal components and even '
     + 'for pressure and the vertical one');
  ok(S.FAM.v.thOff === 0 && S.FAM.p.thOff === 0.5 && S.FAM.u.thOff === 0.5
     && S.FAM.w.thOff === 0.5,
     'the azimuthal velocity sits on the theta faces and everything else on the '
     + 'theta centres');

  /* u_r at the axis. A single-valued vector field needs u_r(0, theta) =
     -u_r(0, theta + pi); the axis row is built as the antisymmetric part of an
     extrapolation, so it satisfies that by construction. It must also NOT be
     identically zero, which is what a single-mode solver is forced to assume and
     why dns/faraday-disc.js refuses m = 1: an m = 1 field is non-zero at the axis,
     and here it survives. */
  const half = S.nth >> 1;
  for (let i = 1; i < S.nr; i++) for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++)
    S.u[S.iu(i,k,j)] = Math.cos(k*S.dth)*S.rf[i]
                     + 0.3*Math.cos(3*k*S.dth)*S.rf[i]*S.rf[i];
  S.axisU();
  let anti = 0, mag = 0;
  for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++){
    anti = Math.max(anti, Math.abs(S.u[S.iu(0,k,j)] + S.u[S.iu(0,k+half,j)]));
    mag = Math.max(mag, Math.abs(S.u[S.iu(0,k,j)]));
  }
  ok(anti === 0,
     'u_r at the axis is exactly antisymmetric under theta -> theta + pi',
     `worst |u(0,k) + u(0,k+pi)| = ${anti.toExponential(2)}`);
  ok(mag > 0,
     'and is not identically zero, so the m = 1 component that lives at the axis '
     + 'is carried rather than assumed away',
     `max |u(0,k)| = ${mag.toExponential(3)}`);
  /* A purely m = 3 field has no axis component, so the same machinery must return
     zero there -- the complement of the check above. */
  S.u.fill(0);
  for (let i = 1; i < S.nr; i++) for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++)
    S.u[S.iu(i,k,j)] = Math.cos(3*k*S.dth)*S.rf[i];
  S.axisU();
  let m3 = 0;
  for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++)
    m3 = Math.max(m3, Math.abs(S.u[S.iu(0,k,j)]));
  ok(m3 < 1e-18,
     'while a purely m = 3 field leaves the axis exactly zero, which is the '
     + 'complement of the check above',
     `max |u(0,k)| = ${m3.toExponential(2)}`);
}

/* ── 4d. the metric interpolation ────────────────────────────────────────── */
section('4d. H interpolated at face midpoints, against the analytic surface');
/* The operator needs H and its two horizontal slopes at face midpoints, where a
   bracket-based difference is centred. Checked against a surface whose H is known
   in closed form.
 *
 * Normalised by the field's own scale, not pointwise: dH/dtheta is proportional to
 * sin(2 theta), which at theta = pi/2 evaluates to 1.2e-16, and a pointwise
 * relative error divided by that reported 156 on correct code. The first version
 * of this check did exactly that. */
{
  const Hf = (r, th) => CELL.h*(1 + 0.3*Math.cos(2*th)*Math.pow(r/CELL.R, 2));
  const Hrf = (r, th) => CELL.h*0.3*Math.cos(2*th)*2*r/(CELL.R*CELL.R);
  const Htf = (r, th) => -CELL.h*0.3*2*Math.sin(2*th)*Math.pow(r/CELL.R, 2);
  const sH = CELL.h, sHr = CELL.h*0.6/CELL.R, sHt = CELL.h*0.6;
  const probe = (nr, nth) => {
    const S = new FaradayCell3D({ nr, nth, nz: 8, ...CELL, rStretch: 0, zStretch: 0 });
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++)
      S.eta[S.ie(i,k)] = Hf(S.rc[i], (k + 0.5)*S.dth) - CELL.h;
    S.refreshMetric();
    let eH = 0, eHr = 0, eHt = 0;
    for (let i = 1; i < nr; i++) for (let k = 0; k < nth; k++){
      const r = 0.5*(S.rc[i-1] + S.rc[i]), th = (k + 0.5)*S.dth, g = S.Hat(r, th);
      eH = Math.max(eH, Math.abs(g.H - Hf(r, th))/sH);
      eHr = Math.max(eHr, Math.abs(g.Hr - Hrf(r, th))/sHr);
    }
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
      const r = S.rc[i], th = k*S.dth, g = S.Hat(r, th);
      eHt = Math.max(eHt, Math.abs(g.Hth - Htf(r, th))/sHt);
    }
    return { eH, eHr, eHt };
  };
  const a = probe(16, 24), b = probe(32, 48);
  const ord = (x, y) => Math.log(x/y)/Math.log(2);
  console.log(`       H ${a.eH.toExponential(2)} -> ${b.eH.toExponential(2)} `
    + `(order ${ord(a.eH, b.eH).toFixed(2)}), dH/dtheta ${a.eHt.toExponential(2)} -> `
    + `${b.eHt.toExponential(2)} (order ${ord(a.eHt, b.eHt).toFixed(2)}), `
    + `dH/dr ${a.eHr.toExponential(2)}`);
  ok(ord(a.eH, b.eH) > 1.8, 'H is second order at face midpoints',
     `order ${ord(a.eH, b.eH).toFixed(3)}`);
  ok(ord(a.eHt, b.eHt) > 1.8, 'and so is dH/dtheta',
     `order ${ord(a.eHt, b.eHt).toFixed(3)}`);
  /* This surface is quadratic in r, and a centred difference differentiates a
     quadratic exactly, so dH/dr is at round-off rather than second order. Asserted
     as such: an order here would be an order on round-off noise. */
  ok(a.eHr < 1e-12 && b.eHr < 1e-12,
     'while dH/dr is exact on this surface, because it is quadratic in r and a '
     + 'centred difference differentiates a quadratic exactly',
     `${a.eHr.toExponential(2)} and ${b.eHr.toExponential(2)}`);
}

/* ── 4e. the metric Laplacian at the staggered nodes ─────────────────────── */
section('4e. the Laplacian at every node family, against calculus');
/* ADMISSIBLE SURFACES ONLY, and that is a physical constraint rather than a
   convenience. A single-valued smooth surface must have its m-th azimuthal
   component vanish as r^m at the axis; a surface like cos(2 theta) with no radial
   factor is multivalued there, and sigma H_theta / r^2 diverges. Probing with one
   reported order -1.99 on a correct operator, which cost a measure-fix cycle to
   understand. The solver will never meet such a surface, because eta is produced by
   the equations rather than imposed.

   Tangential derivatives are taken at a common physical height, which is what makes
   the operator converge at all: see the note on famLaplacian. */
{
  const KZ = 700;
  const fOf = (r, th, z) =>
    (1 + (r/CELL.R)*(r/CELL.R)*Math.cos(2*th))*Math.sin(KZ*z);
  const lapOf = (r, th, z) => -KZ*KZ*fOf(r, th, z);
  /* mixed, and every component admissible: m = 1 as r, m = 2 as r^2, m = 3 as r^3 */
  const surf = (x, th) => 0.6*x*Math.cos(th) + 0.5*x*x*Math.cos(2*th)
                        - 0.3*x*x*x*Math.sin(3*th);
  const errorOn = (famName, nr, nth, nz, amp, skipIn) => {
    const S = new FaradayCell3D({ nr, nth, nz, ...CELL, rStretch: 0, zStretch: 0 });
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++)
      S.eta[S.ie(i,k)] = amp*S.h*surf(S.rc[i]/S.R, (k + 0.5)*S.dth);
    S.refreshMetric();
    const fam = S.FAM[famName];
    const N = famName === 'u' ? S.NU : famName === 'v' ? S.NV
            : famName === 'w' ? S.NW : S.NP;
    const f = new Float64Array(N), got = new Float64Array(N);
    const thOf = k => (k + fam.thOff)*S.dth;
    const zOf = (a, k, b) => fam.sn[b]*S.Hat(fam.rn[a], thOf(k)).H;
    for (let a = 0; a < fam.rn.length; a++) for (let k = 0; k < nth; k++)
      for (let b = 0; b < fam.sn.length; b++)
        f[fam.idx(a,k,b)] = fOf(fam.rn[a], thOf(k), zOf(a,k,b));
    S.famLaplacian(f, got, fam, (kd, r, th, sg) => fOf(r, th, sg*S.Hat(r, th).H));
    let num = 0, den = 0;
    for (let a = Math.max(fam.rLo, skipIn); a <= fam.rHi; a++)
      for (let k = 0; k < nth; k++)
        for (let b = fam.sLo; b <= fam.sHi; b++){
          const want = lapOf(fam.rn[a], thOf(k), zOf(a,k,b));
          const d = got[fam.idx(a,k,b)] - want;
          num += d*d; den += want*want;
        }
    return Math.sqrt(num/den);
  };
  /* The two innermost radial rows are excluded for the families whose reflection is
     odd. The probe is a SCALAR, even under the axis reflection, while u and v carry
     the odd sign a horizontal vector component requires -- so at the axis the probe
     itself has the wrong parity and the reflected neighbour is wrong. That is a
     property of the probe, not the operator: v's order recovers from 0.28 to 1.45,
     matching p and u, once those rows are dropped. The axis treatment is gated on
     its own terms in 4c, where u_r's antisymmetry is exact. */
  for (const [fam, skip] of [['p', 0], ['u', 2], ['v', 2], ['w', 0]]){
    for (const amp of [0, 0.4]){
      const coarse = errorOn(fam, 16, 24, 16, amp, skip);
      const fine = errorOn(fam, 32, 48, 32, amp, skip);
      const order = Math.log(coarse/fine)/Math.log(2);
      console.log(`       ${fam} eta/h = ${amp}: ${coarse.toExponential(2)} -> `
        + `${fine.toExponential(2)}, order ${order.toFixed(2)}`);
      ok(fine < coarse,
         `${fam}, eta/h = ${amp}: refining reduces the error in grad^2 f`,
         `${coarse.toExponential(3)} then ${fine.toExponential(3)}`);
      ok(order > 1.2,
         `${fam}, eta/h = ${amp}: and it converges (order ${order.toFixed(2)})`,
         `observed order ${order.toFixed(3)}`);
    }
  }
}

/* ── 4f. the vector Laplacian and its cylindrical coupling ───────────────── */
section('4f. the vector Laplacian, where the coupling cannot hide');
/* u_r = U cos(theta) g(z), u_theta = -U sin(theta) g(z) is a field uniform in
   Cartesian terms times g(z), so grad^2 u = U xhat g''(z) exactly. The scalar
   Laplacian of each component carries a spurious -u/r^2, and it is precisely the
   two coupling terms that cancel it. Get either coupling term wrong and this field
   cannot pass; a field without an m = 1 part would not notice. */
{
  const KZ = 700, U = 1;
  const g = z => Math.sin(KZ*z), gpp = z => -KZ*KZ*Math.sin(KZ*z);
  const errorOn = (nr, nth, nz, amp) => {
    const S = new FaradayCell3D({ nr, nth, nz, ...CELL, rStretch: 0, zStretch: 0 });
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
      const x = S.rc[i]/S.R, th = (k + 0.5)*S.dth;
      S.eta[S.ie(i,k)] = amp*S.h*(0.6*x*Math.cos(th) + 0.5*x*x*Math.cos(2*th));
    }
    S.refreshMetric();
    const H = (r, th) => S.Hat(r, th).H;
    for (let i = 0; i <= nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
      S.u[S.iu(i,k,j)] = U*Math.cos((k+0.5)*S.dth)*g(S.sc[j]*H(S.rf[i], (k+0.5)*S.dth));
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 0; j < nz; j++)
      S.v[S.iv(i,k,j)] = -U*Math.sin(k*S.dth)*g(S.sc[j]*H(S.rc[i], k*S.dth));
    const LU = new Float64Array(S.NU), LV = new Float64Array(S.NV),
          LW = new Float64Array(S.NW);
    S.viscous(LU, LV, LW,
      (kd, r, th, sg) => U*Math.cos(th)*g(sg*H(r, th)),
      (kd, r, th, sg) => -U*Math.sin(th)*g(sg*H(r, th)),
      () => 0);
    let nu = 0, du = 0, nv = 0, dv = 0;
    for (let i = 2; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 1; j < nz-1; j++){
      const th = (k+0.5)*S.dth, z = S.sc[j]*H(S.rf[i], th);
      const w = U*Math.cos(th)*gpp(z), e = LU[S.iu(i,k,j)] - w;
      nu += e*e; du += w*w;
    }
    for (let i = 2; i < nr; i++) for (let k = 0; k < nth; k++) for (let j = 1; j < nz-1; j++){
      const th = k*S.dth, z = S.sc[j]*H(S.rc[i], th);
      const w = -U*Math.sin(th)*gpp(z), e = LV[S.iv(i,k,j)] - w;
      nv += e*e; dv += w*w;
    }
    return { u: Math.sqrt(nu/du), v: Math.sqrt(nv/dv) };
  };
  for (const amp of [0, 0.2, 0.4]){
    const a = errorOn(16, 24, 16, amp), b = errorOn(32, 48, 32, amp);
    const oU = Math.log(a.u/b.u)/Math.log(2), oV = Math.log(a.v/b.v)/Math.log(2);
    console.log(`       eta/h = ${amp}: (grad^2 u)_r order ${oU.toFixed(2)}, `
      + `(grad^2 u)_theta order ${oV.toFixed(2)}`);
    ok(oU > 1.8, `eta/h = ${amp}: the radial component is second order`,
       `${a.u.toExponential(3)} -> ${b.u.toExponential(3)}, order ${oU.toFixed(3)}`);
    ok(oV > 1.8, `eta/h = ${amp}: and so is the azimuthal one`,
       `${a.v.toExponential(3)} -> ${b.v.toExponential(3)}, order ${oV.toFixed(3)}`);
  }
}

/* ── 4g. Omega and the physical vertical velocity ────────────────────────── */
section('4g. the state is the physical velocity; Omega is derived');
{
  const S = deform(new FaradayCell3D({ nr: 10, nth: 16, nz: 8, ...CELL }), 0.3);
  const r = rnd(9001);
  for (let c = 0; c < S.NU; c++) S.u[c] = 1e-3*r();
  for (let c = 0; c < S.NV; c++) S.v[c] = 1e-3*r();
  for (let c = 0; c < S.NW; c++) S.w[c] = 1e-3*r();
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++) S.w[S.iw(i,k,0)] = 0;
  const keep = Float64Array.from(S.w);
  S.omegaFromW();
  let floorZero = true, slope = 0;
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
    if (S.om[S.iw(i,k,0)] !== 0) floorZero = false;
  for (let c = 0; c < S.NW; c++) slope = Math.max(slope, Math.abs(S.om[c] - keep[c]));
  S.wFromOmega();
  let worst = 0;
  for (let c = 0; c < S.NW; c++) worst = Math.max(worst, Math.abs(S.w[c] - keep[c]));
  ok(worst < 1e-18, 'w to Omega and back is the identity',
     `worst difference ${worst.toExponential(3)}`);
  ok(floorZero,
     'Omega is exactly zero on the floor, because sigma kills the slope term there '
     + 'and no slip kills w',
     `${floorZero}`);
  ok(slope > 0,
     'and the slope term is non-zero on a deformed surface, so the round trip is '
     + 'not two copies of the same array',
     `largest |Omega - w| = ${slope.toExponential(3)}`);
}

/* ── 6. advection ────────────────────────────────────────────────────────── */

/* Kahan summation, because these sums cancel to their last bit by construction and
   a naive accumulation of ten thousand terms hides that behind its own drift. With
   compensation the residual of an exact cancellation reads 1e-16 of the terms' own
   size; without it, 1e-12, and a gate set at 1e-12 would also pass a scheme whose
   curvature terms do not cancel at all. */
function kahan(){
  let s = 0, c = 0, a = 0;
  return { add(x){ const y = x - c, t = s + y; c = (t - s) - y; s = t; a += Math.abs(x); },
           get sum(){ return s; }, get abs(){ return a; } };
}

/* Every pressure cell's absolute flux balance, which is what divergence() forms:
       d_r(r H u) + d_theta(H v) + d_sigma(r Omega)
   integrated over the cell. */
function absDiv(S){
  const D = new Float64Array(S.NP);
  S.divergence(S.u, S.v, S.om, D);
  return D;
}
/* and the rate at which each control volume's own volume grows, taken as the net of
   the mesh fluxes through its faces. The mesh moves only vertically: the sheet at
   sigma rises at sigma dH/dt, so its flux through a face of horizontal area A is
   A sigma dH/dt, and the averaging onto a momentum cell is the same averaging the
   transport fluxes get. Written here from that geometry rather than read off the
   solver -- it is the discrete geometric conservation law, and if the two disagree
   a uniform field over a rising surface accelerates. */
const meshFlux = (S,i,k,b) => S.rc[i]*S.drc[i]*S.dth*S.sf[b]*S.Ht[S.ie(i,k)];
function meshNet(S, which, i, k, b){
  const Q = (a, kk, bb) => meshFlux(S, a, kk, bb);
  if (which === 'u')
    return 0.5*(Q(i-1,k,b+1) + Q(i,k,b+1) - Q(i-1,k,b) - Q(i,k,b));
  if (which === 'v')
    return 0.5*(Q(i,k-1,b+1) + Q(i,k,b+1) - Q(i,k-1,b) - Q(i,k,b));
  return b < S.nz ? 0.5*(Q(i,k,b+1) - Q(i,k,b-1))
                  : Q(i,k,S.nz) - 0.5*(Q(i,k,S.nz-1) + Q(i,k,S.nz));
}
const volU = (S,i,k,b) => S.rf[i]*S.drf[i]*S.dth*S.Hr[i*S.nth + S.kw(k)]*S.dsc[b];
const volV = (S,i,k,b) => S.rc[i]*S.drc[i]*S.dth*S.Hth[i*S.nth + S.kw(k)]*S.dsc[b];
const volW = (S,i,k,b) => S.rc[i]*S.drc[i]*S.dth*S.H[S.ie(i,k)]*S.dsf[b];

/* The net absolute flux a momentum cell must carry if the averaging is right: half
   the sum of the divergences of the pressure cells it straddles. The surface half
   cell straddles one. */
function expectNet(S, D, which, i, k, b){
  if (which === 'u') return 0.5*(D[S.ip(i-1,k,b)] + D[S.ip(i,k,b)]);
  if (which === 'v') return 0.5*(D[S.ip(i,k-1,b)] + D[S.ip(i,k,b)]);
  return b < S.nz ? 0.5*(D[S.ip(i,k,b-1)] + D[S.ip(i,k,b)]) : 0.5*D[S.ip(i,k,S.nz-1)];
}
/* the momentum nodes each family solves, as [i, k, b] triples */
function nodesOf(S, which){
  const out = [];
  if (which === 'u'){ for (let i = 1; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
                        for (let b = 0; b < S.nz; b++) out.push([i,k,b]); }
  else if (which === 'v'){ for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
                             for (let b = 0; b < S.nz; b++) out.push([i,k,b]); }
  else { for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
           for (let b = 1; b <= S.nz; b++) out.push([i,k,b]); }
  return out;
}
const famOf = (S, which) =>
  which === 'u' ? { idx: (i,k,b) => S.iu(i,k,b), vol: volU, f: S.u }
: which === 'v' ? { idx: (i,k,b) => S.iv(i,k,b), vol: volV, f: S.v }
                : { idx: (i,k,b) => S.iw(i,k,b), vol: volW, f: S.w };

/* A state with every boundary condition the advection relies on actually imposed:
   no penetration at the rim, no slip on the floor, and a MATERIAL SURFACE -- the
   mesh velocity dH/dt is Omega at sigma = 1, which is the kinematic condition, so
   the grid-relative flux there is a difference of equals and exactly zero. */
function randomState(S, seed, scale){
  const r = rnd(seed);
  for (let c = 0; c < S.NU; c++) S.u[c] = scale*r();
  for (let c = 0; c < S.NV; c++) S.v[c] = scale*r();
  for (let c = 0; c < S.NW; c++) S.w[c] = scale*r();
  for (let k = 0; k < S.nth; k++)
    for (let b = 0; b < S.nz; b++) S.u[S.iu(S.nr,k,b)] = 0;
  for (let i = 0; i < S.nr; i++)
    for (let k = 0; k < S.nth; k++) S.w[S.iw(i,k,0)] = 0;
  S.omegaFromW();
  for (let i = 0; i < S.nr; i++)
    for (let k = 0; k < S.nth; k++) S.Ht[S.ie(i,k)] = S.om[S.iw(i,k,S.nz)];
  return S;
}

section('6a. a momentum cell carries half the divergence of each cell it straddles');
/* The averaging of the pressure cells' fluxes onto the momentum cells is what makes
   centred advection conservative, and it is an algebraic identity -- true of ANY
   field, divergence free or not -- so it is asserted to round-off and not to a
   tolerance. Read out by holding the advected component at one, which turns every
   face value into one and leaves the transport term equal to minus the net ABSOLUTE
   flux over the volume: the grid-relative flux plus the mesh flux is the absolute
   one, and the moving-mesh term restores exactly the piece that was subtracted. */
for (const [nr, nth, nz, amp] of [[8, 12, 6, 0.4], [10, 16, 8, 0.7]]){
  for (const which of ['u', 'v', 'w']){
    const S = deform(new FaradayCell3D({ nr, nth, nz, ...CELL }), amp);
    randomState(S, 4242 + nr, 1e-2);
    /* the probed component is held at one everywhere, including outside the solved
       range, so every face value is exactly one */
    if (which === 'u') S.u.fill(1);
    if (which === 'v') S.v.fill(1);
    if (which === 'w') S.w.fill(1);
    /* u at the rim stays zero: the v and w cells' outer faces take the wall value
       there, and no penetration is what makes that face's flux zero */
    if (which !== 'u')
      for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++) S.u[S.iu(nr,k,b)] = 0;
    S.omegaFromW();
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++) S.Ht[S.ie(i,k)] = S.om[S.iw(i,k,nz)];
    const D = absDiv(S);
    const TU = new Float64Array(S.NU), TV = new Float64Array(S.NV),
          TW = new Float64Array(S.NW);
    S.advectTransport(TU, TV, TW);
    const T = which === 'u' ? TU : which === 'v' ? TV : TW;
    const fam = famOf(S, which);
    let worst = 0, scale = 0, htSeen = 0;
    for (const [i,k,b] of nodesOf(S, which)){
      const got = -T[fam.idx(i,k,b)]*fam.vol(S,i,k,b);
      const want = expectNet(S, D, which, i, k, b);
      worst = Math.max(worst, Math.abs(got - want));
      scale = Math.max(scale, Math.abs(want));
    }
    for (let c = 0; c < S.NE; c++) htSeen = Math.max(htSeen, Math.abs(S.Ht[c]));
    ok(worst < 1e-13*scale,
       `${nr}x${nth}x${nz}, eta/h = ${amp}, ${which}: the net flux is exactly half `
       + `the sum of the neighbouring divergences`,
       `worst ${worst.toExponential(3)} against a largest net of `
       + `${scale.toExponential(3)}, relative ${(worst/scale).toExponential(2)}`);
    ok(htSeen > 0 && scale > 0,
       `${nr}x${nth}x${nz}, ${which}: and the surface was moving and the fluxes `
       + `non-zero, so that identity was not read off two rows of zeros`,
       `largest |dH/dt| = ${htSeen.toExponential(3)}, largest net `
       + `${scale.toExponential(3)}`);
  }
}

section('6b. no advective flux crosses the axis, the rim, the floor or the surface');
/* Each is zero for its own reason and none of them is clamped: the axis face has
   exactly no area, the rim carries no penetration, the floor no slip, and the
   surface is material. A clamp would make all four read zero even when the state
   said otherwise, which is the failure these checks could not then see. */
{
  const S = randomState(deform(new FaradayCell3D({ nr: 10, nth: 16, nz: 8, ...CELL }),
                               0.5), 31337, 1e-2);
  let axis = 0, rim = 0, floor = 0, surf = 0, inner = 0, vert = 0;
  for (let k = 0; k < S.nth; k++)
    for (let b = 0; b < S.nz; b++){
      axis = Math.max(axis, Math.abs(S.fluxR(0, k, b)));
      rim = Math.max(rim, Math.abs(S.fluxR(S.nr, k, b)));
      for (let i = 1; i < S.nr; i++) inner = Math.max(inner, Math.abs(S.fluxR(i, k, b)));
    }
  for (let i = 0; i < S.nr; i++)
    for (let k = 0; k < S.nth; k++){
      floor = Math.max(floor, Math.abs(S.fluxS(i, k, 0)));
      surf = Math.max(surf, Math.abs(S.fluxS(i, k, S.nz)));
      for (let b = 1; b < S.nz; b++) vert = Math.max(vert, Math.abs(S.fluxS(i, k, b)));
    }
  ok(axis === 0 && rim === 0, 'the radial flux is exactly zero at the axis and the rim',
     `axis ${axis.toExponential(3)}, rim ${rim.toExponential(3)}`);
  ok(floor === 0 && surf === 0,
     'and the grid-relative vertical flux is exactly zero on the floor and at the '
     + 'surface, so the surface is material',
     `floor ${floor.toExponential(3)}, surface ${surf.toExponential(3)}`);
  ok(inner > 0 && vert > 0,
     'while the interior fluxes are non-zero, so those four zeros are the boundary '
     + 'conditions and not an empty field',
     `largest interior radial ${inner.toExponential(3)}, vertical `
     + `${vert.toExponential(3)}`);
}

section('6c. the cylindrical curvature pair cancels exactly in the energy');
/* +u_theta^2/r in the radial equation and -u_r u_theta/r in the azimuthal one cancel
   POINTWISE in the continuum, so they must contribute exactly nothing to the
   discrete energy. Interpolating each component to the other's node leaves
   (mean v)^2 against v^2 and does not cancel: measured that way the imbalance was
   1.7e-2 of the terms' own size, an energy source of the scheme's own truncation
   order. Forming the product once at the cell centre and distributing it as the
   exact adjoint of those averages cancels to the last bit. */
for (const [nr, nth, nz, amp] of [[8, 12, 6, 0], [10, 16, 8, 0.6]]){
  const S = deform(new FaradayCell3D({ nr, nth, nz, ...CELL }), amp);
  randomState(S, 77 + nz, 1e-2);
  /* the axis row is the one term the identity leaves open; it is zeroed here and
     derived in closed form and asserted term for term in 6i */
  for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++) S.u[S.iu(0,k,b)] = 0;
  const CU = new Float64Array(S.NU), CV = new Float64Array(S.NV);
  S.advectCurvature(CU, CV);
  const acc = kahan();
  for (let i = 1; i < nr; i++) for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++)
    acc.add(volU(S,i,k,b)*S.u[S.iu(i,k,b)]*CU[S.iu(i,k,b)]);
  for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++)
    acc.add(volV(S,i,k,b)*S.v[S.iv(i,k,b)]*CV[S.iv(i,k,b)]);
  const rel = Math.abs(acc.sum)/acc.abs;
  ok(rel < 1e-14,
     `${nr}x${nth}x${nz}, eta/h = ${amp}: the two curvature terms sum to zero in the `
     + `energy`,
     `sum ${acc.sum.toExponential(3)} against ${acc.abs.toExponential(3)} of terms, `
     + `relative ${rel.toExponential(2)}`);
  ok(acc.abs > 0,
     `${nr}x${nth}x${nz}, eta/h = ${amp}: and the terms themselves are not zero`,
     `sum of magnitudes ${acc.abs.toExponential(3)}`);
}

section('6d. the transport telescopes exactly, whatever the field');
/* The energy identity in its sharp form. For ANY field, divergence free or not,
       sum_cells u (V T)  =  -sum_cells (u^2/2) (net absolute flux + dV/dt)
   because every interior face contributes Q (u_R^2 - u_L^2)/2 and those regroup onto
   the cells, while the moving-mesh term contributes the volume rate. It needs no
   projection and no tolerance, which is why it is the gate: a defect in the flux
   averaging, in the face value, or in the volume rate breaks it at O(1), whereas a
   loose projection only makes the right-hand side small. */
for (const [nr, nth, nz, amp] of [[8, 12, 6, 0.4], [10, 16, 8, 0.7]]){
  const S = deform(new FaradayCell3D({ nr, nth, nz, ...CELL }), amp);
  randomState(S, 9090 + nth, 1e-2);
  for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++) S.u[S.iu(0,k,b)] = 0;
  const D = absDiv(S);
  const TU = new Float64Array(S.NU), TV = new Float64Array(S.NV),
        TW = new Float64Array(S.NW);
  S.advectTransport(TU, TV, TW);
  const lhs = kahan(), rhs = kahan();
  for (const which of ['u', 'v', 'w']){
    const fam = famOf(S, which);
    const T = which === 'u' ? TU : which === 'v' ? TV : TW;
    for (const [i,k,b] of nodesOf(S, which)){
      const c = fam.idx(i,k,b), V = fam.vol(S,i,k,b), q = fam.f[c];
      lhs.add(q*V*T[c]);
      rhs.add(-0.5*q*q*(expectNet(S, D, which, i, k, b) + meshNet(S, which, i, k, b)));
    }
  }
  const rel = Math.abs(lhs.sum - rhs.sum)/lhs.abs;
  ok(rel < 1e-14,
     `${nr}x${nth}x${nz}, eta/h = ${amp}: the energy the transport moves is exactly `
     + `the regrouped net flux`,
     `${lhs.sum.toExponential(6)} against ${rhs.sum.toExponential(6)}, relative to `
     + `${lhs.abs.toExponential(3)} of terms: ${rel.toExponential(2)}`);
}

section('6e. the geometric conservation law: a uniform field over a rising surface');
/* The one property the flux form alone cannot have. Let the whole layer rise at a
   single rate c -- w = c everywhere, no horizontal motion, and the surface therefore
   material with dH/dt = c -- and the field is exactly divergence free while every
   sigma sheet moves. Nothing should happen to it. Without the volume-rate term the
   transport returns -c (net mesh flux)/V instead of zero: a field made of nothing
   but a uniform rise accelerates. */
{
  const S = deform(new FaradayCell3D({ nr: 10, nth: 16, nz: 8, ...CELL }), 0.45);
  const c = 3.7e-4;
  S.u.fill(0); S.v.fill(0); S.w.fill(c); S.om.fill(c);
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++) S.Ht[S.ie(i,k)] = c;
  const D = absDiv(S);
  let dmax = 0; for (const x of D) dmax = Math.max(dmax, Math.abs(x));
  const TU = new Float64Array(S.NU), TV = new Float64Array(S.NV),
        TW = new Float64Array(S.NW);
  S.advect(TU, TV, TW);
  let worst = 0, mesh = 0;
  for (const [i,k,b] of nodesOf(S, 'w')){
    worst = Math.max(worst, Math.abs(TW[S.iw(i,k,b)]));
    mesh = Math.max(mesh, Math.abs(meshNet(S, 'w', i, k, b)/volW(S,i,k,b)));
  }
  ok(dmax === 0, 'a uniform rise is exactly divergence free, so nothing else is in play',
     `largest |divergence| ${dmax.toExponential(3)}`);
  ok(worst < 1e-16*c/S.h,
     'and the advection leaves it exactly alone, so the volume rate is the net of the '
     + 'same mesh fluxes the transport subtracted',
     `largest |dw/dt| ${worst.toExponential(3)} against the term it cancels, `
     + `${(mesh*c).toExponential(3)}`);
  ok(mesh > 0,
     'while the mesh fluxes themselves are non-zero, so that zero is a cancellation '
     + 'and not an idle grid',
     `largest mesh flux over volume ${mesh.toExponential(3)}`);
}

section('6f. on a divergence-free field the only energy it moves is the mesh volume');
/* The consequence of 6a, 6c, 6d and 6e on the state the solver will actually hand the
   advection: a projected, divergence-free field whose surface is material. The
   identity then reads
       sum u (V A)  =  -sum (u^2/2) dV/dt
   with nothing else in it, which is exactly the moving-mesh statement that the
   advection redistributes kinetic energy and never makes any. It is not asserted
   against a fixed bound -- the projection leaves a divergence of its own tolerance,
   and the identity carries that divergence through -- but against the TOLERANCE:
   tighten the solve by five decades and the residual must follow. This is the check
   that caught the missing volume-rate term, where it sat at 2.5e-3 and would not
   move. */
{
  const nr = 10, nth = 16, nz = 8;
  const S = deform(new FaradayCell3D({ nr, nth, nz, ...CELL }), 0.5);
  randomState(S, 5150, 1e-3);
  S.pressureDiagonal();
  const residual = () => {
    const TU = new Float64Array(S.NU), TV = new Float64Array(S.NV),
          TW = new Float64Array(S.NW);
    S.advect(TU, TV, TW);
    const acc = kahan();
    for (const which of ['u', 'v', 'w']){
      const fam = famOf(S, which);
      const T = which === 'u' ? TU : which === 'v' ? TV : TW;
      for (const [i,k,b] of nodesOf(S, which)){
        const c = fam.idx(i,k,b), q = fam.f[c];
        acc.add(q*fam.vol(S,i,k,b)*T[c] + 0.5*q*q*meshNet(S, which, i, k, b));
      }
    }
    return { rel: Math.abs(acc.sum)/acc.abs, abs: Math.abs(acc.sum), scale: acc.abs };
  };
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
    S.wFromOmega();
    /* the surface stays material: the mesh follows it, so dH/dt is Omega at sigma = 1 */
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++) S.Ht[S.ie(i,k)] = S.om[S.iw(i,k,nz)];
    S.axisU();
    /* the axis row is the one term the identity leaves open; measured in 6g */
    for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++) S.u[S.iu(0,k,b)] = 0;
    return S.maxDivergence();
  };
  const dLoose = project(1e-9), eLoose = residual();
  const dTight = project(1e-14), eTight = residual();
  const decades = Math.log10(eLoose.rel/eTight.rel);
  console.log(`       divergence ${dLoose.toExponential(2)} -> ${dTight.toExponential(2)}`
    + `, residual ${eLoose.rel.toExponential(2)} -> ${eTight.rel.toExponential(2)}`
    + ` (${decades.toFixed(1)} decades)`);
  ok(decades > 3,
     'the residual follows the projection tolerance rather than sitting on a floor of '
     + 'its own',
     `${eLoose.rel.toExponential(3)} at 1e-9 then ${eTight.rel.toExponential(3)} at `
     + `1e-14, ${decades.toFixed(2)} decades`);
  ok(eTight.rel < 1e-12,
     'and at the tight tolerance it is at round-off of the terms it sums',
     `${eTight.abs.toExponential(3)} against ${eTight.scale.toExponential(3)} of terms`);
}

section('6g. the one energy term a moving surface leaves, and its order');
/* The energy identity holds exactly while the surface is still. While it moves, one term
 * is left, and it is measured here rather than asserted away. (The other thing the
 * identity leaves open is the axis row, which is not about the surface moving at all and
 * is pinned in closed form in 6i.)
 *
 * THE RADIAL VOLUME WEIGHTING. The volume of a u cell is rf drf dth Hr dsigma, the
 * same product gradient() and famLaplacian() use, and its time derivative is not
 * exactly the net of the mesh fluxes through its faces, which is half the sum of the
 * two pressure cells'. The two differ because Hr is the DISTANCE-weighted
 * interpolation of H, which is what makes it a consistent face value, while the
 * volume sum wants the volume-weighted one; no single interpolation is both. The v
 * and w families have no such gap -- for them the two coincide to the last bit,
 * because sc and Hth are exact midpoints -- so this is the radial family alone, and
 * it is the one term by which kinetic energy fails to be conserved exactly while the
 * surface is moving. Measured under refinement rather than argued. */
{
  const setup = (nr, nth, nz) => {
    const S = new FaradayCell3D({ nr, nth, nz, ...CELL });
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
      const x = S.rc[i]/S.R, th = (k + 0.5)*S.dth;
      S.eta[S.ie(i,k)] = 0.35*S.h*(x*x*Math.cos(2*th) + 0.5*x*Math.sin(th + 0.4));
      S.Ht[S.ie(i,k)] = 0.11*(x*x*Math.cos(2*th) + 0.5*x*Math.sin(th + 0.4));
    }
    S.refreshMetric();
    /* a fixed smooth field, so the same physical state is sampled on both grids */
    const H = (r, th) => S.Hat(r, th).H;
    const KZ = 520;
    for (let i = 0; i <= nr; i++) for (let k = 0; k < nth; k++) for (let b = 0; b < nz; b++){
      const th = (k + 0.5)*S.dth, z = S.sc[b]*H(S.rf[i], th);
      S.u[S.iu(i,k,b)] = 1e-3*Math.cos(2*th)*Math.sin(KZ*z)*(S.rf[i]/S.R);
    }
    return S;
  };
  const gap = S => {
    const num = kahan(), den = kahan();
    for (let i = 1; i < S.nr; i++) for (let k = 0; k < S.nth; k++)
      for (let b = 0; b < S.nz; b++){
        const q = S.u[S.iu(i,k,b)];
        /* d/dt of the coded volume, exactly: Hr is linear in the two cell values, so
           its rate is the same interpolation of their rates */
        const a = S.drc[i-1], c = S.drc[i];
        const HrDot = (c*S.Ht[S.ie(i-1,k)] + a*S.Ht[S.ie(i,k)])/(a + c);
        const Vdot = S.rf[i]*S.drf[i]*S.dth*HrDot*S.dsc[b];
        const mesh = meshNet(S, 'u', i, k, b);
        num.add(0.5*q*q*(Vdot - mesh));
        den.add(Math.abs(0.5*q*q*mesh));
      }
    return Math.abs(num.sum)/den.sum;
  };
  const coarse = gap(setup(16, 24, 16)), fine = gap(setup(32, 48, 32));
  const order = Math.log(coarse/fine)/Math.log(2);
  console.log(`       radial volume-rate gap ${coarse.toExponential(2)} -> `
    + `${fine.toExponential(2)}, order ${order.toFixed(2)}`);
  ok(order > 1.8,
     'the radial volume-rate gap, the one energy term a moving surface leaves, is '
     + 'second order',
     `${coarse.toExponential(3)} then ${fine.toExponential(3)}, order `
     + `${order.toFixed(3)}`);
  /* and for the other two families it is not second order, it is nothing */
  const S = setup(16, 24, 16);
  const ULP = 8*Number.EPSILON;
  let worstV = 0, worstW = 0, nV = 0, nW = 0;
  const Q = (i,k,bb) => meshFlux(S, i, k, bb);
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++){
    for (let b = 0; b < S.nz; b++){
      const HthDot = 0.5*(S.Ht[S.ie(i,k-1)] + S.Ht[S.ie(i,k)]);
      const want = S.rc[i]*S.drc[i]*S.dth*HthDot*S.dsc[b];
      const mag = Math.max(Math.abs(Q(i,k-1,b)), Math.abs(Q(i,k,b)),
                           Math.abs(Q(i,k-1,b+1)), Math.abs(Q(i,k,b+1)));
      if (mag > 0){
        worstV = Math.max(worstV,
          Math.abs(want - meshNet(S, 'v', i, k, b))/(ULP*mag));
        nV++;
      }
    }
    for (let b = 1; b <= S.nz; b++){
      const want = S.rc[i]*S.drc[i]*S.dth*S.Ht[S.ie(i,k)]*S.dsf[b];
      const mag = Math.max(Math.abs(Q(i,k,b-1)), Math.abs(Q(i,k,b)),
                           Math.abs(Q(i,k,Math.min(b+1, S.nz))));
      if (mag > 0){
        worstW = Math.max(worstW,
          Math.abs(want - meshNet(S, 'w', i, k, b))/(ULP*mag));
        nW++;
      }
    }
  }
  /* This asked for exactly zero and read 3.9e-23, then for a fixed fraction of the
     term and read 2.2e-13 of it. Both expectations were wrong and the code was not.
     The two expressions are the same number but not the same SEQUENCE of operations:
     one multiplies a width, the other differences two mesh fluxes that are nearly
     equal, and on a grid graded towards both ends the widths are a thousandth of the
     fluxes being differenced. The loss is the cancellation in that difference, so the
     bound has to be relative to the FLUXES and not to the width -- which is where it
     is, at well under one unit in their last place. */
  ok(worstV < 1 && worstW < 1,
     'while the azimuthal and vertical families have no gap beyond the round-off of '
     + 'differencing the mesh fluxes themselves, so the term above is the radial '
     + 'weighting alone',
     `azimuthal ${worstV.toExponential(2)} of eight units in the last place over `
     + `${nV} nodes; vertical ${worstW.toExponential(2)} over ${nW}`);
}

section('6h. the advection is the advective term, against calculus, under refinement');
/* Conservation says the scheme cannot make energy. It does not say it computes the
 * right thing, and a wrong-but-conservative operator would pass everything above. So
 * this compares against calculus on a field whose derivatives are known in closed
 * form, on a DEFORMED surface, because a flat one cannot tell a missing metric term
 * from a present one.
 *
 * The field is a curl, u = curl A with A_r = K, A_theta = 0, A_z = Phi, so it is
 * divergence free identically -- for any smooth K and Phi -- rather than by
 * construction of one particular solution:
 *
 *     u_r     = (1/r) dPhi/dtheta
 *     u_theta = dK/dz - dPhi/dr
 *     u_z     = -(1/r) dK/dtheta
 *
 * Phi carries an m = 2 part and an m = 1 part and K an m = 1 part, so every component
 * has azimuthal structure and the axis sees the one mode that is non-zero there, with
 * the m = 1 parts of u_r and u_theta combining at r = 0 into a uniform Cartesian
 * vector, which is what regularity requires. Each potential carries one factor of
 * (R - r), so u_r vanishes at the rim -- no penetration, which is the only wall
 * condition the advection needs -- while u_theta does not, so the comparison there is
 * against a quantity of the field's own size rather than against a number the field
 * has already driven to zero. K carries (z/h)^2, so u_z and its vertical derivative
 * vanish on the floor and Omega there is exactly zero.
 *
 * THE INNER QUARTER OF THE RADIUS IS REPORTED SEPARATELY, and not because it is
 * inconvenient. The residual advection of a uniform Cartesian field is
 * O(dtheta^2)/r -- derived in closed form and pinned in 6i -- so at the innermost
 * cells, where r is itself a spacing, it is first order and no care in this operator
 * removes it: it is the staggered cylindrical discretisation's own property. Outside
 * that the order is the operator's. Both numbers are printed. */
{
  const KZ = 470;
  const A1 = 3.1e-2, B1 = 2.3e-2, C1 = 7.0e-4;   // the three potential amplitudes
  const field = (S) => {
    const R = S.R, h = S.h;
    const P = r => r*r*(R - r),  Pp = r => 2*R*r - 3*r*r,  Ppp = r => 2*R - 6*r;
    const Qf = r => r*(R - r),   Qp = r => R - 2*r,        Qpp = () => -2;
    const W = r => R - r,        Wp = () => -1;
    const Sf = z => Math.sin(KZ*z + 0.3),  Sp = z => KZ*Math.cos(KZ*z + 0.3);
    const S2 = z => Math.cos(KZ*z) + 0.5,  S2p = z => -KZ*Math.sin(KZ*z);
    const Tf = z => (z/h)*(z/h)*Math.sin(KZ*z);
    const Tp = z => (2*z/(h*h))*Math.sin(KZ*z) + (z/h)*(z/h)*KZ*Math.cos(KZ*z);
    const Tpp = z => (2/(h*h))*Math.sin(KZ*z) + (4*z*KZ/(h*h))*Math.cos(KZ*z)
                     - (z/h)*(z/h)*KZ*KZ*Math.sin(KZ*z);
    return {
      ur: (r,t,z) => -2*A1*Qf(r)*Math.sin(2*t)*Sf(z) - C1*W(r)*Math.sin(t)*S2(z),
      ut: (r,t,z) => B1*P(r)*Math.cos(t)*Tp(z) - A1*Pp(r)*Math.cos(2*t)*Sf(z)
                     - C1*Qp(r)*Math.cos(t)*S2(z),
      uz: (r,t,z) => B1*Qf(r)*Math.sin(t)*Tf(z),
      dur_r: (r,t,z) => -2*A1*Qp(r)*Math.sin(2*t)*Sf(z) - C1*Wp(r)*Math.sin(t)*S2(z),
      dur_t: (r,t,z) => -4*A1*Qf(r)*Math.cos(2*t)*Sf(z) - C1*W(r)*Math.cos(t)*S2(z),
      dur_z: (r,t,z) => -2*A1*Qf(r)*Math.sin(2*t)*Sp(z) - C1*W(r)*Math.sin(t)*S2p(z),
      dut_r: (r,t,z) => B1*Pp(r)*Math.cos(t)*Tp(z) - A1*Ppp(r)*Math.cos(2*t)*Sf(z)
                        - C1*Qpp(r)*Math.cos(t)*S2(z),
      dut_t: (r,t,z) => -B1*P(r)*Math.sin(t)*Tp(z) + 2*A1*Pp(r)*Math.sin(2*t)*Sf(z)
                        + C1*Qp(r)*Math.sin(t)*S2(z),
      dut_z: (r,t,z) => B1*P(r)*Math.cos(t)*Tpp(z) - A1*Pp(r)*Math.cos(2*t)*Sp(z)
                        - C1*Qp(r)*Math.cos(t)*S2p(z),
      duz_r: (r,t,z) => B1*Qp(r)*Math.sin(t)*Tf(z),
      duz_t: (r,t,z) => B1*Qf(r)*Math.cos(t)*Tf(z),
      duz_z: (r,t,z) => B1*Qf(r)*Math.sin(t)*Tp(z)
    };
  };
  /* the advective term with the cylindrical curvature terms, from those derivatives */
  const target = (F, r, t, z) => {
    const ur = F.ur(r,t,z), ut = F.ut(r,t,z), uz = F.uz(r,t,z);
    return {
      r: -(ur*F.dur_r(r,t,z) + (ut/r)*F.dur_t(r,t,z) + uz*F.dur_z(r,t,z)) + ut*ut/r,
      t: -(ur*F.dut_r(r,t,z) + (ut/r)*F.dut_t(r,t,z) + uz*F.dut_z(r,t,z)) - ur*ut/r,
      z: -(ur*F.duz_r(r,t,z) + (ut/r)*F.duz_t(r,t,z) + uz*F.duz_z(r,t,z))
    };
  };
  const build = (nr, nth, nz, amp) => {
    const S = new FaradayCell3D({ nr, nth, nz, ...CELL });
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
      const x = S.rc[i]/S.R, th = (k + 0.5)*S.dth;
      S.eta[S.ie(i,k)] = amp*S.h*(x*x*Math.cos(2*th) + 0.5*x*Math.sin(th + 0.4));
    }
    S.refreshMetric();
    S.Ht.fill(0);
    const F = field(S), H = (r, th) => S.Hat(r, th).H;
    for (let i = 0; i <= nr; i++) for (let k = 0; k < nth; k++){
      const th = (k + 0.5)*S.dth, r = S.rf[i];
      for (let b = 0; b < nz; b++) S.u[S.iu(i,k,b)] = F.ur(r, th, S.sc[b]*H(r, th));
    }
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
      const th = k*S.dth, r = S.rc[i];
      for (let b = 0; b < nz; b++) S.v[S.iv(i,k,b)] = F.ut(r, th, S.sc[b]*H(r, th));
    }
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++){
      const th = (k + 0.5)*S.dth, r = S.rc[i];
      for (let b = 0; b <= nz; b++) S.w[S.iw(i,k,b)] = F.uz(r, th, S.sf[b]*H(r, th));
    }
    S.omegaFromW();
    return { S, F, H };
  };

  /* the potentials are differentiated by hand above, so the field is first checked
     against the property that makes it usable at all: a curl has no divergence */
  {
    const { S, F } = build(16, 24, 16, 0.3);
    let worst = 0, scale = 0;
    for (let n = 0; n < 400; n++){
      const r = S.R*(0.08 + 0.85*((n*37) % 101)/101);
      const t = 2*Math.PI*((n*53) % 97)/97;
      const z = S.h*(0.05 + 0.9*((n*29) % 89)/89);
      const d = F.ur(r,t,z)/r + F.dur_r(r,t,z) + F.dut_t(r,t,z)/r + F.duz_z(r,t,z);
      worst = Math.max(worst, Math.abs(d));
      scale = Math.max(scale, Math.abs(F.dur_r(r,t,z)) + Math.abs(F.dut_t(r,t,z)/r)
                              + Math.abs(F.duz_z(r,t,z)));
    }
    ok(worst < 1e-13*scale,
       'the test field is divergence free in closed form, so the hand derivatives it '
       + 'is compared against are the right ones',
       `largest |div u| ${worst.toExponential(3)} against ${scale.toExponential(3)} of `
       + `terms`);
  }

  const errorOn = (nr, nth, nz, amp, rLoFrac) => {
    const { S, F, H } = build(nr, nth, nz, amp);
    const AU = new Float64Array(S.NU), AV = new Float64Array(S.NV),
          AW = new Float64Array(S.NW);
    S.advect(AU, AV, AW);
    const rLo = rLoFrac*S.R;
    let nu = 0, du = 0, nv = 0, dv = 0, nw = 0, dw = 0;
    for (let i = 2; i < nr; i++){ if (S.rf[i] < rLo) continue;
      for (let k = 0; k < nth; k++) for (let b = 1; b < nz-1; b++){
        const th = (k + 0.5)*S.dth, r = S.rf[i], z = S.sc[b]*H(r, th);
        const want = target(F, r, th, z).r, e = AU[S.iu(i,k,b)] - want;
        nu += e*e; du += want*want; } }
    for (let i = 1; i < nr; i++){ if (S.rc[i] < rLo) continue;
      for (let k = 0; k < nth; k++) for (let b = 1; b < nz-1; b++){
        const th = k*S.dth, r = S.rc[i], z = S.sc[b]*H(r, th);
        const want = target(F, r, th, z).t, e = AV[S.iv(i,k,b)] - want;
        nv += e*e; dv += want*want; } }
    for (let i = 1; i < nr; i++){ if (S.rc[i] < rLo) continue;
      for (let k = 0; k < nth; k++) for (let b = 1; b < nz-1; b++){
        const th = (k + 0.5)*S.dth, r = S.rc[i], z = S.sf[b]*H(r, th);
        const want = target(F, r, th, z).z, e = AW[S.iw(i,k,b)] - want;
        nw += e*e; dw += want*want; } }
    return { u: Math.sqrt(nu/du), v: Math.sqrt(nv/dv), w: Math.sqrt(nw/dw) };
  };
  const name = { u: 'radial', v: 'azimuthal', w: 'vertical' };
  for (const amp of [0, 0.2, 0.4]){
    const a = errorOn(16, 24, 16, amp, 0.25), b = errorOn(32, 48, 32, amp, 0.25);
    const a0 = errorOn(16, 24, 16, amp, 0), b0 = errorOn(32, 48, 32, amp, 0);
    const o = c => Math.log(a[c]/b[c])/Math.log(2);
    const o0 = c => Math.log(a0[c]/b0[c])/Math.log(2);
    console.log(`       eta/h = ${amp}: outside r/R = 0.25, order radial ${o('u').toFixed(2)}, `
      + `azimuthal ${o('v').toFixed(2)}, vertical ${o('w').toFixed(2)}; including the `
      + `axis cells ${o0('u').toFixed(2)}, ${o0('v').toFixed(2)}, ${o0('w').toFixed(2)}`);
    for (const c of ['u', 'v', 'w']){
      ok(o(c) > 1.8,
         `eta/h = ${amp}: the ${name[c]} component is second order outside the axis cells`,
         `${a[c].toExponential(3)} -> ${b[c].toExponential(3)}, order ${o(c).toFixed(3)}`);
      ok(o0(c) > 1.0,
         `eta/h = ${amp}: and the ${name[c]} component still converges with them in`,
         `${a0[c].toExponential(3)} -> ${b0[c].toExponential(3)}, order `
         + `${o0(c).toFixed(3)}`);
    }
  }
}

section('6i. what the axis costs, in closed form');
/* A uniform Cartesian field -- u_r = U cos(theta), u_theta = -U sin(theta), nothing
 * else, over a flat surface -- advects itself to exactly nothing, because a constant
 * vector has no gradient. The discrete scheme does not return zero, and the amount it
 * returns is worth knowing exactly rather than bounding, so it is derived here and
 * asserted term for term.
 *
 * Every sigma flux vanishes (flat surface, w = 0), the advected value at both r faces
 * is U cos(theta_c) because the field does not depend on r, and the azimuthal faces
 * reduce with sin(2 theta) identities, giving, with a = dtheta,
 *
 *   A_r = -U^2 [ a cos^2(th) - cos(a/2) sin(a) cos(2 th) ] / (rf a)
 *         + U^2 sin^2(th) cos^2(a/2) / rf
 *
 * where the second term is the curvature pair's own contribution. Expanded in a this
 * is -U^2 a^2 [1/8 + cos(2 th)/6] / rf + O(a^4): SECOND ORDER IN THE AZIMUTHAL
 * SPACING, DIVIDED BY THE RADIUS. At a fixed radius it is second order like everything
 * else. At the first cell, where rf is itself a spacing, it is first order, and no
 * choice of interpolation removes it -- the two conditions that would make it vanish
 * for all theta ask the same coefficient to be 1/4 and -1/3. That is the price of the
 * axis on a staggered cylindrical grid, and it is why 6h reports the inner quarter of
 * the radius separately. */
{
  const residual = (nr, nth, nz) => {
    const S = new FaradayCell3D({ nr, nth, nz, ...CELL });
    const U = 4.3e-3, a = S.dth;
    for (let i = 0; i <= nr; i++) for (let k = 0; k < nth; k++)
      for (let b = 0; b < nz; b++) S.u[S.iu(i,k,b)] = U*Math.cos((k + 0.5)*a);
    for (let i = 0; i < nr; i++) for (let k = 0; k < nth; k++)
      for (let b = 0; b < nz; b++) S.v[S.iv(i,k,b)] = -U*Math.sin(k*a);
    S.omegaFromW();
    const AU = new Float64Array(S.NU), AV = new Float64Array(S.NV),
          AW = new Float64Array(S.NW);
    S.advect(AU, AV, AW);
    /* the closed form above */
    const want = (i, k) => {
      const th = (k + 0.5)*a, c = Math.cos(th), s = Math.sin(th);
      return -U*U*(a*c*c - Math.cos(a/2)*Math.sin(a)*Math.cos(2*th))/(S.rf[i]*a)
             + U*U*s*s*Math.cos(a/2)*Math.cos(a/2)/S.rf[i];
    };
    /* Compared against the size of the TERMS, not of the answer. Both the scheme and
       the closed form reach the answer by subtracting two quantities of size U^2/rf
       from each other to leave one of size U^2 a^2/rf, so a relative bound on the
       answer would be asking for a^2/8 more precision than double carries: measured
       against the answer the agreement reads 2.4e-13 and 1.7e-12, and measured against
       the terms, 1e-15 -- the same numbers, one normalisation honest and one not. */
    let worst = 0, scale = 0, first = 0, mid = 0, rMid = 0;
    const iMid = Math.round(nr/2);
    for (let i = 1; i < nr; i++) for (let k = 0; k < nth; k++)
      for (let b = 0; b < nz; b++){
        const got = AU[S.iu(i,k,b)], w = want(i, k);
        worst = Math.max(worst, Math.abs(got - w)*S.rf[i]/(U*U));
        scale = Math.max(scale, Math.abs(w));
        if (i === 1) first = Math.max(first, Math.abs(got));
        if (i === iMid){ mid = Math.max(mid, Math.abs(got)); rMid = S.rf[i]; }
      }
    return { worst, scale, first, mid, rMid, U };
  };
  const c = residual(16, 24, 16), f = residual(32, 48, 32);
  ok(c.worst < 1e-13 && f.worst < 1e-13,
     'the residual advection of a uniform Cartesian field is exactly the closed form '
     + 'derived by hand, on both grids',
     `${c.worst.toExponential(2)} and ${f.worst.toExponential(2)} of the terms each `
     + `subtracts`);
  /* the same physical radius on both grids, so the order is the azimuthal spacing's */
  const gMid = residual(16, 48, 16), gFine = residual(16, 96, 16);
  const orderTh = Math.log(gMid.mid/gFine.mid)/Math.log(2);
  console.log(`       at r/R = ${(gMid.rMid/CELL.R).toFixed(3)}: `
    + `${gMid.mid.toExponential(2)} -> ${gFine.mid.toExponential(2)} when dtheta halves, `
    + `order ${orderTh.toFixed(2)}; at the first cell `
    + `${c.first.toExponential(2)} -> ${f.first.toExponential(2)} under full refinement, `
    + `order ${(Math.log(c.first/f.first)/Math.log(2)).toFixed(2)}`);
  ok(orderTh > 1.9 && orderTh < 2.1,
     'it is second order in the azimuthal spacing at a fixed radius, as the expansion '
     + 'says',
     `${gMid.mid.toExponential(3)} -> ${gFine.mid.toExponential(3)}, order `
     + `${orderTh.toFixed(3)}`);
  const orderFirst = Math.log(c.first/f.first)/Math.log(2);
  ok(orderFirst > 0.9 && orderFirst < 1.4,
     'and first order at the first cell, where the radius is itself a spacing -- which '
     + 'is the whole of what the axis costs',
     `${c.first.toExponential(3)} -> ${f.first.toExponential(3)}, order `
     + `${orderFirst.toFixed(3)}`);
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
