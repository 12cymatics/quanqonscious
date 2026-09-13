/* Verification of the Faraday DNS. Every check here is a value compared with an
   independently-known answer -- an analytic force, an analytic Laplacian, an
   exact identity, the dispersion relation -- and a nonzero exit if any of them
   misses. Run: node dns/check-dns.mjs

   WHAT THIS FILE EXISTS FOR

   The first version of this solver was 5% to 28% low on the dispersion relation
   and it passed every check it had. It had a check that the divergence and the
   gradient were exact adjoints (1.2e-15), that the pressure operator was
   symmetric (7.0e-16) and negative definite, that the projection left the
   velocity divergence-free (5.4e-13), and that a flat surface at rest stayed
   flat (3.5e-19). All of those were true and none of them could see the defect,
   because the defect was that the operator pair was adjoint in the WRONG INNER
   PRODUCT. On a mapped grid the adjoint identity carries the Jacobian J = H;
   take the plain Euclidean transpose of an unweighted divergence and what comes
   back is not the gradient. Measured against the analytic hydrostatic force it
   returned s * (dp/dx) -- 98% wrong in the bottom cell, exactly HALF on the
   depth average, which halves omega^2 and gives sqrt(1/2) = 0.7071 against a
   measured 0.7231 in shallow water where the depth-averaged force is the whole
   of the dynamics.

   The missing gate was the obvious one: take a pressure field whose gradient is
   known on paper, apply the operator, compare. That is check 2 below. It is
   first because its absence is what let the solver be wrong for a whole PR
   while reporting eight green checks.

   There was a second defect of the same shape. The dynamic surface condition
   was written as p = -gamma*kappa, dropping the viscous normal stress
   2 mu (n.E.n). While a free-surface wave stays irrotational the INTERIOR
   viscous force is identically zero -- nu lap u = nu grad(div u) - nu
   curl(curl u) = 0 -- so the damping does not come from the bulk term at all;
   it comes from the two surface stresses, half from each. Dropping one halved
   the damping exactly, and the damping is what sets a Faraday threshold. That
   is check 10, and like check 2 it compares against a reference derived without
   reference to the code: the exact viscous free-surface root, solved here.

   The lesson generalises and the rest of this file is built on it: an operator
   that is only ever exercised inside a time loop can be checked only by its
   effect on a whole simulation, and a self-consistency property (adjointness,
   symmetry, conservation) can be exactly true of a wrong operator. So the
   momentum step's pieces -- the mapped Laplacian, the contravariant velocity --
   are methods on the solver rather than expressions buried in `step`, and each
   is compared here against something computed without reference to the code.
*/
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { FaradayDNS, curvature } = require('./faraday-dns.js');

const G = 9.80665, rho = 998.2, h0 = 0.003, gam = 0.07274, nuW = 1.0036e-6;
const wEx = (k, g) => Math.sqrt((G*k + g*k*k*k/rho)*Math.tanh(k*h0));

/* Amplitude damping from the bottom Stokes layer, for a standing wave over a
   rigid no-slip bottom. Derived rather than quoted, because the first version
   of this line had a factor of two wrong and the wrong value happened to sit
   inside the tolerance it was decorating:

     Stokes layer under a free stream u = U cos(wt) dissipates
        D = mu Int (du/dz)^2 dz = (rho/2) U^2 sqrt(nu w/2)   per unit area
     standing wave eta = a cos(kx) cos(wt):
        U(x) = a w sin(kx)/sinh(kh),  <U^2>_x = (a w/sinh kh)^2 / 2
        E    = (1/4) rho w^2 a^2/(k tanh kh)     per unit horizontal area
     so the ENERGY decays at <D>/E = sqrt(nu w/2) k tanh(kh)/sinh^2(kh)
     and the AMPLITUDE at half that.

   Valid while the layer is thin against both the depth and the wavelength;
   at kh = 0.31 below, delta/h = 0.11 and k delta = 0.035. */
const bottomLayerDamping = (k, w, nu) =>
  (k/2)*Math.sqrt(nu*w/2)*Math.tanh(k*h0)/Math.pow(Math.sinh(k*h0), 2);
let pass = 0; const fail = [];
const ok = (c, label, detail) => {
  if (c) { pass++; return true; }
  fail.push(`${label}${detail ? '  — ' + detail : ''}`); return false;
};

/* 1. Curvature is the exact one, not the small-slope H_xx. */
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
  ok(worst < 1e-4, 'curvature matches the exact expression', worst.toExponential(2));
  console.log(`1. curvature vs exact, 256 points: ${worst.toExponential(2)} relative`);
}

/* 2. THE GATE THAT WAS MISSING. The projection's gradient, applied to the
   analytic hydrostatic pressure on a deformed surface, against the force that
   is known on paper.

     p = rho g (H(x) - z) = rho g H(x)(1 - s)
     dp/dx|_z = dp/dx|_s - (s H_x/H) dp/ds
              = rho g H_x (1-s) + rho g H_x s = rho g H_x   -- DEPTH-INDEPENDENT
     dp/dz    = -rho g

   Depth-independence is the whole point: the broken operator returned
   s * rho g H_x, which is right only at the very top and wrong by 98% at the
   bottom. Nothing about the shape of that profile shows up in an adjointness
   or symmetry check. Refined, the error here falls 3.97x per doubling. */
{
  const report = [];
  let firstWorst = 0, firstAvg = 0, rate = 0, prev = null;
  for (const [nx, ns] of [[32,16],[64,32],[128,64]]){
    const L = 0.060, A = 1e-6, S = new FaradayDNS({nx, ns, L, h0, rho, nu:0, gamma:0});
    const k = 2*Math.PI/L;
    for (let i = 0; i < nx; i++) S.H[i] = h0 + A*Math.cos(k*(i+0.5)*S.dx);
    const q = new Float64Array(nx*ns);
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ns; j++) q[i*ns+j] = rho*G*S.H[i]*(1 - (j+0.5)/ns);
    const gu = new Float64Array(nx*ns), gw = new Float64Array(nx*(ns+1));
    S.gradient(q, gu, gw);
    let worst = 0, avgWorst = 0, vert = 0;
    for (let i = 0; i < nx; i++){
      const im = (i-1+nx)%nx, ex = rho*G*0.5*(S.Hx(im) + S.Hx(i));
      if (Math.abs(ex) < 1e-4) continue;                 // node of H_x, no scale
      let sum = 0;
      for (let j = 0; j < ns; j++){
        worst = Math.max(worst, Math.abs(gu[i*ns+j] - ex)/Math.abs(ex));
        sum += gu[i*ns+j];
      }
      avgWorst = Math.max(avgWorst, Math.abs(sum/ns - ex)/Math.abs(ex));
    }
    for (let i = 0; i < nx; i++)
      for (let j = 1; j <= ns; j++)
        vert = Math.max(vert, Math.abs(gw[i*(ns+1)+j] + rho*G)/(rho*G));
    report.push(`     nx=${String(nx).padStart(3)} ns=${String(ns).padStart(2)}: `
      + `worst ${worst.toExponential(2)}, depth-avg ${avgWorst.toExponential(2)}, `
      + `vertical ${vert.toExponential(2)}`
      + (prev === null ? '' : `, fell ${(prev/worst).toFixed(2)}x`));
    if (prev !== null) rate = prev/worst;
    if (firstWorst === 0){ firstWorst = worst; firstAvg = avgWorst; }
    prev = worst;
    ok(vert < 1e-12, 'gradient reproduces dp/dz = -rho g exactly', vert.toExponential(2));
  }
  console.log('2. gradient vs the ANALYTIC hydrostatic force (the gate that was missing):');
  for (const r of report) console.log(r);
  // the broken operator was 98% wrong at j=0 and 50% on the depth average
  ok(firstWorst < 2e-2, 'horizontal force is depth-independent rho g H_x', firstWorst.toExponential(2));
  ok(firstAvg < 1e-2, 'and its depth average is the full restoring force', firstAvg.toExponential(2));
  ok(rate > 3.5, 'and the error is second order in the mesh', `${rate.toFixed(2)}x per doubling`);
}

/* 2b. The bottom flux carries no contribution from u, whatever u is: at s = 0
   the metric term is multiplied by zero. That is the property that makes
   uAtSFace's j = 0 branch unreachable, so it is asserted rather than left as an
   untested guard whose value nobody checked. */
{
  const S = new FaradayDNS({nx:8, ns:6, L:0.006, h0, rho, nu:0, gamma:0});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.2*Math.sin(2*Math.PI*i/S.nx));
  const u = new Float64Array(S.nx*S.ns).map(() => Math.random()-0.5);
  const w = new Float64Array(S.nx*(S.ns+1)).map(() => Math.random()-0.5);
  let worst = 0;
  for (let i = 0; i < S.nx; i++)
    worst = Math.max(worst, Math.abs(S.vFlux(u, w, i, 0, S.Hx(i)) - w[i*(S.ns+1)]));
  ok(worst === 0, 'the bottom flux is exactly w, independent of u', String(worst));
  console.log(`2b. bottom flux independent of u: max deviation ${worst}`);
}

/* 3. D and G are adjoint IN THE METRIC INNER PRODUCT, the one the identity
   actually holds in:  sum_c (Du)_c q_c + sum_f (Gq)_f u_f (1/W)_f = 0, with
   1/W the Jacobian weight at each staggered location. Stating it in the
   Euclidean inner product instead is precisely what the broken version
   satisfied exactly while being the wrong operator. */
{
  const S = new FaradayDNS({nx:16, ns:12, L:0.006, h0, rho, nu:nuW, gamma:gam});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.15*Math.sin(2*Math.PI*i/S.nx));
  const R = () => Math.random() - 0.5;
  const u = new Float64Array(S.nx*S.ns).map(R), w = new Float64Array(S.nx*(S.ns+1)).map(R);
  for (let i = 0; i < S.nx; i++) w[i*(S.ns+1)] = 0;       // bottom face is not a dof
  const q = new Float64Array(S.nx*S.ns).map(R);
  const d = new Float64Array(S.nx*S.ns); S.divergence(u, w, d);
  let lhs = 0; for (let c = 0; c < d.length; c++) lhs += d[c]*q[c];
  const gu = new Float64Array(u.length), gw = new Float64Array(w.length); S.gradient(q, gu, gw);
  let rhs = 0;
  for (let i = 0; i < S.nx; i++)
    for (let j = 0; j < S.ns; j++) rhs += u[i*S.ns+j]*gu[i*S.ns+j]*S.metricWeight('u', i, j);
  for (let i = 0; i < S.nx; i++)
    for (let j = 0; j <= S.ns; j++) rhs += w[i*(S.ns+1)+j]*gw[i*(S.ns+1)+j]*S.metricWeight('w', i, j);
  const rel = Math.abs(lhs + rhs)/Math.abs(lhs);
  ok(rel < 1e-13, 'D and G are adjoint in the metric inner product', rel.toExponential(2));
  console.log(`3. <Du,q> + <Gq,u>_H = ${rel.toExponential(2)} relative (15% surface deformation)`);
}

/* 4. L = D G is symmetric and negative definite, so CG is a valid method on it.
   Necessary, and -- as the first version proved -- nowhere near sufficient. */
{
  const S = new FaradayDNS({nx:12, ns:10, L:0.006, h0, rho, nu:nuW, gamma:gam});
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
  console.log(`4. L symmetry ${sym.toExponential(2)} relative, <a,La> = ${qd.toExponential(3)} < 0`);
}

/* 5. The mapped Laplacian against an ANALYTIC Laplacian on a deformed surface.
   With A = s H_x/H,

     d2/dx2|_z = d_xx - 2A d_xs + A^2 d_ss + (A A_s - A_x) d_s,
     A A_s - A_x = s(2 H_x^2/H^2 - H_xx/H)

   and the last group is second order in the wave amplitude, so a linear
   dispersion test cannot see it at all: the u equation carried half of it and
   the w equation none, and nothing noticed. Applying the operator to
   cos(kx)cos(kappa z), whose Laplacian is -(k^2+kappa^2) times itself, does
   see it -- and the ghosts are the analytic values, so the boundary treatment
   is not what is being measured here. */
{
  const lapErr = (nx, ns) => {
    const L = 0.006, S = new FaradayDNS({nx, ns, L, h0, rho, nu:0, gamma:0});
    for (let i = 0; i < nx; i++) S.H[i] = h0*(1 + 0.15*Math.sin(2*Math.PI*(i+0.5)/nx));
    const k = 2*Math.PI/L, kap = 3/h0;
    const f = (x,z) => Math.cos(k*x)*Math.cos(kap*z);
    const lf = (x,z) => -(k*k + kap*kap)*f(x,z);
    const fu = new Float64Array(nx*ns), gB = new Float64Array(nx), gT = new Float64Array(nx);
    for (let i = 0; i < nx; i++){
      const x = i*S.dx, HF = S.Hface(i);
      for (let j = 0; j < ns; j++) fu[i*ns+j] = f(x, (j+0.5)*S.ds*HF);
      gB[i] = f(x, -0.5*S.ds*HF); gT[i] = f(x, (1 + 0.5*S.ds)*HF);
    }
    const ou = new Float64Array(nx*ns); S.lapU(fu, gB, gT, ou);
    let eu = 0, mu = 0;
    for (let i = 0; i < nx; i++){
      const x = i*S.dx, HF = S.Hface(i);
      for (let j = 0; j < ns; j++){
        const ex = lf(x, (j+0.5)*S.ds*HF);
        eu = Math.max(eu, Math.abs(ou[i*ns+j] - ex)); mu = Math.max(mu, Math.abs(ex));
      }
    }
    const fw = new Float64Array(nx*(ns+1)), gTw = new Float64Array(nx);
    for (let i = 0; i < nx; i++){
      const x = (i+0.5)*S.dx, Hi = S.H[i];
      for (let j = 0; j <= ns; j++) fw[i*(ns+1)+j] = f(x, j*S.ds*Hi);
      gTw[i] = f(x, (1 + S.ds)*Hi);
    }
    const ow = new Float64Array(nx*(ns+1)); S.lapW(fw, gTw, ow);
    let ew = 0, mw = 0;
    for (let i = 0; i < nx; i++){
      const x = (i+0.5)*S.dx, Hi = S.H[i];
      for (let j = 1; j <= ns; j++){
        const ex = lf(x, j*S.ds*Hi);
        ew = Math.max(ew, Math.abs(ow[i*(ns+1)+j] - ex)); mw = Math.max(mw, Math.abs(ex));
      }
    }
    return [eu/mu, ew/mw];
  };
  const [u1, w1] = lapErr(32, 24), [u2, w2] = lapErr(64, 48);
  console.log(`5. mapped Laplacian vs analytic (15% deformation):`);
  console.log(`     nx=32 ns=24: lapU ${u1.toExponential(2)}  lapW ${w1.toExponential(2)}`);
  console.log(`     nx=64 ns=48: lapU ${u2.toExponential(2)}  lapW ${w2.toExponential(2)}`
    + `   (fell ${(u1/u2).toFixed(2)}x, ${(w1/w2).toFixed(2)}x)`);
  ok(u1 < 5e-3 && w1 < 5e-3, 'the mapped Laplacian matches an analytic one',
     `lapU ${u1.toExponential(2)}, lapW ${w1.toExponential(2)}`);
  ok(u1/u2 > 3.5 && w1/w2 > 3.5, 'and it is second order in the mesh',
     `${(u1/u2).toFixed(2)}x, ${(w1/w2).toFixed(2)}x`);
}

/* 6. Omega = 0 at s = 1, exactly. Substituting the kinematic condition
   H_t = w - u H_x into Omega = (w - s H_t - s u H_x)/H leaves zero at s = 1 --
   an identity, not a tolerance. It is what "the frame follows the surface"
   means, and dropping the H_t term (which this did) breaks it by exactly H_t/H:
   the mesh moves and the advection is told it does not. */
{
  const S = new FaradayDNS({nx:16, ns:12, L:0.006, h0, rho, nu:nuW, gamma:gam});
  const k = 2*Math.PI/S.L;
  for (let i = 0; i < S.nx; i++) S.H[i] = h0 + 3e-5*Math.cos(k*(i+0.5)*S.dx);
  for (let n = 0; n < 60; n++) S.step(1e-6);
  const Ht = S.surfaceHt();
  let worst = 0, scale = 0;
  for (let i = 0; i < S.nx; i++){
    worst = Math.max(worst, Math.abs(S.omegaAtSFace(S.u, S.w, i, S.ns, Ht)));
    scale = Math.max(scale, Math.abs(Ht[i]/S.H[i]));
  }
  ok(scale > 0, 'the surface is actually moving, so the identity is not vacuous',
     `max|H_t/H| = ${scale.toExponential(2)}`);
  ok(worst < 1e-12*scale, 'Omega vanishes at the free surface identically',
     `${worst.toExponential(2)} against a broken-term size of ${scale.toExponential(2)}`);
  console.log(`6. Omega(s=1) = ${worst.toExponential(2)} s^-1 on a moving surface`
    + ` where dropping H_t would give ${scale.toExponential(2)}`);
}

/* 7. The projection is exact and hydrostatic balance is exact. With the
   Dirichlet condition now landing ON the surface rather than at the ghost
   centre half a cell above it, the pressure profile is not merely parallel to
   rho g (H - z), it IS rho g (H - z): the 0.7342 Pa = rho g H ds/2 offset the
   first version carried is gone, and the assertion is absolute rather than on
   the gradient alone. */
{
  const S = new FaradayDNS({nx:32, ns:24, L:0.006, h0, rho, nu:nuW, gamma:gam});
  const k = 2*Math.PI/S.L;
  for (let i = 0; i < S.nx; i++) S.H[i] = h0 + 1e-7*Math.cos(k*(i+0.5)*S.dx);
  for (let n = 0; n < 20; n++) S.step(2e-6);
  const dv = S.maxDivergence();
  ok(dv < 1e-9, 'velocity is divergence-free after projection', dv.toExponential(2));
  console.log(`7. max|div u| after 20 steps: ${dv.toExponential(2)} s^-1`
    + ` (CG residual ${S.cgResidual.toExponential(1)}, ${S.cgIters} iterations)`);

  const F = new FaradayDNS({nx:16, ns:20, L:0.006, h0, rho, nu:0, gamma:gam});
  for (let n = 0; n < 100; n++) F.step(1e-5);
  let wmax = 0, dH = 0, off = 0;
  for (let c = 0; c < F.w.length; c++) wmax = Math.max(wmax, Math.abs(F.w[c]));
  for (let i = 0; i < F.nx; i++) dH = Math.max(dH, Math.abs(F.H[i] - h0));
  for (let i = 0; i < F.nx; i++)
    for (let j = 0; j < F.ns; j++)
      off = Math.max(off, Math.abs(F.p[i*F.ns+j] - rho*G*(F.H[i] - (j+0.5)*F.ds*F.H[i])));
  ok(wmax < 1e-15, 'a flat surface at rest stays at rest', `max|w| ${wmax.toExponential(2)}`);
  ok(dH === 0, 'and the depth does not drift', dH.toExponential(2));
  ok(off < 1e-10, 'the hydrostatic pressure is rho g (H - z) with no offset',
     `${off.toExponential(2)} Pa`);
  console.log(`   100 steps under gravity: max|w| ${wmax.toExponential(2)} m/s,`
    + ` depth drift ${dH.toExponential(1)} m, max|p - rho g (H-z)| ${off.toExponential(2)} Pa`);
}

/* 8. THE PHYSICS. The dispersion relation omega^2 = (gk + gamma k^3/rho)tanh(kh),
   across shallow, intermediate and deep water and with and without capillarity,
   from a linear-amplitude standing wave released from rest. This is the check
   the first version failed at 0.7231 / 0.7419 / 0.8489 / 0.9536. */
{
  const run = (nx, ns, dt, L, g, nT) => {
    const k = 2*Math.PI/L, A = 1e-9;
    const S = new FaradayDNS({nx, ns, L, h0, rho, nu:0, gamma:g});
    for (let i = 0; i < nx; i++) S.H[i] = h0 + A*Math.cos(k*(i+0.5)*S.dx);
    const we = wEx(k, g), steps = Math.ceil(nT*2*Math.PI/we/dt);
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
  console.log('8. dispersion relation, omega/omega_exact at nx=32 ns=24:');
  const was = {60:0.7231, 20:0.7419, 6:0.8489, 61:0.9536};
  for (const [tag, L, g, dt, wk] of [
    ['gravity, L=60mm (kh=0.31, shallow)', 0.060, 0,   6e-5, 60],
    ['gravity, L=20mm (kh=0.94)         ', 0.020, 0,   3e-5, 20],
    ['gravity, L= 6mm (kh=3.14, deep)   ', 0.006, 0,   2e-5, 6],
    ['gravity+capillary, L=6mm          ', 0.006, gam, 1e-5, 61]]){
    const r = run(32, 24, dt, L, g, 2.2);
    ok(Number.isFinite(r) && Math.abs(1 - r) < 1e-2, `dispersion: ${tag.trim()}`, r.toFixed(5));
    console.log(`     ${tag}: ${r.toFixed(5)}   (was ${was[wk].toFixed(4)})`);
  }
  console.log('   joint refinement, nx, ns and dt all halved together:');
  let prev = null, rate = 0;
  for (const f of [1, 2]){
    const r = run(16*f, 12*f, 2.4e-4/f, 0.060, 0, 2.2), e = Math.abs(1 - r);
    console.log(`     nx=${16*f} ns=${12*f} dt=${(2.4e-4/f).toExponential(1)}: `
      + `${r.toFixed(6)}, error ${e.toExponential(2)}`
      + (prev === null ? '' : `, fell ${(prev/e).toFixed(2)}x`));
    if (prev !== null) rate = prev/e;
    prev = e;
  }
  ok(rate > 3.5, 'and the dispersion error is second order in the mesh',
     `${rate.toFixed(2)}x per halving`);
}

/* 9. The free-surface stress conditions, which are the ONLY source of damping
   for a free-surface wave: while the flow stays irrotational nu lap u is
   identically zero in the interior, so nothing in the bulk dissipates and the
   entire decay rate comes from the two surface stresses. The first version had
   neither right -- copy-ghosts for the tangential condition, and no viscous
   normal stress at all -- and measured exactly half of Lamb's rate.

   (a) is the algebra: n.E.n reduced with incompressibility and zero tangential
   stress, checked against n.E.n computed from its definition.
   (b) is the implementation on an analytic velocity field. */
{
  let worst = 0;
  for (let t = 0; t < 2000; t++){
    const ux = (Math.random()-0.5)*4, Hx = (Math.random()-0.5)*1.2;
    const Exx = ux, Ezz = -ux;                          // incompressibility
    const Exz = 2*Hx*ux/(1 - Hx*Hx);                    // zero tangential stress
    const def = (Hx*Hx*Exx - 2*Hx*Exz + Ezz)/(1 + Hx*Hx);
    const red = -ux*(1 + Hx*Hx)/(1 - Hx*Hx);
    worst = Math.max(worst, Math.abs(def - red)/Math.abs(red));
  }
  ok(worst < 1e-14, 'the n.E.n reduction equals n.E.n from its definition', worst.toExponential(2));
  console.log(`9. n.E.n reduction vs definition, 2000 random states: ${worst.toExponential(2)}`);

  const probe = (nx, ns) => {
    const L = 0.006, S = new FaradayDNS({nx, ns, L, h0, rho, nu:nuW, gamma:gam});
    const k = 2*Math.PI/L, A = 1.5e-4, U = 0.02;
    for (let i = 0; i < nx; i++) S.H[i] = h0 + A*Math.cos(k*(i+0.5)*S.dx);
    for (let i = 0; i < nx; i++){
      const x = i*S.dx, HF = S.Hface(i);
      for (let j = 0; j < ns; j++){
        const z = (j+0.5)*S.ds*HF;
        S.u[i*ns+j] = U*Math.sin(k*x)*(z/h0)*(z/h0);
      }
    }
    const ps = new Float64Array(nx), sux = S.surfaceUx(new Float64Array(nx));
    S.surfacePressure(ps, sux);
    let wUx = 0, wP = 0, mUx = 0, mP = 0;
    for (let i = 0; i < nx; i++){
      const xc = (i+0.5)*S.dx, H = S.H[i];
      const uxEx = U*k*Math.cos(k*xc)*(H/h0)*(H/h0);
      const Hx = -A*k*Math.sin(k*xc), Hxx = -A*k*k*Math.cos(k*xc);
      const pEx = -gam*(Hxx/Math.pow(1+Hx*Hx,1.5))
                - 2*rho*nuW*uxEx*(1+Hx*Hx)/(1-Hx*Hx);
      wUx = Math.max(wUx, Math.abs(sux[i] - uxEx)); mUx = Math.max(mUx, Math.abs(uxEx));
      wP  = Math.max(wP,  Math.abs(ps[i]  - pEx));  mP  = Math.max(mP,  Math.abs(pEx));
    }
    return [wUx/mUx, wP/mP];
  };
  const [a1, p1] = probe(32, 24), [a2, p2] = probe(64, 48);
  console.log(`   du/dx|_z and the surface pressure vs analytic values:`);
  console.log(`     nx=32 ns=24: du/dx|_z ${a1.toExponential(2)}, p_surface ${p1.toExponential(2)}`);
  console.log(`     nx=64 ns=48: du/dx|_z ${a2.toExponential(2)}, p_surface ${p2.toExponential(2)}`
    + `   (fell ${(a1/a2).toFixed(2)}x, ${(p1/p2).toFixed(2)}x)`);
  ok(a1 < 1e-2 && p1 < 1e-2, 'surfaceUx and surfacePressure match analytic values',
     `${a1.toExponential(2)}, ${p1.toExponential(2)}`);
  ok(a1/a2 > 3.5 && p1/p2 > 3.5, 'and both are second order in the mesh',
     `${(a1/a2).toFixed(2)}x, ${(p1/p2).toFixed(2)}x`);
}

/* 10. THE DAMPING RATE, against the EXACT linear viscous free-surface
   dispersion relation -- not against Lamb's 2 nu k^2, which is only its
   delta*k -> 0 limit.

     (2 nu k^2 + lambda)^2 + omega0^2 = 4 nu^2 k^4 sqrt(1 + lambda/(nu k^2))

   Non-dimensionalised with x = lambda/(nu k^2) and W = omega0/(nu k^2) this is
   (2+x)^2 + W^2 = 4 sqrt(1+x), solved below by complex Newton on the principal
   branch. Its leading correction is exactly 1 - delta*k/2 with
   delta = sqrt(2 nu/omega), so at the delta*k = 0.22 of this test the true rate
   is 0.888 of Lamb's, not 1.000 -- and a gate written against 2 nu k^2 would
   have to carry a 12% tolerance to pass, which is far too loose to catch
   anything. Against the exact root the agreement is under 1% and the tolerance
   can be 2%.

   This is the only check here that exercises the free-surface STRESS
   conditions end to end, and it is the one that matters most: while the flow
   stays irrotational the interior viscous force is identically zero, so the
   damping comes entirely from those two boundary stresses -- and the damping
   is what sets the Faraday threshold, which is the whole point of the solver.
   Both surface conditions were wrong before, and this measured 0.497 of the
   asymptotic rate, 0.56 of the exact one. */
{
  const cadd = (a,b) => [a[0]+b[0], a[1]+b[1]];
  const csub = (a,b) => [a[0]-b[0], a[1]-b[1]];
  const cmul = (a,b) => [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]];
  const cdiv = (a,b) => { const d = b[0]*b[0]+b[1]*b[1];
    return [(a[0]*b[0]+a[1]*b[1])/d, (a[1]*b[0]-a[0]*b[1])/d]; };
  const csqrt = (a) => { const r = Math.hypot(a[0],a[1]);
    const re = Math.sqrt((r+a[0])/2); let im = Math.sqrt((r-a[0])/2);
    if (a[1] < 0) im = -im; return [re, im]; };
  const viscousRoot = (W) => {
    let x = [-2, W];
    for (let i = 0; i < 200; i++){
      const sq = csqrt([1+x[0], x[1]]);
      const F  = csub(cadd(cmul(cadd([2,0],x), cadd([2,0],x)), [W*W,0]), [4*sq[0], 4*sq[1]]);
      const dF = csub([2*(2+x[0]), 2*x[1]], cdiv([2,0], sq));
      const st = cdiv(F, dF);
      x = csub(x, st);
      if (Math.hypot(st[0], st[1]) < 1e-15) break;
    }
    return x;
  };
  const measure = (nx, ns, dt, L, nu, nT) => {
    const k = 2*Math.PI/L, A = 1e-9;
    const S = new FaradayDNS({nx, ns, L, h0, rho, nu, gamma:gam});
    for (let i = 0; i < nx; i++) S.H[i] = h0 + A*Math.cos(k*(i+0.5)*S.dx);
    const w0 = wEx(k, gam), steps = Math.ceil(nT*2*Math.PI/w0/dt);
    const pk = []; let p2 = null, p1 = null, t1 = 0;
    for (let n = 0; n < steps; n++){
      S.step(dt);
      let m = 0; for (let i = 0; i < nx; i++) m += S.H[i]; m /= nx;
      const e = S.H[0] - m;
      if (p1 !== null && p2 !== null && p1 > p2 && p1 >= e && p1 > 0) pk.push([t1, p1]);
      p2 = p1; p1 = e; t1 = S.t;
    }
    if (pk.length < 2) return NaN;
    const [ta, aa] = pk[0], [tb, ab] = pk[pk.length-1];
    return -Math.log(ab/aa)/(tb - ta);
  };
  console.log('10. viscous damping vs the EXACT free-surface viscous root:');
  for (const [nu, nT] of [[6.8e-6, 4], [1.7e-6, 10]]){
    const L = 0.006, k = 2*Math.PI/L, w0 = wEx(k, gam), nk2 = nu*k*k;
    const x = viscousRoot(w0/nk2), exact = -x[0]*nk2;
    const dk = k*Math.sqrt(2*nu/w0);
    const got = measure(32, 32, 2e-5, L, nu, nT);
    const rel = Math.abs(got/exact - 1);
    console.log(`     nu=${nu.toExponential(2)} (delta k = ${dk.toFixed(3)}): `
      + `${got.toFixed(4)} s^-1 vs exact ${exact.toFixed(4)}, ${(rel*100).toFixed(2)}% off`
      + `   [Lamb's 2 nu k^2 = ${(2*nk2).toFixed(4)}, exact/Lamb ${(exact/(2*nk2)).toFixed(4)}]`);
    /* The residual is POSITIVE at both points and grows as nu falls, which is
       the bottom Stokes layer: the root above is the deep-water one and this
       cell is kh = 3.14, so the solver carries a damping the reference does
       not. Its rate is `bottomLayerDamping` below -- 0.95% of the total at the
       first point and 1.79% at the second, which accounts for the 0.69% and
       1.99% measured and has the right trend, since it scales as sqrt(nu)
       against the bulk's nu. 3% leaves room for that plus the O(dx^2) of a
       32-cell wavelength; the defect it has to catch missed by 44%. */
    const gb = bottomLayerDamping(k, w0, nu);
    console.log(`       bottom Stokes layer, not in the deep-water root: `
      + `${gb.toFixed(4)} s^-1 = ${(100*gb/exact).toFixed(2)}% of it`);
    ok(Number.isFinite(got) && rel < 0.03,
      `damping matches the exact viscous root at nu = ${nu.toExponential(2)}`,
      `${(rel*100).toFixed(2)}% off`);
  }

  /* SHALLOW, where the bottom layer is the dissipation rather than a 1%
     correction. This row exists because of a regeneration test that came back
     GREEN: replacing the no-slip bottom with free slip passed all 29 checks.
     Nothing above could see it -- at kh = 3.14 the bottom carries under 1% of
     the damping, well inside the 3% tolerance, and no other check touches u at
     the bottom at all. At kh = 0.31 it carries 91%, and free slip then misses
     by a factor of eleven.

     The reference is the bulk rate (Lamb's 2 nu k^2, which is depth-independent
     -- the dissipation integral over the potential flow gives 4 nu k^2 for the
     energy whatever the depth) plus the bottom layer derived above. Both are
     leading-order in delta; here delta/h = 0.11, and the tolerance is sized for
     that, not for round-off.

     Measured, the ratio to that reference falls with refinement rather than
     sitting still: 1.083 at nx=32 ns=32 dt=2e-4, 1.073 at nx=32 ns=48 dt=1e-4,
     1.068 at nx=48 ns=64 dt=1e-4 (168 s, 479 s and 2055 s respectively). So
     part of the gap is resolution and part is the reference's own leading
     order, and it is not worth deciding which without a finer reference than
     two asymptotic formulas added together. 15% covers both with room; free
     slip, which is what this row exists to catch, misses by 95.7%. */
  {
    const L = 0.060, k = 2*Math.PI/L, nu = nuW, w0 = wEx(k, gam);
    const ref = 2*nu*k*k + bottomLayerDamping(k, w0, nu);
    const bot = bottomLayerDamping(k, w0, nu);
    /* nx=16, ns=32, dt=1e-3, five periods -- eight seconds. Checked against
       nx=32, dt=2e-4, eight periods, which costs 168 s and gives 0.53750
       against this configuration's 0.54190: 0.8% apart. dt=5e-4 gives 0.54082,
       so it is dt-converged too, and the whole refinement ladder above moves
       the answer by 2.4%. The cheap one is used because a gate nobody can
       afford to run is not a gate. */
    const got = measure(16, 32, 1e-3, L, nu, 5);
    const rel = Math.abs(got/ref - 1);
    console.log(`     SHALLOW kh=${(k*h0).toFixed(2)}, L=60mm: ${got.toFixed(5)} s^-1 vs`
      + ` bulk+bottom ${ref.toFixed(5)} (${(100*bot/ref).toFixed(0)}% of it is the bottom`
      + ` layer), ${(rel*100).toFixed(1)}% off`);
    ok(Number.isFinite(got) && rel < 0.15,
      'shallow damping matches bulk + bottom Stokes layer (gates the no-slip bottom)',
      `${(rel*100).toFixed(1)}% off`);
  }
}

/* 11. THE FARADAY INSTABILITY ITSELF, against the damped-Mathieu prediction
   that cymatic.html's renderer uses to decide which modes are excited.

   Oscillating the container's gravity, g(t) = g + a cos(omega_d t), modulates
   only the gravitational part of the restoring force, so the mode amplitude
   obeys

       eta'' + 2 gamma eta' + [omega^2 + a k tanh(kh) cos(omega_d t)] eta = 0,

   a damped Mathieu equation whose principal (subharmonic) tongue at
   omega_d = 2 omega grows at

       sigma = eps omega/4 - gamma,   eps = a k tanh(kh)/omega^2,

   so the threshold is a_c = 4 gamma omega /(k tanh kh). This is the ONLY check
   that exercises the time-dependent drive at all, and the drive is what the
   page is about.

   omega and gamma are taken from the solver's OWN free decay rather than from
   theory, so this measures the parametric mechanism and not the dispersion and
   damping that checks 8 and 10 already measure against exact references. It
   also removes a detuning: the solver's omega is 0.9964 of the inviscid one, so
   driving at twice the THEORETICAL frequency sits about half a linewidth off
   resonance and quietly raises the threshold.

   The drive is put well above threshold on purpose. From a rest start the state
   is a mixture of both Floquet branches, and near onset they do not separate
   inside any window worth running in CI -- at a = 4 a_c an envelope fit reads
   4.99 over 10 periods and converges to a rock-steady 6.23 only by 40, against
   a prediction of 6.49. That is a property of the measurement, not of the
   solver. At a = 24 a_c the branches separate by e^10 within five periods and
   the fit is clean immediately. The 8% tolerance covers the leading-order
   Mathieu formula's own O(eps^2) error, which at these amplitudes is the
   largest term in the comparison. */
{
  const L = 0.006, k = 2*Math.PI/L, th = Math.tanh(k*h0);
  const nx = 32, ns = 32, dt = 2e-5, nu = nuW;
  const S = new FaradayDNS({nx, ns, L, h0, rho, nu, gamma:gam});
  for (let i = 0; i < nx; i++) S.H[i] = h0 + 1e-9*Math.cos(k*(i+0.5)*S.dx);
  const w0 = wEx(k, gam);
  const cr = [], pk = []; let prev = null, p2 = null, p1 = null, t1 = 0;
  for (let n = 0, N = Math.ceil(12*2*Math.PI/w0/dt); n < N; n++){
    S.step(dt);
    let m = 0; for (let i = 0; i < nx; i++) m += S.H[i]; m /= nx;
    const e = S.H[0] - m;
    if (prev !== null && prev > 0 && e <= 0){ const f = prev/(prev - e); cr.push(S.t - dt + f*dt); }
    if (p1 !== null && p2 !== null && p1 > p2 && p1 >= e && p1 > 0) pk.push([t1, p1]);
    prev = e; p2 = p1; p1 = e; t1 = S.t;
  }
  const wOwn = 2*Math.PI/((cr[cr.length-1] - cr[0])/(cr.length - 1));
  const gOwn = -Math.log(pk[pk.length-1][1]/pk[0][1])/(pk[pk.length-1][0] - pk[0][0]);
  const ac = 4*gOwn*wOwn/(k*th), T = 2*Math.PI/wOwn;
  console.log(`11. Faraday instability vs the damped Mathieu tongue the renderer uses:`);
  console.log(`     solver's own omega ${wOwn.toFixed(3)} rad/s (${(wOwn/w0).toFixed(5)} of inviscid),`
    + ` gamma ${gOwn.toFixed(4)} s^-1`);
  console.log(`     threshold a_c = 4 gamma omega/(k tanh kh) = ${ac.toFixed(4)} m/s^2`);

  const aRel = 24, nP = 6;
  const D = new FaradayDNS({nx, ns, L, h0, rho, nu, gamma:gam, accel:aRel*ac, omegaD:2*wOwn});
  for (let i = 0; i < nx; i++) D.H[i] = h0 + 1e-9*Math.cos(k*(i+0.5)*D.dx);
  const per = Math.round(T/dt), amp = [];
  for (let p = 0; p < nP; p++){ for (let n = 0; n < per; n++) D.step(dt); amp.push(D.surfaceAmplitude()); }
  let sx = 0, sy = 0, sxx = 0, sxy = 0, c = 0;
  for (let p = Math.floor(nP*0.4); p < nP; p++){
    const t = (p+1)*T, y = Math.log(amp[p]); sx += t; sy += y; sxx += t*t; sxy += t*y; c++;
  }
  const got = (c*sxy - sx*sy)/(c*sxx - sx*sx);
  const eps = aRel*ac*k*th/(wOwn*wOwn), pred = eps*wOwn/4 - gOwn;
  const rel = Math.abs(got/pred - 1);
  console.log(`     driven at a = ${aRel} a_c (eps = ${eps.toFixed(4)}), omega_d = 2 omega:`
    + ` grew at ${got.toFixed(3)} s^-1 against ${pred.toFixed(3)} predicted, ${(rel*100).toFixed(2)}% off`);
  ok(amp[nP-1] > amp[0]*10, 'the parametric drive makes the surface go unstable',
     `grew ${(amp[nP-1]/amp[0]).toExponential(2)}x`);
  ok(Number.isFinite(got) && rel < 0.08,
     'and at the damped-Mathieu growth rate', `${(rel*100).toFixed(2)}% off`);
}

console.log('\n' + '─'.repeat(66));
console.log(`${pass} checks passed, ${fail.length} failed`);
for (const f of fail) console.log('  FAILED  ' + f);
process.exit(fail.length ? 1 : 0);
