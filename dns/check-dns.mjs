import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
import { readFileSync } from 'node:fs';
const { FaradayDNS, curvature } = require('./faraday-dns.js');
const { floquet, hessenbergEigs } = require('./faraday-floquet.js');

const G = 9.80665, rho = 998.2, h0 = 0.003, gam = 0.07274, nuW = 1.0036e-6;
const wEx = (k, g) => Math.sqrt((G*k + g*k*k*k/rho)*Math.tanh(k*h0));

const bottomLayerDamping = (k, w, nu) =>
  (k/2)*Math.sqrt(nu*w/2)*Math.tanh(k*h0)/Math.pow(Math.sinh(k*h0), 2);
function mulberry32(a){
  return function(){
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(0x9E3779B9);
const R = () => rand() - 0.5;

let pass = 0; const fail = [];
let ELEVEN = null;
const ok = (c, label, detail) => {
  if (c) { pass++; return true; }
  fail.push(`${label}${detail ? '  — ' + detail : ''}`); return false;
};

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
      if (Math.abs(ex) < 1e-4) continue;
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

  ok(firstWorst < 2e-2, 'horizontal force is depth-independent rho g H_x', firstWorst.toExponential(2));
  ok(firstAvg < 1e-2, 'and its depth average is the full restoring force', firstAvg.toExponential(2));
  ok(rate > 3.5, 'and the error is second order in the mesh', `${rate.toFixed(2)}x per doubling`);
}

{
  const S = new FaradayDNS({nx:8, ns:6, L:0.006, h0, rho, nu:0, gamma:0});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.2*Math.sin(2*Math.PI*i/S.nx));
  const u = new Float64Array(S.nx*S.ns).map(R);
  const w = new Float64Array(S.nx*(S.ns+1)).map(R);
  let worst = 0;
  for (let i = 0; i < S.nx; i++)
    worst = Math.max(worst, Math.abs(S.vFlux(u, w, i, 0, S.Hx(i)) - w[i*(S.ns+1)]));
  ok(worst === 0, 'the bottom flux is exactly w, independent of u', String(worst));
  console.log(`2b. bottom flux independent of u: max deviation ${worst}`);
}

{
  const S = new FaradayDNS({nx:16, ns:12, L:0.006, h0, rho, nu:nuW, gamma:gam});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.15*Math.sin(2*Math.PI*i/S.nx));
  const u = new Float64Array(S.nx*S.ns).map(R), w = new Float64Array(S.nx*(S.ns+1)).map(R);
  for (let i = 0; i < S.nx; i++) w[i*(S.ns+1)] = 0;
  const q = new Float64Array(S.nx*S.ns).map(R);
  const d = new Float64Array(S.nx*S.ns); S.divergence(u, w, d);
  let lhs = 0, scale = 0;
  for (let c = 0; c < d.length; c++){ lhs += d[c]*q[c]; scale += Math.abs(d[c]*q[c]); }
  const gu = new Float64Array(u.length), gw = new Float64Array(w.length); S.gradient(q, gu, gw);
  let rhs = 0;
  for (let i = 0; i < S.nx; i++)
    for (let j = 0; j < S.ns; j++){
      const t = u[i*S.ns+j]*gu[i*S.ns+j]*S.metricWeight('u', i, j);
      rhs += t; scale += Math.abs(t);
    }
  for (let i = 0; i < S.nx; i++)
    for (let j = 0; j <= S.ns; j++){
      const t = w[i*(S.ns+1)+j]*gw[i*(S.ns+1)+j]*S.metricWeight('w', i, j);
      rhs += t; scale += Math.abs(t);
    }
  const rel = Math.abs(lhs + rhs)/scale;
  ok(rel < 1e-15, 'D and G are adjoint in the metric inner product', rel.toExponential(2));
  console.log(`3. <Du,q> + <Gq,u>_H = ${rel.toExponential(2)} relative (15% surface deformation)`);
}

{
  const S = new FaradayDNS({nx:12, ns:10, L:0.006, h0, rho, nu:nuW, gamma:gam});
  for (let i = 0; i < S.nx; i++) S.H[i] = h0*(1 + 0.12*Math.sin(2*Math.PI*i/S.nx));
  const n = S.nx*S.ns;
  const a = new Float64Array(n).map(R);
  const b = new Float64Array(n).map(R);
  const La = new Float64Array(n), Lb = new Float64Array(n);
  S.applyL(a, La); S.applyL(b, Lb);
  let ab = 0, ba = 0, qd = 0, sc = 0;
  for (let i = 0; i < n; i++){
    ab += La[i]*b[i]; ba += a[i]*Lb[i]; qd += a[i]*La[i];
    sc += Math.abs(La[i]*b[i]) + Math.abs(a[i]*Lb[i]);
  }
  const sym = Math.abs(ab - ba)/sc;
  ok(sym < 1e-15, 'L is symmetric', sym.toExponential(2));
  ok(qd < 0, 'L is negative definite');
  console.log(`4. L symmetry ${sym.toExponential(2)} relative, <a,La> = ${qd.toExponential(3)} < 0`);
}

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

{
  let worst = 0;
  for (let t = 0; t < 2000; t++){
    const ux = R()*4, Hx = R()*1.2;
    const Exx = ux, Ezz = -ux;
    const Exz = 2*Hx*ux/(1 - Hx*Hx);
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
    for (let i = 0; i < 60; i++){
      const sq = csqrt([1+x[0], x[1]]);
      const F  = csub(cadd(cmul(cadd([2,0],x), cadd([2,0],x)), [W*W,0]), [4*sq[0], 4*sq[1]]);
      const dF = csub([2*(2+x[0]), 2*x[1]], cdiv([2,0], sq));
      const st = cdiv(F, dF);
      x = csub(x, st);
      if (Math.hypot(st[0], st[1]) <= 1e-15*Math.hypot(x[0], x[1])) return x;
    }
    throw new Error(`viscousRoot: Newton did not converge for W = ${W}`);
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

    const gb = bottomLayerDamping(k, w0, nu);
    console.log(`       bottom Stokes layer, not in the deep-water root: `
      + `${gb.toFixed(4)} s^-1 = ${(100*gb/exact).toFixed(2)}% of it`);
    ok(Number.isFinite(got) && rel < 0.03,
      `damping matches the exact viscous root at nu = ${nu.toExponential(2)}`,
      `${(rel*100).toFixed(2)}% off`);
  }

  {
    const L = 0.060, k = 2*Math.PI/L, nu = nuW, w0 = wEx(k, gam);
    const ref = 2*nu*k*k + bottomLayerDamping(k, w0, nu);
    const bot = bottomLayerDamping(k, w0, nu);

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

  const per = Math.round(T/dt), dtA = T/per, amp = [];
  for (let p = 0; p < nP; p++){ for (let n = 0; n < per; n++) D.step(dtA); amp.push(D.surfaceAmplitude()); }
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
  ELEVEN = { growth: got, ac, wOwn, gOwn, aRel, L, k, th };
}

{
  console.log('12. the Faraday threshold, by Floquet:');

  const eigCheck = (M, n, want, label) => {
    const got = hessenbergEigs(M, n).map(z => Math.hypot(z.re, z.im)).sort((a,b) => b-a);
    const w = want.slice().sort((a,b) => b-a);
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(got[i] - w[i]));
    ok(worst < 1e-9, `eigensolver: ${label}`, `worst |lambda| error ${worst.toExponential(2)}`);
  };
  eigCheck([[2,1,1],[0,-3,1],[0,0,0.5]], 3, [2,3,0.5], 'upper triangular');
  eigCheck([[0,-1],[1,0]], 2, [1,1], 'rotation, a conjugate pair');
  eigCheck([[0,4],[1,0]], 2, [2,2], 'equal modulus, +-2');

  eigCheck([[1,1],[-1,3]], 2, [2,2], 'defective 2x2, double root at 2');
  eigCheck([[5,1,1,1],[0,3,1,1],[0,0,1,1],[0,0,-1,3]], 4, [5,3,2,2],
           '4x4 whose trailing 2x2 is defective');
  {
    const roots = [1,2,3,4,5];
    let poly = [1];
    for (const r of roots){ const q = [...poly, 0];
      for (let i = 0; i < poly.length; i++) q[i+1] -= r*poly[i]; poly = q; }
    const n = 5, Cm = Array.from({length:n}, () => Array(n).fill(0));
    for (let i = 1; i < n; i++) Cm[i][i-1] = 1;
    for (let i = 0; i < n; i++) Cm[i][n-1] = -poly[n-i];
    eigCheck(Cm, n, roots, 'companion of (x-1)...(x-5)');
  }

  const { ac, wOwn, aRel, L, k, th } = ELEVEN;
  const base = { nx:32, ns:32, L, h0, rho, nu:nuW, gamma:gam, omegaD:2*wOwn, dt:2e-5 };

  const fl6 = floquet({ ...base, accel: aRel*ac, m: 6 });
  const dGrowth = Math.abs(fl6.growth/ELEVEN.growth - 1);
  console.log(`     a = ${aRel} a_c: Floquet |mu| = ${fl6.muMax.toFixed(8)}, growth `
    + `${fl6.growth.toFixed(4)} s^-1 against check 11's envelope fit ${ELEVEN.growth.toFixed(4)}`
    + ` -- ${(dGrowth*100).toFixed(3)}% apart`);
  ok(dGrowth < 0.01, 'Floquet and the envelope fit agree on the growth rate',
     `${(dGrowth*100).toFixed(3)}%`);

  const fl4 = floquet({ ...base, accel: aRel*ac, m: 4 });
  const dM = Math.abs(fl4.muMax/fl6.muMax - 1);
  ok(dM < 1e-5, 'and the multiplier is independent of the Krylov dimension',
     `m=4 vs m=6 differ by ${dM.toExponential(2)}`);

  const lo = floquet({ ...base, accel: 0.95*ac, m: 6 });
  const hi = floquet({ ...base, accel: 1.05*ac, m: 6 });
  console.log(`     |mu|(0.95 a_c) = ${lo.muMax.toFixed(8)}  |mu|(1.05 a_c) = ${hi.muMax.toFixed(8)}`
    + `   (bisected threshold 2.55291, formula ${ac.toFixed(5)}, formula 0.99% low)`);
  ok(lo.muMax < 1, 'below 0.95 a_c the mode is stable', `|mu| = ${lo.muMax.toFixed(8)}`);
  ok(hi.muMax > 1, 'above 1.05 a_c it is unstable', `|mu| = ${hi.muMax.toFixed(8)}`);
  ok(lo.muMax < 1 && hi.muMax > 1,
     'so the damped-Mathieu threshold the renderer uses is right to within 5%',
     `bracketed in [${(0.95*ac).toFixed(4)}, ${(1.05*ac).toFixed(4)}]`);

  {
    const fx = JSON.parse(readFileSync(new URL('./hessenberg-defective-24.json',
                                               import.meta.url), 'utf8'));
    const want = fx.eigenvalueModuliReference.sortedDescending;
    let threw = null, got = null;
    try { got = hessenbergEigs(fx.matrix, fx.matrix.length)
            .map(z => Math.hypot(z.re, z.im)).sort((a, b) => b - a); }
    catch (e) { threw = e.message; }
    ok(threw === null, 'the eigensolver converges on the defective Hessenberg',
       threw || 'no throw');
    let worst = Infinity;
    if (got){
      worst = 0;
      for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]));
      console.log(`     defective 24x24 fixture: |mu|max ${got[0].toFixed(12)} vs LAPACK `
        + `${want[0].toFixed(12)}, worst of 24 moduli ${worst.toExponential(2)}`);
    }
    ok(worst < 5e-8, 'and matches LAPACK across all 24 eigenvalues',
       `worst ${worst.toExponential(2)}`);
  }
}

console.log('\n' + '─'.repeat(66));
console.log(`${pass} checks passed, ${fail.length} failed`);
for (const f of fail) console.log('  FAILED  ' + f);
process.exit(fail.length ? 1 : 0);
