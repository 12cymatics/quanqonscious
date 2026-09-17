'use strict';

const { FaradayDNS } = require('./faraday-dns.js');

const C = (re, im = 0) => ({ re, im });
const cadd = (a, b) => C(a.re + b.re, a.im + b.im);
const csub = (a, b) => C(a.re - b.re, a.im - b.im);
const cmul = (a, b) => C(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
const cdiv = (a, b) => { const d = b.re*b.re + b.im*b.im;
  return C((a.re*b.re + a.im*b.im)/d, (a.im*b.re - a.re*b.im)/d); };
const cabs = (a) => Math.hypot(a.re, a.im);
const cconj = (a) => C(a.re, -a.im);
const csqrtc = (a) => { const r = cabs(a); if (r === 0) return C(0, 0);
  const re = Math.sqrt((r + a.re)/2); let im = Math.sqrt((r - a.re)/2);
  if (a.im < 0) im = -im; return C(re, im); };

function eig2x2(a, b, c, d){
  const tr = cadd(a, d), det = csub(cmul(a, d), cmul(b, c));
  const disc = csqrtc(csub(cmul(tr, tr), C(4*det.re, 4*det.im)));
  const s = (tr.re*disc.re + tr.im*disc.im) >= 0 ? disc : C(-disc.re, -disc.im);
  const r1 = cdiv(cadd(tr, s), C(2, 0));
  const r2 = cabs(r1) > 0 ? cdiv(det, r1) : cdiv(csub(tr, s), C(2, 0));
  return [r1, r2];
}

function hessenbergEigs(Hin, n){
  const H = [];
  for (let i = 0; i < n; i++){ H.push([]); for (let j = 0; j < n; j++) H[i].push(C(Hin[i][j], 0)); }
  const eigs = [];
  let hi = n - 1;
  while (hi >= 0){
    if (hi === 0){ eigs.push(H[0][0]); break; }
    let it = 0, deflated = false;
    for (; it < 500 && !deflated; it++){
      let small = -1;
      for (let i = hi; i > 0; i--){
        const t = cabs(H[i-1][i-1]) + cabs(H[i][i]);
        if (cabs(H[i][i-1]) <= 1e-16*(t || 1)){ small = i; break; }
      }
      if (small === hi){ eigs.push(H[hi][hi]); H[hi][hi-1] = C(0,0); hi--; deflated = true; break; }
      const a = H[hi-1][hi-1], b = H[hi-1][hi], c = H[hi][hi-1], d = H[hi][hi];
      const [r1, r2] = eig2x2(a, b, c, d);

      if (small === hi - 1 || (small < 0 && hi === 1)){
        eigs.push(r1, r2);
        if (hi - 1 > 0) H[hi-1][hi-2] = C(0,0);
        hi -= 2; deflated = true; break;
      }
      const mu = cabs(csub(d, r1)) < cabs(csub(d, r2)) ? r1 : r2;
      for (let i = 0; i <= hi; i++) H[i][i] = csub(H[i][i], mu);
      const cs = [], sn = [];
      for (let i = 0; i < hi; i++){
        const x = H[i][i], y = H[i+1][i];
        const r = Math.hypot(cabs(x), cabs(y));
        if (r === 0){ cs.push(C(1,0)); sn.push(C(0,0)); continue; }
        const cc = cdiv(x, C(r,0)), ss = cdiv(y, C(r,0));
        cs.push(cc); sn.push(ss);
        for (let j = i; j <= hi; j++){
          const u = H[i][j], v = H[i+1][j];
          H[i][j]   = cadd(cmul(cconj(cc), u), cmul(cconj(ss), v));
          H[i+1][j] = cadd(cmul(C(-ss.re, -ss.im), u), cmul(cc, v));
        }
      }
      for (let i = 0; i < hi; i++){
        const cc = cs[i], ss = sn[i];
        for (let j = 0; j <= Math.min(hi, i+2); j++){
          const u = H[j][i], v = H[j][i+1];
          H[j][i]   = cadd(cmul(u, cc), cmul(v, ss));
          H[j][i+1] = cadd(cmul(u, C(-ss.re, ss.im)), cmul(v, cconj(cc)));
        }
      }
      for (let i = 0; i <= hi; i++) H[i][i] = cadd(H[i][i], mu);
    }
    if (!deflated && it >= 500)
      throw new Error(`hessenbergEigs: QR did not converge on the block ending at ${hi}`);
  }
  return eigs.sort((p, q) => cabs(q) - cabs(p));
}

function stateSize(nx, ns){ return nx*ns + nx*(ns+1) + nx; }

function applyPeriodMap(S, h0, v, steps, dt, out, href, uref){
  const nu = S.nx*S.ns, nw = S.nx*(S.ns + 1), nx = S.nx;
  for (let i = 0; i < nu; i++) S.u[i] = v[i]*uref;
  for (let i = 0; i < nw; i++) S.w[i] = v[nu + i]*uref;
  for (let i = 0; i < nx; i++) S.H[i] = h0 + v[nu + nw + i]*href;
  S.p.fill(0);
  S.t = 0;
  for (let n = 0; n < steps; n++) S.step(dt);
  for (let i = 0; i < nu; i++) out[i] = S.u[i]/uref;
  for (let i = 0; i < nw; i++) out[nu + i] = S.w[i]/uref;
  for (let i = 0; i < nx; i++) out[nu + nw + i] = (S.H[i] - h0)/href;
  return out;
}

function arnoldi(applyFn, v0, m){
  const n = v0.length;
  const V = [new Float64Array(n)];
  let nrm = 0; for (let i = 0; i < n; i++) nrm += v0[i]*v0[i];
  nrm = Math.sqrt(nrm);
  if (!(nrm > 0)) throw new Error('arnoldi: starting vector is zero');
  for (let i = 0; i < n; i++) V[0][i] = v0[i]/nrm;
  const H = Array.from({ length: m + 1 }, () => Array(m).fill(0));
  const w = new Float64Array(n);
  let used = m;
  for (let j = 0; j < m; j++){
    applyFn(V[j], w);
    for (let pass = 0; pass < 2; pass++){
      for (let i = 0; i <= j; i++){
        let d = 0; for (let q = 0; q < n; q++) d += V[i][q]*w[q];
        for (let q = 0; q < n; q++) w[q] -= d*V[i][q];
        H[i][j] += d;
      }
    }
    let hn = 0; for (let q = 0; q < n; q++) hn += w[q]*w[q];
    hn = Math.sqrt(hn);
    H[j+1][j] = hn;

    if (hn <= 1e-14*Math.max(1, Math.abs(H[j][j]))){ used = j + 1; break; }
    if (j + 1 < m){
      V.push(new Float64Array(n));
      for (let q = 0; q < n; q++) V[j+1][q] = w[q]/hn;
    }
  }
  const Hm = Array.from({ length: used }, (_, i) =>
    Array.from({ length: used }, (_, j) => H[i][j]));
  return { Hm, used, breakdown: used < m, residual: H[used] ? H[used][used-1] : 0 };
}

function floquet(o){
  const { nx, ns, L, h0, rho, nu, gamma, accel, omegaD } = o;
  const m = o.m || 12;
  if (!(typeof omegaD === 'number' && Number.isFinite(omegaD) && omegaD > 0))
    throw new TypeError(
      `omegaD = ${omegaD}: the monodromy map is the map over one DRIVE period, `
      + `so a positive drive frequency is required. Refusing rather than letting `
      + `Td = 2*PI/omegaD be non-finite, which makes the step count non-finite, `
      + `takes zero time steps and returns the identity map as a Floquet result.`);
  const dtIn = o.dt === undefined ? 2e-5 : o.dt;
  if (!(typeof dtIn === 'number' && Number.isFinite(dtIn) && dtIn > 0))
    throw new TypeError(`dt = ${dtIn}: the step must be a finite positive number.`);
  const Td = 2*Math.PI/omegaD;
  const steps = Math.max(1, Math.round(Td/dtIn));
  const dt = Td/steps;
  const S = new FaradayDNS({ nx, ns, L, h0, rho, nu, gamma, accel, omegaD });
  const n = stateSize(nx, ns);
  const out = new Float64Array(n);

  const href = o.href || 1e-9, uref = href*(omegaD/2);
  const apply = (v, w) => { applyPeriodMap(S, h0, v, steps, dt, out, href, uref); w.set(out); };

  const v0 = new Float64Array(n);
  const k = 2*Math.PI/L, dx = L/nx, base = nx*ns + nx*(ns + 1);
  for (let i = 0; i < nx; i++) v0[base + i] = Math.cos(k*(i + 0.5)*dx);

  const { Hm, used, breakdown, residual } = arnoldi(apply, v0, m);
  const eigs = hessenbergEigs(Hm, used);
  const muMax = cabs(eigs[0]);
  return { eigs, muMax, growth: Math.log(muMax)/Td, Td, dt, steps,
           krylov: used, breakdown, residual };
}

module.exports = { floquet, hessenbergEigs, arnoldi, applyPeriodMap, stateSize };
