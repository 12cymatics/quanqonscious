/* The Faraday threshold by Floquet analysis, not by fitting an envelope.

   WHY THIS EXISTS. dns/check-dns.mjs measures the parametric growth rate by
   releasing the surface from rest and fitting the decay of successive peaks.
   That works well above threshold and it is what check 11 asserts. Near onset
   it does not work at all, and PR #113 said so rather than pretending: released
   from rest the state is a mixture of BOTH Floquet branches, and near onset
   they do not separate inside any window worth running. At a = 2 a_c the fitted
   rate still read 1.92 against a predicted 2.16 after forty periods and was
   still climbing.

   The reason is arithmetic. Inside the tongue the two multipliers are
   mu_+- = -exp((-gamma +- s) T_d). At onset s = gamma, so |mu_+| = 1 while
   |mu_-| = exp(-2 gamma T_d) -- and 2 gamma T_d is 0.045 in the configuration
   below, so the branches differ in modulus by 4%. Power iteration, which is
   what an envelope fit is, separates them as 0.956^n: a hundred drive periods
   for one digit.

   So the multipliers are computed rather than fitted. The one-drive-period map
   is LINEAR on perturbations about the flat base state -- verified, not assumed:
   a flat surface under a 60 m/s^2 oscillating gravity stays flat to 1.8e-17
   after 600 steps, so there is a genuine base state to linearise about. Arnoldi
   builds a Krylov basis of that map from a dozen applications and the Ritz
   values give BOTH branches at once, with no separation required.

   The threshold is then where max|mu| = 1, which is exact for a linear
   Floquet problem rather than a fit to an exponential that is not one.
*/
'use strict';

const { FaradayDNS } = require('./faraday-dns.js');

/* ── complex arithmetic, only what the eigensolver needs ─────────────────── */
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

/* Eigenvalues of a small upper Hessenberg matrix by shifted QR, carried out in
   COMPLEX arithmetic so that a conjugate pair needs no 2x2 block handling: with
   a Wilkinson shift the matrix goes upper triangular and the eigenvalues come
   off the diagonal.

   Complex rather than real is not fastidiousness. The multipliers here are a
   conjugate pair below threshold and a real pair of EQUAL MODULUS at the
   coalescence -- exactly the two cases unshifted real QR cannot separate. */
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
      const tr = cadd(a, d), det = csub(cmul(a,d), cmul(b,c));
      const disc = csqrtc(csub(cmul(tr,tr), C(4*det.re, 4*det.im)));
      const r1 = cdiv(cadd(tr, disc), C(2,0)), r2 = cdiv(csub(tr, disc), C(2,0));
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

/* ── the one-drive-period map ────────────────────────────────────────────── */

/* A perturbation is packed as [u | w | H - h0]. The bottom w face is carried
   even though the projection pins it to zero; it simply stays zero, and
   carrying it keeps the packing a plain concatenation of the solver's own
   arrays rather than a second layout that could disagree with them. */
function stateSize(nx, ns){ return nx*ns + nx*(ns+1) + nx; }

/* Applies the map ONCE, in SCALED variables.

   Two scalings, both necessary and neither of them changing the eigenvalues.

   AMPLITUDE. Arnoldi normalises its basis vectors to unit length, and a
   unit-length state vector is a one-metre surface displacement on a three
   millimetre cell. That is not a linear perturbation, it is not even a physical
   one -- run directly, the first application drove H through zero and the
   pressure solve refused to converge, which is the refusal in faraday-dns.js
   doing exactly its job. So the unit vector is scaled down to `href` before it
   is handed to the solver and the result is scaled back up. The map is linear,
   so this is an identity on the operator, not an approximation of it.

   UNITS. u and w are m/s while H is m, so a unit vector in the raw packing
   mixes quantities that differ by a factor of omega. Carrying velocities in
   units of `uref = omega0 * href` makes every component the same size for a
   wave of amplitude href, which is a diagonal similarity transform: the Ritz
   values are unchanged and the Krylov basis is far better conditioned.

   The drive phase is reset to t = 0 on every call. The monodromy operator is
   the map over one period FROM A FIXED PHASE; two applications started at
   different phases are not the same operator. The pressure warm start is
   cleared for the same reason -- it converges to the same field either way, but
   a carried-over guess makes the map depend on call order, which a linear
   operator must not. */
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

/* ── Arnoldi ─────────────────────────────────────────────────────────────── */

/* Modified Gram-Schmidt, with one reorthogonalisation pass. The second pass is
   not optional here: the map is strongly contracting on everything except the
   two Floquet branches, so the Krylov vectors lose orthogonality fast and the
   Ritz values pick up the loss long before the residual says so. */
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
    /* A happy breakdown means the Krylov space closed on an invariant subspace
       -- the exact answer, not a failure. It is reported so a caller can tell
       that from a truncation. */
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

/* ── the public entry point ──────────────────────────────────────────────── */

/* Floquet multipliers of one Faraday mode over one DRIVE period.

   The subharmonic response has twice the drive period, so a subharmonic
   instability shows here as a multiplier near -1, and the growth rate is
   ln|mu|/T_d. Reporting over the drive period rather than the response period
   is deliberate: it is the period the operator is actually periodic in, and
   using the response period would fold the two branches onto the same value. */
function floquet(o){
  const { nx, ns, L, h0, rho, nu, gamma, accel, omegaD } = o;
  const m = o.m || 12;
  const Td = 2*Math.PI/omegaD;
  const steps = Math.max(1, Math.round(Td/(o.dt || 2e-5)));
  const dt = Td/steps;                       // exactly one period, no remainder
  const S = new FaradayDNS({ nx, ns, L, h0, rho, nu, gamma, accel, omegaD });
  const n = stateSize(nx, ns);
  const out = new Float64Array(n);
  /* href is a linear-regime surface amplitude: nine orders below the depth, the
     same scale the growth-rate tests in check-dns.mjs release from. uref is the
     velocity a wave of that amplitude carries at the response frequency. */
  const href = o.href || 1e-9, uref = href*(omegaD/2);
  const apply = (v, w) => { applyPeriodMap(S, h0, v, steps, dt, out, href, uref); w.set(out); };

  /* Start from a surface perturbation of the mode in question. The linear
     dynamics preserves each x-Fourier sector, so Arnoldi stays inside this
     mode's sector and the effective dimension is 2 ns + 2 rather than the
     full state. Starting from noise would span every sector at once and need a
     far larger m to resolve the one being asked about. */
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
