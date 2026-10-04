'use strict';

/* Resolved rather than required outright: under node this is a CommonJS
   require, in the browser faraday-dns.js has already put itself on globalThis
   as a classic script. It REFUSES if neither is available, rather than
   proceeding with an undefined solver. */
/* Bound under a DIFFERENT name than the class it holds. Classic scripts share
   one global lexical environment, so `const { FaradayDNS }` here and
   `class FaradayDNS` in faraday-dns.js would be a redeclaration and the page
   would fail to parse. Node's module scopes hide that; the browser does not. */
const FaradayDNSCtor = (function(){
  if (typeof require === 'function') return require('./faraday-dns.js');
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DNS)
    return globalThis.FARADAY_DNS;
  throw new Error(
    'faraday-floquet: FaradayDNS is not available. Load dns/faraday-dns.js '
    + 'before this file, or require it under node.');
})().FaradayDNS;

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

function requireFinitePositiveNumber(v, name){
  if (typeof v !== 'number' || !Number.isFinite(v) || !(v > 0)) throw new TypeError(
    `${name} = ${v}: a finite positive number is required.`);
  return v;
}

function requireFiniteInteger(v, name){
  if (typeof v !== 'number' || !Number.isInteger(v) || !(v > 0)) throw new TypeError(
    `${name} = ${v}: a positive integer is required.`);
  return v;
}

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

/* Resumable: pass a previous return as `prev` and the space is EXTENDED from
   where it stopped rather than rebuilt. That is what makes raising the Krylov
   dimension affordable -- each extra vector costs one period map, not m of them
   -- and it is why floquetDisc can escalate on its own instead of telling the
   caller to raise a number they may have no way to reach. */
function arnoldi(applyFn, v0, m, prev){
  const n = v0.length;
  let V, H, start;
  if (prev && prev.V && prev.V.length){
    if (prev.breakdown) throw new RangeError(
      `arnoldi: asked to extend to ${m} from a space that closed at ${prev.done}. `
      + `A breakdown means the Krylov space is invariant under the operator: every `
      + `further vector is already in it, so there is nothing to extend and the `
      + `eigenvalues of the ${prev.done}-dimensional Hessenberg are exact. `
      + `Refusing rather than reading a basis vector that was never built.`);
    /* A shallow copy of the reference array, not of the vectors: basis vectors
       are written once and only read afterwards, so sharing them is exact, while
       the separate array leaves `prev` unchanged by the extension. A caller that
       keeps a result and reads its dimension later gets what it was given. */
    V = prev.V.slice();
    H = Array.from({ length: m + 1 }, (_, i) =>
      Array.from({ length: m }, (_, j) =>
        (prev.H[i] !== undefined && prev.H[i][j] !== undefined) ? prev.H[i][j] : 0));
    start = prev.done;
    if (start >= m) throw new RangeError(
      `arnoldi: asked to extend to ${m} from ${start}, which is not an extension.`);
    if (V.length !== start + 1) throw new RangeError(
      `arnoldi: ${V.length} basis vectors carried in for a space of dimension `
      + `${start}, which needs ${start + 1} to extend. The previous run did not `
      + `leave its next starting vector behind.`);
  } else {
    V = [new Float64Array(n)];
    let nrm = 0; for (let i = 0; i < n; i++) nrm += v0[i]*v0[i];
    nrm = Math.sqrt(nrm);
    if (!(nrm > 0)) throw new Error('arnoldi: starting vector is zero');
    for (let i = 0; i < n; i++) V[0][i] = v0[i]/nrm;
    H = Array.from({ length: m + 1 }, () => Array(m).fill(0));
    start = 0;
  }
  const w = new Float64Array(n);
  let used = m;
  for (let j = start; j < m; j++){
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
    V.push(new Float64Array(n));
    for (let q = 0; q < n; q++) V[j+1][q] = w[q]/hn;
  }
  const Hm = Array.from({ length: used }, (_, i) =>
    Array.from({ length: used }, (_, j) => H[i][j]));
  return { Hm, used, breakdown: used < m, residual: H[used] ? H[used][used-1] : 0,
           V, H, done: used };
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
  const S = new FaradayDNSCtor({ nx, ns, L, h0, rho, nu, gamma, accel, omegaD });
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

/* ---- the disc ----------------------------------------------------------
   The same Floquet machinery over the cylindrical solver: the period map of the
   linearised Navier-Stokes operator in (r, z) at one azimuthal mode number,
   on the actual cell -- no-slip floor and sidewall, contact line free or
   pinned. The box above answers for a periodic strip at the same wavenumber;
   this answers for the disc the renderer draws.

   Linearising is not a reduction here. Floquet stability of the flat state IS
   the linear problem, and it is what makes the azimuthal direction separate
   exactly: with the base state at rest and axisymmetric, each e^{i m theta}
   mode evolves independently, so a three-dimensional question becomes a
   two-dimensional one with nothing thrown away. */
const FaradayDiscCtor = (function(){
  if (typeof require === 'function') return require('./faraday-disc.js');
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DISC)
    return globalThis.FARADAY_DISC;
  throw new Error(
    'faraday-floquet: FaradayDisc is not available. Load dns/faraday-disc.js '
    + 'before this file, or require it under node.');
})().FaradayDisc;

/* The C++ engine, resolved the same way the solver is: required only when a
   caller asks for engine: 'wasm', so a page that never asks does not need the
   module present. */
function discWasmFactory(){
  if (typeof require === 'function') return require('./faraday-disc-wasm.js').createDiscEngine;
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DISC_WASM)
    return globalThis.FARADAY_DISC_WASM.createDiscEngine;
  throw new Error(
    "faraday-floquet: engine 'wasm' was asked for, but dns/faraday-disc-wasm.js "
    + 'is not loaded. Load it before this file, or require it under node. '
    + 'Refusing rather than running the JavaScript solver under that name.');
}

function discStateSize(nr, nz){
  return 2*(nr - 1)*nz + nr*nz + nr;
}

function discPack(S, v, href, uref){
  const nr = S.nr, nz = S.nz;
  let q = 0;
  for (let i = 1; i < nr; i++) for (let j = 0; j < nz; j++) S.u[S.iu(i, j)] = v[q++]*uref;
  for (let i = 1; i < nr; i++) for (let j = 0; j < nz; j++) S.v[S.iu(i, j)] = v[q++]*uref;
  for (let i = 0; i < nr; i++) for (let j = 1; j <= nz; j++) S.w[S.iw(i, j)] = v[q++]*uref;
  for (let i = 0; i < nr; i++) S.eta[i] = v[q++]*href;
  return q;
}

function discUnpack(S, out, href, uref){
  const nr = S.nr, nz = S.nz;
  let q = 0;
  for (let i = 1; i < nr; i++) for (let j = 0; j < nz; j++) out[q++] = S.u[S.iu(i, j)]/uref;
  for (let i = 1; i < nr; i++) for (let j = 0; j < nz; j++) out[q++] = S.v[S.iu(i, j)]/uref;
  for (let i = 0; i < nr; i++) for (let j = 1; j <= nz; j++) out[q++] = S.w[S.iw(i, j)]/uref;
  for (let i = 0; i < nr; i++) out[q++] = S.eta[i]/href;
  return q;
}

function applyDiscPeriodMap(S, v, steps, dt, out, href, uref){
  discPack(S, v, href, uref);
  S.u[S.iu(0, 0)] = 0;
  S.p.fill(0);
  /* The time is SET from the step index rather than accumulated by step().
     Accumulating dt makes t after a full period differ from Td: over the 2759
     steps the default working point needs, the drift is around 3e-13 relative --
     physically nothing, but it also means the map is not quite over one period,
     and it was what first broke bit-for-bit agreement with the C++
     transcription. Measured: with the drive table built from n*dt against a
     JavaScript run accumulating dt, 50 steps at 28x14 diverged to 1.075e-10
     worst relative difference and only 40 of 1204 values stayed identical; with
     both using n*dt, all 1204 agree exactly. */
  for (let n = 0; n < steps; n++){ S.t = n*dt; S.step(dt); }
  discUnpack(S, out, href, uref);
  return out;
}

function floquetDisc(o){
  const { nr, nz, R, h, rho, nu, gamma, accel, omegaD, m } = o;
  /* 16, not 6. The viscous disc has a dense cluster of decaying shear modes
     whose moduli sit within a per cent of each other, so a short Krylov space
     does not merely lose accuracy -- it returns the wrong mode. Measured at
     nr = 28, nz = 14, m = 12, a = 7.0608 m/s^2, drive 111 Hz:

       krylov    4       8      14      20
       |mu|    0.90177 0.94006 0.94320 0.94444

     At krylov 4 the answer was 4% low and, worse, LOWER than the same case
     undriven, which inverts the one qualitative fact about parametric forcing
     that has to hold. It settles by 14. */
  const krylov = o.krylov || 16;
  /* Where escalation stops. Raising the Krylov dimension is cheap here because
     the Arnoldi space is EXTENDED rather than rebuilt -- going from 16 to 20
     costs four period maps, not twenty -- so a caller that asked for 16 and got
     a space still moving does not have to be told to come back with a larger
     number. It is escalated here, up to this cap, and only then refused. */
  const krylovCap = o.krylovCap === undefined ? 40
    : requireFiniteInteger(o.krylovCap, 'krylovCap');
  if (krylovCap < krylov) throw new RangeError(
    `krylovCap = ${krylovCap} is below krylov = ${krylov}, so the escalation has `
    + `nowhere to go.`);
  const krylovStep = o.krylovStep === undefined ? 4
    : requireFiniteInteger(o.krylovStep, 'krylovStep');
  /* How much of the leading modulus may still be moving between the last two
     Krylov sizes before the answer is refused rather than reported. */
  const krylovTol = o.krylovTol === undefined ? 2e-2
    : requireFinitePositiveNumber(o.krylovTol, 'krylovTol');
  const engine = o.engine === undefined ? 'js' : o.engine;
  if (engine !== 'js' && engine !== 'wasm') throw new TypeError(
    `engine = ${JSON.stringify(o.engine)}: the period map runs in 'js' or `
    + `'wasm'. The two are bit-for-bit identical, so this chooses how long it `
    + `takes and nothing else; 'js' is the default because it is the reference.`);
  if (!(typeof omegaD === 'number' && Number.isFinite(omegaD) && omegaD > 0))
    throw new TypeError(
      `omegaD = ${omegaD}: the monodromy map is the map over one DRIVE period, `
      + `so a positive drive frequency is required. Refusing rather than letting `
      + `Td = 2*PI/omegaD be non-finite, which takes zero time steps and returns `
      + `the identity map as a Floquet result.`);
  /* The grading is forwarded, not defaulted. It was omitted here once, and the
     effect was silent: every Floquet run built its grid with the solver's
     default 2.2/2.2 stretch no matter what the caller asked for, so a grid sized
     by suggestGrid to put 2.85 cells in the floor Stokes layer was integrated
     with 1.22 there instead. Nothing announced it -- the answer came back, with
     the damping of a different grid. */
  const S = new FaradayDiscCtor({ nr, nz, R, h, rho, nu, gamma, m,
                                  g: o.g, accel, omegaD, contact: o.contact,
                                  rStretch: o.rStretch, zStretch: o.zStretch });
  const Td = 2*Math.PI/omegaD;
  /* The step is the solver's own stability limit unless the caller overrides it,
     and the override is checked against that limit rather than trusted: a step
     above it does not announce itself, it returns a multiplier. */
  const limit = S.stableStep(0.4);
  let dtIn = o.dt === undefined ? limit : o.dt;
  if (!(typeof dtIn === 'number' && Number.isFinite(dtIn) && dtIn > 0))
    throw new TypeError(`dt = ${dtIn}: the step must be a finite positive number.`);
  if (dtIn > limit/0.4*0.5) throw new RangeError(
    `dt = ${dtIn.toExponential(3)} s exceeds half this grid's explicit stability `
    + `limit of ${(limit/0.4).toExponential(3)} s at m = ${m}, nr = ${nr}, `
    + `nz = ${nz}. Refusing rather than integrating past it: the instability that `
    + `follows is exponential and indistinguishable from a Faraday multiplier.`);
  const steps = Math.max(1, Math.round(Td/dtIn));
  const dt = Td/steps;

  const n = discStateSize(nr, nz);
  const out = new Float64Array(n);
  const href = o.href || 1e-9, uref = href*(omegaD/2);

  /* The starting vector is the surface shape of the mode being asked about --
     J_m(k r) with k the Bessel root -- rather than noise, so the Krylov space
     is built around the physics instead of around round-off. */
  const v0 = new Float64Array(n);
  const base = 2*(nr - 1)*nz + nr*nz;
  if (o.eta0){
    if (o.eta0.length !== nr) throw new RangeError(
      `eta0 has ${o.eta0.length} values for a ${nr}-cell radius.`);
    for (let i = 0; i < nr; i++) v0[base + i] = o.eta0[i];
  } else {
    for (let i = 0; i < nr; i++) v0[base + i] = Math.cos(Math.PI*(i + 0.5)/nr);
  }

  /* ---- the grid has to be able to carry the mode ------------------------
     Two things are measured on the seed profile before a single step is taken,
     and a grid that fails either is refused rather than integrated.

     The first is whether the discrete surface operator acts on this profile the
     way the continuum one does. It is the term that returns the wave: capillary
     pressure is -gamma*D_m eta, and if D_m eta is not close to -k^2 eta then the
     restoring force is wrong and every multiplier that follows is wrong with it.
     Measured on the apparatus the renderer draws, m = 12, n = 3, 138.8 Hz:
     2.56e-2 at 28x14, 8.97e-3 at 48x24, 5.09e-3 at 64x32. At m = 17, n = 46,
     3648 Hz the same 28x14 grid gives 8.57e-1 -- the operator misrepresents the
     mode by 86 per cent, because 46 radial wavelengths do not fit on 28 cells.

     The second is whether the oscillatory boundary layers have any cells in
     them. The Stokes depth sqrt(2 nu/omega) is where the no-slip floor, the
     sidewall and the free surface do their damping, and a layer thinner than one
     cell is a layer the grid cannot see: the damping then comes out of the
     truncation error instead of out of the viscosity. Same point: 1.50 cells in
     the floor layer and 0.95 at the rim at 28x14, 2.96 and 1.68 at 48x24, 4.13
     and 2.27 at 64x32.

     Neither is a tolerance on the answer -- they are conditions on the operator,
     checked before the answer exists. suggestGrid in dns/faraday-disc.js walks
     the grid up until both hold, and says UNREACHABLE when they cannot. */
  for (let i = 0; i < nr; i++) S.eta[i] = v0[base + i];
  const fit = S.surfaceModeFit();
  const kGate = o.k === undefined ? fit.k : requireFinitePositiveNumber(o.k, 'k');
  const opErr = o.k === undefined ? fit.err : S.surfaceOperatorError(kGate);
  const errTol = o.errTol === undefined ? 1e-2
    : requireFinitePositiveNumber(o.errTol, 'errTol');
  const layerCells = o.layerCells === undefined ? 2
    : requireFinitePositiveNumber(o.layerCells, 'layerCells');
  const stokes = S.stokesResolution(omegaD/2);
  const layers = Math.min(stokes.cellsInFloorLayer, stokes.cellsInSurfaceLayer,
                          stokes.cellsInRimLayer);
  const perLambda = S.cellsPerRadialWavelength(kGate);
  if (!(opErr <= errTol) || !(layers >= layerCells)) throw new Error(
    `this grid cannot carry the mode being asked about. At nr = ${nr}, nz = ${nz}`
    + `, m = ${m}, kR = ${(kGate*R).toFixed(2)}: the discrete surface operator `
    + `misses -k^2 eta by ${(opErr*100).toFixed(2)}% against a tolerance of `
    + `${(errTol*100).toFixed(2)}%, the mode gets ${perLambda.toFixed(2)} cells `
    + `per radial wavelength, and the Stokes layer of depth `
    + `${stokes.delta.toExponential(3)} m gets `
    + `${stokes.cellsInFloorLayer.toFixed(2)} cells at the floor, `
    + `${stokes.cellsInSurfaceLayer.toFixed(2)} at the surface and `
    + `${stokes.cellsInRimLayer.toFixed(2)} at the rim, against ${layerCells} `
    + `required. Refusing rather than returning a multiplier: an unresolved `
    + `boundary layer does not make the answer noisy, it makes the damping come `
    + `out of the truncation error, and that error has no sign. Call `
    + `FARADAY_DISC.suggestGrid to size a grid that holds, or say why a looser `
    + `errTol is right for this mode.`);
  S.eta.fill(0);

  /* The period map, in JavaScript or in C++. The two are bit-for-bit identical
     -- dns/check-disc.mjs asserts it on one step and on a full period -- so this
     changes how long the solve takes and nothing about what it returns. */
  const wasm = engine === 'wasm'
    ? discWasmFactory()({ bytes: o.wasmBytes })
        .load(S, steps, dt, accel === undefined ? 0 : accel, omegaD, S.g)
    : null;
  const apply = wasm
    ? (vec, w) => { wasm.applyPeriodMap(vec, out, dt, href, uref); w.set(out); }
    : (vec, w) => { applyDiscPeriodMap(S, vec, steps, dt, out, href, uref); w.set(out); };

  /* Arnoldi's own convergence, for free: the leading principal submatrix of Hm
     is exactly the Hessenberg matrix a shorter run would have produced, so the
     same single integration gives the answer at several Krylov sizes. The
     leading moduli must have stopped moving.

     This is checked rather than assumed because the residual cannot be used
     here: the spectrum is a dense cluster, so H[used][used-1] stays order one
     even where the leading modulus has settled to five figures. A gate on the
     residual would reject every converged answer and accept no others. */
  const ladderOf = (Hm, used) => {
    const l = [];
    for (const kk of [used - 8, used - 4, used]){
      if (kk < 2) continue;
      const sub = Array.from({ length: kk }, (_, i) =>
        Array.from({ length: kk }, (_, j) => Hm[i][j]));
      l.push({ krylov: kk, muMax: cabs(hessenbergEigs(sub, kk)[0]) });
    }
    return l;
  };
  const driftOf = l => l.length > 1
    ? Math.abs(l[l.length-1].muMax/l[l.length-2].muMax - 1) : Infinity;

  /* Escalation, not a message telling the caller to escalate. Each round
     extends the existing space, so reaching dimension 40 from 16 costs 24 more
     period maps in total rather than 40 fresh ones. */
  let run = arnoldi(apply, v0, krylov);
  let ladder = ladderOf(run.Hm, run.used), drift = driftOf(ladder);
  const escalation = [{ krylov: run.used, drift }];
  while (!(drift <= krylovTol) && !run.breakdown && run.used < krylovCap){
    run = arnoldi(apply, v0, Math.min(krylovCap, run.used + krylovStep), run);
    ladder = ladderOf(run.Hm, run.used);
    drift = driftOf(ladder);
    escalation.push({ krylov: run.used, drift });
  }
  const { Hm, used, breakdown, residual } = run;
  const eigs = hessenbergEigs(Hm, used);
  const muMax = cabs(eigs[0]);

  /* The spread across the ladder, not just the last step. The viscous disc's
     decaying shear modes form a dense cluster, so which member is largest is
     genuinely ill conditioned: the answer is a value with a width, and the width
     is reported rather than hidden. */
  const moduli = ladder.map(l => l.muMax);
  const spread = ladder.length > 1
    ? (Math.max(...moduli) - Math.min(...moduli))/2 : 0;
  if (!(drift <= krylovTol) && !breakdown) throw new Error(
    `disc Floquet has not converged in the Krylov dimension: the leading `
    + `modulus reads `
    + ladder.map(l => `${l.muMax.toFixed(8)} at k=${l.krylov}`).join(', ')
    + `, still moving by ${(drift*100).toFixed(3)}% against a tolerance of `
    + `${(krylovTol*100).toFixed(3)}% at the cap of ${krylovCap}. The space was `
    + `extended through `
    + escalation.map(e => `${e.krylov} (${(e.drift*100).toFixed(3)}%)`).join(', ')
    + ` before giving up. Raise krylovCap. Refusing rather than reporting the `
    + `leading modulus of a subspace that has not yet found the dominant mode -- `
    + `at krylov 4 that mistake returned a multiplier BELOW the undriven one, `
    + `which no parametric drive can do.`);

  const growth = Math.log(muMax)/Td;
  const growthWidth = spread > 0 ? Math.abs(Math.log(1 + spread/muMax)/Td) : 0;
  return { eigs, muMax, growth, Td, dt, steps, m, engine,
           muSpread: spread, growthWidth,
           unstable: muMax - spread > 1 ? true : (muMax + spread < 1 ? false : null),
           krylov: used, krylovCap, breakdown, residual, stateSize: n,
           ladder, drift, escalation,
           surfaceOperatorError: opErr, k: kGate, kFit: fit.k,
           cellsPerWavelength: perLambda, cellsInLayer: layers,
           stepLimits: S.stepLimits(), stokes,
           resolution: S.resolution() };
}

const FARADAY_FLOQUET = { floquet, hessenbergEigs, arnoldi, applyPeriodMap, stateSize,
                          floquetDisc, discStateSize, applyDiscPeriodMap };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_FLOQUET;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_FLOQUET = FARADAY_FLOQUET;
