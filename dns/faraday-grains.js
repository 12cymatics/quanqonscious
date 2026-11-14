'use strict';

const O_NEILL_WALL_SHEAR_FACTOR = 1.7009;

const GRAIN_MODEL = Object.freeze({
  name: 'grains resting on the floor in the Floquet-mode flow',
  equation: '(m_p + m_f/2) dv/dt = C_w 3 pi mu d (u - v) + (3/2) m_f du/dt + F_friction',
  wallDragFactor: O_NEILL_WALL_SHEAR_FACTOR,
  wallDragSource: "O'Neill (1968), sphere touching a plane wall in linear shear, Stokes flow",
  fluidVelocityAt: 'grain centre height d/2 above the floor',
  normalForce: '(m_p - m_f)(g + a cos(omega_d t))',
  friction: 'Coulomb: stick while |F| <= mu_s N, slide against mu_k N',
  streaming: 'steady streaming inside the floor Stokes layer, u = (1/w) Re[a(z/delta) (U.grad)U* + b(z/delta) (div U) U*], '
    + 'driven by the response-frequency component of the mode\'s flow taken 4 Stokes depths above the floor and carried '
    + 'to the floor with cosh(k z); edge values reproduce Schlichting (standing) and Longuet-Higgins 1953 (progressive)',
  neglected: ['Basset history force', 'lift', 'grain-grain contact', 'grain feedback on the fluid',
              'mean flow driven at the free surface and in the interior', 'higher harmonics of the mode in the streaming',
              'flight off the floor']
});

function requirePositive(v, name){
  if (!(typeof v === 'number' && Number.isFinite(v) && v > 0))
    throw new RangeError(`${name} = ${v}: a finite positive number is required.`);
  return v;
}
function requireNonNegative(v, name){
  if (!(typeof v === 'number' && Number.isFinite(v) && v >= 0))
    throw new RangeError(`${name} = ${v}: a finite non-negative number is required.`);
  return v;
}

function floorFieldTable(o){
  const { mode, m, Td } = o;
  if (!mode || !mode.frames || !mode.grid) throw new TypeError(
    'floorFieldTable: a Floquet result with its periodic mode is required (floquetDisc with vector: true).');
  const amplitude = requirePositive(o.amplitude, 'amplitude');
  const zGrain = requirePositive(o.zGrain, 'zGrain');
  const { rf, zc } = mode.grid, nz = zc.length, nr = rf.length - 1;
  const frames = mode.frames, perDrive = frames.length - 1;
  const mu = mode.multiplier, sign = Math.sign(mu), mag = Math.abs(mu);
  if (!(mag > 0)) throw new RangeError('floorFieldTable: the multiplier is zero.');
  if (zGrain >= mode.grid.zf[nz]) throw new RangeError(
    `floorFieldTable: a grain centre ${zGrain} m above the floor is not inside the ${mode.grid.zf[nz]} m layer.`);
  let j1 = 0;
  while (j1 < nz && zc[j1] < zGrain) j1++;
  const zLo = j1 === 0 ? 0 : zc[j1 - 1], zHi = j1 >= nz ? zc[nz - 1] : zc[j1];
  if (j1 >= nz) throw new RangeError('floorFieldTable: the grain centre is above the top cell centre.');
  const wHi = (zGrain - zLo)/(zHi - zLo), wLo = 1 - wHi;
  const at = (arr, i, j) => j < 0 ? 0 : arr[i*nz + j];
  const responseSamples = sign < 0 ? 2*perDrive : perDrive;
  const period = sign < 0 ? 2*Td : Td;
  let etaPeak = 0;
  for (let k = 0; k < perDrive; k++){
    const damp = Math.pow(mag, -frames[k].t/Td);
    for (const e of frames[k].eta) etaPeak = Math.max(etaPeak, Math.abs(e)*damp);
  }
  if (!(etaPeak > 0)) throw new RangeError('floorFieldTable: the mode has no surface displacement.');
  const scale = amplitude/etaPeak;
  const faces = nr + 1;
  const knots = [];
  for (let rep = 0; rep < (sign < 0 ? 2 : 1); rep++)
    for (let k = 0; k < perDrive; k++){
      const f = frames[k], s = scale*(rep ? sign : 1)*Math.pow(mag, -f.t/Td);
      const u = new Float64Array(faces), v = new Float64Array(faces);
      for (let i = 0; i < faces; i++){
        u[i] = s*(wLo*at(f.u, i, j1 - 1) + wHi*at(f.u, i, j1));
        v[i] = s*(wLo*at(f.v, i, j1 - 1) + wHi*at(f.v, i, j1));
      }
      knots.push({ t: f.t + rep*Td, u, v });
    }
  const U = new Float64Array(responseSamples*faces), V = new Float64Array(responseSamples*faces);
  for (let k = 0, q = 0; k < responseSamples; k++){
    const t = k*period/responseSamples;
    while (q + 1 < knots.length && knots[q + 1].t <= t) q++;
    const a = knots[q], b = q + 1 < knots.length ? knots[q + 1]
      : { t: period, u: knots[0].u, v: knots[0].v };
    const w = (t - a.t)/(b.t - a.t);
    for (let i = 0; i < faces; i++){
      U[k*faces + i] = (1 - w)*a.u[i] + w*b.u[i];
      V[k*faces + i] = (1 - w)*a.v[i] + w*b.v[i];
    }
  }
  const dts = period/responseSamples;
  const dU = new Float64Array(U.length), dV = new Float64Array(V.length);
  for (let k = 0; k < responseSamples; k++){
    const kp = (k + 1) % responseSamples, km = (k - 1 + responseSamples) % responseSamples;
    for (let i = 0; i < faces; i++){
      dU[k*faces + i] = (U[kp*faces + i] - U[km*faces + i])/(2*dts);
      dV[k*faces + i] = (V[kp*faces + i] - V[km*faces + i])/(2*dts);
    }
  }
  return { m, rf: Float64Array.from(rf), R: rf[nr], faces, samples: responseSamples, period,
           U, V, dU, dV, amplitude, zGrain, multiplier: mu, scale };
}

function streamingProfile(zeta){
  requireNonNegative(zeta, 'zeta');
  const e1 = Math.exp(-zeta), e2 = Math.exp(-2*zeta), c = Math.cos(zeta), s = Math.sin(zeta);
  const a = e1*s - (1 - e2)/4;
  const eRe = e1*c, eIm = e1*s;
  const pRe = eRe*zeta - eIm*(zeta + 1), pIm = eRe*(zeta + 1) + eIm*zeta;
  const bRe = -pRe/2 + (eIm - (1 - eRe))/2;
  const bIm = 0.5 - pIm/2 + ((1 - eRe) + eIm)/2 - (1 - e2)/4;
  return { a, bRe, bIm };
}

function streamingCoefficients(o){
  const { rf, m, omega } = o, prof = o.profile, faces = rf.length, nr = faces - 1;
  const Ar = o.A.re, Ai = o.A.im, Br = o.B.re, Bi = o.B.im;
  const PrCC = new Float64Array(faces), PrSS = new Float64Array(faces), PtSC = new Float64Array(faces);
  const QrCC = new Float64Array(faces), QtSC = new Float64Array(faces);
  const coef = (pRe, pIm, qRe, qIm) => (prof.a*pRe + prof.bRe*qRe - prof.bIm*qIm)/omega;
  const coefP = (pRe) => prof.a*pRe/omega;
  for (let i = 1; i < faces; i++){
    const r = rf[i];
    const il = Math.max(0, i - 1), ih = Math.min(nr, i + 1);
    const dr = rf[ih] - rf[il];
    const dAr = (Ar[ih] - Ar[il])/dr, dAi = (Ai[ih] - Ai[il])/dr;
    const dBr = (Br[ih] - Br[il])/dr, dBi = (Bi[ih] - Bi[il])/dr;
    const Dr = dAr + (Ar[i] + m*Br[i])/r, Di = dAi + (Ai[i] + m*Bi[i])/r;
    const AdA_re = Ar[i]*dAr + Ai[i]*dAi;
    const BA_re = Br[i]*Ar[i] + Bi[i]*Ai[i], BB = Br[i]*Br[i] + Bi[i]*Bi[i];
    const AdB_re = Ar[i]*dBr + Ai[i]*dBi;
    const DA_re = Dr*Ar[i] + Di*Ai[i], DA_im = Di*Ar[i] - Dr*Ai[i];
    const DB_re = Dr*Br[i] + Di*Bi[i], DB_im = Di*Br[i] - Dr*Bi[i];
    PrCC[i] = coefP(AdA_re);
    PrSS[i] = coefP(-(m*BA_re + BB)/r);
    PtSC[i] = coefP(AdB_re + (m*BB + BA_re)/r);
    QrCC[i] = coef(0, 0, DA_re, DA_im);
    QtSC[i] = coef(0, 0, DB_re, DB_im);
  }
  for (const arr of [PrCC, PrSS, PtSC, QrCC, QtSC]) arr[0] = m === 1 ? arr[1] : 0;
  return { PrCC, PrSS, PtSC, QrCC, QtSC };
}

function floorStreaming(o){
  const { mode, m, Td } = o;
  const amplitude = requirePositive(o.amplitude, 'amplitude');
  const nu = requirePositive(o.nu, 'nu'), k = requirePositive(o.k, 'k');
  const zGrain = requirePositive(o.zGrain, 'zGrain');
  const { rf, zc } = mode.grid, nz = zc.length, nr = rf.length - 1, faces = nr + 1;
  const frames = mode.frames, perDrive = frames.length - 1;
  const mu = mode.multiplier, sign = Math.sign(mu), mag = Math.abs(mu);
  const period = sign < 0 ? 2*Td : Td, omega = 2*Math.PI/period;
  const delta = Math.sqrt(2*nu/omega);
  const zTarget = 4*delta;
  let jo = 0;
  for (let j = 1; j < nz; j++) if (Math.abs(zc[j] - zTarget) < Math.abs(zc[jo] - zTarget)) jo = j;
  const zOuter = zc[jo];
  if (zOuter < 2*delta) throw new RangeError(
    `floorStreaming: the cell centre nearest 4 Stokes depths is ${zOuter} m, inside the ${delta} m layer.`);
  let etaPeak = 0;
  for (let q = 0; q < perDrive; q++){
    const damp = Math.pow(mag, -frames[q].t/Td);
    for (const e of frames[q].eta) etaPeak = Math.max(etaPeak, Math.abs(e)*damp);
  }
  const scale = amplitude/etaPeak/Math.cosh(k*zOuter);
  const Ar = new Float64Array(faces), Ai = new Float64Array(faces);
  const Br = new Float64Array(faces), Bi = new Float64Array(faces);
  const reps = sign < 0 ? 2 : 1;
  for (let rep = 0; rep < reps; rep++)
    for (let q = 0; q < perDrive; q++){
      const f0 = frames[q], f1 = frames[q + 1];
      const span = f1.t - f0.t;
      for (const [f, wgt] of [[f0, span/2], [f1, span/2]]){
        const t = f.t + rep*Td, s = scale*(rep ? sign : 1)*Math.pow(mag, -f.t/Td)*wgt*2/period;
        const cr = Math.cos(omega*t)*s, ci = -Math.sin(omega*t)*s;
        for (let i = 0; i < faces; i++){
          const u = f.u[i*nz + jo], v = f.v[i*nz + jo];
          Ar[i] += u*cr; Ai[i] += u*ci; Br[i] += v*cr; Bi[i] += v*ci;
        }
      }
    }
  const prof = streamingProfile(zGrain/delta);
  const { PrCC, PrSS, PtSC, QrCC, QtSC } = streamingCoefficients({ rf, m, A: { re: Ar, im: Ai }, B: { re: Br, im: Bi }, omega, profile: prof });
  return { m, rf: Float64Array.from(rf), faces, omega, delta, zOuter, zetaGrain: zGrain/delta,
           profile: prof, A: { re: Ar, im: Ai }, B: { re: Br, im: Bi },
           PrCC, PrSS, PtSC, QrCC, QtSC };
}

function streamingAt(S, r, th){
  const { rf, faces, m } = S, nr = faces - 1;
  let i = 0;
  while (i < nr - 1 && rf[i + 1] < r) i++;
  const w = Math.min(1, Math.max(0, (r - rf[i])/(rf[i + 1] - rf[i]))), lin = a => (1 - w)*a[i] + w*a[i + 1];
  const c = Math.cos(m*th), s = Math.sin(m*th);
  const ur = (lin(S.PrCC) + lin(S.QrCC))*c*c + lin(S.PrSS)*s*s;
  const ut = (lin(S.PtSC) + lin(S.QtSC))*s*c;
  return { ur, ut };
}

function mulberry32(seed){
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0)/4294967296;
  };
}

function createGrains(o){
  const count = o.count;
  if (!(Number.isSafeInteger(count) && count > 0)) throw new RangeError(`count = ${count}: a positive whole number of grains.`);
  const R = requirePositive(o.R, 'R'), d = requirePositive(o.d, 'd');
  const rhoP = requirePositive(o.rhoP, 'rhoP'), rho = requirePositive(o.rho, 'rho');
  const mu = requirePositive(o.mu, 'mu');
  const muS = requireNonNegative(o.muS, 'muS'), muK = requireNonNegative(o.muK, 'muK');
  if (muK > muS) throw new RangeError(`muK = ${muK} exceeds muS = ${muS}: sliding friction above static friction.`);
  const g = requirePositive(o.g, 'g');
  const accel = requireNonNegative(o.accel, 'accel');
  const omegaD = requirePositive(o.omegaD, 'omegaD');
  if (!(rhoP > rho)) throw new RangeError(
    `rhoP = ${rhoP} kg/m^3 is not above the liquid's ${rho}: such a grain does not rest on the floor.`);
  if (accel >= g) throw new RangeError(
    `peak acceleration ${accel} m/s^2 reaches gravity ${g} m/s^2, so a grain leaves the floor once a cycle. `
    + `Flight off the floor is not modelled; refusing rather than holding the grain down.`);
  if (d/2 >= R) throw new RangeError('the grain is wider than the cell.');
  const vol = Math.PI*d*d*d/6;
  const mp = rhoP*vol, mf = rho*vol;
  const rand = mulberry32(o.seed === undefined ? 1 : o.seed);
  const rMax = R - d/2;
  const x = new Float64Array(count), y = new Float64Array(count);
  const vx = new Float64Array(count), vy = new Float64Array(count);
  const stuck = new Uint8Array(count).fill(1);
  for (let n = 0; n < count; n++){
    const r = rMax*Math.sqrt(rand()), th = 2*Math.PI*rand();
    x[n] = r*Math.cos(th); y[n] = r*Math.sin(th);
  }
  return { count, R, d, rMax, rhoP, rho, mu, muS, muK, g, accel, omegaD,
           mp, mf, M: mp + mf/2, C: O_NEILL_WALL_SHEAR_FACTOR*3*Math.PI*mu*d,
           x, y, vx, vy, stuck, t: 0, steps: 0, slidSteps: 0 };
}

function advanceGrains(G, T, tEnd, dt){
  requirePositive(dt, 'dt');
  if (!(tEnd >= G.t)) throw new RangeError(`tEnd = ${tEnd} is before the grains' time ${G.t}.`);
  const { faces, samples, period, rf, m } = T;
  const nr = faces - 1;
  const bins = 4096, binOf = new Int32Array(bins + 1);
  for (let b = 0, i = 0; b <= bins; b++){
    const r = rf[nr]*b/bins;
    while (i < nr - 1 && rf[i + 1] <= r) i++;
    binOf[b] = i;
  }
  const Ut = new Float64Array(faces), Vt = new Float64Array(faces);
  const dUt = new Float64Array(faces), dVt = new Float64Array(faces);
  const { x, y, vx, vy, stuck, C, M, mf, mp, muS, muK, g, accel, omegaD, rMax } = G;
  const buoyant = mp - mf;
  const SM = T.streaming || null;
  if (SM && (SM.faces !== faces || SM.m !== m || SM.rf.some((v, i) => v !== rf[i]))) throw new RangeError(
    'advanceGrains: the streaming field was built on a different grid or mode from the oscillatory table.');
  while (G.t < tEnd - 1e-15*period){
    const h = Math.min(dt, tEnd - G.t), tm = G.t + h/2;
    const ph = ((tm % period) + period) % period/period*samples;
    const k0 = Math.floor(ph) % samples, k1 = (k0 + 1) % samples, w1 = ph - Math.floor(ph), w0 = 1 - w1;
    for (let i = 0; i < faces; i++){
      Ut[i] = w0*T.U[k0*faces + i] + w1*T.U[k1*faces + i];
      Vt[i] = w0*T.V[k0*faces + i] + w1*T.V[k1*faces + i];
      dUt[i] = w0*T.dU[k0*faces + i] + w1*T.dU[k1*faces + i];
      dVt[i] = w0*T.dV[k0*faces + i] + w1*T.dV[k1*faces + i];
    }
    const N = buoyant*(g + accel*Math.cos(omegaD*tm));
    const stickLimit = muS*N, slideFriction = muK*N;
    for (let n = 0; n < G.count; n++){
      const px = x[n], py = y[n];
      const r = Math.hypot(px, py);
      let ufx = 0, ufy = 0, afx = 0, afy = 0;
      {
        const b = Math.min(bins, Math.floor(r/rf[nr]*bins));
        let i = binOf[b];
        while (i < nr - 1 && rf[i + 1] < r) i++;
        while (i > 0 && rf[i] > r) i--;
        const wR = (r - rf[i])/(rf[i + 1] - rf[i]), wL = 1 - wR;
        const ur = wL*Ut[i] + wR*Ut[i + 1], ut = wL*Vt[i] + wR*Vt[i + 1];
        const ar = wL*dUt[i] + wR*dUt[i + 1], at = wL*dVt[i] + wR*dVt[i + 1];
        const c = r > 0 ? px/r : 1, s = r > 0 ? py/r : 0, th = r > 0 ? Math.atan2(py, px) : 0;
        const cm = Math.cos(m*th), sm = Math.sin(m*th);
        let vr = ur*cm, vt = ut*sm;
        const qr = ar*cm, qt = at*sm;
        if (SM){
          vr += (wL*(SM.PrCC[i] + SM.QrCC[i]) + wR*(SM.PrCC[i + 1] + SM.QrCC[i + 1]))*cm*cm
              + (wL*SM.PrSS[i] + wR*SM.PrSS[i + 1])*sm*sm;
          vt += (wL*(SM.PtSC[i] + SM.QtSC[i]) + wR*(SM.PtSC[i + 1] + SM.QtSC[i + 1]))*sm*cm;
        }
        ufx = vr*c - vt*s; ufy = vr*s + vt*c;
        afx = qr*c - qt*s; afy = qr*s + qt*c;
      }
      const fx = C*ufx + 1.5*mf*afx, fy = C*ufy + 1.5*mf*afy;
      if (stuck[n]){
        const fs = Math.hypot(fx, fy);
        if (fs <= stickLimit) continue;
        stuck[n] = 0;
        const nx = vx[n] = (h*(fx - slideFriction*fx/fs))/(M + h*C);
        const ny = vy[n] = (h*(fy - slideFriction*fy/fs))/(M + h*C);
        x[n] += nx*h; y[n] += ny*h;
      } else {
        const sp = Math.hypot(vx[n], vy[n]);
        const dx = sp > 0 ? vx[n]/sp : 0, dy = sp > 0 ? vy[n]/sp : 0;
        const nvx = (M*vx[n] + h*(fx - slideFriction*dx))/(M + h*C);
        const nvy = (M*vy[n] + h*(fy - slideFriction*dy))/(M + h*C);
        if (sp > 0 && nvx*vx[n] + nvy*vy[n] <= 0){
          vx[n] = 0; vy[n] = 0;
          const fsNow = Math.hypot(fx, fy);
          stuck[n] = fsNow <= stickLimit ? 1 : 0;
        } else { vx[n] = nvx; vy[n] = nvy; }
        x[n] += vx[n]*h; y[n] += vy[n]*h;
        G.slidSteps++;
      }
      const rr = Math.hypot(x[n], y[n]);
      if (rr > rMax){
        const ex = x[n]/rr, ey = y[n]/rr;
        x[n] = ex*rMax; y[n] = ey*rMax;
        const out = vx[n]*ex + vy[n]*ey;
        if (out > 0){ vx[n] -= out*ex; vy[n] -= out*ey; }
      }
    }
    G.t += h; G.steps++;
  }
  return G;
}

const FARADAY_GRAINS = { GRAIN_MODEL, O_NEILL_WALL_SHEAR_FACTOR, floorFieldTable, floorStreaming, streamingProfile, streamingAt, streamingCoefficients,
                         createGrains, advanceGrains, mulberry32 };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_GRAINS;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_GRAINS = FARADAY_GRAINS;
