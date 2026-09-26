'use strict';

const G0 = 9.80665;

function curvature(H, dx, out){
  const n = H.length;
  for (let i = 0; i < n; i++){
    const im = (i - 1 + n) % n, ip = (i + 1) % n;
    const Hx = (H[ip] - H[im])/(2*dx);
    const Hxx = (H[ip] - 2*H[i] + H[im])/(dx*dx);
    out[i] = Hxx/Math.pow(1 + Hx*Hx, 1.5);
  }
  return out;
}

/* Dual-mode export. This solver is loaded two ways: by `require` from
   dns/check-dns.mjs under node, and as a classic <script> by cymatic.html,
   where `module` does not exist and a bare `module.exports` would throw on
   load. The kernel next door uses the same shape. */
const FARADAY_DNS = { G0, curvature };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_DNS;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_DNS = FARADAY_DNS;

function requireFinite(v, name){
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(
    `${name} = ${typeof v === 'number' ? v : typeof v}: the drive must be a finite `
    + `number. Refusing rather than coercing to zero, which would report a `
    + `confident stability verdict for a drive that does not exist.`);
  return v;
}

class FaradayDNS {
  constructor(o){
    this.nx = o.nx; this.ns = o.ns;
    this.L = o.L; this.h0 = o.h0;
    this.rho = o.rho; this.nu = o.nu; this.gamma = o.gamma;
    this.g = o.g === undefined ? G0 : o.g;
    this.accel = o.accel === undefined ? 0 : requireFinite(o.accel, 'accel');
    this.omegaD = o.omegaD === undefined ? 0 : requireFinite(o.omegaD, 'omegaD');
    if (!(this.ns >= 3)) throw new Error(
      `ns = ${this.ns}: the free-surface stress balance needs three cells of `
      + `depth to difference against. Refusing rather than falling back to a `
      + `copy-ghost, which is the approximation this solver exists to avoid.`);
    this.dx = o.L/o.nx; this.ds = 1/o.ns;
    this.t = 0;
    const nx = this.nx, ns = this.ns;
    this.H  = new Float64Array(nx).fill(o.h0);
    this.u  = new Float64Array(nx*ns);
    this.w  = new Float64Array(nx*(ns+1));
    this.p  = new Float64Array(nx*ns);
    this._div = new Float64Array(nx*ns);
    this._gu = new Float64Array(nx*ns);
    this._gw = new Float64Array(nx*(ns+1));
    this._kap = new Float64Array(nx);
    this._r = new Float64Array(nx*ns);
    this._d = new Float64Array(nx*ns);
    this._q = new Float64Array(nx*ns);
    this._ps = new Float64Array(nx);
    this._sux = new Float64Array(nx);
    this._Ht = new Float64Array(nx);
    this._uBot = new Float64Array(nx);
    this._uTop = new Float64Array(nx);
    this._wTop = new Float64Array(nx);
    this._lu = new Float64Array(nx*ns);
    this._lw = new Float64Array(nx*(ns+1));
    this._uN = new Float64Array(nx*ns);
    this._wN = new Float64Array(nx*(ns+1));
    this._Hn = new Float64Array(nx);
    this.cgIters = 0; this.cgResidual = 0;
  }
  idxU(i, j){ return ((i % this.nx + this.nx) % this.nx)*this.ns + j; }
  idxW(i, j){ return ((i % this.nx + this.nx) % this.nx)*(this.ns + 1) + j; }
  idxP(i, j){ return ((i % this.nx + this.nx) % this.nx)*this.ns + j; }

  Hface(i){ const nx = this.nx, im = ((i - 1) % nx + nx) % nx, ic = ((i % nx) + nx) % nx;
            return 0.5*(this.H[im] + this.H[ic]); }
  Hx(i){ const nx = this.nx, im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
         return (this.H[ip] - this.H[im])/(2*this.dx); }
  Hxx(i){ const nx = this.nx, im = (i - 1 + nx) % nx, ic = ((i % nx) + nx) % nx, ip = (ic + 1) % nx;
          return (this.H[ip] - 2*this.H[ic] + this.H[im])/(this.dx*this.dx); }

  divergence(u, w, out){
    const { nx, ns, dx, ds } = this;
    for (let i = 0; i < nx; i++){
      const Hxi = this.Hx(i);
      const HfL = this.Hface(i), HfR = this.Hface(i + 1);
      for (let j = 0; j < ns; j++){
        const c = i*ns + j;
        out[c] = (HfR*u[this.idxU(i+1, j)] - HfL*u[this.idxU(i, j)])/dx
               + (this.vFlux(u, w, i, j+1, Hxi) - this.vFlux(u, w, i, j, Hxi))/ds;
      }
    }
    return out;
  }

  vFlux(u, w, i, j, Hxi){
    const wv = w[this.idxW(i, j)];
    if (j === 0) return wv;
    return wv - (j*this.ds)*this.uAtSFace(u, i, j)*Hxi;
  }

  uAtSFace(u, i, j){
    const { nx, ns } = this;
    const ic = ((i % nx) + nx) % nx, ir = (ic + 1) % nx;
    if (j === ns) return 0.5*(u[ic*ns + ns-1] + u[ir*ns + ns-1]);

    if (j === 0) return 0;
    return 0.25*(u[ic*ns + j-1] + u[ir*ns + j-1] + u[ic*ns + j] + u[ir*ns + j]);
  }

  gradient(q, gu, gw){
    const { nx, ns, dx, ds } = this;
    gu.fill(0); gw.fill(0);
    for (let i = 0; i < nx; i++){
      const Hxi = this.Hx(i);
      const HfL = this.Hface(i), HfR = this.Hface(i + 1);
      for (let j = 0; j < ns; j++){
        const qc = q[i*ns + j];
        gu[this.idxU(i+1, j)] += qc*HfR/dx;
        gu[this.idxU(i,   j)] -= qc*HfL/dx;
        this.vFluxT(gu, gw, i, j+1, Hxi,  qc/ds);
        this.vFluxT(gu, gw, i, j,   Hxi, -qc/ds);
      }
    }

    for (let i = 0; i < nx; i++){
      const wi = -1/this.Hface(i);
      for (let j = 0; j < ns; j++) gu[i*ns + j] *= wi;
    }
    for (let i = 0; i < nx; i++){
      const wi = -1/this.H[i];
      gw[i*(ns+1) + 0] = 0;
      for (let j = 1; j < ns; j++) gw[i*(ns+1) + j] *= wi;
      gw[i*(ns+1) + ns] *= 2*wi;
    }
  }

  vFluxT(gu, gw, i, j, Hxi, k){
    const { nx, ns, ds } = this;
    gw[this.idxW(i, j)] += k;
    if (j === 0) return;
    const ic = ((i % nx) + nx) % nx, ir = (ic + 1) % nx;
    const kk = -k*(j*ds)*Hxi;
    if (j === ns){
      gu[ic*ns + ns-1] += 0.5*kk; gu[ir*ns + ns-1] += 0.5*kk;
    } else {
      gu[ic*ns + j-1] += 0.25*kk; gu[ir*ns + j-1] += 0.25*kk;
      gu[ic*ns + j  ] += 0.25*kk; gu[ir*ns + j  ] += 0.25*kk;
    }
  }

  metricWeight(kind, i, j){
    if (kind === 'u') return this.Hface(i);
    if (j === 0) return 0;
    return j === this.ns ? this.H[i]/2 : this.H[i];
  }
}
FARADAY_DNS.FaradayDNS = FaradayDNS;
if (typeof module !== 'undefined' && module.exports) module.exports.FaradayDNS = FaradayDNS;

FaradayDNS.prototype.surfaceHt = function(out){
  const { nx, ns } = this;
  out = out || new Float64Array(nx);
  for (let i = 0; i < nx; i++)
    out[i] = this.w[i*(ns+1) + ns] - this.uAtSFace(this.u, i, ns)*this.Hx(i);
  return out;
};

FaradayDNS.prototype.omegaAtSFace = function(u, w, i, j, Ht){
  const s = j*this.ds, ic = ((i % this.nx) + this.nx) % this.nx;
  return (w[this.idxW(i, j)] - s*Ht[ic] - s*this.uAtSFace(u, i, j)*this.Hx(i))/this.H[ic];
};

FaradayDNS.prototype.lapU = function(f, gBot, gTop, out){
  const { nx, ns, dx, ds } = this;
  for (let i = 0; i < nx; i++){
    const im = (i - 1 + nx) % nx;
    const HF = this.Hface(i), HxF = 0.5*(this.Hx(im) + this.Hx(i));
    const HxxF = (this.Hx(i) - this.Hx(im))/dx;
    const cS = 2*HxF*HxF/(HF*HF) - HxxF/HF;
    const v = (fi, jj) => { const k = ((fi % nx) + nx) % nx;
      return jj < 0 ? gBot[k] : jj > ns - 1 ? gTop[k] : f[k*ns + jj]; };
    for (let j = 0; j < ns; j++){
      const s = (j + 0.5)*ds, A = s*HxF/HF, fc = f[i*ns + j];
      const fxp = v(i+1, j), fxm = v(i-1, j), fsp = v(i, j+1), fsm = v(i, j-1);
      const d2xs = ((v(i+1,j+1) - v(i+1,j-1)) - (v(i-1,j+1) - v(i-1,j-1)))/(4*dx*ds);
      const d2s = (fsp - 2*fc + fsm)/(ds*ds);
      out[i*ns + j] = (fxp - 2*fc + fxm)/(dx*dx) - 2*A*d2xs + A*A*d2s
                    + s*cS*(fsp - fsm)/(2*ds) + d2s/(HF*HF);
    }
  }
  return out;
};

FaradayDNS.prototype.lapW = function(f, gTop, out){
  const { nx, ns, dx, ds } = this;
  for (let i = 0; i < nx; i++){
    const Hi = this.H[i], Hxi = this.Hx(i);
    const cS = 2*Hxi*Hxi/(Hi*Hi) - this.Hxx(i)/Hi;
    const v = (fi, jj) => { const k = ((fi % nx) + nx) % nx;
      return jj > ns ? gTop[k] : f[k*(ns+1) + jj]; };
    out[i*(ns+1)] = 0;
    for (let j = 1; j <= ns; j++){
      const s = j*ds, A = s*Hxi/Hi, fc = f[i*(ns+1) + j];
      const fxp = v(i+1, j), fxm = v(i-1, j), fsp = v(i, j+1), fsm = v(i, j-1);
      const d2xs = ((v(i+1,j+1) - v(i+1,j-1)) - (v(i-1,j+1) - v(i-1,j-1)))/(4*dx*ds);
      const d2s = (fsp - 2*fc + fsm)/(ds*ds);
      out[i*(ns+1) + j] = (fxp - 2*fc + fxm)/(dx*dx) - 2*A*d2xs + A*A*d2s
                        + s*cS*(fsp - fsm)/(2*ds) + d2s/(Hi*Hi);
    }
  }
  return out;
};

FaradayDNS.prototype.applyL = function(q, out){
  this.gradient(q, this._gu, this._gw);
  this.divergence(this._gu, this._gw, out);
};

FaradayDNS.prototype.solveP = function(rhs, tol, maxIt){
  const n = rhs.length, p = this.p, r = this._r, d = this._d, q = this._q;
  this.applyL(p, q);
  let rr = 0;
  for (let i = 0; i < n; i++){ r[i] = rhs[i] - q[i]; d[i] = r[i]; rr += r[i]*r[i]; }
  const rr0 = rr;
  if (rr0 === 0){ this.cgIters = 0; this.cgResidual = 0; return 0; }
  let it = 0;
  for (; it < maxIt; it++){
    this.applyL(d, q);
    let dq = 0;
    for (let i = 0; i < n; i++) dq += d[i]*q[i];
    if (dq === 0) break;
    const alpha = rr/dq;
    let rr2 = 0;
    for (let i = 0; i < n; i++){ p[i] += alpha*d[i]; r[i] -= alpha*q[i]; rr2 += r[i]*r[i]; }
    if (Math.sqrt(rr2/rr0) < tol){ rr = rr2; it++; break; }
    const beta = rr2/rr; rr = rr2;
    for (let i = 0; i < n; i++) d[i] = r[i] + beta*d[i];
  }
  this.cgIters = it; this.cgResidual = Math.sqrt(rr/rr0);

  if (!(this.cgResidual < tol))
    throw new Error(
      `pressure solve did not converge: residual ${this.cgResidual.toExponential(3)} `
      + `after ${it} iterations against a tolerance of ${tol.toExponential(0)} `
      + `on a ${this.nx}x${this.ns} grid at t = ${this.t.toExponential(3)} s.`);
  return this.cgResidual;
};

FaradayDNS.prototype.step = function(dt){
  const { nx, ns, dx, ds, rho, nu } = this;
  const H = this.H, u = this.u, w = this.w;
  const gT = this.g + this.accel*Math.cos(this.omegaD*this.t);

  const sux = this.surfaceUx(this._sux);
  this.surfacePressure(this._ps, sux);

  const uN = this._uN, wN = this._wN;
  const at = (arr, i, j, n2) => arr[((i % nx + nx) % nx)*n2 + j];
  const Ht = this.surfaceHt(this._Ht);
  this.surfaceGhosts(this._uBot, this._uTop, this._wTop, sux);
  const uBot = this._uBot, uTop = this._uTop, wTop = this._wTop;
  this.lapU(u, uBot, uTop, this._lu);
  this.lapW(w, wTop, this._lw);

  const uG = (fi, jj) => { const k = ((fi % nx) + nx) % nx;
    return jj < 0 ? uBot[k] : jj > ns - 1 ? uTop[k] : u[k*ns + jj]; };
  for (let i = 0; i < nx; i++){
    const im = (i - 1 + nx) % nx;
    const HF = this.Hface(i), HxF = 0.5*(this.Hx(im) + this.Hx(i));
    const HtF = 0.5*(Ht[im] + Ht[i]);
    for (let j = 0; j < ns; j++){
      const s = (j + 0.5)*ds, c = i*ns + j, uc = u[c];
      const du_dx = (at(u, i+1, j, ns) - at(u, i-1, j, ns))/(2*dx);
      const du_ds = (uG(i, j+1) - uG(i, j-1))/(2*ds);
      const wHere = 0.25*(at(w, i, j, ns+1) + at(w, i, j+1, ns+1)
                        + at(w, i-1, j, ns+1) + at(w, i-1, j+1, ns+1));
      const Om = (wHere - s*HtF - s*uc*HxF)/HF;
      const dps_dx = (this._ps[i] - this._ps[im])/dx;
      uN[c] = uc + dt*( -uc*du_dx - Om*du_ds + nu*this._lu[c] - dps_dx/rho );
    }
  }

  const wG = (fi, jj) => { const k = ((fi % nx) + nx) % nx;
    return jj > ns ? wTop[k] : w[k*(ns+1) + jj]; };
  for (let i = 0; i < nx; i++){
    wN[i*(ns+1) + 0] = 0;
    for (let j = 1; j <= ns; j++){
      const c = i*(ns+1) + j, wc = w[c];
      const dw_dx = (at(w, i+1, j, ns+1) - at(w, i-1, j, ns+1))/(2*dx);
      const dw_ds = (wG(i, j+1) - w[i*(ns+1) + j-1])/(2*ds);
      const uHere = this.uAtSFace(u, i, j);
      const Om = this.omegaAtSFace(u, w, i, j, Ht);
      wN[c] = wc + dt*( -uHere*dw_dx - Om*dw_ds + nu*this._lw[c] - gT );
    }
  }

  this.divergence(uN, wN, this._div);
  for (let c = 0; c < this._div.length; c++) this._div[c] *= rho/dt;
  this.solveP(this._div, 1e-10, 40*(nx + ns));
  this.gradient(this.p, this._gu, this._gw);
  for (let c = 0; c < u.length; c++) u[c] = uN[c] - (dt/rho)*this._gu[c];
  for (let c = 0; c < w.length; c++) w[c] = wN[c] - (dt/rho)*this._gw[c];
  for (let i = 0; i < nx; i++) w[i*(ns+1)] = 0;

  const Hn = this._Hn;
  for (let i = 0; i < nx; i++)
    Hn[i] = H[i] + dt*(w[i*(ns+1) + ns] - this.uAtSFace(u, i, ns)*this.Hx(i));
  H.set(Hn);

  this.t += dt;
  return this;
};

FaradayDNS.prototype.surfExtrap = function(f, i0, ns){
  return (15*f[i0 + ns-1] - 10*f[i0 + ns-2] + 3*f[i0 + ns-3])/8;
};
FaradayDNS.prototype.surfDeriv = function(f, i0, ns){
  return (2*f[i0 + ns-1] - 3*f[i0 + ns-2] + f[i0 + ns-3])/this.ds;
};

FaradayDNS.prototype.surfaceUx = function(out){
  const { nx, ns, dx } = this;
  const u = this.u, H = this.H;
  for (let i = 0; i < nx; i++){
    const ir = (i + 1) % nx;

    const uxS = (this.surfExtrap(u, ir*ns, ns) - this.surfExtrap(u, i*ns, ns))/dx;
    const dudsC = 0.5*(this.surfDeriv(u, i*ns, ns) + this.surfDeriv(u, ir*ns, ns));
    out[i] = uxS - (this.Hx(i)/H[i])*dudsC;
  }
  return out;
};

FaradayDNS.prototype.surfacePressure = function(out, ux){
  const { nx, rho, nu, gamma } = this;
  curvature(this.H, this.dx, this._kap);
  for (let i = 0; i < nx; i++){
    const Hxi = this.Hx(i);
    out[i] = -gamma*this._kap[i] - 2*rho*nu*ux[i]*(1 + Hxi*Hxi)/(1 - Hxi*Hxi);
  }
  return out;
};

FaradayDNS.prototype.surfaceGhosts = function(uBot, uTop, wTop, ux){
  const { nx, ns, dx, ds } = this;
  const u = this.u, w = this.w, H = this.H;
  const at = (arr, i, j, n2) => arr[((i % nx + nx) % nx)*n2 + j];
  for (let i = 0; i < nx; i++) uBot[i] = -u[i*ns + 0];
  for (let i = 0; i < nx; i++){
    const im = (i - 1 + nx) % nx;
    const HF = this.Hface(i), HxF = 0.5*(this.Hx(im) + this.Hx(i));

    const ip = (i + 1) % nx;
    const dudsT = this.surfDeriv(u, i*ns, ns);
    const ux_z = (this.surfExtrap(u, ip*ns, ns) - this.surfExtrap(u, im*ns, ns))/(2*dx)
               - (HxF/HF)*dudsT;

    const dwdsF = 0.5*((3*w[i*(ns+1)+ns]  - 4*w[i*(ns+1)+ns-1]  + w[i*(ns+1)+ns-2])
                     + (3*w[im*(ns+1)+ns] - 4*w[im*(ns+1)+ns-1] + w[im*(ns+1)+ns-2]))/(2*ds);
    const wx_z = (w[i*(ns+1)+ns] - w[im*(ns+1)+ns])/dx - (HxF/HF)*dwdsF;
    const uz_S = -wx_z + 4*HxF*ux_z/(1 - HxF*HxF);
    uTop[i] = u[i*ns + ns-1] + ds*HF*uz_S;
  }
  for (let i = 0; i < nx; i++) wTop[i] = w[i*(ns+1) + ns-1] - 2*ds*H[i]*ux[i];
};

FaradayDNS.prototype.maxDivergence = function(){
  this.divergence(this.u, this.w, this._div);
  let m = 0;
  for (let i = 0; i < this.nx; i++)
    for (let j = 0; j < this.ns; j++)
      m = Math.max(m, Math.abs(this._div[i*this.ns + j])/this.H[i]);
  return m;
};
FaradayDNS.prototype.surfaceAmplitude = function(){
  let mean = 0; for (let i = 0; i < this.nx; i++) mean += this.H[i];
  mean /= this.nx;
  let a = 0; for (let i = 0; i < this.nx; i++) a = Math.max(a, Math.abs(this.H[i] - mean));
  return a;
};
