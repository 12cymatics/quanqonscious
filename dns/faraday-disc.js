'use strict';

/* Linearised free-surface Navier-Stokes in a circular cell, at one azimuthal
   mode number, on graded grids in r and z.

   WHAT THIS SOLVES. The incompressible Navier-Stokes equations linearised about
   the flat resting state of a liquid layer in a cylinder of radius R and depth
   h, under gravity g + a cos(omega_d t), with no-slip on the floor and the
   sidewall, zero tangential stress and the full normal-stress balance at the
   free surface, and the contact line either free (dEta/dr = 0) or pinned
   (Eta = 0). Perturbations go as e^{i m theta}, which for a resting axisymmetric
   base state decouples the azimuthal modes EXACTLY -- so a three-dimensional
   question becomes a two-dimensional one with nothing discarded. Linearising is
   not a reduction either: Floquet stability of the flat state is the linear
   problem.

   The real representation carries u_r and w and eta as cos(m theta) and u_theta
   as sin(m theta), so every stored field is real.

   WHY THE GRIDS ARE GRADED. The damping sets the Faraday threshold, and the
   damping lives in Stokes layers. On this apparatus delta = sqrt(2 nu/omega) is
   76 micrometres against a 3 mm depth. Uniform grids measured that badly and,
   worse, inconsistently: at m = 12, nr x nz of 32x12, 40x16 and 48x20 gave
   Floquet growth -5.82, -6.94 and -8.47 per second, drifting further with each
   refinement rather than converging, because each refinement resolved a little
   more of a layer that was never resolved. Grading toward the floor, the free
   surface and the sidewall puts cells where the gradients are. It costs metric
   factors and buys the layers; it is a coordinate stretch and changes no
   equation.

   Grading the radius also relaxes the step limit. The stiffest capillary mode is
   set by m/r at the innermost cell, so widening that cell -- where, for m = 12,
   the mode amplitude is smaller than round-off anyway -- raises the allowed step
   directly.

   ALL DIFFERENCES ARE IN CONSERVATIVE FLUX FORM. That is what makes the axis and
   the contact line fall out rather than needing ghost values: at r = 0 the inner
   face area is exactly zero, so the flux through it is zero, and a free contact
   line is a zero flux at the rim. The projection's gradient is built as the exact
   transpose of its divergence, so divergence(gradient(.)) is symmetric negative
   definite by construction and conjugate gradients is entitled to converge on it.
*/

const DISC_G0 = 9.80665;

function requireFinitePositive(v, name){
  if (typeof v !== 'number' || !Number.isFinite(v) || !(v > 0)) throw new TypeError(
    `${name} = ${v}: a finite positive number is required. Refusing rather than `
    + `coercing, which would report a confident stability verdict for an `
    + `apparatus that does not exist.`);
  return v;
}
function requireFinite(v, name){
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(
    `${name} = ${v}: a finite number is required.`);
  return v;
}
function requireIntegerAtLeast(v, lo, name){
  if (!Number.isInteger(v) || v < lo) throw new TypeError(
    `${name} = ${v}: an integer of at least ${lo} is required.`);
  return v;
}

/* One-sided stretch onto [0, L], clustering toward L. stretch = 0 is uniform;
   larger values pull cells toward the far end.

   The first version of this had the map the other way round -- L*(1 - tanh(s(1 -
   i/n))/tanh(s)) -- which clusters at ZERO, the axis, where for m = 12 the mode
   is smaller than round-off. Measured: the innermost cell centre landed at
   1.4e-5 m, so the azimuthal wavenumber m/r there was 8.4e5 per metre and the
   step limit fell to 3.5e-7 s, eight times smaller than on a uniform grid, for
   resolution in the one place it buys nothing. */
function gradeToEnd(n, L, stretch){
  const f = new Float64Array(n + 1);
  if (stretch === 0){ for (let i = 0; i <= n; i++) f[i] = L*i/n; return f; }
  const th = Math.tanh(stretch);
  for (let i = 0; i <= n; i++)
    f[i] = L*Math.tanh(stretch*i/n)/th;
  f[0] = 0; f[n] = L;
  return f;
}

/* Symmetric stretch onto [0, L], clustering toward BOTH ends. */
function gradeBothEnds(n, L, stretch){
  const f = new Float64Array(n + 1);
  if (stretch === 0){ for (let j = 0; j <= n; j++) f[j] = L*j/n; return f; }
  const th = Math.tanh(stretch);
  for (let j = 0; j <= n; j++)
    f[j] = L*0.5*(1 + Math.tanh(stretch*(2*j/n - 1))/th);
  f[0] = 0; f[n] = L;
  return f;
}

class FaradayDisc {
  constructor(o){
    this.m  = requireIntegerAtLeast(o.m, 0, 'm');
    this.nr = requireIntegerAtLeast(o.nr, 6, 'nr');
    this.nz = requireIntegerAtLeast(o.nz, 4, 'nz');
    this.R  = requireFinitePositive(o.R, 'R');
    this.h  = requireFinitePositive(o.h, 'h');
    this.rho   = requireFinitePositive(o.rho, 'rho');
    this.nu    = requireFinitePositive(o.nu, 'nu');
    this.gamma = requireFinitePositive(o.gamma, 'gamma');
    this.g     = o.g === undefined ? DISC_G0 : requireFinitePositive(o.g, 'g');
    this.accel  = o.accel  === undefined ? 0 : requireFinite(o.accel, 'accel');
    this.omegaD = o.omegaD === undefined ? 0 : requireFinite(o.omegaD, 'omegaD');
    this.contact = o.contact === undefined ? 'free' : o.contact;
    if (this.contact !== 'free' && this.contact !== 'pinned') throw new TypeError(
      `contact = ${JSON.stringify(this.contact)}: the contact line is either `
      + `'free' (dEta/dr = 0 at the rim) or 'pinned' (Eta = 0 there). There is no `
      + `default that covers both, and the two give different thresholds.`);
    if (this.m === 1) throw new RangeError(
      `m = 1 is not implemented. The radial momentum's singular group is `
      + `-[(m^2+1)u + 2m v]/r^2, which at m = 1 reads -2(u + v)/r^2 and stays `
      + `finite only through the cancellation u + v ~ r^2, while u - v tends to a `
      + `nonzero constant: a rigid sideways translation of the axis. Every other m `
      + `has u and v vanishing at r = 0, which is what the conservative flux form `
      + `here imposes. Refusing rather than returning the number that condition `
      + `produces for the one m where it is wrong.`);

    /* Grading strengths. Defaults are chosen so the Stokes layer of the mode
       being asked about falls inside the first few cells; a caller that knows
       the layer thickness can say so directly. */
    const rs = o.rStretch === undefined ? 2.2 : requireFinite(o.rStretch, 'rStretch');
    const zs = o.zStretch === undefined ? 2.2 : requireFinite(o.zStretch, 'zStretch');
    if (rs < 0 || zs < 0) throw new RangeError(
      `rStretch = ${rs}, zStretch = ${zs}: a negative stretch inverts the grading `
      + `and puts the cells where the gradients are not.`);
    this.rStretch = rs; this.zStretch = zs;

    const nr = this.nr, nz = this.nz;
    this.rf = gradeToEnd(nr, this.R, rs);       // r faces, clustered at the rim
    this.zf = gradeBothEnds(nz, this.h, zs);    // z faces, clustered at both
    this.rc = new Float64Array(nr);
    this.zc = new Float64Array(nz);
    for (let i = 0; i < nr; i++) this.rc[i] = 0.5*(this.rf[i] + this.rf[i+1]);
    for (let j = 0; j < nz; j++) this.zc[j] = 0.5*(this.zf[j] + this.zf[j+1]);
    this.drc = new Float64Array(nr);            // cell widths
    this.dzc = new Float64Array(nz);
    for (let i = 0; i < nr; i++) this.drc[i] = this.rf[i+1] - this.rf[i];
    for (let j = 0; j < nz; j++) this.dzc[j] = this.zf[j+1] - this.zf[j];
    /* Centre-to-centre distances, indexed by the face between them. Index 0 and
       n reach the boundary itself, where the value is prescribed. */
    this.drf = new Float64Array(nr + 1);
    this.dzf = new Float64Array(nz + 1);
    this.drf[0] = this.rc[0] - 0;
    for (let i = 1; i < nr; i++) this.drf[i] = this.rc[i] - this.rc[i-1];
    this.drf[nr] = this.R - this.rc[nr-1];
    this.dzf[0] = this.zc[0] - 0;
    for (let j = 1; j < nz; j++) this.dzf[j] = this.zc[j] - this.zc[j-1];
    this.dzf[nz] = this.h - this.zc[nz-1];

    this.t = 0;
    this.u = new Float64Array((nr + 1)*nz);
    this.v = new Float64Array((nr + 1)*nz);
    this.w = new Float64Array(nr*(nz + 1));
    this.p = new Float64Array(nr*nz);
    this.eta = new Float64Array(nr);

    this._us = new Float64Array((nr + 1)*nz);
    this._vs = new Float64Array((nr + 1)*nz);
    this._ws = new Float64Array(nr*(nz + 1));
    this._lu = new Float64Array((nr + 1)*nz);
    this._lv = new Float64Array((nr + 1)*nz);
    this._lw = new Float64Array(nr*(nz + 1));
    this._div = new Float64Array(nr*nz);
    this._gu = new Float64Array((nr + 1)*nz);
    this._gv = new Float64Array((nr + 1)*nz);
    this._gw = new Float64Array(nr*(nz + 1));
    this._ps = new Float64Array(nr);
    this._r = new Float64Array(nr*nz);
    this._d = new Float64Array(nr*nz);
    this._q = new Float64Array(nr*nz);
    this._dudz = new Float64Array(nr + 1);
    this._dvdz = new Float64Array(nr + 1);

    this.cgIters = 0; this.cgResidual = 0;
  }

  iu(i, j){ return i*this.nz + j; }
  iw(i, j){ return i*(this.nz + 1) + j; }
  ip(i, j){ return i*this.nz + j; }

  gravity(){ return this.g + this.accel*Math.cos(this.omegaD*this.t); }

  /* Cell sizes the caller may want to compare against a Stokes depth. */
  resolution(){
    return { drMin: Math.min(...this.drc), drMax: Math.max(...this.drc),
             dzMin: Math.min(...this.dzc), dzMax: Math.max(...this.dzc),
             firstCellFromRim: this.drc[this.nr-1],
             firstCellFromFloor: this.dzc[0],
             firstCellFromSurface: this.dzc[this.nz-1],
             innermostRadius: this.rc[0] };
  }

  /* Surface Laplacian of the elevation, D_m eta = eta_rr + eta_r/r - m^2 eta/r^2,
     in flux form. For eta proportional to J_m(k r) this is exactly -k^2 eta,
     which is what turns the capillary term into sigma k^2. The axis needs no
     ghost: its face area is zero. The rim carries the contact line -- free is a
     zero flux, pinned is the flux to a zero value at r = R. */
  surfaceLaplacian(i){
    const nr = this.nr, m = this.m, rc = this.rc, rf = this.rf, eta = this.eta;
    const outer = i === nr - 1
      ? (this.contact === 'free' ? 0
         : rf[nr]*(0 - eta[nr-1])/this.drf[nr])
      : rf[i+1]*(eta[i+1] - eta[i])/this.drf[i+1];
    const inner = i === 0 ? 0 : rf[i]*(eta[i] - eta[i-1])/this.drf[i];
    return (outer - inner)/(rc[i]*this.drc[i]) - m*m*eta[i]/(rc[i]*rc[i]);
  }

  /* dw/dz at the free surface. Second order on the graded grid: the three
     vertical faces below the surface, weighted by their actual spacings. */
  wzSurface(i){
    const nz = this.nz, zf = this.zf;
    const w0 = this.w[this.iw(i, nz)], w1 = this.w[this.iw(i, nz-1)],
          w2 = this.w[this.iw(i, nz-2)];
    const a = zf[nz] - zf[nz-1], b = zf[nz] - zf[nz-2];
    /* derivative of the quadratic through (0,w0), (-a,w1), (-b,w2) at 0 */
    return (w0*(a + b)/(a*b)) - (w1*b/(a*(b - a))) + (w2*a/(b*(b - a)));
  }

  /* The pressure the free surface carries: the hydrostatic response of the
     displaced elevation to the instantaneous gravity, the capillary term, and
     the viscous normal stress. This is the inhomogeneous Dirichlet value on the
     projection, which is also what removes that projection's null space. */
  surfacePressure(out){
    const rho = this.rho, g = this.gravity();
    for (let i = 0; i < this.nr; i++)
      out[i] = rho*g*this.eta[i] - this.gamma*this.surfaceLaplacian(i)
             + 2*rho*this.nu*this.wzSurface(i);
    return out;
  }

  /* Zero tangential stress at the free surface: du/dz = -dw/dr and
     dv/dz = (m/r) w. Applied as the Neumann value the vertical difference needs
     at z = h, rather than through a ghost row. */
  surfaceSlopes(dudz, dvdz){
    const nr = this.nr, nz = this.nz, m = this.m;
    const wS = i => this.w[this.iw(i, nz)];
    for (let i = 0; i <= nr; i++){
      if (i === 0 || i === nr){ dudz[i] = 0; dvdz[i] = 0; continue; }
      dudz[i] = -(wS(i) - wS(i-1))/this.drf[i];
      const wAtFace = (this.drc[i-1]*wS(i) + this.drc[i]*wS(i-1))
                      /(this.drc[i-1] + this.drc[i]);
      dvdz[i] = (m/this.rf[i])*wAtFace;
    }
  }

  /* Vector Laplacian of the (u_r, u_theta) pair at r faces. The scalar part is
     in flux form; the coupling group -[(m^2+1)f + 2m g]/r^2 is passed in. Both
     components are needed: dropping either changes the equation, not the
     scheme. */
  lapUV(f, other, dfdzTop, sign, out){
    const nr = this.nr, nz = this.nz, m = this.m;
    const rf = this.rf, rc = this.rc, drf = this.drf, drc = this.drc;
    const zc = this.zc, zf = this.zf, dzc = this.dzc, dzf = this.dzf;
    for (let i = 1; i < nr; i++){
      const r = rf[i];
      /* radial flux form for a field living at r faces: the faces of ITS control
         volume are the cell centres either side. */
      const wIn = rc[i-1]/ (drc[i-1]), wOut = rc[i]/(drc[i]);
      const vol = r*0.5*(drc[i-1] + drc[i]);
      for (let j = 0; j < nz; j++){
        const c = this.iu(i, j);
        const fc = f[c];
        const fIn = f[this.iu(i-1, j)], fOut = f[this.iu(i+1, j)];
        const radial = (wOut*(fOut - fc) - wIn*(fc - fIn))/vol;
        const below = j === 0 ? (fc - 0)/dzf[0] : (fc - f[this.iu(i, j-1)])/dzf[j];
        const above = j === nz - 1 ? dfdzTop[i]
                                   : (f[this.iu(i, j+1)] - fc)/dzf[j+1];
        const vertical = (above - below)/dzc[j];
        const coupling = -((m*m + 1)*fc + sign*2*m*other[c])/(r*r);
        out[c] = radial + vertical + coupling;
      }
    }
    return out;
  }

  lapW(out){
    const nr = this.nr, nz = this.nz, m = this.m, w = this.w;
    const rf = this.rf, rc = this.rc, drf = this.drf, drc = this.drc;
    const zf = this.zf, zc = this.zc, dzc = this.dzc, dzf = this.dzf;
    for (let i = 0; i < nr; i++){
      const r = rc[i];
      const inArea = rf[i], outArea = rf[i+1];
      const vol = r*drc[i];
      for (let j = 1; j <= nz; j++){
        const c = this.iw(i, j);
        const wc = w[c];
        /* axis: inArea is exactly zero, so no ghost is needed.
           rim: w vanishes there, so the outward flux is to a zero. */
        const fluxIn  = i === 0 ? 0
                                : inArea*(wc - w[this.iw(i-1, j)])/drf[i];
        const fluxOut = i === nr - 1
          ? outArea*(0 - wc)/drf[nr]
          : outArea*(w[this.iw(i+1, j)] - wc)/drf[i+1];
        const radial = (fluxOut - fluxIn)/vol;
        const below = j === 1 ? (wc - 0)/dzc[0]
                              : (wc - w[this.iw(i, j-1)])/dzc[j-1];
        const above = j === nz
          ? /* dw/dz above the surface is not available; the surface value of
               dw/dz itself is, so the one-sided second difference closes on it */
            below + 2*(this.wzSurface(i) - below)
          : (w[this.iw(i, j+1)] - wc)/dzc[j];
        const vertical = (above - below)/dzf[j];
        out[c] = radial + vertical - m*m*wc/(r*r);
      }
    }
    return out;
  }

  /* Integral-form divergence over each cell: the flux sum, unnormalised. The
     azimuthal term loses its 1/r to the cell volume exactly, which is why it
     appears with no radius in it. */
  divergence(u, v, w, out){
    const nr = this.nr, nz = this.nz, m = this.m;
    const rf = this.rf, rc = this.rc, drc = this.drc, dzc = this.dzc;
    for (let i = 0; i < nr; i++){
      for (let j = 0; j < nz; j++){
        const vbar = 0.5*(v[this.iu(i, j)] + v[this.iu(i+1, j)]);
        out[this.ip(i, j)] =
            dzc[j]*(rf[i+1]*u[this.iu(i+1, j)] - rf[i]*u[this.iu(i, j)])
          + rc[i]*drc[i]*(w[this.iw(i, j+1)] - w[this.iw(i, j)])
          + m*drc[i]*dzc[j]*vbar;
      }
    }
    return out;
  }

  /* Exactly minus the transpose of `divergence`, divided by a positive diagonal
     weight. Built this way rather than differenced independently so that
     divergence(gradient(q)) is symmetric negative definite by construction. The
     weights are chosen so that the result IS the gradient: each is the flux
     coefficient times the distance the difference actually spans, which on a
     graded grid differs face by face and, at the free surface, is the half cell
     from the last pressure centre up to z = h. */
  gradient(q, gu, gv, gw){
    const nr = this.nr, nz = this.nz, m = this.m;
    const rf = this.rf, rc = this.rc, drc = this.drc, dzc = this.dzc;
    gu.fill(0); gv.fill(0); gw.fill(0);
    for (let i = 0; i < nr; i++){
      for (let j = 0; j < nz; j++){
        const qc = q[this.ip(i, j)];
        gu[this.iu(i+1, j)] += qc*dzc[j];
        gu[this.iu(i,   j)] -= qc*dzc[j];
        gw[this.iw(i, j+1)] += qc*rc[i]*drc[i];
        gw[this.iw(i, j  )] -= qc*rc[i]*drc[i];
        gv[this.iu(i,   j)] += qc*m*drc[i]*dzc[j]*0.5;
        gv[this.iu(i+1, j)] += qc*m*drc[i]*dzc[j]*0.5;
      }
    }
    for (let i = 1; i < nr; i++){
      const span = this.drf[i];
      const halfWidth = 0.5*(drc[i-1] + drc[i]);
      for (let j = 0; j < nz; j++){
        gu[this.iu(i, j)] /= -(dzc[j]*span);
        gv[this.iu(i, j)] /= -(rf[i]*dzc[j]*halfWidth);
      }
    }
    for (let j = 0; j < nz; j++){       // walls: u and v are prescribed there
      gu[this.iu(0, j)] = 0; gv[this.iu(0, j)] = 0;
      gu[this.iu(nr, j)] = 0; gv[this.iu(nr, j)] = 0;
    }
    for (let i = 0; i < nr; i++){
      gw[this.iw(i, 0)] = 0;            // no-slip floor
      for (let j = 1; j <= nz; j++)
        gw[this.iw(i, j)] /= -(rc[i]*drc[i]*this.dzf[j]);
    }
    return [gu, gv, gw];
  }

  applyL(q, out){
    this.gradient(q, this._gu, this._gv, this._gw);
    this.divergence(this._gu, this._gv, this._gw, out);
    return out;
  }

  solveP(rhs, tol, maxIt){
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
    if (!(this.cgResidual < tol)) throw new Error(
      `disc pressure solve did not converge: residual `
      + `${this.cgResidual.toExponential(3)} after ${it} iterations against a `
      + `tolerance of ${tol.toExponential(0)} on a ${this.nr}x${this.nz} grid at `
      + `m = ${this.m}, t = ${this.t.toExponential(3)} s.`);
    return this.cgResidual;
  }

  step(dt){
    const nr = this.nr, nz = this.nz, nu = this.nu, rho = this.rho;
    const u = this.u, v = this.v, w = this.w;
    const us = this._us, vs = this._vs, ws = this._ws;

    this.surfaceSlopes(this._dudz, this._dvdz);
    this.lapUV(u, v, this._dudz, +1, this._lu);
    this.lapUV(v, u, this._dvdz, -1, this._lv);
    this.lapW(this._lw);

    us.set(u); vs.set(v); ws.set(w);
    for (let i = 1; i < nr; i++)
      for (let j = 0; j < nz; j++){
        const c = this.iu(i, j);
        us[c] = u[c] + dt*nu*this._lu[c];
        vs[c] = v[c] + dt*nu*this._lv[c];
      }
    for (let i = 0; i < nr; i++)
      for (let j = 1; j <= nz; j++){
        const c = this.iw(i, j);
        ws[c] = w[c] + dt*nu*this._lw[c];
      }

    /* The surface pressure is an inhomogeneous Dirichlet value on the
       projection, not a force on the predictor. As a predictor force it is
       P_s/(rho * half cell) -- O(1/dz) -- and the projection then cancels almost
       all of it, leaving the physical acceleration as the difference of two large
       numbers and a step limit that collapses with the grid. Measured at nr=48,
       nz=20, m=12: the fields went non-finite 0.57 periods in. Resolved inside
       the projection instead, nothing large cancels: the surface face's gradient
       gains P_s/dzf[nz] and the top row of the right-hand side loses that term's
       divergence. */
    this.surfacePressure(this._ps);
    this.divergence(us, vs, ws, this._div);
    for (let c = 0; c < this._div.length; c++) this._div[c] *= rho/dt;
    for (let i = 0; i < nr; i++)
      this._div[this.ip(i, nz-1)] -=
        this.rc[i]*this.drc[i]*this._ps[i]/this.dzf[nz];
    this.solveP(this._div, 1e-11, 40*(nr + nz));

    this.gradient(this.p, this._gu, this._gv, this._gw);
    for (let i = 0; i < nr; i++)
      this._gw[this.iw(i, nz)] += this._ps[i]/this.dzf[nz];

    for (let i = 1; i < nr; i++)
      for (let j = 0; j < nz; j++){
        const c = this.iu(i, j);
        u[c] = us[c] - (dt/rho)*this._gu[c];
        v[c] = vs[c] - (dt/rho)*this._gv[c];
      }
    for (let i = 0; i < nr; i++)
      for (let j = 1; j <= nz; j++){
        const c = this.iw(i, j);
        w[c] = ws[c] - (dt/rho)*this._gw[c];
      }
    for (let i = 0; i < nr; i++) w[this.iw(i, 0)] = 0;

    /* eta from the corrected surface velocity, which pairs the surface pressure
       at the old time with the velocity at the new one: symplectic Euler on the
       surface oscillator, stable for omega dt < 2 rather than merely less
       unstable than forward Euler. */
    for (let i = 0; i < nr; i++) this.eta[i] += dt*w[this.iw(i, nz)];
    this.t += dt;
    return this;
  }

  maxDivergence(){
    this.divergence(this.u, this.v, this.w, this._div);
    let mx = 0;
    for (let i = 0; i < this.nr; i++)
      for (let j = 0; j < this.nz; j++)
        mx = Math.max(mx, Math.abs(this._div[this.ip(i, j)])
                          /(this.rc[i]*this.drc[i]*this.dzc[j]));
    return mx;
  }

  surfaceAmplitude(){
    let a = 0;
    for (let i = 0; i < this.nr; i++) a = Math.max(a, Math.abs(this.eta[i]));
    return a;
  }

  /* Mechanical energy of the perturbation: kinetic, gravitational, and the
     capillary energy of the stretched surface. With no drive this decreases
     monotonically, which is the check that the viscous terms and the surface
     conditions dissipate rather than produce. */
  energy(){
    const nr = this.nr, nz = this.nz, m = this.m, rho = this.rho;
    const rf = this.rf, rc = this.rc, drc = this.drc, dzc = this.dzc;
    let ke = 0, pe = 0;
    for (let i = 1; i < nr; i++){
      const vol = rf[i]*0.5*(drc[i-1] + drc[i]);
      for (let j = 0; j < nz; j++){
        const a = this.u[this.iu(i, j)], b = this.v[this.iu(i, j)];
        ke += 0.5*rho*(a*a + b*b)*vol*dzc[j];
      }
    }
    for (let i = 0; i < nr; i++)
      for (let j = 1; j <= nz; j++){
        const c = this.w[this.iw(i, j)];
        ke += 0.5*rho*c*c*rc[i]*drc[i]*this.dzf[j];
      }
    for (let i = 0; i < nr; i++){
      const e = this.eta[i];
      pe += 0.5*rho*this.g*e*e*rc[i]*drc[i];
      const eOut = i === nr - 1
        ? (this.contact === 'free' ? e : 0)
        : this.eta[i+1];
      const slope = (eOut - e)/this.drf[Math.min(i+1, nr)];
      pe += 0.5*this.gamma*(slope*slope + m*m*e*e/(rc[i]*rc[i]))*rc[i]*drc[i];
    }
    return { kinetic: ke, potential: pe, total: ke + pe };
  }

  /* Explicit step limit, from the discrete operators' own eigenvalue bounds
     rather than from a Cartesian rule of thumb.

     The first version used sqrt(rho dmin^3 / 2 pi sigma) and was wrong by a
     factor of three at m = 12, because it ignored the azimuthal direction: the
     stiffest capillary mode is set by m/r at the INNERMOST cell, not by dr.
     Measured with the old figure at dt = 2.58e-5: m = 12 blew up at step 37 with
     an amplification of 2.15 per step seeded from round-off, while m = 0 and
     m = 2 survived 4000 steps at the same step. That is what named the term. */
  stepLimits(){
    const m = this.m, rMin = this.rc[0];
    const drMin = Math.min(...this.drc), dzMin = Math.min(...this.dzc);
    const kr2 = 4/(drMin*drMin), kz2 = 4/(dzMin*dzMin), ka2 = m*m/(rMin*rMin);
    const kSurf2 = kr2 + ka2;
    const omegaSurf = Math.sqrt((this.g + Math.abs(this.accel)
                                 + (this.gamma/this.rho)*kSurf2)*(1/this.dzf[this.nz]));
    return { capillary: 2/omegaSurf,
             viscous: 2/(this.nu*(kr2 + kz2 + ka2)),
             kRadial: Math.sqrt(kr2), kVertical: Math.sqrt(kz2),
             kAzimuthal: Math.sqrt(ka2) };
  }

  stableStep(safety){
    const s = safety === undefined ? 0.4 : requireFinitePositive(safety, 'safety');
    const L = this.stepLimits();
    return s*Math.min(L.capillary, L.viscous);
  }

  /* The Stokes depth of a mode at the drive's response frequency, against the
     cells that have to resolve it. Reported rather than assumed, because the
     damping is what sets the Faraday threshold and an unresolved layer measures
     it wrongly while still returning a number. */
  stokesResolution(omega){
    const delta = Math.sqrt(2*this.nu/requireFinitePositive(omega, 'omega'));
    const res = this.resolution();
    return { delta,
             cellsInFloorLayer: delta/res.firstCellFromFloor,
             cellsInSurfaceLayer: delta/res.firstCellFromSurface,
             cellsInRimLayer: delta/res.firstCellFromRim };
  }
}

const FARADAY_DISC = { DISC_G0, FaradayDisc, gradeToEnd, gradeBothEnds };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_DISC;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_DISC = FARADAY_DISC;
