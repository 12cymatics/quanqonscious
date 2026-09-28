'use strict';
/* Nonlinear free-surface Navier-Stokes in a circular cell, in three dimensions,
   with every azimuthal mode present and coupled.

   WHAT THIS SOLVES. The incompressible Navier-Stokes equations for a liquid
   layer in a cylinder of radius R over a floor at z = 0, under gravity
   g + a cos(omega_d t), with no slip on the floor and the sidewall, zero
   tangential stress and the full normal-stress balance at the free surface
   z = h + eta(r, theta, t), and the contact line free or pinned. Nothing is
   linearised: the advective term u.grad u is carried in full, the surface
   curvature is the full mean curvature of the deformed surface, and the domain
   follows the surface rather than the surface conditions being applied at a
   fixed plane.

   HOW IT DIFFERS FROM dns/faraday-disc.js, AND WHY BOTH EXIST. That solver
   linearises about the flat state and takes one azimuthal mode at a time, which
   is not a shortcut for the question it asks -- Floquet stability of the flat
   state IS the linear problem, and about a resting axisymmetric base state the
   e^{i m theta} modes decouple exactly. It answers "does this mode grow", to
   eight digits, and it is the instrument for the threshold. It cannot answer
   "what pattern do you see", because a growing linear mode grows without bound:
   the visible figure is a saturated finite-amplitude state, and saturation is
   the nonlinearity. This solver answers that, and is what the renderer draws.

   THE SURFACE-FOLLOWING COORDINATE. z = sigma H(r, theta, t) with H = h + eta
   and sigma in [0, 1], so the top of the grid is the surface exactly, at any
   amplitude. Under that map the continuity equation is, exactly,

       r H div(u) = d_r(r H u) + d_theta(H v) + d_sigma(r Omega)

   with Omega = w - sigma (u dH/dr + (v/r) dH/dtheta) -- derived, not assumed;
   the derivation is a page of chain rule on d/dr|_z = d/dr|_sigma
   - (sigma H_r/H) d/dsigma and its fellows, and the terms group exactly.

   OMEGA IS THE THIRD UNKNOWN, NOT w. That choice is what keeps the projection
   exact. In terms of Omega the divergence above is a clean flux difference, so
   its gradient can be built as its exact transpose over the cell volumes, and
   divergence(gradient(.)) is symmetric negative definite by construction -- the
   same property dns/faraday-disc.js relies on, preserved in a moving frame. In
   terms of physical w it is not: w's flux carries the metric slopes, the
   transpose would smear pressure from one face into three velocity components,
   and the operator would lose the symmetry conjugate gradients needs.

   Two boundary conditions then fall out rather than being imposed:
     Omega = 0 at sigma = 0, because w = 0 on the floor and sigma kills the
       slope terms there;
     Omega at sigma = 1 equals w_s - u_s eta_r - (v_s/r) eta_theta, which IS the
       kinematic free-surface condition, so eta advances by dt times the top
       Omega and no separate surface equation is needed.
   Physical w is recovered where it is needed -- the viscous term and the
   reported field -- as w = H Omega/... no: as w = Omega + sigma(u H_r
   + (v/r) H_theta), which is the definition rearranged.

   ALL DIFFERENCES ARE IN CONSERVATIVE FLUX FORM, as next door: at r = 0 the
   inner face area is exactly zero, so the axis needs no ghost value and no
   special case beyond that; a free contact line is a zero flux at the rim.
   theta is periodic and uniform, so every azimuthal mode the grid can carry is
   present, and the advective term couples them -- which is the point.
*/

const CELL3D_G0 = 9.80665;

function requireFinitePositive(v, name){
  if (typeof v !== 'number' || !Number.isFinite(v) || !(v > 0)) throw new TypeError(
    `${name} = ${v}: a finite positive number is required.`);
  return v;
}
function requireFinite(v, name){
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(
    `${name} = ${v}: a finite number is required.`);
  return v;
}

/* The same graded maps the two-dimensional solver uses, so the grid code has one
   implementation and the two cannot drift. */
const GRADE = (function(){
  if (typeof require === 'function') return require('./faraday-disc.js');
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DISC)
    return globalThis.FARADAY_DISC;
  throw new Error(
    'faraday-cell3d: the graded grid maps live in dns/faraday-disc.js. Load it '
    + 'before this file, or require it under node.');
})();

class FaradayCell3D {
  constructor(o){
    const nr = this.nr = o.nr, nth = this.nth = o.nth, nz = this.nz = o.nz;
    if (!(Number.isInteger(nr) && nr >= 4)) throw new RangeError(
      `nr = ${o.nr}: at least four radial cells are needed for the surface stencil.`);
    if (!(Number.isInteger(nth) && nth >= 8 && nth % 2 === 0)) throw new RangeError(
      `nth = ${o.nth}: an even azimuthal count of at least eight is required. `
      + `Even, because the highest mode the grid carries is nth/2 and an odd count `
      + `leaves that mode without its conjugate; eight, because fewer cannot `
      + `represent the m = 4 patterns this cell shows at its lowest drives.`);
    if (!(Number.isInteger(nz) && nz >= 4)) throw new RangeError(
      `nz = ${o.nz}: at least four vertical cells are needed.`);

    this.R = requireFinitePositive(o.R, 'R');
    this.h = requireFinitePositive(o.h, 'h');
    this.rho = requireFinitePositive(o.rho, 'rho');
    this.nu = requireFinitePositive(o.nu, 'nu');
    this.gamma = requireFinitePositive(o.gamma, 'gamma');
    this.g = o.g === undefined ? CELL3D_G0 : requireFinitePositive(o.g, 'g');
    this.accel = o.accel === undefined ? 0 : requireFinite(o.accel, 'accel');
    this.omegaD = o.omegaD === undefined ? 0 : requireFinite(o.omegaD, 'omegaD');
    this.contact = o.contact === undefined ? 'free' : o.contact;
    if (this.contact !== 'free' && this.contact !== 'pinned') throw new TypeError(
      `contact = ${JSON.stringify(this.contact)}: the contact line is either `
      + `'free' or 'pinned'.`);

    const rs = o.rStretch === undefined ? 2.2 : requireFinite(o.rStretch, 'rStretch');
    const zs = o.zStretch === undefined ? 2.2 : requireFinite(o.zStretch, 'zStretch');
    if (rs < 0 || zs < 0) throw new RangeError(
      `rStretch = ${rs}, zStretch = ${zs}: a negative stretch would cluster cells `
      + `where the gradients are not.`);
    this.rStretch = rs; this.zStretch = zs;

    /* r as next door: faces clustered at the rim, where the sidewall layer is.
       sigma on the unit interval, clustered at both ends, for the floor and the
       free-surface layers. theta uniform, because it is periodic and no point on
       the circle is special. */
    this.rf = GRADE.gradeToEnd(nr, this.R, rs);
    this.sf = GRADE.gradeBothEnds(nz, 1, zs);
    this.dth = 2*Math.PI/nth;

    this.rc = new Float64Array(nr);
    this.sc = new Float64Array(nz);
    for (let i = 0; i < nr; i++) this.rc[i] = 0.5*(this.rf[i] + this.rf[i+1]);
    for (let j = 0; j < nz; j++) this.sc[j] = 0.5*(this.sf[j] + this.sf[j+1]);
    this.drc = new Float64Array(nr);
    this.dsc = new Float64Array(nz);
    for (let i = 0; i < nr; i++) this.drc[i] = this.rf[i+1] - this.rf[i];
    for (let j = 0; j < nz; j++) this.dsc[j] = this.sf[j+1] - this.sf[j];
    this.drf = new Float64Array(nr + 1);
    this.dsf = new Float64Array(nz + 1);
    this.drf[0] = this.rc[0];
    for (let i = 1; i < nr; i++) this.drf[i] = this.rc[i] - this.rc[i-1];
    this.drf[nr] = this.R - this.rc[nr-1];
    this.dsf[0] = this.sc[0];
    for (let j = 1; j < nz; j++) this.dsf[j] = this.sc[j] - this.sc[j-1];
    this.dsf[nz] = 1 - this.sc[nz-1];

    const NU = (nr + 1)*nth*nz, NV = nr*nth*nz, NW = nr*nth*(nz + 1),
          NP = nr*nth*nz, NE = nr*nth;
    this.NU = NU; this.NV = NV; this.NW = NW; this.NP = NP; this.NE = NE;

    this.t = 0;
    this.u   = new Float64Array(NU);     // radial, at r faces
    this.v   = new Float64Array(NV);     // azimuthal, at theta faces
    this.om  = new Float64Array(NW);     // Omega, at sigma faces
    this.p   = new Float64Array(NP);
    this.eta = new Float64Array(NE);

    /* metric: H at cell centres, and interpolated to r and theta faces */
    this.H    = new Float64Array(NE);
    this.Hr   = new Float64Array((nr + 1)*nth);
    this.Hth  = new Float64Array(nr*nth);
    this.Hdr  = new Float64Array(NE);    // dH/dr at cell centres
    this.Hdth = new Float64Array(NE);    // dH/dtheta at cell centres
    this.Ht   = new Float64Array(NE);    // dH/dt, for the grid-relative transport

    this._r = new Float64Array(NP);
    this._d = new Float64Array(NP);
    this._q = new Float64Array(NP);
    this._z = new Float64Array(NP);
    this._div = new Float64Array(NP);
    this._gu = new Float64Array(NU);
    this._gv = new Float64Array(NV);
    this._gw = new Float64Array(NW);
    this._pdiag = new Float64Array(NP);

    this.cgIters = 0; this.cgResidual = 0;
    this.refreshMetric();
  }

  /* Index maps. theta is periodic, so k wraps; the wrap is done here once rather
     than at every use, where a missing modulo would read a neighbouring radius
     and look like a physical asymmetry. */
  kw(k){ const n = this.nth; return ((k % n) + n) % n; }
  iu(i, k, j){ return (i*this.nth + this.kw(k))*this.nz + j; }
  iv(i, k, j){ return (i*this.nth + this.kw(k))*this.nz + j; }
  iw(i, k, j){ return (i*this.nth + this.kw(k))*(this.nz + 1) + j; }
  ip(i, k, j){ return (i*this.nth + this.kw(k))*this.nz + j; }
  ie(i, k){ return i*this.nth + this.kw(k); }

  /* H and its derivatives from the current eta. Called whenever eta changes,
     because every operator below is written against this metric and a stale one
     would solve last step's geometry with this step's fields. */
  refreshMetric(){
    const nr = this.nr, nth = this.nth, h = this.h;
    const H = this.H, eta = this.eta;
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        H[e] = h + eta[e];
        if (!(H[e] > 0)) throw new Error(
          `the free surface has reached the floor: h + eta = ${H[e].toExponential(3)} `
          + `at r = ${this.rc[i].toExponential(3)} m, theta = `
          + `${(k*this.dth).toFixed(3)} rad, t = ${this.t.toExponential(3)} s. The `
          + `surface-following coordinate z = sigma*(h + eta) requires a positive `
          + `depth everywhere; a non-positive one means the layer has broken and `
          + `there is no single-valued surface to follow. Refusing rather than `
          + `continuing with an inverted cell, which would return a field.`);
      }
    /* r faces: linear in r between the neighbouring centres. At the axis the
       face area is zero so the value is never used in a flux, but it is used by
       the slope, where a one-sided value is right. At the rim the wall is at the
       last centre's own H under a free contact line, and at the pinned one the
       surface is undeformed there. */
    for (let k = 0; k < nth; k++){
      this.Hr[0*nth + k] = H[this.ie(0, k)];
      for (let i = 1; i < nr; i++){
        const a = this.drc[i-1], b = this.drc[i];
        this.Hr[i*nth + k] = (b*H[this.ie(i-1, k)] + a*H[this.ie(i, k)])/(a + b);
      }
      this.Hr[nr*nth + k] = this.contact === 'pinned' ? h : H[this.ie(nr-1, k)];
    }
    /* theta faces: uniform spacing, so the arithmetic mean. */
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        this.Hth[i*nth + k] = 0.5*(H[this.ie(i, k-1)] + H[this.ie(i, k)]);
    /* centred slopes, from the face values, so that a linear eta differentiates
       exactly and the slope is consistent with the fluxes that use it. */
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        this.Hdr[e] = (this.Hr[(i+1)*nth + this.kw(k)] - this.Hr[i*nth + this.kw(k)])
                      /this.drc[i];
        this.Hdth[e] = (this.Hth[i*nth + this.kw(k+1)] - this.Hth[i*nth + this.kw(k)])
                       /this.dth;
      }
    return this;
  }

  gravity(){ return this.g + this.accel*Math.cos(this.omegaD*this.t); }

  /* ---- the projection -------------------------------------------------- */

  /* Net flux out of each cell, in the transformed coordinates:
        d_r(r H u) + d_theta(H v) + d_sigma(r Omega)
     integrated over the cell. Unnormalised, as next door: the volume division
     belongs to the gradient, which must be this operator's exact transpose over
     those volumes. */
  divergence(u, v, om, out){
    const nr = this.nr, nth = this.nth, nz = this.nz, dth = this.dth;
    const rf = this.rf, rc = this.rc, drc = this.drc, dsc = this.dsc;
    for (let i = 0; i < nr; i++){
      const rci = rc[i], dr = drc[i];
      for (let k = 0; k < nth; k++){
        const kk = this.kw(k);
        const HrIn = this.Hr[i*nth + kk], HrOut = this.Hr[(i+1)*nth + kk];
        const HthIn = this.Hth[i*nth + kk], HthOut = this.Hth[i*nth + this.kw(k+1)];
        for (let j = 0; j < nz; j++){
          const radial = dth*dsc[j]*(
              rf[i+1]*HrOut*u[this.iu(i+1, k, j)]
            - rf[i]  *HrIn *u[this.iu(i,   k, j)]);
          const azim = dr*dsc[j]*(
              HthOut*v[this.iv(i, k+1, j)]
            - HthIn *v[this.iv(i, k,   j)]);
          const vert = rci*dr*dth*(
              om[this.iw(i, k, j+1)] - om[this.iw(i, k, j)]);
          out[this.ip(i, k, j)] = radial + azim + vert;
        }
      }
    }
    return out;
  }

  /* The exact transpose of the divergence, divided by minus each face's own
     control volume. Written as an accumulation in the same order the divergence
     reads its arguments, which is what makes the transpose exact rather than
     approximately so -- and exactness is the whole point: it is what makes
     divergence(gradient(.)) symmetric, and a symmetric negative definite
     operator is the only kind conjugate gradients is entitled to converge on. */
  gradient(q, gu, gv, gw){
    const nr = this.nr, nth = this.nth, nz = this.nz, dth = this.dth;
    const rf = this.rf, rc = this.rc, drc = this.drc, dsc = this.dsc,
          drf = this.drf, dsf = this.dsf;
    gu.fill(0); gv.fill(0); gw.fill(0);
    for (let i = 0; i < nr; i++){
      const rci = rc[i], dr = drc[i];
      for (let k = 0; k < nth; k++){
        const kk = this.kw(k);
        const HrIn = this.Hr[i*nth + kk], HrOut = this.Hr[(i+1)*nth + kk];
        const HthIn = this.Hth[i*nth + kk], HthOut = this.Hth[i*nth + this.kw(k+1)];
        for (let j = 0; j < nz; j++){
          const qc = q[this.ip(i, k, j)];
          gu[this.iu(i+1, k, j)] += qc*dth*dsc[j]*rf[i+1]*HrOut;
          gu[this.iu(i,   k, j)] -= qc*dth*dsc[j]*rf[i]  *HrIn;
          gv[this.iv(i, k+1, j)] += qc*dr*dsc[j]*HthOut;
          gv[this.iv(i, k,   j)] -= qc*dr*dsc[j]*HthIn;
          gw[this.iw(i, k, j+1)] += qc*rci*dr*dth;
          gw[this.iw(i, k, j  )] -= qc*rci*dr*dth;
        }
      }
    }
    /* Divide by minus the control volume of each face. u at the axis and the rim
       is prescribed, so its gradient there is not solved for and is zeroed; the
       same for Omega on the floor. */
    for (let k = 0; k < nth; k++)
      for (let j = 0; j < nz; j++){
        gu[this.iu(0, k, j)] = 0;
        gu[this.iu(nr, k, j)] = 0;
      }
    for (let i = 1; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const Hf = this.Hr[i*nth + this.kw(k)];
        for (let j = 0; j < nz; j++)
          gu[this.iu(i, k, j)] /= -(rf[i]*drf[i]*dth*Hf*dsc[j]);
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const Hf = this.Hth[i*nth + this.kw(k)];
        for (let j = 0; j < nz; j++)
          gv[this.iv(i, k, j)] /= -(rc[i]*drc[i]*dth*Hf*dsc[j]);
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        gw[this.iw(i, k, 0)] = 0;               // impermeable floor
        for (let j = 1; j <= nz; j++)
          gw[this.iw(i, k, j)] /= -(rc[i]*drc[i]*dth*dsf[j]);
      }
    return [gu, gv, gw];
  }

  applyL(q, out){
    this.gradient(q, this._gu, this._gv, this._gw);
    this.divergence(this._gu, this._gv, this._gw, out);
    return out;
  }

  /* The diagonal of divergence(gradient(.)), exactly, in eight applications.
     The stencil reaches (i+-1, k, j), (i, k+-1, j) and (i, k, j+-1) and no
     diagonal neighbour, so cells whose (i, k, j) parities all agree are never in
     each other's stencil: setting one parity class to one and reading the result
     at those same cells returns their diagonal entries with nothing else
     contributing. Eight classes, eight applications. Two would do in two
     dimensions and do next door; three indices need eight.

     Read from applyL rather than rederived, so the preconditioner cannot drift
     from the operator it preconditions. */
  pressureDiagonal(){
    const nr = this.nr, nth = this.nth, nz = this.nz;
    const d = this._pdiag, probe = new Float64Array(this.NP), q = new Float64Array(this.NP);
    for (let c = 0; c < 8; c++){
      const pi = c & 1, pk = (c >> 1) & 1, pj = (c >> 2) & 1;
      probe.fill(0);
      for (let i = 0; i < nr; i++) if ((i & 1) === pi)
        for (let k = 0; k < nth; k++) if ((k & 1) === pk)
          for (let j = 0; j < nz; j++) if ((j & 1) === pj)
            probe[this.ip(i, k, j)] = 1;
      this.applyL(probe, q);
      for (let i = 0; i < nr; i++) if ((i & 1) === pi)
        for (let k = 0; k < nth; k++) if ((k & 1) === pk)
          for (let j = 0; j < nz; j++) if ((j & 1) === pj)
            d[this.ip(i, k, j)] = q[this.ip(i, k, j)];
    }
    for (let c = 0; c < d.length; c++)
      if (!(d[c] < 0)) throw new Error(
        `the pressure operator has a diagonal entry of ${d[c]} at cell ${c} of `
        + `${d.length}. It is negative definite by construction, so a zero or `
        + `positive diagonal means a cell is decoupled from the pressure field `
        + `and no preconditioner can be formed from it.`);
    return d;
  }

  /* Conjugate gradients with the operator's own diagonal as preconditioner. The
     azimuthal count is even and periodic, which with the graded radius makes the
     diagonal span orders of magnitude; scaling it out is what keeps the
     iteration count near the square root of the unknown count rather than
     proportional to it. */
  solveP(rhs, tol, maxIt){
    const n = rhs.length, p = this.p, r = this._r, d = this._d, q = this._q,
          z = this._z, M = this._pdiag;
    this.applyL(p, q);
    let rr = 0;
    for (let i = 0; i < n; i++){ r[i] = rhs[i] - q[i]; rr += r[i]*r[i]; }
    const rr0 = rr;
    if (rr0 === 0){ this.cgIters = 0; this.cgResidual = 0; return 0; }
    let rz = 0;
    for (let i = 0; i < n; i++){ z[i] = r[i]/M[i]; d[i] = z[i]; rz += r[i]*z[i]; }
    let it = 0;
    for (; it < maxIt; it++){
      this.applyL(d, q);
      let dq = 0;
      for (let i = 0; i < n; i++) dq += d[i]*q[i];
      if (dq === 0) break;
      const alpha = rz/dq;
      let rr2 = 0;
      for (let i = 0; i < n; i++){ p[i] += alpha*d[i]; r[i] -= alpha*q[i]; rr2 += r[i]*r[i]; }
      if (Math.sqrt(rr2/rr0) < tol){ rr = rr2; it++; break; }
      let rz2 = 0;
      for (let i = 0; i < n; i++){ z[i] = r[i]/M[i]; rz2 += r[i]*z[i]; }
      const beta = rz2/rz; rz = rz2; rr = rr2;
      for (let i = 0; i < n; i++) d[i] = z[i] + beta*d[i];
    }
    this.cgIters = it; this.cgResidual = Math.sqrt(rr/rr0);
    if (!(this.cgResidual < tol)) throw new Error(
      `three-dimensional pressure solve did not converge: residual `
      + `${this.cgResidual.toExponential(3)} after ${it} iterations against a `
      + `tolerance of ${tol.toExponential(0)} on a ${this.nr}x${this.nth}x${this.nz} `
      + `grid at t = ${this.t.toExponential(3)} s.`);
    return this.cgResidual;
  }

  /* Largest absolute divergence, per unit volume, over the cells. Reported
     rather than assumed: it is the one number that says whether the projection
     did its job on this metric. */
  maxDivergence(){
    this.divergence(this.u, this.v, this.om, this._div);
    let worst = 0;
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++){
        const vol = this.rc[i]*this.drc[i]*this.dth*this.H[this.ie(i, k)];
        for (let j = 0; j < this.nz; j++){
          const d = Math.abs(this._div[this.ip(i, k, j)])/(vol*this.dsc[j]);
          if (d > worst) worst = d;
        }
      }
    return worst;
  }

  /* Physical vertical velocity, from Omega and the metric: the definition of
     Omega, rearranged. Needed by the viscous term and by anything reporting the
     field, and never stored, so it cannot go stale against Omega. */
  wAt(i, k, j){
    const e = this.ie(i, k);
    const s = this.sf[j];
    /* u and v at this sigma face of this cell: the radial average of the cell's
       two r faces, and the azimuthal average of its two theta faces, each taken
       at the sigma cells either side of the face. */
    const jm = j === 0 ? 0 : j - 1, jp = j === this.nz ? this.nz - 1 : j;
    const uu = 0.25*(this.u[this.iu(i, k, jm)] + this.u[this.iu(i+1, k, jm)]
                   + this.u[this.iu(i, k, jp)] + this.u[this.iu(i+1, k, jp)]);
    const vv = 0.25*(this.v[this.iv(i, k, jm)] + this.v[this.iv(i, k+1, jm)]
                   + this.v[this.iv(i, k, jp)] + this.v[this.iv(i, k+1, jp)]);
    return this.om[this.iw(i, k, j)]
         + s*(uu*this.Hdr[e] + (vv/this.rc[i])*this.Hdth[e]);
  }
}

const FARADAY_CELL3D = { CELL3D_G0, FaradayCell3D };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_CELL3D;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_CELL3D = FARADAY_CELL3D;
