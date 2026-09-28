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
    /* THE STATE IS THE PHYSICAL VELOCITY. (u, v, w) and eta are what the solver
       carries; Omega is formed from them at the start of each projection and read
       back afterwards. Carrying Omega as state instead would make the stored field
       depend on the metric, so the moment eta advanced the stored numbers would
       mean something different -- and the viscous and advective terms need
       physical w in any case. */
    this.u   = new Float64Array(NU);     // radial, at r faces
    this.v   = new Float64Array(NV);     // azimuthal, at theta faces
    this.w   = new Float64Array(NW);     // physical vertical, at sigma faces
    this.om  = new Float64Array(NW);     // Omega, derived for the projection
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

    /* H on a radially extended grid, so a face midpoint anywhere between the axis
       and the rim can be interpolated without a special case. Column 0 is cell 0
       continued across the axis to -rc[0] -- H is a scalar, so the antipodal value
       carries a plus sign -- and the last column is the rim, where a free contact
       line leaves the surface at the last cell's height and a pinned one holds it
       at h. Interpolating a pure function of position is also what keeps the
       viscous operator symmetric: both cells sharing a face compute that face's H
       identically. */
    this.rx = new Float64Array(nr + 2);
    this.Hx = new Float64Array((nr + 2)*nth);
    /* The centred slopes, extended the same way. Hat's own slopes are the slope of
       the bracket it interpolates in, which is centred at a FACE midpoint and
       one-sided at a node -- right for the r-face and theta-face cross terms,
       which are evaluated at faces, and first order for the sigma-face cross
       terms, which are evaluated at nodes. Interpolating the precomputed centred
       slopes instead keeps those second order. Getting this wrong cost the whole
       operator an order and a half: every family read 0.5 instead of 2. */
    this.Hxr = new Float64Array((nr + 2)*nth);
    this.Hxt = new Float64Array((nr + 2)*nth);
    this.rx[0] = -this.rc[0];
    for (let i = 0; i < nr; i++) this.rx[i+1] = this.rc[i];
    this.rx[nr+1] = this.R;

    this.cgIters = 0; this.cgResidual = 0;
    this._es = new Float64Array(6);        // one cell's four eta slopes, then two weights

    /* Node geometry, one descriptor per staggered family. Every family shares the
       same periodic uniform theta and the same flux algebra; they differ only in
       where their nodes sit in r and sigma, by how much their theta nodes are
       offset from the cell centres, and in what happens at each boundary. So the
       viscous operator is written once and instantiated four times, rather than
       copied four times with the indices changed -- which is where a sign or a
       spacing would go wrong unnoticed.

         rn  node coordinate
         rb  control-volume boundaries, rb[a] below node a, so one longer than rn
         lo, hi   the range of node indices that are unknowns; outside it the value
                  is prescribed by the wall
         axisSign  +1 for a scalar, -1 for a horizontal vector component */
    const rbU = new Float64Array(nr + 2);
    rbU[0] = 0;
    for (let i = 0; i < nr; i++) rbU[i+1] = this.rc[i];
    rbU[nr+1] = this.R;
    const sbW = new Float64Array(nz + 2);
    sbW[0] = 0;
    for (let j = 0; j < nz; j++) sbW[j+1] = this.sc[j];
    sbW[nz+1] = 1;

    this.FAM = {
      p: { idx: (i, k, j) => this.ip(i, k, j), axisSign: +1, thOff: 0.5,
           rn: this.rc, rb: this.rf, rLo: 0, rHi: nr - 1,
           sn: this.sc, sb: this.sf, sLo: 0, sHi: nz - 1 },
      u: { idx: (i, k, j) => this.iu(i, k, j), axisSign: -1, thOff: 0.5,
           rn: this.rf, rb: rbU, rLo: 1, rHi: nr - 1,
           sn: this.sc, sb: this.sf, sLo: 0, sHi: nz - 1 },
      v: { idx: (i, k, j) => this.iv(i, k, j), axisSign: -1, thOff: 0,
           rn: this.rc, rb: this.rf, rLo: 0, rHi: nr - 1,
           sn: this.sc, sb: this.sf, sLo: 0, sHi: nz - 1 },
      /* w's nodes run 0..nz, but its momentum equation is solved on 1..nz-1: the
         floor value is zero by no slip and the surface value is set by the
         free-surface conditions rather than by a momentum balance -- it has no
         half cell above it to take a flux through. Both remain readable
         neighbours. */
      w: { idx: (i, k, j) => this.iw(i, k, j), axisSign: +1, thOff: 0.5,
           rn: this.rc, rb: this.rf, rLo: 0, rHi: nr - 1,
           sn: this.sf, sb: sbW, sLo: 1, sHi: nz - 1 }
    };
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
    /* the extended columns. H is even under the axis reflection, so its radial
       derivative is odd and its azimuthal derivative even:
           H(-r, th)   =  H(r, th+pi)
           H_r(-r, th) = -H_r(r, th+pi)
           H_th(-r,th) =  H_th(r, th+pi)
       At the rim a free contact line has dH/dr = 0 by definition, and a pinned one
       has the surface undeformed along the wall, so dH/dtheta = 0 there instead. */
    const half = nth >> 1;
    const free = this.contact === 'free';
    for (let k = 0; k < nth; k++){
      const ka = this.kw(k + half);
      this.Hx[0*nth + k]  =  H[this.ie(0, ka)];
      this.Hxr[0*nth + k] = -this.Hdr[this.ie(0, ka)];
      this.Hxt[0*nth + k] =  this.Hdth[this.ie(0, ka)];
      for (let i = 0; i < nr; i++){
        this.Hx[(i+1)*nth + k]  = H[this.ie(i, k)];
        this.Hxr[(i+1)*nth + k] = this.Hdr[this.ie(i, k)];
        this.Hxt[(i+1)*nth + k] = this.Hdth[this.ie(i, k)];
      }
      this.Hx[(nr+1)*nth + k]  = free ? H[this.ie(nr-1, k)] : h;
      this.Hxr[(nr+1)*nth + k] = free ? 0
        : (h - H[this.ie(nr-1, k)])/this.drf[nr];
      this.Hxt[(nr+1)*nth + k] = free ? this.Hdth[this.ie(nr-1, k)] : 0;
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

  /* H and its two horizontal slopes at an arbitrary position, bilinear on the
     extended grid. theta is periodic and uniform with cell centres at
     (k + 1/2) dtheta; r is bracketed in the extended node list, so the axis and
     the rim need no special case at the call site. */
  Hat(r, th){
    const nth = this.nth, nr = this.nr, rx = this.rx, Hx = this.Hx, dth = this.dth;
    let a = 0;
    while (a < nr && rx[a+1] < r) a++;
    if (a > nr) a = nr;
    const r0 = rx[a], r1 = rx[a+1];
    const fr = (r - r0)/(r1 - r0);
    const tt = th/dth - 0.5;
    const kb = Math.floor(tt);
    const ft = tt - kb;
    const k0 = this.kw(kb), k1 = this.kw(kb + 1);
    const h00 = Hx[a*nth + k0], h01 = Hx[a*nth + k1];
    const h10 = Hx[(a+1)*nth + k0], h11 = Hx[(a+1)*nth + k1];
    const H = (1 - fr)*((1 - ft)*h00 + ft*h01) + fr*((1 - ft)*h10 + ft*h11);
    const Hr = (((1 - ft)*h10 + ft*h11) - ((1 - ft)*h00 + ft*h01))/(r1 - r0);
    const Hth = ((1 - fr)*(h01 - h00) + fr*(h11 - h10))/dth;
    return { H, Hr, Hth };
  }

  /* The centred slopes of H at an arbitrary position, bilinear on the extended
     slope grids. Distinct from Hat's slopes, which are the bracket's own and so
     are centred only at a face midpoint: these are what the sigma-face cross
     terms need, because those are evaluated at nodes. */
  Hslope(r, th){
    const nth = this.nth, nr = this.nr, rx = this.rx, dth = this.dth;
    let a = 0;
    while (a < nr && rx[a+1] < r) a++;
    if (a > nr) a = nr;
    const r0 = rx[a], r1 = rx[a+1];
    const fr = (r - r0)/(r1 - r0);
    const tt = th/dth - 0.5, kb = Math.floor(tt), ft = tt - kb;
    const k0 = this.kw(kb), k1 = this.kw(kb + 1);
    const lerp = A => (1 - fr)*((1 - ft)*A[a*nth + k0] + ft*A[a*nth + k1])
                    + fr*((1 - ft)*A[(a+1)*nth + k0] + ft*A[(a+1)*nth + k1]);
    return { Hr: lerp(this.Hxr), Hth: lerp(this.Hxt) };
  }

  /* ---- the axis ---------------------------------------------------------- */

  /* The radial velocity at the axis. u_r is not stored there as an independent
     value: a single-valued vector field requires u_r(0, theta) = -u_r(0, theta+pi),
     so the axis row is the antisymmetric part of an extrapolation from the two
     nodes outside it. Second order, and it satisfies the constraint by
     construction rather than by an assertion afterwards.

     Setting it to zero instead -- which is what a single-mode solver does, and why
     dns/faraday-disc.js refuses m = 1 -- would be exact for m = 0 and for every
     m >= 2, and wrong for exactly the one mode that is non-zero at the axis. */
  axisU(){
    const nr = this.nr, nth = this.nth, nz = this.nz, half = nth >> 1;
    const r1 = this.rf[1], r2 = this.rf[2];
    const ex = new Float64Array(nth);
    for (let j = 0; j < nz; j++){
      for (let k = 0; k < nth; k++){
        const a = this.u[this.iu(1, k, j)], b = this.u[this.iu(2, k, j)];
        ex[k] = a + (0 - r1)*(b - a)/(r2 - r1);     // linear extrapolation to r = 0
      }
      for (let k = 0; k < nth; k++)
        this.u[this.iu(0, k, j)] = 0.5*(ex[k] - ex[this.kw(k + half)]);
    }
    return this;
  }

  /* A field's value in one column, at an arbitrary physical height.
   *
   * This is the primitive the whole operator is built on. Anything that compares
   * values from two different columns must do it at a COMMON PHYSICAL HEIGHT:
   * under z = sigma H two columns' sigma levels are at different heights whenever
   * the surface is deformed, so comparing them at equal sigma carries an O(dH)
   * error that vanishes when flat and does not converge when not. That error is
   * what made the coupling terms read order 2.00 flat and -0.52 at eta/h = 0.4.
   *
   * Quadratic in sigma, on the stencil centred at level `lev` -- centred on the
   * LEVEL and not chosen by bracketing the target, so that two columns compared at
   * one height use the same node positions and their reconstruction errors cancel
   * instead of jumping as a target crosses a node.
   *
   * Radial index a < 0 is the antipodal column reflected through the axis, carrying
   * the family's own sign: plus for a scalar or the vertical component, minus for a
   * horizontal one. */
  colValueAtZ(f, fam, a, k, z, lev){
    const sn = fam.sn, nJ = sn.length, half = this.nth >> 1;
    let aa = a, kk = k, sign = 1;
    if (a < 0){ aa = -1 - a; kk = k + half; sign = fam.axisSign; }
    const th = (kk + fam.thOff)*this.dth;
    const H = this.Hat(fam.rn[aa], th).H;
    const ss = z/H;
    const at = b => f[fam.idx(aa, kk, b)];
    if (nJ === 1) return sign*at(0);
    if (nJ === 2){
      const t = (ss - sn[0])/(sn[1] - sn[0]);
      return sign*((1 - t)*at(0) + t*at(1));
    }
    let j = lev === undefined ? 1 : lev;
    if (j < 1) j = 1;
    if (j > nJ - 2) j = nJ - 2;
    const s0 = sn[j-1], s1 = sn[j], s2 = sn[j+1];
    const L0 = ((ss - s1)*(ss - s2))/((s0 - s1)*(s0 - s2));
    const L1 = ((ss - s0)*(ss - s2))/((s1 - s0)*(s1 - s2));
    const L2 = ((ss - s0)*(ss - s1))/((s2 - s0)*(s2 - s1));
    return sign*(at(j-1)*L0 + at(j)*L1 + at(j+1)*L2);
  }

  /* ---- the viscous operator -------------------------------------------- */

  /* The metric Laplacian at any of the four node families.
   *
   * TANGENTIAL DERIVATIVES ARE TAKEN AT A COMMON PHYSICAL HEIGHT. This is the
   * whole design and it is not an optimisation; the obvious discretisation does
   * not converge. Under z = sigma H every physical derivative is a difference of
   * two terms,
   *
   *     df/dr|_z = d_r f - (sigma H_r / H) d_sigma f
   *
   * and for a field that depends on z alone those two terms are individually O(1)
   * and cancel EXACTLY. Discretely they cancel only to O(dtheta^2); the Laplacian
   * then divides a difference of face fluxes by dtheta, leaving O(dtheta); and the
   * 1/r^2 factor near the axis amplifies that by 1/dr^2, so refining the grid
   * makes it worse. Measured with f = sin(kz) over a surface varying only in
   * theta, eta/h = 0.3: family p read 1.77e-1 then 1.72e-1, order 0.04, and family
   * v read 8.63e-1 then 3.42e+0, order -1.99. This is the pressure-gradient error
   * known from terrain-following ocean and atmosphere models.
   *
   * The cure is to remove the subtraction rather than to compute it more
   * carefully. Each column is reconstructed as a function of physical height and
   * the two reconstructions are differenced at the SAME height:
   *
   *     df/dr|_z  ~  [ f_{a+1}(z) - f_a(z) ] / (r_{a+1} - r_a)
   *
   * with f_a(z) obtained by interpolating column a in sigma at sigma = z/H_a. For
   * f = f(z) both reconstructions return f(z) and the difference is exactly zero,
   * whatever the surface does. Quadratic in sigma, so the reconstruction error is
   * smooth across node boundaries: with linear interpolation the error is
   * piecewise and the two columns' errors fail to cancel when their targets
   * straddle a node.
   *
   * The families differ only in where their nodes sit in r and sigma, their theta
   * offset, and their boundaries, so this is written once and instantiated four
   * times. `bc(kind, r, th, sigma)` gives the field's value on a boundary face,
   * kind being 'rim', 'floor' or 'surface'; omitted, boundary faces carry zero
   * flux. The axis needs no entry: at r = 0 the face area is exactly zero, and
   * inward stencils use the antipodal column with the family's reflection sign. */
  famLaplacian(f, out, fam, bc){
    const nth = this.nth, dth = this.dth;
    const rn = fam.rn, rb = fam.rb, sn = fam.sn, sb = fam.sb;
    const nI = rn.length, nJ = sn.length;
    const idx = fam.idx, sgn = fam.axisSign, thOff = fam.thOff;
    const half = nth >> 1;
    const thOf = k => (k + thOff)*dth;
    const at = (a, k, b) => f[idx(a, k, b)];

    /* A column, resolved through the axis, for its radial coordinate only; the
       value comes from colValueAtZ, which applies the same reflection. */
    const colR = a => a >= 0 ? rn[a] : -rn[-1 - a];

    const atZ = (a, k, z, lev) => this.colValueAtZ(f, fam, a, k, z, lev);

    /* df/dr|_z and df/dtheta|_z, each a difference of reconstructions at ONE
       height. Beyond the node list the wall value comes from bc, which already
       delivers a value at a requested sigma and therefore at a requested height. */
    /* At the rim the face IS the wall, so a two-point difference between the wall
       and the nearest column is centred at their midpoint and only first order AT
       the face -- and a flux error of that order does not converge. It cost the w
       family its last radial row, 2.9e-3 growing to 3.7e-3 while every interior row
       ran at second order; the same three-point quadratic the sigma boundaries use
       fixes it. u never showed it, because its node list reaches the wall and the
       wall is an ordinary neighbour there rather than a boundary. */
    const dPhysR = (aL, aR, k, z, lev) => {
      const th = thOf(k);
      const HR = this.Hat(this.R, th).H;
      const outL = aL > nI - 1, outR = aR > nI - 1;
      if (outR){
        if (!bc) return 0;
        const fo = bc('rim', this.R, th, z/HR);
        const fc = atZ(aL, k, z, lev);
        const d1 = this.R - colR(aL);
        if (aL - 1 < 0) return (fo - fc)/d1;
        const ff = atZ(aL - 1, k, z, lev);
        const d2 = this.R - colR(aL - 1);
        return fo*(d1 + d2)/(d1*d2) - fc*d2/(d1*(d2 - d1)) + ff*d1/(d2*(d2 - d1));
      }
      const vL = outL ? (bc ? bc('rim', this.R, th, z/HR) : 0) : atZ(aL, k, z, lev);
      const vR = atZ(aR, k, z, lev);
      const rL = outL ? this.R : colR(aL), rR = colR(aR);
      return (vR - vL)/(rR - rL);
    };
    const dPhysTh = (a, kL, kR, z, lev) =>
      (atZ(a, kR, z, lev) - atZ(a, kL, z, lev))/((kR - kL)*dth);

    /* d f / d sigma at a node, centred where both neighbours exist. This one needs
       no common-height treatment: it is already a derivative along the coordinate,
       with nothing to cancel against. */
    const dfds = (a, k, b) => {
      if (nJ < 2) return 0;
      if (b === 0) return (at(a, k, 1) - at(a, k, 0))/(sn[1] - sn[0]);
      if (b === nJ - 1) return (at(a, k, nJ-1) - at(a, k, nJ-2))/(sn[nJ-1] - sn[nJ-2]);
      return (at(a, k, b+1) - at(a, k, b-1))/(sn[b+1] - sn[b-1]);
    };
    /* and at a boundary face, second order, from the three-point quadratic --
       a plain one-sided difference is second order at the midpoint between face
       and node and only first order AT the face, and a flux error of that order
       does not converge at all. dns/faraday-disc.js does the same in wzSurface. */
    const faceDeriv = (fo, fc, ff, d1, d2) =>
      fo*(d1 + d2)/(d1*d2) - fc*d2/(d1*(d2 - d1)) + ff*d1/(d2*(d2 - d1));

    for (let a = fam.rLo; a <= fam.rHi; a++){
      const dra = rb[a+1] - rb[a];
      for (let k = 0; k < nth; k++){
        const th = thOf(k);
        const mid = this.Hat(rn[a], th);
        const midS = this.Hslope(rn[a], th);
        for (let b = fam.sLo; b <= fam.sHi; b++){
          const dsb = sb[b+1] - sb[b];
          const fc = at(a, k, b);
          let flux = 0;

          /* ---- the two r faces: normal r-hat, so the flux is df/dr|_z ---- */
          for (const side of [-1, +1]){
            const rface = side < 0 ? rb[a] : rb[a+1];
            if (rface === 0) continue;                  // the axis: zero area
            const g = this.Hat(rface, th);
            if (!bc && (side < 0 ? a - 1 : a + 1) > nI - 1) continue;
            const z = sn[b]*g.H;
            const d = side < 0 ? dPhysR(a - 1, a, k, z, b) : dPhysR(a, a + 1, k, z, b);
            flux += side*rface*dth*g.H*dsb*d;
          }

          /* ---- the two theta faces: normal theta-hat ---- */
          for (const side of [-1, +1]){
            const thf = th + side*0.5*dth;
            const g = this.Hat(rn[a], thf);
            const z = sn[b]*g.H;
            const d = side < 0 ? dPhysTh(a, k - 1, k, z, b) : dPhysTh(a, k, k + 1, z, b);
            flux += side*dra*g.H*dsb*d/rn[a];
          }

          /* ---- the two sigma faces: the curved sheets z = sigma H, whose normal
                  is proportional to (-sigma H_r, -sigma H_theta / r, 1) ---- */
          for (const side of [-1, +1]){
            const sface = side < 0 ? sb[b] : sb[b+1];
            const proj = rn[a]*dra*dth;
            const bn = side < 0 ? b - 1 : b + 1;
            let dsg;
            if (bn < 0 || bn > nJ - 1){
              if (!bc) continue;
              const fo = bc(side < 0 ? 'floor' : 'surface', rn[a], th, sface);
              if (side < 0){
                const d1 = sn[b] - sface;
                dsg = (b + 1 <= nJ - 1)
                  ? -faceDeriv(fo, fc, at(a, k, b+1), d1, sn[b+1] - sface)
                  : (fc - fo)/d1;
              } else {
                const d1 = sface - sn[b];
                dsg = (b - 1 >= 0)
                  ? faceDeriv(fo, fc, at(a, k, b-1), d1, sface - sn[b-1])
                  : (fo - fc)/d1;
              }
            } else {
              dsg = side < 0 ? (fc - at(a, k, bn))/(sn[b] - sn[bn])
                             : (at(a, k, bn) - fc)/(sn[bn] - sn[b]);
            }
            /* the two tangential gradients at this sheet, at its own height */
            const z = sface*mid.H;
            const gR = dPhysR(a - 1, a + 1, k, z, b);
            const gT = dPhysTh(a, k - 1, k + 1, z, b);
            flux += side*proj*( dsg/mid.H
                              - sface*midS.Hr*gR
                              - (sface*midS.Hth/(rn[a]*rn[a]))*gT );
          }

          out[idx(a, k, b)] = flux/(rn[a]*dra*dth*mid.H*dsb);
        }
      }
    }
    return out;
  }

  /* The viscous term for the velocity, as the vector Laplacian in cylindrical
     coordinates:

         (grad^2 u)_r     = grad^2 u_r - u_r/r^2 - (2/r^2) d u_th / d theta
         (grad^2 u)_theta = grad^2 u_th - u_th/r^2 + (2/r^2) d u_r  / d theta
         (grad^2 u)_z     = grad^2 w

     The scalar part is famLaplacian at each component's own nodes; the rest is the
     curvature of the coordinate system, algebraic and pointwise. The two coupling
     terms are not optional and not small: for a field that is uniform in Cartesian
     terms -- u_r = U cos(theta) g(z), u_th = -U sin(theta) g(z) -- the scalar
     Laplacian of each component carries a spurious -u/r^2, and it is exactly the
     coupling that cancels it to leave U g''(z). The gate uses that field, because
     it is the one where getting the coupling wrong cannot hide.

     `bcU`, `bcV`, `bcW` close the three scalar operators at the walls. */
  viscous(outU, outV, outW, bcU, bcV, bcW){
    const nr = this.nr, nth = this.nth, nz = this.nz, half = nth >> 1;
    this.famLaplacian(this.u, outU, this.FAM.u, bcU);
    this.famLaplacian(this.v, outV, this.FAM.v, bcV);
    this.famLaplacian(this.w, outW, this.FAM.w, bcW);

    /* d v / d theta at a u node, and d u / d theta at a v node.
     *
     * Both are taken at the target node's own PHYSICAL HEIGHT. u sits at
     * (rf, theta centre) and v at (rc, theta face), so on a deformed surface their
     * sigma levels are at different heights, and averaging across at equal sigma
     * carries an O(dH) error: it vanishes when flat and does not converge when
     * not. Measured with the equal-sigma average: order 2.00 at eta/h = 0 for both
     * components and -0.52 at eta/h = 0.4. Reconstructing each column at the
     * target's height removes it. */
    const FU = this.FAM.u, FV = this.FAM.v;
    for (let i = FU.rLo; i <= FU.rHi; i++){
      const r = this.rf[i], inv = 1/(r*r);
      for (let k = 0; k < nth; k++){
        const Hu = this.Hat(r, (k + 0.5)*this.dth).H;
        for (let j = 0; j < nz; j++){
          const c = this.iu(i, k, j);
          const z = this.sc[j]*Hu;
          /* v on this u cell's two theta faces, each the mean of the two radial
             columns either side, reconstructed at z */
          const vA = 0.5*(this.colValueAtZ(this.v, FV, i - 1, k, z, j)
                        + (i <= nr - 1 ? this.colValueAtZ(this.v, FV, i, k, z, j) : 0));
          const vB = 0.5*(this.colValueAtZ(this.v, FV, i - 1, k + 1, z, j)
                        + (i <= nr - 1 ? this.colValueAtZ(this.v, FV, i, k + 1, z, j) : 0));
          outU[c] += -inv*this.u[c] - 2*inv*(vB - vA)/this.dth;
        }
      }
    }
    for (let i = FV.rLo; i <= FV.rHi; i++){
      const r = this.rc[i], inv = 1/(r*r);
      for (let k = 0; k < nth; k++){
        const Hv = this.Hat(r, k*this.dth).H;
        for (let j = 0; j < nz; j++){
          const c = this.iv(i, k, j);
          const z = this.sc[j]*Hv;
          const uA = 0.5*(this.colValueAtZ(this.u, FU, i, k - 1, z, j)
                        + this.colValueAtZ(this.u, FU, i + 1, k - 1, z, j));
          const uB = 0.5*(this.colValueAtZ(this.u, FU, i, k, z, j)
                        + this.colValueAtZ(this.u, FU, i + 1, k, z, j));
          outV[c] += -inv*this.v[c] + 2*inv*(uB - uA)/this.dth;
        }
      }
    }
    return [outU, outV, outW];
  }

  /* ---- advection -------------------------------------------------------- */

  /* The volumetric face fluxes of a PRESSURE cell, in the transformed coordinates.
   * The radial and azimuthal ones are exactly the terms divergence() forms. The
   * vertical one comes in three pieces because the sigma sheets themselves move
   * when eta does:
   *
   *     fluxSabs    r Omega, the absolute transport, what continuity balances
   *     fluxSmesh   r sigma dH/dt, the sheet's own motion
   *     fluxS       their difference, the GRID-RELATIVE transport, which is what
   *                 carries momentum across a moving sheet
   *
   * NOTHING IS CLAMPED TO ZERO AT A BOUNDARY, and that is the point. Each of the
   * four boundary fluxes is already exactly zero for a physical reason:
   *
   *     r = 0     rf[0] is exactly zero, so the axis face has no area
   *     r = R     no penetration, so u at the rim is zero
   *     sigma = 0 no slip, so Omega on the floor is zero, and sigma kills the mesh
   *               term there in any case
   *     sigma = 1 the surface is material, so Omega there IS dH/dt and the two
   *               pieces are a floating-point difference of equals
   *
   * Writing the expressions out rather than clamping means a violated boundary
   * condition shows up as a divergence or an energy imbalance, instead of being
   * masked by a branch that answers zero whatever the state says. */
  fluxR(i, k, b){
    return this.dth*this.dsc[b]*this.rf[i]*this.Hr[i*this.nth + this.kw(k)]
         * this.u[this.iu(i, k, b)];
  }
  fluxTh(i, k, b){
    return this.drc[i]*this.dsc[b]*this.Hth[i*this.nth + this.kw(k)]
         * this.v[this.iv(i, k, b)];
  }
  fluxSabs(i, k, b){
    return this.rc[i]*this.drc[i]*this.dth*this.om[this.iw(i, k, b)];
  }
  fluxSmesh(i, k, b){
    return this.rc[i]*this.drc[i]*this.dth*this.sf[b]*this.Ht[this.ie(i, k)];
  }
  fluxS(i, k, b){ return this.fluxSabs(i, k, b) - this.fluxSmesh(i, k, b); }

  /* The transport part of the advective term, in conservative flux form, centred.
   *
   * NO UPWINDING. Upwinding adds numerical dissipation, which is indistinguishable
   * from viscosity in the answer and would fake the damping that sets the Faraday
   * threshold. Centred flux-form advection is exactly energy neutral instead, which
   * is an identity the gate checks rather than a tolerance it tolerates.
   *
   * THE FORM IS THE MOVING-MESH ONE, and it is not the same as minus the flux
   * divergence. On a mesh that moves, what the finite-volume balance conserves is
   * the momentum CONTENT of a cell, V u, not u:
   *
   *     d(V u)/dt + sum_faces F_rel phi = 0   ==>   V du/dt = -f - u dV/dt
   *
   * so the rate of change of the cell's own volume appears, and it must be taken as
   * the net of the SAME mesh fluxes that were subtracted to make F_rel -- the
   * discrete geometric conservation law. Drop the term and a field that is uniform
   * over a surface rising uniformly accelerates out of nothing: measured without it,
   * the energy residual on a divergence-free field sat at 2.5e-3 of the terms it
   * sums and would not move when the projection tolerance was tightened by five
   * decades, because it was not the projection's error at all.
   *
   * THE ADVECTED QUANTITY IS A PLAIN ARITHMETIC MEAN, not a common-height
   * reconstruction, and that is deliberate. The common-height rule exists because a
   * physical DERIVATIVE under z = sigma H is a difference of two O(1) terms that
   * must cancel; an average has no cancellation to protect, and the flux form in
   * these coordinates never forms such a difference. What the mean must do instead
   * is telescope: with the face value equal to the mean of its two neighbouring
   * nodes, each interior face contributes Q (u_R^2 - u_L^2)/2 and the sum regroups
   * onto the cells. Reconstructing at a common height breaks that and the scheme
   * stops conserving energy.
   *
   * THE TRANSPORT FLUXES ARE THE PRESSURE CELLS' OWN, AVERAGED ONTO THE MOMENTUM
   * CONTROL VOLUMES, and the averaging is what makes the net flux collapse:
   *
   *     net absolute flux out of a momentum cell
   *         = 1/2 (divergence of one neighbour + divergence of the other)
   *
   * exactly, for any field whatever. Building the momentum fluxes independently
   * would leave a residual behaving like a spurious source. The one-sided cases are
   * the same statement: the u cell at i = 1 and the w half cell at sigma = 1 each
   * straddle a single pressure cell and collapse to half that one cell's divergence.
   *
   * w IS SOLVED AT SIGMA = 1, on the half cell between sc[nz-1] and the surface.
   * Its horizontal faces carry half of the last pressure cell's fluxes -- half
   * because sc[nz-1] is exactly the midpoint of that cell -- and its top face
   * carries the grid-relative flux there, which the kinematic condition makes zero.
   * Leaving that node out would put a face with non-zero flux on the edge of the
   * energy sum, and the identity would hold only up to the work done through it. */
  advectTransport(outU, outV, outW){
    const nr = this.nr, nth = this.nth, nz = this.nz, dth = this.dth;
    const rf = this.rf, rc = this.rc, drc = this.drc, drf = this.drf;
    const dsc = this.dsc, dsf = this.dsf;
    const u = this.u, v = this.v, w = this.w, half = nth >> 1;

    const Qr = (i, k, b) => this.fluxR(i, k, b);
    const Qt = (i, k, b) => this.fluxTh(i, k, b);
    const Qs = (i, k, b) => this.fluxS(i, k, b);
    const Qm = (i, k, b) => this.fluxSmesh(i, k, b);

    outU.fill(0); outV.fill(0); outW.fill(0);

    /* ---- radial momentum, on the u control volumes ------------------------
       rc[i-1] .. rc[i] in r, one pressure cell in theta and in sigma. Its r faces
       sit at the two pressure centres either side, so each carries the mean of the
       fluxes through the two cell faces bracketing it; its theta and sigma faces
       each span half of each of the two cells it straddles, so each carries the
       mean of those two cells' fluxes there. Only the sigma faces move. */
    for (let i = 1; i < nr; i++){
      for (let k = 0; k < nth; k++){
        const Hf = this.Hr[i*nth + this.kw(k)];
        for (let b = 0; b < nz; b++){
          const c = this.iu(i, k, b);
          const V = rf[i]*drf[i]*dth*Hf*dsc[b];
          const QrIn  = 0.5*(Qr(i-1, k, b) + Qr(i, k, b));
          const QrOut = 0.5*(Qr(i, k, b)   + Qr(i+1, k, b));
          const QtLo  = 0.5*(Qt(i-1, k, b) + Qt(i, k, b));
          const QtHi  = 0.5*(Qt(i-1, k+1, b) + Qt(i, k+1, b));
          const QsLo  = 0.5*(Qs(i-1, k, b) + Qs(i, k, b));
          const QsHi  = 0.5*(Qs(i-1, k, b+1) + Qs(i, k, b+1));
          const QmLo  = 0.5*(Qm(i-1, k, b) + Qm(i, k, b));
          const QmHi  = 0.5*(Qm(i-1, k, b+1) + Qm(i, k, b+1));
          /* below the first sigma node the wall value is zero by no slip; above the
             last there is no u node, so the nearest value stands. Both faces carry
             exactly zero flux, so neither choice enters the answer -- they exist
             because a face value has to be a number. */
          const uDn = b === 0      ? 0 : u[this.iu(i, k, b-1)];
          const uUp = b === nz - 1 ? u[c] : u[this.iu(i, k, b+1)];
          const f = QrOut*0.5*(u[c] + u[this.iu(i+1, k, b)])
                  - QrIn *0.5*(u[this.iu(i-1, k, b)] + u[c])
                  + QtHi *0.5*(u[c] + u[this.iu(i, k+1, b)])
                  - QtLo *0.5*(u[this.iu(i, k-1, b)] + u[c])
                  + QsHi *0.5*(u[c] + uUp)
                  - QsLo *0.5*(uDn + u[c]);
          outU[c] = -(f + u[c]*(QmHi - QmLo))/V;
        }
      }
    }

    /* ---- azimuthal momentum, on the v control volumes --------------------- */
    for (let i = 0; i < nr; i++){
      for (let k = 0; k < nth; k++){
        const Hf = this.Hth[i*nth + this.kw(k)];
        for (let b = 0; b < nz; b++){
          const c = this.iv(i, k, b);
          const V = rc[i]*drc[i]*dth*Hf*dsc[b];
          const QrIn  = 0.5*(Qr(i, k-1, b)   + Qr(i, k, b));
          const QrOut = 0.5*(Qr(i+1, k-1, b) + Qr(i+1, k, b));
          const QtLo  = 0.5*(Qt(i, k-1, b) + Qt(i, k, b));
          const QtHi  = 0.5*(Qt(i, k, b)   + Qt(i, k+1, b));
          const QsLo  = 0.5*(Qs(i, k-1, b) + Qs(i, k, b));
          const QsHi  = 0.5*(Qs(i, k-1, b+1) + Qs(i, k, b+1));
          const QmLo  = 0.5*(Qm(i, k-1, b) + Qm(i, k, b));
          const QmHi  = 0.5*(Qm(i, k-1, b+1) + Qm(i, k, b+1));
          /* inward of the first column the continuation is the antipodal one with
             u_theta's own sign; outward of the last the sidewall holds v at zero.
             Both faces carry exactly zero flux. */
          const vIn  = i === 0      ? -v[this.iv(0, k + half, b)] : v[this.iv(i-1, k, b)];
          const vOut = i === nr - 1 ? 0 : v[this.iv(i+1, k, b)];
          const vDn  = b === 0      ? 0 : v[this.iv(i, k, b-1)];
          const vUp  = b === nz - 1 ? v[c] : v[this.iv(i, k, b+1)];
          const f = QrOut*0.5*(v[c] + vOut)
                  - QrIn *0.5*(vIn + v[c])
                  + QtHi *0.5*(v[c] + v[this.iv(i, k+1, b)])
                  - QtLo *0.5*(v[this.iv(i, k-1, b)] + v[c])
                  + QsHi *0.5*(v[c] + vUp)
                  - QsLo *0.5*(vDn + v[c]);
          outV[c] = -(f + v[c]*(QmHi - QmLo))/V;
        }
      }
    }

    /* ---- vertical momentum, on the w control volumes ---------------------
       sc[b-1] .. sc[b] in sigma for b below nz, and sc[nz-1] .. 1 for the surface
       half cell. There is no pressure cell at b = nz, so that half cell's
       horizontal faces carry half of cell nz-1's fluxes and nothing else; its top
       face carries the full flux at sigma = 1. */
    for (let i = 0; i < nr; i++){
      for (let k = 0; k < nth; k++){
        const Hc = this.H[this.ie(i, k)];
        for (let b = 1; b <= nz; b++){
          const c = this.iw(i, k, b);
          const V = rc[i]*drc[i]*dth*Hc*dsf[b];
          const top = b === nz;
          const QrIn  = 0.5*(Qr(i, k, b-1)   + (top ? 0 : Qr(i, k, b)));
          const QrOut = 0.5*(Qr(i+1, k, b-1) + (top ? 0 : Qr(i+1, k, b)));
          const QtLo  = 0.5*(Qt(i, k, b-1)   + (top ? 0 : Qt(i, k, b)));
          const QtHi  = 0.5*(Qt(i, k+1, b-1) + (top ? 0 : Qt(i, k+1, b)));
          const QsLo  = 0.5*(Qs(i, k, b-1) + Qs(i, k, b));
          const QsHi  = top ? Qs(i, k, nz) : 0.5*(Qs(i, k, b) + Qs(i, k, b+1));
          const QmLo  = 0.5*(Qm(i, k, b-1) + Qm(i, k, b));
          const QmHi  = top ? Qm(i, k, nz) : 0.5*(Qm(i, k, b) + Qm(i, k, b+1));
          /* inward of the first column the continuation is the antipodal one, and w
             is a scalar under that reflection; outward of the last the sidewall
             holds w at zero. Above the surface node there is nothing, so the node
             itself stands -- against a flux the kinematic condition makes zero. */
          const wIn  = i === 0      ? w[this.iw(0, k + half, b)] : w[this.iw(i-1, k, b)];
          const wOut = i === nr - 1 ? 0 : w[this.iw(i+1, k, b)];
          const wUp  = top ? w[c] : w[this.iw(i, k, b+1)];
          const f = QrOut*0.5*(w[c] + wOut)
                  - QrIn *0.5*(wIn + w[c])
                  + QtHi *0.5*(w[c] + w[this.iw(i, k+1, b)])
                  - QtLo *0.5*(w[this.iw(i, k-1, b)] + w[c])
                  + QsHi *0.5*(w[c] + wUp)
                  - QsLo *0.5*(w[this.iw(i, k, b-1)] + w[c]);
          outW[c] = -(f + w[c]*(QmHi - QmLo))/V;
        }
      }
    }
    return [outU, outV, outW];
  }

  /* The two terms the rotating basis contributes to the cylindrical momentum
   * equations, +u_theta^2/r in the radial one and -u_r u_theta/r in the azimuthal.
   *
   * THEY CANCEL EXACTLY IN THE ENERGY, and getting that cancellation is the whole
   * design of this method. In the continuum the cancellation is POINTWISE:
   * u_r (u_theta^2/r) - u_theta (u_r u_theta / r) = 0 at every point, so the basis
   * contributes no energy. On a staggered grid u_r and u_theta live in different
   * places, and interpolating each to the other's node -- the obvious thing --
   * leaves (mean v)^2 on one side against v^2 on the other, which do not cancel:
   * the pair then acts as an energy source of the scheme's own truncation order.
   *
   * What does cancel is to form the product at ONE place, the pressure cell centre,
   * where both components have a single value, and to distribute it to the two
   * momentum equations as exact adjoints of those cell-centre averages. With
   *
   *     uc = (u(i) + u(i+1))/2,  vc = (v(k) + v(k+1))/2,  Vc the cell volume
   *
   * the radial term at node i takes half of each neighbouring cell's Vc vc^2/rc and
   * the azimuthal term at node k takes half of each neighbouring cell's
   * Vc (uc/rc) vc. Summing V u C over the radial nodes then gives
   * +sum_cells Vc uc vc^2/rc and over the azimuthal nodes -sum_cells Vc uc vc^2/rc,
   * the same number twice with opposite signs, and the two cancel in floating point
   * to the last bit rather than to an order.
   *
   * Two terms survive at the ends of the radial direction, where a cell has only
   * one momentum node inside the sum: 1/2 Vc(0) Gv(0) u(0) at the axis and
   * 1/2 Vc(nr-1) Gv(nr-1) u(nr) at the rim. The rim one is identically zero, because
   * no penetration holds u(nr) at zero. The axis one is not zero, and it is not
   * hidden either: u_r at r = 0 is not an unknown but the antisymmetric extrapolation
   * of the two columns outside it, so its control volume has exactly zero measure and
   * the faces it shares with node 1 have no partner inside the energy sum. What that
   * costs is derived in closed form and asserted term for term in the gate, on a
   * uniform Cartesian field, which advects itself to exactly nothing: the residual is
   * second order in dtheta at a fixed radius and first order at the first cell, where
   * the radius is itself a spacing, and no interpolation removes it. */
  advectCurvature(outU, outV){
    const nr = this.nr, nth = this.nth, nz = this.nz, dth = this.dth;
    const rf = this.rf, rc = this.rc, drc = this.drc, drf = this.drf, dsc = this.dsc;
    const u = this.u, v = this.v, H = this.H;

    const Vcell = (i, k, b) => rc[i]*drc[i]*dth*H[this.ie(i, k)]*dsc[b];
    const ucell = (i, k, b) => 0.5*(u[this.iu(i, k, b)] + u[this.iu(i+1, k, b)]);
    const vcell = (i, k, b) => 0.5*(v[this.iv(i, k, b)] + v[this.iv(i, k+1, b)]);
    /* u_theta^2/r and (u_r/r) u_theta at a cell centre */
    const Gv = (i, k, b) => { const t = vcell(i, k, b); return t*t/rc[i]; };
    const Pu = (i, k, b) => ucell(i, k, b)*vcell(i, k, b)/rc[i];

    for (let i = 1; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const Hf = this.Hr[i*nth + this.kw(k)];
        for (let b = 0; b < nz; b++)
          outU[this.iu(i, k, b)] +=
            0.5*(Vcell(i-1, k, b)*Gv(i-1, k, b) + Vcell(i, k, b)*Gv(i, k, b))
            / (rf[i]*drf[i]*dth*Hf*dsc[b]);
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const Hf = this.Hth[i*nth + this.kw(k)];
        for (let b = 0; b < nz; b++)
          outV[this.iv(i, k, b)] -=
            0.5*(Vcell(i, k-1, b)*Pu(i, k-1, b) + Vcell(i, k, b)*Pu(i, k, b))
            / (rc[i]*drc[i]*dth*Hf*dsc[b]);
      }
    return [outU, outV];
  }

  advect(outU, outV, outW){
    this.advectTransport(outU, outV, outW);
    this.advectCurvature(outU, outV);
    return [outU, outV, outW];
  }

  /* ---- the free surface -------------------------------------------------- */

  /* The four face slopes of eta around one cell, written ONCE because the curvature is
   * the exact adjoint of exactly these and the two must not be able to drift apart.
   * Filled into a caller-supplied four-element array: [inner r, outer r, lower theta,
   * upper theta].
   *
   * The axis needs no condition. Its face area rf[0] is exactly zero, so the area
   * functional weights that slope by nothing and the curvature's coefficient for it is
   * exactly zero -- which is the only correct treatment, because eta_r at r = 0 is
   * non-zero for every azimuthal mode but m = 0 and any single value assigned there
   * would be wrong for some mode.
   *
   * The rim needs one, and which one is the contact condition: a free contact line
   * leaves the surface with zero radial slope at the wall, a pinned one holds eta at
   * zero there, so the pinned case takes a two-point difference against the wall value.
   *
   * THAT IS DELIBERATELY NOT THE THREE-POINT QUADRATIC the rest of this solver uses at a
   * boundary face, and the reason is specific to a functional whose derivative is then
   * taken. A three-point rim slope makes the last cell's AREA depend on the column two
   * cells in, so the functional stops being a sum of local cell areas, and its
   * derivative inherits that: the extra term lands on cell nr-2, where it is not part of
   * that cell's divergence. Measured: with the three-point slope the last two rows read
   * 35 and 8.1 relative and got WORSE under refinement, orders -1.20 and -0.71, while
   * the two-point slope leaves every row but the rim second order and the rim row first.
   *
   * THREE WAYS OF MAKING THE RIM SLOPE SECOND ORDER WERE TRIED AND ALL THREE MADE THE
   * CURVATURE WORSE, because within this construction the weights and the slope are not
   * free to choose: it is the FACE-AREA weighting that makes rc*drc/(rf[i] + rf[i+1])
   * collapse to drc/2 and hence makes the derivative a divergence at all. Measured at the
   * rim row, relative, on grids of 16, 32 and 64 radial cells:
   *
   *     three-point quadratic through the wall         35, 80   (diverging, -1.2)
   *     weights 1/3 and 2/3, centring the estimate     38, 84, 176
   *     slope extrapolated to R as (4 sWall - sIn)/3   39, 84, 176
   *     the plain local two-point difference          0.013, 0.016, 0.017
   *
   * so the plain difference is not a shortcut, it is the only one of the four that leaves
   * a small error rather than a large one.
   *
   * First order at the rim row is not a consequence of the two-point slope; it is a
   * consequence of the functional being local, and it cannot be removed without giving
   * that up. A free contact line's rim slope is EXACTLY right -- it is zero, which is the
   * condition itself -- and its rim row is first order all the same, because what is
   * one-sided there is the face's own coefficient 1/sqrt(1 + |grad eta|^2), which has
   * only the cell inside it to come from. The trade is deliberate: locality buys the
   * exact identity kappa = -(1/vol) dA/d eta, and that identity is what makes the
   * exchange between kinetic and surface energy exact rather than approximate. One
   * annulus of width h carrying a first-order force contributes at the scheme's own
   * order to anything integrated. */
  etaSlopes(i, k, out){
    const nr = this.nr, eta = this.eta, drf = this.drf, rf = this.rf;
    const e = this.ie(i, k);
    const pinnedRim = i === nr - 1 && this.contact !== 'free';
    out[0] = i === 0 ? 0 : (eta[e] - eta[this.ie(i-1, k)])/drf[i];
    if (i < nr - 1) out[1] = (eta[this.ie(i+1, k)] - eta[e])/drf[i+1];
    else if (!pinnedRim) out[1] = 0;
    else out[1] = (0 - eta[e])/drf[nr];
    out[2] = (eta[e] - eta[this.ie(i, k-1)])/this.dth;
    out[3] = (eta[this.ie(i, k+1)] - eta[e])/this.dth;
    /* The weights with which the two radial slopes make the CELL-CENTRE value the area
     * element needs. Ordinarily they are the faces' own areas, and that is second order
     * because the two faces bracket the centre symmetrically: weighting positions by
     * rf gives rc + O(drc^2/rc). It is also what makes the curvature's radial face
     * coefficient come out as the width-weighted mean of 1/sqrt(1 + |grad eta|^2), which
     * is what consistency asks for.
     *
     * The pinned rim is the exception, and the reason is where its slope LIVES. A
     * two-point difference against the wall value is the exact slope at the midpoint of
     * rc[nr-1] and R, not at R; the inner face's slope is at rf[nr-1]. Since
     * rc[nr-1] - rf[nr-1] and R - rc[nr-1] are both half the last cell's width, that
     * midpoint sits at rc + drc/4 and the inner face at rc - drc/2, so the weights that
     * centre the estimate on rc are exactly one third and two thirds, on any grid.
     * Weighting them by face area instead puts the estimate a quarter of a cell off
     * centre, which is first order, and it showed: rows nr-1 and nr-2 sat at 1.3% and
     * 0.7% and did not converge at all while every interior row ran at second order. */
    const w = rf[i] + rf[i+1]; out[4] = rf[i]/w; out[5] = rf[i+1]/w;
    return out;
  }

  /* 1 + |grad eta|^2 at a cell centre. The radial pair is weighted by the FACE AREAS,
     which is what makes the axis face drop out of its own accord; the azimuthal faces
     have equal area, so they are weighted equally. */
  surfaceMetric(i, k){
    const s = this.etaSlopes(i, k, this._es);
    const rc = this.rc[i];
    return 1 + s[4]*s[0]*s[0] + s[5]*s[1]*s[1]
             + 0.5*(s[2]*s[2] + s[3]*s[3])/(rc*rc);
  }

  /* The area of the free surface, discretely:
   *
   *     A = sum_cells rc drc dtheta sqrt(1 + |grad eta|^2)
   *
   * This exists because the curvature is defined as its derivative. */
  surfaceArea(){
    let A = 0;
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++)
        A += this.rc[i]*this.drc[i]*this.dth*Math.sqrt(this.surfaceMetric(i, k));
    return A;
  }

  /* The mean curvature of the free surface, div(grad eta / sqrt(1 + |grad eta|^2)).
   *
   * NOT the linearised Laplacian of eta. At the drives this cell runs |grad eta| is of
   * order one, where sqrt(1 + |grad eta|^2) is two, and the difference is the difference
   * between a capillary pressure that saturates the pattern and one that does not.
   *
   * IT IS DEFINED AS THE VARIATIONAL DERIVATIVE OF THE DISCRETE AREA, and that is the
   * design rather than a way of writing it down:
   *
   *     kappa_j = -(1/(rc drc dtheta)) dA/d eta_j
   *
   * Three things follow that a discretised formula would only approximate. The
   * expression IS the finite-volume divergence form, because differentiating a sum over
   * face slopes gathers exactly a difference of face fluxes -- worked out, the radial
   * face coefficient comes to the width-weighted mean of 1/sqrt(1 + |grad eta|^2) across
   * the face, which is what consistency asks for, and the area weighting of the slopes
   * is what makes it come out that way. It is second order, because the area is. And the
   * work the capillary term does is exactly -gamma dA/dt, so the exchange between
   * kinetic energy and surface energy is an identity in floating point rather than a
   * tolerance -- which is what will let S7 assert that the whole step conserves energy
   * with viscosity and the drive off.
   *
   * Written as a scatter, in the same accumulate-then-divide shape `gradient` uses,
   * because that is what makes it the exact adjoint of the slopes rather than
   * approximately so. */
  curvature(out){
    const nr = this.nr, nth = this.nth, dth = this.dth;
    const rf = this.rf, rc = this.rc, drc = this.drc, drf = this.drf;
    const free = this.contact === 'free';
    const s = this._es;
    out.fill(0);
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        this.etaSlopes(i, k, s);
        const rr = rc[i]*rc[i];
        const X = 1 + s[4]*s[0]*s[0] + s[5]*s[1]*s[1]
                    + 0.5*(s[2]*s[2] + s[3]*s[3])/rr;
        const W = rc[i]*drc[i]*dth/(2*Math.sqrt(X));
        /* the radial pair. The axis face carries rf[0] = 0, so its weight is exactly zero
           and the neighbour that does not exist is never touched. */
        const cIn = W*2*s[4]*s[0], cOut = W*2*s[5]*s[1];
        if (i > 0){
          out[e] += cIn/drf[i];
          out[this.ie(i-1, k)] -= cIn/drf[i];
        }
        if (i < nr - 1){
          out[e] -= cOut/drf[i+1];
          out[this.ie(i+1, k)] += cOut/drf[i+1];
        } else if (!free){
          out[e] -= cOut/drf[nr];
        }
        /* the azimuthal pair, periodic, so both neighbours always exist */
        const cLo = W*s[2]/rr, cHi = W*s[3]/rr;
        out[e] += (cLo - cHi)/dth;
        out[this.ie(i, k-1)] -= cLo/dth;
        out[this.ie(i, k+1)] += cHi/dth;
      }
    for (let i = 0; i < nr; i++){
      const vol = rc[i]*drc[i]*dth;
      for (let k = 0; k < nth; k++) out[this.ie(i, k)] /= -vol;
    }
    return out;
  }

  /* Omega from the physical velocity, and back. The definition is
         Omega = w - sigma (u dH/dr + (v/r) dH/dtheta)
     so the two directions differ only in the sign of the slope term. Both are
     written out rather than one calling the other, because each is a full pass and
     the caller always knows which way it is going.

     Floor and surface both come out right without a special case: at sigma = 0 the
     slope term vanishes and Omega is w, which is zero by no slip; at sigma = 1 the
     slope term is exactly the horizontal advection of the surface, so Omega there
     is the kinematic condition, dEta/dt. */
  omegaFromW(){
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++)
        for (let j = 0; j <= this.nz; j++){
          const c = this.iw(i, k, j);
          this.om[c] = this.w[c] - this.slopeTerm(i, k, j);
        }
    return this;
  }
  wFromOmega(){
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++)
        for (let j = 0; j <= this.nz; j++){
          const c = this.iw(i, k, j);
          this.w[c] = this.om[c] + this.slopeTerm(i, k, j);
        }
    return this;
  }
  /* sigma (u dH/dr + (v/r) dH/dtheta) at a sigma face, with u and v averaged from
     the four faces of the cell that meet there. */
  slopeTerm(i, k, j){
    const e = this.ie(i, k), s = this.sf[j];
    if (s === 0) return 0;
    const jm = j === 0 ? 0 : j - 1, jp = j === this.nz ? this.nz - 1 : j;
    const uu = 0.25*(this.u[this.iu(i, k, jm)] + this.u[this.iu(i+1, k, jm)]
                   + this.u[this.iu(i, k, jp)] + this.u[this.iu(i+1, k, jp)]);
    const vv = 0.25*(this.v[this.iv(i, k, jm)] + this.v[this.iv(i, k+1, jm)]
                   + this.v[this.iv(i, k, jp)] + this.v[this.iv(i, k+1, jp)]);
    return s*(uu*this.Hdr[e] + (vv/this.rc[i])*this.Hdth[e]);
  }

}

const FARADAY_CELL3D = { CELL3D_G0, FaradayCell3D };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_CELL3D;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_CELL3D = FARADAY_CELL3D;
