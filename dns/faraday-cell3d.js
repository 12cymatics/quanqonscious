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

/* The derivative at x of the polynomial through the first n of (xs, ys).
   Written as the sum over j of ys[j] times the derivative of the j-th Lagrange
   basis polynomial, and that derivative as the sum over i != j of the product
   over m != i, j of (x - xs[m]) -- which has no removable singularity, so x may
   be a node as well as a point between nodes.

   WHY A CUBIC IS NEEDED, not merely convenient. A finite-volume row's Laplacian
   is a difference of two face fluxes divided by the row's own thickness. In the
   interior the two fluxes carry the same truncation error to leading order and it
   cancels, so second-order fluxes leave a second-order operator. At a boundary row
   one of the two faces is the boundary, where the stencil is one-sided or the flux
   is prescribed outright, so there is nothing for the interior face's error to
   cancel against and it is divided by the row's thickness undiminished -- which
   costs an order. With the two-point difference the surviving term is the offset
   between the face and the midpoint of the two nodes differenced there,
   (ds_neighbour - ds_row)/4 times f_ss, and on a graded grid that is first order:
   measured on the sigma = 0 row, relative error 1.748e-1, 7.679e-2, 3.598e-2,
   1.741e-2 over nz = 16, 32, 64, 128, against the closed form above of 1.730e-1,
   7.623e-2, 3.575e-2, 1.731e-2 -- three digits, so the mechanism is not in doubt.
   Four points make each face flux third order, the division by the row thickness
   leaves second order, and no cancellation is relied on: the same rows then read
   6.98e-5, 6.38e-6, 9.32e-7, 1.76e-7. */
/* The four columns that straddle a theta node symmetrically with the node itself left out,
   as offsets in k -- on a uniform grid the classical fourth-order central difference. Module
   level, because it is a constant and the innermost loop must not allocate. */
const TH_NODE = new Int8Array([-2, -1, 1, 2]);

function polyDerivAt(xs, ys, n, x){
  let d = 0;
  for (let j = 0; j < n; j++){
    let den = 1;
    for (let m = 0; m < n; m++) if (m !== j) den *= xs[j] - xs[m];
    let num = 0;
    for (let i = 0; i < n; i++){
      if (i === j) continue;
      let prod = 1;
      for (let m = 0; m < n; m++) if (m !== j && m !== i) prod *= x - xs[m];
      num += prod;
    }
    d += ys[j]*num/den;
  }
  return d;
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
    /* The same chain again for d eta/d t, because a renderer needs the surface's RATE and its
       gradient, not only its height. `Ht` is d eta/d t exactly -- step() sets it from the
       corrected Omega at sigma = 1, which IS the kinematic condition -- so what is missing is
       only the face, slope and extended-column machinery H already has. It is built alongside
       rather than by generalising H's, because H's is bit-exact load-bearing for every
       measured order in this file and the two differ anyway at the pinned rim: eta is zero on
       the wall for all time, so d eta/d t is zero there where H is h. */
    this.Tr   = new Float64Array((nr + 1)*nth);
    this.Tth  = new Float64Array(nr*nth);
    this.Tdr  = new Float64Array(NE);
    this.Tdth = new Float64Array(NE);
    /* AND THE SAME CHAIN FOR ETA'S OWN SLOPES, which is not the same thing as H's even though
       eta_r is H_r in exact arithmetic. H's face values are formed as
       (h + eta_lo) + f*((h + eta_hi) - (h + eta_lo)), and that inner difference loses the low
       bits of eta_hi - eta_lo: with h = 3e-3 m and eta differing by 1.7e-6 across a cell, the
       ulp of h + eta is 4.3e-19 and the difference carries 2.5e-13 of relative error. The
       solver does not care -- Hdr and Hdth feed flux terms where 2.5e-13 is far below the
       discretisation -- but the RENDERER does, because its transport forms
       Re grad Im - Im grad Re, a difference of two products that cancels exactly for a
       standing wave. Measured: with H's slopes that cancellation left 2.046e-12 of the terms
       instead of round-off, which is a drift the page would have drawn under a pattern going
       nowhere. Differencing eta directly is the fix, and the two chains are kept separate so
       that H's -- bit-exact load-bearing for every measured order in this file -- is untouched. */
    this.Er   = new Float64Array((nr + 1)*nth);
    this.Eth  = new Float64Array(nr*nth);
    this.Edr  = new Float64Array(NE);
    this.Edth = new Float64Array(NE);

    this._r = new Float64Array(NP);
    this._d = new Float64Array(NP);
    this._q = new Float64Array(NP);
    this._z = new Float64Array(NP);
    this._div = new Float64Array(NP);
    this._gu = new Float64Array(NU);
    this._gv = new Float64Array(NV);
    this._gw = new Float64Array(NW);
    this._gom = new Float64Array(NW);
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
    /* eta on the same extended nodes, so a renderer can have it without subtracting h */
    this.Ex = new Float64Array((nr + 2)*nth);
    /* and d eta/d t with its slopes, on those same nodes */
    this.Tx  = new Float64Array((nr + 2)*nth);
    this.Txr = new Float64Array((nr + 2)*nth);
    this.Txt = new Float64Array((nr + 2)*nth);
    this.Exr = new Float64Array((nr + 2)*nth);
    this.Ext = new Float64Array((nr + 2)*nth);
    this.rx[0] = -this.rc[0];
    for (let i = 0; i < nr; i++) this.rx[i+1] = this.rc[i];
    this.rx[nr+1] = this.R;

    this.cgIters = 0; this.cgResidual = 0;
    this._es = new Float64Array(6);        // one cell's four eta slopes, then two weights
    this._st = new Float64Array(6);        // one surface point's rate-of-strain tensor
    this._sg = new Float64Array(9);        // and its nine covariant derivatives
    this._sf3 = new Float64Array(3);       // the three surface Laplacian fluxes
    this._sx = new Float64Array(4);        // one sigma-face stencil's abscissae
    this._sy = new Float64Array(4);        // and its values
    this._rx4 = new Float64Array(4);       // one radial stencil's abscissae
    this._ry4 = new Float64Array(4);       // and its values
    this._tx4 = new Float64Array(4);       // one azimuthal stencil's abscissae
    this._ty4 = new Float64Array(4);       // and its values
    this._hA = new Float64Array(3);        // (H, H_r, H_theta) at a cell's own centre
    this._hB = new Float64Array(3);        // and at one of its six faces
    this._kap = new Float64Array(NE);      // the surface's mean curvature
    this._ps  = new Float64Array(NE);      // and the pressure it carries
    this._lu = new Float64Array(NU); this._au = new Float64Array(NU);
    this._lv = new Float64Array(NV); this._av = new Float64Array(NV);
    this._lw = new Float64Array(NW); this._aw = new Float64Array(NW);

    /* No slip on the floor and on the sidewall, which is the physical condition and not a
       convenient zero: the wall does not move. They are written once here rather than at
       every call so that `step` cannot pass a different closure than the gates do. The free
       surface is NOT among them -- its condition is a traction, which `viscous` takes as a
       flux through `surfaceLapFluxes`, and famLaplacian never asks a `bc` for the surface
       while a flux is supplied. */
    this._bcU = () => 0; this._bcV = () => 0; this._bcW = () => 0;
    this._fsr = new Float64Array(NE);      // and those three, over the whole surface
    this._fst = new Float64Array(NE);
    this._fsz = new Float64Array(NE);

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
      p: { idx: (i, k, j) => this.ip(i, k, j), axisSign: +1, thOff: 0.5, stride: nz,
           rn: this.rc, rb: this.rf, rLo: 0, rHi: nr - 1,
           sn: this.sc, sb: this.sf, sLo: 0, sHi: nz - 1 },
      u: { idx: (i, k, j) => this.iu(i, k, j), axisSign: -1, thOff: 0.5, stride: nz,
           rn: this.rf, rb: rbU, rLo: 1, rHi: nr - 1,
           sn: this.sc, sb: this.sf, sLo: 0, sHi: nz - 1 },
      v: { idx: (i, k, j) => this.iv(i, k, j), axisSign: -1, thOff: 0, stride: nz,
           rn: this.rc, rb: this.rf, rLo: 0, rHi: nr - 1,
           sn: this.sc, sb: this.sf, sLo: 0, sHi: nz - 1 },
      /* w's nodes run 0..nz and its viscous term is solved on 1..nz, the surface node
         included -- S5 made that node a solved ADVECTION unknown and this closes its
         viscous side, over the same half control volume [sc[nz-1], 1] the advection uses,
         because it is that shared volume which makes the discrete energy identity exact.

         AT THAT NODE THE UNKNOWN IS THE HALF CELL'S AVERAGE, not the point value at
         sigma = 1, and the two differ by O(1/nz). A staggered face-centred unknown on the
         domain boundary BOUNDS its control volume instead of straddling it, so the flux
         balance is a statement about the average and that average's centroid sits
         ds_top/4 below the node. Measured, with the surface flux supplied analytically:
         against the exact average over the half cell 6.78e-5, 1.80e-5, 4.58e-6 over
         16/32/64, order 1.92 then 1.97; against the point value at sigma = 1, 4.63e-3 at
         order 1.21 then 1.11. No choice of volume closes that gap -- a node cannot be the
         centroid of a cell it bounds -- and the alternative is the one
         dns/faraday-disc.js takes in lapW, a pointwise one-sided second derivative, which
         is second order at the node and not conservative. Conservation is kept here
         instead: an advective term in flux form beside a pointwise viscous one would
         leave the energy identity neither exactly dissipative nor exactly conservative. */
      w: { idx: (i, k, j) => this.iw(i, k, j), axisSign: +1, thOff: 0.5, stride: nz + 1,
           rn: this.rc, rb: this.rf, rLo: 0, rHi: nr - 1,
           sn: this.sf, sb: sbW, sLo: 1, sHi: nz }
    };
    /* Each family's node radii sit in fixed brackets of the extended radial node list, so
       the linear search HatH does can be done once per family node here instead of on every
       reconstruction. colValueAtZ was 38.5 per cent of a step before this, and at the rim
       node that search walked the whole list. The bracket found is the same one, so what
       follows it is unchanged arithmetic. */
    for (const key of ['p', 'u', 'v', 'w']){
      const fam = this.FAM[key], rn = fam.rn, br = new Int32Array(rn.length);
      for (let a = 0; a < rn.length; a++){
        let b = 0;
        while (b < nr && this.rx[b+1] < rn[a]) b++;
        br[a] = b > nr ? nr : b;
      }
      fam.hBr = br;
    }
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
        const lo = H[this.ie(i-1, k)], hi = H[this.ie(i, k)];
        /* written as an increment from one end rather than as a weighted sum, so that two
           equal depths interpolate to EXACTLY that depth. The weighted form
           (b*lo + a*hi)/(a + b) does not: for lo = hi it rounds twice and lands within an
           ulp, and the centred slope below then differences two such values over drc, which
           amplifies that ulp by h/drc. A flat surface would carry slopes of 1e-13 instead of
           zero, and the free surface's flat limit -- where the radial flux must reduce to
           exactly minus dw/dr, the condition dns/faraday-disc.js imposes -- would hold only
           to 2.4e-15 rather than exactly. Measured, before the change. */
        this.Hr[i*nth + k] = lo + (a/(a + b))*(hi - lo);
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
    /* THE SAME EXTENSION OF ETA ITSELF, and not H minus h, because that subtraction has no
       low bits to give back. Here h is 3e-3 m and the elevations a renderer draws are 1e-4 m
       and smaller, so h + eta has an ulp of 4.3e-19 and (h + eta) - h loses everything below
       it. Measured, before this array existed: `etaAt` as `HatH(r, th) - this.h` missed the
       stored eta at a cell centre by 6.505e-19 where it should have been exact, the
       axisymmetric surface's elevation on the axis spread 4.337e-19 over theta where it
       should have been one number, and the free rim missed the last cell's own elevation by
       5.421e-19. All three are that one ulp, and all three are the same defect the capillary
       energy had in section 11: a small quantity reconstructed as the difference of two large
       ones. Interpolating eta on its own nodes makes a cell centre exact again, because the
       bilinear weights there are fr = 1 and ft = 0 and the value is the stored one. */
    for (let k = 0; k < nth; k++){
      const ka = this.kw(k + half);
      this.Ex[0*nth + k] = eta[this.ie(0, ka)];
      for (let i = 0; i < nr; i++) this.Ex[(i+1)*nth + k] = eta[this.ie(i, k)];
      this.Ex[(nr+1)*nth + k] = free ? eta[this.ie(nr-1, k)] : 0;
    }

    /* eta's OWN faces and slopes, differenced from eta and never from H -- see the comment on
       `Er` in the constructor for the 2.046e-12 that made this necessary. Same shape as H's,
       with the rim value 0 rather than h, since a pinned line holds eta at zero there. */
    for (let k = 0; k < nth; k++){
      this.Er[0*nth + k] = eta[this.ie(0, k)];
      for (let i = 1; i < nr; i++){
        const a = this.drc[i-1], b = this.drc[i];
        const lo = eta[this.ie(i-1, k)], hi = eta[this.ie(i, k)];
        this.Er[i*nth + k] = lo + (a/(a + b))*(hi - lo);
      }
      this.Er[nr*nth + k] = free ? eta[this.ie(nr-1, k)] : 0;
    }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        this.Eth[i*nth + k] = 0.5*(eta[this.ie(i, k-1)] + eta[this.ie(i, k)]);
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        this.Edr[e] = (this.Er[(i+1)*nth + this.kw(k)] - this.Er[i*nth + this.kw(k)])
                      /this.drc[i];
        this.Edth[e] = (this.Eth[i*nth + this.kw(k+1)] - this.Eth[i*nth + this.kw(k)])
                       /this.dth;
      }
    for (let k = 0; k < nth; k++){
      const ka = this.kw(k + half);
      this.Exr[0*nth + k] = -this.Edr[this.ie(0, ka)];
      this.Ext[0*nth + k] =  this.Edth[this.ie(0, ka)];
      for (let i = 0; i < nr; i++){
        this.Exr[(i+1)*nth + k] = this.Edr[this.ie(i, k)];
        this.Ext[(i+1)*nth + k] = this.Edth[this.ie(i, k)];
      }
      this.Exr[(nr+1)*nth + k] = free ? 0
        : (0 - eta[this.ie(nr-1, k)])/this.drf[nr];
      this.Ext[(nr+1)*nth + k] = free ? this.Edth[this.ie(nr-1, k)] : 0;
    }

    /* d eta/d t, through the same faces, slopes and extended columns. Identical in shape to
       H's chain above, with the rim value 0 rather than h: a pinned contact line holds eta at
       zero for all time, so its rate there is zero too. Under a free line deta/dr = 0 at the
       wall, hence so is the rate's radial slope, exactly as for H. */
    const Ht = this.Ht;
    for (let k = 0; k < nth; k++){
      this.Tr[0*nth + k] = Ht[this.ie(0, k)];
      for (let i = 1; i < nr; i++){
        const a = this.drc[i-1], b = this.drc[i];
        const lo = Ht[this.ie(i-1, k)], hi = Ht[this.ie(i, k)];
        this.Tr[i*nth + k] = lo + (a/(a + b))*(hi - lo);
      }
      this.Tr[nr*nth + k] = free ? Ht[this.ie(nr-1, k)] : 0;
    }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        this.Tth[i*nth + k] = 0.5*(Ht[this.ie(i, k-1)] + Ht[this.ie(i, k)]);
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        this.Tdr[e] = (this.Tr[(i+1)*nth + this.kw(k)] - this.Tr[i*nth + this.kw(k)])
                      /this.drc[i];
        this.Tdth[e] = (this.Tth[i*nth + this.kw(k+1)] - this.Tth[i*nth + this.kw(k)])
                       /this.dth;
      }
    for (let k = 0; k < nth; k++){
      const ka = this.kw(k + half);
      this.Tx[0*nth + k]  =  Ht[this.ie(0, ka)];
      this.Txr[0*nth + k] = -this.Tdr[this.ie(0, ka)];
      this.Txt[0*nth + k] =  this.Tdth[this.ie(0, ka)];
      for (let i = 0; i < nr; i++){
        this.Tx[(i+1)*nth + k]  = Ht[this.ie(i, k)];
        this.Txr[(i+1)*nth + k] = this.Tdr[this.ie(i, k)];
        this.Txt[(i+1)*nth + k] = this.Tdth[this.ie(i, k)];
      }
      this.Tx[(nr+1)*nth + k]  = free ? Ht[this.ie(nr-1, k)] : 0;
      this.Txr[(nr+1)*nth + k] = free ? 0
        : (0 - Ht[this.ie(nr-1, k)])/this.drf[nr];
      this.Txt[(nr+1)*nth + k] = free ? this.Tdth[this.ie(nr-1, k)] : 0;
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
    const Hr = this.Hr, Hth = this.Hth, nw = nz + 1;
    for (let i = 0; i < nr; i++){
      const rci = rc[i], dr = drc[i], rIn = rf[i], rOut = rf[i+1];
      for (let k = 0; k < nth; k++){
        const kp = k + 1 === nth ? 0 : k + 1;
        const HrIn = Hr[i*nth + k], HrOut = Hr[(i+1)*nth + k];
        const HthIn = Hth[i*nth + k], HthOut = Hth[i*nth + kp];
        const bIn = (i*nth + k)*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
        const bW = (i*nth + k)*nw;
        for (let j = 0; j < nz; j++){
          const radial = dth*dsc[j]*(
              rOut*HrOut*u[bOut + j]
            - rIn *HrIn *u[bIn + j]);
          const azim = dr*dsc[j]*(
              HthOut*v[bK + j]
            - HthIn *v[bIn + j]);
          const vert = rci*dr*dth*(
              om[bW + j + 1] - om[bW + j]);
          out[bIn + j] = radial + azim + vert;
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
    const Hrm = this.Hr, Hthm = this.Hth, nw = nz + 1;
    for (let i = 0; i < nr; i++){
      const rci = rc[i], dr = drc[i], rIn = rf[i], rOut = rf[i+1];
      for (let k = 0; k < nth; k++){
        const kp = k + 1 === nth ? 0 : k + 1;
        const HrIn = Hrm[i*nth + k], HrOut = Hrm[(i+1)*nth + k];
        const HthIn = Hthm[i*nth + k], HthOut = Hthm[i*nth + kp];
        const bIn = (i*nth + k)*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
        const bW = (i*nth + k)*nw;
        for (let j = 0; j < nz; j++){
          const qc = q[bIn + j];
          gu[bOut + j] += qc*dth*dsc[j]*rOut*HrOut;
          gu[bIn  + j] -= qc*dth*dsc[j]*rIn *HrIn;
          gv[bK  + j] += qc*dr*dsc[j]*HthOut;
          gv[bIn + j] -= qc*dr*dsc[j]*HthIn;
          gw[bW + j + 1] += qc*rci*dr*dth;
          gw[bW + j    ] -= qc*rci*dr*dth;
        }
      }
    }
    /* THE SLOPE OPERATOR'S TRANSPOSE, which is what makes this the PHYSICAL pressure
       gradient rather than the covariant one, and without which nothing else here is a
       Navier-Stokes solver.

       The divergence above reads Omega. Expressed in the physical w it reads
       w - sigma(u H_r + (v/r) H_theta), so u and v enter it through that term too, and the
       transpose must return part of every sigma face's contribution to the four r faces and
       the four theta faces that meet there. Adding it converts

           d p / d r at constant SIGMA   into   d p / d r at constant Z

       and likewise azimuthally, which is exactly what the momentum equations for the
       physical components ask for. Without it the projection's corrector applies the
       covariant gradient: measured on p = A r + B z over a surface at eta/h = 0.4, the
       radial component returned 86.99 at r/R = 0.888, sigma = 0.63, against the physical
       137.00 -- a 36 per cent error at O(1), matching A + B sigma H_r to five digits.

       The file's header used to say that carrying w instead of Omega would cost the
       symmetry conjugate gradients needs. That is not so, and it is worth saying why: the
       divergence in the physical variables is D composed with this change of variable, its
       transpose is the change of variable's transpose composed with D's, and
       D W^-1 D^T is symmetric for ANY diagonal positive W whatever D is. What carrying w
       costs is a wider stencil -- a sigma face's pressure reaches eight horizontal faces --
       and nothing else. Gate 1 measures the symmetry either way.

       Also: the sigma face's control volume includes H, exactly as the r and theta faces' do.
       That was missing, and with any positive weight the operator stays symmetric negative
       definite and the projection still removes the divergence, so gates 1 and 2 passed
       either way -- but the vertical component then returned B H rather than B, smaller than
       the physical gradient by a factor of H, 333 on this cell. */
    const Hdr = this.Hdr, Hdth = this.Hdth, sfa = this.sf;
    for (let i = 0; i < nr; i++){
      const ri = rc[i];
      for (let k = 0; k < nth; k++){
        const e = i*nth + k;
        const cr = -0.25*Hdr[e], ct = -0.25*Hdth[e]/ri;
        const kp = k + 1 === nth ? 0 : k + 1;
        const bIn = (i*nth + k)*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
        const bW = (i*nth + k)*nw;
        for (let j = 1; j <= nz; j++){
          const raw = gw[bW + j], s = sfa[j];
          const jm = j - 1, jp = j === nz ? nz - 1 : j;
          const du = cr*s*raw, dv = ct*s*raw;
          gu[bIn + jm] += du; gu[bOut + jm] += du;
          gu[bIn + jp] += du; gu[bOut + jp] += du;
          gv[bIn + jm] += dv; gv[bK + jm] += dv;
          gv[bIn + jp] += dv; gv[bK + jp] += dv;
        }
      }
    }
    /* Divide by minus the control volume of each face. u at the axis and the rim
       is prescribed, so its gradient there is not solved for and is zeroed; the
       same for Omega on the floor. */
    const Hm = this.H;
    for (let k = 0; k < nth; k++)
      for (let j = 0; j < nz; j++){
        gu[k*nz + j] = 0;
        gu[(nr*nth + k)*nz + j] = 0;
      }
    for (let i = 1; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const Hf = Hrm[i*nth + k], b = (i*nth + k)*nz;
        for (let j = 0; j < nz; j++)
          gu[b + j] /= -(rf[i]*drf[i]*dth*Hf*dsc[j]);
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const Hf = Hthm[i*nth + k], b = (i*nth + k)*nz;
        for (let j = 0; j < nz; j++)
          gv[b + j] /= -(rc[i]*drc[i]*dth*Hf*dsc[j]);
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const H = Hm[i*nth + k], b = (i*nth + k)*nw;
        gw[b] = 0;                              // impermeable floor
        for (let j = 1; j <= nz; j++)
          gw[b + j] /= -(rc[i]*drc[i]*dth*H*dsf[j]);
      }
    return [gu, gv, gw];
  }

  /* The pressure operator, D W^-1 D^T, where D is the divergence read in the PHYSICAL
     velocity: `divergence` takes Omega, so the composition puts the gradient's three
     physical components through the same change of variable the transpose above went
     through. Leaving that out would compose D with the transpose of a different operator
     and the result would not be symmetric -- which gate 1 measures, so it cannot drift. */
  applyL(q, out){
    this.gradient(q, this._gu, this._gv, this._gw);
    this.omegaOf(this._gu, this._gv, this._gw, this._gom);
    this.divergence(this._gu, this._gv, this._gom, out);
    return out;
  }

  /* The diagonal of divergence(gradient(.)), exactly, by colouring. Set one colour class
     to one, apply the operator, and read the result at those same cells: if no two cells of
     a class are in each other's stencil, nothing but the diagonal contributes.

     THE STRIDES ARE (2, 2, 3) AND THAT IS NOT A MARGIN, it is the stencil. Once `gradient`
     carries the slope operator's transpose, a sigma face's pressure reaches the eight r and
     theta faces that meet there, and the reach becomes

         delta i in [-1, 1],  delta k in [-1, 1],  delta sigma in [-2, 2]

     including the corners -- delta i = +-1 together with delta sigma = +-2. Two cells of one
     class differ by an even delta i, an even delta k and a multiple of three in sigma, and
     the only such triple inside that box is the zero one. It used to be (2, 2, 2) in eight
     classes, which was right for the narrower stencil and is not right for this one: with it
     the diagonal came out wrong, the Jacobi preconditioner with it, and a projection asked
     for 1e-14 left a divergence of 6.7e-2 where the same projection at 1e-9 left 3.3e-8.
     Twelve classes, twelve applications, and gate 3 compares the result against the diagonal
     read one unit vector at a time, so a wrong stride cannot pass.

     Read from applyL rather than rederived, so the preconditioner cannot drift
     from the operator it preconditions. */
  pressureDiagonal(){
    const nr = this.nr, nth = this.nth, nz = this.nz;
    const d = this._pdiag, probe = new Float64Array(this.NP), q = new Float64Array(this.NP);
    for (let c = 0; c < 12; c++){
      const pi = c & 1, pk = (c >> 1) & 1, pj = c >> 2;
      probe.fill(0);
      for (let i = 0; i < nr; i++) if ((i & 1) === pi)
        for (let k = 0; k < nth; k++) if ((k & 1) === pk)
          for (let j = 0; j < nz; j++) if (j % 3 === pj)
            probe[this.ip(i, k, j)] = 1;
      this.applyL(probe, q);
      for (let i = 0; i < nr; i++) if ((i & 1) === pi)
        for (let k = 0; k < nth; k++) if ((k & 1) === pk)
          for (let j = 0; j < nz; j++) if (j % 3 === pj)
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

  /* The same bilinear interpolation, without the object. `Hat` is called from inside the
     face loops of `famLaplacian` and was 12.7 per cent of a step's time on a 16x24x10 grid,
     most of it allocating and collecting one three-field object per face per stencil point.
     These two write the same arithmetic in the same order -- `HatH` when only H is wanted,
     which is five of the nine call sites, and `HatInto` into a caller-owned triple for the
     three that want the slopes as well. Nothing here rounds differently from `Hat`, and the
     bit-for-bit gate over forty driven and undriven steps is what says so rather than the
     claim. `Hat` itself stays, because the gate calls it and an object is the right shape
     for a one-off. */
  HatH(r, th){
    const nr = this.nr, rx = this.rx;
    let a = 0;
    while (a < nr && rx[a+1] < r) a++;
    if (a > nr) a = nr;
    return this.HatHBr(a, r, th);
  }
  /* The same, with the radial bracket already known -- which it is at every family node,
     from the table the constructor builds. */
  HatHBr(a, r, th){
    const nth = this.nth, rx = this.rx, Hx = this.Hx, dth = this.dth;
    const r0 = rx[a], r1 = rx[a+1];
    const fr = (r - r0)/(r1 - r0);
    const tt = th/dth - 0.5;
    const kb = Math.floor(tt);
    const ft = tt - kb;
    const k0 = this.kw(kb), k1 = this.kw(kb + 1);
    const h00 = Hx[a*nth + k0], h01 = Hx[a*nth + k1];
    const h10 = Hx[(a+1)*nth + k0], h11 = Hx[(a+1)*nth + k1];
    return (1 - fr)*((1 - ft)*h00 + ft*h01) + fr*((1 - ft)*h10 + ft*h11);
  }
  HatInto(r, th, o){
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
    o[0] = (1 - fr)*((1 - ft)*h00 + ft*h01) + fr*((1 - ft)*h10 + ft*h11);
    o[1] = (((1 - ft)*h10 + ft*h11) - ((1 - ft)*h00 + ft*h01))/(r1 - r0);
    o[2] = ((1 - fr)*(h01 - h00) + fr*(h11 - h10))/dth;
    return o;
  }

  /* THE SURFACE ELEVATION AT AN ARBITRARY POSITION, which is what a renderer asks for.
   *
   * It is `HatH` minus the still depth and not a separate interpolation, deliberately. The
   * extended grid `Hx` that HatH reads already carries the two things a resampler would
   * otherwise have to reinvent, and would reinvent differently: the row below the axis is the
   * ANTIPODAL continuation, `H[ie(0, k + nth/2)]` at r = -rc[0], so a point at or near r = 0
   * is interpolated across the axis rather than extrapolated up to it; and the row at r = R is
   * the contact condition itself, the last cell's own H under a free line and h under a pinned
   * one. A renderer that interpolated eta on its own would have to get both right to draw the
   * same surface the solver is solving, and any difference would appear as a defect in the
   * physics rather than in the drawing.
   *
   * Exact at a cell centre: rx[i+1] is rc[i] and Ex[(i+1)*nth + k] is eta[ie(i,k)], so the
   * bilinear weights there are fr = 1 and ft = 0 and the value is the stored one to the bit.
   * Second order between centres, which is the interpolation's own order and is gated as such.
   *
   * It reads `Ex`, the extension of ETA, and NOT `HatH(r, th) - this.h`. That was the first
   * version and it was wrong by exactly one ulp of h + eta everywhere it mattered; the reason,
   * and the three measurements, are in `refreshMetric` where Ex is filled. */
  /* d eta/d t and the two gradients, at an arbitrary position, on the same extended nodes and
   * the same bilinear weights `etaAt` uses.
   *
   * WHY A RENDERER NEEDS ALL FOUR. The page's grain transport is driven by a COMPLEX surface
   * amplitude, not a height: it forms
   *
   *     PG = -K_I grad |A|^2 + K_F (Re grad Im - Im grad Re)
   *
   * an intensity gradient plus a phase flux. The modal renderer built A from each mode's own
   * phase offset. A direct simulation has no per-mode phase, and does not need one: for any
   * oscillation the instantaneous pair (eta, -eta_t/omega) IS that amplitude, and it reduces
   * to the modal answer. Check the two limits. For a standing mode eta and eta_t share one
   * spatial profile, so |A|^2 = A0^2 J_m^2 cos^2(m theta) is time independent and
   * Re grad Im - Im grad Re vanishes: a standing wave has an intensity pattern and no phase
   * flux, which is right. For a travelling one the flux is nonzero. So the pair carries what
   * the transport asks for, and carries it from the instantaneous state rather than from an
   * assumption that each mode has a single frequency. */
  etaDotAt(r, th){
    const nth = this.nth, nr = this.nr, rx = this.rx, Tx = this.Tx, dth = this.dth;
    let a = 0;
    while (a < nr && rx[a+1] < r) a++;
    if (a > nr) a = nr;
    const r0 = rx[a], r1 = rx[a+1];
    const fr = (r - r0)/(r1 - r0);
    const tt = th/dth - 0.5;
    const kb = Math.floor(tt);
    const ft = tt - kb;
    const k0 = this.kw(kb), k1 = this.kw(kb + 1);
    const t00 = Tx[a*nth + k0], t01 = Tx[a*nth + k1];
    const t10 = Tx[(a+1)*nth + k0], t11 = Tx[(a+1)*nth + k1];
    return (1 - fr)*((1 - ft)*t00 + ft*t01) + fr*((1 - ft)*t10 + ft*t11);
  }
  /* The gradient of eta at an arbitrary position, from ETA's own extended slope columns.
     Deliberately NOT `Hslope`: eta_r is H_r in exact arithmetic and not in doubles, because
     H's face values difference two h + eta sums and lose the low bits of eta's own difference.
     That was 2.046e-12 of relative error in the renderer's phase flux, which is a difference
     of two products that must cancel exactly for a standing wave. See `Er` in the constructor. */
  etaSlopeAt(r, th){
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
    return { Hr: lerp(this.Exr), Hth: lerp(this.Ext) };
  }
  /* And the gradient of d eta/d t, from its own extended slope columns. */
  etaDotSlopeAt(r, th){
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
    return { Tr: lerp(this.Txr), Tth: lerp(this.Txt) };
  }

  etaAt(r, th){
    const nth = this.nth, nr = this.nr, rx = this.rx, Ex = this.Ex, dth = this.dth;
    let a = 0;
    while (a < nr && rx[a+1] < r) a++;
    if (a > nr) a = nr;
    const r0 = rx[a], r1 = rx[a+1];
    const fr = (r - r0)/(r1 - r0);
    const tt = th/dth - 0.5;
    const kb = Math.floor(tt);
    const ft = tt - kb;
    const k0 = this.kw(kb), k1 = this.kw(kb + 1);
    const e00 = Ex[a*nth + k0], e01 = Ex[a*nth + k1];
    const e10 = Ex[(a+1)*nth + k0], e11 = Ex[(a+1)*nth + k1];
    return (1 - fr)*((1 - ft)*e00 + ft*e01) + fr*((1 - ft)*e10 + ft*e11);
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
   * CUBIC in sigma, on the four nodes lev-1 .. lev+2 -- anchored on the LEVEL and not
   * chosen by bracketing the target, so that two columns compared at one height use the
   * same node positions and their reconstruction errors cancel instead of jumping as a
   * target crosses a node.
   *
   * Cubic and not quadratic for the reason polyDerivAt gives, one step removed. A
   * quadratic leaves the reconstruction wrong by O(ds^2 dr) at a height one column's own
   * surface fixes, so the difference of two columns over dr carries O(ds^2); between the
   * two sigma faces of an interior row that cancels, but at the SURFACE row -- where the
   * other face's flux comes from the stress condition and is not a discretisation of
   * anything -- there is nothing to cancel against, and dividing by the top row's
   * thickness turns it into O(ds). Measured with a quadratic and the surface flux supplied
   * analytically, family p's surface row read 1.68e-2, 1.58e-2, 1.05e-2 over 16, 32, 64:
   * order 0.08 then 0.59. The cubic makes it O(ds^3) instead, which the same division
   * leaves second order.
   *
   * Radial index a < 0 is the antipodal column reflected through the axis, carrying
   * the family's own sign: plus for a scalar or the vertical component, minus for a
   * horizontal one. */
  colValueAtZ(f, fam, a, k, z, lev, bracket){
    const sn = fam.sn, nJ = sn.length, half = this.nth >> 1;
    let aa = a, kk = k, sign = 1;
    if (a < 0){ aa = -1 - a; kk = k + half; sign = fam.axisSign; }
    const th = (kk + fam.thOff)*this.dth;
    const H = this.HatHBr(fam.hBr[aa], fam.rn[aa], th);
    const ss = z/H;
    const base = (aa*this.nth + this.kw(kk))*fam.stride;
    if (nJ === 1) return sign*f[base];
    const n = nJ < 4 ? nJ : 4;
    /* `bracket` puts the stencil around the target instead of around `lev`, by a binary
       search for the interval holding it -- for the callers whose target is far from the
       level, where anchoring would extrapolate. famLaplacian's sigma-face cross terms say
       why, with the numbers; every other caller leaves it off. */
    let j0;
    if (bracket){
      let lo = 0, hi = nJ - 1;
      while (hi - lo > 1){ const m = (lo + hi) >> 1; if (sn[m] <= ss) lo = m; else hi = m; }
      j0 = lo - 1;
    } else j0 = (lev === undefined ? 1 : lev) - 1;
    if (j0 + n > nJ) j0 = nJ - n;
    if (j0 < 0) j0 = 0;
    let v = 0;
    for (let i = 0; i < n; i++){
      let L = 1;
      for (let m = 0; m < n; m++)
        if (m !== i) L *= (ss - sn[j0 + m])/(sn[j0 + i] - sn[j0 + m]);
      v += f[base + j0 + i]*L;
    }
    return sign*v;
  }

  /* The same stencil as colValueAtZ, differentiated instead of evaluated: df/dz in one
   * column at one physical height. Since z = sigma H at fixed (r, theta), d/dz is
   * (1/H) d/dsigma, so this is the quadratic's sigma-derivative over that column's own H.
   *
   * It sits beside colValueAtZ and shares its conventions -- the stencil centred on `lev`
   * rather than chosen by bracketing the target, the antipodal reflection for a negative
   * radial index, the family's own sign -- because the two are read at the same points and
   * a difference in stencil between them would be a difference nothing would catch. */
  colDerivAtZ(f, fam, a, k, z, lev){
    const sn = fam.sn, nJ = sn.length, half = this.nth >> 1;
    let aa = a, kk = k, sign = 1;
    if (a < 0){ aa = -1 - a; kk = k + half; sign = fam.axisSign; }
    const th = (kk + fam.thOff)*this.dth;
    const H = this.HatHBr(fam.hBr[aa], fam.rn[aa], th);
    const ss = z/H;
    if (nJ === 1) return 0;
    const n = nJ < 4 ? nJ : 4;
    let j0 = (lev === undefined ? 1 : lev) - 1;
    if (j0 + n > nJ) j0 = nJ - n;
    if (j0 < 0) j0 = 0;
    const sx = this._sx, sy = this._sy;
    const base = (aa*this.nth + this.kw(kk))*fam.stride;
    for (let m = 0; m < n; m++){ sx[m] = sn[j0 + m]; sy[m] = f[base + j0 + m]; }
    return sign*polyDerivAt(sx, sy, n, ss)/H;
  }

  /* The nine covariant derivatives of the velocity at the free surface above one pressure
   * cell, filled into a caller-supplied nine-element array in the order
   *
   *     [u_r,r  u_r,th  u_r,z   u_th,r  u_th,th  u_th,z   u_z,r  u_z,th  u_z,z]
   *
   * where u_{i,j} is the j-th covariant derivative of the i-th component, so the two that
   * carry the rotating basis are
   *
   *     u_r,theta    = (1/r) du_r/dtheta - u_theta/r
   *     u_theta,theta = (1/r) du_theta/dtheta + u_r/r
   *
   * The strain tensor and the surface flux are both built from these, so they are formed
   * once: two functions each forming their own nine derivatives would be two chances for
   * them to disagree about one.
   *
   * Every horizontal derivative is taken BETWEEN COLUMNS AT ONE PHYSICAL HEIGHT -- the
   * height of this cell's own surface -- which is rule 1 and is not optional: under
   * z = sigma H the neighbouring columns' sigma = 1 sits at a different height, and
   * differencing there carries an O(dH) error that vanishes on a flat surface and does not
   * converge on a deformed one. Measured with the neighbour read at its own sigma = 1
   * instead: E_rr at order -0.007 while the flat case still passed.
   *
   * Every vertical derivative is the quadratic through that column's three topmost sigma
   * nodes, differentiated at the target height. For w that interpolates, because w has a
   * node at sigma = 1; for u and v it extrapolates half a cell, which is what a field whose
   * vertical nodes are cell centres costs at a boundary.
   *
   * At the rim the wall supplies the outward neighbour, zero for all three components by no
   * slip. At the axis the inward neighbour is the antipodal column, carried with the
   * family's own reflection sign by colValueAtZ. */
  surfaceGradient(i, k, out){
    const nr = this.nr, dth = this.dth, rc = this.rc, R = this.R;
    const FU = this.FAM.u, FV = this.FAM.v, FW = this.FAM.w;
    const r = rc[i], z = this.H[this.ie(i, k)];
    const lu = FU.sn.length - 2, lw = FW.sn.length - 2;
    const uV = (a, kk) => this.colValueAtZ(this.u, FU, a, kk, z, lu);
    const uD = (a, kk) => this.colDerivAtZ(this.u, FU, a, kk, z, lu);
    const vV = (a, kk) => a > nr - 1 ? 0 : this.colValueAtZ(this.v, FV, a, kk, z, lu);
    const vD = (a, kk) => a > nr - 1 ? 0 : this.colDerivAtZ(this.v, FV, a, kk, z, lu);
    const wV = (a, kk) => a > nr - 1 ? 0 : this.colValueAtZ(this.w, FW, a, kk, z, lw);
    const wD = (a, kk) => a > nr - 1 ? 0 : this.colDerivAtZ(this.w, FW, a, kk, z, lw);
    const rcOf = a => a < 0 ? -rc[-1 - a] : (a > nr - 1 ? R : rc[a]);
    const span = rcOf(i + 1) - rcOf(i - 1);

    const uIn = uV(i, k), uOut = uV(i + 1, k);
    const ur = 0.5*(uIn + uOut);
    const vLo = vV(i, k), vHi = vV(i, k + 1);
    const ut = 0.5*(vLo + vHi);

    out[0] = (uOut - uIn)/this.drc[i];
    out[1] = 0.25*(uV(i, k+1) - uV(i, k-1) + uV(i+1, k+1) - uV(i+1, k-1))/(dth*r) - ut/r;
    out[2] = 0.5*(uD(i, k) + uD(i + 1, k));
    out[3] = 0.5*(vV(i+1, k) - vV(i-1, k) + vV(i+1, k+1) - vV(i-1, k+1))/span;
    out[4] = (vHi - vLo)/(dth*r) + ur/r;
    out[5] = 0.5*(vD(i, k) + vD(i, k+1));
    out[6] = (wV(i+1, k) - wV(i-1, k))/span;
    out[7] = 0.5*(wV(i, k+1) - wV(i, k-1))/(dth*r);
    out[8] = wD(i, k);
    return out;
  }

  /* The rate-of-strain tensor at the free surface above one pressure cell, as
   *
   *     [E_rr, E_thetatheta, E_zz, E_rtheta, E_rz, E_thetaz]
   *
   * which is the symmetric part of the nine derivatives above, and nothing more. */
  surfaceStrain(i, k, out){
    const g = this.surfaceGradient(i, k, this._sg);
    out[0] = g[0];
    out[1] = g[4];
    out[2] = g[8];
    out[3] = 0.5*(g[1] + g[3]);
    out[4] = 0.5*(g[2] + g[6]);
    out[5] = 0.5*(g[5] + g[7]);
    return out;
  }

  /* The flux the vector Laplacian's sigma = 1 face must carry, for each velocity
   * component, once the free-surface stress conditions are imposed: grad u_i . N at the
   * surface, with N the unnormalised outward normal.
   *
   * WHY IT IS NOT JUST THE TRACTION. famLaplacian's sigma-face flux is proj*(grad f . N),
   * and the free surface gives a traction, 2 rho nu E . n. For an incompressible flow with
   * constant viscosity 2 nu div E and nu grad^2 u are the same VOLUME operator, but their
   * face fluxes differ by nu u_{j,i} n_j -- a term that integrates to zero over a closed
   * surface and does not vanish face by face. Putting a traction on this one face while
   * every other face carries grad u_i . N would be a consistent discretisation of neither.
   *
   * The conversion is an identity, not an approximation. From E_ij = (u_{i,j} + u_{j,i})/2,
   *
   *     grad u_i . n  =  2 (E . n)_i  -  u_{j,i} n_j
   *
   * and the tangential stress condition is imposed by PROJECTION -- E . n is replaced by
   * (n . E . n) n, which is exact at any slope -- so with the unnormalised normal, where
   * E . N = lambda N and lambda = n . E . n,
   *
   *     grad u_i . N  =  2 lambda N_i  -  u_{j,i} N_j
   *
   * Nothing divides by 1 - |grad eta|^2 anywhere, so the forty-five degree degeneracy that
   * defeats solving the tangential conditions for du/dz never appears. The derivation of
   * that degeneracy is in dns/PLAN-cell3d.md.
   *
   * Filled into a caller-supplied three-element array as [radial, azimuthal, vertical],
   * at the surface above one PRESSURE cell. Each momentum family's own sigma = 1 face takes
   * its own component from here, interpolated to where that face sits. */
  surfaceLapFluxes(i, k, out){
    const g = this.surfaceGradient(i, k, this._sg);
    const n = this.surfaceNormal(this.rc[i], (k + 0.5)*this.dth);
    /* the unnormalised normal, whose radial and azimuthal parts are minus the surface
       slopes and whose vertical part is exactly one -- which is what makes the tangential
       traction vanish to the last bit rather than to an order */
    const Nr = -n.sr, Nt = -n.st, Nz = 1;
    const E0 = g[0], E1 = g[4], E2 = g[8];
    const E3 = 0.5*(g[1] + g[3]), E4 = 0.5*(g[2] + g[6]), E5 = 0.5*(g[5] + g[7]);
    /* lambda = n.E.n, formed with N and divided by |N|^2 */
    const NEN = E0*Nr*Nr + E1*Nt*Nt + E2*Nz*Nz
              + 2*(E3*Nr*Nt + E4*Nr*Nz + E5*Nt*Nz);
    const lam = NEN/(n.len*n.len);
    for (let c = 0; c < 3; c++)
      out[c] = 2*lam*(c === 0 ? Nr : c === 1 ? Nt : Nz)
             - (g[c]*Nr + g[3 + c]*Nt + g[6 + c]*Nz);
    return out;
  }

  /* The viscous normal stress at the free surface above one pressure cell, 2 rho nu n.E.n.
   *
   * This is the term the normal-stress condition contributes to the surface pressure, and
   * it is contracted with the FULL normal rather than with z-hat. The flat-normal form
   * 2 rho nu dw/dz is what the two-dimensional solver next door is entitled to, because it
   * linearises about a flat surface; this one is not.
   *
   * It is NOT obtained by eliminating anything. Writing E.n = lambda n and solving for
   * lambda divides by 1 - |grad eta|^2, which vanishes at a forty-five degree slope and
   * changes sign beyond it -- the derivation is in dns/PLAN-cell3d.md. The strain comes
   * from the interior field and the contraction is then just a contraction, well defined
   * at any slope. */
  surfaceNormalStress(i, k){
    const e = this.ie(i, k);
    const n = this.surfaceNormal(this.rc[i], (k + 0.5)*this.dth);
    const E = this.surfaceStrain(i, k, this._st);
    const a = n.nr, b = n.nth, c = n.nz;
    const nEn = E[0]*a*a + E[1]*b*b + E[2]*c*c
              + 2*(E[3]*a*b + E[4]*a*c + E[5]*b*c);
    return 2*this.rho*this.nu*nEn;
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
   * flux. `sFlux(a, k)`, if given, overrides the sigma = 1 face with grad f . N
   * directly -- which is how the free surface is closed, because its condition is a
   * traction and a traction is a flux, not a value. The axis needs no entry: at r = 0 the face area is exactly zero, and
   * inward stencils use the antipodal column with the family's reflection sign. */
  famLaplacian(f, out, fam, bc, sFlux){
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

    /* df/dr|_z at a chosen radius, third order there, from the cubic through the four
       columns that straddle it -- each column reconstructed at the SAME physical height,
       which is rule 1 above and the reason for colValueAtZ. Four and not two for the
       reason polyDerivAt gives: the radial grid is graded too, so the two-point difference
       is centred at the midpoint of its columns rather than at the face, and at the rim
       column -- where the outward face is the wall and its stencil is one-sided -- there is
       nothing for that error to cancel against. Measured on family p over a flat surface,
       the rim column read 2.27e-3 at order 1.71 then 1.43 while every interior column ran
       at 2.04 or better; the corner where the rim meets a sigma boundary was first order
       for the same reason in both directions at once.

       A column past the rim is the wall, whose value bc delivers at the requested height;
       one inside the axis is the antipodal column, which colValueAtZ carries with the
       family's own reflection sign. The wall is therefore an ordinary fourth point rather
       than a special case, and the three-point rim closure this replaces is gone with it.
       `a0` is the first column of the stencil; it is clamped so the four exist. */
    const aHi = (bc && rn[nI-1] < this.R) ? nI : nI - 1;
    /* How far inward the stencil may reach. A family whose first node is ON the axis --
       u, whose nodes are the r faces -- has no antipodal continuation to offer: column
       -1 would be that same node reflected, at the same radius zero, and two stencil
       points at one abscissa is a division by zero rather than a wide stencil. Such a
       family stops at its own axis node, which carries the reflection already (axisU).
       The others, whose nodes are cell centres, continue across as far as the stencil
       needs. */
    const aLo = rn[0] > 0 ? -nI : 0;
    const rx = this._rx4, ry = this._ry4;
    const dPhysR = (k, z, lev, x, a0) => {
      let j0 = a0;
      if (j0 + 3 > aHi) j0 = aHi - 3;
      if (j0 < aLo) j0 = aLo;
      const th = thOf(k);
      const HR = this.HatH(this.R, th);
      for (let m = 0; m < 4; m++){
        const a = j0 + m;
        rx[m] = a > nI - 1 ? this.R : colR(a);
        ry[m] = a > nI - 1 ? bc('rim', this.R, th, z/HR) : atZ(a, k, z, lev);
      }
      return polyDerivAt(rx, ry, 4, x);
    };
    /* df/dtheta|_z AT A THETA FACE, two points, and two points deliberately.
       theta is uniform and periodic, so the difference over the interval it spans is
       centred exactly at the face and its truncation coefficient is the same at every k --
       which cancels between opposite faces. More than that, THIS STENCIL IS LOAD-BEARING
       for the cylindrical coupling: for a field uniform in Cartesian terms the azimuthal
       part of the scalar Laplacian is a spurious -u/r^2 and it is the coupling term
       (2/r^2) du_theta/dtheta that cancels it, and that cancellation is between two
       DISCRETE expressions, so it holds only while the two use matching stencils. Raising
       this one to four points on its own took the vector Laplacian from order 2.10 to 0.67
       on a flat surface -- the mismatch is O(dtheta^2)/r^2, which at the first cell is
       O(1). */
    const dPhysThFace = (a, kL, kR, z, lev) =>
      (atZ(a, kR, z, lev) - atZ(a, kL, z, lev))/((kR - kL)*dth);
    const tx = this._tx4, ty = this._ty4;

    /* d f / d sigma AT a sigma face, third order there, from the cubic through the
       four values that straddle it -- polyDerivAt says why four and not two, with
       the measurement, and the short of it is that a boundary row has nothing for
       its interior face's error to cancel against.

       A boundary value counts as one of the four, at the boundary's own sigma. It
       exists only where the family's outermost node is off the boundary AND a `bc`
       is given to supply it: w's first and last nodes ARE sigma = 0 and sigma = 1,
       so w reads only its own nodes, and no family reads a surface value while
       `sFlux` closes that face -- a traction is not a value, and there is no surface
       Dirichlet datum to read. Where a boundary value is unavailable the four run
       one-sided into the interior, which is third order there too. */
    const sLoB = sb[0], sHiB = sb[nJ];
    const jLo = (bc && sn[0] > sLoB) ? -1 : 0;
    const jHi = (bc && !sFlux && sn[nJ-1] < sHiB) ? nJ : nJ - 1;
    if (jHi - jLo < 3) throw new Error(
      `famLaplacian: this family offers ${jHi - jLo + 1} values in sigma and the `
      + `face derivative needs four. nz >= 4 guarantees them, so this is a `
      + `descriptor error rather than a grid that is too coarse.`);
    const sAbs = j => j < 0 ? sLoB : (j > nJ - 1 ? sHiB : sn[j]);
    const sVal = (a, k, j, th) => j < 0 ? bc('floor', rn[a], th, sLoB)
                               : (j > nJ - 1 ? bc('surface', rn[a], th, sHiB)
                                             : at(a, k, j));
    const sx = this._sx, sy = this._sy;
    /* The four values straddling one sigma face of row b, loaded into sx and sy, and the
       index the first of them sits at -- returned because the cross terms below interpolate
       the SAME four levels in the neighbouring columns, so that every column's interpolation
       error carries the same coefficient and the r and theta differences see a smooth one. */
    const sStencil = (b, side) => {
      let j0 = side < 0 ? b - 2 : b - 1;
      if (j0 < jLo) j0 = jLo;
      if (j0 + 3 > jHi) j0 = jHi - 3;
      return j0;
    };
    const loadS = (a, k, j0, th) => {
      for (let m = 0; m < 4; m++){
        sx[m] = sAbs(j0 + m);
        sy[m] = sVal(a, k, j0 + m, th);
      }
    };
    /* THE TANGENTIAL GRADIENTS A SIGMA FACE WANTS, and the one place in this operator where
       rule 1's common height must NOT be used -- with a construction that keeps what rule 1
       is for and drops the height matching that makes it fail here.
       .
       Rule 1 exists because f_r|_z = f_r|_sigma - sigma H_r f_z is a difference of two O(1)
       terms that must cancel exactly for f = f(z), and reading both columns at one height
       makes it cancel by construction. On the r and theta faces that works, because the
       height wanted differs from a neighbour's own level by only H_r dr/H or H_theta dtheta/H
       of a sheet. On a SIGMA face near the surface it does not: the sheet is at this column's
       sigma H, and a neighbour's level at that height is off by H_theta dtheta / H, measured
       at 18 to 20 times the top row's thickness over 16/32/64 -- a ratio refinement does not
       reduce, since dtheta and ds_top both fall like 1/n. Anchored on the level the cubic was
       extrapolating ten stencil widths past its own nodes and the azimuthal cross term read
       3.05e-3, 1.94e-3, 5.91e-4, order 0.65 then 1.71. Choosing the stencil to bracket the
       target instead removed the extrapolation but made the error jump as the stencil changed
       between adjacent theta columns, which a theta derivative divides by dtheta: 3.73e-3,
       9.06e-4, 3.94e-4, 1.25e-4 over four grids, order 2.04, 1.20, 1.66. With the
       reconstruction replaced by its analytic value the same term read 1.95e-3, 3.15e-4,
       4.53e-5, 5.84e-6, order 2.63, 2.80, 2.96 -- so neither stencil is the difficulty, the
       reconstruction is.
       .
       Writing the subtraction in sigma instead, with the same discrete operator on H as on
       the field -- so that f = z cancels to the last bit whatever the stencil -- was tried
       and is WORSE, by a factor of thirty, and the reason is worth keeping because it is what
       rule 1 is really about. In sigma coordinates the field carries the surface's own
       azimuthal variation multiplied by the vertical wavenumber: f(r, theta, sigma H(theta))
       oscillates in theta at an effective wavenumber k_z H_theta, which here is 3.15 radians
       per radian of theta, so 0.82 radians per azimuthal cell -- barely resolved. The
       four-point difference's O(dtheta^4 f^(5)) error then evaluates to 0.05 against a term of
       size 2, and the surface row divides it by its own thickness: predicted 0.16, measured
       1.566e-1. Radially the same substitution is harmless (measured indistinguishable from
       the form below), because the radial slope is gentler; azimuthally it is not. At a
       COMMON HEIGHT the field varies in theta only through its own shape, which is smooth,
       and that is the whole of rule 1's content.
       .
       So: common height, bracketed. */
    const atZB = (a, k, z) => this.colValueAtZ(f, fam, a, k, z, undefined, true);
    const dSigR = (a, k, z) => {
      let j = a - 1;
      if (j + 3 > aHi) j = aHi - 3;
      if (j < aLo) j = aLo;
      const th2 = thOf(k), HR = this.HatH(this.R, th2);
      for (let m = 0; m < 4; m++){
        const a2 = j + m;
        rx[m] = a2 > nI - 1 ? this.R : colR(a2);
        ry[m] = a2 > nI - 1 ? bc('rim', this.R, th2, z/HR) : atZB(a2, k, z);
      }
      return polyDerivAt(rx, ry, 4, rn[a]);
    };
    const dSigTh = (a, k, z) => {
      for (let m = 0; m < 4; m++){
        tx[m] = (k + TH_NODE[m])*dth;
        ty[m] = atZB(a, k + TH_NODE[m], z);
      }
      return polyDerivAt(tx, ty, 4, k*dth);
    };

    for (let a = fam.rLo; a <= fam.rHi; a++){
      const dra = rb[a+1] - rb[a];
      for (let k = 0; k < nth; k++){
        const th = thOf(k);
        const mid = this.HatInto(rn[a], th, this._hA);
        const midS = this.Hslope(rn[a], th);
        for (let b = fam.sLo; b <= fam.sHi; b++){
          const dsb = sb[b+1] - sb[b];
          /* The sigma CENTROID of this row's control volume, which is where the r and theta
             faces' one-point quadrature belongs -- not the node. For the families whose
             nodes are cell centres the two are the same number bit for bit; for w, whose
             nodes are the sigma faces, they differ by O(ds^2) in the interior and by
             ds_top/4 at the surface, where the control volume is the half cell the node
             bounds rather than straddles. Evaluating there instead cost the surface row an
             order: against the exact average over that half cell it read 1.22e-4, 6.53e-5,
             3.43e-5, order 0.91 then 0.93, and reads second order once the quadrature sits
             at the centroid. */
          const sMid = 0.5*(sb[b] + sb[b+1]);
          let flux = 0;

          /* ---- the two r faces: normal r-hat, so the flux is df/dr|_z ---- */
          for (const side of [-1, +1]){
            const rface = side < 0 ? rb[a] : rb[a+1];
            if (rface === 0) continue;                  // the axis: zero area
            const g = this.HatInto(rface, th, this._hB);
            if (!bc && (side < 0 ? a - 1 : a + 1) > nI - 1) continue;
            const z = sMid*g[0];
            const d = dPhysR(k, z, b, rface, side < 0 ? a - 2 : a - 1);
            flux += side*rface*dth*g[0]*dsb*d;
          }

          /* ---- the two theta faces: normal theta-hat ---- */
          for (const side of [-1, +1]){
            const thf = th + side*0.5*dth;
            const g = this.HatInto(rn[a], thf, this._hB);
            const z = sMid*g[0];
            const d = side < 0 ? dPhysThFace(a, k - 1, k, z, b)
                               : dPhysThFace(a, k, k + 1, z, b);
            flux += side*dra*g[0]*dsb*d/rn[a];
          }

          /* ---- the two sigma faces: the curved sheets z = sigma H, whose normal
                  is proportional to (-sigma H_r, -sigma H_theta / r, 1) ---- */
          for (const side of [-1, +1]){
            const sface = side < 0 ? sb[b] : sb[b+1];
            const proj = rn[a]*dra*dth;
            const bn = side < 0 ? b - 1 : b + 1;
            if (bn > nJ - 1 && sFlux){
              /* THE FREE SURFACE, closed by a FLUX rather than by a value. The bracket below
                 computes grad f . N, where N is the sheet's unnormalised outward normal, and
                 proj is its projected area -- so a caller that knows what grad f . N must be
                 at the surface, because the stress conditions fix it, supplies exactly that
                 and the rest of the branch is skipped. `bc` cannot express this: a traction is
                 a flux, and there is no boundary VALUE that carries it. */
              flux += side*proj*sFlux(a, k);
              continue;
            }
            if ((bn < 0 || bn > nJ - 1) && !bc) continue;
            loadS(a, k, sStencil(b, side), th);
            const dsg = polyDerivAt(sx, sy, 4, sface);
            const z = sface*mid[0];
            flux += side*proj*( dsg/mid[0]
                              - sface*midS.Hr*dSigR(a, k, z)
                              - (sface*midS.Hth/(rn[a]*rn[a]))*dSigTh(a, k, z) );
          }

          out[idx(a, k, b)] = flux/(rn[a]*dra*dth*mid[0]*dsb);
        }
      }
    }
    return out;
  }

  /* The free surface's own flux, per velocity component, at every pressure cell. Formed once
     per viscous evaluation because all three components of `surfaceLapFluxes` come from one
     rate-of-strain tensor, and forming it three times would be three chances for them to
     disagree about one surface. */
  refreshSurfaceFluxes(){
    const t3 = this._sf3;
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++){
        this.surfaceLapFluxes(i, k, t3);
        const e = this.ie(i, k);
        this._fsr[e] = t3[0]; this._fst[e] = t3[1]; this._fsz[e] = t3[2];
      }
    return this;
  }

  /* and that flux where one family's own sigma = 1 face sits, which is not where the
     pressure cells are. u's faces are at r faces, v's at theta faces, and w's already at the
     pressure cell's own position.
     .
     THE RADIAL ONE IS A WEIGHTED INTERPOLATION RATHER THAN A MEAN, and that is a smaller
     matter than it looks -- said here because the opposite was expected and measured wrong.
     An r face is the midpoint of its two cell centres only when the two cells are equally
     wide, and this radius is graded towards the rim, so the arithmetic mean sits at
     (rc[a-1] + rc[a])/2 instead of at rf[a]. The offset is O(dr^2) on a smoothly graded grid,
     not O(dr), so BOTH forms are second order at the face: measured against the analytic flux
     there, 5.12e-2, 1.17e-2, 2.80e-3 over 16/32/64 at order 2.12 then 2.07 for the weights
     below, and 6.10e-2, 1.43e-2, 3.46e-3 at order 2.09 then 2.05 for the mean. The weights
     are kept because they are the right interpolation and cost two arithmetic operations, and
     because a surface-flux error is divided by the top row's thickness in the surface row --
     the identity 8e gates -- so a nineteen per cent smaller constant is worth having there.
     They are refreshMetric's own weights for H at an r face, written the same way, as an
     increment from one end so that two equal fluxes interpolate to exactly that flux.
     Azimuthally the mean is right rather than merely close: theta is uniform, so a theta face
     IS the midpoint of its two cells, exactly. */
  surfaceFluxFace(which, a, k){
    if (which === 'v') return 0.5*(this._fst[this.ie(a, k - 1)] + this._fst[this.ie(a, k)]);
    if (which === 'w') return this._fsz[this.ie(a, k)];
    const lo = this._fsr[this.ie(a - 1, k)], hi = this._fsr[this.ie(a, k)];
    const wa = this.drc[a-1], wb = this.drc[a];
    return lo + (wa/(wa + wb))*(hi - lo);
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
    this.refreshSurfaceFluxes();
    this.famLaplacian(this.u, outU, this.FAM.u, bcU,
      (a, k) => this.surfaceFluxFace('u', a, k));
    this.famLaplacian(this.v, outV, this.FAM.v, bcV,
      (a, k) => this.surfaceFluxFace('v', a, k));
    this.famLaplacian(this.w, outW, this.FAM.w, bcW,
      (a, k) => this.surfaceFluxFace('w', a, k));

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
        const Hu = this.HatH(r, (k + 0.5)*this.dth);
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
        const Hv = this.HatH(r, k*this.dth);
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
    /* Written out rather than as 1 + surfaceSlopeSquared(i, k), because ((1 + A) + B) + C and
       1 + ((A + B) + C) are not the same double and this quantity feeds the curvature, the
       surface pressure and every measured order in this file. On the fingerprint case the two
       happened to agree bit for bit, which is luck at one amplitude and not a property: with
       A, B, C all far below one, the first form truncates each addend against a leading 1
       three times and the second only once. The three lines are duplicated so that nothing
       downstream depends on that coincidence holding at another amplitude. */
    const s = this.etaSlopes(i, k, this._es);
    const rc = this.rc[i];
    return 1 + s[4]*s[0]*s[0] + s[5]*s[1]*s[1]
             + 0.5*(s[2]*s[2] + s[3]*s[3])/(rc*rc);
  }
  /* |grad eta|^2 alone, which is what `surfaceMetric` adds one to. Separate because the
     excess area needs it WITHOUT the one: sqrt(1 + q) - 1 loses every significant digit for
     small q, and so does surfaceArea() - pi R^2, where the two operands are 4.6e-4 and their
     difference at eta = 1e-9 m is 1.2e-18. Measured: the excess area came out 1.1926e-18
     against the 1.1596e-18 its own amplitude scaling demands, 2.8 per cent wrong, and the
     total energy then drifted 25.3 per cent over a quarter period where the same run at
     eta = 1e-7 drifts 0.0356. That was the DIAGNOSTIC failing, not the solver: the drift is
     -0.0344, -0.0349 and -0.0356 per cent at 1e-5, 1e-6 and 1e-7, flat across three decades
     as a linear regime must be. Gate 9c happened to sit at 1e-7 and so never saw it. */
  surfaceSlopeSquared(i, k){
    const s = this.etaSlopes(i, k, this._es);
    const rc = this.rc[i];
    return s[4]*s[0]*s[0] + s[5]*s[1]*s[1]
         + 0.5*(s[2]*s[2] + s[3]*s[3])/(rc*rc);
  }

  /* The outward unit normal of the free surface at an arbitrary position, from the
   * surface's own slopes. With z = H(r, theta) and h constant, eta_r is H_r and
   * eta_theta is H_theta, so
   *
   *     N = (-H_r, -H_theta/r, 1),   |N| = sqrt(1 + H_r^2 + (H_theta/r)^2)
   *
   * and n = N/|N|. Returned unnormalised as well as normalised, because the tangent
   * vectors (1, 0, H_r) and (0, 1, H_theta/r) are orthogonal to N without normalising and
   * the traction algebra is cleaner in those terms.
   *
   * THIS IS NOT THE SAME QUANTITY AS `surfaceMetric`, AND THEY MUST NOT BE UNIFIED. That
   * one is a CELL-AVERAGED squared slope, built from face differences weighted by face
   * area, and it is that particular average which makes the curvature the exact
   * derivative of the area. This one is a POINTWISE slope at an arbitrary position, which
   * is what a normal at a face is. They agree to second order and they are different
   * objects; replacing either with the other would break the thing it was built for. */
  surfaceNormal(r, th){
    const g = this.Hslope(r, th);
    const sr = g.Hr, st = g.Hth/r;
    const len = Math.sqrt(1 + sr*sr + st*st);
    return { sr, st, len, nr: -sr/len, nth: -st/len, nz: 1/len };
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

  /* The area the surface has IN EXCESS of flat, which is the quantity the capillary energy
     wants and is not the same computation as taking the difference of two areas.

     Sum_cells rc drc dtheta is exactly pi R^2 -- rc drc telescopes to R^2/2 and dtheta sums
     to 2 pi -- so the excess is Sum rc drc dtheta (sqrt(1 + q) - 1) term by term, with q the
     same cell-averaged squared slope `surfaceMetric` adds one to. Written as
     q/(1 + sqrt(1 + q)), which is the same number in exact arithmetic and keeps every digit
     for small q, where the subtraction has none: at eta = 1e-9 m the difference of areas is
     1.2e-18 out of operands of 4.6e-4, twenty times the double's own resolution.

     This is deliberately NOT how `surfaceArea` is written and `surfaceArea` is deliberately
     not written in terms of this. The curvature is the exact variational derivative of the
     area as `surfaceArea` computes it (gate 7c), and that identity is in floating point, not
     to a tolerance. Adding pi R^2 back to this function returns `surfaceArea` to round-off
     and the gate says by how much. */
  surfaceExcessArea(){
    let dA = 0;
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++){
        const q = this.surfaceSlopeSquared(i, k);
        dA += this.rc[i]*this.drc[i]*this.dth*(q/(1 + Math.sqrt(1 + q)));
      }
    return dA;
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
  /* ---- the free surface's pressure, and the time step ------------------- */

  /* The pressure the free surface carries, at every pressure column: the hydrostatic
     response of the displaced elevation to the instantaneous gravity, the capillary term
     with the FULL mean curvature rather than its linearisation, and the viscous normal
     stress with the full rate-of-strain contraction against the true normal.

         p_s = rho g_eff eta - gamma kappa + 2 rho nu (n.E.n)/|n|^2

     This is the same expression `surfacePressure` carries in dns/faraday-disc.js, whose two
     right-hand terms are that solver's flat-surface linearisations of these: `surfaceLaplacian`
     for the curvature and `wzSurface` for the stress. Nothing is linearised here.

     It is the inhomogeneous DIRICHLET value on the projection, never a force on the
     predictor -- see `step`. */
  surfacePressure(out){
    const rho = this.rho, g = this.gravity();
    this.curvature(this._kap);
    for (let i = 0; i < this.nr; i++)
      for (let k = 0; k < this.nth; k++){
        const e = this.ie(i, k);
        out[e] = rho*g*this.eta[e] - this.gamma*this._kap[e]
               + this.surfaceNormalStress(i, k);
      }
    return out;
  }

  /* ONE STEP OF THE NAVIER-STOKES EQUATIONS. Everything above is an operator; this is what
     makes the file a solver.

     The order is not a matter of taste, and three parts of it were established by
     measurement rather than by choice:

     1. THE SURFACE PRESSURE IS AN INHOMOGENEOUS DIRICHLET VALUE INSIDE THE PROJECTION, not
        a force on the predictor. As a predictor force it is p_s/(rho H dsigma) -- O(1/dsigma)
        -- and the projection then cancels almost all of it, leaving the physical
        acceleration as the difference of two large numbers and a step limit that collapses
        with the grid. dns/faraday-disc.js went non-finite 0.57 periods in that way, at
        nr = 48, nz = 20, m = 12. Resolved inside the projection nothing large cancels: the
        surface face's gradient gains p_s/(H dsigma) and the top row of the right-hand side
        loses that term's divergence.
     2. THE AXIS AND WALL VALUES ARE SET BEFORE Omega IS FORMED FROM THE VELOCITY. Omega is
        derived from u, v and w, and u at the axis face enters it through the innermost cell's
        slope term, so setting that value after forming Omega leaves the two inconsistent and
        the next projection undoes the one before it: measured, a projection asked for 1e-14
        reported a divergence of 6.74e-2 where the same projection at 1e-9 reported 3.40e-8.
     3. ETA ADVANCES ON THE CORRECTED Omega AT sigma = 1, which IS the kinematic condition --
        Omega there is dH/dt by construction, not by differencing H. Pairing the surface
        pressure at the old time with the surface velocity at the new one is symplectic Euler
        on the surface oscillator, stable for omega dt < 2 rather than merely less unstable
        than forward Euler.

     The predictor carries the viscous and advective terms in full. Gravity does not appear in
     it: this pressure is the total one, and the whole of gravity's effect on the interior is
     the hydrostatic head the surface value carries. */
  step(dt){
    if (!(typeof dt === 'number' && Number.isFinite(dt) && dt > 0)) throw new TypeError(
      `dt = ${dt}: a finite positive time step is required.`);
    const nr = this.nr, nth = this.nth, nz = this.nz, nu = this.nu, rho = this.rho;
    const u = this.u, v = this.v, w = this.w;
    const lu = this._lu, lv = this._lv, lw = this._lw;
    const au = this._au, av = this._av, aw = this._aw;

    /* the surface's own state, read BEFORE anything moves: the stresses and the curvature
       belong to the surface the velocity is being advanced over */
    this.omegaFromW();
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++) this.Ht[this.ie(i, k)] = this.om[this.iw(i, k, nz)];
    this.surfacePressure(this._ps);

    /* the explicit right-hand side: viscous, with the free surface's own traction on the
       sigma = 1 face, and advective, in the conservative grid-relative form */
    this.viscous(lu, lv, lw, this._bcU, this._bcV, this._bcW);
    this.advect(au, av, aw);

    for (let i = 1; i < nr; i++)
      for (let k = 0; k < nth; k++)
        for (let j = 0; j < nz; j++){
          const c = this.iu(i, k, j);
          u[c] += dt*(nu*lu[c] + au[c]);
        }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        for (let j = 0; j < nz; j++){
          const c = this.iv(i, k, j);
          v[c] += dt*(nu*lv[c] + av[c]);
        }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        for (let j = 1; j <= nz; j++){
          const c = this.iw(i, k, j);
          w[c] += dt*(nu*lw[c] + aw[c]);
        }

    /* the prescribed values, and only then Omega -- reason 2 above */
    for (let k = 0; k < nth; k++)
      for (let j = 0; j < nz; j++) u[this.iu(nr, k, j)] = 0;      // no penetration at the rim
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++) w[this.iw(i, k, 0)] = 0;      // no slip on the floor
    this.axisU();
    this.omegaFromW();

    /* the projection, with the surface pressure as its Dirichlet value -- reason 1 above */
    this.divergence(u, v, this.om, this._div);
    const scale = rho/dt;
    for (let c = 0; c < this._div.length; c++) this._div[c] *= scale;
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        this._div[this.ip(i, k, nz - 1)] -= this.rc[i]*this.drc[i]*this.dth
          *this._ps[this.ie(i, k)]/(this.H[this.ie(i, k)]*this.dsf[nz]);
    this.pressureDiagonal();
    this.p.fill(0);
    this.solveP(this._div, 1e-11, 400*(nr + nth + nz));

    this.gradient(this.p, this._gu, this._gv, this._gw);
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        this._gw[this.iw(i, k, nz)] += this._ps[e]/(this.H[e]*this.dsf[nz]);
      }

    const s2 = dt/rho;
    for (let i = 1; i < nr; i++)
      for (let k = 0; k < nth; k++)
        for (let j = 0; j < nz; j++){
          const c = this.iu(i, k, j);
          u[c] -= s2*this._gu[c];
        }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        for (let j = 0; j < nz; j++){
          const c = this.iv(i, k, j);
          v[c] -= s2*this._gv[c];
        }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++)
        for (let j = 1; j <= nz; j++){
          const c = this.iw(i, k, j);
          w[c] -= s2*this._gw[c];
        }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++) w[this.iw(i, k, 0)] = 0;
    this.axisU();
    this.omegaFromW();

    /* eta on the corrected Omega at the surface, which IS dH/dt -- reason 3 above */
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.ie(i, k);
        this.eta[e] += dt*this.om[this.iw(i, k, nz)];
        this.Ht[e] = this.om[this.iw(i, k, nz)];
      }
    this.t += dt;
    this.refreshMetric();
    return this;
  }

  /* The explicit step's limits, from the discrete operators' own wavenumber bounds rather
     than from a Cartesian rule of thumb. Four of them, and the azimuthal one is not
     decoration: dns/faraday-disc.js's first version of this ignored it and was wrong by a
     factor of three at m = 12, because the stiffest capillary mode is set by m/r at the
     INNERMOST cell and not by dr. Here every azimuthal mode the grid carries is present at
     once, so the stiffest is nth/2 at rc[0] always, with no mode number to be lucky about.

       capillary   the surface oscillator at the largest wavenumber the grid carries, from
                   the DISPERSION RELATION, omega^2 = (g_eff k + gamma k^3/rho) tanh(k h)
       viscous     the explicit diffusion bound, 2/(nu sum k_i^2)
       advective   1/max(|u|/dr + |v|/(r dtheta) + |w|/dz) over the cells, from the field
                   as it stands
       gravity     the shallow-water signal speed across the finest horizontal cell

     THE CAPILLARY ONE IS THE DISPERSION RELATION AND NOT THE HALF CELL, and the difference
     was measured rather than argued. dns/faraday-disc.js takes the surface stiffness to be
     (g + gamma k^2/rho) divided by the top half cell's thickness, on the reasoning that the
     surface pressure lands there; that is right for its own scheme and is too strict for
     this one, because the projection distributes that pressure through the whole column and
     the surface then responds at the PHYSICAL frequency. Measured directly: released from
     rest with eta = eps J_m(kr) cos(m theta), one step gives d eta/dt = -omega^2 eta dt with
     omega the continuum gravity-capillary frequency, to 2.1 per cent on a 32x48x20 grid --
     so the physical dispersion relation is what the discrete surface obeys and what its step
     limit follows from.

     It is still conservative, by a factor measured on two grids: the scheme is stable at
     1.35 and 2.03 times this limit on 10x16x8 and diverges at 2.71, and on 14x24x10 it is
     stable at 1.40 and diverges at 2.80. With `stableStep`'s safety factor of 0.4 that puts
     the default step between 3.5 and 7 times below where the scheme actually breaks, which
     gate 9 asserts on both sides rather than leaving as a claim.

     Returned together, so a caller can see which one binds. */
  stepLimits(){
    const nr = this.nr, nth = this.nth, nz = this.nz;
    let drMin = Infinity, dzMin = Infinity, Hmin = Infinity;
    for (let i = 0; i < nr; i++) drMin = Math.min(drMin, this.drc[i]);
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const H = this.H[this.ie(i, k)];
        Hmin = Math.min(Hmin, H);
        for (let j = 0; j < nz; j++) dzMin = Math.min(dzMin, H*this.dsc[j]);
      }
    const kr2 = 4/(drMin*drMin), kz2 = 4/(dzMin*dzMin);
    const mMax = nth >> 1, ka2 = (mMax*mMax)/(this.rc[0]*this.rc[0]);
    const kSurf2 = kr2 + ka2, kSurf = Math.sqrt(kSurf2);
    const gEff = this.g + Math.abs(this.accel);
    const omegaSurf = Math.sqrt((gEff*kSurf + this.gamma*kSurf2*kSurf/this.rho)
                                *Math.tanh(kSurf*Hmin));

    let adv = 0;
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const r = this.rc[i], H = this.H[this.ie(i, k)];
        for (let j = 0; j < nz; j++){
          const uu = Math.max(Math.abs(this.u[this.iu(i, k, j)]),
                              Math.abs(this.u[this.iu(i+1, k, j)]));
          const vv = Math.max(Math.abs(this.v[this.iv(i, k, j)]),
                              Math.abs(this.v[this.iv(i, k+1, j)]));
          const ww = Math.max(Math.abs(this.w[this.iw(i, k, j)]),
                              Math.abs(this.w[this.iw(i, k, j+1)]));
          adv = Math.max(adv, uu/this.drc[i] + vv/(r*this.dth)
                            + ww/(H*this.dsc[j]));
        }
      }
    return { capillary: 2/omegaSurf,
             viscous: 2/(this.nu*(kr2 + kz2 + ka2)),
             advective: adv > 0 ? 1/adv : Infinity,
             gravity: Math.min(drMin, this.rc[0]*this.dth)/Math.sqrt(gEff*Hmin),
             kRadial: Math.sqrt(kr2), kVertical: Math.sqrt(kz2),
             kAzimuthal: Math.sqrt(ka2) };
  }

  stableStep(safety){
    const s = safety === undefined ? 0.4 : requireFinitePositive(safety, 'safety');
    const L = this.stepLimits();
    return s*Math.min(L.capillary, L.viscous, L.advective, L.gravity);
  }

  /* The energy, by the same control volumes the projection weights its faces with -- which
     is what makes the projection non-increasing in this kinetic energy rather than in some
     other one.

     The capillary part is gamma times the surface's EXCESS area over the flat disc, and it
     uses `surfaceArea`, of which the curvature is the exact variational derivative: that is
     what makes the capillary work exactly -gamma dA/dt rather than approximately so.

     With a drive the total is not a conserved quantity -- the shaker does work on the cell --
     and the hydrostatic term is reported against the INSTANTANEOUS effective gravity, so it
     is a diagnostic of the state under the force acting on it rather than a state function.
     With accel = 0 and nu = 0 the total is conserved and that is the gate. */
  energy(){
    const nr = this.nr, nth = this.nth, nz = this.nz, rho = this.rho, dth = this.dth;
    let ke = 0;
    for (let i = 1; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const vol = this.rf[i]*this.drf[i]*dth*this.Hr[i*nth + this.kw(k)];
        for (let j = 0; j < nz; j++){
          const a = this.u[this.iu(i, k, j)];
          ke += 0.5*rho*a*a*vol*this.dsc[j];
        }
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const vol = this.rc[i]*this.drc[i]*dth*this.Hth[i*nth + this.kw(k)];
        for (let j = 0; j < nz; j++){
          const a = this.v[this.iv(i, k, j)];
          ke += 0.5*rho*a*a*vol*this.dsc[j];
        }
      }
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const vol = this.rc[i]*this.drc[i]*dth*this.H[this.ie(i, k)];
        for (let j = 1; j <= nz; j++){
          const a = this.w[this.iw(i, k, j)];
          ke += 0.5*rho*a*a*vol*this.dsf[j];
        }
      }
    const g = this.gravity();
    let hyd = 0;
    for (let i = 0; i < nr; i++)
      for (let k = 0; k < nth; k++){
        const e = this.eta[this.ie(i, k)];
        hyd += 0.5*rho*g*e*e*this.rc[i]*this.drc[i]*dth;
      }
    const cap = this.gamma*this.surfaceExcessArea();
    return { kinetic: ke, hydrostatic: hyd, capillary: cap, total: ke + hyd + cap };
  }

  /* sigma (u dH/dr + (v/r) dH/dtheta) at a sigma face, with u and v averaged from
     the four faces of the cell that meet there. Taken over GIVEN arrays rather than over
     the state, because the projection needs the same operator's transpose applied to the
     pressure gradient, not only to the velocity. */
  slopeTerm(i, k, j){ return this.slopeTermOf(this.u, this.v, i, k, j); }
  slopeTermOf(u, v, i, k, j){
    const e = this.ie(i, k), s = this.sf[j];
    if (s === 0) return 0;
    const jm = j === 0 ? 0 : j - 1, jp = j === this.nz ? this.nz - 1 : j;
    const uu = 0.25*(u[this.iu(i, k, jm)] + u[this.iu(i+1, k, jm)]
                   + u[this.iu(i, k, jp)] + u[this.iu(i+1, k, jp)]);
    const vv = 0.25*(v[this.iv(i, k, jm)] + v[this.iv(i, k+1, jm)]
                   + v[this.iv(i, k, jp)] + v[this.iv(i, k+1, jp)]);
    return s*(uu*this.Hdr[e] + (vv/this.rc[i])*this.Hdth[e]);
  }
  /* Omega from a given (u, v, w) triple into a given array -- the same definition
     omegaFromW applies to the state. */
  omegaOf(u, v, w, out){
    /* slopeTermOf inlined, with the index arithmetic hoisted out of the innermost loop.
       This is the second half of the conjugate-gradient matvec and runs once per iteration
       -- a hundred and more times a step -- so a per-point call that recomputes eight wrapped
       indices is most of it. Same expressions, same order, and the bit-for-bit gate says so. */
    const nr = this.nr, nth = this.nth, nz = this.nz, nw = nz + 1;
    const Hdr = this.Hdr, Hdth = this.Hdth, sfa = this.sf, rc = this.rc;
    for (let i = 0; i < nr; i++){
      const rci = rc[i];
      for (let k = 0; k < nth; k++){
        const e = i*nth + k;
        const hr = Hdr[e], hth = Hdth[e];
        const kp = k + 1 === nth ? 0 : k + 1;
        const bIn = e*nz, bOut = ((i+1)*nth + k)*nz, bK = (i*nth + kp)*nz;
        const bW = e*nw;
        for (let j = 0; j <= nz; j++){
          const c = bW + j, s = sfa[j];
          if (s === 0){ out[c] = w[c]; continue; }
          const jm = j === 0 ? 0 : j - 1, jp = j === nz ? nz - 1 : j;
          const uu = 0.25*(u[bIn + jm] + u[bOut + jm]
                         + u[bIn + jp] + u[bOut + jp]);
          const vv = 0.25*(v[bIn + jm] + v[bK + jm]
                         + v[bIn + jp] + v[bK + jp]);
          out[c] = w[c] - s*(uu*hr + (vv/rci)*hth);
        }
      }
    }
    return out;
  }

}

const FARADAY_CELL3D = { CELL3D_G0, FaradayCell3D };
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_CELL3D;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_CELL3D = FARADAY_CELL3D;
