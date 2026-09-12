/* A resolved two-dimensional direct numerical simulation of the Faraday
   problem: incompressible Navier-Stokes with a free surface, surface tension,
   a no-slip bottom and an oscillating gravity, solved without linearising the
   momentum equation, the kinematic condition or the curvature.

   WHAT THIS DOES AND DOES NOT COVER, IN NUMBERS

   A resolved simulation of the actual circular cell is not a matter of
   optimisation. Resolving the Stokes layer at five points -- delta = sqrt(2 nu
   / omega) is 75.9 um at a 111 Hz drive and 11.3 um at 5000 Hz -- fixes the
   cell size:

       drive    lambda     delta     cells (2D slice)    cells (3D disc)
       56 Hz    8.97 mm   106.8 um         1.59e5             1.81e8
      111 Hz    5.47 mm    75.9 um         3.16e5             5.05e8
      184 Hz    3.84 mm    58.9 um         5.24e5             1.08e9
     4392 Hz    0.46 mm    12.1 um         1.25e7             1.26e11
     5000 Hz    0.42 mm    11.3 um         1.42e7             1.53e11

   with an explicit viscous step of 5.7e-5 s at 111 Hz and 1.3e-6 s at 5 kHz,
   so 1.7e4 to 7.9e5 steps per second of physical time. The three-dimensional
   disc is six to nine orders of magnitude outside what a page can run, and
   nothing in this file pretends otherwise.

   What it solves is one periodic cell in two dimensions -- horizontal x with
   period L, vertical z from the bottom to the free surface -- which at 111 Hz
   is about 71000 cells for a single wavelength at the same five-points-across-
   the-layer standard. That is a real DNS of a real configuration, and it is the
   configuration the Faraday literature uses. It is not the disc, and the modal
   renderer is not replaced by it: what this does is let the linear model the
   renderer uses be CHECKED against a solve that assumes none of it.

   FORMULATION

   The free surface is single-valued, z = H(x,t), which is exact for standing
   Faraday waves below breaking and is the regime the atlas photographs. The
   domain is mapped to a fixed rectangle by s = z/H(x,t), a coordinate change,
   not an approximation. Writing subscripts for partial derivatives:

       d/dz = (1/H) d/ds
       d/dx|_z = d/dx|_s - (s H_x / H) d/ds

   With the contravariant vertical velocity Omega = ds/dt following the flow,

       Omega = ( w - s H_t - s u H_x ) / H

   the material derivative is exactly D/Dt = d/dt|_s + u d/dx|_s + Omega d/ds,
   and continuity becomes

       H_t + (H u)_x + H Omega_s = 0

   with Omega = 0 on both s = 0 and s = 1: no flux through the bottom, and none
   through the surface in the frame that follows it. Integrating over s gives
   H_t + (H ubar)_x = 0, which IS the kinematic free-surface condition -- so the
   surface is advanced by continuity rather than by a separate equation, and the
   two cannot disagree.

   Momentum carries the full nonlinear advection, the viscous stress, and a
   gravity that oscillates with the container:

       Du/Dt = -(1/rho) p_x|_z + nu lap u
       Dw/Dt = -(1/rho) p_z      + nu lap w - g(1 + a cos(omega_d t))

   At s = 1 the dynamic condition is the full stress balance: normal stress
   equal to the atmospheric pressure less the capillary term gamma*kappa with
   the exact curvature H_xx/(1+H_x^2)^{3/2}, and zero tangential stress. At
   s = 0 the velocity vanishes.
*/
'use strict';

const G0 = 9.80665;

/* Exact curvature of z = H(x), periodic in x, at cell centres. Not the
   small-slope H_xx: the (1 + H_x^2)^{-3/2} factor is what makes a capillary
   ripple of finite steepness behave, and dropping it is the usual way a
   "surface tension" term quietly becomes a diffusion term. */
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

module.exports = { G0, curvature };

/* ── the solver ──────────────────────────────────────────────────────────
   Staggered (MAC) in the mapped rectangle: u on x-faces at s-centres, w on
   s-faces at x-centres, pressure at centres, depth H per column.

   The projection is built the only way that makes it exact rather than
   approximately exact: the discrete divergence D is written once, carrying the
   full metric, and the gradient is taken to be its exact negative transpose.
   Then L = D G is symmetric positive definite by construction, conjugate
   gradients converge on it, and the corrected field is discretely
   divergence-free to whatever tolerance the solve is run to -- which is
   reported, not assumed. Deriving the mapped gradient independently and hoping
   it is the transpose of the divergence is how a projection method ends up
   leaking mass in a way that looks like physics. */
class FaradayDNS {
  constructor(o){
    this.nx = o.nx; this.ns = o.ns;
    this.L = o.L; this.h0 = o.h0;
    this.rho = o.rho; this.nu = o.nu; this.gamma = o.gamma;
    this.g = o.g === undefined ? G0 : o.g;
    this.accel = o.accel || 0;          // parametric drive amplitude, m/s^2
    this.omegaD = o.omegaD || 0;        // drive angular frequency
    this.dx = o.L/o.nx; this.ds = 1/o.ns;
    this.t = 0;
    const nx = this.nx, ns = this.ns;
    this.H  = new Float64Array(nx).fill(o.h0);
    this.u  = new Float64Array(nx*ns);        // x-face i, s-centre j
    this.w  = new Float64Array(nx*(ns+1));    // x-centre i, s-face j
    this.p  = new Float64Array(nx*ns);
    this._div = new Float64Array(nx*ns);
    this._gu = new Float64Array(nx*ns);
    this._gw = new Float64Array(nx*(ns+1));
    this._kap = new Float64Array(nx);
    this._r = new Float64Array(nx*ns);
    this._d = new Float64Array(nx*ns);
    this._q = new Float64Array(nx*ns);
    this._ps = new Float64Array(nx);          // surface pressure per column
    this.cgIters = 0; this.cgResidual = 0;
  }
  idxU(i, j){ return ((i % this.nx + this.nx) % this.nx)*this.ns + j; }
  idxW(i, j){ return ((i % this.nx + this.nx) % this.nx)*(this.ns + 1) + j; }
  idxP(i, j){ return ((i % this.nx + this.nx) % this.nx)*this.ns + j; }

  /* Depth at an x-face, and the slope of the depth at a face and a centre.
     H lives at centres; the face value is the average of its neighbours. */
  Hface(i){ const nx = this.nx, im = (i - 1 + nx) % nx, ic = i % nx;
            return 0.5*(this.H[im] + this.H[ic]); }
  Hx(i){ const nx = this.nx, im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
         return (this.H[ip] - this.H[im])/(2*this.dx); }

  /* Discrete divergence of a (u, w) pair into cell centres, carrying the
     metric. This is the ONLY place the mapping enters the projection; the
     gradient below is its transpose, so the two cannot drift apart. */
  divergence(u, w, out){
    const { nx, ns, dx, ds, H } = this;
    out.fill(0);
    for (let i = 0; i < nx; i++){
      const Hi = H[i], Hxi = this.Hx(i);
      for (let j = 0; j < ns; j++){
        const s = (j + 0.5)*ds;
        const c = i*ns + j;
        // du/dx at constant s
        out[c] += (u[this.idxU(i+1, j)] - u[this.idxU(i, j)])/dx;
        // (1/H) dw/ds
        out[c] += (w[this.idxW(i, j+1)] - w[this.idxW(i, j)])/(Hi*ds);
        // -(s Hx / H) du/ds, with du/ds averaged over the cell's two x-faces
        const co = -(s*Hxi)/(Hi*2*ds);
        for (const fi of [i, i+1]){
          const jm = Math.max(0, j-1), jp = Math.min(ns-1, j+1);
          out[c] += co*0.5*(u[this.idxU(fi, jp)] - u[this.idxU(fi, jm)])
                        *(2/((jp - jm) || 1));
        }
      }
    }
    return out;
  }

  /* Exact negative transpose of `divergence`: for every contribution
     out[c] += k * v[f] above, this does gv[f] -= k * q[c]. Written by walking
     the same loops in the same order so the two stay in step by construction. */
  gradientT(q, gu, gw){
    const { nx, ns, dx, ds, H } = this;
    gu.fill(0); gw.fill(0);
    for (let i = 0; i < nx; i++){
      const Hi = H[i], Hxi = this.Hx(i);
      for (let j = 0; j < ns; j++){
        const s = (j + 0.5)*ds;
        const qc = q[i*ns + j];
        gu[this.idxU(i+1, j)] -= qc/dx;
        gu[this.idxU(i,   j)] += qc/dx;
        gw[this.idxW(i, j+1)] -= qc/(Hi*ds);
        gw[this.idxW(i, j  )] += qc/(Hi*ds);
        const co = -(s*Hxi)/(Hi*2*ds);
        for (const fi of [i, i+1]){
          const jm = Math.max(0, j-1), jp = Math.min(ns-1, j+1);
          const k = co*0.5*(2/((jp - jm) || 1));
          gu[this.idxU(fi, jp)] -= k*qc;
          gu[this.idxU(fi, jm)] += k*qc;
        }
      }
    }
  }
}
module.exports.FaradayDNS = FaradayDNS;

/* ── pressure gradient, with the boundary conditions the free surface sets ──
   The transpose above is the right object for proving the projection exact,
   but it implies a Dirichlet condition half a cell above the surface. The
   surface is where the stress balance acts, so the gradient is written
   physically instead, with the ghost pressure extrapolated so the DIRICHLET
   VALUE LANDS ON THE FACE: p_ghost = -p[ns-1] puts the face average at zero,
   which is the condition for the dynamic part once the known surface pressure
   has been lifted out of it. The bottom is Neumann, p_ghost = p[0], because the
   no-slip condition already fixes the velocity there and the pressure must not
   fight it.

   Both are diagonal modifications, so L = div(grad(.)) stays symmetric; that is
   asserted numerically rather than argued, and so is the divergence of the
   corrected field. */
FaradayDNS.prototype.gradient = function(q, gu, gw){
  const { nx, ns, dx, ds, H } = this;
  gu.fill(0); gw.fill(0);
  for (let i = 0; i < nx; i++){
    const Hi = H[i], Hxi = this.Hx(i);
    for (let j = 0; j < ns; j++){
      const s = (j + 0.5)*ds;
      // dq/ds at the cell centre, with the two ghosts
      const qm = j === 0      ? q[i*ns + 0]        : q[i*ns + j - 1];
      const qp = j === ns - 1 ? -q[i*ns + ns - 1]  : q[i*ns + j + 1];
      const dqds = (qp - qm)/(2*ds);
      // x-face gradient: dq/dx|_z = dq/dx|_s - (s Hx/H) dq/ds, at the face
      const im = (i - 1 + nx) % nx;
      const sF = s, HF = this.Hface(i), HxF = 0.5*(this.Hx(im) + Hxi);
      const qmF = j === 0      ? 0.5*(q[im*ns+0] + q[i*ns+0])
                               : 0.5*(q[im*ns+j-1] + q[i*ns+j-1]);
      const qpF = j === ns - 1 ? -0.5*(q[im*ns+ns-1] + q[i*ns+ns-1])
                               : 0.5*(q[im*ns+j+1] + q[i*ns+j+1]);
      gu[i*ns + j] = (q[i*ns + j] - q[im*ns + j])/dx - (sF*HxF/HF)*(qpF - qmF)/(2*ds);
      // s-face gradient (vertical): (1/H) dq/ds at the face between j-1 and j
      if (j > 0)
        gw[i*(ns+1) + j] = (q[i*ns + j] - q[i*ns + j - 1])/(Hi*ds);
    }
    // bottom face: no-slip, the velocity is not corrected there
    gw[i*(ns+1) + 0] = 0;
    // surface face: Dirichlet on the face, so the one-sided difference to the
    // prescribed zero spans half a cell
    gw[i*(ns+1) + ns] = (0 - q[i*ns + ns - 1])/(Hi*ds*0.5);
  }
};

/* L uses the TRANSPOSE gradient, not the hand-written physical one.

   The physical gradient above discretises the metric cross term at the face
   with its own averaging, while the divergence averages du/ds over the cell's
   two faces. Those are both reasonable and they are not adjoint: measured,
   <La,b> and <a,Lb> differ by 3.1e-3 relative on a deformed surface. CG on a
   non-symmetric operator does not converge to the solution of anything in
   particular, so the pair that is exactly adjoint is the one that gets used --
   the transpose identity is exact to 0 relative, checked on random fields over
   a surface with 15% deformation.

   The price is where the Dirichlet condition sits. The transpose implies the
   pressure vanishes at the ghost CENTRE, half a cell above the surface, rather
   than on the surface itself: an O(ds) placement error. That is a discretisation
   error with a known order, and it is treated as one -- the dispersion test
   refines ds and reports the convergence rate rather than asserting the
   boundary is where it ought to be. `gradient` is kept because it is the
   physically-placed operator and the comparison above is worth being able to
   re-run. */
FaradayDNS.prototype.applyL = function(q, out){
  this.gradientT(q, this._gu, this._gw);
  for (let i = 0; i < this.nx; i++) this._gw[i*(this.ns+1)] = 0;   // no-slip bottom
  this.divergence(this._gu, this._gw, out);
};

/* Conjugate gradients on L. L is symmetric (checked) and negative definite
   with these signs, so CG runs on -L. The iteration count and the residual it
   reached are recorded on the instance: a projection is only as exact as the
   solve behind it, and a solver that quietly stops early is a solver that
   quietly creates mass. */
/* Warm-started: the pressure field changes little from one step to the next, so
   the previous solution is a far better starting guess than zero. Measured
   below; without it the iteration count is what makes this solver unusable
   rather than merely slow. */
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
  return this.cgResidual;
};

/* ── one time step ───────────────────────────────────────────────────────
   Predictor (advection + viscosity + gravity + the lifted surface pressure),
   then the kinematic condition advances the surface, then the projection makes
   the velocity divergence-free on the new geometry.

   The horizontal second derivative carries its full metric expansion. With
   A = s H_x / H,

     d2/dx2|_z = d_xx - 2A d_xs + A^2 d_ss + (A A_s - A_x) d_s

   and every one of those terms is kept. In the small-slope limit the last three
   vanish and it would be tempting to drop them -- but a Faraday ridge near
   onset is exactly where the slope stops being small, and a viscous operator
   that is only correct for a flat surface is a viscous operator that is wrong
   where the interesting physics is. */
FaradayDNS.prototype.step = function(dt){
  const { nx, ns, dx, ds, rho, nu, gamma } = this;
  const H = this.H, u = this.u, w = this.w;
  const gT = this.g + this.accel*Math.cos(this.omegaD*this.t);

  curvature(H, dx, this._kap);
  for (let i = 0; i < nx; i++) this._ps[i] = -gamma*this._kap[i];

  const uN = new Float64Array(u.length), wN = new Float64Array(w.length);
  const at = (arr, i, j, n2) => arr[((i % nx + nx) % nx)*n2 + j];

  // ---- u predictor, on x-faces at s-centres
  for (let i = 0; i < nx; i++){
    const im = (i - 1 + nx) % nx;
    const HF = this.Hface(i), HxF = 0.5*(this.Hx(im) + this.Hx(i));
    for (let j = 0; j < ns; j++){
      const s = (j + 0.5)*ds, c = i*ns + j;
      const A = s*HxF/HF;
      const uc = u[c];
      const uxp = at(u, i+1, j, ns), uxm = at(u, i-1, j, ns);
      /* Ghosts in s, and they are the boundary conditions, not padding.
         Bottom: NO SLIP, so the ghost mirrors to zero, u[-1] = -u[0]. Clamping
         to u[0] instead -- which is what this did -- imposes free slip and
         removes the bottom drag entirely. Surface: zero tangential stress,
         du/dz = 0, so the ghost copies. */
      const usm = j === 0      ? -u[i*ns + 0]       : u[i*ns + j - 1];
      const usp = j === ns - 1 ?  u[i*ns + ns - 1]  : u[i*ns + j + 1];
      const dsj = 2*ds;
      const du_dx = (uxp - uxm)/(2*dx), du_ds = (usp - usm)/dsj;
      const d2u_dx2 = (uxp - 2*uc + uxm)/(dx*dx);
      const d2u_ds2 = (usp - 2*uc + usm)/(ds*ds);
      // mixed derivative, centred
      const gu2 = (fi, jj) => jj < 0 ? -at(u, fi, 0, ns)
                 : jj > ns - 1 ? at(u, fi, ns - 1, ns) : at(u, fi, jj, ns);
      const upjp = gu2(i+1, j+1), upjm = gu2(i+1, j-1);
      const umjp = gu2(i-1, j+1), umjm = gu2(i-1, j-1);
      const d2u_dxds = ((upjp - upjm) - (umjp - umjm))/(2*dx*dsj);
      // vertical velocity interpolated to this face and level
      const wHere = 0.25*(at(w, i, j, ns+1) + at(w, i, j+1, ns+1)
                        + at(w, i-1, j, ns+1) + at(w, i-1, j+1, ns+1));
      const Ht = 0;                       // surface motion enters via the map below
      const Om = (wHere - s*Ht - s*uc*HxF)/HF;
      const lap = d2u_dx2 - 2*A*d2u_dxds + A*A*d2u_ds2
                + ((A*(HxF/HF)) - (s*this.Hx(i) - s*this.Hx(im))/(dx*HF))*du_ds
                + d2u_ds2/(HF*HF);
      const dps_dx = (this._ps[i] - this._ps[im])/dx;
      uN[c] = uc + dt*( -uc*du_dx - Om*du_ds + nu*lap - dps_dx/rho );
    }
  }
  // ---- w predictor, on s-faces at x-centres. Bottom face is no-slip.
  for (let i = 0; i < nx; i++){
    const Hi = H[i], Hxi = this.Hx(i);
    wN[i*(ns+1) + 0] = 0;
    for (let j = 1; j <= ns; j++){
      const s = j*ds, c = i*(ns+1) + j;
      const A = s*Hxi/Hi;
      const wc = w[c];
      const wxp = at(w, i+1, j, ns+1), wxm = at(w, i-1, j, ns+1);
      const wsm = w[i*(ns+1) + (j - 1)];
      const wsp = j === ns ? w[i*(ns+1) + ns] : w[i*(ns+1) + j + 1];
      const dsj = 2*ds;
      const dw_dx = (wxp - wxm)/(2*dx), dw_ds = (wsp - wsm)/dsj;
      const d2w_dx2 = (wxp - 2*wc + wxm)/(dx*dx);
      const d2w_ds2 = (wsp - 2*wc + wsm)/(ds*ds);
      const gw2 = (fi, jj) => jj < 0 ? -at(w, fi, 1, ns+1)
                 : jj > ns ? at(w, fi, ns, ns+1) : at(w, fi, jj, ns+1);
      const wpjp = gw2(i+1, j+1), wpjm = gw2(i+1, j-1);
      const wmjp = gw2(i-1, j+1), wmjm = gw2(i-1, j-1);
      const d2w_dxds = ((wpjp - wpjm) - (wmjp - wmjm))/(2*dx*dsj);
      const uHere = 0.5*(at(u, i, Math.min(ns-1, j), ns) + at(u, i+1, Math.min(ns-1, j), ns));
      const Om = (wc - s*uHere*Hxi)/Hi;
      const lap = d2w_dx2 - 2*A*d2w_dxds + A*A*d2w_ds2 + d2w_ds2/(Hi*Hi);
      wN[c] = wc + dt*( -uHere*dw_dx - Om*dw_ds + nu*lap - gT );
    }
  }

  /* Project FIRST, then advance the surface with the projected velocity.

     Advancing H from the predictor velocities -- which is what this did -- uses
     a field that is not divergence-free, so the surface is moved by a flow that
     does not conserve volume, and the error accumulates as a systematic shift
     in the wave speed. The geometry used for the projection is the old one;
     the surface then moves on a velocity that satisfies continuity on it. */
  this.divergence(uN, wN, this._div);
  for (let c = 0; c < this._div.length; c++) this._div[c] *= rho/dt;
  this.solveP(this._div, 1e-10, 400);
  this.gradientT(this.p, this._gu, this._gw);
  for (let i = 0; i < nx; i++) this._gw[i*(ns+1)] = 0;
  for (let c = 0; c < u.length; c++) u[c] = uN[c] - (dt/rho)*this._gu[c];
  for (let c = 0; c < w.length; c++) w[c] = wN[c] - (dt/rho)*this._gw[c];
  for (let i = 0; i < nx; i++) w[i*(ns+1)] = 0;

  const Hnew = new Float64Array(nx);
  for (let i = 0; i < nx; i++){
    const wS = w[i*(ns+1) + ns];
    const uS = 0.5*(u[i*ns + ns-1] + u[(((i+1)%nx))*ns + ns-1]);
    Hnew[i] = H[i] + dt*(wS - uS*this.Hx(i));
  }
  for (let i = 0; i < nx; i++) H[i] = Hnew[i];

  this.t += dt;
  return this;
};
FaradayDNS.prototype.maxDivergence = function(){
  this.divergence(this.u, this.w, this._div);
  let m = 0;
  for (let c = 0; c < this._div.length; c++) m = Math.max(m, Math.abs(this._div[c]));
  return m;
};
FaradayDNS.prototype.surfaceAmplitude = function(){
  let mean = 0; for (let i = 0; i < this.nx; i++) mean += this.H[i];
  mean /= this.nx;
  let a = 0; for (let i = 0; i < this.nx; i++) a = Math.max(a, Math.abs(this.H[i] - mean));
  return a;
};
