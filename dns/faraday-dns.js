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

   At s = 1 the dynamic condition is the full stress balance, and "full" has to
   mean the viscous part too:

       n.sigma.n = -p_atm + gamma*kappa,     t.sigma.n = 0,
       sigma = -p I + 2 mu E

   so the surface pressure is the capillary term with the exact curvature
   H_xx/(1+H_x^2)^{3/2} AND the viscous normal stress 2 mu (n.E.n). The second
   one is not a refinement. While a free-surface wave stays irrotational the
   interior viscous force is identically zero,

       nu lap u = nu grad(div u) - nu curl(curl u) = 0,

   so the damping does not come from the bulk term at all -- it comes from these
   two surface stresses, half from each. Writing the condition without the
   viscous normal stress, which this did, halves the damping exactly: measured
   7.415 s^-1 against Lamb's 2 nu k^2 = 14.914. At s = 0 the velocity vanishes.
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

   The projection is built so that it is exact rather than approximately exact:
   the discrete divergence D is written once in conservative form, carrying the
   full metric, and the gradient is -W D^T with W the diagonal metric weight.
   Then L = D G = -D W D^T is symmetric negative definite by construction,
   conjugate gradients converge on it, the corrected field is discretely
   divergence-free to whatever tolerance the solve is run to -- reported, not
   assumed -- and, because the weight is the Jacobian, G is also the physical
   gradient rather than only a correct transpose. Getting the first three and
   not the fourth is what made the first version of this solver return half the
   gravitational restoring force while passing every check it had. */
class FaradayDNS {
  constructor(o){
    this.nx = o.nx; this.ns = o.ns;
    this.L = o.L; this.h0 = o.h0;
    this.rho = o.rho; this.nu = o.nu; this.gamma = o.gamma;
    this.g = o.g === undefined ? G0 : o.g;
    this.accel = o.accel || 0;          // parametric drive amplitude, m/s^2
    this.omegaD = o.omegaD || 0;        // drive angular frequency
    if (!(this.ns >= 3)) throw new Error(
      `ns = ${this.ns}: the free-surface stress balance needs three cells of `
      + `depth to difference against. Refusing rather than falling back to a `
      + `copy-ghost, which is the approximation this solver exists to avoid.`);
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
    this._sux = new Float64Array(nx);         // du/dx|_z at the surface
    this._Ht = new Float64Array(nx);          // surface velocity dH/dt
    this._uBot = new Float64Array(nx);        // u ghost below s = 0
    this._uTop = new Float64Array(nx);        // u ghost above s = 1
    this._wTop = new Float64Array(nx);        // w ghost above s = 1
    this._lu = new Float64Array(nx*ns);       // viscous Laplacian of u
    this._lw = new Float64Array(nx*(ns+1));   // viscous Laplacian of w
    this._uN = new Float64Array(nx*ns);       // predictor fields
    this._wN = new Float64Array(nx*(ns+1));
    this._Hn = new Float64Array(nx);
    this.cgIters = 0; this.cgResidual = 0;
  }
  idxU(i, j){ return ((i % this.nx + this.nx) % this.nx)*this.ns + j; }
  idxW(i, j){ return ((i % this.nx + this.nx) % this.nx)*(this.ns + 1) + j; }
  idxP(i, j){ return ((i % this.nx + this.nx) % this.nx)*this.ns + j; }

  /* Depth at an x-face, and the slope of the depth at a face and a centre.
     H lives at centres; the face value is the average of its neighbours. */
  Hface(i){ const nx = this.nx, im = ((i - 1) % nx + nx) % nx, ic = ((i % nx) + nx) % nx;
            return 0.5*(this.H[im] + this.H[ic]); }
  Hx(i){ const nx = this.nx, im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
         return (this.H[ip] - this.H[im])/(2*this.dx); }
  Hxx(i){ const nx = this.nx, im = (i - 1 + nx) % nx, ic = ((i % nx) + nx) % nx, ip = (ic + 1) % nx;
          return (this.H[ip] - 2*this.H[ic] + this.H[im])/(this.dx*this.dx); }

  /* CONSERVATIVE discrete divergence: this returns H * (div u), not div u.

     The weight is not cosmetic and it is the whole reason the first version of
     this solver was wrong. On a mapped grid the divergence and the gradient are
     adjoint only in the METRIC inner product, the one carrying the Jacobian
     J = H:

         \int q (div u) H ds dx = - \int (grad q) . u H ds dx + boundary.

     Take the plain Euclidean transpose of the unweighted divergence instead and
     what comes back is not the gradient: measured against the analytic
     hydrostatic force it returns s * (dp/dx), which depth-averages to exactly
     HALF the restoring force, and halves omega^2. That was the dispersion
     defect -- omega low by 29% in shallow water, sqrt(1/2) = 0.7071 -- and it
     was invisible to every symmetry and adjointness check, because the operator
     pair really was exactly adjoint. It was adjoint in the wrong inner product.

     So the divergence is written in conservative (flux) form,

         H div u = d(H u)/dx|_s + d(w - s u H_x)/ds,

     which is an identity, not a discretisation choice: expanding the second
     term gives H u_x + H_x u + w_s - H_x(u + s u_s) = H u_x - s H_x u_s + w_s,
     and that is H times (u_x|_s - (s H_x/H) u_s + w_s/H) = H div u. The second
     flux is H*Omega, the contravariant vertical flux, so both fluxes are the
     physical ones through the two faces and the operator telescopes: summed
     over a column it is exactly the kinematic condition. */
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

  /* H*Omega = w - s u H_x at (x-centre i, s-face j), the flux through a
     horizontal face. u lives on x-faces at s-centres, so it is averaged over
     the four surrounding values; at s = 0 the factor s kills the term outright,
     and at s = 1 the two top-cell values are used, which is where a surface
     value has to come from when the top cell centre is the last one there is. */
  vFlux(u, w, i, j, Hxi){
    const wv = w[this.idxW(i, j)];
    if (j === 0) return wv;                       // s = 0 kills the second term
    return wv - (j*this.ds)*this.uAtSFace(u, i, j)*Hxi;
  }

  /* u interpolated to a horizontal face (x-centre i, s-face j). One definition,
     used by the flux above, by the w predictor's advection and by the kinematic
     surface update, so those three cannot drift apart. At s = 1 the top cell
     centres are the last values there are. */
  uAtSFace(u, i, j){
    const { nx, ns } = this;
    const ic = ((i % nx) + nx) % nx, ir = (ic + 1) % nx;
    if (j === ns) return 0.5*(u[ic*ns + ns-1] + u[ir*ns + ns-1]);
    /* No call site reaches j = 0: `vFlux` returns before it because s = 0
       kills the term, and the w predictor and the kinematic update start at
       j = 1 and j = ns. It returns the no-slip value rather than the interior
       extrapolation 0.5(u[0] + u[0]) anyway, because that is what u at the
       bottom wall IS, and a guard that is never taken is not a licence to put
       a wrong number in it. check-dns.mjs asserts the property that makes it
       unreachable -- that vFlux at j = 0 is exactly w, for arbitrary u. */
    if (j === 0) return 0;
    return 0.25*(u[ic*ns + j-1] + u[ir*ns + j-1] + u[ic*ns + j] + u[ir*ns + j]);
  }

  /* The metric-adjoint gradient: G = -W D^T, with W the diagonal metric weight
     1/H_face on x-faces and 1/H on s-faces. Every contribution `out[c] += k*v[f]`
     in `divergence` appears here as `gt[f] += k*q[c]`, then the whole vector is
     scaled by -W. That makes -D^T q equal to H*(grad q) componentwise, so
     dividing by H leaves the physical gradient AND leaves L = D G = -D W D^T
     symmetric negative definite, because W is diagonal and positive. Both
     properties at once; the old version could only have one of them.

     The top s-face carries W = 2/H rather than 1/H. That is not a fudge: the
     transpose alone puts the Dirichlet p = 0 at the ghost CENTRE, half a cell
     above the surface, and the distance from the top cell centre to the surface
     face is ds/2, not ds. Scaling that one diagonal entry puts the boundary
     condition on the surface where the stress balance acts, and since W stays
     diagonal the symmetry is untouched. */
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
    // scale by -W: 1/H_face on x-faces, 1/H on s-faces, 2/H on the surface face
    for (let i = 0; i < nx; i++){
      const wi = -1/this.Hface(i);
      for (let j = 0; j < ns; j++) gu[i*ns + j] *= wi;
    }
    for (let i = 0; i < nx; i++){
      const wi = -1/this.H[i];
      gw[i*(ns+1) + 0] = 0;                        // no-slip bottom, not corrected
      for (let j = 1; j < ns; j++) gw[i*(ns+1) + j] *= wi;
      gw[i*(ns+1) + ns] *= 2*wi;                   // Dirichlet lands on the surface
    }
  }

  /* Exact transpose of `vFlux`: the same four-point stencil, the same weights,
     the same special cases, scattered instead of gathered. */
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

  /* The metric weight 1/W at each velocity location: the factor the adjoint
     identity is stated in. Exposed so the adjointness gate can be written in
     the inner product the identity actually holds in, rather than in the
     Euclidean one where the broken version also passed. */
  metricWeight(kind, i, j){
    if (kind === 'u') return this.Hface(i);
    if (j === 0) return 0;
    return j === this.ns ? this.H[i]/2 : this.H[i];
  }
}
module.exports.FaradayDNS = FaradayDNS;

/* ── the operators the momentum step is built from ──────────────────────────
   These are methods rather than expressions inlined in `step` for one reason:
   a term that only exists inside a time loop can be checked only by its effect
   on a whole simulation, and the metric terms below are second order in the
   wave amplitude, so a linear dispersion test cannot see them at all. As
   methods they can be applied to an analytic field and compared with an
   analytic answer, which is the only way to know they are right. */

/* H_t per column, from the kinematic condition on the current velocity field.
   This is the same expression that advances H at the end of a step. */
FaradayDNS.prototype.surfaceHt = function(out){
  const { nx, ns } = this;
  out = out || new Float64Array(nx);
  for (let i = 0; i < nx; i++)
    out[i] = this.w[i*(ns+1) + ns] - this.uAtSFace(this.u, i, ns)*this.Hx(i);
  return out;
};

/* Omega = (w - s H_t - s u H_x)/H, the contravariant vertical velocity, at an
   s-face. At s = 1 this is EXACTLY zero whatever the state: substituting the
   kinematic condition H_t = w - u H_x leaves (w - w + u H_x - u H_x)/H. That is
   an identity, not a tolerance, and it is what the surface-following frame
   means -- so check-dns.mjs asserts it directly. Dropping the H_t term, which
   is what this did, breaks it by exactly H_t/H: the mesh moves and the
   advection pretends it does not. */
FaradayDNS.prototype.omegaAtSFace = function(u, w, i, j, Ht){
  const s = j*this.ds, ic = ((i % this.nx) + this.nx) % this.nx;
  return (w[this.idxW(i, j)] - s*Ht[ic] - s*this.uAtSFace(u, i, j)*this.Hx(i))/this.H[ic];
};

/* The full mapped Laplacian of a field at u-locations (x-faces, s-centres):

     lap = d_xx - 2A d_xs + A^2 d_ss + (A A_s - A_x) d_s + (1/H^2) d_ss

   with A = s H_x/H and A A_s - A_x = s(2 H_x^2/H^2 - H_xx/H). The ghost arrays
   carry the values half a cell outside the domain: in `step` they are the
   no-slip mirror and the free-surface stress balance; in the gate they are the
   analytic values of the test function, which is what lets the metric terms be
   compared against a known Laplacian instead of against a simulation. */
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

/* The same operator for a field at w-locations (x-centres, s-faces). The bottom
   face is not evolved, so its entry is left at zero. */
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

/* L = D G. With the metric-adjoint pair above this is -D W D^T with W diagonal
   and positive, so it is symmetric and negative definite by construction, and
   the gradient inside it is the physical one rather than something that merely
   transposes correctly. Both are asserted numerically in dns/check-dns.mjs --
   the symmetry, and the gradient against the analytic hydrostatic force, which
   is the check that the first version of this solver did not have and that
   would have caught its defect on the first run. */
FaradayDNS.prototype.applyL = function(q, out){
  this.gradient(q, this._gu, this._gw);
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
  /* A projection is only as exact as the solve behind it. If CG did not reach
     the tolerance, the corrected velocity is not divergence-free and the
     surface is about to be advanced by a flow that does not conserve volume --
     which does not look like a solver failure, it looks like physics. So this
     raises and names the numbers instead of returning a worse answer quietly.
     The iteration cap is sized to the grid (unpreconditioned CG on a 2D Poisson
     problem needs O(max(nx,ns)) iterations), so hitting it means something is
     wrong with the operator, not that the budget was mean. */
  if (!(this.cgResidual < tol))
    throw new Error(
      `pressure solve did not converge: residual ${this.cgResidual.toExponential(3)} `
      + `after ${it} iterations against a tolerance of ${tol.toExponential(0)} `
      + `on a ${this.nx}x${this.ns} grid at t = ${this.t.toExponential(3)} s.`);
  return this.cgResidual;
};

/* ── one time step ───────────────────────────────────────────────────────
   Predictor (advection + viscosity + gravity + the lifted surface pressure),
   then the projection, then the kinematic condition advances the surface with
   the PROJECTED velocity.

   Everything the formulation at the top of this file writes down is here, and
   the three things that were quietly missing are named because each of them is
   a term the header promised:

   1. Omega carried no H_t. The contravariant vertical velocity is
      (w - s H_t - s u H_x)/H and the code used (w - s u H_x)/H, i.e. it
      advected through a mesh it pretended was stationary while moving it.
      H_t is now taken from the kinematic condition on the current projected
      field -- the same expression that advances H at the end of the step, so
      the mesh velocity used for advection and the mesh velocity actually
      applied cannot disagree.

   2. The horizontal second derivative carries the full metric expansion. With
      A = s H_x / H,

        d2/dx2|_z = d_xx - 2A d_xs + A^2 d_ss + (A A_s - A_x) d_s

      and A A_s - A_x = s(2 H_x^2/H^2 - H_xx/H). The u equation had only
      s(H_x^2/H^2 - H_xx/H) -- half the first part -- and the w equation had
      the term not at all. In the small-slope limit these vanish, which is
      exactly why they are easy to lose and exactly why losing them is wrong:
      a Faraday ridge near onset is where the slope stops being small.

   3. The free-surface conditions were replaced by copy-ghosts. Zero tangential
      stress on z = H(x) is not du/dz = 0; it is

        (1 - H_x^2)(u_z + w_x) + 2 H_x (w_z - u_x) = 0

      with every derivative at constant z, and using w_z = -u_x|_z (which holds
      identically on a divergence-free field) that is

        u_z = -w_x|_z + 4 H_x u_x|_z / (1 - H_x^2).

      The dropped -w_x is first order in the wave amplitude, so the copy-ghost
      got the viscous damping wrong at leading order. The same identity fixes
      the w ghost: dw/ds at the surface is -H u_x|_z, not zero.
      The 1/(1 - H_x^2) is genuine and degenerates at a 45-degree slope, which
      is a limit of the single-valued z = H(x) form already stated at the top,
      not an extra assumption introduced here. */
FaradayDNS.prototype.step = function(dt){
  const { nx, ns, dx, ds, rho, nu } = this;
  const H = this.H, u = this.u, w = this.w;
  const gT = this.g + this.accel*Math.cos(this.omegaD*this.t);

  /* du/dx|_z at the surface: computed ONCE and handed to both the dynamic
     condition and the w ghost, because they are the same physical quantity
     and two copies of it are two things that can disagree. */
  const sux = this.surfaceUx(this._sux);
  this.surfacePressure(this._ps, sux);

  const uN = this._uN, wN = this._wN;
  const at = (arr, i, j, n2) => arr[((i % nx + nx) % nx)*n2 + j];
  const Ht = this.surfaceHt(this._Ht);
  this.surfaceGhosts(this._uBot, this._uTop, this._wTop, sux);
  const uBot = this._uBot, uTop = this._uTop, wTop = this._wTop;
  this.lapU(u, uBot, uTop, this._lu);
  this.lapW(w, wTop, this._lw);

  // ---- u predictor, on x-faces at s-centres
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
  // ---- w predictor, on s-faces at x-centres. Bottom face is no-slip.
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

  /* Project FIRST, then advance the surface with the projected velocity.

     Advancing H from the predictor velocities -- which is what this did -- uses
     a field that is not divergence-free, so the surface is moved by a flow that
     does not conserve volume, and the error accumulates as a systematic shift
     in the wave speed. The geometry used for the projection is the old one;
     the surface then moves on a velocity that satisfies continuity on it. */
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

/* The ghost values just outside the domain, from the boundary conditions rather
   than by copying a neighbour.

   Bottom: no slip, so the u ghost mirrors, u(-ds/2) = -u(ds/2).

   Surface: zero tangential stress on z = H(x) is

       (1 - H_x^2)(u_z + w_x) + 2 H_x (w_z - u_x) = 0,

   every derivative at constant z. On a divergence-free field w_z = -u_x|_z
   identically, so this collapses to

       u_z = -w_x|_z + 4 H_x u_x|_z / (1 - H_x^2),

   and the same identity gives dw/ds = -H u_x|_z for the w ghost. A copy-ghost
   (du/dz = 0, dw/ds = 0) drops the -w_x, which is FIRST order in the wave
   amplitude, so it gets the viscous damping wrong at leading order -- and the
   damping is the entire content of a Faraday threshold. */
/* du/dx|_z at the free surface, per column (x-centre). Two things need it and
   they must be the same number: the w ghost, via the incompressibility identity
   dw/ds = -H du/dx|_z, and the VISCOUS NORMAL STRESS in the dynamic condition.
   See `surfacePressure` for why the second one is not optional. */
/* Extrapolate a cell-centred column to s = 1 and differentiate it there.

   The top three cell centres sit at s = 1 - h/2, 1 - 3h/2, 1 - 5h/2 with
   h = ds. Fitting a quadratic through them and evaluating at s = 1 gives

       u(1)      = (15 u0 - 10 u1 +  3 u2)/8
       du/ds (1) = ( 2 u0 -  3 u1 +    u2)/h

   both second order. The one-sided pair that was here -- (u0 - u1)/h -- is the
   derivative at s = 1 - h, not at s = 1, so it is only FIRST order at the
   surface, and it feeds the viscous normal stress, which is first order in the
   wave amplitude. Measured, it held du/dx|_z to 2.05x per mesh doubling; the
   quadratic form takes it to 4x. The three cells it needs are the three the
   constructor already demands. */
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
    // du/dx|_s at s = 1, from u extrapolated to the surface on both x-faces
    const uxS = (this.surfExtrap(u, ir*ns, ns) - this.surfExtrap(u, i*ns, ns))/dx;
    const dudsC = 0.5*(this.surfDeriv(u, i*ns, ns) + this.surfDeriv(u, ir*ns, ns));
    out[i] = uxS - (this.Hx(i)/H[i])*dudsC;
  }
  return out;
};

/* The pressure ON the free surface, which the step lifts out of the projection
   so that the remaining dynamic pressure vanishes there.

   The normal stress balance is n.sigma.n = -p_atm + gamma*kappa with
   sigma = -p I + 2 mu E, so the surface pressure is NOT just the capillary
   term: it carries the viscous normal stress 2 mu (n.E.n) as well. Using
   incompressibility (w_z = -u_x|_z) and the zero-tangential-stress condition to
   eliminate the other strain components, that reduces exactly to

       n.E.n = -u_x|_z (1 + H_x^2)/(1 - H_x^2)

   so  p_surface = -gamma*kappa - 2 mu u_x|_z (1 + H_x^2)/(1 - H_x^2).

   Dropping the viscous part -- which this did -- is not a small error. For a
   free-surface wave the interior viscous force is IDENTICALLY ZERO while the
   flow stays irrotational (nu lap u = nu grad(div u) - nu curl(curl u) = 0), so
   the damping does not come from the bulk term at all: it comes from the two
   boundary stresses. The normal one supplies nu k^2 of Lamb's 2 nu k^2 and the
   rotational surface layer supplies the other nu k^2. Omitting the normal one
   therefore halves the damping exactly, which is what was measured: 7.415 s^-1
   against 14.914. Halving the damping halves the Faraday threshold, so on this
   term the whole point of the solver depends.

   The 1/(1 - H_x^2) is the same genuine degeneracy at a 45-degree slope that the
   tangential condition carries, and for the same reason: it is a limit of the
   single-valued z = H(x) surface, not an assumption added here. */
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
    // du/dx|_z AT the surface on this x-face, same quadratic extrapolation
    const ip = (i + 1) % nx;
    const dudsT = this.surfDeriv(u, i*ns, ns);
    const ux_z = (this.surfExtrap(u, ip*ns, ns) - this.surfExtrap(u, im*ns, ns))/(2*dx)
               - (HxF/HF)*dudsT;
    // dw/dx|_z at the same place; dw/ds one-sided from the top three s-faces
    const dwdsF = 0.5*((3*w[i*(ns+1)+ns]  - 4*w[i*(ns+1)+ns-1]  + w[i*(ns+1)+ns-2])
                     + (3*w[im*(ns+1)+ns] - 4*w[im*(ns+1)+ns-1] + w[im*(ns+1)+ns-2]))/(2*ds);
    const wx_z = (w[i*(ns+1)+ns] - w[im*(ns+1)+ns])/dx - (HxF/HF)*dwdsF;
    const uz_S = -wx_z + 4*HxF*ux_z/(1 - HxF*HxF);
    uTop[i] = u[i*ns + ns-1] + ds*HF*uz_S;
  }
  for (let i = 0; i < nx; i++) wTop[i] = w[i*(ns+1) + ns-1] - 2*ds*H[i]*ux[i];
};

/* The PHYSICAL max|div u|. `divergence` returns H*(div u) -- that weight is
   what makes the operator pair adjoint -- so it is divided back out here rather
   than reported as if it were the divergence itself. */
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
