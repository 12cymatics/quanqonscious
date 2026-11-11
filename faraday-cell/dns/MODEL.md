# Nonlinear Faraday Cell: equations and supported physical problem

The requested production problem uses a **pinned contact line**, a rigid circular
cylinder, and a single incompressible Newtonian liquid with constant density,
viscosity and surface tension. The interior free surface evolves nonlinearly.
The reference experiment is optional comparison data, not a source of forcing,
fluid properties, geometry, pressure, or the nonlinear surface field.

This specification describes the continuum equations. The JavaScript and C++
implementations solve a finite-grid numerical discretization of them; their
floating-point results are not exact continuum solutions. Resolution, time-step,
pressure residual, physical time and boundary-condition tests remain part of the
meaning of every result.

## Variables and domain

Use cylindrical coordinates `(r, theta, z)` in the translating container frame:

```
0 <= r < R, theta periodic over 2*pi, 0 < z < H(r,theta,t)
H = h + eta > 0
u = (ur, utheta, uz)
rho > 0                     density [kg/m^3]
nu > 0, mu = rho*nu          kinematic/dynamic viscosity [m^2/s, Pa*s]
gamma > 0                   surface tension [N/m]
g >= 0                      gravitational acceleration [m/s^2]
G(t) = g + a*cos(omegaD*t)   effective downward acceleration [m/s^2]
omegaD = 2*pi*f              drive angular frequency [rad/s]
```

`R`, `h`, `rho`, `nu`, `gamma`, `g`, `a` and `f` are specified physical inputs.
Example inputs are examples, not measurements or a shaker calibration. No
apparatus-voltage conversion is needed in the general experiment path.

## Complete bulk equations for this model

Mass conservation and vector momentum are

```
div(u) = 0
rho * (partial_t u + (u dot grad)u)
    = -grad(p) + mu*laplacian(u) - rho*G(t)*ez
```

For a scalar `v`, define

```
L(v) = v_rr + v_r/r + v_theta_theta/r^2 + v_zz.
```

The cylindrical components retain the geometric terms:

```
ur_r + ur/r + utheta_theta/r + uz_z = 0

ur_t + ur*ur_r + utheta*ur_theta/r + uz*ur_z - utheta^2/r
    = -p_r/rho + nu*(L(ur) - ur/r^2 - 2*utheta_theta/r^2)

utheta_t + ur*utheta_r + utheta*utheta_theta/r + uz*utheta_z
    + ur*utheta/r
    = -p_theta/(rho*r) + nu*(L(utheta) - utheta/r^2 + 2*ur_theta/r^2)

uz_t + ur*uz_r + utheta*uz_theta/r + uz*uz_z
    = -p_z/rho + nu*L(uz) - G(t).
```

There is no small-amplitude deletion of advection, no one-azimuthal-mode
restriction in nonlinear evolution, and no linearized replacement for the
surface curvature. The separate modal visualization and Floquet calculation
answer different questions and are not substitutes for this solver.

## Free surface: geometry, kinematics and stress

At `z = H`, let

```
s = sqrt(1 + eta_r^2 + (eta_theta/r)^2)
n = (-eta_r, -eta_theta/r, 1)/s
C = (1/r)*partial_r(r*eta_r/s)
    + (1/r^2)*partial_theta(eta_theta/s)
D = (grad(u) + transpose(grad(u)))/2
T = -p*I + 2*mu*D.
```

`C` is the divergence of the horizontal surface slope divided by `s`. It is the
negative of `div(n)` under the outward-normal convention above. Declaring this
sign is necessary to interpret capillary pressure correctly.

The full kinematic boundary condition is

```
eta_t + ur_s*eta_r + (utheta_s/r)*eta_theta = uz_s.
```

The exterior has uniform pressure `p_atm` and zero viscous shear in this
single-liquid model. The stress conditions are

```
(I - n*n^T)*T*n = 0
n^T*T*n = -p_atm + gamma*C
p_s - p_atm = 2*mu*n^T*D*n - gamma*C.
```

Both tangential traction components and the normal viscous traction are required.
Surface tension is constant, so this model has no Marangoni stress term. A
variable-tension or two-fluid problem would require its additional constitutive
and interface equations rather than being inferred from this one.

## Solid boundary, rim and axis

The floor and vertical wall impose no slip and no penetration: `u = 0` at
`z = 0` and `r = R`. The pinned contact line imposes

```
eta(R,theta,t) = 0
eta_t(R,theta,t) = 0.
```

The pinned height is relative to the specified reference level `h`. Pinning does
not impose a 90-degree contact angle and does not freeze the interior surface.
The axis uses regular cylindrical fields and zero inner radial face area;
azimuthal boundaries are periodic.

The historical `free` option extrapolates a moving rim while retaining no-slip
wall velocities. It does not supply a moving-contact-line closure. Production
nonlinear runs therefore require the user-selected pinned condition; no clamp
of the free rim is used to disguise this inconsistency. Historical free-slope
operator tests remain diagnostic evidence, not validation of that moving
contact-line model.

## Pressure representation and the corrected boundary operation

The stored pressure is a reduced pressure, not absolute pressure:

```
q = p - p_atm + rho*G(t)*(z-h)
q_s = rho*G(t)*eta - gamma*C + 2*mu*n^T*D*n.
```

This change of pressure variable absorbs the spatially uniform body force; it
does not delete gravity or the drive. Recover physical pressure using the first
identity. The solver's surface-following coordinate is `z = sigma*H`, with

```
Omega = uz - sigma*(ur*H_r + (utheta/r)*H_theta)
r*H*div(u) = partial_r(r*H*ur) + partial_theta(H*utheta)
             + partial_sigma(r*Omega)
eta_t = Omega at sigma=1.
```

The pressure gradient must include the prescribed surface pressure **before**
the slope-transpose operation. Adding it to the vertical component afterwards
omits horizontal terms on a deformed surface. The corrected projection uses
the same affine operator `G(q,q_s)` in its right-hand side and correction:

```
D*G(q,0) = (rho/dt)*D*u_star - D*G(0,q_s)
u_new = u_star - (dt/rho)*G(q,q_s).
```

Here `D` in the projection identities denotes the implemented divergence,
not the strain-rate tensor. A matching constant interior and boundary pressure
must produce zero pressure gradient even on a sloping surface. Adding the same
constant to both pressures must leave the evolution unchanged. These are
independent regression checks of the corrected boundary operation.

## Initial conditions and numerical evidence

An initial surface and divergence-compatible initial velocity must be supplied.
The interface and terminal runner use a visible small Bessel perturbation obeying
the selected pinned rim, with initially zero velocity. The seed is an initial
condition, not a restriction of subsequent nonlinear mode coupling or a shape
copied from the reference experiment.

The browser worker and terminal runner recompute their time-step limits from the evolving depth and
velocity after each completed step, including when using WebAssembly or worker
threads. The terminal runner's last step ends at the requested physical duration. These stability
estimates do not replace time-step convergence checks.

Negative layer depth, failed pressure convergence, invalid input, unavailable
backend and an infeasible requested grid are failures. They must not yield
clipped states, a cheaper solver, or a modal image presented as a nonlinear
result. Image colour limits and interpolation into display pixels do not modify
the solver state. A saved nonlinear image uses the actual completed solver
surface, not a newly generated modal approximation.

Required evidence includes constant-pressure invariance, manufactured operator
checks, pinned boundary compatibility, nonlinear harmonic generation, mass and
energy/balance diagnostics, spatial/time convergence, non-paper SI parameter
sets, JavaScript/WebAssembly consistency, and browser/export behavior. Passing
a finite set of checks is not proof of correctness for every input or independent
experimental validation.

## Corrected book arithmetic

The sutra source is Bharati Krishna Tirthaji,
[*Vedic Mathematics*](https://archive.org/details/vedic-mathematics-bharati-krishna-tirth-ji-maharaj), revised
edition Delhi 1992, the 2010/2012 reprint identified by the project's book
register. Printed page numbers, not PDF page numbers, apply. The current
`vedic_book.py` and `docs/TRUE_SUTRA_REGISTER.md` contain the page-specific
procedures; no older invented correspondence is substituted for them.

The corrected procedures include the constant remainder on printed page 19,
unchanged remainder after divisor normalization on page 57, and the deficiency
cube on pages 288–289. The latter is a cube procedure, not a renamed square.
These corrections are recorded in commit
`aee61f40a967ea08bf5e8879edd1da9c9e47d624` and
[Claude's PR #172](https://github.com/12cymatics/quanqonscious/pull/172) and are independently
checked in this work.

Exact coefficient convolution and differentiation from those book procedures
are used to derive manufactured pressure/coordinate identities in
`check-book-pressure.py`. They are algebraic verification of the numerical
implementation; the Navier–Stokes momentum law, stress balance and pressure
projection are not relabelled as sutras. A book arithmetic identity is not by
itself a derivation or empirical validation of fluid dynamics.

## Free-surface geometry by the book's procedures

`check-book-surface.py` applies the book to the geometry of the free surface.
For a polynomial surface `z = h + eta(x, y)` the slopes, the squared slope
`q = eta_x^2 + eta_y^2`, the unnormalised normal `(-eta_x, -eta_y, 1)` and the
numerator of the mean curvature are polynomials, and it derives them with three
procedures as the pinned `vedic_book.py` implements them:

| Procedure | Printed pages | Use |
| --- | --- | --- |
| Calana-Kalanabhyam | 145, 175-178 | every derivative of `eta` |
| Urdhva-Tiryagbhyam | 33-36 | every product of two polynomials |
| Gunita-samuccaya | 76 | the test of each product: the sum of its coefficients is the product of the factors' sums |

With `s = sqrt(1 + q)` the curvature `C` of this document is `N / s^3`, where

```
N = (1 + eta_y^2)*eta_xx - 2*eta_x*eta_y*eta_xy + (1 + eta_x^2)*eta_yy.
```

`N` is derived a second way, from the divergence,
`(1 + q)*(eta_xx + eta_yy) - (eta_x*q_x + eta_y*q_y)/2`, and the two must be the
same polynomial. The cylindrical slopes come from `r*eta_r = x*eta_x + y*eta_y`
and `eta_theta = x*eta_y - y*eta_x`, and `(r*eta_r)^2 + eta_theta^2 = r^2*q` is
required as a polynomial identity. sympy then differentiates the formulas of the
free-surface section above, in cylindrical coordinates and with the square root
in place, and must arrive at the same `N`.

### What is held to it

`book-surface-probe.mjs` gives `FaradayCell3D` each surface, sampled at its own
cell centres and correctly rounded to doubles, and returns the solver's
`curvature()`, `surfaceSlopeSquared()` and `surfaceNormal()`. These are compared
with the exact values at the same cell centres.

No comparison is made in floating point. A double is an exact rational. The
cell-centre angles `pi*(2k + 1)/nth`, their cosines and sines, and `s` are not
rational, so each is carried as an interval with integer ends at `2^-320`,
rounded outward at every operation: `pi` from Machin's alternating series,
cosine and sine from their Taylor series with the remainder added, the root from
the integer square root. Every claim is decided on the ends of those intervals.
It is true of the real numbers they enclose, or the check fails.

The solver is a second-order discretisation in doubles, so it is not asked to
equal the exact value. It is asked to converge on it. The measure is the
area-weighted root mean square of the error over a region, relative to that of
the exact quantity. The inner region is `r <= 9R/10`; the rim is beyond it. The
cut is a physical radius, so every grid measures the same region.

Two surfaces are used, on `R = 1/64 m`, `h = 1/32 m`, both with zero elevation
and zero radial slope at the rim, so each meets the free and the pinned contact
condition:

```
cap     eta = (R/2)*(1 - rho^2/R^2)^2                               slope up to 0.77
lobed   eta = (R/2)*(1 - rho^2/R^2)^2 * (1 + 2*(x^2 - y^2)/R^2)     slope up to 1.11
```

### What was measured

Grids `16x24`, `32x48`, `64x96`, `128x192` (radial by azimuthal). Relative error
on each, and the fall per halving that the check requires (second order is 4):

| Quantity, region, contact | cap | lobed | Required |
| --- | --- | --- | --- |
| curvature, inner, either | 2.38e-2, 6.18e-3, 1.56e-3, 3.92e-4 | 3.91e-2, 1.06e-2, 2.72e-3, 6.85e-4 | 7/2 |
| squared slope, inner, either | 3.00e-2, 7.63e-3, 1.92e-3, 4.80e-4 | 3.96e-2, 1.02e-2, 2.56e-3, 6.40e-4 | 7/2 |
| radial slope of the normal, inner, either | 1.22e-2, 3.09e-3, 7.75e-4, 1.94e-4 | 1.87e-2, 4.76e-3, 1.19e-3, 2.99e-4 | 7/2 |
| azimuthal slope of the normal, inner and rim, either | exactly zero | 4.51e-2, 1.14e-2, 2.85e-3, 7.14e-4 | 7/2 |
| squared slope, rim, either | 5.08e-2, 1.26e-2, 3.13e-3, 7.82e-4 | 3.23e-2, 8.10e-3, 2.02e-3, 5.06e-4 | 7/2 |
| radial slope of the normal, rim, free | 1.81e-2, 4.60e-3, 1.16e-3, 2.90e-4 | 1.40e-2, 3.62e-3, 9.16e-4, 2.30e-4 | 7/2 |
| radial slope of the normal, rim, pinned | 2.18e-2, 5.85e-3, 1.65e-3, 4.91e-4 | 1.93e-2, 5.44e-3, 1.61e-3, 5.06e-4 | 2 |
| curvature, rim, free | 3.64e-2, 1.23e-2, 4.23e-3, 1.47e-3 | 3.91e-2, 1.39e-2, 4.86e-3, 1.71e-3 | 2 |
| curvature, rim, pinned | 8.83e-2, 7.53e-2, 5.76e-2, 4.22e-2 | 1.09e-1, 9.01e-2, 6.87e-2, 5.03e-2 | any fall |

That is 88 claims decided, and two references the same decision refuses on each
surface and contact, eight in all: the linearised curvature `eta_xx + eta_yy`
(error 26% and 35%, not falling) and the curvature of a surface one per cent
taller (error near 1%, not falling at the required rate).

### The pinned rim row does not converge

Under the pinned contact condition the curvature in the outermost row of cells
does not approach the exact value on these surfaces. The check prints the
relative error of that row alone on the four grids:

```
          pinned                        free
cap       0.181, 0.217, 0.234, 0.242    0.0733, 0.0349, 0.0170, 0.00839
lobed     0.190, 0.220, 0.235, 0.243    0.0672, 0.0334, 0.0165, 0.00821
```

Pinned, it grows toward about a quarter instead of falling. Free, it halves
with the cell. The cause is the pinned rim closure: the slope at the wall face
is taken as the two-point difference between the last cell centre and the
wall, which is the slope a quarter of a cell inside the wall. On a surface
whose rim slope is zero that is a slope error of first order in the cell
width, and the curvature divides it by the cell width. The area-weighted error
over the rim region still falls, because the row is one cell wide, which is
why the table above requires only a fall there.

`check-cell3d.mjs` section 7d bounds the pinned rim at nine per cent on its own
surface, `eta = A*(1 - x^2)`, where the solver's source records 0.013, 0.016
and 0.017 for this row. That bound does not hold on a surface with zero rim
slope.

This check verifies the solver's surface geometry against an exact derivation.
It does not test the fluid equations, the pressure solve, time stepping or
agreement with an experiment.
