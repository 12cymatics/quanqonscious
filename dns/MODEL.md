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
