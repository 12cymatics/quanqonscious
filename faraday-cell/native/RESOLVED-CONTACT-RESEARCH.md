# Resolving liquid gaps around physical grains

Research and implementation note, updated 2026-11-22. Exact positive-gap geometry and
its verification are implemented in `native/resolved-gap.hpp` and
`native/check-resolved-gap.cpp`, runnable with
`node native/check-resolved-gap.mjs`. A fixed-geometry three-dimensional
Navier–Stokes reference patch is now implemented in
`native/resolved-gap-flow.hpp` and `native/resolved-gap-flow.cpp`, verified by
`node native/check-resolved-gap-flow.mjs`. Physical mixed-boundary and SI-unit
verification is in `node native/check-resolved-gap-boundary.mjs`.
The patch has not been joined to the
vessel mesh or moving-grain equations; it does not implement a depositing bed.

## What the current refusal means

`native/rigid-coupler.cpp` computes sphere/floor, sphere/cylinder and sphere/sphere
distances. A positive gap smaller than `resolvedGapCells * grainSpacing` is
refused, including a swept-gap check during motion. Exactly touching geometry
uses the separate contact-impulse system. The fluid calculation supplies no
empirical drag or lubrication correction.

`native/rigid-interpolation.cpp` derives one scalar spacing from the largest
spacing of independent velocity nodes inside the grain, and uses it to build
isotropic volume-marker supports. This is a conservative admission rule, not a
measurement of the normal and tangential resolution of the liquid gap. It also
does not certify that interpolation supports remain distinct across two nearby
solid boundaries. Merely reducing the gap threshold would not resolve the
missing pressure and shear gradients.

There are two different problems to fix:

1. Numerical resolution and time integration must resolve the positive-gap
   fluid without moving a grain across an unresolved region.
2. The physical contact model must describe how the specified real surfaces
   can contact. Infinite refinement of an inappropriate smooth-surface model
   cannot supply the missing material physics.

## Physical model: a smooth sphere is not a sand grain

For the particular problem of a smooth rigid sphere falling toward a smooth
wall in constant-density incompressible viscous liquid, no-slip conditions
lead to a no-collision result in finite time. Slip changes that result under
the conditions analyzed by Gérard-Varet, Hillairet and Wang. This does not
prove a universal statement about every driven many-grain/free-surface
configuration; it does demonstrate why the current canonical sphere-wall
model cannot be made into ordinary deposition just by finding a faster
integrator. [Primary analysis, introduction and theorems](https://arxiv.org/html/1302.7098v1).

Roughness must be specified as geometry or as an independently justified wall
law. Adding smooth bumps is not, by itself, a general proof of finite-time
contact: the small-gap boundary behavior and surface regularity matter.
Gérard-Varet and Hillairet analyze roughness, reduced regularity and slip, and
identify distinct drag asymptotics. Their results support investigating the
actual near-contact geometry; they do not give permission to invent a universal
roughness cutoff. [Primary roughness analysis](https://www.numdam.org/item/10.1051/m2an/2012001.pdf).

For the intended beach-sand calculation, the remaining physical inputs are
explicit grain shape/size geometry, near-contact surface topography or a
measured interfacial law, solid deformation/contact parameters where relevant,
and the vessel's corresponding surface properties. Density alone does not
provide those. An illustrative geometry can be used for numerical verification
if its provenance is explicit; it must not be described as a measured south-east
Queensland sample. A measured Navier slip law would change a boundary condition
of the resolved PDE, but it is not currently specified and must not silently
replace no slip.

The commonly cited Costa et al. collision method includes a subgrid lubrication
correction, a roughness-scale saturation and a soft-sphere collision duration.
Those are explicitly documented choices in that work, but they do not satisfy
the user's stricter request. The title's term “fully resolved” is not enough to
establish compatibility. [Primary method description](https://arxiv.org/abs/1506.01880).

## Supported numerical routes

Dynamic local refinement of a constrained fluid solve is a demonstrated route.
Selçuk, Ghigo, Popinet and Wachs extend a distributed-Lagrange-multiplier /
fictitious-domain method to adaptive quad/octrees and report direct calculation
of lubrication interactions without an ad-hoc force correction. This establishes
that the positive-gap resolution problem is solvable numerically; it does not
make zero-gap deposition automatic. The accessible abstract and section outline
were checked; the HAL full-text endpoint denied automated access during this
research. [Primary paper](https://doi.org/10.1016/j.jcp.2020.109954).

An unsplit velocity/pressure/rigidity solve can avoid the artificial leakage of
loosely split constraint methods. Kallemov et al. describe such a constrained
immersed-boundary method and a Schur-complement preconditioner with multigrid
fluid solves. Their observation about marker spacing is a conditioning result
for that discretization, not evidence that any eight-cell grain is physically
resolved. [Author-hosted primary paper](https://math.nyu.edu/inmemoriam/donev/CFD/RigidIBAMR.pdf).

IBAMR has adaptive Cartesian meshes and a concrete rigid-body constraint
implementation. It is useful for comparison and operator design. Adopting it
would still require the vessel/free-surface problem, boundary conditions,
geometry and force verification to be implemented and tested; it is not a
drop-in completion of this application. [Official CIB implementation interface](https://ibamr.github.io/IBAMR-docs/ibamr/classIBAMR_1_1CIBMethod.html).

## Contained implementation for the existing solver

The following architecture is derived from the exact geometry and conservation
identities. The geometry and standalone patch described below are implemented;
the global vessel joining work remains to be done.

The existing solver is a fixed polar tensor grid with a surface-following
vertical coordinate. Its pressure storage and rigid interpolation are not
octree operators. Replacing the gap test with an octree tag, while leaving
those operators unchanged, would be ineffective. A contained extension is a
locally fitted three-dimensional liquid patch coupled conservatively to the
existing vessel grid. Both regions must solve the same incompressible
Navier–Stokes equations. No algebraic lubrication force is added.

### First component: exact gap geometry and metric operators

For a sphere of radius a above a plane, choose local coordinates with the plane
at z=0 and the closest sphere surface at z=g, with g>0. For r²=x²+y²<a²:

```
z_minus = 0
z_plus  = g + r² / (a + sqrt(a²-r²))
H       = z_plus-z_minus
```

The rational expression evaluates the exact spherical sag without subtracting
nearly equal radii. It is not a parabolic lubrication approximation.

For two spheres of radii a and b whose closest surface separation is g:

```
z_minus = -r² / (a + sqrt(a²-r²))
z_plus  = g + r² / (b + sqrt(b²-r²))
H       = z_plus-z_minus
```

Here the frame's z-axis joins the centers, and the chart is confined to
r<min(a,b). Rotation and translation into world coordinates are explicit.
The patch can be meshed with x=xi, y=eta and

```
z(xi,eta,sigma,t) = z_minus(xi,eta,t) + sigma*H(xi,eta,t)
det(J) = H > 0
partial_z = (1/H)*partial_sigma
partial_x = partial_xi - (z_xi/H)*partial_sigma
partial_y = partial_eta - (z_eta/H)*partial_sigma
```

The new component implements the geometry, positive-Jacobian check, physical
surface normals, Jacobian and cofactor vectors, inverse coordinates, and
transformed gradient/divergence. Its independent tests use Cartesian polynomial
fields, rotated/translated configurations, implicit sphere equations, finite
differences of mapped coordinates and fluxes, and an independently evaluated
volume antiderivative. The surface-flux integral converges to the volume
integral under quadrature refinement. This component does not calculate
hydrodynamic forces or authorize crossing the existing refusal boundary.

For a stationary chart, exact incompressibility becomes

```
partial_xi(H*u) + partial_eta(H*v)
  + partial_sigma(w-u*z_xi-v*z_eta) = 0.
```

For a moving graph with stationary horizontal axes the relative sigma flux is
`w-u*z_xi-v*z_eta-z_t`. For general translated/rotated charts the implemented
relative flux is the cofactor projection of `velocity-gridVelocity`. The grid
velocity includes frame translation, frame angular velocity and prescribed
gap rate. The tests independently differentiate that flux and verify that
its coordinate divergence equals the Jacobian time derivative (gap rate),
including simultaneous translation and rotation. Comoving fluid has exactly
zero relative flux. This is a metric/geometric-conservation check, not a
completed constant-field transport solve or freely moving grain calculation.

### Second component: solve and join the actual liquid domains

#### Implemented fixed-geometry full Navier–Stokes patch

The new reference patch uses continuous tensor-product Q2 velocity and Q1
pressure on three-dimensional hexahedral elements mapped by the exact gap
chart. The polynomial pair is the standard Taylor–Hood choice documented in
the [official deal.II mixed-element tutorial](https://dealii.org/current/doxygen/deal.II/step_22.html).
That reference is not a proof of robust conditioning on arbitrarily thin,
curved gaps; the supplied refinement results apply to the tested meshes.
The patch assembles physical Cartesian gradients, the complete symmetric
viscous stress, conservative nonlinear momentum convection, and an optional
backward-Euler acceleration term. The steady and transient cases use the same
nonlinear solver. Density and viscosity must be positive; setting density to
zero to substitute a Stokes problem is refused.

For velocity test functions v, its residual is

```
R_u(v) = integral_Omega [rho*(u-u_old)/dt . v
            - rho*(u tensor u) : grad(v)
            + 2*mu*D(u):D(v) - p*div(v) - f.v] dV
          + integral_boundary rho*u*(u.n).v dA
          - integral_traction_boundary prescribed_traction.v dA
R_p(q) = integral_Omega q*div(u) dV
```

The time term is omitted only when the caller explicitly requests a steady
problem. The complete derivative of the nonlinear convective residual is
assembled in Newton's Jacobian. The divergence and pressure-gradient blocks
use the same quadrature contributions with opposite transpose signs. The
pressure-pressure block is zero: no pressure penalty, added viscosity or
force correction is supplied.

The standalone solve supports physical velocity or prescribed physical traction
`sigma*n` on each of the six faces. At a shared edge a velocity condition takes
precedence for that velocity node. With velocity on all faces, one pressure DOF
fixes the gauge; **every** continuity equation, including the omitted gauge row,
is verified before an accepted state is returned. Incompatible boundary flux
is refused. With any traction face, the applied traction fixes absolute pressure,
so all pressure DOFs and continuity equations are retained and no gauge is pinned.
Initial unknown velocities are zero in the verification cases; the manufactured
interior solution is not injected into the unknowns. The **uneliminated** `linearize` API retains all
boundary velocity DOFs and all pressure DOFs for a subsequent global interface
solve. It must receive the actual boundary/interface equations before being
used as a complete coupled problem.

The boundary API provides physical normals, area weights, normal velocity,
pressure/viscous traction `sigma*n`, advective momentum flux
`rho*u*(u.n)`, and total outward momentum flux
`rho*u*(u.n)-sigma*n`. Weak boundary reaction loads are the assembled nodal
momentum residuals. Their sum balances the body force, acceleration and actual
boundary momentum flux to the nonlinear solve tolerance. This conservation
statement does not depend on claiming that a Q2 velocity is divergence free
pointwise. The measured pointwise divergence error is reported separately.

Momentum and incompressibility residuals have different physical units. Newton
acceptance and the dense linear solve therefore use dimensionless equations:
with `L=halfWidth`, the explicit numerical reference speed `U=referenceVelocity`,
and `P=rho*U^2+mu*U/L`, the momentum row scale is `P*L^2` and the continuity row
scale is `U*L^2`. Velocity and pressure increments are scaled by `U` and `P`.
The same physical equations and output units are retained; the reference speed
does not prescribe a physical velocity. This fixes the earlier unit-sized
absolute force tolerance that could accept a zero interior initial velocity
before solving a micron-scale SI problem. Reported Newton residual norms are
dimensionless; `maximumContinuityMoment` remains a physical volume/time moment.

Uniform gravity requires another exact rearrangement because physical `z` is
not represented by the mapped Q1 pressure basis on the exact spherical chart.
The caller can supply `pressureGradientOffset=G` and write
`p_total=p_Q1+G.x`. The assembled force is the supplied **physical** `force-G`,
while `G.x*n` is subtracted from every boundary residual. On open faces this
transforms the supplied physical traction; on velocity faces it retains the
actual physical wall/grain reaction. Fields and traction samples reconstruct
total pressure and its gradient. The reduced pressure is interpolated directly
in the solve, avoiding subtraction of two large hydrostatic pressures.
This changes no continuum force or equation and supplies no hydrodynamic force
law. The `State.pressure` coefficients are reduced pressure when an offset is
present; consumers must use reconstructed physical pressure for output/loads.

No external algebra library was installed or vendored. The portable pivoted
dense solver is a **reference-size implementation**, with explicit unknown and
matrix-memory limits checked before allocating a requested solve. Exceeding
the limit throws; it does not coarsen, reduce grain count or change physics.
Production refinement requires a sparse global mixed solve/preconditioner.
Newton line search changes numerical iterates, not accepted physical velocity
or gap values; failed convergence returns no accepted approximate state.

#### Independent verification results

The supplied test uses an independently differentiated three-dimensional
Cartesian polynomial curl velocity, a nonconstant pressure, and nonzero body
forcing containing the complete nonlinear advection. Density is 2 and dynamic
viscosity 0.13 in this verification problem; these are mathematical test inputs,
not measurements of water or Queensland sand. Errors are evaluated with a
different volume quadrature rule from the assembly rule. Six-point Gauss
quadrature in each element direction is checked on the tested curved meshes;
polynomial basis degree alone is not sufficient to choose quadrature for
nonpolynomial sphere geometry. The small measured hydrostatic error is finite
quadrature error, not an exact well-balanced identity for arbitrary geometry.

| Elements per axis | Mixed interior unknowns | Velocity RMS error | Pressure RMS error | Traction RMS error | Divergence RMS error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 107 | 1.385e-3 | 3.763e-2 | 4.026e-2 | 2.215e-2 |
| 3 | 438 | 4.545e-4 | 1.219e-2 | 1.353e-2 | 1.039e-2 |
| 4 | 1,153 | 1.985e-4 | 8.418e-3 | 9.175e-3 | 6.033e-3 |

The nonlinear manufactured advection RMS is 2.317e-2. Each mesh takes three
Newton steps from zero interior velocity; final dimensionless algebraic residuals
are below 9e-13. The weak integral momentum defect is below 3e-14 and net boundary flux
is below 8e-17; all pressure-test continuity moments are below 8e-18. These
values verify the standalone discrete equations; the
finite spatial errors above must not be described as zero or as a converged
grain-force prediction.

A nonzero-acceleration backward-Euler case also converges, with velocity RMS
error 4.541e-4 on the three-element mesh. A stationary no-slip pressure/body-force
balance checks the same full solver at rest, with maximum velocity 3.11e-12
and maximum pressure error 3.70e-11. Additional checks differentiate
the unrestricted nonlinear residual numerically, confirm `G=-B^T` exactly,
verify a zero pressure penalty block, check the pressure contribution to actual
boundary traction, and refuse missing previous states or excessive resources.

The mathematical manufactured boundary data are verification inputs. They do
not represent a validated vibrating grain configuration. The standalone patch
currently has fixed geometry; ALE transport, changing grain frames, free
surfaces, interface joining and contact/deposition are not enabled by this test.

#### Physical rotating-cap and SI-unit verification

`check-resolved-gap-boundary` also solves a physical boundary-value problem with
**zero body forcing**: a sphere rotates about the axis normal to a stationary
wall, with no slip on both surfaces and zero physical traction on the four open
lateral faces. Rotation preserves the exact spherical geometry. This is an
actual steady full Navier–Stokes calculation with nonzero nonlinear momentum
transport, rather than a prescribed manufactured interior solution or a moving
geometry approximation. The reference problem uses radius 1, positive gap .05,
square half-width .4, density 1, dynamic viscosity .1 and angular velocity .2.
These normalized benchmark inputs do not describe measured Queensland sand.

| Elements per axis | Mixed unknowns | Integrated cap torque | Weak cap reaction torque | Magnitude of their difference |
| --- | ---: | ---: | ---: | ---: |
| 2 | 252 | -0.01192743513 | -0.01179065965 | 1.368e-4 |
| 3 | 799 | -0.01179852006 | -0.01173374231 | 6.478e-5 |
| 4 | 1,826 | -0.01171929232 | -0.01168335216 | 3.594e-5 |

All meshes require three Newton steps. Whole-patch weak force and axial torque
defects are below 1.6e-17 and 6.2e-18 respectively. The torque opposes the imposed
rotation; integrated stress torque approaches the independently assembled weak
reaction torque under spatial refinement. This establishes convergence behavior
for the specified finite cap/reservoir problem. It is **not whole-sphere torque**,
does not include the exterior of the cap, and is not a comparison with the
whole-sphere Brenner translation force. The finite stress/reaction difference
is retained in the results rather than described as converged away.

A mixed-boundary hydrostatic problem fixes absolute pressure through traction,
despite an unrelated supplied gauge value. Maximum velocity is 1.51e-11 and
maximum pressure error 1.46e-10 on its two-element curved mesh. Missing traction
data and an incompatible all-velocity total flux are explicitly rejected.

Independent vertical-gravity tests exposed a genuine failure of relying on
algebraic convergence alone: without the affine decomposition, maximum spurious
velocity was 3.27e-4 and 1.46e-4 on two- and three-element curved meshes, despite
algebra residuals around 2.4e-16. With the explicit physical gravity offset, the
two-element vertical test has maximum velocity 6.93e-12 and sampled pressure
error 7.31e-11. Adding the same conservative force and exact pressure to the
nonzero rotating full-NS flow changes velocity by only 6.94e-18; the measured
physical hydrostatic reaction increment agrees with the integrated traction
increment to 6.92e-18. Gravity and its grain load remain present.

A separate unit-similarity check solves the same slow rotating-cap equations
in reference units and in SI units for a 200 micrometre sphere in water
(`rho=1000 kg/m^3`, `mu=.001 Pa s`, gap 5 micrometres, angular velocity .2/s).
Both solves start with zero unknown velocities and take two Newton iterations.
After converting units, maximum velocity and pressure differences are below
4.1e-20 and 9.9e-20. The SI cap torque is -1.1927426e-16 N m. This is a numerical
unit-invariance test with explicit illustrative inputs; it supplies no validation
of real beach-sand shape, loading, roughness, or deposition.
A controlled verification build restoring the former unscaled residual norm
fails this test by accepting zero unknown interior flow before a Newton step.
The normal dimensionless build passes. This checks that the test detects the
actual former defect rather than only repeating the repaired implementation.

#### Implemented conforming nonoverlapping patch pair

`native/resolved-gap-join.hpp/.cpp` now joins two adjacent rectangular regions
of the **same exact physical chart**, verified with
`node native/check-resolved-gap-join.mjs`. Chart-coordinate footprint centers
and aspect ratios locate the regions; the sphere frame itself is not shifted
to fabricate an adjacent surface. Shared Q2 velocity and Q1 pressure nodes
have one global unknown each. The two complete uneliminated element systems
are gathered into one nonlinear system before exterior boundary conditions
and, when applicable, one global pressure gauge are applied. No interface
velocity, reservoir pressure, force correction, or duplicate fluid mass is
inserted between the patches.

The constructor rejects overlapping or separated footprints, different sphere
geometry or fluid material, nonmatching face meshes, and resource excess.
The actual Newton solve uses the same full transient/advective/viscous equations
and physical pressure decomposition as each patch. Its dense reference budget
covers concurrently live matrices. It is not a scalable general mesh manager.

For a 4-by-2-by-2 mesh split into two 2-by-2-by-2 regions, the complete gathered
transient residual and Jacobian match independently assembled monolithic
operators to 4.29e-16 and 2.01e-16 relative. This comparison uses arbitrary
nonzero velocity, pressure, previous-time velocity, forcing, traction and
affine gravity pressure. Independently integrated physical liquid mass agrees
with the assembled constant-field mass to 3.63e-12 relative, confirming that
the interface does not duplicate fluid mass.

An actual nonlinear rotating-cap solve crosses the internal face freely.
It takes three Newton iterations; joined and monolithic velocities, pressures
and physical exterior reactions differ by at most 2.42e-17, 5.43e-17 and
1.22e-18. The weak interface reaction defect is 8.28e-19 and interface work
defect 2.03e-20. Shared velocity, conservative advective flux and affine-pressure
interface flux cancel algebraically on the matching face quadrature.

The direct pointwise stress jump is **not** zero: its maximum is 1.229e-3 on
this finite mesh. Continuous Q2 velocity has discontinuous element derivative
traces. Independently sampled one-sided monolithic traces reproduce the same
jump within 3.03e-16, identifying it as the existing spatial approximation
rather than a new interface force. The code reports that error instead of
forcing a pointwise traction-continuity claim.

This supplies tested conservative joining between conforming FEM regions.
It does not join a fitted gap to the vessel's different staggered discretization,
generate its transition mesh, move/remesh grain boundaries, or implement a
deposited sand bed. Those remaining changes cannot be inferred from the pair
benchmark.

#### Joining to the actual polar vessel grid

Use shared face fluxes and a pressure-gradient operator that is the appropriate
mass-weighted negative adjoint of divergence. Apply the physical rigid-body
velocity on the actual solid faces. A patch/vessel interface must exchange both
normal mass flux and total momentum flux, with equal and opposite transfer.
For a moving interface, the latter is
`rho*u*((u-gridVelocity).n)-sigma*n`; exchanging stress alone omits advected
momentum. The fixed-geometry reference patch exposes the zero-grid-velocity
case and must gain the ALE terms/GCL before being used on moving interfaces.
Interpolation of prescribed boundary velocities alone does not establish a
conservative coupling. The global pressure/rigidity solve must include the
patch interface unknowns; overlapping physical volume must not be counted twice.

The existing `NativeFluid` exposes radial faces `rf`, azimuth spacing `dtheta`,
and vertical faces `sf`. Its physical coordinates are of the form
`(r*cos(theta), r*sin(theta), sigma*H(r,theta))`. A fitted Cartesian gap's four
lateral faces generally do not coincide with those polar faces. Therefore the
present patch cannot be attached by simply reading background velocities and
adding its force back to the unchanged background grid.

A concrete nonoverlap extension is:

1. Select a connected block of existing pressure cells bounded by existing
   `rf`, azimuth and `sf` faces. Remove that block's fluid volume/operators from
   the background solve; do not retain a second copy of its mass.
2. Build a replacement fluid mesh containing the fitted gap and transition
   cells, ending exactly on those original polar interfaces. Grain surfaces
   within the block must be actual solid boundaries of the replacement fluid
   domain. The current gap chart alone does not mesh all this surrounding fluid.
3. Repartition the staggered velocity dual volumes crossing the interface.
   The existing `NativeFluid::velocityMass` weights cannot be copied unchanged
   after removing neighboring fluid volume. Compute matching physical face
   quadrature/geometry on both sides and rebuild the discrete divergence and
   its mass adjoint with the same interface trace/normal-flux unknowns.
4. Assemble the patch's uneliminated mixed blocks with the background pressure,
   rigidity and interface equations. Use one global pressure gauge, equal and
   opposite total momentum flux, and single-counted interface work/mass.
5. Verify a manufactured solution crossing the interface, then conservative
   motion/remeshing transfer, before allowing real near-contact grains through
   the current gap admission threshold.

This transition geometry and background operator exclusion are required work,
not functionality claimed by the present patch. They are a more specific next
step than an overlapping force addition or a lower gap-refusal threshold.

The local patch uses the full advective and viscous operators, including metric
terms. Viscous stiffness can be handled implicitly without dropping viscosity.
The gap is refined before it becomes unresolved, with accepted output retaining
the actual mesh and physical time. Positive-Jacobian failure at g=0 cannot be
fixed by clipping H, changing a radius, or applying contact early.

Resolve normal and lateral geometry separately. The lateral scale near a smooth
sphere-plane gap is proportional to sqrt(a*g); for a sphere pair the effective
radius satisfies 1/a_eff=1/a+1/b. These scales guide mesh allocation only. A
force-convergence measurement, not a fixed number of cells or that asymptotic
scaling, determines whether a particular finite-gap computation is accurate.

## Independent verification that changes the decision

Ghigo's public sphere-wall test supplies a reproducible three-dimensional
embedded-boundary calculation and Brenner-series drag references at several
positive gaps. It is a Stokes-limit verification case, not a replacement for
the nonlinear production equations. Its implementation also uses small cut-cell
cleanup tolerances, so its source must not be imported wholesale under a claim
of satisfying this project's constraints. [Official test and references](https://basilisk.fr/sandbox/ghigo/src/test-stokes/sphere-towards-wall.c).

The new check contains a **test-only** independent Brenner-series evaluator.
Its exponential rearrangement avoids large hyperbolic intermediates and uses
an analytic geometric-series bound on the omitted tail. Six gap/radius ratios
from 0.001 to 10 are compared with independent 75-decimal evaluations of the
original hyperbolic formula; the comparison tolerance is 2e-11 in the
dimensionless force multiplier. The tail bound controls truncation, not
floating-point roundoff. The evaluator is deliberately absent from the
production header and supplies no hydrodynamic force to the simulator. Its
verified domain must not be extrapolated to arbitrarily small gaps.

For the new patch, run the following before connecting it to the sand display:

1. Geometry/metric identities above, including rotated gaps and moving-chart
   constant-field conservation; compare against independent analytic Cartesian
   derivatives, not a duplicate of the implementation.
2. Manufactured three-dimensional incompressible velocity and pressure fields
   with nonzero advection. Refine space and time separately and retain measured
   errors and boundary traction residuals.
3. Prescribed slow sphere-wall approach at fixed positive gaps. Compare the
   computed integrated traction against the independent Brenner-series limit,
   while separately reducing Reynolds number, changing the outer domain size,
   and refining the normal and lateral meshes. An analytic force must never be
   fed into the numerical force balance being tested.
4. Translate/rotate the same setup relative to the vessel mesh. Require
   convergence to the same force and torque, and verify equal/opposite patch
   interface momentum and fluid/grain power exchange.
5. Freely moving, initially separated grains. Verify momentum, work, physical
   time-step refinement and changing-geometry constraints before a near-contact
   run. Then compare two-sphere approach/shear against an independently resolved
   calculation.
6. Only after a physical near-contact surface model is explicit, test actual
   deposition, sustained bed contacts, and lift-off with measured or explicitly
   illustrative material inputs. A successful positive-gap benchmark is not a
   completed settling-bed demonstration.

Exact gap geometry, a standalone full Navier–Stokes patch with physical
mixed-boundary force/torque tests, a conservative conforming patch-pair solve,
and an independent test-only force oracle are now implemented. The next required
work is whole-grain gap-force convergence plus conservative joining to the
actual vessel discretization and motion described above. The current fixed-grid gap refusal must remain
until that work supplies a valid computed continuation.
