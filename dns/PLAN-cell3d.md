# The renderer's picture becomes a Navier-Stokes solve

**Session counter: 2**
**Stages complete: 5 of 14**
**Next action: S5 — conservative centred advection, grid-relative in sigma. The
primitive it needs already exists: `colValueAtZ` reconstructs any field in any
column at any physical height, which is how every cross-column comparison in this
solver must be done (see below). No upwinding: it would add numerical dissipation,
which is indistinguishable from viscosity in the answer and would fake the damping
that sets the threshold.**

### Read this before touching any operator: compare at a common physical height

Four separate faults in S4b were the same fault. Under z = sigma H two columns'
sigma levels sit at DIFFERENT physical heights whenever the surface is deformed, so
any quantity formed by comparing columns at equal sigma carries an O(dH) error that
vanishes when flat and does not converge when not. It bit:

  - the tangential derivatives in the Laplacian (order -1.99 at worst),
  - the interpolation of v to u's nodes in the coupling (2.00 flat, -0.52 deformed),

and would bite the advection identically. `colValueAtZ` is the primitive: quadratic
in sigma on a stencil centred on the LEVEL being differenced, not on the target, so
two columns compared at one height use the same node positions and their
reconstruction errors cancel rather than jumping as a target crosses a node.

A second rule, from the same stage: a boundary face needs a THREE-POINT derivative.
A two-point difference between a wall value and the nearest node is centred at their
midpoint and only first order at the face, and a flux error of that order does not
converge at all. This was fixed once, lost in a rewrite, and found again by the w
family's last radial row diverging while every interior row ran at second order.

### The sigma-coordinate cancellation, and why it defeats the obvious discretisation

Worth reading before touching `famLaplacian`, because it cost four measure-fix
cycles to find and it will look like a small accuracy problem until it is
understood.

Under z = sigma H, each physical derivative is a difference of two terms:
d/dr|_z = d_r - (sigma H_r/H) d_sigma. For a field that depends on z alone those
two terms are individually O(1) and cancel EXACTLY. Discretely they cancel only to
O(dtheta^2); the Laplacian then divides a difference of face fluxes by dtheta,
leaving O(dtheta); and the (1/r^2) factor near the axis amplifies that by 1/dr^2.
Refining the grid therefore makes it WORSE. Measured with f = sin(kz), a surface
varying only in theta, eta/h = 0.3:

    family p:  1.77e-1 -> 1.72e-1   order  0.04
    family v:  8.63e-1 -> 3.42e+0   order -1.99

This is the same defect known in terrain-following ocean and atmosphere models as
the pressure-gradient error over steep topography. No amount of care in the
metric slopes fixes it, because the problem is the subtraction itself. Evaluating
the tangential difference between two columns at a common physical height removes
the subtraction: for f = f(z) both interpolated values are equal, so the
difference is exactly zero.

### Faults found and fixed in `famLaplacian` so far

| Fault | How it showed | Fix |
|-------|---------------|-----|
| Hat's slopes used for the sigma-face cross terms | every family order 0.5 instead of 2 | interpolate the precomputed centred slopes (`Hslope`) instead; Hat's own are centred only at face midpoints |
| Boundary branch keyed on the unknown range, not the node list | NaN for the whole w family, from a zero-width half cell above its surface node | key it on the node list: a node that exists but is not solved for is an ordinary neighbour |
| One-sided boundary derivative | surface row stuck at 2.4e-1 on both grids | three-point quadratic at the face, as `wzSurface` already does in the two-dimensional solver |

A fourth, still open, is the cancellation above.

Update the counter, the stage table and the next action in the same commit that
changes a stage's status. A stage is `done` only when a gate in
`dns/check-cell3d.mjs` asserts it against something other than this solver's own
opinion, and the measured figure is recorded in the row.

---

## Why this exists

The page draws a modal superposition from `faraday/kernel.js`. The Navier-Stokes
solvers sit in a panel beside it and are not what you see. The instruction is that
the solver **is** the visuals, which forces three things the linear per-mode solver
in `dns/faraday-disc.js` cannot do, none of them optional:

- **Nonlinearity.** A growing linear mode grows without bound. The visible figure
  is a saturated finite-amplitude state and saturation is the advective term.
- **Every azimuthal mode at once, coupled.** The figures are not one `m`, and what
  couples modes is again the nonlinearity.
- **A domain that follows the surface.** At the drives this cell runs, eta/h is of
  order a half; applying the surface conditions at a fixed plane would be exactly
  the shortcut this work refuses.

`dns/faraday-disc.js` is not superseded. It answers "does this mode grow" to eight
digits and stays the instrument for the threshold; linearising is not a shortcut
for that question, because Floquet stability of the flat state IS the linear
problem. The new solver answers "what do you see", which that one cannot.

## The one hard constraint, stated once

This will not run at real time, and nothing does: a DNS of this cell at a grid
that resolves the Stokes layers costs thousands of pressure solves per second of
physics. The page shows the physics clock against the wall clock and reports the
ratio. That is what DNS costs, not a compromise chosen here.

Second, from the hardware: **no GPU on an Intel Mac can do double precision.**
The Metal Shading Language Specification has no `double` type, which is also why
WGSL has no `f64`. So the GPU cannot own the arithmetic without dropping to single
precision.

**Settled, confirmed by the owner on 2026-09-28, not to be reopened:** all eight
CPU cores carry the physics in full `f64`; the GPU carries the rendering, where
`f32` is correct because the output is pixels; and the GPU may additionally carry
an `f32` preconditioner inside the `f64` solve, which changes the iteration count
and not the answer. The physics is never moved to `f32`. S11 and S12 are written
against this and need no further decision.

## Stages

| # | Stage | Status | Evidence |
|---|-------|--------|----------|
| S1 | Exact projection under the surface-following metric | **done** | symmetric to 2.7e-16 at eta/h = 0.7; divergence falls 3.7e+10 at tol 1e-11 and five decades further at 1e-14 |
| S2 | Metric Laplacian with the sigma-face cross terms | **done** | order 2.05 flat, 2.00 at eta/h = 0.3, 1.52 at 0.6; cross terms proven load-bearing by injection (order -0.01 without them) |
| S3 | Axis as a reflection, so m = 1 needs no special case | **done** | axis cell converges at the interior's order, 2.01 against 2.00 |
| S4a | Staggered node geometry, u_r at the axis, metric interpolation | **done** | descriptors shape-checked; axis row exactly antisymmetric and non-zero for m = 1, exactly zero for m = 3; `Hat` second order at face midpoints (1.96, 1.95) and exact in r on a quadratic surface |
| S4b | The Laplacian instantiated at u, v, w, plus the cylindrical vector coupling | **done** | vector Laplacian second order at every deformation: (grad^2 u)_r and (grad^2 u)_theta both 2.00 at eta/h = 0, 0.2, 0.4; grad^2 w 2.00 flat and 1.97 at 0.4; scalar probe convergent at all four families |
| S5 | Conservative centred advection, grid-relative in sigma | todo | discrete kinetic energy conserved to round-off with viscosity and drive off; no upwinding, which would fake viscosity |
| S6 | Free surface: full mean curvature, normal stress, tangential stress, kinematic update | todo | curvature against the analytic mean curvature of a known surface, under refinement |
| S7 | `step()`, its stability limit, and the energy diagnostic | todo | amplification below one at the stated limit and above it at twice the limit |
| S8 | Validation against the independent linear solver | todo | at small amplitude, per-mode growth rate agrees with `faraday-disc.js`; energy conserved as nu goes to zero; harmonics appear at finite amplitude |
| S9 | The renderer draws this solver's surface | todo | the page's field equals the solver's eta to the digit; physics and wall clocks both shown |
| S10 | C++ port of the three-dimensional step | todo | bit-for-bit against the JavaScript over a full drive period, as `faraday_disc.cpp` already is |
| S11 | All eight cores | todo | measured speedup against core count; identical answer on any count |
| S12 | GPU render path, and the optional f32 preconditioner | todo | f64 answer unchanged by the preconditioner; render timing measured |
| S13 | Ship: one zip, README, everything gated in CI | todo | suites pass from a fresh unpack, as `faraday-cell.zip` already does |

## Rules this build keeps

Inherited from `CLAUDE.md` and from what has already gone wrong here:

- No upwinding in the advection. It would add numerical dissipation, which is
  indistinguishable from viscosity in the answer and would fake the damping that
  sets the threshold.
- No linearised surface conditions, no fixed-plane surface, no dropped metric
  term. S2's injection test shows why: a flat test grid cannot detect a missing
  cross term, so every operator gate deforms the surface.
- Expected values come from calculus, from an independently written solver, or
  from an identity the discretisation must satisfy. Never from this solver.
- Every gate must be shown able to fail, and the injection is done on a copy
  under `/tmp`, never in the working tree.
- When a measurement contradicts an expectation, the expectation is corrected in
  place with the number that overturned it. Three have been so far: the pressure
  operator does not annihilate constants and must not, the divergence-reduction
  bound confused a 2-norm with a maximum, and a grid large enough to overflow the
  module arena was not the one assumed.
