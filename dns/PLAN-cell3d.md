# The renderer's picture becomes a Navier-Stokes solve

**Session counter: 1**
**Stages complete: 4 of 14**
**Next action: S4b — instantiate the metric Laplacian at the u, v and w families
and add the cylindrical vector coupling. The geometry it needs is in place
(`this.FAM`), the metric interpolation it calls is verified (`Hat`), and the one
non-mechanical piece, u_r at the axis, is done.**

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
precision. The plan therefore puts all eight CPU cores on the physics in full
`f64`, the GPU on rendering where `f32` is correct because the output is pixels,
and the GPU optionally on an `f32` preconditioner inside the `f64` solve, which
changes the iteration count and not the answer.

## Stages

| # | Stage | Status | Evidence |
|---|-------|--------|----------|
| S1 | Exact projection under the surface-following metric | **done** | symmetric to 2.7e-16 at eta/h = 0.7; divergence falls 3.7e+10 at tol 1e-11 and five decades further at 1e-14 |
| S2 | Metric Laplacian with the sigma-face cross terms | **done** | order 2.05 flat, 2.00 at eta/h = 0.3, 1.52 at 0.6; cross terms proven load-bearing by injection (order -0.01 without them) |
| S3 | Axis as a reflection, so m = 1 needs no special case | **done** | axis cell converges at the interior's order, 2.01 against 2.00 |
| S4a | Staggered node geometry, u_r at the axis, metric interpolation | **done** | descriptors shape-checked; axis row exactly antisymmetric and non-zero for m = 1, exactly zero for m = 3; `Hat` second order at face midpoints (1.96, 1.95) and exact in r on a quadratic surface |
| S4b | The Laplacian instantiated at u, v, w, plus the cylindrical vector coupling | todo | manufactured solution at each staggered family; the coupling against the analytic vector Laplacian |
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
