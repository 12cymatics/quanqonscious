# The renderer's picture becomes a Navier-Stokes solve

**Session counter: 3**
**Stages complete: 6 of 14, and S6 is under way**
**Next action: S6d -- the traction on the momentum faces, and w solved at sigma = 1.
The strain tensor and the viscous NORMAL stress are done and gated (8b, 8c). What
remains of S6: form the traction T = 2 rho nu E.n at each momentum family's sigma = 1
face, replace it by (n.T)n so the tangential part is exactly zero at any slope, use it
as that face's viscous flux, and extend `famLaplacian`'s w family from nz-1 to nz --
`dns/faraday-disc.js`'s `lapW` (line 322) is the pattern. Then the surface pressure
`p_s = p_ext - gamma kappa + n.T` and the kinematic update, and S6 is closed.
(Superseded: S6c -- the surface stresses. The curvature is done and gated (7a to 7d)
and so is the outward normal (8a); 140 checks, 0 failed. The count was written as 138
here for one commit, which was the total before 8a was added -- measured, not
transcribed, and corrected. What remains: the full normal stress
`rho g eta - gamma kappa + 2 rho nu n.E.n` with the WHOLE rate-of-strain contraction and
not the flat-normal `2 rho nu dw/dz` the two-dimensional solver is entitled to; the two
tangential conditions `t_j.(2 rho nu E).n = 0` supplying du/dz and dv/dz at sigma = 1;
and `famLaplacian`'s w family extended from nz-1 to nz, which is what closes the surface
node S5 made a solved advection unknown. `dns/faraday-disc.js`'s `lapW` (line 322) is the
pattern for that last one: it solves w at the surface and closes the top face on
`wzSurface`, the three-point quadratic of w's own column.**

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

A THIRD rule, from S5, and it is the limit of the first: the ADVECTION must NOT use
`colValueAtZ`. The common-height rule exists because a physical DERIVATIVE under
z = sigma H is a difference of two O(1) terms that must cancel. An AVERAGE has no
cancellation to protect, and the flux form in these coordinates never forms such a
difference -- it is the same flux algebra `divergence` uses, which is exact. What the
average must do instead is telescope, and that requires the face value to be the plain
arithmetic mean of its two neighbouring nodes; reconstructing at a common height breaks
the telescoping and the scheme stops conserving energy. The planning note that said to
use `colValueAtZ` here was wrong and is corrected in place.

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
| S5 | Conservative centred advection, grid-relative in sigma | **done** | every momentum cell's net flux is exactly half the sum of its neighbours' divergences (worst 1e-16 of the largest net, three families, two grids, deformed and moving); the transport telescopes exactly on an arbitrary field (1e-16 of the terms summed); the curvature pair cancels to 7e-18; a uniform rise over a rising surface is left exactly alone; on a projected field the residual follows the solve tolerance over 6.2 decades to 1.8e-17; second order against calculus outside the axis cells, 1.90 to 2.08 in all three components at eta/h = 0, 0.2 and 0.4; seven injected defects all red |
| S6 | Free surface: full mean curvature | **done** | the curvature of a sphere is -2/Rs wherever you stand on it, second order at 1.90, 1.88, 1.94 with \|grad eta\| up to 1.61; the curvature of a tilted plane is zero at second order outside r/R = 0.4, and the whole residual is alpha dtheta^2/r -- the same constant 8.01e-2, 8.13e-2, 8.16e-2 on three grids; kappa is the EXACT variational derivative of the discrete area, residual falling as eps^2 with ratio 4.00 on both contact branches; both contact conditions second order inside r/R = 0.9 (1.85/1.96 free, 1.97/1.99 pinned) and bounded by 9% without converging in the two rows at the rim; five injected defects all red |
| S6b | Free surface: the outward normal | **done** | second order against the analytic normal of a surface with \|grad eta\| up to 0.68, order 1.98, over r/R in [0.25, 0.9]; and a unit vector to 1e-16 |
| S6c | Free surface: the strain tensor at the surface, and the viscous normal stress | **done** | all six components second order against calculus on a deformed surface, 1.97 to 4.35 at eta/h = 0 and 0.3, rim row included; 2 rho nu n.E.n second order (1.98) against the analytic contraction with the analytic normal at eta/h = 0.3 and 0.6; on a FLAT surface it equals `wzSurface` in the independently written two-dimensional solver to 1.1e-13 relative, and on a deformed one the flat form is wrong by 95.5% of the stress; five injected defects all red |
| S6d | Free surface: the traction on the momentum faces, and w solved at sigma = 1 | todo | the tangential traction exactly zero after projection; grad^2 w convergent including the surface row; the small-slope limit reproduces `surfaceSlopes` next door |
| S7 | `step()`, its stability limit, and the energy diagnostic | todo | amplification below one at the stated limit and above it at twice the limit |
| S8 | Validation against the independent linear solver | todo | at small amplitude, per-mode growth rate agrees with `faraday-disc.js`; energy conserved as nu goes to zero; harmonics appear at finite amplitude |
| S9 | The renderer draws this solver's surface | todo | the page's field equals the solver's eta to the digit; physics and wall clocks both shown |
| S10 | C++ port of the three-dimensional step | todo | bit-for-bit against the JavaScript over a full drive period, as `faraday_disc.cpp` already is |
| S11 | All eight cores | todo | measured speedup against core count; identical answer on any count |
| S12 | GPU render path, and the optional f32 preconditioner | todo | f64 answer unchanged by the preconditioner; render timing measured |
| S13 | Ship: one zip, README, everything gated in CI | todo | suites pass from a fresh unpack, as `faraday-cell.zip` already does |

## What S5 found, in the order it was found

### The moving mesh needs the volume-rate term, and the energy gate is what caught it

The flux form alone is not the momentum balance on a mesh that moves. What a
finite-volume cell conserves is its momentum CONTENT:

    d(V u)/dt + sum_faces F_rel phi = 0    ==>    V du/dt = -f - u dV/dt

and the volume rate has to be the net of the SAME mesh fluxes that were subtracted to
make the transport grid-relative -- the discrete geometric conservation law. Written
without it, the energy residual on a projected divergence-free field sat at **2.5e-3 of
the terms it sums and did not move when the projection tolerance was tightened by five
decades**, which is what proved it was not the projection's error. The same omission
makes a field consisting of nothing but a uniform rise accelerate out of nothing; that
is now its own gate, 6e.

### The curvature pair cancels exactly, but only if the product is formed once

+u_theta^2/r and -u_r u_theta/r cancel POINTWISE in the continuum. On a staggered grid
u_r and u_theta live in different places, and interpolating each to the other's node --
the obvious thing, and what was written first -- leaves (mean v)^2 against v^2, which do
not cancel: **measured, 1.7e-2 of the terms' own size**, an energy source of the
scheme's own truncation order. Forming the product once at the pressure cell centre and
distributing it to the two equations as the exact adjoints of those cell-centre averages
cancels to the last bit (7e-18). The injection test reproduces the old behaviour at
1.4e-2, so the gate sees exactly this.

### Two terms the identity leaves open, both measured rather than assumed

- **The axis row.** u_r at r = 0 is not an unknown -- it is the antisymmetric
  extrapolation of the two columns outside it -- so its control volume has exactly zero
  measure and the faces it shares with node 1 have no partner inside the energy sum.
  The residual advection of a uniform Cartesian field is derived in closed form and
  pinned in gate 6i:

      A_r = -U^2 [ a cos^2(th) - cos(a/2) sin(a) cos(2 th) ] / (rf a)
            + U^2 sin^2(th) cos^2(a/2) / rf        with a = dtheta

  which expands to -U^2 a^2 [1/8 + cos(2 th)/6]/rf + O(a^4). Second order in dtheta at a
  fixed radius (measured 1.99), first order at the first cell where rf is itself a
  spacing (measured 0.98). No interpolation removes it: the two conditions that would
  make it vanish for all theta ask one coefficient to be 1/4 and -1/3. This is why the
  accuracy gate reports the inner quarter of the radius separately, and it is the whole
  of what the axis costs.

- **The radial volume weighting.** A u cell's volume is rf drf dth Hr dsigma, the
  product `gradient` and `famLaplacian` already use, and its time derivative is not
  exactly the net of the mesh fluxes through its faces. The two differ because Hr is the
  DISTANCE-weighted interpolation of H, which is what makes it a consistent face value,
  while the volume sum wants the volume-weighted one, and no single interpolation is
  both. The v and w families have no gap at all -- for them the two agree to the last
  bit, because sc and Hth are exact midpoints -- so this is the radial family alone, and
  it is the one term by which kinetic energy fails to be conserved exactly while the
  surface is moving. Measured order 2.02. Choosing the other side of that trade would
  have bought exact energy at the price of an advective flux through the free surface,
  which is not a trade worth making.

### Nothing is clamped at a boundary any more

All four boundary fluxes are zero for a physical reason -- zero area at the axis, no
penetration at the rim, no slip on the floor, and a material surface at sigma = 1, where
Omega IS dH/dt and the grid-relative flux is a floating-point difference of equals. They
are now written out rather than branched to zero, so a violated boundary condition shows
up as a divergence or an energy imbalance instead of being masked. Injecting a wrong mesh
flux is caught by exactly that check.

### Two more of my own expectations overturned by measurement

- Two algebraically identical expressions were expected to agree bit for bit and read
  3.9e-23 apart. They are the same number and not the same sequence of operations, and
  on a grid graded towards both ends the widths one of them multiplies are a thousandth
  of the fluxes the other differences, so the bound has to be relative to the fluxes.
- The closed form in 6i was expected to match to 1e-12 of the answer and read 1.7e-12.
  Both sides reach an answer of size U^2 a^2/rf by subtracting two quantities of size
  U^2/rf, so a bound relative to the answer asks for a^2/8 more precision than double
  carries. Measured against the terms each subtracts, the agreement is 1e-15.

## What S6 found about the curvature

### Define it as the derivative of the area, not as a discretised formula

The curvature is `kappa_j = -(1/(rc drc dtheta)) dA/d eta_j` for the discrete area

    A = sum_cells rc drc dtheta sqrt(1 + |grad eta|^2)

and three things follow that a discretised formula would only approximate. It IS the
finite-volume divergence form: worked out, the radial face coefficient comes to the
width-weighted mean of `1/sqrt(1 + |grad eta|^2)` across the face, which is what
consistency asks for. It is second order, because the area is. And the work the capillary
term does is exactly `-gamma dA/dt`, so the exchange between kinetic and surface energy is
an identity in floating point rather than a tolerance -- which is what will let S7 assert
that the whole step conserves energy with viscosity and the drive off. Gate 7c holds the
identity directly: the central difference of A along a random direction matches
`-sum (rc drc dtheta) kappa delta` with a residual that falls as eps^2, ratio 4.00.

### The axis needs no condition, and that is not a convenience

`eta_r` at r = 0 is non-zero for every azimuthal mode but m = 0, so any single value
assigned there would be wrong for some mode. Weighting each cell's two radial slopes by
their FACE AREAS makes the axis face -- whose area rf[0] is exactly zero -- drop out of
its own accord, in the area and in the derivative alike.

### The face-area weighting is load-bearing, and the rim cannot be fixed inside it

The weighting is not free to choose. It is exactly what makes `rc drc/(rf[i] + rf[i+1])`
collapse to `drc/2`, and hence what makes the derivative a divergence at all. Three ways
of making the pinned rim slope second order were tried and every one made the curvature
far worse. Measured at the rim row, relative, on grids of 16, 32 and 64 radial cells:

| rim closure | rim row error |
|---|---|
| three-point quadratic through the wall | 35, 80 -- diverging, order -1.2 |
| weights 1/3 and 2/3, centring the estimate on rc | 38, 84, 176 |
| slope extrapolated to R as (4 sWall - sIn)/3 | 39, 84, 176 |
| **the plain local two-point difference** | **0.013, 0.016, 0.017** |

So the plain difference is not a shortcut; it is the only one of the four that leaves a
small error instead of a large one. What is left is a couple of per cent in the outermost
two rows, not converging, and it is asserted as a BOUND rather than as a rate because it
is not a rate. The free branch proves the cause is not the slope: a free contact line's
rim slope is exactly right -- zero is the condition itself -- and its rim row is first
order anyway, because what is one-sided there is the face's own coefficient, which has
only the cell inside it to come from. The trade was taken this way round because locality
buys the exact energy identity, and one annulus of width h carrying a first-order force
contributes at the scheme's own order to anything integrated.

### Rule 2 has an exception, and it is specific to functionals

"A boundary face needs a three-point derivative" holds for a flux computed directly. It
does NOT hold for a slope inside a functional whose derivative is then taken: the
three-point slope makes the last cell's area depend on the column two cells in, so the
functional stops being a sum of local cell areas and its derivative lands a term on cell
nr-2 where that term is not a divergence. That is the first row of the table above.

## What S6b must not do: eliminate the surface z-derivatives

Worked out before writing any of it, because the obvious move is singular and the
singularity is at a slope this cell reaches.

The two tangential conditions are `t_j . E . n = 0` for the two surface tangents
`t1 = (1, 0, s)` and `t2 = (0, 1, t)`, with `s = eta_r` and `t = eta_theta/r`. Written out
they are two equations for the two strain components that carry the vertical derivatives:

    (1 - s^2) E_rz  -    s t    E_theta_z  =  s E_rr + t E_r_theta     - s E_zz
      - s t   E_rz  + (1 - t^2) E_theta_z  =  s E_r_theta + t E_tt     - t E_zz

and their determinant is `1 - s^2 - t^2`. **It vanishes at |grad eta| = 1 and changes sign
beyond it.** So solving them for `du_r/dz` and `du_theta/dz` -- which is what
`surfaceSlopes` does next door, in its linearised form, and what the obvious
generalisation would do -- is singular at a forty-five degree slope. The same denominator
appears if one tries to eliminate the normal stress instead: `E . n = lambda n` gives

    lambda (1 - s^2 - t^2) = E_zz - s^2 E_rr - 2 s t E_r_theta - t^2 E_tt

so `n . E . n` cannot be obtained that way either.

That is not a defect in the physics. At `s = 1, t = 0` the tangential condition in the
r-z plane reduces to `E_zz = E_rr` -- it constrains the horizontal strains and says
nothing about `du_r/dz`, because a forty-five degree rotation turns the shear in that
plane into the difference of the two normal strains. The traction is then left to the
momentum equations, which is where it belongs. The elimination is simply the wrong move.

**What to do instead is impose the conditions on the FLUX**, which is where a
finite-volume method wants them anyway:

1. Form the six strain components at the sigma = 1 face from the interior field:
   horizontal derivatives at the surface's own height (rule 1), vertical derivatives
   one-sided from each family's own column with the three-point quadratic (rule 2, as
   `wzSurface` does in `dns/faraday-disc.js`).
2. Form the traction `T = 2 rho nu E . n`.
3. Replace `T` by `(n . T) n`. That imposes zero tangential traction exactly, at any
   slope, with no division by `1 - |grad eta|^2` anywhere.
4. The sigma = 1 face's viscous momentum flux is `T_i` times the face area, and `n . T` is
   the viscous part of the surface pressure, `p_s = p_ext - gamma kappa + n . T`.

This is exact at any slope and it reduces to the two-dimensional solver's linearised pair
as the slope goes to zero: `n` goes to `z`, `T` goes to `2 rho nu (E_rz, E_theta_z, E_zz)`,
and projecting out the tangential part sets `E_rz = E_theta_z = 0`, which is exactly
`du_r/dz = -du_z/dr` and `du_theta/dz = -(1/r) du_z/dtheta` -- `surfaceSlopes` itself.
**That limit is a gate**, and it is a gate against an independently written code rather
than against this one's own opinion.

## The gate runs in CI now

`dns/check-cell3d.mjs` was not run by any job until 2026-09-28. It is now the first step
of the `dns` job in `.github/workflows/javascript.yml`, ahead of `dns/check-dns.mjs`, so
its signal arrives in about seven seconds rather than behind a two-and-a-half minute
suite. It shares that job rather than taking a runner of its own, which seven seconds does
not earn. Confirmed to be a gate and not a decoration: with a defect injected it exits 1,
clean it exits 0.

A suite that exists and is not run is not a gate, and this repository has already paid for
that once -- a red `tests/test_documented_paths.py` sat unobserved on the default branch
because the workflow carrying it was keyed on a branch that does not exist.

## What S6c measured

The rate-of-strain tensor at the free surface, and the viscous normal stress from it.

Every horizontal derivative is taken between columns at ONE physical height -- this cell's
own surface height, so a neighbour is read above or below its own surface as the slope
dictates. That is rule 1 and the gate sees it: evaluating each column at its own sigma = 1
instead leaves E_rr at order -0.007 on a deformed surface while the flat case still passes,
which is the signature of that rule being broken.

Every vertical derivative is the quadratic through a column's three topmost sigma nodes,
differentiated at the target height. For w that interpolates, since w has a node at
sigma = 1; for u and v it extrapolates half a cell, which is what a field whose vertical
nodes are cell centres costs at a boundary.

`colDerivAtZ` sits beside `colValueAtZ` and shares its stencil conventions deliberately:
the two are read at the same points, and a difference in stencil between them is a
difference nothing would catch.

**The flat limit is a check against another code.** On a flat surface the normal is exactly
z-hat, so `2 rho nu n.E.n` must collapse to `2 rho nu dw/dz` -- which is `wzSurface` in
`dns/faraday-disc.js`. It agrees to 1.1e-13 relative, the round-off of the cancellation in
a three-point derivative rather than bit equality, since the two work in sigma and in z
respectively. On a deformed surface at |grad eta| up to 0.30 the flat form is wrong by
**95.5% of the stress**, which is the whole reason for carrying the full contraction.

**One gate exists solely for the contraction.** The six components being right does not make
n.E.n right: a dropped or mis-signed cross term leaves every component correct and the
stress wrong, and the flat limit cannot see it either, because on a flat surface every cross
term is multiplied by a normal component that is zero. Dropping the radial-vertical cross
term is red only on that gate, at order -0.01.

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
  place with the number that overturned it. Five have been so far: the pressure
  operator does not annihilate constants and must not, the divergence-reduction
  bound confused a 2-norm with a maximum, a grid large enough to overflow the
  module arena was not the one assumed, two algebraically identical expressions
  were expected to agree bit for bit, and a residual formed by cancellation was
  normalised against the answer instead of against the terms that cancelled.
