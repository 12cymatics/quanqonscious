# The renderer's picture becomes a Navier-Stokes solve

**Session counter: 5**
**Stages complete: 8 of 14 -- S6, S7 and S8 are closed. The file is a SOLVER, and it has been
validated against an independently written one, driven and undriven. A step cost 146 ms on
16x24x10 when S7 landed, so an oscillation period cost eleven minutes and S8 was not
affordable; S7b halved it to 85.6 ms with the operators proven bit-for-bit unchanged.**
**Next action: S9b, the renderer draws this solver (S9a, the resampler, is done). Replace the modal field in
`recompute()`/`renderSurface()` with eta from the 3-D solver, in a worker; show the physics
clock against the wall clock and report the ratio; size the grid by the same measured-resolution
logic `suggestGrid` already applies. Gate by extending `dns/check-page.mjs`: the page's field
equals the solver's eta to the digit, and both clocks are present.**

**What session 5 did.** `faraday-cell3d.js` had no `step()` at all: forty-three methods, every
one an operator, and nothing that advanced anything in time. Building the assembly found two
O(1) defects in the committed projection that nothing had noticed BECAUSE nothing used the
gradient for anything but its own transpose -- the Omega control volume was missing its H, and
the horizontal components were the covariant gradient rather than the physical one (86.99
against 137.00 at r/R = 0.888). Both are fixed and gated. `surfacePressure`, `step`,
`stepLimits`, `stableStep` and `energy` exist, and the assembled solver reproduces the linear
gravity-capillary dispersion relation. See "S7: the file becomes a solver" below.

**What session 4 did, in one line each.** The surface-row defect session 3 diagnosed was
mostly the GATE's: its test surface was inadmissible at the axis and contradicted the free
contact line at the rim. Underneath it were three real first-order closures, all the same
mechanism -- a boundary row has nothing for its interior face's error to cancel against --
cured by making every face derivative cubic. One quadrature sat at the node instead of the
control volume's sigma centroid. `FAM.w.sHi` is now `nz`, so w is solved at the surface, and
gate 4g says what that unknown means and pins its centroid offset in closed form. The
azimuthal cross term needed the reconstruction stencil to bracket its target rather than be
anchored on the level, and what remains there is a property of the grid, not the operator.
The regeneration test then found a gate gap -- one of the five injected defects left the whole
suite green -- and closing it needed an AXISYMMETRIC probe beside the azimuthal one. 195
checks, 0 failed, 44 seconds. Every number is in "The surface-row defect, resolved" below.

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
| S6d | Free surface: the flux the Laplacian's surface face wants, from the traction | **done** | the tangential traction exactly zero at both tangents, at \|grad eta\| up to 0.30 -- exactly, not nearly; on a FLAT surface the radial and azimuthal fluxes are EXACTLY minus dw/dr and -(1/r)dw/dtheta, which is `surfaceSlopes` in the independently written two-dimensional solver; second order against the identity formed analytically, 1.98 to 1.99, at eta/h = 0.3 and 0.6; a flat surface's slopes and normal exactly zero and exactly vertical; five injected defects all red, two of them caught only by the exactness gates |
| S6e | Free surface: the flux hook in famLaplacian, and u, v, w closed by it | **done** | the hook changes the surface row by exactly the flux over the volume it crosses and changes no other row at all; `viscous` supplies it for all three; `FAM.w.sHi` is `nz`, so w is solved at sigma = 1 over the half cell its advection already uses |
| S6f | Free surface: the Laplacian second order at EVERY row, and what w's surface unknown means | **done** | every face derivative cubic, so no boundary row relies on a cancellation: 4e reads 1.99 to 2.00 at all eight (family, amplitude) pairs against a floor raised from 1.2 to 1.9; the sigma = 0 row's first-order term identified in closed form and measured to three digits against it before the fix; 4g asserts grad^2 w at the surface row second order against the exact average over its half cell, AND pins its gap to the point value at sigma = 1 as the centroid offset in closed form, over eight cases -- azimuthal and axisymmetric probes, flat and deformed, at the renderer's grid aspect and at one where the azimuthal surface-slope resolution matches the radial; five injected defects red, the fifth only after the axisymmetric probe was added, which is why it is there |
| S6g | Free surface: the flux interpolation, gated | **done** | the interpolation of each surface-flux component onto its OWN family's sigma = 1 face -- radially for u, azimuthally for v -- is second order against `wantFlux` evaluated at that face, 2.12 then 2.07 for u and 1.96 then 1.98 for v over 16/32/64 at eta/h = 0.3 and 0.6; one cell's value in place of either reads 0.92 and 0.99. The plan said this needed a probe satisfying zero tangential stress; it does not, because the analytic flux can be asked for the face's own position |
| S6h | Free surface: the surface pressure | **done** | `rho g eta - gamma kappa + 2 rho nu n.E.n`, entering the projection's right-hand side top row and the corrector's sigma = 1 face and nowhere else -- never the predictor, where it is O(1/dsigma) and the projection cancels almost all of it |
| S7 | `step()`, its stability limit, and the energy diagnostic | **done** | the assembled solver reproduces the linear gravity-capillary frequency for m = 3, error 13.11% -> 6.62% -> 3.92% -> 2.08% over four grids, which no single operator in the file could produce alone; building it found two O(1) defects in the committed projection (the Omega control volume missing its H; the horizontal components covariant rather than physical, 86.99 against 137.00); `stableStep` from the DISPERSION relation, with no growth at 1x (0.997392 and 0.997441 per step on two grids) and divergence at 10x (by step 14 and step 15), so the limit is falsifiable on both sides; over a quarter period at nu = 1e-12 the energy drifts -0.0343%, never rises, and 99.19% of it becomes kinetic |
| S8a | Validation: a quarter period's energy decay against `faraday-disc.js` | **done** | a quantity with no closed form, from two solvers sharing nothing but the physics and not even a vertical coordinate: 1.26121/1.33232, 1.29079/1.32578, 1.30927/1.32165, 1.33430/1.33377, 1.35070/1.34333 over five grid pairs -- gap +5.64%, +2.71%, +0.95%, -0.04%, -0.55%, each code still moving in its own grid, so they converge to one limit from opposite sides; m = 2 likewise +6.13% then +3.35%. Found one real defect, in the ENERGY DIAGNOSTIC: the capillary part was the difference of two areas agreeing to thirteen digits, which at eta = 1e-9 m reported a 9.4% energy GAIN where the scheme drifts -0.035%. Free-slip walls read -59% and are red; half the radial viscosity reads -2.47% and is red only since the third pair was added, which is why it was |
| S8b | Validation: harmonics, mode coupling, and step() as a composition | **done** | a single m = 3 mode generates m = 0 and m = 6 at exponent 1.998/1.999 in the amplitude and m = 9 at 2.853/2.960, all from 1e-22; m = 2 and m = 3 together generate m = 1 and m = 5 bilinearly (1.9803, 1.9991, 1.9992, 1.9992 on halving either parent) and neither parent alone produces them at better than 1e-22. The regeneration test then found that the whole ADVECTIVE TERM can be deleted from `step()` with all 239 checks green -- at eta/h = 0.2 it is one part in a hundred of the inertia -- so section 14 now reassembles one step from the public operators and asserts the eight state arrays bit for bit. Four step-composition defects red |
| S8c | Validation: the driven growth rate against the Arnoldi multiplier | **done** | six drive periods at 20x24x12 give \|mu\| = 1.235690, 1.762186, 1.987387, 2.041031, 2.053208, 2.055543 -- increments 0.0536, 0.0122, 0.0023 -- against `floquetDisc`'s Arnoldi value 2.08649212 for the same nr and nz: **-1.49%**, two solvers with different vertical coordinates agreeing on a driven Floquet multiplier. That costs 400 s per period so it is recorded, not gated; section 15 gates the disc's own time-domain driven run at a matched 12x8 / 12x20x8, where the third period reads 2.01139 against 1.96195 (-2.46%) and the window's amplification 4.4801 against 4.1887 (-6.50%), plus the off-resonance contrast (1.836e-1 and 2.058e-2 against 4.48 and 4.19) |
| S9a | The resampler: eta at an arbitrary position | **done** | `etaAt` reads the extended grid `HatH` reads, so the axis is interpolated ACROSS (antipodal row) and the rim IS the contact condition, rather than a renderer reinventing both. Exact at a cell centre to four ulp of the amplitude (5.421e-19 of 1.188e-3, and the residual is the sample position, not the interpolation), second order between centres at 1.965 then 1.919, spread 0.00e+0 over theta on the axis for an axisymmetric surface and 9.26e-23 for m = 3, and exactly the contact condition at r = R. Found a real defect in its own first version: `HatH - h` cannot return eta, because h + eta has an ulp of 4.3e-19 against elevations of 1e-9 to 1e-4 m |
| S9b | The renderer draws this solver's surface | **done** | the page's drawn field agrees with `CELL3D.etaAtPixel` at 379 pixels to a relative 1e-6, through four resamplers and a polar-to-Cartesian rotation in the normalised radius; the deck carries the physics clock, the wall clock, their ratio and the step count; a worker built from the two `<script id="dns...">` sources, slices bounded at 120 ms of wall clock. Found a real precision defect on the way in -- `st.depthMm` does not exist on the resolved state, so `h` arrived as NaN -- and the page suite caught the standalone build still carrying a `<script src>` for the solver, twice over |
| S10 | C++ port of the three-dimensional step | todo | bit-for-bit against the JavaScript over a full drive period, as `faraday_disc.cpp` already is |
| S11 | All eight cores | todo | measured speedup against core count; identical answer on any count |
| S12 | GPU render path, and the optional f32 preconditioner | todo | f64 answer unchanged by the preconditioner; render timing measured |
| S13a | The zip, the terminal runner, and the gate that unpacks it | **done** | `faraday/build-zip.mjs` writes `faraday-cell.zip`: 34 files, 516 KB, 31 from the checkout plus a generated README, the single-file page and a generated suite runner. `faraday/check-zip.mjs` (49 checks, 9 s) builds one, unpacks it into a temp directory and runs four suites from THERE -- 10 + 5 + 36 + 102, counts asserted, because a suite that collects nothing also exits zero. Seven injected defects all red, restored green, listed below. `dns/run-cell3d.mjs` runs the solver with no browser at all: ASCII plan view through `etaAt`, energy split, divergence, and the clocks. **This row used to claim a zip already existed and already passed from a fresh unpack**, which was false; it does now |
| S13b | Ship: the README's claims, and the whole suite set from the unpack | todo | the slow suite run from the unpack too, in CI rather than by hand; `dns/check-page.mjs` from the unpack, which needs a browser on the machine doing the unpacking |

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

## S6d: the surface flux is not the traction, and the fix is an identity

Found while writing S6d, before writing any of it, and it changes the design.

`famLaplacian`'s sigma-face flux is exactly `proj * (grad f . N)` with
`N = (-sigma H_r, -sigma H_theta/r, 1)` the sheet's unnormalised normal and
`proj = rn dra dtheta` its projected area -- which is right, because
`dA = |N| dA_proj` and `n = N/|N|`, so `grad f . n dA` is `(grad f . N) dA_proj` with the
normalisation cancelling exactly.

The free-surface condition, though, gives a TRACTION: `tau . n = 2 rho nu E . n`. And for
an incompressible flow with constant viscosity

    2 nu div E = nu grad^2 u        (as volume operators, since div u = 0)

but their FACE FLUXES are not the same: `2 nu E . n` and `nu grad u_i . n` differ by
`nu u_{j,i} n_j`, a term that integrates to zero over a closed surface and does not vanish
face by face. **So substituting the traction for the Laplacian's flux on the surface face,
while every other face carries `grad u_i . N`, is a consistent discretisation of neither
operator.** Two ways out:

1. Rebuild the viscous term as `2 nu div E`, so that every face carries a traction. Correct,
   and a large change: the full strain tensor at every interior face, and S4b's gates redone.
2. Keep `nu grad^2 u` and convert the traction into the flux that operator wants. This is an
   identity, not an approximation, and it needs no elimination:

       E_ij = 1/2 (u_{i,j} + u_{j,i})   =>   grad u_i . n = 2 (E . n)_i - u_{j,i} n_j

   After the tangential projection `E . n = lambda n` with `lambda = n . E . n`, so with the
   unnormalised normal (`E . N = lambda N`)

       grad u_i . N  =  2 lambda N_i  -  u_{j,i} N_j

   and every term of `u_{j,i} N_j` is a derivative the surface strain machinery already
   forms:

       i = r      du_r/dr,  du_th/dr,  du_z/dr
       i = theta  (1/r)du_r/dtheta - u_th/r,  (1/r)du_th/dtheta + u_r/r,  (1/r)du_z/dtheta
       i = z      du_r/dz,  du_th/dz,  du_z/dz

   No division by `1 - |grad eta|^2` anywhere, so the forty-five degree degeneracy never
   appears; the projection is what disposes of the tangential stress, and this identity only
   changes bases.

**Take route 2.** It keeps an operator that is already gated second order at every family
and every deformation, and the conversion is exact.

One thing route 2 needs that is not yet built: `surfaceStrain` is written for the PRESSURE
cell's position, and the flux is wanted at each momentum family's own sigma = 1 face --
`(rf, theta_c)` for u, `(rc, theta_k)` for v, `(rc, theta_c)` for w. So it has to take a
family rather than assume one. That is the first task of S6d, and it is a generalisation of
working code rather than new numerics.

`famLaplacian` also needs its sigma = 1 boundary branch to accept a FLUX instead of a
boundary value, since a traction is a flux; and with that in place the w family's range goes
from `1..nz-1` to `1..nz`, which closes the surface node S5 made a solved advection unknown.

## What S6d built, and one round-off fix that turned out to matter

`surfaceLapFluxes` gives, for each velocity component, the flux famLaplacian's sigma = 1
face wants -- not the traction, for the reason recorded above, but the traction converted by
the identity

    grad u_i . N  =  2 lambda N_i  -  u_{j,i} N_j ,      lambda = n . E . n

after the tangential stress has been disposed of by PROJECTION. Nothing divides by
1 - |grad eta|^2, so the forty-five degree degeneracy never appears.

The nine covariant derivatives are formed once, in `surfaceGradient`, and both the strain
tensor and the flux are built from them: two functions each forming their own nine
derivatives would be two chances to disagree about one.

**The tangential traction is exactly zero, not nearly.** After projection the traction is
2 lambda N, the tangents are (1, 0, eta_r) and (0, 1, eta_theta/r), and N is
(-eta_r, -eta_theta/r, 1) with a vertical part of exactly one -- so t.N is a difference of
equals and vanishes to the last bit, at any slope.

**The flat limit is exactly the two-dimensional solver's condition.** With eta flat the
radial flux collapses to -du_z/dr and the azimuthal one to -(1/r)du_z/dtheta, so the flux the
Laplacian wants gives du_r/dz = -du_z/dr and du_theta/dz = -(1/r)du_z/dtheta: `surfaceSlopes`
in `dns/faraday-disc.js`, exactly and not asymptotically.

### The round-off fix

That last property held only to 2.4e-15 at first, and chasing why found something worth
fixing. `refreshMetric` interpolated the face depth as `(b*lo + a*hi)/(a + b)`, which for
two EQUAL depths rounds twice and lands within an ulp rather than on the common value. The
centred slope then differences two such values over drc, amplifying that ulp by h/drc: a flat
surface carried slopes of about 1e-13 instead of zero. Written as an increment from one end,
`lo + (a/(a+b))*(hi - lo)`, two equal depths interpolate to exactly that depth, and a flat
surface's slopes and normal are now exactly zero and exactly vertical. Both facts are gated.

That fix is the one the injection test says is load-bearing: putting the weighted sum back
leaves every order test passing and is caught ONLY by the two exactness gates.

## What S6e closed, and the two things it did not

**Closed.** famLaplacian's sigma = 1 boundary branch now accepts a FLUX where `bc` takes a
value, because the free surface's condition is a traction and a traction is a flux -- there
is no boundary value that carries one. `viscous` supplies it for u and v, interpolating each
component to where its own face sits: radially for u, azimuthally for v. The hook is gated
exactly, not under refinement: changing the supplied flux by C changes the surface row by
exactly C/(H dsigma) and changes no other row at all.

**Not closed, and measured rather than assumed.**

*w's surface row.* Extending the w family's viscous range from 1..nz-1 to 1..nz was tried and
reverted. With the surface flux supplied ANALYTICALLY, so that the flux itself cannot be the
suspect:

| | interior rows | surface row, at the node | at the CV centroid |
|---|---|---|---|
| flat | 1.99 | 1.16 | 1.40 |
| eta/h = 0.4 | 1.65 | **0.06**, 32% error | 0.06 |

Two things are wrong and only one is understood. The node at sigma = 1 sits ON its control
volume's boundary rather than at its centroid, so a flux balance over that half cell is
second order half a cell away from where the value is read -- which is the flat row's 1.16
against 1.40, and which raises a design question S6f has to answer: is w[nz] a node value or
a half-cell average? The two-dimensional solver next door takes the other route entirely, a
pointwise one-sided second derivative at the node (`lapW`, line 322), which is second order
at the node but not conservative. The deformed case's 0.06 is NOT explained by the centroid
offset and is not yet diagnosed.

Until that is settled the range stays at nz-1, where w's control volumes never touch
sigma = 1, exactly as before S6e. The surface flux itself is right and gated in 8d; u and v
do take it.

*An untested interpolation.* The three flux components are interpolated to their own
families' faces, and nothing checks that interpolation. Proven by injection: replacing u's
mean of the two adjacent cells with one cell's value leaves all 170 checks passing. It is a
second-order error, so only a convergence gate on the composed operator at the surface row
would see it -- and gate 4f excludes the top row. Closing it needs a probe field that
satisfies zero tangential stress at the surface, which is the same construction w's row
needs, so both belong to S6f.

## The surface-row defect, resolved -- and most of it was in the gate

The section this replaces diagnosed a defect in `famLaplacian` from measurements taken over
an INADMISSIBLE TEST SURFACE, and the larger part of what it measured was that surface
rather than the operator. Kept here because the diagnosis was wrong in a way worth not
repeating: rule 3 was written about probe FIELDS, and a surface is a field.

### The gate's own surface was not one the solver can be in

`deform` carried its m = 3 mode as rho^2 and its m = 5 as rho^1. A mode m must vanish like
r^m at the axis or the field is not a smooth function of position -- r is not -- so
`dH/dtheta / r^2`, which every sigma face's azimuthal cross term carries, diverged like 1/r.
That term alone was the whole of the "defect": family w's rows near the surface read
1.4e-2, 8.0e-3, 2.4e-3 over 16/32/64, order 0.84 then 1.75, and with each mode carried as
rho^m the same rows read second order, deformed exactly as flat.

`deform` also had a nonzero radial slope at the rim. A free contact line means a 90 degree
contact angle, deta/dr = 0 at r = R exactly, and `refreshMetric` closes the rim with that
(`Hxr = 0` there), so a surface with a rim slope contradicts its own metric by O(1). Family
w's rim column read 4.39e-2, 2.59e-2, 1.44e-2, order 0.76 then 0.85, against 1.85 then 1.88
with the slope removed. 4e's own surface had the same rim slope, and family u showed it
because its node list reaches the wall, so the wall is an ordinary member of its stencils:
1.79e-3, 7.41e-4, 4.49e-4, order 1.27 then 0.72, where p, v and w read 1.99.

`deform` is now admissible at the axis and has zero radial slope at the rim, and is
normalised by the sup of its shape over the unit disc (0.803963385774, from a 20000 x 7200
scan) so that `ampFraction` is max|eta|/h exactly.

### Three real first-order closures, found underneath that

A row's Laplacian is a difference of two face fluxes over the row's thickness. In the
interior the two fluxes carry the same truncation error and it cancels; at a boundary row
one face IS the boundary -- one-sided stencil, or a flux prescribed by the stress condition
-- and there is nothing for the interior face's error to cancel against, so it is divided by
the row's thickness undiminished. On a graded grid that costs an order, three times over:

| where | with two points | measured | with four |
|---|---|---|---|
| sigma = 0 row, relative error | (ds_neighbour - ds_row)/4 times f_ss | 1.748e-1, 7.679e-2, 3.598e-2, 1.741e-2 over nz = 16..128, against the closed form's 1.730e-1, 7.623e-2, 3.575e-2, 1.731e-2 | 6.98e-5, 6.38e-6, 9.32e-7, 1.76e-7 |
| rim column, family p, flat | the same offset in r | 2.27e-3, order 1.71 then 1.43 | 1.58e-3, order 2.19 then 2.05 |
| surface row, with the flux supplied analytically | the height reconstruction's O(ds^2) in a cross-column difference | 1.68e-2, 1.58e-2, 1.05e-2, order 0.08 then 0.59 | 6.68e-4, order 1.80 then 2.00 |

So `polyDerivAt` -- one generic Lagrange derivative -- and the sigma-face derivative, the
radial face derivative and `colValueAtZ`/`colDerivAtZ` all cubic. The three-point rim
closure and the dead `dfds` are gone with them. 4e's floor is raised from order 1.2, which
is what an operator that loses an order at a boundary can manage, to 1.9.

### One quadrature, in the wrong place

The r and theta faces' one-point rule sat at the node. For the families whose nodes are cell
centres that IS the control volume's sigma centroid, bit for bit; for w, whose nodes are the
sigma faces, it is not, and at the surface the node is a quarter cell above the centroid.
Against the exact average over the half cell the surface row read 1.22e-4, 6.53e-5, 3.43e-5,
order 0.91 then 0.93; with the quadrature at the centroid, second order.

### w is now solved at sigma = 1, and what its unknown means

`FAM.w.sHi` is `nz`. The control volume is the half cell [sc[nz-1], 1] that S5's advection
already uses, because a shared volume is what makes the discrete energy identity exact.

AT THAT NODE THE UNKNOWN IS THE HALF CELL'S AVERAGE, not the point value at sigma = 1, and
the two differ by O(1/nz). A staggered face-centred unknown on the domain boundary BOUNDS
its control volume instead of straddling it, so the flux balance is a statement about the
average, whose centroid is ds_top/4 below the node. No choice of volume closes that gap: a
node cannot be the centroid of a cell it bounds. The alternative is `lapW` in
`dns/faraday-disc.js`, a pointwise one-sided second derivative -- second order at the node
and not conservative -- and an advective term in flux form beside a pointwise viscous one
would leave the energy identity neither exactly dissipative nor exactly conservative.

Gate 4g therefore asserts three things, and the third is the one that turns "first order at
the node" from an excuse into a prediction:

1. second order against the exact average over the half cell (seven-point Gauss-Legendre,
   exact for degree 13);
2. that the gap to the point value at sigma = 1 is the centroid offset in closed form,
   `got = grad^2 f(1) - hw H d(grad^2 f)/dz + O(hw^2)`, to the same order;
3. that the point value alone is first order and larger.

### The azimuthal cross term, and the resolution it needs

The last term to fall, and the one that cost the most. A sigma face's cross terms want the
tangential derivatives AT THAT SHEET'S HEIGHT, and a neighbouring theta column's own sigma
level at that height is off by `|H_theta| dtheta / H` -- measured at 18 to 20 times the top
row's thickness, a ratio refinement does not reduce, since dtheta and ds_top both fall like
1/n. Four things were measured:

| form | surface row, eta/h = 0.4, 16/32/64 (max norm) | order |
|---|---|---|
| common height, stencil anchored on the level | 3.99e-3, 1.91e-3, 5.45e-4 | 1.06, 1.81 |
| common height, stencil bracketing the target | 4.75e-3, 8.02e-4, 3.54e-4 | 2.57, 1.18 |
| sigma coordinates, metric slope consistent with the field's | 1.57e-1, 5.23e-2, 8.94e-3 | 1.58, 2.55 |
| bracketed, with the azimuthal grid resolving the surface slope as well as the radial one | 3.66e-4, 9.41e-5, 2.38e-5 | 1.96, 1.98 |

Anchored on the level, the cubic extrapolates ten stencil widths past its own nodes.
Bracketing removes that; what is left is that the stencil changes between adjacent theta
columns and a theta derivative divides the jump by dtheta.

**The sigma-coordinate form is worse, and the reason is what rule 1 is really about.**
Substituting `f_r|_z = f_r|_sigma - sigma H_r f_z` needs the H-slope in it to be the same
discrete operator as the field's, or the residual is `sigma^2 H_r (H_r - D_r H)`, an O(dr^2)
flux error the surface row divides by its own thickness -- measured 1.75e-1 at order 0.14.
With `D_r H` that residual is identically zero for f = z, whatever the stencil, and
`grad^2 z` reads 1.8e-9 instead of 1.3e-10; but the azimuthal term is then thirty times
worse, because in sigma coordinates the field carries the surface's own azimuthal variation
multiplied by the vertical wavenumber. Here k_z H_theta is 3.15 radians per radian of theta,
so 0.82 radians per azimuthal cell -- barely resolved -- and the four-point difference's
O(dtheta^4 f^(5)) error evaluates to 0.05 against a term of size 2: predicted 0.16, measured
1.566e-1. Radially the substitution is harmless (indistinguishable from the common-height
form) because the radial slope is gentler. At a common height the field varies in theta only
through its own shape, which is smooth. That is the whole of rule 1's content, and it is why
the common-height form stays.

Narrowing the azimuthal stencil does not help: three points and two points both read
3.86e-2 at order 0.68 then 0.83, ten times worse, their O(dtheta^2) truncation being what
the row divides by its own thickness.

**What is left is a property of the grid, not of the operator.** With the azimuthal and
radial surface-slope resolutions matched the surface row is second order (1.99, 1.98 in RMS);
at the renderer's aspect, nth = 1.5 nr, the azimuthal ratio is 3.3 times the radial one and
the row reads 2.41 then 1.74, average 2.07. Both are gated. Whether to REFUSE above some
ratio is S7's to decide, once `step()` exists and the cost of a coarse azimuthal grid can be
measured in the answer rather than in one operator: refusing at nth = 1.5 nr would refuse
the grid the renderer needs, so it is not a free choice.

### A gate gap the regeneration test found, and what closed it

Of the five defects injected for this pass, four were red at once and one -- moving the r and
theta faces' one-point quadrature from the control volume's sigma centroid back to the node --
left all 183 checks PASSING. That change is a real second-order restoration, measured before
it was made, and nothing in the suite saw it undone.

The reason is instructive. 4g's probe carried cos 2theta, and with an azimuthal component in
the field the sigma faces' cross terms are twenty times larger than the quadrature term and
bury it: the surface row read 4.34e-4 at order 2.00 with the quadrature at the node against
4.25e-4 at 1.98 with it at the centroid -- indistinguishable. With an AXISYMMETRIC probe the
cross terms vanish analytically and the quadrature is all that is left: 1.22e-4 at order 0.91
then 0.93 against 4.74e-5 at 2.74 then 2.38. So 4g now runs both probes, and the gate that
catches this defect is the axisymmetric one.

That axisymmetric probe then said the same thing about the azimuthal cross term a third time.
On a deformed surface its cross terms ought to be identically zero, and what survives is the
reconstruction's own error: at the renderer's aspect the row reads 1.37 then 1.14, and with
the azimuthal surface-slope resolution matched to the radial, 3.44 then 2.41. Established by
injection on a disposable copy: zeroing the azimuthal cross term takes the same case to
7.22e-5 at order 2.81 then 2.36, while zeroing the radial one changes nothing.

### What is left there, stated as an open item rather than a tolerance

The surface row's order is limited by the surface slope the grid resolves,
`max(|H_r| dr / H, |H_theta| dtheta / H)`, and by ONE mechanism in both directions: the
bracketed reconstruction follows its target, so its stencil CHANGES between adjacent columns
whenever the target moves by more than a cell, and a derivative divides that jump by dr or
dtheta. Measured, at eta/h = 0.4, against the exact average over the half cell:

| probe | grid aspect | slope ratios theta / r | order |
|---|---|---|---|
| cos 2theta | nth = 1.5 nr | 0.169 / 0.051 | 2.41, 1.74 |
| axisymmetric with radial structure | nth = 1.5 nr | 0.169 / 0.051 | 1.42, 1.17 |
| cos 2theta | nth = 6 nr | 0.098 / 0.093 | 2.35, 1.99 |
| axisymmetric with radial structure | nth = 6 nr | 0.098 / 0.093 | 2.02, 1.59 |

so the azimuthal jumps limit the first two and the radial jumps the last, which is why
matching the ratios does not by itself buy second order once the probe has radial structure.
Every case is gated at the floor it is measured to clear, with those numbers beside it, and
the strong claims are untouched: 4e reads 1.99 to 2.00 at all eight (family, amplitude) pairs
against a floor of 1.9, `grad^2 z` is zero to 1.3e-10, and 4g pins the surface row's centroid
offset in closed form.

**The fix, for a pass of its own.** Remove the jump rather than bound it: a reconstruction
that is C1 in its target position -- a cubic Hermite on the bracketing interval, or two
adjacent cubics blended with weights smooth in the target -- has an error that is a continuous
function of the target, so a difference quotient of it cannot divide a discontinuity. That is
a change to `colValueAtZ` alone, and it is gateable by exactly the table above. It is not
attempted here because S6's remaining work (the flux-interpolation gate and the surface
pressure) and S7 are ahead of it, and because the row is already second order in every case
where the slope ratios are small enough that the stencil does not change -- which is what the
flat cases show, at 1.98 to 2.74.

### The untested interpolation, and an expectation it overturned

`surfaceLapFluxes` gives the flux at a pressure cell; u's sigma = 1 face is at an r face and
v's at a theta face, so each is interpolated, and nothing checked that -- replacing u's with
one cell's value left the whole suite passing. The plan said closing it needed a probe field
satisfying zero tangential stress at the surface. **It does not.** 8d's `wantFlux` is the
analytic flux at any (r, theta), so it can be asked for the face's own position and the
interpolation gated directly against calculus, which is both simpler and stronger than gating
it through a composed operator. `refreshSurfaceFluxes` and `surfaceFluxFace` exist so the gate
can reach it.

The defect expected was not the one found. The arithmetic mean was expected to be FIRST order
at a u face, since an r face is the midpoint of its two cell centres only when the cells are
equally wide and this radius is graded. Measured, it is second order: 6.10e-2, 1.43e-2,
3.46e-3 over 16/32/64, order 2.09 then 2.05, because on a smoothly graded grid the offset is
O(dr^2) and not O(dr). The weighted interpolation -- refreshMetric's own weights for H at an r
face -- reads 5.12e-2, 1.17e-2, 2.80e-3 at 2.12 then 2.07, a nineteen per cent smaller error,
and is kept for that constant, because a surface-flux error is divided by the top row's
thickness in the surface row. A coarser slip than a mean IS first order and the gate catches
it: one cell's value reads 0.92 then 0.96 radially and 0.99 then 1.00 azimuthally.

### The gate that changed, with the measurement

8c pinned EXACT agreement with the two-dimensional solver's `wzSurface`, a three-point
quadratic, and `colDerivAtZ` is now a cubic, so that had to change. It is not a weakening:
by the identity 8e gates, a surface-flux error is divided by the top row's thickness, which
falls like 1/nz here, so a second-order surface quantity leaves the surface row first order.
The two-dimensional solver keeps its quadratic -- it is the independent reference for S8 and
changing it would spend that independence. The check is now made twice and the first is
still exact: on a w profile quadratic in z, which both stencils differentiate exactly, they
agree to 7.7e-13, 2.1e-12, 6.3e-12 relative (a three-point derivative's cancellation gets
relatively worse as the spacing shrinks); on the sinusoid the gap between them falls at
order 2.69 then 2.32, the order the quadratic can promise.

## S7: the file becomes a solver, and two O(1) defects the projection had been hiding

`faraday-cell3d.js` had forty-three methods and not one that advanced anything in time. It was
a library of gated operators, and the two defects below had survived every gate in it because
**nothing used `gradient` for anything except its own transpose**: `divergence(gradient(.))` is
symmetric negative definite for any positive face weight and any consistent divergence, so the
symmetry, definiteness and divergence-removal gates all passed while the gradient itself was
not the pressure gradient.

### The Omega control volume was missing its H

Every other face divided by its physical volume; the sigma faces divided by
`rc dr dtheta dsigma` with no H. Measured on `p = A r + B z`, where every centred difference is
exact, one face returned `-gw = 2.73300e+0` against `B H = -2.73300e+0` at H = 3.00e-3 m: the
vertical pressure gradient was smaller than the physical one by a factor of H, 333 on this
cell. Gate 3 now asks for EXACTNESS there, on a probe linear in z, which is where a missing H
shows as a factor of H rather than as an order.

### The horizontal components were the COVARIANT gradient, not the physical one

`divergence` reads Omega. Written in the physical w it reads `w - sigma(u H_r + (v/r) H_theta)`,
so u and v enter it through that term and the transpose must carry the composition -- returning
part of every sigma face's contribution to the eight r and theta faces that meet there. Without
it the horizontal components are the derivatives at constant SIGMA and not at constant HEIGHT,
and the two differ by `sigma H_r dp/dz`, which is not a correction: measured on the same probe
over a surface at eta/h = 0.4, the radial component read **86.99 where the physical value is
137.00** at r/R = 0.888, sigma = 0.63 -- 36 per cent, and matching `A + B sigma H_r` to five
digits.

**The file's header was wrong about why Omega is the unknown.** It said that carrying w would
cost "the symmetry conjugate gradients needs". It does not: the divergence in the physical
variables is D composed with a change of variable, its transpose is that change of variable's
transpose composed with D's, and `D W^-1 D^T` is symmetric for any diagonal positive W whatever
D is. What carrying w costs is a WIDER STENCIL, and that had a second consequence:
`pressureDiagonal` read its diagonal by eight parity classes on the assumption that the stencil
reaches no diagonal neighbour. With the slope transpose it reaches `delta sigma = +-2` together
with `delta i = +-1`, so the colouring must be (2, 2, 3) in twelve classes. With the old strides
the diagonal came out wrong, the Jacobi preconditioner with it, and a projection asked for 1e-14
left a divergence of **6.7e-2** where the same projection at 1e-9 left 3.3e-8. Gate 3 now
compares the coloured diagonal against the diagonal read one unit vector at a time, exactly, so
a stride that is too short cannot pass.

### And one ordering constraint, established the same way

The axis and wall values must be set BEFORE Omega is formed from the velocity. Omega is derived
from u, v and w, and u at the axis face enters it through the innermost cell's slope term, so
setting that value afterwards leaves the two inconsistent and the next projection undoes the one
before it: measured, a projection asked for 1e-14 reported a divergence of 6.74e-2 where the
same projection at 1e-9 reported 3.40e-8. `step` carries that order, and so does the gate.

### What step() is

Predictor (viscous with the surface traction on the sigma = 1 face, plus the conservative
grid-relative advection) -> prescribed values -> Omega -> the projection with the surface
pressure as its inhomogeneous DIRICHLET value -> corrector on the physical velocity -> eta on
the corrected Omega at sigma = 1, which IS the kinematic condition.

`p_s = rho g_eff eta - gamma kappa + 2 rho nu (n.E.n)/|n|^2`, the same expression
`dns/faraday-disc.js` carries with its two right-hand terms linearised; nothing is linearised
here. It is the Dirichlet value and never a predictor force, because as a predictor force it is
O(1/dsigma) and the projection cancels almost all of it -- which made the two-dimensional solver
non-finite 0.57 periods in.

### The check that everything else was for

Released from rest with `eta = eps J_m(kr) cos(m theta)` and `J_m'(kR) = 0` -- the free contact
line, so the profile is admissible -- one step must leave `d eta/dt = -omega^2 eta dt` with
`omega^2 = (g k + gamma k^3/rho) tanh(k h)`. That single step runs the curvature, the normal,
the strain, the viscous normal stress, the surface pressure, the pressure solve, the corrector
and the kinematic condition. Measured, m = 3, omega = 70.709 rad/s:

| grid | measured/theory | error |
|---|---|---|
| 12x16x10 | 0.86886 | 13.11% |
| 16x24x12 | 0.93382 | 6.62% |
| 24x32x16 | 0.96077 | 3.92% |
| 32x48x20 | 0.97921 | 2.08% |

First order, and that is expected rather than disappointing: the capillary term is about half
the restoring force at this wavenumber and the curvature's rim closure is first order (7d).

### The step limit, and why it is the dispersion relation

`dns/faraday-disc.js` takes the surface stiffness to be `(g + gamma k^2/rho)` divided by the top
half cell's thickness. That is right for its scheme and too strict for this one, because the
projection distributes the surface pressure through the whole column and the surface responds at
the PHYSICAL frequency -- which the table above measures directly. So the capillary limit here is
`2/omega` with omega from the dispersion relation at the largest wavenumber the grid carries,
`k^2 = 4/dr_min^2 + (nth/2)^2/rc[0]^2`. Every azimuthal mode is present at once, so unlike next
door there is no mode number to be lucky about.

Still conservative, and the margin is measured rather than assumed: on 10x16x8 the scheme is
stable at 1.35 and 2.03 times that limit and diverges at 2.71; on 14x24x10 stable at 1.40 and
divergent at 2.80. With the safety factor 0.4 the default step is 3.5 to 7 times below where the
scheme breaks, and gate 9b asserts both sides -- no growth at `stableStep()`, divergence at ten
times it (measured: by step 14 and step 15 on the two grids).

### The energy

Kinetic by the same control volumes the projection weights its faces with, which is what makes
the projection non-increasing in THIS kinetic energy rather than some other one; hydrostatic
against the instantaneous effective gravity; capillary as gamma times the surface's excess area,
whose variational derivative IS the curvature the surface pressure carries (7c).

Measured, released from rest with eta = 1e-7 J_3(kr) cos(3 theta) on 10x16x8 and nu = 1e-12, over
a quarter of the 88.860 ms period in 377 steps: the total goes 1.8784e-15 -> 1.8777e-15 J, a drift
of -0.0333%, and it falls at every step rather than rising anywhere -- worst single-step rise
1.52e-3% of the initial total. The kinetic part reaches 99.20% of that initial total, which is the
whole point of running a quarter period rather than a fixed number of steps.

(Those three figures read -0.0343%, 5.22e-3% and 99.19% when S7 landed, and the difference is the
capillary energy, which is now the surface's excess area computed directly instead of the
difference of two areas agreeing to thirteen digits. The defect that change fixes, and why nothing
about the solver was wrong, is under "S8a" below.)

**That last figure is why this gate was restructured, and the first version of it is worth
recording.** It ran two hundred steps on 14x24x10 and asserted that the kinetic energy took "a
large share" of the total, reading 14.70%. Nothing was wrong with the solver: two hundred steps at
that grid's `stableStep` is 5.714 ms of an 88.860 ms period, six per cent of it, and
sin^2(2 pi 0.0643) = 15.5% -- so 14.70% is very nearly the right answer to a question about six
per cent of a period. The defect was in the gate, and it was not a loose threshold but a
meaningless one: no number that can be asserted about 14.70% distinguishes an oscillator from a
scheme leaking a little energy into motion. A quarter period is the only duration at which the
assertion says something, because at a quarter period an oscillator must have converted nearly all
of it and anything else cannot.

### What the regeneration test found about these three gates

Five defects, one at a time, on a worktree under `/tmp`; all five red on the gate they target
and 214 passed, 0 failed on restore. Two of them said something about the gates rather than
about the solver, and both are worth keeping.

| injected | what went red |
|---|---|
| the hydrostatic head dropped from the surface pressure | gate 9's absolute check, 54.47% error at 32x48x20; 9c's drift, -38.84% |
| `stableStep` ten times too large | 9b's "no growth at 1x", Infinity on both grids, and then the floor-contact refusal fires in 9c |
| `stableStep` ten times too small | 9b's "and at ten times it does", 0.9974 per step on both grids |
| eta advanced on the PRE-corrector Omega (forward Euler on the surface) | 9b's "no growth at 1x", 1.0335 per step; 9c's drift, +6.3e+9%, its direction, and its single-step rise |
| no surface pressure at all | gate 9 entirely, 100% error and zero response; 9c's exchange, 0.00% kinetic |

**Gate 9's two assertions are not redundant, and the convergence one is the weaker.** With the
hydrostatic head missing the error still fell on every refinement -- 62.13%, 57.70%, 55.77%,
54.47% -- so the sequence was monotone and the check passed while the surface pressure was
missing a quarter of itself. Only the absolute bound caught it. A convergence assertion says the
discretisation is consistent with *something*; it does not say what.

**And one assertion passed for exactly the reason it exists to exclude, so it was rewritten.**
"By the quarter period nearly all of it has become kinetic" was `kemax > 0.9*e0.total`. Under
forward Euler on the surface the energy grew by a factor of 6.3e+7, and 3.4e+9 times the initial
total is indeed more than nine tenths of it, so that assertion went GREEN on a scheme that was
manufacturing motion out of nothing. The other three in 9c went red, so the gate caught the
defect -- but the assertion that names the exchange did not, and a one-sided bound on a quantity
that can run away is not a bound. It is now bracketed above by the same drift tolerance:
between 90% and 100.5% of the energy the surface started with, measured 99.19%. Re-injected
against the bracket, it is red.

## S7b: the step made affordable, and not one bit changed

S8 is a comparison against `dns/faraday-disc.js` over drive periods, so before writing it I
measured what a period costs. At 16x24x10, the coarsest grid that carries m = 3 with eight
cells per azimuthal wavelength, **one oscillation period of the m = 3 mode was 3876 steps at
146 ms each: eleven minutes**. At 20x32x12 it was thirty-six. Several grids times several
periods times two solvers is not a stage, it is a week, so the cost had to be understood
before the physics could be validated.

Profiled with V8's sampler over a hundred steps rather than guessed at:

| | before | after |
|---|---|---|
| `colValueAtZ` | 38.5% of the step | (still the largest, 35%) |
| `gradient` | 16.4% | 13.1% |
| `Hat` | 12.7% | gone from the profile |
| one `applyL` (the CG matvec) | 0.50 ms | 0.16 ms |
| a step at 16x24x10 | 146 ms | **85.6 ms** |
| a step at 20x32x12 | 316 ms | **174 ms** |
| the whole gate | 3m00s | **1m18s** |

None of it was arithmetic. Three things were being done again and again that did not need
doing once:

- **`divergence`, `gradient` and `omegaOf` are the conjugate-gradient matvec**, so they run a
  hundred and more times per step, and each was calling `this.ip/iu/iv/iw` -- a wrapped modulo
  behind a method call -- six to fourteen times per cell. Every one of those indices is
  `(i*nth + kw(k))*stride + j` with `kw(k) = k` for the loop variable, so the column base is
  one multiply hoisted to the k loop and the wrap is one comparison for `k+1`.
- **`colValueAtZ` allocated a closure per call** over `fam.idx`, and called `Hat`, which
  searched the radial node list linearly from the start every time -- at the rim node that walk
  is the whole list. The bracket for a family node is fixed at construction, so the constructor
  now tabulates it; `HatHBr` takes it, `HatH` finds it as before for an arbitrary position.
- **`Hat` allocated one three-field object per face per stencil point.** `HatH` returns the
  scalar for the five call sites that want only H, `HatInto` fills a caller-owned triple for
  the three that want the slopes.

**The gate for this is bit-for-bit equality, and it is not a tolerance.** Section 10 carries
its OWN implementation of each operator -- the obvious loop, written against the public index
accessors and `Hat` -- and asserts `===` on a deformed surface with every degree of freedom
excited. That is the only assertion that can distinguish "faster" from "different", and one
change was caught by it while the work was being done: hoisting `(vv/rc[i])*Hdth[e]` as
`vv*(Hdth[e]*(1/rc[i]))` is the same number in exact arithmetic and a different one in doubles,
and two of the sixteen digits of eta[0] moved. Reverted; the loops now hoist only what can be
hoisted without reordering a single operation.

### The regeneration test found a probe that could not see its own defect

Seven injected defects, each red on the assertion it targets, 221 passed 0 failed on restore.
The third was **green at first, and the reason is worth more than the fix.** It replaced `h01`
by `h00` in `HatHBr` -- an outright wrong azimuthal interpolation node -- and the section did
not notice, because of what the probe positions were:

- a family node with `thOff = 0.5` sits on a theta cell centre, so the bilinear azimuthal
  weight `ft` is exactly **0**;
- a family node in r sits on `rx[a+1]`, so the radial weight `fr` is exactly **1**;
- and the off-node positions I had added used `theta = (k + 1/2) dtheta`, which is a cell
  centre too, so `ft` was 0 at every single one of the 602 points.

Each of the eight bilinear products is multiplied by zero somewhere on that set, so a wrong one
is invisible. The probe now sweeps azimuthal offsets 0, 0.19, 0.5 and 0.73 of a cell and radial
positions strictly between nodes -- 910 points, deliberately not 0, 1/2 or 1 -- and the same
injection reads 1.31e-4. This is the same lesson as the axisymmetric probe in S6f and the
one-sided input range in `CLAUDE.md`: **a probe that lands on the symmetry points of the thing
it is testing tests nothing.**

### What was NOT done, and why

`colValueAtZ` is still the largest single cost, and what is left in it is twelve divisions per
call in the Lagrange product, `L *= (ss - sn[j0+m])/(sn[j0+i] - sn[j0+m])`. The denominators
depend only on `j0` and could be tabulated as reciprocals -- but `a/b` and `a*(1/b)` do not
round alike, so that would forfeit the bit-for-bit argument and put every measured order in
this file back in question for perhaps another 20%. It is not taken. The answer to the step
cost is S10 and S11, where the same discretisation compiled and spread over eight cores buys
far more than reassociating a product.

## S8a: the two solvers agree on a number that has no closed form

This is the check the file exists to pass, and it is the first one that is not against calculus,
against an identity, or against itself. `dns/faraday-disc.js` is a different solver: written
separately, linear rather than nonlinear, one azimuthal mode at a time rather than all of them,
and -- the part that makes the comparison mean anything -- **in a different vertical
coordinate**. It solves in z on a fixed grid with the surface conditions applied at the top;
this one solves in sigma = z/H on a grid that follows the surface. They share the physics and
the cell, and nothing else.

The quantity is the total mechanical energy's fractional decay over a fixed physical window,
`-ln(E/E0)/(2t)`, from a state released from rest in one Bessel mode. It is chosen because there
is **no closed form for it**. The frequency has one and gate 9 already checks against it. The
dissipation is set by the Stokes layers at the floor, the sidewall and the surface, and the only
reference for it is another solver -- the disc gate measures the exponent of its own
nu-dependence at 0.755, between the bulk term's 1 and a pure boundary layer's 1/2, so most of
this number comes from exactly the part the two codes do differently.

**It is not the asymptotic modal damping rate, and this section called it that until the number
was measured.** Over a quarter period the energy is still sloshing between kinetic and potential
and the boundary layers are still forming, so the ratio is a transient. The same figure on
`dns/faraday-disc.js` at 20x12 over lengthening windows:

| window | 0.125T | 0.25T | 0.5T | 1T | 2T | 4T | 8T |
|---|---|---|---|---|---|---|---|
| rate | 0.641 | **1.334** | 1.242 | 1.468 | 1.522 | 1.553 | 1.565 |

and that code's own Floquet multiplier for the same grid gives 1.513. The quarter-period value
is 1.334 -- exactly the number the table below compares, and fifteen per cent below where the
sequence is heading.

That does not weaken the comparison and arguably sharpens it. What is compared is a well-defined
functional of one initial-value problem: identical initial condition, identical physical window,
matched nr and nz. It includes the transient formation of the boundary layers, which is where
two discretisations of a viscous free surface differ most, rather than only an asymptotic
eigenvalue. What it must not be called is the modal damping rate. S8c is where the asymptotic
quantity gets compared, against `floquetDisc`.

m = 3, n = 1, free contact, eta = 1e-9 m, a quarter of the 88.860 ms period, matching nr and nz:

| grid | disc | cell3d | gap |
|---|---|---|---|
| 12x8 / 12x20x8 | 1.26121 | 1.33232 | +5.64% |
| 14x10 / 14x24x10 | 1.29079 | 1.32578 | +2.71% |
| 16x10 / 16x24x10 | 1.30927 | 1.32165 | +0.95% |
| 20x12 / 20x32x12 | 1.33430 | 1.33377 | **-0.04%** |
| 24x14 / 24x40x14 | 1.35070 | 1.34333 | -0.55% |

Each code is still moving in its own grid over that range -- the disc from 1.261 to 1.351, this
solver from 1.332 to 1.343 -- so the four parts in ten thousand at 20x12 is the two of them
converging to the same limit from opposite sides, not either one being right. m = 2 was measured
too: +6.13% at 16x10, +3.35% at 20x12, converging the same way. The first three pairs are gated
in section 12; the last two cost 288 s and 862 s and are recorded here instead.

### The comparison found a defect, and it was in the energy diagnostic

The first run reported the 3-D solver's rate as **-2.034 s^-1: a growth**, against the disc's
+1.309. At the real viscosity the energy rose 9.4% over a quarter period.

It was not the solver, and the sequence that established that is worth recording, because the
first suspicion -- the viscous surface traction, which is the newest and most intricate thing in
the file -- was wrong. The rate at nu = 1e-12, 1e-10, 1e-8, 1e-7 and 1e-6 read -2.750, -2.749,
-2.711, -2.541, -2.034: the growth is **already there at nu = 1e-12**, and raising nu makes it
monotonically less negative by 0.72 s^-1, which is viscosity damping correctly on top of it. So
the viscous term was doing its job and something else was adding energy.

What was adding energy was the *measurement*. `energy().capillary` was
`gamma*(surfaceArea() - PI*R*R)`, and at eta = 1e-9 m those two operands are both
4.618632074629e-4 and their difference is 1.2e-18 -- twenty times a double's own resolution at
that magnitude. The excess area came out 1.1926e-18 where its own amplitude scaling demands
1.1596e-18, 2.8 per cent wrong, and some runs reported `kinetic/total` of 1.29 and 1.82, which
is a negative potential energy. Sweeping the amplitude settled it: the drift over a quarter
period reads -0.0344, -0.0349, -0.0356, +0.1166 and +25.256 per cent at eta = 1e-5, 1e-6, 1e-7,
1e-8 and 1e-9 m. Flat over three decades and then nonsense -- a diagnostic breaking down, not a
scheme.

`surfaceExcessArea` sums `rc drc dtheta (sqrt(1+q) - 1)` cell by cell as `q/(1 + sqrt(1+q))`,
the same number in exact arithmetic and one that keeps every digit for small q. `Sum rc drc
dtheta` is exactly `pi R^2` -- `rc drc` telescopes to `R^2/2`, `dtheta` sums to `2 pi` -- so
nothing is approximated by taking the excess term by term. The drift now reads -0.0344,
-0.0349, -0.0349, -0.0349, -0.0349 across the same five decades, and the damping rate came out
+1.32165 against the disc's +1.30927.

**Gate 9c never saw any of this because it happens to be written at eta = 1e-7 m**, the last
amplitude where the subtraction still had digits. A diagnostic that silently loses its
significance below a bound nobody has written down is worse than a wrong one, because every
energy claim made with it is conditional on that bound. Section 11 now gates the excess area
against the amplitude-squared law over eight decades -- the excess form departs by 2.85e-10, the
difference form by 9.90e+1 -- and gates that the drift over a fixed time is the same at 1e-5,
1e-7 and 1e-9 m, which is what makes it a property of the scheme.

One expectation of mine was overturned in passing: I bounded `surfaceExcessArea() + pi R^2`
against `surfaceArea()` at a relative 1e-15 and it failed at 1.07e-15. That is about ten units
in the last place accumulated over a 240-cell sum, which is what a correct sum does; the bound
is 1e-14 now, still thirteen orders tighter than the injections it has to catch.

### What this gate can and cannot resolve, measured

- Free-slip instead of no-slip at the floor and sidewall -- removing the layers that carry most
  of the damping -- takes the rate from 1.326 to 0.526: a **-59%** gap, red at every pair.
- Halving the viscosity in the **radial predictor alone** moves it by about three per cent, and
  that was **GREEN** on the first version of section 12, which stopped at 14x24x10 inside a four
  per cent window: 1.32578 became 1.28233. It also made the *coarse* pair agree better, so a
  two-point convergence test passed as well. The third pair is there because of that: at
  16x24x10 the clean gap is +0.95% and the defect reads -2.47%, so a two per cent window
  separates them, and the convergence assertion now runs over all three. Red on both.

So: this gate resolves a defect of a few per cent in the damping and not one of a few tenths.
That is its measured power and it is stated in the section rather than implied by a threshold.

## S8b: the nonlinearity, and the term the whole suite could not see

Two claims, both falsifiable by a SCALING LAW rather than by a threshold on a value -- which
matters, because a harmonic's size depends on the grid and the window and its exponent in the
amplitude does not.

**One mode in, its own harmonics out.** A quadratic nonlinearity acting on cos(3 theta) gives
cos^2 = (1 + cos 6 theta)/2, so a single m = 3 mode must generate m = 0 and m = 6 at order A^2
and m = 9 at order A^3. Measured on 10x24x8 over a tenth of a period, at eta/h = 0.2, 0.1 and
0.05, as the r-weighted L2 norm of each azimuthal component of eta:

| | m = 0 | m = 3 | m = 6 | m = 9 |
|---|---|---|---|---|
| eta/h = 0.2 | 4.896e-9 | 1.273e-6 | 1.485e-8 | 3.002e-11 |
| eta/h = 0.1 | 1.226e-9 | 6.361e-7 | 3.718e-9 | 4.156e-12 |
| eta/h = 0.05 | 3.066e-10 | 3.180e-7 | 9.297e-10 | 5.342e-13 |
| exponent | **1.998/1.999** | 1.001/1.000 | **1.998/1.999** | **2.853/2.960** |

The seeded mode is linear, the two quadratic channels are square, and m = 9 is cubic, which no
quadratic term can produce. All of it grew from 1e-22: the harmonics before the first step are
at round-off, thirteen orders below where they finish. A linear solver returns exactly zero for
every one of them.

**Two modes in, their sum and difference out.** m = 2 and m = 3 together must give m = 5 and
m = 1, at order a2 a3, and those two channels are reachable from neither parent alone:

| | m = 1 | m = 5 |
|---|---|---|
| a2 = a3 = 0.1 h | 2.563e-9 | 7.358e-9 |
| a2 halved | 1.294e-9 | 3.681e-9 |
| a3 halved | 1.282e-9 | 3.681e-9 |
| m = 3 alone | 1.956e-22 | 3.722e-22 |
| m = 2 alone | 2.463e-22 | 5.642e-22 |

Halving either parent divides the child by 1.9803, 1.9991, 1.9992, 1.9992; removing either
parent drops it thirteen orders to round-off. And m = 1 arrives with no special case anywhere,
which is worth noting because `dns/faraday-disc.js` REFUSES m = 1 outright -- its radial
singular group `-[(m^2+1)u + 2m v]/r^2` stays finite there only through a cancellation its flux
form does not impose. Here it is a product of S3 having made the axis a reflection.

### The advective term can be deleted from step() and 239 checks pass

That was the regeneration test of the section above, and it came back **green**. `au`, `av` and
`aw` removed from all three predictors: the harmonics moved by under three per cent -- 4.930e-9
against 4.896e-9 for m = 0 -- and every exponent stayed right at 2.001/2.000, 2.000/2.000,
2.997/2.999. Freezing the metric flat instead cut the harmonics by a factor of **28**, to
1.769e-10, and the exponents still read 1.998/2.000 and the coupling still halved on halving
either parent.

The arithmetic says why. At eta/h = 0.2 the surface velocity is of order omega eta ~ 2e-2 m/s,
so `u.grad u` is of order u^2/R ~ 3e-2 m/s^2 against a gravity-capillary acceleration of
omega^2 eta ~ 3 m/s^2: **one part in a hundred**, and its share of a quadratic harmonic is a
correction to a correction. The harmonics at that amplitude come from the surface -- the metric
H = h + eta, the full mean curvature, the traction on a sloped face.

**An exponent is a structural property.** It says a quadratic coupling exists; it is blind to
which term supplies it and to how large the result is. Section 13's claim -- that this solver
couples modes where the linear one cannot -- is true, and it is not the claim that every
nonlinear term is present. `advect` itself is thoroughly gated in section 6 (the net-flux
identity to 1e-16, exact telescoping, the curvature pair cancelling to 7e-18, second order
against calculus, seven injected defects red). What had no gate anywhere was that **`step`
uses it**.

### Section 14: step() is the composition it documents, term for term

The section reassembles one step from the solver's own public operators -- `omegaFromW`,
`surfacePressure`, `viscous`, `advect`, `divergence`, `pressureDiagonal`, `solveP`, `gradient`,
`axisU` -- in the order the file documents, on a deformed surface at eta/h = 0.35 with a
divergent random velocity field, and asserts all eight state arrays are **bit for bit** what
`step` produces.

It is a composition check and not a physics check. The physics of each operator is sections 1
to 8; the physics of the assembly is section 9 against the dispersion relation and section 12
against another solver. What it catches is a term dropped, added or scaled, the surface pressure
moved out of the projection, the axis prescribed after Omega instead of before, or the kinematic
update taken from w instead of the corrected Omega. Four injections, all red:

| injected | what section 14 reports |
|---|---|
| the advective term dropped -- the defect the suite missed | every field differs: u 896/1008, v 896/896, w 896/1024, Omega 896/1024, eta, p, H, Ht all |
| the axis prescribed after Omega instead of before | every field differs, identically wide |
| eta advanced on the physical w instead of the corrected Omega | eta 128/128 and H 128/128 -- exactly the two it should touch |
| the surface pressure left out of the right-hand side top row | every field differs, and seven other sections go red too |

The third is the one to read carefully: it changes eta and H and **nothing else** on the first
step, which is right, and it is the kind of defect that takes many steps to show anywhere else.

**The general lesson, and it is the fourth time this session.** A gate that measures a
*property* of the answer -- an exponent, a convergence order, an invariant -- cannot tell you
which code path produced it. Sections 10 and 14 are the two that ask the other question, "is
this the computation it says it is", and both were written after something got past everything
else: section 10 after the index hoisting could have changed bits silently, section 14 after an
entire nonlinear term did.

## S8c: the driven growth rate, against the Arnoldi multiplier next door

Section 12 compares the two solvers with the drive off. S8c turns it on, which is the regime the
renderer runs in: the cell shaken vertically at twice a mode's frequency, where that mode grows
out of nothing. The subharmonic point for m = 3, n = 1, at a = 20 m/s^2, comfortably above
threshold.

**The reference is `floquetDisc` in `dns/faraday-floquet.js`**, which does Arnoldi on the disc's
period map rather than integrating a seed, so it returns the asymptotic multiplier directly:
|mu| = **2.08649212** at nr = 20, nz = 12, with a Krylov spread of 5.81e-6.

Matching that grid, this solver integrated six drive periods at 20x24x12 -- 2787 steps and 400 s
each -- and the per-period amplification of the m = 3 component of eta:

| period | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| \|mu\| | 1.235690 | 1.762186 | 1.987387 | 2.041031 | 2.053208 | **2.055543** |
| growth s^-1 | 4.763 | 12.752 | 15.458 | 16.058 | 16.192 | 16.217 |

Increments 0.0536, 0.0122, 0.0023: converging, and the limit is **1.49 per cent below** the
Arnoldi value for the same nr and nz. That residual is the discretisation difference between the
two -- same radial and vertical resolution, a different vertical coordinate, and an azimuthal
direction one of them does not have.

**The first period is not the multiplier**, and the sequence above is why the gated window is
three periods and not one. Released from rest the initial condition is not the Floquet
eigenvector, which carries a particular phase between eta and the velocity field, so the first
period's ratio is a projection onto both Floquet modes; the growing one takes over as
(2.09/0.4)^n.

### What is gated, and why the Arnoldi comparison is not

`floquetDisc` refuses every grid coarser than 20x12 for this mode -- its surface-operator
tolerance is one per cent and 18x10 misses by 1.12, 16x10 by 1.37, 14x10 by 1.71 -- and matching
20x12 costs this solver 400 s per drive period. Six periods is 40 minutes. So the Arnoldi
comparison is a recorded measurement and section 15 gates the affordable equivalent: the disc's
own TIME-DOMAIN driven run, which has no such grid restriction, beside this solver at a matched
12x8 / 12x20x8, three periods each.

| | period 1 | 2 | 3 | total |
|---|---|---|---|---|
| disc 12x8 | 1.24704 | 1.78613 | 2.01139 | 4.4801 |
| cell3d 12x20x8 | 1.22821 | 1.73828 | 1.96195 | 4.1887 |
| gap | | | **-2.46%** | **-6.50%** |

The coarse grid tracks the fine one closely -- 1.22821, 1.73828, 1.96195 against 1.235690,
1.762186, 1.987387 -- which is what makes the cheap gate worth having.

### Off resonance, the same drive must do nothing, and the per-period ratio stops meaning anything

Detuning omega_D by 35 per cent at the same amplitude:

| | period 1 | 2 | total |
|---|---|---|---|
| disc 12x8 | 0.73793 | 0.24879 | 1.836e-1 |
| cell3d 12x20x8 | 0.68113 | 0.03022 | 2.058e-2 |

Both amplify nothing, which is what separates parametric resonance from a drive that pumps energy
into whatever is there. Note what the per-period ratio does off resonance in a three-period
window: 0.73793, 0.24879, **2.12167** for the disc and 0.68113, 0.03022, **28.34476** for this
solver. Nothing is growing -- the amplitude beats instead, and a period boundary landing near a
node makes the ratio meaningless. The measure has to be the amplification over the whole window,
and that is what is asserted.

### Regeneration

| injected | on-resonance mu per period | red |
|---|---|---|
| the drive dropped from the effective gravity | 0.92913, 0.89060, 0.86590 (total 0.7165) | 4 of 5 |
| the drive frequency doubled, so the subharmonic point is not one | 0.93930, 0.91614, 0.91171 (total 0.7845) | 4 of 5 |
| the drive amplitude 30 per cent too large | 1.43678, 2.26450, 2.50018 (total 8.1346) | 3 of 5 |

The third is the discriminating one and it is why the cross-code assertions are there at all. The
mode still grows, so "a drive at twice the mode frequency makes the mode grow" stays GREEN and so
does the off-resonance contrast -- a qualitative gate would have passed a thirty per cent error in
the drive. What catches it is the agreement with the other solver, at +24.30% on the third
period's multiplier and +81.57% on the window, and the climb toward the Arnoldi value, which
2.50018 overshoots rather than approaches from below.

Clean: 247 checks, 0 failed, 11m37s.

## S9a: eta where the renderer asks for it, and one ulp that was a real defect

The page draws a GR x GR Cartesian raster; the solver holds eta on a graded polar grid.
Something has to resample, and the point of `etaAt` is that it is **not a new interpolation**.

It reads the extended grid that `HatH` already reads, because that grid carries the two
conventions a renderer would otherwise reinvent, and reinvent differently:

- the row below the axis is the **antipodal continuation**, `eta[ie(0, k + nth/2)]` at
  r = -rc[0], so a point near r = 0 is interpolated ACROSS the axis rather than extrapolated up
  to it;
- the row at r = R is the **contact condition itself**: the last cell's own elevation under a
  free line, zero under a pinned one.

A renderer interpolating eta on its own would have to reproduce both to draw the surface the
solver is solving, and any difference would appear as a defect in the physics rather than in the
drawing.

### One ulp, and it was not cosmetic

The first version was `etaAt(r, th) = HatH(r, th) - this.h`. Section 16 caught it on its first
run:

| | measured | owed |
|---|---|---|
| at a cell centre | off by 6.505e-19 | exact |
| on the axis, axisymmetric surface | spread 4.337e-19 over theta | one value |
| at the free rim | off by 5.421e-19 | exact |

All three are **one ulp of h + eta**. With h = 3e-3 m and the elevations a renderer draws at
1e-4 m and below, that sum has an ulp of 4.3e-19 and the subtraction cannot give the low bits
back. It is the same defect section 11 found in the capillary energy -- a small quantity
reconstructed as the difference of two large ones -- and it would have put a floor of 4.3e-19 m
under every elevation the page could draw. That is 1.4e-16 of the depth, which sounds harmless
until you notice the amplitudes S8 validates the solver at: 1e-9 m. The renderer would have been
drawing round-off.

So eta has its own extended grid, `Ex`, filled from `eta` and never from `H`. Afterwards the axis
assertions are satisfied exactly: spread **0.00e+0** over theta for an axisymmetric surface, and
9.26e-23 against a 1.25e-5 mid-radius amplitude for an m = 3 one. The step fingerprint is
bit-for-bit unchanged, so nothing else in the solver moved.

### And one expectation that cannot be met, corrected rather than engineered around

The cell-centre and free-rim checks still asserted `=== 0` and still failed, at 5.421e-19 and
4.336e-19. That residual is the **probe**, not the code. The sample position is `(k + 0.5)*dth`,
and recovering k from it inside `etaAt` costs a multiply and a divide: `th/dth - 0.5` is k plus a
few ulp, not k, so the floor and the remainder put the azimuthal weight a few ulp off the corner
instead of on it, and what survives is that weight times the difference between two neighbouring
elevations. 5.421e-19 of a 1.188e-3 deformation is 4.56e-16 relative -- four ulp of the
amplitude. The bound is now 1e-13: three orders above that residual and twelve below any defect
the section has to catch. Asking for zero was asking the coordinates to express something they
cannot.

The **pinned** rim is still asserted exact, and is: both corner values of that row are zero and
no weight can make anything else of them.

Second order between centres against the analytic surface: 1.072e-4, 2.747e-5, 7.264e-6 over
12x20, 24x40, 48x80, order 1.965 then 1.919 -- the bilinear interpolation's own order and not
better.

### Regeneration: four injections, and two of them were about the gate

| injected | what section 16 reported |
|---|---|
| the eta row below the axis at the same azimuth, not the antipode | **GREEN at first.** The m = 3 elevation on the axis went 9.26e-23 -> 4.68e-8, fifteen orders, and the threshold was two per cent of the 1.25e-5 mid-radius amplitude, so 4.68e-8 slipped under 2.5e-7 by a factor of five. Re-run against the round-off bound: RED at 4.677e-8 of 1.250e-5, **3.742e-3 relative** |
| the rim row extrapolated instead of carrying the free contact condition | RED, 2.322e-6 of a 1.040e-3 rim elevation, 2.234e-3 relative |
| `etaAt` reverted to `HatH - h` | RED, on the axis spread: 4.337e-19 where one value is owed |
| the radial weight snapped to the nearer node | RED on the order: 1.293 then 0.925, against the clean 1.965 then 1.919 |

**The axis bound had to become round-off, not a percentage.** "An m = 3 mode has no elevation on
the axis" was written as though it were an approximate statement. It is not. At r = 0 the radial
weight is `(0 - rx[0])/(rx[1] - rx[0])` = 1/2 **exactly**, and the two rows it averages are
`eta[ie(0, k)]` and `eta[ie(0, k + nth/2)]`, which for m = 3 differ by cos(3 pi) = -1 and sum to
zero to the last bit of the cosine. So the bound is 1e-12 relative: six orders above the clean
7.41e-18 and nine below the defect's 3.7e-3. That is the third time this session an injection has
shown a threshold loose enough to be meaningless -- after `kemax > 0.9*e0.total` passing on a
scheme that manufactured energy, and section 12's four per cent window swallowing a three per cent
error in the radial viscous term -- and the second in this section alone.

Clean, with the tightened bound: **254 checks, 0 failed**, and the axis figure reads 9.26e-23,
7.41e-18 of the mid-radius amplitude.

**And the cell-centre assertion cannot resolve the `HatH - h` defect at all.** It reads 5.48e-16
with the defect against 4.56e-16 clean: twenty per cent apart, which no honest threshold
separates. What catches that defect is the axis spread, because that assertion demands exact
round-off rather than a fraction of something. Worth recording because it is not the check one
would guess: the section works, and not through the assertion whose name mentions exactness.

### What S9 still needs

`etaAt` is the resampler. The rest of S9 is the wiring: a worker carrying `faraday-cell3d.js`
that steps and streams eta back, the page calling `etaAt(RAD[i]*R, ANG[i])` per covered pixel in
place of the modal superposition in `buildSurface`, the physics and wall clocks with their ratio,
and a grid sizing rule for the three-dimensional case. The existing Floquet panel in
`cymatic.html` already has the worker pattern to follow -- `ensureWorker` builds one from the
`<script id="dns...">` sources -- so S9b is that panel's structure applied to a running
simulation rather than a one-shot solve.

## The zip: what was claimed, what was false, and what is there now

**It did not exist, and this file said it did.** The S13 row asserted that a
`faraday-cell.zip` already existed and already passed its suites from a fresh unpack --
"as `faraday-cell.zip` already does". Nothing on disk, nothing in git history;
`faraday/build-standalone.mjs` emits `faraday-cell-standalone.html`, one file with the
scripts inlined, and no archive. The claim was borrowed from the shape of the sentence
around it and was found the moment the owner asked where the zip was.

**It exists now, it is generated, and it is not committed.** `node faraday/build-zip.mjs`
writes it; `.gitignore` carries it for the same reason it carries the single-file build.
A committed zip is a second copy of every file inside it, and the first time one side is
edited the copy drifts -- and a drifted copy that still *runs* is the worst failure mode
this repository has, because it returns an answer.

**A zip is only ever wrong in what is missing from it, and everything missing still
builds cleanly.** So the gate does not check a zip, it checks an unpack: build, unzip into
a temp directory, compare every file against the checkout byte for byte, rebuild the
single-file page and compare it, read the README and require every path it names to
resolve, run the terminal runner, run four suites from inside that directory. Seven
injected defects, each required to go red on a disposable copy and green on restore:

| injected | what went red |
|---|---|
| `dns/faraday-disc.js` dropped from `MANIFEST` | 7 checks: the terminal runner cannot load the solver out of the unpack at all, and the README names the file |
| a suite file in the checkout that `SUITES` does not list | 1: the generated runner would ship without it, which is the silent omission a runner must never make |
| `faraday-cell3d.js` left out of the single-file inliner | 3: the page still carries a `<script src>`, which `file://` gives no origin to fetch |
| the README naming a file that does not travel | 1 |
| the generated runner filtering out the slow suite | 1: `--list` names eight of nine |
| a suite exiting zero after 7 of its 102 checks | 1: the count, not the exit code |
| `faraday/benchmark.js` or `faraday/zip-README.md` dropped | 1, and **only** the inverse-manifest check -- measured: without it the gate stays entirely green while the zip ships without them |

The last row is why the manifest is also read backwards: every source file in `faraday/`,
`dns/`, `boundary/` and `fsi/` must travel, and the deliberate-exclusion list is empty.
The first example that comment gave was `boundary/boundaries.html`, and the injection
disproved it -- `boundary/boundarykernel.test.js` reads that page, and dropping it fails 2
of that suite's 36 checks from the unpack. It was covered all along, and the comment now
says so.

**What the zip does not yet do.** `dns/check-cell3d.mjs` is not run from the unpack by the
gate: it is 264 checks over many grids and it costs minutes, and a gate nobody can afford
to run by hand stops being run. `dns/check-dns.mjs` was in the gate's fast set and was
taken out for the same reason -- measured at 506 s on this container, against 2.2 s for
`dns/check-wasm-build.mjs`, which crosses the same directory boundary and additionally
rebuilds the C++ with clang. Both still run in CI on the checkout. Closing that gap is
S13b. While measuring it, the CI comment calling `check-dns` "a two-and-a-half minute run"
was found stale and corrected in place.

**There is no GPU path, and the physics is never going to have one.** Zero references to WebGL,
WebGPU or WGSL in `dns/faraday-cell3d.js`, anywhere under `dns/`, or in `cymatic.html`; the only
mention in the project is S12 in this file. That is not an oversight waiting to be corrected.
Metal has no `double`, so WGSL has no `f64`, and no GPU on an Intel Mac carries a
double-precision type -- the constraint is recorded under "The one hard constraint" above and
was confirmed by the owner. S12 is a RENDER path plus an optional f32 preconditioner inside the
f64 conjugate gradient, legitimate because a preconditioner steers the search direction and
never the converged answer. Moving the physics to f32 would discard the validation S8 bought:
the cross-solver agreement is four parts in ten thousand, and f32 carries seven decimal digits
in total.

What makes the solver faster is S10, the same discretisation compiled to wasm, and S11, eight
CPU cores with a deterministic reduction order. Today it is single-core node at 85.6 ms a step
on 16x24x10, and 64.3 ms on 10x24x8, where `dns/run-cell3d.mjs` measured the ratio the page
reports: **1278x slower than real time** on this container, with the step set by the capillary
limit.

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
