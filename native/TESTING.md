Run the native fluid regression from the repository root:

```sh
node native/check-fluid-runtime.mjs
```

The command needs Node, a POSIX shell, and a C++20 compiler (`CXX`, default
`c++`) on a little-endian IEEE 754 host. It builds its driver and generates
fixtures in a temporary directory, which it removes on success or failure.
The allocation check uses approximately 121 MB of native field storage.

The checks compare every one of the 67 exposed arrays against both JavaScript
and the checked-in WebAssembly solver after 20 driven steps on two graded,
deformed grids. They cover the full step, separated predictor/surface step,
checkpoint recovery after a latched metric error, and stability bounds.

Projection checks verify physical-state isolation, storage alias refusal,
mass-weighted symmetry, idempotence, and preservation of divergence after
dependent-axis reconstruction. Resource checks exercise storage beyond the
former 96 MB arena, explicit budget refusal, partial-allocation cleanup,
signed-index rejection, and singleton recovery. The production native FFT
residue has a compile-time regression for the large-prime overflow case.

These are implementation regressions. They do not establish grain-scale
hydrodynamic accuracy, convergence with grid refinement, contact accuracy, or
practical full-vessel HPC performance.

Build only the runtime object with `native/build.sh`; use `native/build.sh
--help` for explicit executable source selection. The default output is
`native/build/fluid-runtime.o`. Compilation disables floating-point contraction
and fast-math to preserve the canonical arithmetic.

To verify that native-only hooks leave WebAssembly unchanged, when the WASM
toolchain described in `dns/build-wasm-cell3d.sh` is available:

```sh
check_dir=$(mktemp -d)
sh dns/build-wasm-cell3d.sh "$check_dir/faraday_cell3d.wasm"
cmp dns/faraday_cell3d.wasm "$check_dir/faraday_cell3d.wasm"
rm -rf "$check_dir"
```

Keep the output basename `faraday_cell3d.wasm`: the module's custom name section
includes that basename, so a different name changes metadata bytes.

Run the coupled mechanics regression with:

```sh
node native/check-coupler.mjs
```

To retain its generated case, native executable, build log and JSON results,
choose an evidence directory outside the source checkout:

```sh
FARADAY_NATIVE_QA=/tmp/faraday-coupler-evidence node native/check-coupler.mjs
```

Without `FARADAY_NATIVE_QA`, successful coupling checks remove their temporary
directory. Failed checks retain it and report its path. The executable also
accepts a test-name substring after its prepared `.fns` input for a focused
rerun. These checks need the same host/compiler requirements as the fluid
regression and can take several minutes on a shared machine.

The generated case is a declared 250 micrometre sphere of density 2650 kg/m³
in liquid of density 1000 kg/m³ and viscosity 0.001 Pa s. Its vessel is only
1 mm in diameter and 0.5 mm deep, with a 20 × 64 × 20 grid. These are
illustrative verification inputs, not measured beach-sand properties or the
user's full vessel. Trajectories use the default 81 volume collocation markers.
Passing the eight-cells-per-diameter refusal gate is not a convergence result.

The ten coupling regressions check:

- Exact preservation of the canonical zero-grain fluid step.
- Unresolved geometry, overlapping geometry and working-memory budget refusals
  without changing the caller's fluid state or projection mode.
- Complete state rollback after constraint nonconvergence and after a rejected
  candidate has advanced its pressure, surface, position and clock.
- An independent dense kinetic-energy Gram solve for the joint fluid/body
  response, including excess inertia and the sign and magnitude of reduced
  pressure. This algebra-only oracle deliberately uses seven markers to keep
  the direct matrix small; it is separate from the default-marker trajectories.
- Twenty nonzero steps from rest, with the body impulse checked independently
  against excess mass times acceleration plus effective weight.
- Three finite-speed steps, including exact accepted position integration and
  reporting of the actual time-step refinements.
- A supported floor sphere, with separate hydrodynamic and contact impulses,
  weight balance, unchanged position and the cancellation of induced flow in
  static equilibrium.

Historical baseline: one host run on 2026-11-14 passed all ten checks. Its measured motion
and elapsed times were:

| Case | Accepted steps | Simulated time | Host elapsed time | Vertical displacement magnitude |
| --- | ---: | ---: | ---: | ---: |
| Default sphere, initially at rest | 20 | 100 ns | 16.70 s | 2.987 × 10⁻¹⁴ m |
| Sphere initially moving at −0.001 m/s | 3 | 1.875 ns | 11.03 s | 6.347 × 10⁻¹³ m |
| Supported sphere | 2 | 6.25 ns | 27.69 s | 0 m |

Host times depend on the machine and concurrent load. The rest-start case
accepted 5 ns steps without refinement, with maximum rigid slip
9.99 × 10⁻¹⁴ m/s and divergence 5.87 × 10⁻¹⁴ s⁻¹. The finite-speed case accepted
0.625 ns steps after nine total refinements. The supported case's largest
residual fluid and body speeds were 9.55 × 10⁻¹³ and 9.09 × 10⁻¹⁵ m/s.
The independent dense oracle differed by 8.54 × 10⁻²¹ m/s in fluid velocity,
1.72 × 10⁻¹⁹ across rigid velocity components, and 2.29 × 10⁻¹⁴ Pa in reduced
pressure against a 0.460 Pa pressure scale. Translational and angular rigid
components have their respective m/s and rad/s units; their reported maximum
is an algebra check, not one physical norm.

The finite-speed regression exposed a stopping-criterion defect: a linear
solve measured against the incoming mismatch could stop above the tolerance
measured against the smaller final velocity. Shrinking the time step did not
remove that residual floor. The coupled solve now recomputes and corrects the
actual velocity defect, accumulating every pressure and hydrodynamic impulse,
until the original final-velocity criterion passes. Contact responses use the
same principle; their mobility is assembled from kinetic energy and checked
against the actual response. No physical acceptance tolerance is loosened.

These tests establish coupling algebra and transactional behavior, not useful
settling duration, drag accuracy, hydrodynamic convergence or sand deposition.
That baseline constrained the old geometry and only checked the updated
geometry, which could force the accepted step toward the algebraic tolerance
at finite velocity. The moving-geometry repair instead solves the endpoint
constraints and synchronizes the dependent axis and surface-rate fields. The
independent fixed-time refinement check is `native/check-coupled-convergence.mjs`;
it compares trajectories at equal physical times and checks the actual endpoint
fields, volume and Newton–Euler impulse accounting. These checks do not certify
spatially converged hydrodynamics. Refining a full polar vessel also introduces severe near-axis
stiffness: the surface and viscous stability estimates include the highest
azimuthal mode divided by the first radial cell's radius. Cell counts and
memory alone therefore understate the full-vessel computation cost.

Positive lubrication gaps below the declared resolution are refused. A
suspended grain cannot be advanced through that unresolved gap into a bed;
an initially touching grain likewise cannot lift off through an unresolved
positive gap. The static floor check does not establish deposition,
resuspension or a usable vibrating sand bed. No replacement drag, lubrication,
packing, particle-speed cap or position repair is supplied by these tests.

## Moving-geometry repair verification, 2026-11-17

Run the independent equal-physical-time check with:

```sh
node native/check-coupled-convergence.mjs
```

At a common 160 ns horizon, requested steps of 20, 10 and 5 ns were accepted
without reduction. The observed refinement orders were 1.028 for fluid
mass-weighted velocity RMS, 0.993 for grain velocity and 1.000 for grain
position. The corresponding finest differences were 2.745e-11 m/s,
1.401e-12 m/s and 2.200e-15 m. The gate checks these differences against
numerical floors before drawing a convergence conclusion. Each refinement
starts after the same explicitly reported 1 ps conditioning step; that step
is part of the numerical fixture, not application initialization.

A displaced sphere with lateral velocity and rotation also passed three
20 ns steps. Freshly rebuilt endpoint interpolation, the reconstructed axis
rows, surface-rate fields, divergence, body impulse accounting and vessel
volume are checked independently. Across those runs the largest slip was
9.95e-14 m/s, divergence 1.11e-13 per second, body impulse-accounting defect
6.08e-28 N s and relative volume change 2.73e-17. Body accounting is not a
claim of total closed-vessel momentum conservation without wall tractions.

The same fixed-time gate fails the earlier implementation because it reduces
the requested step. Thus the check distinguishes the repair from the old
behavior rather than merely asserting that a run produced output. It proves
first-order temporal behavior on one mesh, not spatial convergence.

The actual command-line runner also completed a compatible rest-start case
to 1 microsecond in 50 requested 20 ns steps, with no event refinements. Its
six saved frames passed the native-recording importer. This remains a short
one-grain numerical run, not a visible depositing sand bed.

`node native/check-cli.mjs` additionally verifies that the first saved frame
contains measured initial constraints and that incompatible initial fluid/grain
velocities are refused instead of being silently projected or reported as zero
slip.

## Moving-grid conservation and joint endpoint solve, 2026-11-22

The radial kinetic mass now uses the same pressure-cell mass transfer as its
transport fluxes. Its time derivative therefore equals the discrete mesh-flux
balance on uniform and graded grids. Pressure, viscosity, cylindrical curvature,
kinetic energy and native grain coupling use this same mass. The independent
public-energy check fails on the earlier kernel with a relative mass-rate defect
of about 0.00927; the repaired cases give at most 3.11e-13. See
[`dns/ALE-MASS-CONSERVATION.md`](../dns/ALE-MASS-CONSERVATION.md) for the derivation,
mass-aware step limits and remaining axis/finite-time caveats. Run
`node dns/check-cell3d-gcl.mjs`.

The optional joint solver now computes normal viscous stress from the final
fluid velocity, including its rigid-grain response. At each trial geometry it
solves the pressure feedback equation, then recomputes the full endpoint
kinematic residual. Interior advection and viscous acceleration remain explicit;
this is not a fully implicit viscous or contact solver.

Independent tests use an initially resting off-axis sphere below a finite
deformation of about 44.8 micrometres in 500 micrometres of liquid. At a common
200 ns horizon, steps of 100, 50 and 25 ns give refinement orders 0.987 for
fluid velocity, 0.964 for grain velocity, 0.99995 for surface height and 1.00086
for pressure. The largest endpoint normal-stress defect is 1.11e-9 Pa; the
previous lagged stress target misses about 0.01994 Pa in the coarse case.
The 100 ns step exceeds this fixture's raw capillary bound, approximately
57.1 ns, while satisfying its retained viscosity/advection/gravity bound.

With the same endpoint equations, a direct comparison of diagonal and cached
constraint preconditioning completed the same 200 ns trajectory in 221.63 s
and 14.82 s respectively on the development host, including cache setup.
Constraint iterations fell from 6,484 to 102; the largest fluid-velocity
difference was 6.21e-13 m/s and grain-velocity difference 1.02e-14 m/s, within
the propagated algebraic tolerances. This is a measured small-case improvement,
not a dense-bed scaling result. The actual launcher also completed a seeded,
initially resting one-grain case to 1 microsecond in ten 100 ns steps. Its six
saved frames passed the recording importer with the method and residuals intact.

The tests also reconstruct actual fluid/body constraint exchange. The maximum
force-impulse defect is 7.62e-28 N s and torque-impulse defect 1.56e-31 N m s.
These are discrete exchange checks, not independent validation of sphere drag,
wall reactions or spatially converged hydrodynamics. Physical clocks must
advance and all three refinements must end at the same time.

```sh
FARADAY_IMPLICIT_MODE=cached node native/check-implicit-coupled.mjs
node native/check-cli.mjs
node faraday/check-native-recording.mjs
```

The command-line gate runs both explicit and joint grain steps, imports their
actual records, verifies a capillary-limited case is admitted only by the chosen
implicit method, and verifies the retained explicit limits still refuse an
oversized step. Recording tests preserve numerical provenance and reject missing
joint diagnostics. No initial normal-stress residual is fabricated.

After the mass correction, the explicit moving-grain test still passes at
20/10/5 ns with no refinements: fluid/grain-velocity/position orders are
1.028/0.997/1.000. The 12 native-fluid groups and 10 coupling groups also pass,
including bit parity with the synchronized JS/WASM kernel, rollback and the
independent algebra oracle. These component and temporal tests do not complete
the missing resolved-gap connection to the vessel or deposition through contact.

The positive-gap tests now include a rotating spherical cap with physical
no-slip and traction boundaries, nonzero nonlinear advection, dimensional
similarity at a 200 micrometre scale, and gravity with physical pressure/load
reconstruction. Two nonoverlapping meshes also share velocity and pressure
unknowns and reproduce a single-mesh residual/Jacobian within 4.3e-16.
That proves the tested conforming finite-element connection, not a connection
to the vessel's staggered grid. Spatial stress errors remain measured rather
than declared zero. See `RESOLVED-CONTACT-RESEARCH.md` and run:

```sh
node native/check-resolved-gap-flow.mjs
node native/check-resolved-gap-boundary.mjs
node native/check-resolved-gap-join.mjs
```

## Positive-gap geometry

```sh
node native/check-resolved-gap.mjs
```

This gate verifies the exact sphere-wall and sphere-pair coordinate maps,
gradient/flux/divergence identities, rotating/translating chart conservation,
surface normals, volume integrals and finite positive-gap behavior. A test-only
Brenner-series reference is checked against independent high-precision values;
it supplies no force to the production solver. See
`native/RESOLVED-CONTACT-RESEARCH.md` for the sources and the remaining full
Navier–Stokes patch/interface requirements. Passing this geometry gate does
not change the production gap refusal or establish deposition.
