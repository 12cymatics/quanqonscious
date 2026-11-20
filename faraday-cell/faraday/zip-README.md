# faraday-cell

The velocity arrows are visible by default. Their fixed drawing scale is 1000 seconds: arrow length equals velocity multiplied by that scale, not a predicted particle trajectory. The displacement magnification is likewise display-only; neither scale changes the computed fluid or tracer coordinates.

## Start here

Extract the ZIP, open the faraday-cell folder, then open **index.html** in a
full desktop browser such as Safari, Chrome or Firefox. This is the complete
standalone simulator, not a link or a launcher requiring Node. No installation
is needed. A static attachment or iPhone Files preview may display the controls
without executing JavaScript; the startup notice will remain visible there.

The app opens one **3D Navier–Stokes** scene. Its free surface, sand-coloured
flow tracers and velocity vectors share the same computed state and physical
clock. Drag to orbit, scroll to zoom, and use the view controls to inspect the
vessel from above or return to the initial camera. Surface and particle controls
show or hide layers of this scene; they do not switch to another simulation.

The ZIP includes both index.html and faraday-cell-standalone.html with identical
contents. The source cymatic.html is also included, but requires the local HTTP
server described below. Running from a folder does not itself accelerate the
numerical solver. The 3D drawing requires WebGL2; all fluid calculations remain
in double precision on the CPU.

The page draws the specified initial surface at **t = 0**, then starts the
nonlinear solver. **Pause/Resume** retains the computed fluid and particle state;
**Reset** returns to the initial condition and holds until **Run** is requested.
Open with `?paused=1` to inspect the initial state before starting. The physical
clock and measured wall time make progress visible. The C++ and JavaScript
engines solve the same discretized equations. Detailed fluid, geometry,
initial-condition and numerical settings remain explicit inputs.

The vertical displacement scale is a visible drawing setting, initially
**5000×**, because the default initial displacement is microscopic relative to
the vessel diameter. Select **1×** for metric displacement. This fixed setting
never changes the solver or renormalizes a new frame to its own peak.

The sand-coloured particles are **passive flow tracers**, advected by the actual
computed fluid velocity and physical time. They reveal the same flow as the
surface and vectors. They do not represent physical sediment settling, particle
inertia, grain contacts or feedback on the fluid; those require a separate
material model and parameters. The earlier empirical drift, agitation and
packing calculation is not used to animate this nonlinear scene. The fluid
solver, its pressure correction, accuracy requirements and book-derived checks
are retained.

A nonlinear free-surface simulation of an incompressible Newtonian liquid in a
rigid circular cell, with a pinned rim and no-slip floor and wall. Diameter,
depth, density, viscosity, surface tension, gravity and drive acceleration are
explicit inputs. The reference paper's apparatus presets and atlas are optional
comparisons. They do not determine the general experiment's nonlinear physics.

The packaged application needs a desktop browser. JavaScript checks need Node;
the exact book-derived verification needs Python 3 and SymPy 1.14.0, and rebuilding
the WebAssembly checks needs clang with the wasm32 target and lld. The application
itself does not call an external API or require those verification tools.

## Book polynomial methods

All 29 named book methods and their derived polynomial extensions are mapped in
`docs/BOOK_POLYNOMIAL_METHODS.md`. The executable modules in `dns/vedic-*.js`
construct exact rational interpolation, derivative, moment and flux coefficients.
The selected JavaScript or WebAssembly solver consumes the numerical tables in
its pressure, viscosity, traction and adaptive time-step calculations.

The line below the solver status reports polynomial integrals for the displayed
frame. **Save polynomial coefficients and frame integrals** downloads a JSON
record containing exact rational coefficients, the actual runtime stencil bytes,
frame values and a SHA-256 digest of the serialized coefficient payload. Recorded
playback exports the selected recorded frame. An analytic preview before solver
initialization has no computed integral export.

The coefficient arithmetic is exact. The fluid state and time stepping use
binary64 arithmetic and remain a numerical discretization. SHA-256 checks byte
integrity, not physical correctness. Independent symbolic, numerical, backend,
worker and browser checks are included in the complete test runner.

## Nonlinear model and corrections

`dns/MODEL.md` specifies every bulk equation and the normal, tangential,
kinematic, pinned-rim and solid-wall boundary conditions for the supported model.
It distinguishes the full nonlinear equations from their finite-grid numerical
solution. Compressible flow, a second viscous fluid, moving contact lines,
overturning interfaces and variable material properties are not silently
represented by this constant-property, single-liquid graph-surface model.

The pressure correction now includes the prescribed free-surface pressure before
the horizontal slope-transpose terms, in both JavaScript and C++. Previously a
matching constant pressure could cause spurious horizontal acceleration on a
deformed surface. `dns/check-surface-pressure.mjs` checks constant-pressure
cancellation, pressure-gauge invariance, the pinned rim and both engines.

`dns/check-book-pressure.py` uses the page-specific procedures in `vedic_book.py`
to derive exact polynomial pressure-coordinate identities. It incorporates the
book corrections recorded in PR #172 and independently checks them. Those
arithmetic checks do not turn the fluid equations or pressure projection into
sutras and do not claim physical validation.

Run the focused checks from this folder:

```
node dns/check-surface-pressure.mjs
node faraday/check-general.mjs
python3 dns/check-book-pressure.py
```

## One physical clock and recorded playback

Live rendering displays completed solver frames. Camera movement and drawing
speed do not advance fluid or tracer time. Recorded playback lets you inspect
already-computed frames, with their original timestamps; it never invents a
future oscillation while the solver is computing. The displayed history is
bounded to **180 frames and 32 MiB**. The available recorded interval is shown in
the interface, and returning to Live resumes the latest computed view.

Image export freezes the displayed geometry, tracer positions, camera and
physical timestamp. Choosing 2048 or 4K changes only rendering resolution. It
does not rerun a separate particle-settling simulation.

The solver can take substantial wall time to advance a small physical interval
on its accuracy-selected grid. Physical and wall clocks remain visible; replay
and camera motion do not claim real-time fluid computation.

## What is in here

| | |
|---|---|
| `faraday-cell-standalone.html` | The same self-contained application as index.html. No server, no siblings, nothing fetched. |
| `cymatic.html` | The same page as it lives in the repository, loading its scripts from `faraday/` and `dns/`. Needs a local server (below). |
| `dns/run-cell3d.mjs` | The three-dimensional solver in a terminal, no browser: an ASCII plan view of the surface, the energy split, and the clocks. |
| `run-checks.mjs` | Runs every suite in this directory, in one command. |
| `dns/PLAN-cell3d.md` | The working record: what each stage measured, what is still open, and every expectation measurement has overturned. |

The solvers:

| | |
|---|---|
| `faraday/kernel.js` | The linear theory: Bessel modes, viscous damping, the Kumar–Tuckerman-style threshold problem, the mode map the page draws its labels from. |
| `dns/faraday-disc.js` | A two-dimensional axisymmetric Navier–Stokes solver, **one azimuthal mode at a time**, linear in the surface amplitude. Its Floquet growth rates are the reference the three-dimensional solver is measured against. |
| `dns/faraday-cell3d.js` | The three-dimensional **nonlinear** solver: every azimuthal mode at once and coupled, in a domain that follows the surface. This is what the page draws. |
| `dns/cell3d-observation.js` | Physical-height velocity reconstruction and geometry from the actual staggered solver fields. |
| `dns/cell3d-tracers.js` | Passive fluid-tracer transport on the solver clock. |
| `faraday/scene3d.js` | The WebGL2 vessel, surface, particle and velocity-vector renderer, orbit camera and captured-state image export. |
| `dns/faraday_disc.cpp`, `dns/faraday_disc.wasm` | The two-dimensional solver again in C++, compiled to WebAssembly, bit for bit against the JavaScript. `dns/build-wasm.sh` rebuilds it. |
| `dns/faraday_cell3d.cpp`, `dns/faraday_cell3d.wasm` | **The three-dimensional solver in C++**, likewise compiled to WebAssembly and likewise bit for bit — operator by operator and then over a whole drive period, 894 steps at 12 × 24 × 8, every element by `Object.is`. `dns/build-wasm-cell3d.sh` rebuilds it; neither needs Emscripten, only `clang` with the wasm32 target and `lld`. |
| `dns/faraday-floquet.js`, `dns/faraday-dns.js` | The Floquet machinery and the periodic-box solver the disc one was built from. |
| `boundary/boundarykernel.js`, `fsi/coupled-affine-benchmark.js` | A boundary-condition kernel and an exact fluid–structure benchmark in rational arithmetic. |

## Running it

**The single file.** Double-click `faraday-cell-standalone.html`. The
three-dimensional solver runs in a worker and feeds the visible 3D scene.

**The repository page.** `cymatic.html` loads its scripts by relative path and
its compiled WebAssembly by fetch, and a page opened from `file://` has no
origin a fetch can satisfy. Serve the directory instead:

```
python3 -m http.server 8000
```

then open <http://localhost:8000/cymatic.html>.

**The terminal.** No browser needed:

```
node dns/run-cell3d.mjs
node dns/run-cell3d.mjs --nr 16 --nth 32 --nz 10 --m 4 --accel 12 --periods 4
node dns/run-cell3d.mjs --engine cpp
```

Options: `nr nth nz` (grid), `m` (azimuthal mode seeded), `rel` (seed amplitude
as a fraction of the depth), `accel` and `freq` (the shaker; with no `freq` it
drives at twice the mode's own frequency, which is the subharmonic resonance
Faraday waves live on), `periods`, `frames`, `contact` (`pinned`),
`R h rho nu gamma g`, `rStretch zStretch`, `width`, `engine` (`js` or
`cpp`) and `threads`. An unknown option is refused rather than ignored.

**Threads.** The page's *solver threads* control, and `--threads N` in
`dns/run-cell3d.mjs`, spread each step's viscous and advective terms over that
many threads, the solver's own included. The page offers up to the number of
cores the browser reports and starts at one fewer, which leaves the drawing a
core. The answer is the same to the bit on any number of threads:
`dns/check-cell3d-pool.mjs` holds it to the single-threaded step with `Object.is`
on one to five workers in both engines, and `dns/check-page.mjs` replays the
page's own threaded run on one thread and compares again. What changes is the
wait, and only for the part that is divided. The pressure solve stays on one
thread: each of its iterations needs two global sums, and a page opened from a
file has no shared memory to make those cheap.

`--engine cpp` is not a faster-but-looser mode. The C++ is a transcription of the
JavaScript — same discretisation, same flux forms, same conjugate gradient with
the same tolerance and cap, same order of operations down to the grouping of each
sum, and no fused multiply-add, because an FMA rounds once where JavaScript rounds
twice. It computes no transcendental at all: the `cos` in the drive is evaluated in
JavaScript and passed in, which is what makes bit-for-bit parity achievable rather
than approximate. Measured on the container this was built on, best of three: **4.0× at
10 × 24 × 8 and 3.5× at 16 × 24 × 10**, with identical fields. If the module cannot be loaded it
refuses and says so, rather than running the JavaScript under the C++ engine's
name and reporting a time that means something else.

**The suites.**

```
node run-checks.mjs --list      # what there is
node run-checks.mjs --fast      # partial diagnostics; exit 2 if selected suites pass
node run-checks.mjs             # everything
```

`dns/check-cell3d.mjs` and `dns/check-cell3d-wasm.mjs` take minutes — the first
integrates the solver over many grids to measure convergence orders, the second
rebuilds the C++ and holds it to the JavaScript over a whole drive period — which
is why `--fast` exists and why those two are the only things it leaves out.
The runner names those omitted suites and reports an incomplete execution.
It returns exit code 2 even when all selected suites pass; full verification requires
running without `--fast`. The full run includes the free-contact wall-kinematics
check through `dns/check-surface-pressure.mjs --audit-free`.
`dns/check-page.mjs` drives a real Chrome over
the DevTools protocol; it **refuses** rather than skipping if it cannot find one,
and `CHROME=/path/to/chrome` tells it where to look. `faraday/check-clean.mjs` reads
the comment-free build back with [acorn](https://github.com/acornjs/acorn), a parser
independent of the one that built it, and likewise refuses without it:
`npm install --prefix /tmp/tok acorn@8.18.0`, then `ACORN=/tmp/tok/node_modules/acorn`.

`node faraday/build-clean.mjs` followed by a name for the new file writes the single
file again with code comments removed. Runtime error reasons, warnings, input
provenance and model limitations remain visible. The token comparison uses the
original full page, and the browser checks require computed warnings and error
reasons to remain on screen.

## Two things to know before you file a bug

**It does not run at real time, and it cannot.** On the cloud container this was
built on, a 10 × 24 × 8 grid takes 18.8 ms per step in JavaScript and 5.3 ms in
C++, and the step is set by the capillary limit: 374× slower than real time for
the first and 106× for the second, measured with `dns/run-cell3d.mjs`. On three
threads, 15.2 and 4.3 ms: 302× and 86×. (When the solver first ran on the page
those were 1278× and 489×; `dns/PLAN-cell3d.md` has what changed, under S10b, S12b
and S11.) The page shows
you the physics clock, the wall clock and their ratio rather than hiding the
gap; `dns/run-cell3d.mjs` prints the same three numbers. Your machine will give a
different ratio and the number it prints is measured on it, not estimated.

**The page sizes its grid for the entered experiment.** Its requirements are the
surface-operator accuracy and Stokes-layer resolution stated below. The active
grid, physical step and measured progress appear in the diagnostics. Historical
timings for earlier apparatus defaults in `dns/PLAN-cell3d.md` are not promises
for the current inputs or your computer.

**The fluid calculation uses double precision on the CPU.** WebGL draws the
computed fields in single precision. Rendering never supplies the velocity,
pressure or surface evolution used by the solver.

**Drawing the 3D scene.** `faraday/scene3d.js` uses WebGL2 for perspective,
depth testing, lighting and point rendering. Camera orbit, zoom, layer visibility
and displacement magnification affect display only. If the graphics context is
lost, the renderer reports it and restores the same frame and camera when the
browser restores the context. No substitute fluid calculation is started.

`faraday/check-scene3d.mjs` checks scene geometry, camera transforms, fixed
magnification and immutable export capture. `dns/check-cell3d-observation.mjs`
checks the conversion of actual solver fields into observation data;
`dns/check-cell3d-tracers.mjs` checks transport against known velocity fields.
`faraday/check-unified-playback.mjs` exercises the page's actual history and
export controller, including timestamp/generation consistency, bounded memory,
pause on seeking and immutable capture across a new incoming solver frame.
The browser checks exercise actual drawing and controls. The retained legacy
field renderer has its own CPU/GPU parity checks; those are separate from the
three-dimensional geometry checks and are not claimed as validation of grain
material physics.

## Physical reference and numerical scope

`docs/PINNED_CYLINDER_REFERENCE.md` derives the coupled pinned-cylinder reference,
including the surface/potential basis overlap, depth-dependent mass, convergence
and independent Floquet calculation. The physical tests use that reference and
the actual evolved velocity and surface fields. They retain the original
accuracy targets and refine the grids where the former grids were insufficient.

For the documented pinned m = 3 example, restoring-acceleration error falls from
10.79% to 0.56% over the four tested meshes. The near-inviscid quarter-period run
changes total discrete energy by -0.0495% and transfers 99.08% into kinetic energy.
The refined damping comparison closes from 7.456% through 2.743% to 1.376%, with
an h-squared extrapolated difference of -0.1654%. The second m = 1 mode's decay
agrees within 0.70%. The driven multiplier comparison is within 2.98% and the
growth-rate comparison within 6.67%; the independent reference multiplier is
1.56280794 at the documented timestep refinement.

These results concern the stated modes, grids, material parameters and time
intervals. They do not establish equal accuracy for arbitrary input or a
comparison with a laboratory experiment. The numerical model is a Newtonian,
incompressible liquid with a single-valued graph surface, stationary no-slip
walls and a pinned contact line. It does not describe overturning, droplet
pinch-off, entrained air or a moving contact line. An unsupported moving-contact
request fails before time or state is advanced.

The page sizes the grid for the selected initial mode's surface-operator target
and Stokes-layer requirement and reports when the cell budget cannot satisfy
them. Higher spatial modes can have larger errors on the same grid. Exact
rational coefficient construction preserves the algebra used to build each
operator; the finite spatial grid and binary64 time integration still introduce
numerical error. `dns/PLAN-cell3d.md` retains measurements of earlier revisions
as history, separately from the current pinned reference.

## The one rule the suites are built on

The suites compare actual outputs against independent algebraic and physical
references. Coefficient and worker fault injections also verify that changed
inputs affect the consumed result or cause a visible failure. Historical mutation
experiments are recorded in `dns/PLAN-cell3d.md`; current polynomial-consumption
checks are in `dns/check-vedic-integration.mjs` and `dns/check-vedic-fluid.mjs`.
The full runner executes every listed suite. A requested partial selection
reports the omitted suites and cannot return full-verification success.
