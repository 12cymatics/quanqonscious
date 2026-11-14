# Experimental native grain–fluid calculation

This source prototype couples finite spherical grains to the existing three-dimensional liquid solver. It follows the selected resolved-fluid route: rigid volume constraints exchange momentum with the liquid, without an empirical particle drag law. The browser imports its recorded states into the same interactive 3D scene as the liquid surface and velocity field.

**This is not a completed beach-sand simulator or a commercially validated product.** The included verification case covers two accepted steps, totalling **10 nanoseconds**, in a deliberately small vessel. It checks execution and output, not settling, pattern formation, deposition or experimental agreement. Motion over this interval is extremely small; playback does not invent more motion.

A separate 20-step, 100 ns execution of this geometry completed in about 20 seconds on the shared Linux development host, using 81 volume markers. That is a limited execution observation, not a portable performance benchmark. It already shows that the present implementation is far too slow for useful full-vessel sand evolution.

The current mesh must resolve grains and positive liquid gaps. It refuses an approaching grain when its lubrication gap becomes unresolved. Consequently this fixed-mesh prototype cannot yet carry settling grains through that unresolved region into a deposited sand bed. More RAM alone does not remove this limitation.

## Requirements and platforms

- A 64-bit computer, Node.js 20 or newer, and a C++20 compiler supporting the flags in `build.sh` (GCC or Clang).
- A POSIX shell and standard command-line tools. The development execution was on Linux with Node.js 24. macOS and Windows builds have not been validated.
- On macOS, use Terminal with a suitable C++ compiler and Node.js installed. On Windows, these commands require a Linux environment such as WSL2, with Node.js and the compiler installed inside that environment. They are not PowerShell or Command Prompt commands; no native Windows executable or installer is supplied.
- The full source tree, including `dns/` and `faraday/`. Copying this directory alone is insufficient. The HTML viewer does not compile or launch the native process.

`CXX` may name a compiler executable, for example `CXX=clang++`. It is not a string of compiler flags. This executable is currently a single-process CPU program, with one live fluid arena. There is no distributed-memory, MPI or GPU backend here; “native” does not establish HPC scalability.

## Run the small verification case

From the repository or extracted source package root:

```sh
sh native/run-case.sh \
  --case native/examples/single-sphere-verification.json \
  --out verification-run
```

The output directory must be new, and its parent directory must exist. The launcher builds the program, prepares exactly the supplied case, and runs it. It never chooses a grain count, coarsens the grid, substitutes fluid or material properties, or accepts an input time step above the fluid stability bound. Accepted steps can be shorter than `dt` during contact-event or constraint-residual refinement, and at the final duration. Their actual duration and refinement count are recorded.

The directory contains:

| File | Content |
| --- | --- |
| `faraday-native` | Locally compiled executable |
| `build.log` | Compiler output |
| `prepared.fns` | Prepared grid, initial state, pressure preconditioner and original case values |
| `preparation.json` | Resource estimates and initial fluid stability bound |
| `preparation.log` | Preparation progress or refusal reason |
| `solver.log` | Accepted-step diagnostics and any native refusal |
| `recording.jsonl` | Recording after successful completion of the declared interval |
| `recording.partial.jsonl` | Retained output if calculation fails or is interrupted |

An exit status of zero means the declared interval completed. It is not a physical validation certificate. A failed calculation must not be presented as a completed run. Logs can be inspected while execution continues, for example `tail -f verification-run/solver.log`.

The same operations can be run separately:

```sh
mkdir verification-manual
sh native/build.sh verification-manual/faraday-native \
  native/run.cpp native/rigid-coupler.cpp \
  native/rigid-contact.cpp native/rigid-interpolation.cpp
node native/prepare-case.mjs \
  --case native/examples/single-sphere-verification.json \
  --out verification-manual/prepared.fns \
  > verification-manual/preparation.json
verification-manual/faraday-native verification-manual/prepared.fns \
  > verification-manual/recording.jsonl \
  2> verification-manual/solver.log
```

With these manual commands, the filename alone does not indicate success: check the exit status and the final `complete` record. Preparation requires both `--case` and `--out`; the native executable takes one prepared-case path. Running `build.sh` without entry sources only compiles the fluid object, not the coupled executable.

## View the recording

Open the current packaged `index.html` in a full desktop browser, then choose **Import native recording** and select `recording.jsonl`. The development page is `cymatic.html`; a source checkout may be served locally to load its modules. The import accepts `.jsonl`, `.ndjson` and `.txt` files from the local file picker.

The liquid surface, velocity display and finite grain spheres use the same recorded physical time and camera. Use **Play recorded frames**, the timeline or frame navigation to inspect accepted states. Playback does not advance physics, interpolate new grain trajectories or produce future frames. **Return to fluid solver** leaves the imported recording and returns to the browser liquid solver; its passive flow tracers are not sand grains.

An error record, malformed tail or interrupted file is visibly reported. Earlier coherent frames may remain available, labelled incomplete. Imported pressure is unavailable in this recording version. History is bounded to 180 frames / 32 MiB, with its retained interval shown. Large outputs can exceed the reader's per-frame or record-size limits and are refused rather than reduced. This is a local recording viewer, not a live remote HPC connection.

## Declare an experiment

`case-schema.mjs` defines strict version 1 inputs. Every listed field is required; unrecognised fields, non-finite numbers and missing physical inputs are rejected. The example is a template for a numerical verification case, not a measured material specification.

| Input | Units and interpretation |
| --- | --- |
| `format`, `version` | `faraday-native-case`, `1` |
| `fluid.radius`, `fluid.depth` | Vessel radius and initial liquid depth, metres |
| `fluid.density` | kg/m³ |
| `fluid.dynamicViscosity` | Pa·s; kinematic viscosity is derived as μ/ρ |
| `fluid.surfaceTension` | N/m |
| `fluid.gravity` | Nonnegative m/s² |
| `fluid.contactLine` | Must be `pinned` |
| `drive.frequency`, `drive.acceleration` | Hz and signed m/s²; effective gravity is `g + a cos(2π f t)` |
| `grid.nr`, `grid.nth`, `grid.nz` | Explicit radial, azimuthal and vertical counts; `nth` is even |
| `grid.rStretch`, `grid.zStretch` | Nonnegative mesh-stretch parameters; zero means uniform |
| `initial` | `{"type":"flat"}` or `{"type":"seed","azimuthalOrder":2,"amplitude":1e-7}`; seed amplitude is metres |
| `grains` | Explicit array, including `[]` for a zero-grain case; no loading is inferred |
| Each grain | Unique integer `id`, `centre` [x,y,z] in metres, `radius` in metres, `density` in kg/m³, `velocity` in m/s, `omega` in rad/s |
| `materialProvenance` | `kind` (`illustrative` or `measured`), `description`, and `source`; a claim of measurements is recorded, not certified |
| `contact.friction`, `contact.restitution` | Explicit coefficients; friction ≥ 0 and restitution from 0 to 1 |
| `resourceBudgetBytes` | Explicit preparation and native working-array budget, described below |
| `duration`, `dt`, `outputInterval` | Positive seconds; `dt` must satisfy the fluid stability bound |

Coordinates use the cylinder axis as x=y=0 and the floor as z=0. Grains must be denser than the liquid for the present positive excess-inertia formulation. Overlapping initial grains, vessel penetration, inadequate local resolution and unsupported submerged geometry are refused. The seeded Bessel surface is only an initial condition; it does not prescribe later fluid patterns.

The supplied example has a **1 mm diameter, 0.5 mm deep vessel**, a `20 × 64 × 20` grid, and one **250 µm diameter sphere** with density **2650 kg/m³**. These are illustrative silica-rich material and sphere inputs. They are not measurements of south-east Queensland beach-sand shape, purity, grading, loading, friction or restitution. The vessel differs from the original 30 mm experiment. The example's contact coefficients are explicit inputs but its tiny separated-grain run does not exercise bed contact. A `16 × 64 × 16` version of this geometry fails the current support-resolution check.

## Governing equations and assumptions

The continuum target is the full three-dimensional, nonlinear incompressible Newtonian momentum equation and continuity, with constant liquid density ρ and dynamic viscosity μ. In the translating vessel frame, let z point upward and `g_eff(t) = g + a cos(2π f t)`. With physical pressure p and constraint-force density λ directed **onto the rigid grains**, the fictitious-domain equations are

```math
\nabla\cdot\mathbf u=0,\qquad
\rho\left(\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u\right)
=-\nabla p+\mu\nabla^2\mathbf u-\rho g_{\rm eff}\mathbf e_z-\boldsymbol\lambda.
```

The fluid fills the vessel computationally, including each sphere's interior; λ is supported there and vanishes in unconstrained liquid. The stored pressure array is reduced pressure `q = p − p_atm + ρ g_eff (z − h)`, so the equivalent momentum equation is `ρ D_t u = −∇q + μ∇²u − λ`. Gravity is therefore not added again as a fluid acceleration after this pressure transformation.

For sphere i with volume `V_i = 4π r_i³/3`, centre X, velocity U and angular velocity Ω, rigidity requires `u(x) = U + Ω × (x − X)` inside that sphere. The body equations use excess mass `m_ex = (ρ_i − ρ)V_i` and excess inertia `I_ex = (2/5)m_ex r_i²`:

```math
m_{\rm ex}\dot{\mathbf U}
=-m_{\rm ex}g_{\rm eff}\mathbf e_z+\int_{B_i}\boldsymbol\lambda\,dV+\mathbf F_c,
\qquad
I_{\rm ex}\dot{\boldsymbol\Omega}
=\int_{B_i}(\mathbf x-\mathbf X)\times\boldsymbol\lambda\,dV+\mathbf T_c,
\qquad \dot{\mathbf X}=\mathbf U.
```

The isotropic sphere inertia makes the gyroscopic cross term zero. The excess quantities avoid counting the inertia and weight of the interior fictitious liquid twice. `F_c` and `T_c` are contact forces/torques, implemented as impulses with the declared friction and restitution. Non-spherical shape, size distribution and measured mineral composition are not inferred from the phrase “beach sand”. No empirical drag, added-mass or history-force formula is added on top of this resolved-fluid constraint formulation.

The discrete constraint system uses fluid interpolation J, rigid motion K, liquid mass matrix `M_f`, excess body mass/inertia matrix `M_ex` and the incompressible projector P. For integrated marker impulses Λ directed onto grains, it solves

```math
\left(JP M_f^{-1}J^T+K M_{\rm ex}^{-1}K^T\right)\Lambda
=J\mathbf u^*-K\mathbf v^*,\qquad
\mathbf u^+=\mathbf u^*-P M_f^{-1}J^T\Lambda,\qquad
\mathbf v^+=\mathbf v^*+M_{\rm ex}^{-1}K^T\Lambda.
```

`v` here collects all grain translation/rotation components. The transpose spreading gives opposite force and torque exchange; the mass-weighted projection couples incompressibility to rigidity. If a trial velocity increment b has `P b = b − Gφ` with `DGφ = Db`, adding `s P b` to the physical velocity requires reduced-pressure update `q += s ρ φ / dt`. The initial rigid correction has `s = −1`; its pressure correction has the same negative sign. Contact-induced liquid corrections use their corresponding signed potential. The C++ `CoupledStepReport.rigidImpulse` includes hydrodynamic impulses induced by contact as well as the initial rigid correction; `contactImpulse` separately contains the direct contact impulses. These impulse arrays are internal reports, not fields in recording version 1.

The free surface is the single-valued height `z = h + η(r,θ,t)`. It obeys the kinematic condition `∂tη + u_h·∇hη = w`, zero tangential viscous traction, and the reduced normal-stress boundary value

```math
q_s=\rho g_{\rm eff}\eta-\gamma\kappa+2\mu\mathbf n\cdot E\mathbf n,
\qquad E=\tfrac12(\nabla\mathbf u+\nabla\mathbf u^T),\qquad
\kappa=\nabla_h\cdot\frac{\nabla_h\eta}{\sqrt{1+|\nabla_h\eta|^2}}.
```

Here n is the outward unit normal, γ is constant surface tension and atmospheric pressure is prescribed. The wall contact line is pinned, `η(R,θ,t)=0`; the vessel floor and side wall use no slip, the azimuth is periodic and the axis uses regularity. The solver retains nonlinear advection, full surface curvature and the normal strain contraction. Its discretization and time integration are still numerical approximations requiring convergence verification. This is not a validated complete general physical simulator: it excludes overturning surfaces, gas dynamics, moving contact lines, wetting/drying grains and non-spherical grains.

## Physical and computational limits

Volume markers are collocation points. The minimum eight local cells across a grain diameter is an admission check, not proof of resolved surface stresses or converged hydrodynamics. Positive liquid gaps require at least the configured two local cell widths. Force/torque and discrete-adjoint checks do not establish settling-speed, near-contact lubrication or bed-pattern accuracy. Those need refinement studies and physical benchmarks before scientific or commercial claims.

Resource preflight runs before creating the JavaScript fluid arrays. It includes multiple full fluid fields, preparation's second fluid instance and temporary arrays, and the pressure band storage:

```text
pcBand doubles = (nth/2 + 1) × nr × nz × (nz + 1)
```

This extra vertical factor matters. Fine grains throughout a 30 mm vessel require far more cells than the tiny example. The current signed indexing and band-preconditioner limits can reject such a case even on a machine with more RAM. No automatic coarse replacement is made.

The full-vessel target needs scalable pressure solving, mesh resolution and time integration, plus verified treatment of narrowing liquid gaps. Increasing the allocation limit does not supply those algorithms or establish a usable HPC implementation.

The preparer's report covers preparation and native fluid arrays. The native coupler separately applies conservative allocation checks for fluid checkpoints, grain/interpolation storage, constraint workspaces and contact matrices before those allocations. A case can pass preparation and still be refused at this later stage.

`resourceBudgetBytes` is **not a hard limit on total process memory**. Allocator and operating-system overhead, compiler memory and recording/browser memory fall outside that array budget. Do not interpret a passing preflight as assurance that an arbitrary grain case fits or completes. Large production runs need a revised solver/memory architecture and measured scaling, not simply a larger budget value.

## File protocols

Prepared cases use `FNSCASE1`, version 1. All integers and IEEE-754 doubles are little-endian with no structure padding. `prepare-case.mjs` is the writer and `run.cpp` is the reader. The fixed 160-byte header contains magic, version, grid dimensions, pinned flag, resource budget, fourteen physical/time/contact doubles, then grain/array/JSON counts. It is followed by UTF-8 case JSON, 96-byte grain records (u64 ID and eleven f64 values), and keyed arrays (u32 key, u64 element count, f64 elements). Array keys are `0…14`, `20`, `62…64`. This is a prepared initial state, not a restart/checkpoint format.

The UTF-8 JSONL recording is `faraday-native-recording`, version 1, with one object per line:

1. `native-case`: format/version, string `caseJson` and `nativeFluidBytes`.
2. `frame`: time, accepted-step count, grid and free-surface geometry, staggered `u/v/w`, surface displacement and rate, grain centres/radii/densities/linear and angular velocities, and constraint diagnostics. The first frame is the actual initial state at zero time.
3. `complete`: final accepted time and step count; or `error`: refusal message. A process interruption may leave no terminator and is incomplete.

Frames are emitted after accepted steps when the next `outputInterval` threshold is reached, and at completion. Event subdivision can change which accepted time first reaches a threshold. No line is a manufactured intermediate state. Standard output is the recording; progress and refusal details go to standard error. The browser reader is `faraday/native-recording.js`. Preserve the case, recording, preparation report and logs together when sharing results.

Basic schema/serialization verification is available with `node native/check-case-schema.mjs`. Numerical component tests and native/browser comparison checks are separate from a physical convergence study.
