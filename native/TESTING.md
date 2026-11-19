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

One complete host run on 2026-11-14 passed all ten checks. Its measured motion
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
The current time integration constrains the old geometry and verifies the
updated geometry, which can require very small accepted steps at finite
velocity. A useful longer trajectory needs geometry-consistent coupled time
integration. Refining a full polar vessel also introduces severe near-axis
stiffness: the surface and viscous stability estimates include the highest
azimuthal mode divided by the first radial cell's radius. Cell counts and
memory alone therefore understate the full-vessel computation cost.

Positive lubrication gaps below the declared resolution are refused. A
suspended grain cannot be advanced through that unresolved gap into a bed;
an initially touching grain likewise cannot lift off through an unresolved
positive gap. The static floor check does not establish deposition,
resuspension or a usable vibrating sand bed. No replacement drag, lubrication,
packing, particle-speed cap or position repair is supplied by these tests.
