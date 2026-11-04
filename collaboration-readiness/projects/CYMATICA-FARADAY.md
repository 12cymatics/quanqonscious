# Cymatica and Faraday project instructions

Status: instruction overlay. It does not change or certify any existing solver.

## Mathematical foundation

Start from the user's supplied formulas, parameters and geometric/Vedic framework. Build an explicit chain from source definition to the complete supplied formulation, implementation, observable and validation. Do not insert a discretised substitute for a requested continuum formulation. Investigate higher-dimensional, algebraic, topological and geometric constructions where relevant; do not claim a physical correspondence merely because an attractive pattern or shared symbol exists. If a requested correspondence is unresolved, record the missing derivation and continue independent work without inventing it.

Every model must identify its variables, units, parameter provenance, domain, boundary and initial conditions, forcing, damping, constitutive laws and coupling assumptions. Record whether a value is supplied, measured, fitted or selected for an experiment. Resolve inconsistent supplied inputs openly without replacing them.

## Model integrity

- Keep modal visualisation, linearised fluid dynamics, nonlinear free-surface dynamics, plate elasticity and coupled fluid-solid evolution distinct. Display the active model, not a generic accuracy claim.
- Preserve boundary geometry, liquid depth, material properties, forcing and requested coupling. Do not substitute a cheaper solver, reduced dimensionality, lower resolution, capped state or altered equation.
- Use exact domains where the specification requires them. A proof or exact structural kernel does not remove discretisation error or measurement uncertainty from a physical simulation.
- Implement all specified equations, terms, couplings, modes, domains, initial conditions and boundary conditions. No approximations, simplifications, truncations, regularised substitutes, reduced models, transformed formulations or replacement formulas. Identify an existing noncompliant path as incomplete and repair it; disclosure, equivalence and approval fields do not authorize it.
- Do not insert guessed material constants. Each material option must reference the properties actually used, their units, applicability and provenance. Missing required data must be visible.
- Preserve numerical failures and explain their cause. Do not hide divergence, non-convergence, unsupported hardware or missing inputs behind a plausible animation.
- Deliver code only, without code comments, commentary docstrings, explanations or separate explanatory artifacts. Keep verification records internal; disclose material blockers and failed or incomplete calculations briefly when required.

## GPU and performance contract

Report separately the processor used for solving, rendering and export. GPU drawing does not prove GPU fluid solving. If GPU use is mandatory, check the required capabilities before execution and show an explicit unsupported state when absent; do not continue with a CPU substitute.

Benchmark the same equations, geometry, inputs, precision, tolerance and output requirements. Record hardware, browser/runtime, grid or basis size, step count, residuals, memory, wall time and simulated physical time. Report initialisation and sustained throughput separately. A smaller grid, relaxed tolerance or different model is a different experiment, not an unqualified speed-up.

Optimise storage, data movement, caching and parallelisation only when the supplied formulation and every required operation remain intact. Validate every change against the complete requested equations and actual implementation outputs. Do not use algebraic equivalence, a change of basis or a transformed formulation as an exception. Do not claim real-time behaviour unless sustained physical-time advancement supports it.

## Scientific and product gates

Require applicable dimensional checks, manufactured or analytic solutions, boundary-condition tests, convergence studies, conservation/energy checks with stated tolerances, stability checks and reproducibility. Define which checks apply and justify exclusions; do not claim every model obeys the same conservation law.

Verify every required calculation in its specified domain. An approximate solver, tolerance pass or convergence study cannot satisfy the full no-approximation requirement; preserve the affected calculation as incomplete until its required implementation works. For experimental calibration, keep held-out measurements separate from fitted data. Agreement with another implementation is useful but does not itself establish physical truth.

Test controls, workers, interrupted/repeated operations, pause/resume semantics, error visibility, model labels and exported results in an actual browser. Distinguish UI time, solver physical time and wall time. A static screen, mocked DOM or screenshot is not proof of evolving physics.

Build and test the exact unpacked package. Give it an obvious entrypoint and required launch route. State browser/hardware coverage, untested modes and known performance limits. Do not present a server-dependent source page as a universal double-click application. Publish only the authorized revision after its applicable gates pass.

For geometric organisation and retrieval, follow GEOMETRIC-STRUCTURE.md. Storage or indexing changes must not alter the physical domain, supplied operators or evidence identity. Carry stable IDs and revision/hash bindings across file moves; revalidate affected references.
