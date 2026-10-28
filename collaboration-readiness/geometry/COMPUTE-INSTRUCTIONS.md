# Geometric compute overlay — proposed instructions

Load for simulations, spatial algorithms, solver acceleration, CPU/GPU mapping, or toroidal/hypercube architecture work. This overlay does not authorize device activation, deployment, changed physical assumptions, or modified exactness requirements.

## Required design record

Before implementation, record: source mathematical definition; domain; boundaries per axis; operator and orientation; exact/numerical arithmetic contract; state shape; index map and inverse; storage layout; task graph; synchronization scopes; device capability evidence; baseline; acceptance test; rollback. Keep physical topology, mathematical graph, storage order, execution mapping and hardware topology separate.

1. Begin with the supplied geometry and laws. Preserve formal dimension and operator; identify historical inspiration separately from verified mathematical and physical claims.
2. Use toroidal wrapping only on explicitly periodic dimensions. Verify seams, corners, halo width, self/duplicate neighbors and conservation under the declared operator. Never impose periodicity on a finite vessel for convenience.
3. Use hypercube bit addressing only where the model or justified algorithm needs it. Preserve Q4 incidence signs, orientation, multiplicities and coefficients. Record how logical nodes map to memory and workers. Do not promise logarithmic total runtime from logarithmic graph diameter.
4. Prefer a measured contiguous baseline. Compare blocked row-major, SoA/AoSoA, Morton/Hilbert or graph partitioning only when the access trace motivates them. Include indexing, sorting, conversion, padding and halo costs.
5. A storage-only relabeling/permutation P must preserve the supplied formulation and every required operation. Establish `A_new = P A P^-1` and `x_new = P x` only as identities of storage addressing, not permission to change mathematical basis or formulation. Check inverse round trips, exact values, coefficients, orientations, dependencies, boundaries and operation order. Do not alter arithmetic accumulation or replace the original calculation.
6. Partition each timestep into dependency-safe phases. Each output must have a defined owner, full input version and explicit interphase ordering. Halo exchange does not license stale values or in-place order-dependent updates.
7. Query actual API limits. Separate dispatch counts from workgroup sizes. Never assume subgroup/SIMD width, nonuniform-group support or arbitrary-precision support. Barrier participation must remain valid on incomplete tiles.
8. Preserve every specified equation, term, coupling, mode, domain, initial condition and boundary condition. Do not approximate, simplify, truncate, reduce, transform or replace the requested formulation; change precision, timestep, grid, dimension, boundary, nonlinear terms, iteration limits, convergence thresholds, integration method or operation ordering; or introduce floating-point contamination. Disclosure, equivalence, approval fields and passing tests provide no exception.
9. Do not equate sparse storage with sparse-grid approximation, cache locality with semantic recall, drawing throughput with solver throughput, or a geometric directory diagram with machine efficiency.
10. Release only after the gates below pass; report unsupported/blocked paths explicitly. Never invent benchmark numbers or claim project speedups from vendor examples.

## Fidelity-preserving benchmark gates

G0 — Lock source revision, build flags, input hashes, geometry/operator version, precision, initial state, units, boundary conditions, time horizon and acceptance criteria before tuning. Keep a reference implementation and reproducible commands.

G1 — Algebra/index correctness: bijective mapping, range/overflow checks, corner/seam/halo tests, all supported dimensions, awkward non-power-of-two shapes, one/two-cell extents where valid, and empty inputs where valid. For designated DEC complexes verify applicable boundary-of-boundary identities in the declared coefficient domain. Do not assert a DEC identity for an unrelated stencil.

G2 — Complete requested calculation: compare exact values and invariant identities in the required arithmetic domain, and verify every specified equation and operation remains active. Residual, conservation and convergence checks must exercise that actual implementation without replacing it with a finite fixture or approximate solver. Preserve arithmetic ordering and summation/FMA behavior; a tolerance pass does not authorize a changed result or formulation.

G3 — Parallel safety: race and bounds checks, barrier validity, phase ordering, resource lifetime, device-loss handling, cancellation and repeated runs. Test partial tiles and halo overlaps. A timeout is failure, not a passing small sample.

G4 — Measurement: separate cold initialization/compile, warm steady-state and complete user workflow; report p50/p95 with sample count, uncertainty and timing method. Include transfers, serialization, layout conversions, synchronization, allocations, solver iterations, memory peak and energy/thermal state when available. Distinguish GPU timestamps from CPU submission time; await actual completion.

G5 — Device matrix and release: record tested hardware/OS/API/browser/compiler versions and actual feature probes. Compare equal fidelity, equal problem and equal output work. Specify minimum useful gain and allowed regressions before tuning. A faster kernel with slower complete simulation does not qualify as a user-facing improvement. Preserve a working rollback with no degraded or substitute solver.

No benchmarks or device probes have been run by this research overlay.

## Edition 5 exceptional arithmetic and timing

Do not rely on NaN propagation alone to detect GPU divergence: WGSL permits indeterminate runtime results in specified exceptional cases. Establish operation preconditions and numeric bounds, independent model-appropriate residual/balance diagnostics and observable error readback. Test actual backend boundary cases. Exact integer representations need carry/borrow and intermediate-width validation; overflow wrapping must match the declared algebra, not silently corrupt it. Source: https://www.w3.org/TR/2026/CRD-WGSL-20261021/

Record timer resolution/quantization and dispatches per sample. A tiny Q4 operation may be below measurable resolution. Preserve observable outputs when batching, and report amortised throughput separately from single-operation latency. Chrome 120 documented a particular timestamp quantization; this is historical evidence, not a universal current-device value. Probe the actual target without requiring disabled security controls. Source: https://developer.chrome.com/blog/new-in-webgpu-120#timestamp_queries_quantization

For ETERNAL include frame-time and input-response distributions under representative collision/audio load. For Cymatica hold simulated duration, transient handling and output cadence fixed. See ../DOMAIN-EVIDENCE.md and ../reviews/V5-GEOMETRY-VALIDATION.md; these additions are requirements for later target testing, not results from such testing.
