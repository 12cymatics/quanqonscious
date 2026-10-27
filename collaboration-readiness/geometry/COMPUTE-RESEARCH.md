# Geometric compute structure: evidence and critique

Research date: 2026-10-23. Scope: instruction design, not repository implementation or GPU activation. Device and project runtime capability remain unverified. Recommendations below are engineering inferences from the cited sources, not demonstrated project speedups.

## Six distinct layers

1. Physical domain and boundary conditions: disk, plate, periodic quotient, finite basin.
2. Mathematical operator: incidence matrices, stencil, graph Laplacian, constitutive and coupling terms.
3. Storage: row-major, column-major, blocked arrays, Morton/Hilbert permutation, sparse representation.
4. Execution: loop order, CPU tasks, GPU workgroups, dependency DAG and synchronization.
5. Hardware: SIMD execution width, caches, memory system and interconnect.
6. Retrieval: indices linking definitions, files, evidence and checkpoints (covered in the companion retrieval review).

A justified map between layers is required. Similar-looking geometry is not proof of equivalent topology, locality or correctness. “Geometric calls” is not a standard CPU/GPU API category: use the concrete terms kernel dispatch, iteration space, memory layout, task graph and topology map. A 3D dispatch does not itself create a torus; a folder named hypercube does not change CPU routing.

## Toroidal and hypercube structures

MPI explicitly offers *virtual* Cartesian and graph topologies, with periodicity per Cartesian dimension. These are communication descriptions, not a guarantee of matching physical links. [MPI topology overview](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node224.htm), [Cartesian constructor](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node192.htm).

For an explicitly periodic lattice, use per-axis modular coordinates and verified halo exchange. For finite/nonperiodic axes, implement their actual boundary operator instead. Degenerate extents of one or two cells can produce duplicate/self-neighbors; preserve the specified multigraph/stencil weights rather than deduplicating automatically. A ring buffer is temporal storage reuse, not a periodic physical domain.

A d-dimensional hypercube graph has binary vertex labels and edges between labels differing in one bit; the neighbor in dimension k is `v XOR (1 << k)`. Dimension-ordered exchanges support particular reductions, transforms and communication patterns. Their benefit depends on message size, mapping, synchronization and physical communication. [Argonne hypercube template](https://www.mcs.anl.gov/~itf/dbpp/text/node124.html), [Bertsekas et al., optimal communication algorithms](https://www.mit.edu/people/dimitrib/OptimalCA.pdf).

For a supplied Q4 graph, preserve its 16 vertices, adjacency, orientation and weights. Do not equate a simple Q4 graph with an arbitrary sampled four-torus: periodic lattices can have different degree and edge multiplicity. A Gray-code ordering changes traversal, not the operator. Hyperspace dimensionality is mathematical data, not a requirement for a GPU API with more than three launch dimensions; use checked flattening and an inverse map.

## Storage and execution choices

- Establish a contiguous row/column-major baseline first. Match the fast-changing loop index to actual contiguous storage. Kokkos makes memory layout and multidimensional tiling explicit and recommends matching iteration order to layout. [MDRangePolicy](https://kokkos.org/kokkos-core-wiki/API/core/policies/MDRangePolicy.html), [multidimensional parallelism](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Multi-Dimensional-Parallelism.html).
- Cache blocking and recursive traversal are real candidates, not aesthetic choices. Cache-oblivious results rely on a specified memory model; do not convert theoretical asymptotics into a claimed laptop speedup. [Prokop, MIT 1999](https://dspace.mit.edu/entities/publication/504b13f7-fc01-41d2-8537-269b0267d51f).
- SoA can make field-wise operations contiguous; AoSoA can balance vector lanes and record locality. AoS may still suit accesses requiring most fields together. Measure the actual access pattern, not a universal layout slogan. Reject precision reduction even when a vendor guide lists it as an optimization. [Intel memory layout transformations](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-layout-transformations.html).
- Morton ordering is useful in hierarchical spatial organization; p4est uses it for ownership/partitioning of adaptive forests. This supports an indexing candidate, not a universal improvement for dense stencil arrays. [p4est paper](https://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf).
- Compact Hilbert indices accommodate unequal dimension cardinalities. Treat index generation, inverse mapping, padding, neighbor lookup and layout conversion as costs. A one-dimensional order cannot keep every multidimensional adjacency consecutive. [Hamilton and Rau-Chaplin paper](https://www.cs.dal.ca/~arc/publications/1-30/paper.pdf).
- For irregular operators, partition their dependency graph using measured work/communication weights. METIS distinguishes edge-cut and communication-volume objectives; minimum edge-cut is not automatically minimum runtime. Reordering must leave all cross-partition dependencies present. [METIS manual](https://karypis.github.io/glaros/files/sw/metis/manual.pdf).
- Distinguish sparse **storage of the same operator** from a **sparse-grid discretization**. The latter changes the approximation space and uses regularity assumptions; its error claims are conditional, not a free exact compression. [Bungartz and Griebel, Sparse grids](https://www.cambridge.org/core/journals/acta-numerica/article/abs/sparse-grids/47EA2993DB84C9D231BB96ECB26F615C).

## GPU launch and synchronization

WebGPU distinguishes workgroup size from dispatch workgroup counts and validates both against device limits. Workgroup coordinates label execution; they do not prescribe physical locality. Query adapter/device features and limits instead of treating defaults as measured hardware capability. [WebGPU specification](https://www.w3.org/TR/webgpu/).

WGSL barriers have workgroup execution scope and require uniform control flow. Avoid placing a barrier after a divergent early return for partial edge tiles. Use masked work while keeping required participants at barriers. A workgroup barrier cannot order all tiles in a global simulation step; use correctly ordered dispatch phases and explicit resource dependencies. [WGSL synchronization and uniformity](https://www.w3.org/TR/WGSL/).

Metal exposes pipeline-specific `threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`. Test group shapes against these and memory limits. Never copy CUDA's warp-width assumptions onto Metal or WebGPU. [Apple threadExecutionWidth](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth).

Coalesced adjacent accesses and reusable shared-memory tiles are well-supported optimization mechanisms in CUDA. But their transaction details, bank behavior and occupancy limits are device-specific. GPU local/threadgroup tiles need full halos, correct loading ownership and barriers; larger tiles may reduce halo overhead while increasing resource pressure. [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

## Spectral structure and project application

FFTW documents precise transform conventions and different even/odd extensions for cosine/sine transforms. The cited basis choices describe library behavior; they do not authorize replacing the supplied formulation or introducing a transform-based solver. Transform normalization and zero modes are part of the contract. A finite circular vessel is not made periodic by using array wrapping. FFT-based algorithms are not inherently exact-arithmetic algorithms. [FFTW transform definitions](https://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html), [even/odd boundary conventions](https://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029.html).

Cymatica/Faraday: preserve supplied vessel geometry, material laws, fluid/plate coupling and nonlinear terms. Start with layout/tiling/phase scheduling of the existing operator. Preserve the supplied formulation, every equation, term, coupling, mode, domain and condition. Do not introduce a change of basis, transformed formulation, approximation, reduced model or replacement formula. Storage or scheduling changes are permitted only when the supplied formulation, all required operations and exact results remain intact. Equivalence proofs, labels and approval fields supply no exception. GPU visualization is not evidence that the pressure solve or coupled solver runs on GPU.

ETERNAL: optimize actual measured simulation dependencies and spatial queries; preserve collision, motion and audio timing contracts. Headless import success cannot validate natural movement, audiovisual synchronization or interactive performance. Geometry-inspired organization is welcome; all speed claims require same-workload end-to-end measurements.

Exact kernels: preserve the designated integer/rational/algebraic/symbolic domain through every intermediate and serialized representation. Fixed-width arithmetic requires proved bounds or checked overflow. Float texture transport, FFT libraries, reassociated reductions and approximate transcendental functions cannot enter an exact path. Where hardware cannot execute the required domain, report unsupported capability rather than substituting a different calculation.
