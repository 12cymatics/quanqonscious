# Geometric structure: definitions, recall and execution

Edition 4 research extension, 23 October 2026. These are design instructions and finite structural checks, not an activated index, repository migration or accelerated simulator.

## Required separation

Keep six explicit contracts: physical domain/boundaries; mathematical graph or complex; operator and coefficient domain; data/storage ordering; task/communication/device mapping; and retrieval structure. A valid map between them is required. Do not make a finite basin periodic because toroidal indexing is convenient, or infer GPU locality from a graph drawing.

“Toroidal hypercube” must identify its exact meaning. The binary hypercube graph Q_d, a periodic sampled lattice, a continuous torus formed by opposite-face identifications, and a project-specific quotient are not interchangeable. A product or quotient needs its definition, metric, adjacency, multiplicity, orientation and invariant tests. Preserve degenerate-axis semantics without deduplicating required neighbors.

The inspected project Q4 implementation is pinned in geometry/q4-source-inspection.json. Its Laplacian is Δ=A−D, not D−A. Its Walsh eigenvalues are nonpositive. This is a source-specific convention to preserve, not a universal naming convention for every Laplacian. Four unsigned neighbors are distinct from eight incoming/outgoing incidences.

## Apply the relevant overlay

- Code/folder organisation, mathematical lookup and fast recall: geometry/RETRIEVAL-INSTRUCTIONS.md, INDEX-AND-FOLDER-GUIDE.md and RETRIEVAL-BENCHMARKS.md.
- Simulation state, CPU memory access, GPU kernels, workgroups and communication: geometry/COMPUTE-INSTRUCTIONS.md and DEVICE-CAPABILITY-MATRIX.md.
- Definitions and primary/official source links: geometry/RETRIEVAL-RESEARCH.md and COMPUTE-RESEARCH.md.

Use readable canonical modules plus a typed cross-reference index. Map definitions/formulas to operators, implementations, tests and evidence using stable IDs, source revision and content hashes. Prefer exact-ID/symbol retrieval for precise questions, then bounded graph expansion; assess semantic retrieval separately. Do not reorganise directories merely to resemble a torus or populate artificial binary categories to fill a hypercube.

Test layout candidates against the actual access trace. A Gray sequence, Morton/Hilbert order, tiled array or hypercube exchange has specific properties and costs; none is a universal speed guarantee. Measure retrieval accuracy and latency separately from cache behaviour, solver throughput and complete user-workflow latency. Preserve the supplied formulation and required arithmetic throughout, with all specified terms, couplings, modes, domains and conditions intact. Layout changes cannot authorize approximations, reduced models, transformed formulations or replacement formulas.

## Structural example checks

Run `python3 geometry/verify_structures.py` from this bundle. It exhaustively checks the small Q4 example, its specified negative Laplacian/Walsh characters, shortest bit-flip routes, Gray adjacency, selected periodic-grid semantics and a finite Morton mapping.

The length-two periodic-axis example intentionally retains both directional stencil slots. It demonstrates why duplicate slots cannot be equated with the simple unsigned Q4 graph. It does not define the user's physical fluid stencil.

These tests establish only their finite structural assertions. They do not execute the production Q4 module, prove general performance, validate a physical model, benchmark recall, or establish GPU capability. The empty index template is a design aid, not a populated schema-validated index or implemented search engine. Actual adoption requires the corpus, loader, update/invalidation process, strict schema and project acceptance evidence.

## Edition 5 review

The current independent review is reviews/V5-GEOMETRY-VALIDATION.md. It adds finite edge cases, exceptional-arithmetic and timing requirements and retrieval concurrency checks. DOMAIN-EVIDENCE.md governs semantic inspection of acceptance evidence. The 57 finite assertions are not a production solver or device certification.
