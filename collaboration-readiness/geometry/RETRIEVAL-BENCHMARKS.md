# Bounded verification and benchmark plan

Status: broader test and benchmark plan proposed; the finite structural subset described under Edition 5 was executed. Production, device and retrieval experiments were not run. Success means either an evidence-supported recommendation or rejection of an ineffective layout; not a predetermined speedup.

## A. Exact structural verification before timing

- Q4: enumerate 16 unique bit addresses; each has 4 distinct symmetric neighbors; count 32 unique undirected edges; verify Hamming distance equals BFS distance. Include invalid dimension/address rejection.
- Gray Q4: 16 unique addresses; inverse round-trip; consecutive and closing-pair XOR have popcount 1. Do not claim all Q4 edges appear in this cycle.
- Q4 Laplacian: exact integer matrix/reference comparisons; each Walsh character satisfies L chi_s=2 popcount(s) chi_s. Verify all 16, plus spectrum multiplicities. Apply only to the defined unweighted simple graph. For the inspected project function Δ=A-D instead test Δ chi_s=-2 popcount(s) chi_s; do not overwrite its sign.
- Periodic lattice: modulo wrap and inverse shifts for each axis; explicit fixtures with sizes 1,2,3. Compare unique simple-graph neighbor count and directional-stencil contribution count separately. Include a nonperiodic-axis fixture to prevent unintended wrapping.
- Morton/Hilbert implementation: pin exact variant; exhaustive small-grid bijection/inverse; random larger-grid round-trips; key overflow/coordinate bounds; periodic seam access; compare permuted outputs against the original exact reference. Do not pad the physical domain or change a mesh to satisfy an index. Preserve the supplied formulation and its full domain.
- Actual DEC code, once accessible: test storage-index round trips and d_(k+1)d_k=0 in the required arithmetic domain; preserve Hodge matrices, orientation signs, boundary and quotient maps. Graph tests alone cannot validate DEC or Faraday physics.

## B. Retrieval experiment

Use 40–60 frozen real project questions with adjudicated source IDs: exact formula, symbol, kernel/test linkage, configuration, conflicting/stale records and cross-project ambiguity. Separate tuning and held-out questions. Keep project access permissions equal. Measure candidate retrieval and final evidence-supported answers separately.

Compare current repository search against exact-ID registry plus lexical search, then typed-graph expansion; test optional vector/HNSW only if corpus size or semantic queries justify it. A hypercube-facet experiment requires actual independent facets and must use the same corpus; do not invent facets to improve scores. Report Recall@k, precision@k, MRR for queries with a ranked primary target, unsupported-answer rate, stale-source rate, source coverage, p50/p95 latency, tokens read, memory and index build/update time. For ANN additionally compare against exhaustive same-metric vector search: ANN neighbor recall and human relevance recall are separate metrics.

Use identical corpus snapshots and source hashes. Run multiple independent builds if randomized; record seed/configuration; randomize query order and explicitly separate cold and warm caches. Include deleted/moved files and stale-index updates. Retrieval of a source is not proof the agent followed it: run agent tasks with a controlled token budget and score actual behavior. No fixed acceptance threshold is invented here; set tolerances against the user's current working baseline before running. Exact identifiers and critical specification retrieval must not become approximate.

## C. Layout experiment

Select one actual kernel with known access pattern, not an invented microbenchmark alone. Compare unchanged layout, blocked row-major, Morton and one named Hilbert variant. Keep equations, precision, sizes and boundary conditions identical. Measure end-to-end runtime and separate indexing, permutation, CPU/GPU upload/download, dispatch, synchronization and compute. Include memory overhead and cache/memory-transaction metrics where available. Verify results against the original oracle before accepting timing. For exact Q4, benchmark batch sizes and report overhead: a 16-node kernel is too small to presume GPU benefit.

Stop when all defined comparisons have correctness results and reproducible measurements, or report a specific hardware/source/access blocker. Adopt only a change that meets predeclared correctness requirements and demonstrates relevant benefit. Keep unsupported performance ideas as experiments, not mandatory production architecture.

## Edition 5 reproducibility additions

Record thread counts, library/BLAS versions, query batching and actual concurrency. If Faiss is selected, account for its documented CPU/GPU thread-safety constraints and nondeterministic construction cases; a seed is not a universal reproducibility guarantee. Verify the exhaustive oracle under the same deployment concurrency. No particular library is required. Source: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls

Predeclare top-k tie handling, relevance adjudication, no-answer/conflicting-source cases and critical-spec failure criteria. Report denominators and uncertainty; 40–60 questions are an initial diagnostic set, not a guarantee of zero failures. Recheck permissions, tombstones and source freshness on graph-expanded results as well as seeds. Community bug reports in ../reviews/V5-GEOMETRY-VALIDATION.md motivate fixtures but have not been reproduced against a project retrieval installation.

The supplied finite example suite now executes 57 named assertions, including inverses, Gray closure, size-one axes and mixed adjacency. It does not implement physical wall/free-surface boundary conditions, production DEC complexes, Hilbert encoding, fixed-width GPU overflow checks or actual retrieval/performance experiments. Those portions of the broader plan remain proposed.
