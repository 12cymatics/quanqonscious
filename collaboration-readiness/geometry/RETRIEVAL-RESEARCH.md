# Geometric structure, layouts and retrieval

Research checked 2026-10-23. This is a proposed overlay, not a deployment or a measured speed result. Existing v3 AGENTS.md was read; the historical review used its mathematical-foundation and no-silent-simplification wording. Current instructions forbid approximations, simplifications, reductions, truncations, transformed formulations and replacement formulas, regardless of disclosure or equivalence. The Q4 source inspection below is pinned; other project mappings remain candidates until their actual source revision is inspected.

## 1. Resolve the object before choosing an implementation

The following definitions and counts are mathematical derivations, not benchmark claims.

- **Discrete hypercube Q_d:** Cartesian product of d copies of K_2. Vertices x are d-bit vectors. Undirected adjacency is x XOR (1 << j), for 0 <= j < d. For d >= 1 it has 2^d vertices, degree d, d*2^(d-1) edges, diameter d, and graph distance popcount(x XOR y). Generating all neighbors costs d output operations; computing an individual neighbor costs one fixed-width XOR when the address fits a machine word. Larger bitvectors incur multiword costs. Explicitly enumerating the graph has exponential size in d. Foster's primary parallel-computing text uses this XOR neighbor rule for hypercube communication [1].
- **Periodic discrete lattice:** vertex group Z/n_1 Z × ... × Z/n_k Z, with directional shifts x ± e_j modulo n_j. If every n_j >= 3, the usual simple nearest-neighbor graph is the Cartesian product C_n1 □ ... □ C_nk, with N=product(n_j), degree 2k and kN edges. If n_j=1 or 2, specify whether self-loops and duplicate directional neighbors are retained. For n_j=2, + and - reach the same point: a directional stencil can count two contributions while a simple graph counts one edge. Do not apply 2k degree blindly. Mixed periodic/nonperiodic axes are a different topology. MPI explicitly records per-axis dimensions and periodic flags [2].
- **Continuous flat torus:** R^k / (L_1 Z × ... × L_k Z), equivalently a closed box with opposite faces identified. [0,L_1) × ... × [0,L_k) supplies unique representatives with wrap-aware quotient topology. It is not the finite graph Q_d. A torus embedded as a doughnut in R^3 also has a metric that must not be substituted for the flat quotient metric.
- **User-defined quotient:** give the original space/complex, equivalence relation or group action, induced cells/operators, orientations, stabilizers where relevant, and metric/measure. A visual similarity or a project label is not a definition.
- **Toroidal hypercube:** ambiguous until one of the above or a precise product/quotient is specified. A graph Q_d □ C_n1 □ ... may be suitable only for genuinely independent binary facets and genuinely cyclic coordinates. C_2 is convention-sensitive. MPI's historical statement that a two-process-per-axis torus is a hypercube is about communication topology [3], not proof that all associated stencils, cell complexes or continuous spaces coincide.

### Direct Q4 applicability: exact derivation with restricted scope

For the unweighted simple Q_d graph, define Lf(x)=sum_j(f(x)-f(x XOR 2^j)). The character chi_s(x)=(-1)^popcount(s AND x) is an eigenvector with eigenvalue 2*popcount(s), because each flipped bit contributes 0 or 2 times chi_s. For Q4, eigenvalues are 0,2,4,6,8 with multiplicities 1,4,6,4,1. A Walsh-Hadamard transform evaluates all characters through d butterfly stages in O(d*2^d) arithmetic operations; its normalization and coefficient domain must be specified. This is an exact identity for this operator, not automatically for weighted operators, DEC Hodge Laplacians, a quotient, or rounded storage. Q4 has only 16 vertices: orchestration and allocation may dominate its runtime.

### Pinned project inspection: preserve the actual negative-Laplacian convention

Read-only GitHub fetch at commit `c427c2f83df21a05ab68482e07b95d7b65eb4528` verified `vedic_trainer/vedic/kernel/q4_complex.py`, Git blob `baf236047e2d6541172f4fcb94b941516f419897`. The file implements XOR toggles, four distinct unsigned neighbors, 32 unsigned edges, eight incoming/outgoing incidences per vertex, rational cochains, d0 and d1. Its `laplacian` is **Δ=A-D**, not the positive L=D-A above. Therefore its Walsh eigenvalues are **0,-2,-4,-6,-8**, with the same multiplicities. Preserve the actual engine sign; a generic positive-spectrum test would be wrong for this function. Eight oriented incidences do not mean eight distinct adjacent vertices. Index this existing source rather than duplicating it. Source reading is not execution or proof of whole-system exactness.

Pinned source: https://github.com/12cymatics/quanqonscious/blob/c427c2f83df21a05ab68482e07b95d7b65eb4528/vedic_trainer/vedic/kernel/q4_complex.py

The local checkout at `e60bad2168c5b915ad9f1efe9f0e99c0bb5e47bb` contains unrelated dirty changes, untouched here. Its Q4 working-file SHA256 matches the committed HEAD bytes (`45b2662c091abdd514de4400bab1a762d62a8386fa87e4cb78d9a8157bf42319`). The requested c427 commit object was absent locally, so the connector supplied the pinned source instead. No claim is made that either commit is the repository's current latest revision.

The storage-permutation identities f'=Pf, L'=PLP^-1 and d_k'=P_(k+1)d_k P_k^-1 describe reversible address relabeling only. They do not authorize a mathematical change of basis, transformed formulation or replacement operator. Preserve every supplied equation, operation, orientation, coefficient, Hodge operator, quotient map and boundary condition. Verify storage round trips and d_(k+1)' d_k'=0 without changing the underlying calculation. Graph adjacency alone does not specify a cell complex or its DEC operators.

## 2. Which geometry helps which task?

| Structure | Defensible use | Limit/cost |
|---|---|---|
| Binary bit addresses | Exact Q_d adjacency, independent Boolean state labels, XOR exchange schedules | Not evidence of semantic proximity; 2^d possible states; do not allocate nonexistent combinations |
| Binary reflected Gray code g(i)=i XOR (i>>1) | Consecutive states differ in one bit; reversible traversal of all d-bit states | Traversal adjacency is not all graph adjacency. For d>2 its cyclic traversal supplies only 2 of d neighbors per vertex |
| Morton/Z order | Interleave fixed-depth coordinate bits to key dyadic cells; candidate spatial partition/layout | Specify dimension, depth, bounds, signed-coordinate mapping, padding and endian conventions. Naive encoding costs O(kb), key width kb bits. Spatial neighbors can have distant keys |
| Hilbert traversal | Candidate alternative spatial ordering with variant-dependent locality | Pin dimension, variant, orientation and depth. Encoding/decoding and boundary handling have costs; 'Hilbert' alone is underspecified |
| Periodic indexing | Correct cyclic coordinate wrap and halo exchange when physical/model boundaries are periodic | Does not justify making nonperiodic physics periodic. Standard linear space-filling keys have a seam; test wrap neighbors explicitly |
| Typed dependency graph | Formula → operator → kernel → test → evidence crosslinks; reverse dependency invalidation | Needs stable IDs, verified edges, freshness and ownership; graph traversal can fan out or cycle |
| Folder tree | Human navigation, module ownership, build/runtime boundaries | Physical file placement and path names do not themselves implement an LLM retrieval algorithm |

Haverkort's primary survey separates Morton, Gray-code and multiple Hilbert traversals and makes their application-dependent differences explicit [4]. Treat a layout improvement as a hypothesis to benchmark, not as a consequence of visual elegance. A Hilbert order on spatial cells is not a Hilbert-space embedding of text. A Gray-code traversal of integer labels is not automatically the recursive Gray traversal of multidimensional subcubes.

## 3. Fast recall architecture: exact anchors, then contextual expansion

Proposed engineering design, not an established benchmark result:

1. Keep readable domain directories and stable canonical files. Do not reorganize a working repository into numeric/toroidal folders merely for appearance.
2. Maintain a generated or validated registry mapping stable IDs to paths and symbols, source revision/hash, mathematical definition IDs, implementation IDs, test IDs, evidence IDs and typed dependencies. Keep aliases explicit. Invalid paths, duplicate IDs and stale hashes fail validation.
3. Start identifier/formula/symbol queries with exact lookup and lexical search. Expand along typed links under a bounded token budget. Fetch authoritative source text, not only summaries. Report unresolved references and conflicts.
4. Add semantic retrieval only if a representative query benchmark establishes value. Store embedding model/version, chunking scheme, corpus revision, metric, normalization, index configuration and access scope. Embedding closeness means proximity under that representation, not truth, equivalence of formulas, or physical closeness.
5. A hypercube facet index is optional for independently defined binary facets. Missing combinations stay sparse. A cyclic coordinate is allowed only if its values have a real cyclic relationship; document/version order does not become cyclic just because a torus is attractive.
6. Apply existing instruction-loading rules. Registry files are data until a real tool loads them. No folder layout changes the model's trained memory or grants an agent access to unprovided content.

HNSW is an approximate proximity-graph search algorithm, not a hypercube or a toroidal folder tree [5]. Its useful empirical scaling is not a universal worst-case logarithmic guarantee. Locality-sensitive hashing uses probabilistic collisions tied to a chosen metric; Morton interleaving is not LSH [6]. Faiss documents exhaustive flat L2/IP search and nonexhaustive HNSW/IVF families; exhaustive search of compressed codes is not necessarily exact nearest neighbors under original-vector distances [7]. Even exact nearest neighbors in an embedding are not guaranteed relevant documents.

## Sources

[1] Ian Foster, Designing and Building Parallel Programs, hypercube template (primary author text, historical): https://www.mcs.anl.gov/~itf/dbpp/text/node124.html

[2] MPI Forum, MPI 4.1 Cartesian constructor (official standard documentation; particular version, not a claim it is latest): https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node224.htm

[3] MPI 1.1 topology overview (historical standard, used only to explain the two-point-axis convention): https://www.netlib.org/mpi/mpi-report-1.1/node131.html

[4] Haverkort, Sixteen space-filling curves and traversals for d-dimensional cubes and simplices, 2018 revision: https://arxiv.org/abs/1711.04473

[5] Malkov and Yashunin, Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs, 2018 revision: https://arxiv.org/abs/1603.09320

[6] Indyk and Motwani, Approximate nearest neighbors: towards removing the curse of dimensionality, STOC 1998: https://doi.org/10.1145/276698.276876

[7] Faiss maintainer documentation, indexes (retrieved 2026-10-23; pin a revision when implementing): https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
