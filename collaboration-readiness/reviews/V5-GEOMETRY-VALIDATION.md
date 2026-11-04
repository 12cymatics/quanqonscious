# Independent geometry, retrieval and compute audit — 23 October 2026

Historical review retained with its original attribution and observations. Its prescriptions are superseded by the current AGENTS.md and INTEGRITY_RULES.md wherever they permit approximations, transformed or replacement formulations, merely declared acceptance, or separate explanatory deliverables. These historical results are not results of the repaired validator.

## Useful findings and disposition

The v4 overlay correctly rejects mystical speed claims, silent model substitutions and confusion between structural algebra, numerical verification and physical validation. Preserve those rules. The material remaining gaps are executable coverage, scientific evidence linkage, GPU exceptional arithmetic, and benchmark operating conditions—not a reason to replace the user's full finite-basin Faraday model or exact Q4/Vedic mathematics.

**Executed evidence:** `v5_opaque_evidence_repro.py` reproduces the opaque-evidence limit; `V5-OPAQUE-EVIDENCE.json` records it. Separate candidate `../geometry/verify_structures.py` executes **57 named finite structural assertions**, compared with v4's 31. These are not 57 independent production validations. Candidate results are in `../geometry/STRUCTURAL-CHECKS.json`. No original bundle was edited; no actual project solver or device was exercised.

## P1 — readiness must remain evidence consistency, not scientific approval

`validate_readiness.py` correctly checks evidence identity, confinement and subject labels, and returns `scientific_truth: not_established`. Nevertheless, a synthetic completed Cymatica record with every gate pointing to one unrelated text file passes. This is an acknowledged architectural limitation, not a newly discovered file-security escape. `FOUNDATION_REGISTER.template.yaml` and `geometry-index.template.json` are not validated or linked by the readiness checker. Repeated prose requirements do not close that gap.

**Actionable revision:** explicitly call the command an *attestation-consistency check* wherever reporting its result. Add a domain evidence checklist with links from a feature/claim to foundation record, mathematical statement, geometry/operator revision and applicable test evidence. Do not pretend a generic schema proves physics. Require the release reviewer to verify relevance, scope and observed outcomes, with unmet required obligations blocking the corresponding claim. Do not require physical calibration to release an honestly labelled uncalibrated research tool; do require it to claim validated physical accuracy.

For Cymatica, the checklist should require explicit requested model scope and actual implementation scope, finite basin wall/bottom/free-surface conditions, forcing/damping and material provenance, spatial/temporal and solver-error evidence, physical-time advancement and held-out observations with uncertainty when physical validation is claimed. Any simplified comparator remains separately named and authorized. For exact kernels, require exact input parsing/serialization, arithmetic-domain and overflow policy, symbolic normalization, independent certificate checking where applicable, and renderer-only numeric conversion. A finite fixture is not a universal symbolic proof. For ETERNAL, require separate movement/collision/audio/visual evidence, target export/runtime and measured responsiveness; physics-rate tests alone do not measure input-to-presentation latency.

## P1 — promised structural coverage exceeds executed coverage

`RETRIEVAL-BENCHMARKS.md` honestly marks its test plan proposed. Its list is broader than `verify_structures.py`: the v4 script lacked independent BFS comparison, Gray cycle closure/inverse, size-one periodic fixtures, mixed finite/periodic fixtures, Morton inverse and invalid-input tests. Its old label “Periodic wrapping is per declared axis” was too broad: the original helper wraps **all** axes and has no per-axis declaration input. Do not describe the 31 old assertions as completing the proposed plan.

**Candidate implementation:** preserves the original 31 checks and adds 26: independent Q4 BFS/Hamming comparison; Gray closure and 256-value inverse round trips; length-one self-slot/simple-graph distinctions; mixed finite/periodic adjacency; finite corners; inverse mixed edges; Morton encode/decode over 1024 points/keys; awkward rectangle injectivity; and 15 rejected invalid-input cases. Negative Gray inputs are now rejected. The added `lattice_slots` is an explicitly scoped adjacency fixture: omitting out-of-domain neighbors is **not** an implementation or validation of no-slip, contact-line, free-surface, or other physical boundary conditions.

Still proposed, not executed: actual signed DEC complexes/Hodge maps, Hilbert variants, large/random machine-width overflow tests, production kernel relabelings, all supported physical geometry and GPU races. The Python Morton routine uses arbitrary-size integers; its result does not certify a fixed-width GPU encoding. Keep these limitations adjacent to test counts.

## P1 — GPU divergence cannot rely on NaN propagation

The current [WGSL Candidate Recommendation Draft, 21 October 2026](https://www.w3.org/TR/2026/CRD-WGSL-20261021/) specifies indeterminate runtime results in certain exceptional cases; implementations need not preserve familiar CPU IEEE exception behavior. Thus a NaN-only sentinel is not a reliable universal divergence detector. This is a spec-derived risk, not a reproduced project bug.

**Insert into compute/device instructions:** document permitted numeric range, underflow/overflow behavior and precision boundaries for each backend. Guard operation preconditions before unsafe arithmetic, retain independent residual/conservation/boundedness diagnostics suited to the model, expose error flags through readback, and test boundary values on actual target devices. For exact limb-based GPU arithmetic, prove carry/borrow, signed representation and intermediate width bounds; wraparound is acceptable only where it implements the declared ring. Reject float injection and out-of-range host serialization. Do not change scientific precision to match available hardware.

## P2 — benchmark timer resolution and workload shape

The gates already require completion-aware timing and separate end-to-end costs. Add timer resolution/quantization, number of repeated dispatches per timed sample and detectable-effect threshold. Tiny Q4 timings can lie below timer resolution. Batched timing should preserve observable results and report batch amortization separately from single-operation interactive latency; batching cannot be sold as a faster individual user operation. [Chrome's timestamp-query documentation](https://developer.chrome.com/blog/new-in-webgpu-120#timestamp_queries_quantization) documents 100-microsecond quantization for Chrome 120. This is version-specific historical evidence, not an assertion that every current browser uses that value: probe and record the actual target.

Community firsthand report [gpuweb #3952](https://github.com/gpuweb/gpuweb/issues/3952) describes confusing timestamp deltas. Treat as an unverified report for this project, useful for choosing adversarial timing cases, not measured project evidence. No flags disabling browser security should be required for ordinary benchmark claims.

For ETERNAL, report frame-time and input-latency distributions under representative destruction/audio load, not only average FPS or a solver throughput number. For Cymatica, hold simulated duration, startup/transient treatment and output sampling fixed; sustained physical-time advancement and interactive rendering responsiveness are different outcomes.

## P2 — retrieval reproducibility needs concurrency and relevance protocol

The existing corpus freezing, exact-ID route, held-out queries, same-metric exhaustive oracle and ANN/relevance separation are strong. Add thread counts, BLAS/library versions, query batch size and actual deployment concurrency. Seed alone is insufficient: official [Faiss threading documentation](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls) records nondeterministic HNSW insertion order and backend-dependent reproducibility exceptions, and distinguishes CPU/GPU thread safety. These are constraints if Faiss is selected, not reasons to require it.

Community firsthand report [Faiss #5702, opened 3 October 2026](https://github.com/facebookresearch/faiss/issues/5702) provides a nested-OpenMP reproducer reporting wrong exact-search results on a specific main revision. It remains **unverified locally** and must not be generalized to all versions. Use it to justify running the exhaustive oracle under actual concurrency, not merely trusting a library's “exact” label. No Faiss installation or project retrieval benchmark was run here.

Predeclare tie handling for top-k ANN recall, human relevance adjudication rules, abstention/no-answer cases, conflicting-authority cases and critical-spec failure criteria. Forty to sixty questions are a useful initial diagnostic set, not a universal statistical certification. Report denominators and uncertainty; a low observed critical-failure rate in a small sample cannot establish zero risk. Recheck tombstones, moved sources and permissions on graph-expanded results, not only the seed retrieval. A source hash identifies content but does not establish its authority.

## Scope and blockers

- **Completed:** read v4 geometry/compute/retrieval instructions, project overlays, foundation/readiness templates, both validators, readiness tests and prior validation audit; executed the isolated reproduction and candidate structural suite; inspected current primary documentation and named community issue reports.
- **Not attempted / inputs unavailable in this audit:** actual Cymatica/Godot solver correctness, target GPU capability and timing, physical experiment validation, production retrieval recall. Required next action: obtain authorized exact project revisions, target-device access and frozen input/evaluation records. Reason: this bundle is instructions and fixtures, not those runtime artifacts. Status **uncertain/unassessed**, not denied permission and not a known runtime failure.
- No external mutation, activation or scientific performance claim was made.
