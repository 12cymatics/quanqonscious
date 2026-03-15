# Formal Asset Register: Hybrid Quantum-Classical 29-Sutra Platform

## Scope and Objective
This register isolates the commercial and technical asset base of the repository
into explicit units that can be defended, benchmarked, licensed, and productized
without collapsing doctrine into generic simulation framing.

The guiding product surface for external packaging is:

> **Deterministic 29-operator hybrid execution engine with serial, concurrent,
> and parallel benchmarking plus structured report artifacts.**

## Asset Register Table

| Asset ID | Asset Name | Evidence (Repo/Branch) | Maturity | Uniqueness | Commercial Route | Primary Weaknesses | Priority |
|---|---|---|---|---|---|---|---|
| A1 | 29-Sutra Runtime Kernel | `sutra_simulator.py` (`HybridQuantumClassicalSimulator`, `run_serial`, `run_concurrent`, `run_parallel`) | High | High | Core runtime licensing, OEM embedding, SDK/API base | Dependency fragility in default environment; requires clearer packaging/install baseline | P0 |
| A2 | Sutra Callable Corpus + Repository Semantics | `sutra_repository.py`, `primarysutra.py` (callable sutra inventory and invocation path) | Medium-High | High | Method IP core, deterministic transform library, enterprise private runtimes | Inconsistent doctrinal purity across implementations; mixed symbolic/numeric styles | P0 |
| A3 | Tri-Modal Structured Reporting Layer | `SimulationReport` and `to_dict()` serialization in `sutra_simulator.py`; JSON output pattern in `hybrid_sutra_platform.py` | High | Medium-High | Benchmark artifacts, audit trails, certification bundles, regression baselines | Report schema governance/versioning not yet formalized | P0 |
| A4 | Hybrid Platform Product Shell | `hybrid_sutra_platform.py`, `README.md`, `docs/hybrid_sutra_platform.md` | High | Medium | Product packaging, paid API, operator workflows, partner demos | Needs hardened dependency lock + reproducible deployment profile | P1 |
| A5 | One-Click Operator Console Surface | `web_server.py` and platform documentation tie-in | Medium-High | Medium | Sales demos, onboarding, investor/partner evaluation workflows | Risk of demo drift unless tightly pinned to runtime invariants | P1 |
| A6 | Argument Resolver and Execution Generality Layer | Argument resolver hooks and runtime argument adaptation in simulator/orchestration path | Medium | Medium-High | SDK ergonomics, integration portability, reduced bespoke wiring costs | Needs explicit conformance tests across full 29-sutra signature set | P1 |
| A7 | Signature Vault Concept (Data Product Layer) | Product doctrine in `docs/hybrid_sutra_platform.md` plus structured execution bundles | Medium | High | Proprietary corpus licensing, anomaly detection baselines, comparative performance index | Requires persistent storage model, schema evolution policy, integrity controls | P1 |
| A8 | GRVQ/TGCR Hybrid Bridge Modules | Branch/PR-linked hybrid simulation modules (PDE/FCI bridge implementations) | Medium | Medium-High | Applied vertical demos, premium research integrations, domain pilots | Float-heavy implementations dilute exact-symbolic differentiation | P2 |
| A9 | Exposed Runtime Scalar Governance | PR history notes and runtime scalar surfacing patterns (attenuation/gating/rebuild controls) | Medium | Medium | Tunable enterprise profiles, auditable control planes, premium configuration support | Parameter semantics need canonical docs and calibration protocol | P2 |
| A10 | Audio/FM/IPC Experimental Control Surfaces | PR history assets for multimodal controls and launcher integration | Low-Medium | Medium | Specialized interactive products, experiential control lanes | Peripheral to primary wedge; can distract from kernel commercialization | P3 |
| A11 | Formula/Proof Archive Corpus | Text/PDF proof assets and formula documents under repo/docs | Medium | Medium-High | Whitepaper support, partner diligence, technical marketing collateral | Limited direct monetization without executable/provable tie-back | P3 |
| A12 | Historical Chat/Design Memory Corpus | Historical narrative artifacts and design records | Low (direct) / High (extractive) | Low (direct) | Internal synthesis, roadmap mining, doctrine consistency checks | Not externally saleable in raw form; high curation overhead | P4 |

## Asset Class Separation (Do Not Blur)

### Core Doctrine Assets (Defining IP)
- 29-sutra operator semantics and callable corpus.
- Deterministic orchestration invariants across serial/concurrent/parallel modes.
- GRVQ/MSTVQ/TGCR coupling principles and exact-law commitments.
- Canonical aggregation and signature semantics.

### Demonstration Assets (Exposure + Distribution)
- Launchers, dashboards, web console, notebooks, and run wrappers.
- PDE/FCI bridge modules and visualization surfaces.
- Sales-facing reporting and benchmarking outputs.

**Control rule:** demonstration assets must remain strictly subordinate to doctrine
assets. Demo convenience is permitted only when it does not overwrite core
identity, invariants, or symbolic-method claims.

## Maturity Heatmap (Condensed)

| Zone | Definition | Current Items |
|---|---|---|
| Green | Production-near and externally packageable | A1, A3, A4 |
| Yellow | Valuable but requiring hardening/specification | A2, A5, A6, A7, A8, A9 |
| Orange | Supportive but non-core for near-term revenue wedge | A10, A11 |
| Gray | Internal extraction value only | A12 |

## Commercialization Wedge (Narrow External Surface)

Primary external wedge should be constrained to:
1. Deterministic 29-operator execution engine.
2. Tri-modal benchmarking (serial/concurrent/parallel).
3. Structured signature bundle output (machine-parseable artifact).

Secondary expansions (post-wedge):
- Hosted API metering,
- Enterprise benchmark certification,
- Domain-specific hybrid bridge integrations.

## Defensibility Upgrades Required

1. **Schema formalization:** versioned report/signature schema with compatibility
   guarantees and migration policy.
2. **Runtime reproducibility:** pinned dependencies, environment lockfiles,
   deterministic run recipe, and baseline CI execution checks.
3. **Doctrine fidelity tags:** explicitly label modules as `exact-symbolic core`,
   `hybrid bridge`, or `demo adapter` to prevent conceptual drift.
4. **Parameter governance:** canonical scalar registry with documented ranges,
   sensitivity notes, and audit trails.
5. **Benchmark protocol:** fixed corpus of seed inputs and mode comparisons for
   repeatable external claims.

## Priority Execution Plan

### P0 (Immediate)
- Preserve and harden A1/A2/A3 as canonical nucleus.
- Freeze artifact semantics for tri-modal report bundles.
- Publish minimal runtime contract for invoking all 29 sutras across three modes.

### P1 (Near-Term)
- Lock deployment reproducibility for platform shell + console.
- Add conformance tests for argument resolver against full sutra inventory.
- Begin persistent signature vault implementation with schema versioning.

### P2 (Selective)
- Refactor bridge modules to isolate numerical approximations from doctrine core.
- Expose runtime scalars via explicit config contracts.

### P3/P4 (Deferred)
- Keep multimodal controls and archives as optional support lanes, not primary
  product claims.

## Bottom-Line Position
The repository already contains an inspectable computational nucleus with real
commercial potential: a deterministic 29-sutra hybrid runtime with tri-modal
execution and structured outputs. Value is strongest where doctrine is encoded
as repeatable software behavior and weakest where implementation drift obscures
core identity. The highest-leverage path is not adding more conceptual breadth,
but enforcing clear asset boundaries and shipping a narrow, defensible product
surface first.
