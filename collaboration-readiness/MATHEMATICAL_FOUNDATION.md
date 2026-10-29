# Mathematical foundation and traceability

## Required starting point

For Daniel's research, simulations, games, tools and commercial methods, begin every substantive design with the supplied ancient mathematical corpus, its geometric constructions and the project's formally defined higher structures. Investigate that foundation before selecting an unrelated method. Trace every claimed application from an identified source or user-supplied definition to the mathematical object, derivation, implementation and verification evidence.

This is a strict method and traceability requirement, not permission to assert that every historical attribution is correct or that one mathematical construction governs every physical system. Pursue the foundation rigorously. Where a requested connection has not been derived, record it as unresolved and investigate it; do not invent the connection, substitute a conventional solver, or label an aesthetic resemblance a proof.

The requirement applies across the project portfolio. Its application differs by layer: mathematics requires definitions and proofs; physical models also require empirical justification; gameplay may contain clearly labelled authored rules; software infrastructure must satisfy its actual technical contracts; commercial claims require evidence of customer value. Do not describe administrative YAML, pricing decisions or an engine's graphics API as consequences of a mathematical law without a valid derivation.

## Corpus authority and provenance

1. Inventory the actual uploaded formulas, parameters, papers, source files and user-approved definitions before implementation. Record exact filenames or stable identifiers, versions and content hashes where possible.
2. Use those supplied materials as the required project inputs. Do not replace them with a recalled formula or a similarly named theory. If an input is missing or ambiguous, identify the missing item and continue independent work without fabricating it.
3. Separate source transcription from modern formalisation and project extensions. Preserve the original expression alongside the formal version, with a justified symbol mapping.
4. Attribute an ancient source only when its provenance is supported. A modern reconstruction, translation, project-specific sutra interpretation or new theorem must be labelled as such. Age, sacred status, geometric beauty and repeated occurrence are not mathematical proofs or empirical validation.
5. Do not infer definitions of GRVQ, MSTVQ or TGCR from their acronyms. Treat the relevant supplied specification and verified source revision as authoritative; otherwise mark the definition unresolved.
6. Resolve conflicting formulas, units, parameter values or boundary assumptions explicitly. Do not choose whichever version makes a test pass.

## Foundation contract for each substantive feature

Create a foundation record before claiming implementation completion. It must state:

- Feature and scientific or product objective.
- Source identifiers, version and provenance status.
- Formal objects: sets, spaces, domains, dimensions, topology, metrics, operators, arithmetic domains and units, as applicable.
- The exact statement being used, including hypotheses, quantifiers, restrictions and exceptional cases.
- Whether it is a definition, conjecture, proven result, numerical observation, empirical observation or authored creative rule.
- The derivation connecting it to this feature, including intermediate identities and conditions under which each step holds.
- How Vedic, GRVQ, MSTVQ or TGCR constructions enter the derivation, if claimed. A name appearing in a file is not an implementation mapping.
- Implementation symbols and modules, at a specific source revision.
- Required exact arithmetic and representation boundaries, every specified equation and operation, and any existing approximation or transformed formulation that must be removed to meet the request.
- Required proof certificates, regression tests, conservation checks, calibration evidence and failure criteria.
- Known limitations, unresolved calculations and the conditions under which the result must not be used.

An unknown field remains explicitly unknown. Do not complete a record with invented sources, guessed identifiers, meaningless hashes or a generic statement such as “based on geometry.”

## Consumed foundation register

`FOUNDATION_REGISTER.template.yaml` uses JSON-compatible YAML. Its exact fields are `schema_version`, `kind`, `task_id` and `equations`. The unassigned template cannot establish readiness. A project record uses `schema_version: 1`, `kind: project_record`, the signed contract's task ID and the complete signed equation inventory. The actual file path and SHA-256 are included in both the source manifest and signed contract.

Each equation identifies its statement file and digest, implementation paths, check IDs, all requirement components and every category applicability decision. Each component has an ID, kind, statement, implementation paths and references to the actual check/assertion pair that assesses it. The supported kinds are equation, term, coupling, mode, domain, initial condition, boundary condition and arithmetic. Equation, domain and arithmetic requirements cannot be excluded. Every other category that occurs in the supplied formulation must be populated; its absence cannot be justified by a worker-written approval field.

`validate_readiness.py` reads the register, compares it with the signed inventory, resolves the geometry-index links and requires executable assertions for every listed component. It executes the bound checks on the verified source. The trusted operator must establish that the complete supplied specification is represented before pinning the contract. The validator cannot derive an omitted scientific law or recover an absent production implementation from a register name.

## Higher constructs

A higher-dimensional, topological, algebraic or geometric construct is admissible only after its objects and operations are defined. Specify the relevant dimension, field or ring, orientation, inner product or metric, boundary, quotient relation and operator domain. Include only structures actually needed by the derivation.

For example, a proposed discrete differential construction must define its cells, orientations and incidence operators before claiming identities about their compositions. A proposed quotient must define its equivalence relation and show that the intended operation is well-defined on equivalence classes. These are verification requirements, not assertions that a particular project already satisfies them.

Do not identify two constructions solely because their pictures resemble one another or their vocabulary overlaps. Do not infer a physical energy, frequency, force, quantum effect or causal mechanism from an abstract geometric object without the additional model and evidence that support that interpretation.

## Evidence labels

Maintain separate fields rather than one ambiguous “verified” badge:

- Source status: supplied, independently located, attribution uncertain, or unavailable.
- Mathematical status: defined, conjectured, proved under stated hypotheses, or unresolved.
- Implementation status: absent, partial, implemented, or independently checked against the specification.
- Numerical status: not run, passed stated tests, failed stated tests, or blocked.
- Physical status: uncalibrated, calibrated within a stated domain, independently validated within a stated domain, or contradicted by stated observations.
- Product status: prototype, usability-tested, deployment-tested, or commercially validated by stated evidence.

A proof does not establish that a physical model describes reality. A numerical match does not prove an identity for every admissible input. A successful build does not establish scientific accuracy. A visually attractive cymatic image does not establish the correctness of its generating equations. Keep the scope attached to every label.

## Exceptions and unresolved mappings

If the required ancient/geometric foundation does not yet yield an implementable method for the requested objective, do not disguise another method as the requested one. Identify the exact unresolved definition, calculation, proof obligation or empirical mapping and implement its resolution. Do not replace the requested method or narrow its scope. Continue all independent work already authorised; keep required unresolved calculations visibly incomplete.

A separately requested comparison must remain separate from the required implementation and cannot satisfy any of its completion checks. Never introduce a classical, reduced or approximate comparator as a replacement for the supplied calculation or present its results as that calculation.

## Domain boundaries

- Cymatica: map supplied equations and parameters to geometry, boundary conditions, constitutive assumptions, forcing, the full requested formulation and observable outputs. An existing discretisation that replaces a requested continuum calculation remains an unresolved implementation gap.
- ETERNAL: distinguish scientifically supported mechanics and measured audio from fictional worldbuilding, authored movement tuning and visual symbolism. Historical-site claims require sources separate from game lore.
- Solver commercialisation: connect mathematical claims to reproducible objective-level results, exact certificates where applicable, resource costs and a named classical comparator. Do not claim quantum advantage, uniqueness or market value from terminology or visualisations alone.
- Sites and interfaces: display the active model, its relevant assumptions and its actual execution state. A presentation layer must not conceal an unresolved foundation or a failed calculation.
