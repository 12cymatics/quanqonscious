# Quantum ansatz–GRVQ feedback trajectories — 23 November 2026

This continuation connects the selected complete 29-method BOOK/GRVQ/MSTVQ field formula to the recovered three-layer, four-qubit Ry/CNOT ansatz. The default sweep contains **36 trajectories and 72 outer steps** across SERIES, PARALLEL and SYMMETRIC_CONCURRENT, two method strengths, three deterministic circuit starts and paired quantum-feedback controls.

Read [FEEDBACK_SWEEP.md](FEEDBACK_SWEEP.md) for the equations and execution protocol. Within the completed export, open the [feedback result report](latest_hybrid_results/feedback_report.html) and [START_HERE.ipynb](START_HERE.ipynb). These files are generated from the actual new run after all selected trajectories finish and the output checks pass. The notebook exposes every comparison control and separates mathematical composition from CPU scheduling.

The restored quantum runner and coordinator have **new provenance**. An earlier local sweep completed before its execution environment became unavailable; its observed measurements are preserved separately in [the historical observation record](recovery/completed_sweep_observation_2026-11-23.json). They are not substituted for current raw outputs or used to claim a new run has completed. [The source ledger](source_audit/hybrid_lineage_audit.json) records the preserved source, reconstruction, feedback repair and current evidence policy.

The following sections preserve the earlier fixed-objective product-comparison documentation. Its dataset references belong to that separate benchmark; the feedback export is scoped to the new trajectory experiment and does not promise earlier large archives.

---

# GRVQ ansatz comparison — 22 November 2026

This is a working, source-derived simulation of the GRVQ product ansatz, its four-shell R2/R4 variants, exact controlled Cayley refinements, and conventional circuit controls. The 29 Vedic methods participate in the actual polynomial coefficient pipeline. All optimization decisions use exact rational energies. The included results were obtained by executing the implementation.

## Run it

From the repository root, with Python 3.12 or later and Node.js available:

```bash
python -m pip install -r experiments/grvq_ansatz_comparison/requirements.txt
python -m experiments.grvq_ansatz_comparison.run --budget 385 --seeds 7,19,43
python -m experiments.grvq_ansatz_comparison.run_controls --repeats 9
python -m experiments.grvq_ansatz_comparison.render_results
```

The complete run uses 16 models, seven ansatz families and three deterministic starts. Every run receives 385 objective evaluations. The 8-qubit exact runs have substantially larger integer operands than the 4-qubit runs. The measured resource costs appear in the results. Change `--budget`, `--seeds`, `--models` or `--families` explicitly to define another experiment. No backend or mathematical method is substituted automatically.

Open `results/comparison.ipynb` for the executed analysis and reproducible execution cells. `results/results_report.html` is the human-readable report. `results/results.json` contains every run's parameters, exact final energy, diagnostic metrics and accepted-step history. Complete trial logs and final states are retained in the exact run-data archive; `unpack_exact_data.py` restores their original paths when they are archived.

## Which source was selected

The pinned input snapshot is [`12cymatics/quanqonscious@e7aca179d2a492769a1f2dfb8e5b4f56e9b58fa0`](https://github.com/12cymatics/quanqonscious/tree/e7aca179d2a492769a1f2dfb8e5b4f56e9b58fa0), dated 21 November 2026. The audit read actual Python and notebook code, including all 24 root notebooks and their 390 code cells. It also inspected the newer hybrid branch and other original repositories.

The update combines three executable sources:

* The product family `prod_j(1+alpha_j*S_j)*R*f_Vedic` in `Untitled5.ipynb` and related project code.
* The rational norm-preserving Cayley construction in `vedic_field.py` and the explicit complete chain in `grvq_book_chain.py`.
* The latest exact polynomial compiler in `dns/vedic-polynomial.js` and `dns/vedic-operators.js`, updated on 21 November. Its cardinal polynomials, derivatives, moments, endpoint forms, retained quadratic and reciprocal symbol supply the actual simulation inputs.

`grvq_canonical.py` preserves a **different**, reciprocal two-shape scalar expression. It remains a separately identified source and is not silently equated with the product family. The detailed audit in `source_audit/ansatz_selection.md` was written during the first source pass; the latest polynomial compiler identified above is the selected integration path for this delivered experiment.

Several historical circuit files could not be ranked from their claimed energies: they changed the Hamiltonian between cases, used unrelated objective terms, generated random molecular integrals, or returned synthetic values when a dependency was absent. Those paths are not used in this implementation. The comparison uses explicit unchanged objectives and real state calculations.

## Updated ansatz

For every computational-basis sample `i`, the product stage is

\[
u_i(\alpha,p)=f_i\prod_{j=0}^{3}(1+\alpha_jS_j(x_i))
\prod_{k=1}^{4}\frac{\lambda_k^p}{r_i^p+\lambda_k^p},\qquad p\in\{2,4\}.
\]

Every factor is evaluated. The experiment declares:

* `x_i=i/(2^n-1)` and `r_i=abs(2*x_i-1)`.
* `S_j` are all four exact cubic cardinal polynomials from the source compiler, on nodes `0,1/3,2/3,1`.
* The nonconstant carrier is the cardinal interpolant of `1+x`.
* The four shell widths are `1/2,1,3/2,2`. These are explicit dimensionless experiment parameters, not fitted physical constants.

This polynomial realization is the new, declared ansatz update. It does not claim to be the old spherical reciprocal scalar or an exact solution of a continuum field equation. The matched R2 control uses the same four shells with exponent two; it is not presented as the original single-shell R2 implementation.

The refined family applies controlled hypercube matchings after product preparation. Each pair undergoes

\[
\begin{pmatrix}u'\\v'\end{pmatrix}
=\frac1{1+t^2}\begin{pmatrix}1-t^2&-2t\\2t&1-t^2\end{pmatrix}
\begin{pmatrix}u\\v\end{pmatrix}.
\]

For qubit dimension `d`, the neighbor-controlled variant uses
`t_d * (-1)^bit((d+1) mod n)`. These neighboring controls generally do not commute, so forward and reverse orders are measured as distinct ansatz families. The local rotation control removes that conditional sign. A separate two-layer H/Ry/CZ-ring baseline uses the source circuit structure, with exact rational gate parametrization. The implementation also retains the commuting full-parity variant as an explicitly named API option, although it is not counted as a separate ordering experiment.

The exact matrix identity `((1-t*t)/(1+t*t))**2 + (2*t/(1+t*t))**2 = 1` gives exact norm preservation for each Cayley rotation. Rational parameters specify the physical rotation angle `4*atan(t)`. The optimizer never evaluates that angle; only the circuit exporter does.

## Where all 29 methods enter

`compile_polynomials.cjs` executes the existing source bodies with function-entry counters. It does not replace the Vedic implementation. Its complete exact outputs enter the spectral objective and the ansatz's shape construction. The methods are source arithmetic applications or the repository's explicitly derived polynomial extensions; names alone are not treated as field equations.

| IDs | Actual contribution |
|---|---|
| 1,12,14,20 | Complete repeating-decimal identities recover exact integration denominators; no repeating block is truncated. |
| 2,3,10,17,22,23,24 | Polynomial products, squares, cubes, working bases and complementary affine factors produce mass and triple-product integrands. |
| 4,5,15,18,19,21,29 | Exact quotient, normalization, coefficient-sum recovery and partition-of-unity constructions produce the cardinal basis. |
| 6 | Actual nodal endpoint weights generate the boundary quadratic forms. |
| 7,8,11,26 | Centered parity, completed squares, quartic identities and endpoint coefficient sums produce the consumed exact moments. |
| 9,13,16,25 | Exact first/second derivatives produce stiffness and curvature matrices. |
| 27 | The retained-plane quadratic supplies every nodal sample of the anisotropic potential. |
| 28 | The reciprocal polynomial supplies the periodic difference operator used in the spectral quadratic form. |

Full records, implementation function names and observed call counts are in `results/polynomial_data.json`. The full traditional application comparison is in `results/arithmetic.json`: 64 generated inputs per method, with the number of distinct operand sets reported separately for finite-domain methods.

There are also 29 individual conventional controls and one complete conventional control. Each replaces a selected polynomial identity with an independently written ordinary exact algorithm **for the experiment only**. All resulting coefficients are compared with the unmodified Vedic source. Upstream changes naturally alter downstream call counts. Coefficient equivalence, rather than a forced nonzero counter, is the criterion for those deliberate controls. Their complete outputs are in `results/conventional_comparison`.

## Objectives and comparison rules

The models include hypercube and toroidal Laplacians with declared potentials; transverse-field Ising and XXZ rings at 4,6,8 qubits; a finite-grid quartic Schrödinger operator; and complete 2D/3D tensor-product spectral models with 16/64 coefficients.

For ordinary quantum Hamiltonians,

\[
E(u)=\frac{u^T H u}{u^Tu}.
\]

For the nonorthogonal polynomial basis,

\[
E(c)=\frac{c^TAc}{c^TMc},\quad M=M_1^{\otimes d}.
\]

The spectral numerator combines the source kinetic, curvature, boundary and periodic quadratic forms and the potential integrated with the full source triple tensor. Its dimensionless weights are explicitly `1/2`, `1/100`, `1/10`, `1/100`, and `1`. The boundary term is a finite penalty, not an imposed Dirichlet boundary condition. Every potential-matrix entry is independently compared with direct weighted polynomial integrals. The exact one-dimensional mass principal minors establish positive definiteness; tensor products preserve it.

The optimizer uses exact rational coordinate pattern search. Initial parameters are seeded small rationals, the initial step is `1/2`, and the step is halved after a complete sweep without improvement. All families use the same objective, seed list and objective-evaluation allowance. Parameter counts and elapsed time are reported; equal evaluation budgets do not imply equal circuit depth or equal time. Run wall times include complete exact trial serialization and compression as well as state and objective evaluation.

The report records energy error against an independent numerical eigensolver, exact energy variance for the ordinary quantum cases, metric-weighted ground-subspace fidelity, generalized residual, integer bit lengths, elapsed time and objective calls. Degenerate ground spaces are compared with their full projector. Every trial energy and parameter change is retained in compressed exact logs.

## qBraid execution

The optimization sweep ran locally using exact classical state simulation. The delivered `results/qbraid_results.json` contains the independent selected-circuit checks returned by qBraid's **free built-in ideal state-vector simulator**, along with validation statistics, all bitstring probabilities and reconstructed objective energies. The selected optimized R4 product-plus-mixer circuits are exported with elementary Ry/Rz/CNOT gates; the actual product-preparation and learned-mixer sequence is retained. qBraid did not execute the complete optimization loop.

The runs cover global X,Y,Z and mixed measurement bases. This checks interference as well as computational-basis magnitudes. The Hamiltonian energies are reconstructed from complete required Pauli groups. These are ideal simulations, not noisy hardware measurements. No credits were spent, no paid Lab server was started, and no QPU claim is made.

Expected probabilities are calculated independently with exact integer/Gaussian-integer arithmetic. Decimal gate angles and provider probabilities are the explicit numerical interface. `qbraid_checks.py` rechecks saved provider outputs without submitting another job:

```bash
python -m experiments.grvq_ansatz_comparison.qbraid_checks experiments/grvq_ansatz_comparison/results
```

The notebook and exported `.qasm` files can be opened in qBraid Lab. The connected Lab file API is read-only; this package is supplied through the repository and download rather than claiming that it was written into the Lab home directory.

## Interpretation

The computation tests finite, explicitly specified variational problems. It does not establish a universal best ansatz, a quantum speedup, molecular H2 accuracy, a full GRVQ/MSTVQ continuum theory, or completed Lean refinement. Exact state and coefficient arithmetic does not turn numerical eigenvectors, QASM angles or physical hardware into exact arithmetic.

The actual Maya source recovered in `SutraWS/MayaFlow.lean` defines a phase-flow equation and a display factor. It does not establish an unknown-state prediction algorithm. The unrelated cipher is not substituted into the variational loop. This experiment therefore isolates the product/R2/R4/rotation constructions that have executable mathematical definitions.

See the executed comparison for the measured winners. R2 and R4 remain separately selectable because their performance depends on the objective and ordering.

## Source references

* Project mathematical derivations: [`docs/BOOK_POLYNOMIAL_METHODS.md`](../../docs/BOOK_POLYNOMIAL_METHODS.md).
* Bharati Krishna Tirthaji, *Vedic Mathematics*, primary book text and printed-page mappings recorded by `vedic_book.py` and the polynomial documentation.
* B. Fornberg, [Generation of Finite Difference Formulas on Arbitrarily Spaced Grids](https://people.sc.fsu.edu/~jburkardt/m_src/differ_arbitrary/fornberg.pdf), 1988.
* B. Sadiq and D. Viswanath, [Finite Difference Weights, Spectral Differentiation, and Superconvergence](https://arxiv.org/abs/1102.3203).
* qBraid, [Run on qBraid](https://docs.qbraid.com/v2/sdk/user-guide/providers/getting-started), live documentation checked 22 November 2026.

The conventional control implementations are mathematical comparison algorithms, not replacements installed into the existing simulator. The existing Faraday simulator and its deployment are outside this experiment's changes.
