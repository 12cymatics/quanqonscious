# Quantum ansatz–GRVQ feedback trajectories

## Restoration provenance

This source package restores the declared feedback experiment after the original execution environment became unavailable. The quantum runner and coordinator were reconstructed under new provenance; each new run records its actual source hashes. The report and notebook are generated from the new run's real completed outputs. Previously observed measurements are retained separately in `recovery/completed_sweep_observation_2026-11-23.json`; they are historical observations, not replacement trial data or proof of a new run.

This experiment executes the complete selected 29-method field formula, solves the corresponding finite quantum variational problem, and feeds the resulting quantum expectations into the next field step. Its exact state, method operands, outputs, parameter decisions and feedback are retained for every trajectory.

Open **START_HERE.ipynb** for browser-based execution in qBraid Lab or Jupyter. The notebook works from the complete supplied source ZIP and provides separate controls for mathematical composition and CPU scheduling. It does not require a Linux computer: the selected notebook environment performs the computation. That environment must provide Python 3.12 or later and Node.js.

## The complete sweep

The default comparison has 36 trajectories and 72 outer steps:

| Factor | Values |
| --- | --- |
| Mathematical composition | SERIES, PARALLEL, SYMMETRIC_CONCURRENT |
| Strength of each of the 29 active methods | 1/29, 1/58 |
| Deterministic circuit starts | Seeds 7, 19, 43 |
| Quantum feedback | Enabled and disabled, paired from the same start |
| Outer steps per trajectory | 2 |
| Quantum coordinate sweeps per step | 4 |
| Rational Cayley candidate grid | -1, -7/8, …, 0, …, 7/8, 1, plus the current coordinate |

Changing an input defines another explicitly recorded experiment. All 29 field methods remain active at nonzero strength in every default trajectory. The disabled-feedback control still calculates the full field formula, circuit and expectations; it applies zero quantum feedback by the declared control definition.

From the extracted repository root:

```bash
python -m experiments.grvq_ansatz_comparison.run_latest_hybrid \
  --output experiments/grvq_ansatz_comparison/latest_hybrid_results \
  --modes SERIES,PARALLEL,SYMMETRIC_CONCURRENT \
  --strengths 1/29,1/58 --seeds 7,19,43 --feedback both \
  --steps 2 --sweeps 4 --grid-denominator 8 \
  --workers 2 --parallel-backend serial --branch-workers 2
```

`--workers` schedules independent trajectories concurrently. `--parallel-backend workers` schedules the independent branches of a PARALLEL field calculation on actual worker threads. It does not change SERIES or SYMMETRIC_CONCURRENT dependency order.

## What a step calculates

1. Start with the current 16-component exact field. The first step uses the source `initState`: the signed transform of `[3,4,5,6]` plus its source dielectric field.
2. Run all 29 source BOOK transformations under the selected composition, using operands derived from the current field. Their returned coefficients enter the source rotation matrices and their outputs enter the evolving field.
3. Complete the source distributed GRVQ terms and MSTVQ stress injection. Four designated GRVQ updates occur inside the core transformations; the remaining 25 occur in `distributedGRVQ`.
4. Compute the four signed source coefficients `lambda = inverseL(afterB)`. These coefficients define the current quantum Hamiltonian.
5. Optimize the exact three-layer circuit for that fixed Hamiltonian, using the declared rational candidate grid and exact energy comparisons.
6. Calculate all four expectations `<Zq>`. In the enabled trajectory, inject `-<Zq>/1000` into the corresponding field mode. The resulting field is consumed by the next complete source step. Advance time by the source `SIM_DT = 16667/50000`.

The recorded checkpoint after the last step retains the field, circuit parameters and time needed for the next step. Retention is distinguished from actual subsequent consumption.

## The quantum ansatz and changing Hamiltonian

The circuit begins in `|0000>`. Each of its three layers applies `Ry` on qubits 0, 1, 2 and 3, followed by CNOTs `0→1`, `1→2`, `2→3`. The final layer also contains its CNOTs. There are 12 variable rotations and nine CNOTs.

The rational parameter `t` specifies

\[
R_y(4\arctan t)=\frac{1}{1+t^2}
\begin{pmatrix}1-t^2&-2t\\2t&1-t^2\end{pmatrix}.
\]

The exact rotation matrix is evaluated directly. Angles are required only for numerical circuit exports. The Hamiltonian follows the recovered source:

\[
H(\lambda)=\sum_{q=0}^{3}\lambda_q Z_q
+\frac12\sum_{q=0}^{2}Z_q Z_{q+1}.
\]

For a fixed circuit, its derivative with respect to `lambda[q]` is `<Zq>`, so the specified feedback is

\[
\Delta\lambda_q=-\frac{\langle Z_q\rangle}{1000}.
\]

The original April implementation passed this nudge through floating-point thresholds and `Math.round`, producing approximately `round(-<Zq>)/1000000`. The new implementation performs the intended rational expression exactly and retains even small nonzero feedback. No probability is omitted because it is numerically small.

The research optimizer and outer-step schedule are explicit updates. The April browser used floating-point parameter-shift updates and frame-dependent hooks. Those schedules are not silently presented as the schedule of this experiment.

## Mathematical modes and execution backends

| Mode | Source equation | Method applications in one field step |
| --- | --- | ---: |
| SERIES | Ordered composition, ascending method IDs | 29 |
| PARALLEL | `psi + sum_s (U_s(psi) - psi)` using a shared input | 29 |
| SYMMETRIC_CONCURRENT | Ascending half-strength composition, then descending half-strength composition | 58 |

The mathematical modes are allowed to produce different fields. Equivalent serial and worker implementations of the same PARALLEL calculation must produce the same exact field. SERIES and SYMMETRIC_CONCURRENT contain state dependencies that remain in their source order. Independent trajectories can run concurrently in all modes.

## Exact representation and retained evidence

Large intermediate rational values use a shared exact expression graph. Each node stores an exact rational operation. Directed dyadic enclosures establish branch comparisons and integer-floor decisions; they do not replace evolving values. A decision that cannot be established within the explicitly configured precision limit raises an error with its retained checkpoint. No substitute state or method is inserted.

The source repetend helper contained an artificial denominator-at-most-999 guard. A second outer step required larger denominators, so that guard is removed and every required repetend is computed in full: denominator 4269 has period 158 and denominator 4267 has period 400. No coefficient scaling, clamp or substitute operand is introduced.

The source BOOK rotation matrices are unchanged. The representation adapter applies those same integer matrices and denominators to exact expression nodes instead of expanding a common denominator at every step. Raw BOOK operands and coefficients remain available in the frame records. Source-defined zero coefficients correspond to identity rotations and are recorded.

Long repetends use one exact tensor representation of the same BOOK matrix. The source matching axes are `[0,1,2,3,0,1,2,3]`, so each complete axis rotation occurs twice and the rotations commute. If a coefficient gives source-normalized integers `(c,s,d)`, the executor computes `C+iS = product(c+i*s)^2`, `D1 = product(d^2)`, and reconstructs every one of the 256 entries as `N[i,j] = sign(i,j) C^(4-h) S^h`, with `h = popcount(i xor j)` and `D = D1^4`. Equal factors retain their complete multiplicity. This changes the multiplication organization while preserving the exact source matrix and denominator. It is the deterministic representation for all inputs.

`latest_hybrid_validation/matrix_representation.json` records the actually regenerated comparison against literal source dense multiplication, including unreduced denominators and complete matrix entries. Long-period construction is checked by exact recomposition, and the complete coefficient sequence remains in its frame record. Source parity, the matrix proof, two complete worker-versus-serial feedback frames, and the independent full-output audit are regenerated from the restored executable source. Their actual status and counts come from their output files.

The original twenty-trajectory representation-equivalence checkpoints are unavailable. That historical comparison is recorded only as an earlier observation; it is not linked as a current proof or claimed to have been regenerated. The new worker comparison is retained in `latest_hybrid_backend_validation/backend_equivalence.json`, and the new full-output audit is in `latest_hybrid_results/independent_output_audit.json`.

Every trajectory retains its request, per-frame data, exact graph, parameter-search records and checkpoint. The sweep summary records completed trajectories, frame counts and paired feedback comparisons. The report is generated from those files.

## How to interpret the results

Because this Hamiltonian is diagonal, an independent diagnostic can calculate its exact ground energy by comparing all 16 computational-basis levels. That diagnostic does not select variational parameters. The per-step energy gap and ground-space probability measure performance against the Hamiltonian used in that same step.

Different mathematical modes can produce different coefficients and therefore different Hamiltonians. Their trajectories compare the coupled method dynamics and the circuit's quality against each resulting problem. They do not establish one universal best ansatz from the smallest raw energy.

The coefficients change between outer steps. A decrease in raw energy across those steps is therefore not a fixed-Hamiltonian convergence result. The feedback-enabled/disabled pair isolates the influence of the specified quantum feedback on subsequent fields and Hamiltonians. There is no general stability or joint ground-state claim for unconstrained lambda evolution.

This study integrates the selected `formulaOf` path—BOOK transformations, GRVQ and MSTVQ—with the recovered quantum circuit loop. The remaining browser MAG, TGCR, ZPE and auxiliary Maya time-step stages are outside this experiment. Source diagnostic quantities are labeled as such instead of being assigned invented feedback paths.

The earlier R2/R4 product comparison is a separate project benchmark. This feedback export does not promise its earlier large data archives or the unavailable historical refinement export. The four signed evolving Hamiltonian coefficients are not silently identified with positive radial shell widths. Earlier fixed-objective ansatz rankings are not claimed as completed closed-loop results.

## Source provenance

- The field kernel is the November 17 source `vedic_v18.51.1_exact_phi.html`, internally version 18.52.0, pinned by SHA-256 in each execution record.
- The circuit, Hamiltonian and intended feedback expression are recovered from `vedic_v21 4.html`, uploaded April 18. Version numbering alone does not establish recency.
- `source_audit/hybrid_lineage_audit.json` records the source decisions, reconstruction provenance and changes. `source_audit/april_ansatz_source_excerpt.json` preserves the previously retrieved circuit/Hamiltonian/feedback section and its Library identity; that excerpt was committed at `8d21a1e` with blob `8b60af34419a8c491e85c15c3e91d1809f10fc68`.
- Executable hashes and runtime versions belong to each actual new run. Historical measurements and their original result hash are isolated in the recovery observation file.

For the browser workflow, qBraid's [notebook guide](https://docs.qbraid.com/v2/lab/user-guide/notebooks) documents the active-kernel `%pip` setup, and its [file guide](https://docs.qbraid.com/v2/lab/user-guide/files) documents uploads and downloads. The supplied notebook uses those ordinary notebook and file operations.

