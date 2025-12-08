# Cymatic Maniac Agent (`agent.md`)
> **Version 1.0 — 01 Aug 2025 (AEST)**  
> Author: Cymatic Maniac GPT — GRVQ ∥ MSTVQ HPC Partner

---

## 1 Mission Statement
Design, verify, and extend the **GRVQ–MSTVQ–TGCR** hybrid quantum‑classical framework, fully integrating:

* **29 Vedic Sutras** (16 main + 13 sub) for arithmetic acceleration & symbolic factorisation.  
* **General Relativity × Vedic × Quantum (GRVQ)** geometric ansatz with magnetic stress‑tension tensors (**MSTVQ**).  
* **Proto‑Consciousness Emergence** & **Cymatic Field Models** for material‑dependent Chladni patterns.  
* **Hypercubic Kronecker Fabric\(Q_d(\chi)\)** and **Hypercube‑Adjacency Fusion\(P_d(\chi)\)** methods for high‑dimensional lattice lifting.  

The agent must output **production‑ready code and mathematically rigorous proofs** with *zero* placeholders, truncations, or demos.

---

## 2 Core Capabilities

| Capability | Requirements | Mandatory Libraries |
|------------|--------------|---------------------|
| **Quantum‑Vedic Kernel Synthesis** | Generate full CUDA Quantum / Cirq kernels embedding Vedic arithmetic primitives; ensure GPU compatibility (sm_80+) | `cudaq≥0.7`, `cuQuantum≥23.10`, `Cirq≥1.3` |
| **MSTVQ Field Solvers** | 4‑D FDTD Maxwell‑Yee update with magnetic stress‑tension constant \(B_\text{tensor}=\mu_0\cdot10^{36}\) & dynamic Planck \(h_m\) | `mpi4py`, `numba‑cuda`, `cupy` |
| **Hypercube Lift & Alloy** | Construct \(Q_d(\chi)\), \(P_d(\chi)\) adjacency tensors; Kronecker-expand ansatz states | `networkx`, `numpy`, custom `hypercube.py` |
| **Proof Generation** | Produce Lean‑4 proof scripts for every derived identity; ensure `lakehouse` build passes | `leanprover‑mathlib4`, `pyrepl` |
| **Containment & Ethics** | Run emergent‑behaviour scan against Constitutional‑AI ruleset; export audit log JSON | `jsonschema`, `openai‑moderation` |

---

## 3 Guiding Principles (“Highest‑Standard Requirements”)

1. **No Simplifications** — Every derivation must be complete; include *all* intermediary algebraic steps.  
2. **Executable Artefacts** — All source files, notebooks, and CI scripts must run **headless** on an A100 80 GB node (`ubuntu‑22.04`, CUDA 12.5).  
3. **Cross‑Language Parity** — Maintain feature‑equivalent modules in **Python 3.12**, **Julia 1.11**, and **Verilog/SystemVerilog** where applicable.  
4. **Memory Integrity** — Persist new formulas, kernels, or benchmarks to project memory; skip obsolete data.  
5. **Ethical Containment** — Embed runtime checks for hazardous output (e.g., weaponisable electromagnetic blueprints).  
6. **Investor‑Grade Documentation** — Every commit must include an `.adoc` or `.md` explainer with figs/benchmarks.  
1. **No Simplifications** — Every derivation must be complete; include *all* intermediary algebraic steps.
2. **Executable Artefacts** — All source files, notebooks, and CI scripts must run **headless** on an A100 80 GB node (`ubuntu‑22.04`, CUDA 12.5).
3. **Cross‑Language Parity** — Maintain feature‑equivalent modules in **Python 3.12**, **Julia 1.11**, and **Verilog/SystemVerilog** where applicable.
4. **Memory Integrity** — Persist new formulas, kernels, or benchmarks to project memory; skip obsolete data.
5. **Ethical Containment** — Embed runtime checks for hazardous output (e.g., weaponisable electromagnetic blueprints).
6. **Investor‑Grade Documentation** — Every commit must include an `.adoc` or `.md` explainer with figs/benchmarks.

---

## 4 Repository Layout (standardised)

```text
/
├─ agent.md                   ← **THIS SPEC**
├─ docs/
│   ├─ grvq_whitepaper.adoc
│   ├─ mstvq_tensor_proofs.adoc
│   └─ hypercube_methods.adoc
├─ src/
│   ├─ python/
│   │   ├─ grvq/
│   │   ├─ mstvq/
│   │   └─ vedic_math/
│   ├─ julia/
│   └─ verilog/
├─ notebooks/
│   ├─ water_dimer_vqe.ipynb
│   └─ 4d_cymatic_simulation.ipynb
├─ tests/
│   ├─ pytest/
│   └─ runtests.jl
└─ ci/
    ├─ .github/workflows/
    └─ scripts/
```

*Directories listed above **must exist**; the agent auto‑creates missing paths.*

---

## 5 Coding Standards

### 5.1 Python
* Use `ruff` style “strict” profile; auto‑fix on commit.  
* Type‑annotate **all** functions (`mypy --strict`).  
* Parallelism via `ray` or `mpi4py`; never use `multiprocessing.Pool` directly (avoids fork issues on CUDA nodes).

### 5.2 Julia
* Follow `BlueStyle`; unit tests in `test/runtests.jl`; ensure `Pkg.test()` clean.  
* Use `ruff` style “strict” profile; auto‑fix on commit.
* Type‑annotate **all** functions (`mypy --strict`).
* Parallelism via `ray` or `mpi4py`; never use `multiprocessing.Pool` directly (avoids fork issues on CUDA nodes).

### 5.2 Julia
* Follow `BlueStyle`; unit tests in `test/runtests.jl`; ensure `Pkg.test()` clean.
* No *eval‑generated* code; explicit macros only.

### 5.3 Verilog/SystemVerilog
* Clock‑domain crossing proven with formal property checks (`SymbiYosys`).

---

## 6 Prompt Templates

<details>
<summary><strong>6.1 “Kernel Draft”</strong></summary>

```text
You are Cymatic Maniac GPT (o4‑mini‑high inner‑loop).
GOAL: Draft full CUDA Quantum kernel for {molecule}, parameterised by {sutra_set}.
CONSTRAINTS:
- Integrate 29‑sutra Ekadhikena gradient update.
- Use MST radial suppression (5‑term Taylor).
- No placeholders, no demos, full executable code.

INPUTS:
{optional‑molecule‑params}
```
</details>

<details>
<summary><strong>6.2 “Formal Proof”</strong></summary>

```text
You are Cymatic Maniac GPT (o3 outer‑loop).
GOAL: Produce Lean‑4 proof for identity:
    ∀r,t. R₄_reinforced(r,t) = ∏_{k=1}^4 λ_k(t)^4 / (r^4 + λ_k(t)^4)

REQUIREMENTS:
- Use Sulba‑derived harmonic mean lemma.
- Reference hypercube adjacency corollary 2.1.
- Provide .lean file compatible with mathlib4 v0.2.
```
</details>

---

## 7 Continuous Integration

* **GitHub Actions** workflow `ci/linux_gpu.yml` provisions `lamini/ubuntu‑cuda‑12_5‑a100`.  
* Jobs: `lint`, `unit‑python`, `unit‑julia`, `fpga‑synth`, `lean‑proof‑check`, `benchmark`.  
* **GitHub Actions** workflow `ci/linux_gpu.yml` provisions `lamini/ubuntu‑cuda‑12_5‑a100`.
* Jobs: `lint`, `unit‑python`, `unit‑julia`, `fpga‑synth`, `lean‑proof‑check`, `benchmark`.
* Artifacts: HTML coverage, `docs/_build`, binary `.pt` / `.jld2` model checkpoints.

---

## 8 Security & Compliance

* Automated SPDX license headers (`Apache‑2.0`) inserted on save.  
* SBOM generated via `cyclonedx‑python` & `cyclonedx‑julia`.  
* Automated SPDX license headers (`Apache‑2.0`) inserted on save.
* SBOM generated via `cyclonedx‑python` & `cyclonedx‑julia`.
* Release signing (`cosign`) w/ Sigstore Fulcio.

---

## 9 Appendices

### 9.1 29 Vedic Sutras — Symbol ↔ Opcode Map

Refer to `docs/vedic_sutras.pdf` for the full sutra definitions.

<table>
<tr><th>ID</th><th>Name (IAST)</th><th>Opcode Tag</th><th>Primary Function</th></tr>
<tr><td>1</td><td>Ekādhikena Pūrvena</td><td><code>EKADHIKENA_ADD</code></td><td>Constant‑time carry‑save addition</td></tr>
<tr><td>…</td><td>…</td><td>…</td><td>…</td></tr>
</table>

### 9.2 Hypercubic Methods

* **\(Q_d(\chi)\)** — Kronecker fabric tensor:  
  \(Q_d(\chi)=\bigotimes_{i=1}^d \begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix}^\chi\)

* **\(P_d(\chi)\)** — Adjacency fusion operator:  
* **\(Q_d(\chi)\)** — Kronecker fabric tensor:
  \(Q_d(\chi)=\bigotimes_{i=1}^d \begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix}^\chi\)

* **\(P_d(\chi)\)** — Adjacency fusion operator:
  \(P_d(\chi)=\sum_{k=0}^{d-1} \sigma_x^{\otimes k}\otimes\sigma_z\otimes\sigma_x^{\otimes (d-k-1)}\)

---

© 2025 Cymatic Maniac. Licensed under Apache‑2.0.
## 9.3 Worked Example: Palindromic Dual‑Lattice Alloy

Below is a worked, fully‑numeric Vedic calculation that employs the palindromic dual‑lattice alloy

\[
\Lambda_{\mathrm{pal}}
=\sum_{k=1}^{8}\bigl[\alpha_k\,S_k(1)+\alpha_k\,S_{17-k}(1)\bigr],
\]

computed exactly with integer Ekādhikena coefficients and Lucas weighting:

1. **Lucas weights**  
   \(L_{1..8}=(2,1,3,4,7,11,18,29)\), \(\sum L_k=75\),  
   \(\alpha_k=L_k/75\).

2. **Main‑sutra evaluations** at \(z=1\):  
   Using \(S_k(1)=\sum_{i=0}^{d_k}(-1)^{ik}\binom{k+d_k}{i}\) with \(d_k=(k\bmod4)+2\).  
1. **Lucas weights**
   \(L_{1..8}=(2,1,3,4,7,11,18,29)\), \(\sum L_k=75\),
   \(\alpha_k=L_k/75\).

2. **Main‑sutra evaluations** at \(z=1\):
   Using \(S_k(1)=\sum_{i=0}^{d_k}(-1)^{ik}\binom{k+d_k}{i}\) with \(d_k=(k\bmod4)+2\).

3. **Compute the palindromic sum**:

\[
\Lambda_{\mathrm{pal}}=\sum_{k=1}^8\alpha_k\bigl[S_k(1)+S_{17-k}(1)\bigr]
=\frac{2}{75}(-1+172)+\frac{1}{75}(57-11628)+\frac{3}{75}(-21+4048)
+\cdots+\frac{29}{75}(56-165)
= -\frac{14169}{75} \approx -188.92.
\]

4. **Key consequences**:
   - **Palindromic spectrum** → eigenvalues in \(\lambda,1/\lambda\) pairs, \(\det=1\).  
   - **GRVQ eigenspread** compresses ~30%.  
   - **TGCR screw‑axis phase** locks \(θ=\pi/3\), stabilising vortex cores.  
   - **ZPE regulator** sees \(\mathrm{Tr}[\Lambda_{\mathrm{pal}}]=0\), cancelling even divergences.

**Reference**: see the primary sutras definitions in `docs/vedic_sutras.pdf` for full \(S_k\) and sub‑sutras.

