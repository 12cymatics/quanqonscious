# Ansatz source selection

Repository snapshot: `12cymatics/quanqonscious` at `e7aca179d2a492769a1f2dfb8e5b4f56e9b58fa0`, default branch `codex/create-github-repository-for-pcfe-v3.0`, head date 2026-11-21 07:07:44 UTC.

The audit read executable source. Notebook code was extracted from all 24 root notebooks (see notebook_inventory.json); output cells and narrative claims were not treated as evidence that computations succeeded.

## Selection

The strongest recent, exact and executable foundation is the combination of `grvq_canonical.py` (literal scalar ansatz) and `grvq_book_chain.py` / `vedic_field.py` (actual 29-method Q4 transitions). This is a source-quality judgment. No stored benchmark establishes that it is the most accurate ansatz on every physical objective.

`grvq_canonical.py` last changed at `ea53e9f288b563513c15bd67ff4a4d003ba0e78e`, 2026-11-04 12:37:10 UTC. Blob `0a8b4cbceba10f077d427d365c0700fd22e2a7c3`.

`grvq_book_chain.py` last changed at `e65a4bda17d7a3e10cbb15aac40162df48fa1a63`, 2026-11-04 14:11:11 UTC. Blob `300ffedae6e3446806dd135fa45d942fd704f3d2`.

The canonical scalar explicitly computes two shape functions `S1 = sin(theta)*cos(phi)*exp(-a1*r)` and `S2 = cos(theta)*sin(phi)*exp(-a2*r**2)`, reciprocal factors `(1-1/(abs(S1)+epsilon))*(1-2/(abs(S2)+epsilon))`, radial factor `1-r**2/(r**2+r0_squared)`, carrier `sin(k*(r+theta+phi)) + cos(2*k*(r+theta+phi))/2`, and modulation `1+gamma*sin(pi*r*theta*phi)`. Unevaluated transcendentals remain symbolic. The code itself has no variational optimization and no full GRVQ dynamics.

The user also retains a distinct product ansatz `prod_j(1+alpha_j*S_j)*R*f_vedic` in `Untitled5.ipynb`, cells 7, 8, 10, 11, 12, 85 and 88. Its `HybridAnsatz.build_quantum_circuit` is `pass`, so the field structure is useful but its quantum bridge must be implemented. These two forms must not be silently declared equivalent.

## Other candidates

| Candidate | Source date | Actual computation and reason not selected unchanged |
| --- | --- | --- |
| `ansatz.py` | 2025-12-26 last file history event | `p/(1+p**4)`, optional main sutras hidden in `except: pass`, H/Ry/CZ chain, fixed-plaintext cipher-derived Rz. The claimed R4 is parameter damping, not the four-factor spatial R4. No objective. |
| `H2_GRVQ_MSTVQ_Simulation_Upgraded_Part3.py` | 2025-12-28 | Three layers of four-qubit Rx/Ry/Rz and CZ ring. Useful hardware-efficient circuit structure. Default cost is Z0*Z1 only, absent Cirq returns `-1+mean(parameters)`, and floating expectation is rationalized. Not a verified H2 Hamiltonian objective. |
| `core/operators/grvq_ansatz.py` | 2026-04-10 | Rational polynomial replacements for Chladni, Bessel, harmonic and radial shapes; radial cap via min. The code differs from supplied functions. |
| `grvq_ttgcr_fci_cirq.py` | inspected at snapshot | Uses random one-/two-electron integrals and constant offdiagonal 0.05; wavefunction factor `1-r**4/R0**4` with R0=1e-16. Not a molecular FCI reference. |
| `Untitled10.ipynb`, cell 0 | uploaded 2025-06-10 | Rx(q0), Ry(q1), CX(0,1), Rz(p2*R2(p2)), CX(1,0); then equal Maya Rz rotations. Hamiltonian XX and ZZ coefficients are altered by radial suppression. The same Hamiltonian must be held fixed across comparisons. The `maya_zne` routine is a positive weighted mean, not an extrapolation to zero noise. |
| `Untitled5.ipynb`, cell 93 | uploaded 2025-06-10 | Four-qubit H/Rx/Ry/CX circuit, two-qubit Hamiltonian tensored with two spectators. Parameter updates are independent of energy and contain arbitrary perturbations/thresholds. |
| `Untitled44.ipynb`, cells 0/1 | uploaded 2025-06-10 | Full matrix ansatz builds nonunitary nearest-index band matrix then replaces it by the SVD polar factor. It is executable numerics, but not a declared gate construction. |
| `Untitled44.ipynb`, cell 8 | uploaded 2025-06-10 | CUDA-Q hypercube circuit with parameters split into theta/phi arrays but a shared incrementing index skips parameters. Host graph iteration and helper functions embedded in quantum kernel need compiler support. |
| `Untitled28.ipynb`, cell 2 | uploaded 2025-06-10 | Named four-factor radius suppression is `r/prod(1+a*r)` with rounded constants 0.618,0.318,1.414,1.732, not the requested `prod(lambda**4/(r**4+lambda**4))`. Quantum sim.run is called without measurements. |
| `notebooks/vedic_grvq_hybrid_simulator.ipynb` | 2026-01-29 | Nikhilam product uses `cross*base - deficiency_product`; the correct identity uses plus. Not trustworthy unchanged. |

## Branch check

`claude/hybrid-quantum-algorithms-2Rm7W`, head `1518de9b6ea393300aa3a916909b453b0d9fbb72`, dated 2026-10-16, has the same ansatz/hybrid/notebook file blobs as the default branch. The only matching Maya difference is `SutraWS/MayaFlow.lean`; it is not a newer variational circuit implementation.

## Concrete update recommendation

Keep both authored scalar families visible. Add the specified four-factor R4 as an explicit new variant with user-visible widths, and compare against R2 using identical objectives, grids, starts and budgets. Do not alter the Hamiltonian when changing the ansatz. Build a genuine unitary variational circuit from exact rational Q4 rotations already implemented in the repository, incorporating all 29 actual method outputs into state preparation. Compare original ordered-edge, matched-edge and symmetric orderings as distinct algorithms; they are not interchangeable when edge rotations overlap. Record optimized objective error, energy residual/variance, state fidelity, evaluation counts and gate resources. qBraid checks should run the exported circuit and compare measured-basis probabilities to the same state generated locally; hardware advantage is not established by these checks.
