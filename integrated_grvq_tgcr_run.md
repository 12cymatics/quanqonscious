# Integrated GRVQ/TGCR Pipeline Run Log

This log captures a full execution of `integrated_grvq_tgcr.py` on the current environment.

## Command

```bash
python integrated_grvq_tgcr.py
```

## Key Outputs
- Updated parameters after 29 sutras: `[0.30784265 0.4177345  0.55021098 0.64867843 0.74595192]`
- FCI eigenvalues: `[-2.20702412 -2.02273278 -1.89468964 -1.80531036 -1.67726722 -1.49297588]`
- Ground state energy: `-2.2070241200933634`
- Quantum parameter (determinant 0 amplitude squared): `0.970877`
- Lambda vectors after entanglement synchronization:
  - Entity 0: `[0.37290774 0.37290774 0.61513567 0.61513567]`
  - Entity 1: `[0.38067783 0.38067783 0.62847756 0.62847756]`
  - Entity 2: `[0.38965015 0.38965015 0.64302169 0.64302169]`
- Cosmology mapping:
  - Effective mean lambda scalar: `0.5049784404921636`
  - Mapped scalar: `0.7330117106932809`
  - Omega_m: `0.32796140528319373`
  - Omega_lambda: `0.6720385947168063`
  - Universe age estimate: `13.227743` billion years

## Full Console Output
```
==============================================================
  FULL PIPELINE: 29 SUTRAS + FCI + R4 ENTANGLEMENT + AGE
==============================================================

Initial parameters: [0.75 0.2  0.91 0.47 0.01]
Updated parameters after 29 sutras: [0.30784265 0.4177345  0.55021098 0.64867843 0.74595192]

FCI Hamiltonian shape: (6, 6)
Slater determinants:
   Index 0 => (0, 1)
   Index 1 => (0, 2)
   Index 2 => (0, 3)
   Index 3 => (1, 2)
   Index 4 => (1, 3)
   Index 5 => (2, 3)
Eigenvalues: [-2.20702412 -2.02273278 -1.89468964 -1.80531036 -1.67726722 -1.49297588]
Ground state energy: -2.2070241200933634
Ground state wavefunction (lowest eigenvector): [-9.85331162e-01  1.49195659e-01  4.78632140e-02 -6.48883354e-02
 -1.90183652e-02 -2.72301092e-04]

Quantum parameter (amplitude^2 of determinant 0) = 0.970877

R4 entanglement: initial lambda vectors for entities:
  Entity 0: lambda_vec = [0.31509046 0.42756959 0.56316508 0.66395083]
  Entity 1: lambda_vec = [0.32233827 0.43740468 0.57611918 0.67922324]
  Entity 2: lambda_vec = [0.32958608 0.44723977 0.58907328 0.69449564]

R4 entanglement: lambda vectors after synchronization steps:
  Entity 0: lambda_vec = [0.37290774 0.37290774 0.61513567 0.61513567]
  Entity 1: lambda_vec = [0.38067783 0.38067783 0.62847756 0.62847756]
  Entity 2: lambda_vec = [0.38965015 0.38965015 0.64302169 0.64302169]

GRVQ/TGCR-style cosmology parameters after entanglement mapping:
  Effective mean lambda scalar: 0.5049784404921636
  Mapped scalar in [0,1]: 0.7330117106932809
  Omega_m: 0.32796140528319373
  Omega_lambda: 0.6720385947168063
  Omega_k, Omega_r: 0.0 0.0
  R0 (km/s/Mpc): 69.5
  w (equation-of-state): -1.0
Recalculated universe age: 13.227743 billion years

[Pipeline complete]
```
