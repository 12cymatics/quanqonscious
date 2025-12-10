SPDX-License-Identifier: Apache-2.0
QuanQonscious: GRVQ-TTGCR Hybrid Quantum-Classical Simulation Framework
=======================================================================

Overview:
---------
QuanQonscious is a production-grade library integrating General Relativity,
Vedic Mathematics, and Quantum Computing into a unified framework—the GRVQ-TTGCR.
It provides:
  • A complete 29-sutra Vedic library (with 16 main sutras and 13 sub-sutras)
    implemented as a dedicated dependency module.
  • An advanced GRVQ Ansatz construction using 4th-order radial suppression and
    adaptive constant modulation.
  • A Full Configuration Interaction (FCI) solver with GRVQ corrections.
  • TTGCR hardware driver simulation (frequency setting, sensor feedback, entropy
    monitoring) without kill switch routines.
  • An HPC 4D PDE solver with MPI-based block-cyclic memory management using
    ``mpi4py`` for parallel processing.
  • An HPC 4D PDE solver with MPI-based block-cyclic memory management and GPU
    acceleration (leveraging CuPy and Numba CUDA kernels for A100).
  • A Bioelectric DNA Encoder module employing fractal Hilbert curve transformation.
  • Extended quantum circuit simulation using Cirq.
  • Automated performance profiling, dynamic dependency updating, and integrated
    bottleneck evaluation routines.
  • A custom Vedic quantum cipher (Maya key cryptography) for cryptographically secure
    watermarking of mathematical proofs and algorithmic outputs.

Installation:
-------------
This package requires **Python 3.12** or later, along with the following
dependencies:

  - numpy, scipy, mpi4py
This package requires Python 3.10+, along with the following dependencies:
  - numpy, scipy, cupy, numba
  - mpi4py
  - cirq
  - hashlib (standard library)
  - Other standard packages

To install the required dependencies, run:

```
pip install numpy scipy mpi4py cirq
```
    pip install numpy scipy cupy numba mpi4py cirq

The ``jaxlib`` entry pins the CPU build (``jaxlib==0.7.0``).  To use GPU
acceleration on CUDA 12 hardware you may instead install the corresponding
CUDA build, for example ``jaxlib==0.7.0+cuda12.cudnn98``.

Usage:
------
Import the main module in your application:
    from grvq_ttgcr import (VedicSutraLibrary, GRVQAnsatz, FCISolver, TTGCRDriver,
                             hpc_quantum_simulation, BioelectricDNAEncoder,
                             extended_quantum_simulation_cirq, orchestrate_simulation,
                             run_full_benchmark, FutureExtensions)

Then use the provided classes and functions to build your simulation workflow.

Visualization Utility
---------------------
The repository includes a helper script `palindromic_alloy_visual.py` that
computes the palindromic dual-lattice alloy described in the documentation and
produces a bar chart of the integer evaluations `S_k(1)`. This script requires
the `matplotlib` package. Install it with:

```
pip install matplotlib
```

Then run the script with:

```
python palindromic_alloy_visual.py
```

This prints the numeric value of the alloy and writes the figure to
`palindromic_alloy.png`. Use `--no-show` to skip opening the plot window or
`--output PATH` to save it elsewhere.

Documentation:
--------------
For detailed API documentation, please refer to the “docs/” folder included in the package.
This includes:
  - Detailed descriptions of each module and function.
  - Performance optimization guidelines.
  - Examples of integration with HPC and quantum backends.

Contact:
--------
Daniel James Elliot Meyer
Email: danmeyer85@gmail.com
Company: Daniel James Elliot Meyer

Version: 5.0 (Prototype Release)
Date: March 25, 2025
