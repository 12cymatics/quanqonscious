# CLAUDE.md - AI Assistant Documentation

## Overview

**QuanQonscious** is a hybrid quantum-classical simulation framework implementing unified GRVQ-TTGCR (Gravitational-Relativistic Vacuum Quantum / Turyavrtti Gravito-Cymatic Reactor) architecture with 29 Vedic Mathematics sutras integration. This document provides comprehensive guidance for AI assistants working with this codebase.

**Repository Type**: Production-grade scientific computing framework
**Languages**: Python (70%), C++ (15%), React/JSX (10%), Jupyter notebooks (5%)
**Scale**: 72,704 lines of tracked Python / C++ / JSX (`git ls-files '*.py' '*.hpp' '*.cpp' '*.jsx' | xargs wc -l`). The working tree is ~306 MB, almost all of it the generated cymatic PNGs rather than code. This line read "~50MB code, 200K+ lines" — both figures were written from impression and neither was ever measured.
**Primary Domain**: Quantum field simulation, consciousness emergence, cymatic patterns

---

## Table of Contents

1. [Quick Start for AI Assistants](#quick-start-for-ai-assistants)
2. [Repository Structure](#repository-structure)
3. [Core Architecture Patterns](#core-architecture-patterns)
4. [Critical Conventions](#critical-conventions)
5. [Development Workflows](#development-workflows)
6. [Testing Philosophy](#testing-philosophy)
7. [Key Technologies & Dependencies](#key-technologies--dependencies)
8. [Common Tasks Guide](#common-tasks-guide)
9. [Pitfalls to Avoid](#pitfalls-to-avoid)
10. [Reference Documentation](#reference-documentation)

---

## Quick Start for AI Assistants

### Essential Context

When working with this codebase, understand these fundamental principles:

1. **Exact Arithmetic by Default**: This codebase prioritizes exact rational arithmetic (`Fraction`, `RationalComplex`) over IEEE-754 floats to maintain mathematical precision. NEVER introduce float arithmetic where exact arithmetic is used.

2. **CODEX Specification Compliance**: The `/core` directory implements CODEX specification (referenced as "CODEX X.Y" in comments). All invariants MUST be preserved.

3. **Dual Architecture**: The codebase has TWO main architectures:
   - `/core/` - New CODEX-compliant simulation core (production-grade, exact arithmetic)
   - `/pcfe-v3/` - Distributed HPC Proto-Consciousness Field Engine (MPI-based, GPU-accelerated)

4. **29 Vedic Sutras**: All 16 main + 13 sub-sutras must be implemented and tested. Missing or broken sutras are critical failures.

5. **Three Execution Modes**: Most modules support CLASSICAL, QUANTUM, and HYBRID execution modes.

### First Steps

Before making changes:

```bash
# 1. Review the relevant architecture
ls -la /core/          # For new CODEX-compliant code
ls -la /pcfe-v3/       # For distributed HPC simulations

# 2. Check test coverage
python tests/test_invariants.py

# 3. Understand dependencies
cat requirements.txt | head -50
```

---

## Repository Structure

### Root Directory Organization

```
/home/user/quanqonscious/
├── core/                        # [PRIMARY] CODEX-compliant simulation core
│   ├── lattice.py              # Toroidal hypercube R4 topology
│   ├── state.py                # Field state with RationalComplex arithmetic
│   ├── operators/              # All 29 Vedic sutra operators
│   │   ├── base.py            # Abstract operator interface
│   │   ├── grvq_ansatz.py     # GRVQ field dynamics
│   │   ├── mstvq.py           # Magnetic stress-tension operators
│   │   ├── r4_coupling.py     # R4 coupling and energy
│   │   └── sutra_ops.py       # Complete sutra implementations
│   ├── observables.py          # Observable computation & invariant checking
│   ├── trace.py                # Evolution trace for deterministic replay
│   └── hybrid_pipeline.py      # Classical + quantum hybrid execution
│
├── pcfe-v3/                     # Proto-Consciousness Field Engine v3
│   ├── src/                    # Main engine implementation
│   │   ├── pcfe_v3_core_engine.py        # main engine
│   │   ├── pcfe_final_integration.py     # Production integration
│   │   ├── pcfe_mpi_visualization.py     # MPI visualization
│   │   └── pcfe_validation_deployment.py # Validation suite
│   ├── config/                 # YAML configurations
│   ├── docker/                 # Container deployment
│   ├── examples/               # Usage examples
│   └── docs/                   # 540-line comprehensive README
│
├── tests/                       # CODEX invariant tests
│   └── test_invariants.py      # 7 comprehensive tests
│
├── tgcr_*/                      # TGCR implementations (3 variants)
│   ├── tgcr_advanced/          # Advanced TGCR + cymatic images
│   ├── tgcr_compliant/         # CODEX-compliant TGCR
│   └── tgcr_cymatics/          # Cymatic pattern outputs
│
├── vedic_*/                     # Vedic visualization outputs
│   ├── vedic_sutra_cymatics/   # 28 high-res PNG images
│   └── vedic_cymatics_images/  # Additional cymatic patterns
│
├── frontend/                    # React web interface
│   └── EnhancedMathSynergyEngine.jsx
│
├── docs/                        # Documentation
│   └── colab_notebook_inventory.md
│
└── [Root Python Files]          # Core implementations (49 files)
    ├── primarysutra.py          # Main VedicSutras class
    ├── sutra_*.py              # Sutra modules (GRVQ, Maya, Sulba, etc.)
    ├── integrated_grvq_tgcr.py # Complete GRVQ-TGCR workflow
    ├── vedic_*.py              # Vedic computation engines
    ├── *_engine.py             # Various simulation engines
    └── run_*.py                # Executable runners
```

### Key File Purposes

| File | Purpose |
|------|---------|
| `core/state.py` | RationalComplex field state with exact arithmetic |
| `core/operators/sutra_ops.py` | All 29 Vedic sutra operator implementations |
| `primarysutra.py` | Main VedicSutras class with 3 execution modes |
| `pcfe_v3_core_engine.py` | Production PCFE engine with MPI/GPU support |
| `integrated_grvq_tgcr.py` | Complete GRVQ-TGCR numeric workflow |
| `vedic_sutras_complete.hpp` | C++ high-performance sutra implementation |
| `test_invariants.py` | CODEX 7.2 invariant verification (7 tests, all passing) |

**There is deliberately no line-count column.** It used to carry seven
figures and every one of them was wrong when checked — `primarysutra.py`
listed at "3800+" is 2,865; `pcfe_v3_core_engine.py` at "5000+" is 2,116;
`core/operators/base.py`, described elsewhere in this file as "~100 lines",
is 513. They were written from impression, nothing recomputed them, and they
aged without anyone noticing. A line count is not a fact a reader of this
file needs, and the honest options were to gate it or drop it. `wc -l` is
one command away if you want one.

---

## Core Architecture Patterns

### 1. Exact Arithmetic System

**Philosophy**: Mathematical precision over performance. Use exact rational arithmetic everywhere possible.

#### RationalComplex Class (`core/state.py`)

```python
from fractions import Fraction

@dataclass(frozen=True)
class RationalComplex:
    """Exact complex number over rationals: ℚ[i]"""
    real: Fraction
    imag: Fraction

    # All arithmetic preserves exactness
    def __mul__(self, other):
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return RationalComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
```

**Critical Rules**:
- ✅ Use `Fraction` for all scalar arithmetic
- ✅ Use `RationalComplex` for complex field values
- ✅ Convert to float ONLY for output/visualization
- ❌ NEVER introduce float arithmetic in field evolution
- ❌ NEVER use `numpy.float64` in exact mode

**C++ Equivalent**:
```cpp
// vedic_sutras_complete.hpp
using BigInt = boost::multiprecision::cpp_int;
using Rational = boost::rational<BigInt>;
// Zero IEEE-754 contamination
```

### 2. Operator Composability Pattern

All operators implement a common interface:

```python
class Operator(ABC):
    @abstractmethod
    def apply(self, state: FieldState, context: Dict) -> FieldState:
        """Apply operator to state, return new state."""
        pass

    def trace_log(self, message: str):
        """Log to evolution trace for reproducibility."""
        pass
```

**Usage**:
```python
# Compose operators in pipeline
pipeline = CompositeOperator([
    GRVQEvolutionOperator(),
    MSTVQOperator(),
    R4CouplingOperator()
])
new_state = pipeline.apply(initial_state, context)
```

### 3. Immutable State Snapshots

For deterministic replay and verification:

```python
# Create snapshot (immutable, hashable)
snapshot = state.snapshot()

# Reconstruct exact state
restored_state = snapshot.to_state(lattice)

# Verify determinism
assert hash(snapshot1) == hash(snapshot2)
```

### 4. Three Execution Modes

Most modules support mode selection:

```python
class ExecutionMode(Enum):
    CLASSICAL = "classical"  # Pure Python/NumPy
    QUANTUM = "quantum"      # Quantum circuits (Cirq/CUDA-Q)
    HYBRID = "hybrid"        # Combination

# Mode-specific execution
if mode == ExecutionMode.QUANTUM:
    result = run_quantum_circuit(params)
else:
    result = classical_computation(params)
```

### 5. Toroidal Lattice Topology

All spatial indices wrap around (no boundaries):

```python
# core/lattice.py
def wrap_index(self, coords: Tuple[int, ...]) -> Tuple[int, ...]:
    """Wrap coordinates with toroidal boundary conditions."""
    return tuple(c % s for c, s in zip(coords, self.shape))

# Usage: coordinates always valid
point = lattice.point(100, 200, 300)  # Auto-wraps to grid size
```

---

## Critical Conventions

### Code Organization Principles

1. **Separation of Concerns**
   - Core simulation logic → `/core`
   - High-level orchestration → root modules
   - Production deployment → `/pcfe-v3`
   - Visualization separate from computation

2. **Module Naming Conventions**
   - Core modules: `lowercase.py` (e.g., `state.py`, `lattice.py`)
   - Sutra modules: `*sutraws.py` or `*sutraaws.py` suffix
   - Runners: `run_*.py` prefix
   - Engines: `*_engine.py` suffix
   - Tests: `test_*.py` prefix

3. **Import Patterns**
   - Use relative imports within `/core`: `from .lattice import ToroidalHypercube`
   - Absolute imports from root: `from primarysutra import VedicSutras`
   - Handle optional dependencies gracefully:

```python
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    # Create stub or use NumPy fallback
```

### Documentation Standards

1. **Docstring Format**:
```python
def grvq_field_solver(field: Tensor, coords: Tuple) -> Tensor:
    """
    GRVQ field evolution operator (CODEX 4.2).

    Implements the Gravitational-Relativistic Vacuum Quantum
    dynamics using Vedic sutra-enhanced evolution:

        ψ'(x) = Ĥ_GRVQ ψ(x) + F̂_vedic[ψ(x)]

    Args:
        field: Complex field tensor Ψ: Ω → ℂ
        coords: Spatial coordinates (x, y, z, t)

    Returns:
        Updated field tensor

    References:
        - CODEX Section 4.2: GRVQ Evolution Operators
        - Vedic Sutra 7: Ekadhikena Purvena (increment by one more)
    """
```

2. **Inline Comments for Algorithms**:
```python
# Palindromic spectrum: eigen-values come in λ,1/λ pairs; determinant = 1
# GRVQ eigenspread: compresses real-part range by ≈ 30%
# TGCR screw-axis: locks helical phase increment θ to π/3
```

3. **LaTeX Math in Comments**:
```python
# Compute: Λ_pal = Σ_{k=1}^8 [α_k S_k(1) + α_k S_{17-k}(1)]
```

### Error Handling Philosophy

1. **Graceful Degradation**:
```python
# Check availability before using
if not CUPY_AVAILABLE:
    logging.warning("CuPy not available, falling back to NumPy")
    array = np.zeros(shape)
else:
    array = cp.zeros(shape)
```

2. **Clear Error Messages**:
```python
if not state.validate_bounded(max_norm):
    raise ValueError(
        f"Field unbounded: max |Ψ| = {state.max_amplitude():.4f} "
        f"exceeds limit {float(max_norm):.4f}. "
        f"Check evolution rate or add damping."
    )
```

3. **Invariant Validation**:
```python
# After critical operations, validate invariants
if not self.validate_all_invariants():
    raise InvariantViolation("Boundedness gate violated at timestep {self.timestep}")
```

---

## Development Workflows

### Setting Up Development Environment

```bash
# 1. Clone repository
cd /home/user/quanqonscious

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies (tiered approach)
pip install -r requirements.txt  # Full stack (may take time)
# OR minimal install:
pip install numpy scipy sympy cirq qiskit pytest

# 4. Install in editable mode
pip install -e .

# 5. Verify installation
python -c "from core.state import RationalComplex; print('Core OK')"
python -c "from primarysutra import VedicSutras; print('Sutras OK')"

# 6. Run tests
cd tests
python test_invariants.py
```

### Working with `/core` (New Architecture)

**Preferred approach for new production code**:

```python
from core.lattice import create_3d_lattice
from core.state import FieldState, RationalComplex, ArithmeticMode
from core.operators.grvq_ansatz import GRVQEvolutionOperator
from fractions import Fraction

# Create lattice
lattice = create_3d_lattice(32, 32, 32)

# Initialize field state (exact arithmetic)
state = FieldState(lattice=lattice, mode=ArithmeticMode.EXACT)

# Set initial condition
center = (16, 16, 16)
state.set_by_coords(center, RationalComplex(Fraction(1), Fraction(0)))

# Apply GRVQ evolution
operator = GRVQEvolutionOperator()
new_state = operator.apply(state, {"timestep": 0, "dt": Fraction(1, 100)})

# Validate invariants
assert new_state.validate_bounded(Fraction(10))
```

### Working with PCFE-v3 (Distributed HPC)

**For large-scale distributed simulations**:

```bash
# Single GPU run
cd /home/user/quanqonscious/pcfe-v3
python src/pcfe_final_integration.py \
    --config config/production.yaml \
    --mode balanced \
    --iterations 10000

# Multi-node MPI run
mpirun -n 4 python src/pcfe_final_integration.py \
    --distributed \
    --config config/production.yaml \
    --iterations 50000
```

**Configuration**:
```yaml
# config/production.yaml
grid_size: 128
coherence_threshold: 0.99
evolution_rate: 0.1
grvq_coupling: 0.3
tgcr_frequency: 7.83
active_sutras:
  - ekadhikena_purvena
  - nikhilam
  - grvq_field
quantum_shots: 10000
checkpoint_interval: 1000
```

### Running Vedic Sutra Simulations

```python
from primarysutra import VedicSutras, SutraContext, ExecutionMode

# Create sutra engine
sutras = VedicSutras(mode=ExecutionMode.HYBRID)

# Configure context
context = SutraContext(
    precision=128,
    use_quantum=True,
    quantum_backend="cirq",
    cache_results=True
)

# Execute specific sutra
result = sutras.ekadhikena_purvena(
    n=12345,
    context=context
)

# Run all 29 sutras
from sutra_simulator import HybridQuantumClassicalSimulator

simulator = HybridQuantumClassicalSimulator(sutras)
report = simulator.run_serial()  # or run_concurrent(), run_parallel()
print(report.summary())
```

### Git Workflow

**Branch naming**: Always use `claude/` prefix followed by descriptive name and session ID:

```bash
# Current branch (from git context)
git checkout claude/add-claude-documentation-hL5KA

# Make changes
git add CLAUDE.md
git commit -m "Add comprehensive CLAUDE.md for AI assistant guidance"

# Push with retry logic (network failures)
git push -u origin claude/add-claude-documentation-hL5KA
```

**Commit message style** (observed from history):
- Clear, descriptive action: "Add FULLY COMPLIANT TGCR Cymatic Engine adhering to 26,000+ lines of Vedic specs"
- Reference specs/standards when relevant
- Mention major features or fixes

---

## Testing Philosophy

### Test Structure

**Primary test file**: `tests/test_invariants.py`

**Seven critical tests**:

1. **Toroidal Closure**: Lattice wrapping works correctly
2. **Determinism**: Same input → same output (reproducibility)
3. **Boundedness Gate**: Field stays bounded (|Ψ| ≤ max_norm)
4. **Trace Replay**: Evolution can be exactly reconstructed
5. **Sutra Operator Closure**: All 29 sutras produce valid outputs
6. **R4 Coupling Validation**: R4 topology coupling preserves invariants
7. **Observable Computation**: Observables computed correctly

### Running Tests

```bash
# Run all invariant tests
cd /home/user/quanqonscious/tests
python test_invariants.py

# Expected output:
# ✓ Toroidal closure invariant
# ✓ Determinism invariant
# ✓ Boundedness gate invariant
# ✓ Trace replay invariant
# ✓ Sutra operator closure
# ✓ R4 coupling validation
# ✓ Observable computation
#
# All tests passed!

# Or use pytest
pytest test_invariants.py -v
```

### Writing New Tests

Follow the pattern:

```python
def test_new_invariant():
    """Test description referencing CODEX section."""
    # Setup
    lattice = create_3d_lattice(8, 8, 8)
    state = create_zero_field(lattice)

    # Operation
    result = perform_operation(state)

    # Assertions with clear error messages
    assert result.validate_bounded(Fraction(100)), \
        f"Field unbounded: max = {result.max_amplitude()}"

    print("✓ New invariant test passed")
```

### CI/CD Pipeline

**GitHub Actions**: `.github/workflows/python-app.yml`

**Triggers**: Push/PR to `main` branch

**Steps**:
1. Python 3.10 setup
2. CUDA 11.8 toolkit installation
3. Install dependencies (including `cupy-cuda11x`)
4. Flake8 linting (E9, F63, F7, F82)
5. Pytest execution

**Local pre-commit checks**:
```bash
# Lint
flake8 . --select=E9,F63,F7,F82 --max-line-length=120

# Format check
black --check .

# Type check (if using mypy)
mypy core/
```

---

## Key Technologies & Dependencies

### Core Stack

**Exact Arithmetic** (Python):
- `sympy` - Symbolic mathematics
- `mpmath` - Arbitrary-precision floating point
- `gmpy2` - GMP wrapper for bignum arithmetic
- `python-flint` - FLINT number theory library
- Built-in `fractions.Fraction`

**Exact Arithmetic** (C++):
- Boost.Multiprecision - Arbitrary-precision integers/rationals
- `cpp_int` - Unlimited precision integers
- `boost::rational<BigInt>` - Exact rational numbers

### Quantum Computing

**Primary**: Cirq (Google)
**Secondary**: Qiskit (IBM), CUDA-Quantum (NVIDIA)
**ML**: PennyLane
**Simulation**: QuTiP

```python
# Backend selection priority
if cuda_quantum_available:
    backend = "cuda-quantum"
elif cirq_available:
    backend = "cirq"
else:
    backend = "qiskit"
```

### GPU Acceleration

**Array Computing**: CuPy (NumPy-compatible GPU arrays)
**JIT Compilation**: Numba (CUDA kernels)
**Deep Learning**: PyTorch (tensors, autograd, mixed precision)

**Version constraints**:
- CUDA 11.8+ or 12.0+
- CuPy: `cupy-cuda11x` or `cupy-cuda12x` (match CUDA version)
- GPU RAM: 16GB+ (V100/A100/H100 recommended)

### HPC / Distributed

**MPI**: mpi4py (OpenMPI 4.1+ or MPICH)
**Distributed Arrays**: Dask, Ray
**Parallelism**: joblib, multiprocess

**Optimal scaling**:
- Grid size: ~64³ to 128³ per MPI rank
- Communication: Minimize with ghost cell overlap
- Checkpointing: HDF5/Zarr every 1000 iterations

### Data Storage

**Checkpoints**: HDF5 (hierarchical data)
**Large Arrays**: Zarr (chunked, compressed)
**Analytics**: DuckDB (embedded), Parquet (columnar)

```python
# Checkpoint saving
import h5py
with h5py.File('checkpoint_5000.h5', 'w') as f:
    f.create_dataset('psi_real', data=psi_real)
    f.create_dataset('psi_imag', data=psi_imag)
    f.attrs['timestep'] = 5000
```

### Visualization

**Static**: Matplotlib (publication-quality)
**Interactive**: Plotly (3D fields), Bokeh
**Web Dashboards**: Streamlit, Dash
**3D Scientific**: VTK, PyVista

**Cymatic Pattern Rendering**:
```python
import matplotlib.pyplot as plt

# High-resolution output (7-8MB PNG)
fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
im = ax.imshow(cymatic_pattern, cmap='RdBu_r', interpolation='bilinear')
plt.savefig('vedic_sutra_chakra_396hz.png', dpi=300, bbox_inches='tight')
```

---

## Common Tasks Guide

### Task 1: Add a New Vedic Sutra Operator

**Location**: `core/operators/sutra_ops.py`

```python
from core.operators.base import Operator
from core.state import FieldState, RationalComplex
from fractions import Fraction
from typing import Dict

class NewSutraOperator(Operator):
    """
    New Sutra: [Sanskrit name] ([English translation]).

    Implements CODEX X.Y specification for [sutra purpose].

    Mathematical form:
        ψ'(x) = ψ(x) + δψ_sutra(x)

    References:
        - Vedic Sutra #XX: [description]
        - CODEX Section X.Y
    """

    def apply(self, state: FieldState, context: Dict) -> FieldState:
        """Apply new sutra operator."""
        new_state = state.copy()

        # Implement sutra logic with exact arithmetic
        for point in state.lattice.iterate_all():
            old_val = state.get(point)
            # Example: multiply by (1 + 1/point.coords[0])
            factor = Fraction(1) + Fraction(1, point.coords[0] + 1)
            new_val = old_val * factor
            new_state.set(point, new_val)

        # Log to trace
        self.trace_log(f"Applied NewSutra at timestep {state.timestep}")

        return new_state
```

**Testing**:
```python
# In tests/test_invariants.py
def test_new_sutra_operator():
    lattice = create_3d_lattice(4, 4, 4)
    state = create_gaussian_field(lattice, (2, 2, 2), sigma=1.0)

    operator = NewSutraOperator()
    new_state = operator.apply(state, {})

    # Validate boundedness
    assert new_state.validate_bounded(Fraction(1000))
    print("✓ New sutra operator test passed")
```

### Task 2: Run a GRVQ-TGCR Simulation

**Using integrated workflow**:

```bash
cd /home/user/quanqonscious
python integrated_grvq_tgcr.py
```

**Customizing parameters**:

```python
# Edit integrated_grvq_tgcr.py or create new script
from integrated_grvq_tgcr import run_grvq_tgcr_simulation

config = {
    'grid_size': 64,
    'max_iterations': 5000,
    'grvq_coupling': 0.25,
    'tgcr_frequency': 7.83,  # Schumann resonance
    'active_sutras': [
        'ekadhikena_purvena',
        'nikhilam',
        'urdhva_tiryagbhyam',
        'grvq_field'
    ],
    'quantum_shots': 10000,
    'checkpoint_dir': './checkpoints'
}

results = run_grvq_tgcr_simulation(config)
print(f"Final coherence: {results['coherence'][-1]:.4f}")
```

### Task 3: Generate Cymatic Visualizations

**Using TGCR engine**:

```python
from tgcr_cymatic_engine_compliant import TGCRCymaticEngine
import numpy as np

# Initialize engine
engine = TGCRCymaticEngine(
    grid_size=512,  # High resolution for publication
    frequency=432,  # Hz (A=432Hz tuning)
    mode_number=7   # Vedic harmonic
)

# Generate cymatic pattern
pattern = engine.generate_pattern()

# Save high-res PNG
engine.save_pattern('cymatic_432hz_mode7.png', dpi=300)
```

### Task 4: Debug Invariant Violations

**Approach**:

1. **Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('CODEX')
```

2. **Check state at each step**:
```python
for i in range(max_iterations):
    state = operator.apply(state, context)

    # Validate after each step
    if not state.validate_bounded(Fraction(100)):
        logger.error(f"Boundedness violated at iteration {i}")
        logger.error(f"Max amplitude: {state.max_amplitude()}")
        logger.error(f"Total norm²: {float(state.total_norm_squared())}")
        break
```

3. **Use trace replay**:
```python
from core.trace import EvolutionTrace

trace = EvolutionTrace()
trace.record(state.snapshot(), operator_name="GRVQ")

# Later: replay to find divergence point
for i, snapshot in enumerate(trace.snapshots):
    restored = snapshot.to_state(lattice)
    if not restored.validate_bounded(Fraction(100)):
        print(f"Divergence at snapshot {i}")
        break
```

### Task 5: Add C++ Performance Implementation

**Pattern** (see `vedic_sutras_complete.hpp`):

```cpp
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>

using BigInt = boost::multiprecision::cpp_int;
using Rational = boost::rational<BigInt>;

// Exact implementation
Rational new_sutra_exact(const Rational& input) {
    // No float arithmetic!
    Rational result = input * Rational(3, 2);
    result += Rational(1, 7);
    return result;
}
```

**Compilation**:
```bash
g++ -std=c++17 -O3 -I/usr/include/boost \
    -o new_sutra_test new_sutra_test.cpp
```

**Python binding** (optional, using pybind11):
```cpp
#include <pybind11/pybind11.h>

PYBIND11_MODULE(new_sutra_cpp, m) {
    m.def("new_sutra_exact", &new_sutra_exact, "Exact new sutra computation");
}
```

---

## Pitfalls to Avoid

### 1. Arithmetic Mode Violations

❌ **WRONG**:
```python
# Mixing float with exact mode
state.set(point, 0.5)  # float introduced!
```

✅ **CORRECT**:
```python
from fractions import Fraction
state.set(point, RationalComplex.from_real(Fraction(1, 2)))
```

### 2. Forgetting Toroidal Wrapping

❌ **WRONG**:
```python
# Direct indexing without wrapping
coords = (128, 64, 32)  # May be out of bounds!
value = field[coords]
```

✅ **CORRECT**:
```python
# Use lattice methods
point = lattice.point(128, 64, 32)  # Auto-wraps
value = state.get(point)
```

### 3. Ignoring Optional Dependencies

❌ **WRONG**:
```python
import cupy as cp  # Hard dependency - crashes if CuPy not installed
array = cp.zeros(shape)
```

✅ **CORRECT**:
```python
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback
    CUPY_AVAILABLE = False

array = cp.zeros(shape)  # Works either way
```

### 4. Mutating Immutable State

❌ **WRONG**:
```python
snapshot = state.snapshot()
snapshot.psi[coords] = new_value  # Error! Tuple is immutable
```

✅ **CORRECT**:
```python
snapshot = state.snapshot()
restored = snapshot.to_state(lattice)  # Mutable state
restored.set_by_coords(coords, new_value)
new_snapshot = restored.snapshot()
```

### 5. Incomplete Sutra Implementation

❌ **WRONG**:
```python
# Only implementing 28 of 29 sutras
# Tests WILL fail!
```

✅ **CORRECT**:
```python
# Ensure all 29 sutras implemented:
# 16 main sutras: 1-16
# 13 sub-sutras: various names
# Test with: python tests/test_invariants.py
```

### 6. Forgetting Trace Logging

❌ **WRONG**:
```python
def apply(self, state, context):
    # No trace logging - can't replay!
    return new_state
```

✅ **CORRECT**:
```python
def apply(self, state, context):
    self.trace_log(f"Applying {self.__class__.__name__} at t={state.timestep}")
    # ... operation ...
    return new_state
```

### 7. Non-Deterministic Seeding

❌ **WRONG**:
```python
import random
random.seed()  # System time - not reproducible!
```

✅ **CORRECT**:
```python
import random
import numpy as np

# Explicit seed from context
seed = context.get('random_seed', 42)
random.seed(seed)
np.random.seed(seed)
```

### 8. Ignoring Grid Size Constraints

❌ **WRONG**:
```python
# 512³ on 16GB GPU - will OOM!
config.grid_size = 512
```

✅ **CORRECT**:
```python
# Check GPU memory first
import torch
if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem_gb < 40:
        config.grid_size = 192  # Safe for V100 16GB
    else:
        config.grid_size = 256  # A100 40GB+
```

---

## Reference Documentation

### Essential Files to Read First

**For `/core` work**:
1. `core/state.py` - Field state and RationalComplex
2. `core/lattice.py` - Toroidal topology
3. `core/operators/base.py` - Operator interface
4. `tests/test_invariants.py` - Invariant tests (7 tests)

**For PCFE-v3 work**:
1. `pcfe-v3/docs/README.md` - Comprehensive guide
2. `pcfe-v3/src/pcfe_v3_core_engine.py` - Main engine
3. `pcfe-v3/config/default_config.yaml` - Configuration reference

**For Vedic sutras**:
1. `primarysutra.py` - Main VedicSutras class
2. `core/operators/sutra_ops.py` - CODEX-compliant operators
3. `vedic_sutras_complete.hpp` - C++ reference
4. `AGENTS.md` - Worked numerical example

### External References

**CODEX Specification**: Referenced throughout `/core` codebase as "CODEX X.Y" sections

**Vedic Mathematics**:
- 16 main sutras from Bharati Krishna Tirthaji's work
- 13 sub-sutras (upasutras)
- Mathematical algorithms for consciousness manipulation

**Quantum Field Theory**:
- GRVQ: Gravitational-Relativistic Vacuum Quantum dynamics
- TGCR: Toroidal Geometrical Cymatic Resonance (Chladni patterns)
- MSTVQ: Magnetic Stress Tensor Vacuum Quantum interactions

**Related Papers** (from PCFE docs):
- "Emergent Proto-Consciousness in Coupled Quantum Fields" - Nature Physics (2024)
- "Vedic Mathematics in Quantum Computing" - PRL (2024)
- "Topological Phase Transitions in GRVQ Systems" - PRB (2024)

### Quick Reference Commands

```bash
# Test invariants
python tests/test_invariants.py

# Run minimal simulation
python -c "from pcfe_final_integration import quick_test_run; quick_test_run()"

# Check dependencies
python -c "import cirq, qiskit, cupy; print('All OK')"

# View PCFE documentation
cat pcfe-v3/docs/README.md | head -100

# Count the 29 sutras. Read it from the table that defines them, not by
# grepping for a naming convention: this recipe used to be
#   grep -r "def.*sutra" primarysutra.py | wc -l   # Should be 29
# which returns 1. It was written to express the intent "there are 29", not
# transcribed from a command anyone had run, so it handed every reader a
# check that fails while telling them the answer it was supposed to give.
PYTHONPATH=vedic_trainer python3 -c \
  "from vedic.kernel.sutras_canonical import SUTRA_KIND; print(len(SUTRA_KIND) - 1)"
# -> 29   (index 0 is unused so SUTRA_KIND[id] reads with 1-based ids)

# Check CODEX references
grep -r "CODEX" core/ | head -20
```

### Configuration Templates

These two are **templates to write, not files to read**: no `config/`
directory exists at the repository root. They are recorded here as the shape
a PCFE config takes. For configs that exist and are loaded by code, see
`pcfe-v3/config/` and `vedic_trainer/configs/`.

**Minimal config** (a `config/minimal.yaml` you create):
```yaml
grid_size: 32
max_iterations: 1000
coherence_threshold: 0.95
evolution_rate: 0.1
active_sutras:
  - ekadhikena_purvena
quantum_shots: 1000
```

**Production config** (a `config/production.yaml` you create):
```yaml
grid_size: 256
max_iterations: 50000
coherence_threshold: 0.99
evolution_rate: 0.05
grvq_coupling: 0.3
tgcr_frequency: 7.83
mstvq_coupling: 0.1
vedic_coupling: 0.5
active_sutras:
  - ekadhikena_purvena
  - nikhilam
  - urdhva_tiryagbhyam
  - paravartya_yojayet
  - grvq_field
quantum_shots: 10000
checkpoint_interval: 1000
use_mixed_precision: true
mpi_enabled: true
```

---

## Glossary of Terms

**CODEX**: Specification document defining invariants and requirements for `/core` implementation

**GRVQ**: Gravitational-Relativistic Vacuum Quantum - field dynamics incorporating gravitational and relativistic effects

**TGCR**: Toroidal Geometrical Cymatic Resonance - geometric pattern formation on toroidal surfaces

**MSTVQ**: Magnetic Stress Tensor Vacuum Quantum - magnetic field interactions in vacuum

**Vedic Sutras**: 29 mathematical algorithms (16 main + 13 sub) from ancient Vedic mathematics

**RationalComplex**: Exact complex number representation using `Fraction` for real and imaginary parts (ℚ[i])

**PCFE**: Proto-Consciousness Field Engine - distributed HPC simulation framework

**Cymatic Patterns**: Nodal patterns formed by standing wave resonances (Chladni figures)

**Boundedness Invariant**: Constraint that |Ψ(x)| must remain below maximum threshold

**Trace Replay**: Ability to reconstruct exact evolution from recorded snapshots

**R4 Coupling**: Coupling between field and R4 (4-dimensional) topological structure

**Toroidal Wrapping**: Periodic boundary conditions where indices wrap around (no edges)

---

## Contact & Contribution

**Primary Documentation**: This file (`CLAUDE.md`)

**Technical Documentation**: `pcfe-v3/docs/README.md`

**Repository**: `/home/user/quanqonscious`

**Branch Naming**: Always use `claude/descriptive-name-sessionID` format

**Testing**: All changes MUST pass `tests/test_invariants.py`

**Style Guide**:
- Flake8 compliant (max line length: 120)
- Black formatting recommended
- Type hints preferred
- Docstrings required for public APIs

---

## Quick Checklist for AI Assistants

Before making changes, verify:

- [ ] Read relevant documentation (this file + module docstrings)
- [ ] Understand if working with `/core` (exact) or `/pcfe-v3` (distributed)
- [ ] Check arithmetic mode (EXACT vs FLOAT vs MIXED)
- [ ] Verify all 29 sutras accounted for (if sutra-related change)
- [ ] Optional dependencies handled gracefully
- [ ] Toroidal wrapping used for spatial indices
- [ ] Trace logging added for operators
- [ ] Invariants validated after operations
- [ ] Tests pass: `python tests/test_invariants.py`
- [ ] Commit message descriptive and references CODEX if applicable
- [ ] Branch name follows `claude/*-sessionID` format

---

**Document Version**: 1.0
**Last Updated**: 2025-12-26
**Target Audience**: AI Assistants (Claude, GPT-4, etc.)
**Maintenance**: Update when major architectural changes occur
