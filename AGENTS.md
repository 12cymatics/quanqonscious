# AGENT.md — Codex Implementation Playbook (GRVQ · MSTVQ · 29‑Sutra Stack)

**Purpose.** Exact, automatable instructions for Codex to implement the three requested synergies across this repo:

1. **Expose primitives once** via a free‑function API layer; 2) **Extend optimiser moves** to include Λ, Ω, Q_d; 3) **Unify metrics** so every call (primitive or composite) lands in one performance dataframe.

**Authoritative modules.**

* `primarysutra.py`, `primarysutraaws2.py` — Vedic primitives & sub‑sutras (class methods).
* `sutraws qubic.py` — composite/hyper‑cube operators: **Λ**, **Ω**, **Q_d**.
* `intersutraws.py` — optimiser / sequence runner.
* `sulbasutraws.py`, `mayasutraaws.py`, `utilitysutraws3.py`, `grvqsutraws.py` — geometry, Māyā transforms, tensor lifts, Φ³/GRVQ kernels.

---

## 0) Mathematical contracts (must hold after implementation)

We rely on the following properties; CI tests enforce them.

**(C1) Wrapper equivalence.** Let $S_k$ be any primitive sutra. The free‑function wrapper $F_k$ must satisfy $F_k(x)=S_k(x)\;\forall x$. (No state skew, identical side‑effects on logger.)

**(C2) Composite closure.** Λ, Ω, Q_d are built from the same generators {S₅,S₆,S₉,S₁₀, S₁₁, …}. Sums and Kronecker lifts keep us inside the finitely generated ring; optimiser convergence assumptions remain valid.

**(C3) Optional nilpotent Ω.** When coefficients are chosen so that **Ω² = 0** (nilpotent index 2), then $e^{\Omega}=I+\Omega$. This yields an exact **finite‑depth** implementation (depth 1 for Ω). We ship a check & a construction helper to encourage (but not require) this variant.

> **Note:** The commutator $[\Omega,\Omega]=0$ is tautologically true for any Ω and does **not** imply truncation. Finite‑depth comes from **nilpotency** (Ω²=0), not from the self‑commutator.

---

## 1) Expose primitives once (create `sutra_primitives.py`)

### 1.1 Add the free‑function API layer

Create `sutra_primitives.py` at repo root with the exact content below (autogenerates wrappers for **all** public callables of `VedicSutras`, preserving signatures & docstrings, and reusing the central logger/perf store):

```python
"""
Vedic Sutra Primitives – Free-Function API Layer
"""
from __future__ import annotations
import inspect as _inspect
import functools as _functools
from typing import Callable as _Callable
from primarysutra import VedicSutras
_VS: VedicSutras = VedicSutras()
__all__: list[str] = []

def _make_wrapper(method_name: str, target: _Callable) -> _Callable:
    @_functools.wraps(target)
    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)
    wrapper.__sutra_primitive__ = True  # for registry checks
    return wrapper

for _name, _method in _inspect.getmembers(_VS, predicate=callable):
    if _name.startswith("_"):  # skip private
        continue
    globals()[_name] = _make_wrapper(_name, _method)
    __all__.append(_name)

for _sym in ("_VS", "_inspect", "_functools", "_make_wrapper", "_name", "_method"):
    globals().pop(_sym, None)
```

### 1.2 Refactor `primarysutra*.py` to forward to wrappers

* Replace any duplicate method bodies that re‑implement primitives with thin forwards **or** leave class methods as‑is but ensure **no code is duplicated elsewhere**.
* Wherever composite code redefines primitives, delete those bodies and `from sutra_primitives import <names>`.

**Acceptance test:** `pytest -k test_wrapper_equivalence` (see §4).

---

## 2) Extend optimiser move‑set (Λ, Ω, Q_d)

### 2.1 Import composites and primitives centrally

In `intersutraws.py` top‑level imports:

```python
from sutraws_qubic import Lambda as Lambda_operator, Omega as Omega_operator, Q_d
from sutra_primitives import *  # unified primitive namespace
```

### 2.2 Build a registry with default params

Add near optimiser init:

```python
SUTRA_REGISTRY = {
    # primitives (automatically exported; include a curated subset if needed)
    # 'ekadhikena_purvena': ekadhikena_purvena, ...
}

COMPOSITE_REGISTRY = {
    'Lambda_operator': (Lambda_operator, { 'alpha': [0.25, 0.25, 0.25, 0.25] }),
    'Omega_operator':  (Omega_operator,  { 'chi': 1.0, 'alpha': [0.25]*4, 'lambda0': 1.0 }),
    'Q_d':             (Q_d,             { 'chi': 1.0, 'd': 4 }),
}

AVAILABLE_MOVES = {**SUTRA_REGISTRY, **COMPOSITE_REGISTRY}
```

### 2.3 Allow the optimiser to mutate composite params

* When sampling a move, if value is `(callable, params_dict)`, deep‑copy and mutate within bounds (e.g., Dirichlet jitter for `alpha`, integer jitter for `d ∈ {2..8}`, bounded noise for `chi`, `lambda0`).

**Acceptance test:** `pytest -k test_registry_and_sampling` (see §4).

---

## 3) Unify metrics & logging

### 3.1 Single logger + dataframe

Ensure **every** move (primitive or composite) records to the same structure managed by `primarysutra.VedicSutras`.

* In the optimiser constructor (`InterSutraWS.__init__` or equivalent):

```python
from primarysutra import VedicSutras
self._vs = VedicSutras()               # or receive an injected instance
self.logger = self._vs.logger          # reuse central logger
self.performance_history = self._vs.performance_history
```

* In the sequence evaluation path, **do not** build an independent dataframe. Call the same perf‑recording hook the class uses (e.g., `_record_performance(...)`). If that hook is private, add a public delegator on `VedicSutras` to record external runs.

### 3.2 Minimal schema (must contain)

`timestamp, move_name, params_hash, device, t_wall_s, cpu_pct, gpu_mem_MB, error_value, iter_idx, seq_id`.

**Acceptance test:** `pytest -k test_unified_metrics_schema` (see §4).

---

## 4) Tests (CI‑enforced)

Create `tests/` with the following files.

### 4.1 `test_wrapper_equivalence.py`

```python
import numpy as np
from primarysutra import VedicSutras
import sutra_primitives as sp

def test_all_public_callables_equivalent():
    vs = VedicSutras()
    # probe a stable input set; extend as needed
    xs = [0.0, 1.0, -1.5, np.pi]
    for name in dir(sp):
        if name.startswith('_'): continue
        f = getattr(sp, name)
        if not getattr(f, '__sutra_primitive__', False):
            continue
        m = getattr(vs, name)
        for x in xs:
            y1 = f(x)
            y2 = m(x)
            assert np.allclose(y1, y2, rtol=1e-12, atol=1e-12), name
```

### 4.2 `test_registry_and_sampling.py`

```python
from intersutraws import AVAILABLE_MOVES

def test_composites_present():
    assert 'Lambda_operator' in AVAILABLE_MOVES
    assert 'Omega_operator' in AVAILABLE_MOVES
    assert 'Q_d' in AVAILABLE_MOVES
```

### 4.3 `test_unified_metrics_schema.py`

```python
from primarysutra import VedicSutras

def test_perf_schema_keys():
    vs = VedicSutras()
    entry = {
        'timestamp': 0, 'move_name': 'dummy', 'params_hash': '0',
        'device': 'cpu', 't_wall_s': 0.0, 'cpu_pct': 0.0, 'gpu_mem_MB': 0.0,
        'error_value': 0.0, 'iter_idx': 0, 'seq_id': 'test'
    }
    # assume VedicSutras exposes `validate_perf_entry`
    assert vs.validate_perf_entry(entry)
```

### 4.4 `test_nilpotent_truncation.py` (optional, guarded)

```python
import numpy as np
from sutraws_qubic import Omega as Omega_operator

EPS = 1e-12

def is_nilpotent2(M):
    return np.linalg.norm(M @ M) < 1e-9

def test_exp_truncates_when_nilpotent():
    # Choose parameters known (or tuned) to yield a strictly upper‑triangular Ω
    Om = Omega_operator(chi=1.0, alpha=[0.25]*4, lambda0=0.0, enforce_upper=True)
    assert is_nilpotent2(Om), 'Ω must be nilpotent of index 2 for this test'
    I = np.eye(Om.shape[0])
    exp_series = I + Om  # exact when Ω²=0
    # numerical exp via 20‑term series (safe for small Ω)
    term = I.copy(); exp_num = I.copy()
    for k in range(1, 20):
        term = term @ (Om / k)
        exp_num = exp_num + term
    assert np.allclose(exp_series, exp_num, atol=1e-10)
```

> If `Omega_operator` currently returns a callable instead of a matrix, adapt by evaluating on the standard basis or enabling a `matrix=True` flag.

---

## 5) Implementation details for composites (reference)

Let the four “main” sutras used in the composites be $S_5, S_6, S_9, S_{10}$. Define:

**Λ (convex alloy).** $\displaystyle \Lambda(\boldsymbol{\alpha}) = \sum_{j\in\{5,6,9,10\}} \alpha_j\, S_j$, with $\alpha_j\ge 0$, $\sum\alpha_j=1$.

**Q_d (hypercube lift).** A Kronecker lift of selected sub‑sutras: $\displaystyle Q_d(\chi) = \bigotimes_{\ell=1}^{d} S_{sub,\ell}(\chi)$. Provide `d∈{2..8}` initially.

**Ω (stacked move).** $\displaystyle \Omega(\chi;\boldsymbol{\alpha},\lambda_0) = S_{11}(\chi) + \Lambda(\boldsymbol{\alpha}) + \lambda_0\, P_d(\chi)$. Here $P_d$ is the projector used by the optimiser’s fabric; when `enforce_upper=True`, project to strictly upper‑triangular part to obtain $\Omega^2=0$.

**Exponential truncation (if Ω²=0).** $e^{\Omega} = I + \Omega$. This underpins the “finite‑depth” claim.

---

## 6) Developer workflow & CI

1. **Install.** `pip install -e .[dev]` (ensure `pytest`, `ruff`, `black`, `mypy` in extras).
2. **Format & lint.** `ruff check . && black . && mypy .` (mypy in `--strict`).
3. **Tests.** `pytest -q` (CI must run §4 tests at minimum).
4. **Bench script (optional).** Add `scripts/bench_moves.py` to print a 2D scatter of time vs. error for primitive vs composite moves using the unified dataframe.

**Commit policy.** Conventional commits; one PR per task. Example subjects:

* `feat(core): add sutra_primitives free‑function API`
* `feat(opt): register Λ/Ω/Q_d and param mutations`
* `feat(metrics): unify performance logger & schema`
* `test(core,opt,metrics): add wrapper, registry, schema, nilpotent tests`

**Rollback.** Revert commits individually; no shared migrations.

---

## 7) Exact checklist for Codex

*

**Done when:** CI green; benchmark shows composite moves present and logging unified; wrapper equivalence holds across a probe set.
