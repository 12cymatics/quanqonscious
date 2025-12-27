# Vedic Sutra Compliance Fixes

**Date**: 2025-12-27
**Critical Issue**: Core simulation code violates exact arithmetic and Vedic sutra principles

## Problem Summary

The `/core` directory simulation code contains **25+ violations** of the fundamental rule:

**ALL mathematical operations must use ONLY the 29 Vedic sutra functions - NO standard math library functions**

### Violations Found

1. **math.sqrt()** - 8 instances
2. **math.sin()** - 6 instances
3. **math.cos()** - 11 instances
4. **math.atan2()** - 3 instances
5. **math.exp()** - 1 instance
6. **RationalComplex.norm()** - 15+ instances (uses sqrt internally)
7. **RationalComplex.phase()** - 10+ instances (uses atan2 internally)

These violate the exact rational arithmetic (ℚ[i]) requirement and do not use Vedic sutras.

## Files Fixed ✅

### 1. core/state.py
- ❌ Removed `import math`
- ✅ Made `.norm()` raise NotImplementedError
- ✅ Made `.phase()` raise NotImplementedError
- ✅ Replaced all `.norm()` calls with `.norm_squared()` (exact Fraction)
- ✅ All methods now return Fraction instead of float where applicable

**Changes**:
- `amplitude()` now returns |Ψ|² (Fraction) instead of |Ψ| (float)
- `max_amplitude()` now returns max |Ψ|² (Fraction)
- `mean_amplitude()` now returns mean |Ψ|² (Fraction)
- `compute_phase_field()` raises NotImplementedError

### 2. core/operators/grvq_ansatz.py
- ❌ Removed `import math`
- ✅ Added import for VedicSutras from primarysutra
- ✅ All shape functions raise NotImplementedError with TODOs:
  - `_chladni()` - Need polynomial/sutra-based pattern
  - `_bessel()` - Need rational Bessel approximations
  - `_harmonic()` - Need discrete lattice Fourier modes
  - `_radial()` - Need r² based patterns (no sqrt)
- ✅ VedicCarrier.evaluate() raises NotImplementedError

**TODO in docs**: Each function documents which sutras should be used for reimplementation.

## Files Needing Fixes ❌

### 3. core/operators/sutra_ops.py (IN PROGRESS)
**7 math function calls + 20+ .norm()/.phase() calls**

Critical violations in Vedic sutra operators themselves:
- Line 510: `math.sqrt()` in ratio adjustment
- Line 544: `math.pi` for phase quantization
- Line 550-551: `math.cos()`, `math.sin()` for phase mixing
- Line 840: `math.cos(7 * phase)` in kevala operator
- Line 943-944: `math.pi` for phase complement
- Line 950-951: `math.cos()`, `math.sin()` for phase reconstruction
- Line 1076: `math.sqrt()` for phase standard deviation

Plus extensive use of:
- `.norm()` for amplitude comparisons (should use `.norm_squared()`)
- `.phase()` for angular operations (FORBIDDEN - no atan2)

**Required Changes**:
1. Replace all `.norm()` with `.norm_squared()` and adjust thresholds
   - `if value.norm() > 0.001` → `if value.norm_squared() > Fraction(1, 1000000)`
2. Remove ALL phase operations - sutras must work without angles
3. Replace `math.sqrt(ratio)` with ratio squared comparisons
4. Remove trigonometric phase mixing - use rational operations only

### 4. core/operators/r4_coupling.py
**1 math.exp() call**

- Line 116: `math.exp(-diff.norm() / 0.5)` for exponential decay

**Fix**: Replace exponential decay with rational polynomial approximation or series using sutras.

### 5. core/operators/mstvq.py
**1 math function call**

**Fix**: Similar pattern to r4_coupling.

### 6. core/observables.py
**Multiple .norm() and .phase() calls**

**Fix**: Replace with .norm_squared() exact comparisons.

### 7. core/hybrid_pipeline.py
**1 math function call**

**Fix**: Remove math dependency.

## Correct Implementation Pattern

### ❌ WRONG (Forbidden):
```python
# Uses sqrt (float approximation)
amplitude = value.norm()
if amplitude > 0.01:
    normalized = value / amplitude

# Uses trigonometry
phase = value.phase()
new_value = RationalComplex.from_complex(
    complex(norm * math.cos(phase), norm * math.sin(phase))
)
```

### ✅ CORRECT (Exact Rational):
```python
# Uses norm_squared (exact Fraction)
norm_sq = value.norm_squared()
threshold_sq = Fraction(1, 10000)  # 0.01²
if norm_sq > threshold_sq:
    # NO normalization - violates exact arithmetic
    # Instead use ratio operations
    ...

# NO phase operations allowed
# Use real/imag parts directly with Vedic sutras
from primarysutra import VedicSutras
sutras = VedicSutras()

# Example: use nikhilam (complement) instead of phase
complement = sutras.nikhilam_navatashcaramam_dashatah(
    value.real, base=Fraction(10)
)
```

## The 29 Vedic Sutras (Reference)

### 16 Main Sutras:
1. **ekadhikena_purvena** - "By one more than the previous"
2. **nikhilam_navatashcaramam_dashatah** - "All from 9, last from 10"
3. **urdhva_tiryagbhyam** - "Vertically and crosswise"
4. **paravartya_yojayet** - "Transpose and apply"
5. **shunyam_samyasamuccaye** - "When sum is same, sum is zero"
6. **anurupye_shunyamanyat** - "If one is in ratio, other is zero"
7. **sankalana_vyavakalanabhyam** - "By addition and by subtraction"
8. **puranapuranabhyam** - "By completion or non-completion"
9. **chalana_kalanabyham** - "Differences and similarities"
10. **yaavadunam** - "Whatever the deficiency"
11. **vyashtisamanstih** - "Part and whole"
12. **shesanyankena_charamena** - "Remainders by last digit"
13. **sopaantyadvayamantyam** - "Ultimate and twice penultimate"
14. **ekanyunena_purvena** - "By one less than the previous"
15. **gunitasamuccayah** - "Product of sum"
16. **gunakasamuccayah** - "Factors of sum"

### 13 Sub-Sutras:
1. **anurupyena** - "Proportionately"
2. **shishyate_sheshasamjnah** - "Remainder remains constant"
3. **adyamadyenantyamantyena** - "First by first, last by last"
4. **kevalaih_saptakam_gunyat** - "For 7, multiplication is done"
5. **vestanam** - "By osculation"
6. **yavadunam_tavadunikritya** - "Lessen by deficiency"
7. **yavadunam_tavadunam** - "Whatever deficiency, lessen"
8. **antyayordasakepi** - "Penultimate is 10"
9. **antyayoreva** - "Only last terms"
10. **samuccayagunitah** - "Whole product"
11. **lopanasthapanabhyam** - "Alternate elimination/retention"
12. **vilokanam** - "By observation"
13. **gunitasamuccayah_samuccayagunitah** - "Product of sums"

## Testing Requirements

After fixes are complete, ALL tests in `tests/test_invariants.py` must pass:

```bash
cd /home/user/quanqonscious/tests
python test_invariants.py
```

Expected: 7/7 tests passing with ZERO math module usage.

## Next Steps

1. ✅ Complete core/state.py fixes
2. ✅ Complete core/operators/grvq_ansatz.py fixes
3. ❌ Complete core/operators/sutra_ops.py fixes (20+ violations)
4. ❌ Fix core/operators/r4_coupling.py (1 violation)
5. ❌ Fix core/operators/mstvq.py (1 violation)
6. ❌ Fix core/observables.py (multiple violations)
7. ❌ Fix core/hybrid_pipeline.py (1 violation)
8. ❌ Run full test suite
9. ❌ Commit all changes

## Notes

- **NO exceptions** - exact arithmetic is NON-NEGOTIABLE
- **NO approximations** - limit_denominator() should be avoided where possible
- **NO normalization** - division by norms introduces floats
- **NO phase operations** - atan2 is transcendental and forbidden
- **ONLY Vedic sutras** - all operations must use the 29 sutra functions

The entire framework must operate in ℚ[i] (rational complex numbers) using ONLY the 29 Vedic mathematical sutras.
