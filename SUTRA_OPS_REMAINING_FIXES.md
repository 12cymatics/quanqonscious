# Remaining Fixes for core/operators/sutra_ops.py

**File**: `core/operators/sutra_ops.py`
**Size**: 1268 lines
**Violations**: 20+ instances of forbidden math functions

## Status

✅ **COMPLETED**: Import statement fixed - `import math` removed
❌ **REMAINING**: 20+ violations in the sutra operator implementations themselves

## Violations Found

### Math Module Calls (9 instances):
1. Line 513: `math.sqrt(target_ratio / ratio)` - In ratio adjustment
2. Line 547: `math.pi` - Phase quantization
3. Line 553-554: `math.cos()`, `math.sin()` - Phase mixing
4. Line 843: `math.cos(7 * phase)` - Kevala modulation
5. Line 946-947: `math.pi` - Phase complement
6. Line 953-954: `math.cos()`, `math.sin()` - Phase reconstruction
7. Line 1079: `math.sqrt()` - Phase standard deviation

### .norm() Calls (15+ instances):
- Line 146, 148: Ekadhikena operator
- Line 298, 300: Anurupyena operator
- Line 380, 382, 385, 386: Completion operator
- Line 702, 704: Factor operator
- Line 741, 742: Ratio comparisons
- Line 810, 813: Product normalization
- Line 874: Difference calculation
- Line 911: Excess calculation

### .phase() Calls (10+ instances):
- Line 539-540: Shesanya operator
- Line 839: Kevala operator
- Line 939: Phase complement
- Line 1072: Regularity operator

## Required Fixes

### Pattern 1: Replace `.norm()` with `.norm_squared()`

❌ **WRONG**:
```python
if value.norm() > 0.01:
    ratio = value.norm() / neighbor_val.norm()
```

✅ **CORRECT**:
```python
threshold_sq = Fraction(1, 10000)  # 0.01²
if value.norm_squared() > threshold_sq:
    # Use squared ratio or avoid division entirely
    ratio_sq = value.norm_squared() / neighbor_val.norm_squared()
```

### Pattern 2: Remove `.phase()` calls entirely

❌ **WRONG**:
```python
phase = value.phase()
quantized_phase = round(phase * n_levels / (2 * math.pi))
return RationalComplex.from_complex(
    complex(norm * math.cos(new_phase), norm * math.sin(new_phase))
)
```

✅ **CORRECT**:
```python
# Phase operations FORBIDDEN in exact arithmetic
# Use real/imag parts directly or raise NotImplementedError
raise NotImplementedError(
    "Phase operations forbidden - must use Vedic sutra functions only"
)
```

### Pattern 3: Replace `math.sqrt()` with variance

❌ **WRONG**:
```python
adjustment = math.sqrt(target_ratio / ratio)
phase_std = math.sqrt(sum((p - mean_phase)**2 for p in phases) / len(phases))
```

✅ **CORRECT**:
```python
# Use squared adjustment or variance (no sqrt)
adjustment_sq = target_ratio / ratio
phase_var = sum((p - mean_phase)**2 for p in phases) / len(phases)
```

### Pattern 4: Replace `math.pi` with rational approximation

❌ **WRONG**:
```python
quantized_phase = round(phase * n_levels / (2 * math.pi))
target = math.pi
```

✅ **CORRECT**:
```python
pi_approx = Fraction(22, 7)  # Rational approximation of π
quantized_phase = round(phase * n_levels / (2 * pi_approx))
target = pi_approx
```

### Pattern 5: Remove trigonometric functions

❌ **WRONG**:
```python
modulation = math.cos(7 * phase) + 1
rotation = RationalComplex.from_complex(
    complex(math.cos(phase_shift), math.sin(phase_shift))
)
```

✅ **CORRECT**:
```python
# Trigonometric functions FORBIDDEN - violate exact arithmetic
# Either:
# 1. Use polynomial approximations
# 2. Use discrete phase sectors
# 3. Raise NotImplementedError until sutra-based implementation ready
raise NotImplementedError(
    "Trigonometric functions forbidden - must use Vedic sutra functions only"
)
```

## Implementation Strategy

### Option 1: Quick Fix (Mark violations as forbidden)

Add `raise NotImplementedError(...)` for all phase-based operations:

```python
# In each sutra operator that uses phase:
if uses_phase_operations:
    raise NotImplementedError(
        f"{self.__class__.__name__} uses forbidden phase operations. "
        "Must be reimplemented using ONLY Vedic sutra rational functions."
    )
```

### Option 2: Proper Fix (Reimpl with sutras)

Reimplement each operator using ONLY:
- `value.real` and `value.imag` (direct access)
- `value.norm_squared()` (exact Fraction)
- `value.conjugate()` (exact)
- Vedic sutra functions from `primarysutra.py`
- Rational arithmetic only

## Files to Fix

1. **SheshanyankenaCharamenaOperator** (Line 535-555)
   - Uses `.norm()` and `.phase()`
   - Uses `math.pi`, `math.cos()`, `math.sin()`

2. **KevalahSaptakamGunvatOperator** (Line 837-851)
   - Uses `.phase()`
   - Uses `math.cos()`

3. **YavadunamTavadunamOperator** (Line 937-956)
   - Uses `.norm()` and `.phase()`
   - Uses `math.pi`, `math.cos()`, `math.sin()`

4. **VilokanamOperator** (Line 1067-1092)
   - Uses `.phase()`
   - Uses `math.sqrt()`

5. **Multiple operators** using `.norm()`:
   - EkadhikenaOperator
   - AnurupyenaOperator
   - PurnaApurnaOperator
   - GunitasamuccayahOperator
   - AnurupyenaSubOperator
   - AdyamadyenaOperator
   - VestanaOperator
   - SamuccayagunitahOperator

## Testing After Fixes

After all fixes, verify:

```bash
# No math module usage (except in comments)
grep "math\." core/operators/sutra_ops.py | grep -v "^#" | grep -v "# OLD"

# No .norm() calls (except .norm_squared())
grep "\.norm()" core/operators/sutra_ops.py | grep -v "norm_squared"

# No .phase() calls
grep "\.phase()" core/operators/sutra_ops.py

# All tests pass
python tests/test_invariants.py
```

## Estimated Effort

- Quick fix (NotImplementedError): ~30 minutes
- Proper fix (sutra-based): ~4-6 hours (requires mathematical redesign)

## Priority

**HIGH** - These are the 29 Vedic sutra operators themselves violating the exact arithmetic requirement. This is a fundamental architectural issue that must be fixed.

Until fixed, the entire CODEX simulation framework cannot run properly because the sutra operators that are supposed to enforce exact arithmetic are themselves using forbidden float operations.
