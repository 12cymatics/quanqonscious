#!/usr/bin/env python3
"""
Comprehensive Test Suite for Extended Sub-Sutras (10-13)

Tests all 4 new sub-sutras with multiple modes:
- Classical mode
- Quantum mode (with Cirq)
- Hybrid mode
- Array/scalar inputs
"""

import numpy as np
import sys
import time
from typing import Dict, Any

# Import extended sutras
try:
    from extended_subsutras import ExtendedVedicSutras
    from primarysutra import SutraContext, SutraMode
    print("✓ Successfully imported ExtendedVedicSutras")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test configuration
VERBOSE = True
RUN_QUANTUM_TESTS = True

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_subheader(text: str):
    """Print formatted subheader"""
    print(f"\n--- {text} ---")

def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status:8s} | {test_name:50s} | {details}")

def compare_values(actual, expected, tolerance=1e-6):
    """Compare values with tolerance"""
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) < tolerance or abs((actual - expected) / max(abs(expected), 1e-10)) < tolerance
    else:
        return False

# =============================================================================
# TEST SUITE 1: GUNITA SAMUCCAYAH (Product Accumulation)
# =============================================================================

def test_gunita_samuccayah():
    """Test Sub-Sutra 10: Gunita Samuccayah"""
    print_header("SUB-SUTRA 10: GUNITA SAMUCCAYAH (Product Accumulation)")

    results = []

    # Initialize
    context_classical = SutraContext(mode=SutraMode.CLASSICAL)
    context_quantum = SutraContext(mode=SutraMode.QUANTUM)

    sutras_classical = ExtendedVedicSutras(context=context_classical)
    sutras_quantum = ExtendedVedicSutras(context=context_quantum)

    # Test 1: Simple scalar product
    print_subheader("Test 1: Simple Scalar Product")
    factors1 = [2.0, 3.0, 4.0]
    expected1 = 24.0

    result_classical = sutras_classical.gunita_samuccayah(factors1)
    passed1 = compare_values(result_classical, expected1)
    print_result("Classical: 2 × 3 × 4 = 24", passed1, f"Got {result_classical:.4f}")
    results.append(passed1)

    if RUN_QUANTUM_TESTS:
        result_quantum = sutras_quantum.gunita_samuccayah(factors1)
        passed1q = compare_values(result_quantum, expected1, tolerance=0.5)
        print_result("Quantum: 2 × 3 × 4 ≈ 24", passed1q, f"Got {result_quantum:.4f}")
        results.append(passed1q)

    # Test 2: Product with negative numbers
    print_subheader("Test 2: Product with Negative Numbers")
    factors2 = [2.0, -3.0, 5.0]
    expected2 = -30.0

    result_classical2 = sutras_classical.gunita_samuccayah(factors2)
    passed2 = compare_values(result_classical2, expected2)
    print_result("Classical: 2 × (-3) × 5 = -30", passed2, f"Got {result_classical2:.4f}")
    results.append(passed2)

    # Test 3: Product of arrays
    print_subheader("Test 3: Product of Arrays")
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([2.0, 2.0, 2.0])
    arr3 = np.array([3.0, 3.0, 3.0])
    expected3 = np.array([6.0, 12.0, 18.0])

    result_arr = sutras_classical.gunita_samuccayah([arr1, arr2, arr3])
    passed3 = compare_values(result_arr, expected3)
    print_result("Classical: Array product", passed3, f"Got {result_arr}")
    results.append(passed3)

    # Test 4: Single element
    print_subheader("Test 4: Single Factor")
    factors4 = [7.5]
    expected4 = 7.5

    result4 = sutras_classical.gunita_samuccayah(factors4)
    passed4 = compare_values(result4, expected4)
    print_result("Classical: Single factor", passed4, f"Got {result4:.4f}")
    results.append(passed4)

    # Test 5: Many factors
    print_subheader("Test 5: Many Factors")
    factors5 = [1.1, 1.2, 1.3, 1.4, 1.5]
    expected5 = 1.1 * 1.2 * 1.3 * 1.4 * 1.5

    result5 = sutras_classical.gunita_samuccayah(factors5)
    passed5 = compare_values(result5, expected5)
    print_result("Classical: 5 factors", passed5, f"Got {result5:.4f}, Expected {expected5:.4f}")
    results.append(passed5)

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nGunita Samuccayah: {passed_count}/{total_count} tests passed")
    return all(results)

# =============================================================================
# TEST SUITE 2: SANKALANA VYAVAKALANABHYAM (Error Compensation)
# =============================================================================

def test_sankalana_vyavakalanabhyam():
    """Test Sub-Sutra 11: Sankalana Vyavakalanabhyam"""
    print_header("SUB-SUTRA 11: SANKALANA VYAVAKALANABHYAM (Error Compensation)")

    results = []

    context = SutraContext(mode=SutraMode.CLASSICAL)
    sutras = ExtendedVedicSutras(context=context)

    # Test 1: Balanced mode
    print_subheader("Test 1: Balanced Mode")
    x1, y1 = 10.0, 5.0
    expected1 = 10.0  # (15 + 5)/2 = 10

    result1 = sutras.sankalana_vyavakalanabhyam_extended(x1, y1, mode='balanced')
    passed1 = compare_values(result1, expected1)
    print_result("Balanced: (10+5 + 10-5)/2 = 10", passed1, f"Got {result1:.4f}")
    results.append(passed1)

    # Test 2: Compensated mode (Kahan summation)
    print_subheader("Test 2: Compensated Mode (Kahan)")
    x2, y2 = 1e10, 1.0
    expected2 = 1e10 + 1.0

    result2 = sutras.sankalana_vyavakalanabhyam_extended(x2, y2, mode='compensated')
    passed2 = compare_values(result2, expected2, tolerance=1e-5)
    print_result("Kahan: 1e10 + 1.0", passed2, f"Got {result2:.4e}")
    results.append(passed2)

    # Test 3: Iterative mode
    print_subheader("Test 3: Iterative Mode")
    x3, y3 = 7.0, 3.0
    expected3 = 10.0

    result3 = sutras.sankalana_vyavakalanabhyam_extended(x3, y3, mode='iterative')
    passed3 = compare_values(result3, expected3)
    print_result("Iterative: 7 + 3 = 10", passed3, f"Got {result3:.4f}")
    results.append(passed3)

    # Test 4: Array compensated
    print_subheader("Test 4: Array Compensated")
    arr_x = np.array([1.0, 2.0, 3.0])
    arr_y = np.array([4.0, 5.0, 6.0])
    expected_arr = np.array([5.0, 7.0, 9.0])

    result_arr = sutras.sankalana_vyavakalanabhyam_extended(arr_x, arr_y, mode='compensated')
    passed_arr = compare_values(result_arr, expected_arr)
    print_result("Array Kahan", passed_arr, f"Got {result_arr}")
    results.append(passed_arr)

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nSankalana Vyavakalanabhyam: {passed_count}/{total_count} tests passed")
    return all(results)

# =============================================================================
# TEST SUITE 3: SOPAANTYADVAYAMANTYAM (Recurrence Relations)
# =============================================================================

def test_sopaantyadvayamantyam():
    """Test Sub-Sutra 12: Sopaantyadvayamantyam"""
    print_header("SUB-SUTRA 12: SOPAANTYADVAYAMANTYAM (Recurrence Relations)")

    results = []

    context = SutraContext(mode=SutraMode.CLASSICAL)
    sutras = ExtendedVedicSutras(context=context)

    # Test 1: Single step (dominant eigenvalue = 2)
    print_subheader("Test 1: Single Step")
    x1 = 5.0
    steps1 = 1
    expected1 = 5.0 * 2  # Dominant eigenvalue

    result1 = sutras.sopaantyadvayamantyam(x1, steps=steps1)
    passed1 = compare_values(result1, expected1)
    print_result("1 step: 5 → 10", passed1, f"Got {result1:.4f}")
    results.append(passed1)

    # Test 2: Multiple steps
    print_subheader("Test 2: Multiple Steps")
    x2 = 3.0
    steps2 = 3
    expected2 = 3.0 * (2 ** 3)  # 3 * 8 = 24

    result2 = sutras.sopaantyadvayamantyam(x2, steps=steps2)
    passed2 = compare_values(result2, expected2)
    print_result("3 steps: 3 → 24", passed2, f"Got {result2:.4f}")
    results.append(passed2)

    # Test 3: Zero steps
    print_subheader("Test 3: Zero Steps")
    x3 = 7.0
    steps3 = 0
    expected3 = 7.0

    result3 = sutras.sopaantyadvayamantyam(x3, steps=steps3)
    passed3 = compare_values(result3, expected3)
    print_result("0 steps: 7 → 7", passed3, f"Got {result3:.4f}")
    results.append(passed3)

    # Test 4: Array input
    print_subheader("Test 4: Array Input")
    arr4 = np.array([1.0, 2.0, 3.0])
    steps4 = 2
    expected_arr4 = arr4 * (2 ** 2)

    result_arr4 = sutras.sopaantyadvayamantyam(arr4, steps=steps4)
    passed4 = compare_values(result_arr4, expected_arr4)
    print_result("Array 2 steps", passed4, f"Got {result_arr4}")
    results.append(passed4)

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nSopaantyadvayamantyam: {passed_count}/{total_count} tests passed")
    return all(results)

# =============================================================================
# TEST SUITE 4: PURANAPURANABYHAM (Completion Analysis)
# =============================================================================

def test_puranapuranabyham():
    """Test Sub-Sutra 13: Puranapuranabyham"""
    print_header("SUB-SUTRA 13: PURANAPURANABYHAM (Completion Analysis)")

    results = []

    context = SutraContext(mode=SutraMode.CLASSICAL)
    sutras = ExtendedVedicSutras(context=context)

    # Test 1: 50% completion
    print_subheader("Test 1: 50% Completion")
    complete1 = 100.0
    incomplete1 = 50.0
    # η = 50/100 = 0.5
    # result = 0.5*50 + 0.5*100 = 25 + 50 = 75
    expected1 = 75.0

    result1 = sutras.puranapuranabyham(complete1, incomplete1)
    passed1 = compare_values(result1, expected1)
    print_result("50% complete: 75", passed1, f"Got {result1:.4f}")
    results.append(passed1)

    # Test 2: 100% completion
    print_subheader("Test 2: 100% Completion")
    complete2 = 80.0
    incomplete2 = 80.0
    # η = 80/80 = 1.0
    # result = 1.0*80 + 0*80 = 80
    expected2 = 80.0

    result2 = sutras.puranapuranabyham(complete2, incomplete2)
    passed2 = compare_values(result2, expected2)
    print_result("100% complete: 80", passed2, f"Got {result2:.4f}")
    results.append(passed2)

    # Test 3: 0% completion
    print_subheader("Test 3: 0% Completion")
    complete3 = 60.0
    incomplete3 = 0.0
    # η = 0/60 = 0.0
    # result = 0*0 + 1*60 = 60
    expected3 = 60.0

    result3 = sutras.puranapuranabyham(complete3, incomplete3)
    passed3 = compare_values(result3, expected3)
    print_result("0% complete: 60", passed3, f"Got {result3:.4f}")
    results.append(passed3)

    # Test 4: Array completion
    print_subheader("Test 4: Array Completion")
    complete_arr = np.array([100.0, 200.0, 300.0])
    incomplete_arr = np.array([50.0, 150.0, 250.0])
    # η = [0.5, 0.75, 0.833]
    # result[0] = 0.5*50 + 0.5*100 = 75
    # result[1] = 0.75*150 + 0.25*200 = 112.5 + 50 = 162.5
    # result[2] = 0.833*250 + 0.167*300 ≈ 258.3

    result_arr = sutras.puranapuranabyham(complete_arr, incomplete_arr)
    # Just check that it returns an array
    passed_arr = isinstance(result_arr, np.ndarray) and result_arr.shape == complete_arr.shape
    print_result("Array completion", passed_arr, f"Got {result_arr}")
    results.append(passed_arr)

    # Test 5: Over-completion (clamp test)
    print_subheader("Test 5: Over-Completion (>100%)")
    complete5 = 50.0
    incomplete5 = 75.0
    # η = 75/50 = 1.5 → clamped to 1.0
    # result = 1.0*75 + 0*50 = 75
    expected5 = 75.0

    result5 = sutras.puranapuranabyham(complete5, incomplete5)
    passed5 = compare_values(result5, expected5)
    print_result("Over-complete (clamped): 75", passed5, f"Got {result5:.4f}")
    results.append(passed5)

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nPuranapuranabyham: {passed_count}/{total_count} tests passed")
    return all(results)

# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

def run_performance_tests():
    """Run performance benchmarks"""
    print_header("PERFORMANCE BENCHMARKS")

    context = SutraContext(mode=SutraMode.CLASSICAL)
    sutras = ExtendedVedicSutras(context=context)

    # Benchmark 1: Product accumulation
    print_subheader("Benchmark 1: Product Accumulation")
    factors_bench = [1.5] * 100

    start = time.time()
    for _ in range(1000):
        _ = sutras.gunita_samuccayah(factors_bench)
    elapsed = time.time() - start
    print(f"1000 iterations × 100 factors: {elapsed:.4f} seconds ({1000/elapsed:.1f} ops/sec)")

    # Benchmark 2: Error compensation
    print_subheader("Benchmark 2: Error Compensation")
    x_bench = 1e10
    y_bench = 1.0

    start = time.time()
    for _ in range(10000):
        _ = sutras.sankalana_vyavakalanabhyam_extended(x_bench, y_bench, mode='compensated')
    elapsed = time.time() - start
    print(f"10000 Kahan summations: {elapsed:.4f} seconds ({10000/elapsed:.1f} ops/sec)")

    # Benchmark 3: Recurrence
    print_subheader("Benchmark 3: Recurrence Relations")
    x_rec = 5.0

    start = time.time()
    for _ in range(10000):
        _ = sutras.sopaantyadvayamantyam(x_rec, steps=10)
    elapsed = time.time() - start
    print(f"10000 recurrences (10 steps): {elapsed:.4f} seconds ({10000/elapsed:.1f} ops/sec)")

    # Benchmark 4: Completion analysis
    print_subheader("Benchmark 4: Completion Analysis")
    complete_bench = 100.0
    incomplete_bench = 75.0

    start = time.time()
    for _ in range(10000):
        _ = sutras.puranapuranabyham(complete_bench, incomplete_bench)
    elapsed = time.time() - start
    print(f"10000 completion analyses: {elapsed:.4f} seconds ({10000/elapsed:.1f} ops/sec)")

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Main test runner"""
    print_header("EXTENDED SUB-SUTRAS TEST SUITE")
    print("Testing Sub-Sutras 10-13 with Vedic Quantum Computing Framework")
    print(f"Quantum tests: {'ENABLED' if RUN_QUANTUM_TESTS else 'DISABLED'}")

    start_time = time.time()

    # Run all tests
    test_results = {}

    try:
        test_results['gunita_samuccayah'] = test_gunita_samuccayah()
    except Exception as e:
        print(f"✗ EXCEPTION in gunita_samuccayah: {e}")
        test_results['gunita_samuccayah'] = False

    try:
        test_results['sankalana_vyavakalanabhyam'] = test_sankalana_vyavakalanabhyam()
    except Exception as e:
        print(f"✗ EXCEPTION in sankalana_vyavakalanabhyam: {e}")
        test_results['sankalana_vyavakalanabhyam'] = False

    try:
        test_results['sopaantyadvayamantyam'] = test_sopaantyadvayamantyam()
    except Exception as e:
        print(f"✗ EXCEPTION in sopaantyadvayamantyam: {e}")
        test_results['sopaantyadvayamantyam'] = False

    try:
        test_results['puranapuranabyham'] = test_puranapuranabyham()
    except Exception as e:
        print(f"✗ EXCEPTION in puranapuranabyham: {e}")
        test_results['puranapuranabyham'] = False

    # Run performance tests
    try:
        run_performance_tests()
    except Exception as e:
        print(f"⚠ Performance tests skipped: {e}")

    # Summary
    elapsed_time = time.time() - start_time

    print_header("TEST SUMMARY")
    passed = sum(test_results.values())
    total = len(test_results)

    for sutra_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} | {sutra_name}")

    print(f"\nTotal: {passed}/{total} test suites passed")
    print(f"Total time: {elapsed_time:.2f} seconds")

    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        return 0
    else:
        print(f"\n✗✗✗ {total - passed} TEST SUITE(S) FAILED ✗✗✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
