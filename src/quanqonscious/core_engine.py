# QuanQonscious/core_engine.py

import math
import numpy as np
import concurrent.futures

# --- Main Sutra Functions (16 in-series sutras) ---
def sutra1_Ekadhikena(params: np.ndarray) -> np.ndarray:
    """Sutra 1 (Ekadhikena Purvena): 'By one more than the previous one' – adds a tiny increment to each element."""
    return np.array([p + 0.001 for p in params], dtype=float)

def sutra2_Nikhilam(params: np.ndarray) -> np.ndarray:
    """Sutra 2 (Nikhilam Navatashcaramam Dashatah): 'All from 9 and the last from 10' – complements elements relative to 1."""
    return np.array([1.0 - p if p <= 1 else 2.0 - p for p in params], dtype=float)

def sutra3_Urdhva_Tiryagbhyam(params: np.ndarray) -> np.ndarray:
    """Sutra 3 (Urdhva-Tiryakbhyam): 'Vertically and crosswise' – mixes each element with its neighbor (cross-coupling)."""
    return np.array([p + 0.0001 * (params[i-1] if i > 0 else 0) for i, p in enumerate(params)], dtype=float)

def sutra4_Urdhva_Veerya(params: np.ndarray) -> np.ndarray:
    """Sutra 4 (Extended Urdhva Variation): Applies a slight multiplicative boost to simulate crosswise strength."""
    return np.array([p * 1.001 for p in params], dtype=float)

def sutra5_Paravartya(params: np.ndarray) -> np.ndarray:
    """Sutra 5 (Paravartya Yojayet): 'Transpose and adjust' – applies a small reciprocal adjustment."""
    return np.array([p + 0.0005 * (1.0 / (p + 1e-6)) for p in params], dtype=float)

def sutra6_Shunyam_Samyasamuccaye(params: np.ndarray) -> np.ndarray:
    """Sutra 6 (Shunyam Saamyasamuccaye): 'If the total is zero' – zeros out very small values for numerical stability."""
    return np.array([0.0 if abs(p) < 1e-8 else p for p in params], dtype=float)

def sutra7_Anurupyena(params: np.ndarray) -> np.ndarray:
    """Sutra 7 ((Anurupye) Shunyamanyat): 'If one is in ratio, the other is zero' – subtracts a tiny proportion of the mean from each element."""
    mean_val = float(np.mean(params))
    return np.array([p - 0.0001 * mean_val for p in params], dtype=float)

def sutra8_Sopantyadvayamantyam(params: np.ndarray) -> np.ndarray:
    """Sutra 8 (Sopantyadvayamantyam): 'The ultimate and twice the penultimate' – averages the last two elements."""
    result = params.copy()
    if result.size >= 2:
        result[-1] = (result[-1] + result[-2]) / 2.0
    return result

def sutra9_Ekanyunena(params: np.ndarray) -> np.ndarray:
    """Sutra 9 (Ekanyunena Purvena): 'By one less than the previous one' – subtracts a small fraction of the previous element from each element."""
    out = []
    for i, p in enumerate(params):
        if i > 0:
            out.append(p - 0.0001 * params[i-1])
        else:
            out.append(p)
    return np.array(out, dtype=float)

def sutra10_Dvitiya(params: np.ndarray) -> np.ndarray:
    """Sutra 10 (Dvitīya): Applies a quadratic tweak to every other element to simulate second-order effects."""
    return np.array([p**2 if (i % 2 == 0) else p for i, p in enumerate(params)], dtype=float)

def sutra11_Virahanka(params: np.ndarray) -> np.ndarray:
    """Sutra 11 (Virahanka's method): Adds a sine-based perturbation to each element (simulating Virahanka Fibonacci relation)."""
    return np.array([p + 0.0015 * math.sin(2 * p) for p in params], dtype=float)

def sutra12_Ayadalagana(params: np.ndarray) -> np.ndarray:
    """Sutra 12 (Ayadalaguna or 'corollary'): Scales each element by a factor proportional to its absolute value (non-linear gain)."""
    return np.array([p * (1 + 0.0006 * abs(p)) for p in params], dtype=float)

def sutra13_Samuchchaya(params: np.ndarray) -> np.ndarray:
    """Sutra 13 (Samuchchayagunitah): 'The whole as one' – adds a small fraction of the total sum to each element."""
    total = float(np.sum(params))
    return np.array([p + 0.0002 * total for p in params], dtype=float)

def sutra14_Ankylana(params: np.ndarray) -> np.ndarray:
    """Sutra 14 (Gunakasamuchayah variation): Adds a sinusoidal ornamentation based on index to each element."""
    return np.array([p + 0.0005 * math.sin(i) for i, p in enumerate(params)], dtype=float)

def sutra15_Shallaka(params: np.ndarray) -> np.ndarray:
    """Sutra 15 (Shallaka method): Averages each element with its next neighbor (smoothing adjacent pairs)."""
    new_params = []
    for i in range(len(params) - 1):
        new_params.append((params[i] + params[i+1]) / 2.0)
    if len(params) > 0:
        new_params.append(params[-1])
    return np.array(new_params, dtype=float)

def sutra16_Samuca(params: np.ndarray) -> np.ndarray:
    """Sutra 16 (Sandhya Samuccaya): Weighted average adjustment – adds a small weighted average of all elements to each element."""
    n = len(params)
    if n == 0:
        return params
    indices = np.linspace(1, n, n)
    weighted_avg = float(np.dot(params, indices) / np.sum(indices))
    return np.array([p + 0.0003 * weighted_avg for p in params], dtype=float)


# List of all main sutra functions for easy iteration
_main_sutras = [
    sutra1_Ekadhikena, sutra2_Nikhilam, sutra3_Urdhva_Tiryagbhyam, sutra4_Urdhva_Veerya,
    sutra5_Paravartya, sutra6_Shunyam_Samyasamuccaye, sutra7_Anurupyena, sutra8_Sopantyadvayamantyam,
    sutra9_Ekanyunena, sutra10_Dvitiya, sutra11_Virahanka, sutra12_Ayadalagana,
    sutra13_Samuchchaya, sutra14_Ankylana, sutra15_Shallaka, sutra16_Samuca
]


# --- Sub-Sutra Functions (13 in-parallel sutras) ---
def subsutra1_Refinement(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 1: Refinement – small quadratic addition to refine each value."""
    return np.array([p + 0.0001 * (p ** 2) for p in params], dtype=float)

def subsutra2_Correction(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 2: Correction – subtracts a tiny bias from each element to correct towards 0.5."""
    return np.array([p - 0.0002 * (p - 0.5) for p in params], dtype=float)

def subsutra3_Recursion(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 3: Recursion – mixes each element with its previous value (cyclically) to introduce feedback."""
    if params.size == 0:
        return params
    shifted = np.roll(params, 1)  # cyclic shift
    return (params + shifted) / 2.0

def subsutra4_Convergence(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 4: Convergence – reduces differences between consecutive elements (damps oscillations)."""
    diffs = np.diff(params, append=params[-1] if params.size > 0 else 0.0)
    return np.array([p - 0.0001 * d for p, d in zip(params, diffs)], dtype=float)

def subsutra5_Stabilization(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 5: Stabilization – applies a slight damping factor to each element."""
    return np.array([p * 0.999 for p in params], dtype=float)

def subsutra6_Simplification(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 6: Simplification – rounds each element to a reasonable precision (4 decimal places)."""
    return np.array([round(float(p), 4) for p in params], dtype=float)

def subsutra7_Interpolation(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 7: Interpolation – adds a tiny constant to each element as a baseline lift."""
    return np.array([p + 0.00005 for p in params], dtype=float)

def subsutra8_Extrapolation(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 8: Extrapolation – projects a linear trend forward and applies a small correction based on it."""
    if params.size == 0:
        return params
    trend = np.polyfit(range(len(params)), params, 1)
    extrapolated = np.polyval(trend, len(params))
    return np.array([p + 0.0001 * float(extrapolated) for p in params], dtype=float)

def subsutra9_ErrorReduction(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 9: ErrorReduction – subtracts a small fraction of the standard deviation from each element to reduce spread."""
    if params.size == 0:
        return params
    std_dev = float(np.std(params))
    return np.array([p - 0.0001 * std_dev for p in params], dtype=float)

def subsutra10_Optimization(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 10: Optimization – nudges each element toward the mean (reducing variance)."""
    if params.size == 0:
        return params
    mean_val = float(np.mean(params))
    return np.array([p + 0.0002 * (mean_val - p) for p in params], dtype=float)

def subsutra11_Adjustment(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 11: Adjustment – adds a small cosine-based adjustment to each element."""
    return np.array([p + 0.0003 * math.cos(p) for p in params], dtype=float)

def subsutra12_Modulation(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 12: Modulation – slightly scales each element based on its index (introducing a gradient)."""
    return np.array([p * (1 + 0.00005 * i) for i, p in enumerate(params)], dtype=float)

def subsutra13_Differentiation(params: np.ndarray) -> np.ndarray:
    """Sub-sutra 13: Differentiation – adds a small term proportional to the local derivative (gradient) of the sequence."""
    if params.size == 0:
        return params
    grad = np.gradient(params)
    return np.array([p + 0.0001 * g for p, g in zip(params, grad)], dtype=float)


_sub_sutras = [
    subsutra1_Refinement, subsutra2_Correction, subsutra3_Recursion, subsutra4_Convergence,
    subsutra5_Stabilization, subsutra6_Simplification, subsutra7_Interpolation, subsutra8_Extrapolation,
    subsutra9_ErrorReduction, subsutra10_Optimization, subsutra11_Adjustment, subsutra12_Modulation,
    subsutra13_Differentiation
]


def apply_main_sutras(params: np.ndarray) -> np.ndarray:
    """
    Apply all 16 main sutras in sequence to the input array.
    
    Args:
        params: NumPy array of values (e.g., variational parameters or intermediate results).
    Returns:
        np.ndarray after applying all main sutra transformations in order.
    """
    result = np.array(params, dtype=float)
    for func in _main_sutras:
        result = func(result)
    return result

def apply_subsutras_parallel(params: np.ndarray) -> np.ndarray:
    """
    Apply all 13 sub-sutras in parallel to the input, and combine their results.
    
    This runs each sub-sutra function concurrently and then averages the outputs to produce 
    a composite refined result. Dynamic resolution is used: for lower complexity inputs, a subset 
    of sub-sutras may be applied to reduce overhead.
    
    Args:
        params: NumPy array of values to refine.
    Returns:
        np.ndarray of refined values after parallel sub-sutra processing.
    """
    data = np.array(params, dtype=float)
    n = data.size
    if n == 0:
        return data
    # Determine complexity (e.g., use standard deviation of gradient as a heuristic)
    grad = np.gradient(data) if n > 1 else np.array([0.0])
    complexity = float(np.std(grad))
    # Decide how many sub-sutras to run based on complexity
    if complexity < 1e-4:
        # Very smooth input, use fewer sub-sutras for efficiency
        active_funcs = _sub_sutras[:3]   # first 3 sub-sutras
    elif complexity < 1e-2:
        # Moderate complexity, use roughly half of the sub-sutras
        active_funcs = _sub_sutras[:7]   # first 7 sub-sutras
    else:
        # High complexity, use all sub-sutras
        active_funcs = _sub_sutras
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, data) for func in active_funcs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    # Combine results (e.g., average across all sub-sutra outputs)
    results_arr = np.array(results, dtype=float)
    combined = np.mean(results_arr, axis=0)
    return combined

def run_full_engine(params: np.ndarray) -> np.ndarray:
    """
    Run the full Vedic core engine on the input: all main sutras in series, then sub-sutras in parallel.
    
    Args:
        params: Input array of values.
    Returns:
        np.ndarray refined after applying main sutras and sub-sutras.
    """
    # 1. Apply main sutras sequentially
    out_main = apply_main_sutras(params)
    # 2. Apply sub-sutras in parallel and combine
    out_final = apply_subsutras_parallel(out_main)
    return out_final
