"""
Integrated GRVQ/TGCR hybrid quantum-classical simulation pipeline.

This module provides a complete numeric workflow that combines:
* 29-sutra Vedic arithmetic updates (serial + concurrent + parallel fusion)
* Minimal full configuration interaction (FCI) solver for a toy spin system
* R4 entanglement topology operations with FFT-based harmonic resynchronization
* Cosmological age estimation driven by entanglement-mapped parameters

All components are implemented with explicit numeric operations—no placeholders
or demo shortcuts. Functions are organized for reuse in larger simulators.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Sequence, Tuple

import numpy as np

########################################################################
# SECTION A: VEDIC ARITHMETIC & 29-SUTRA RECURSIVE UPDATES
########################################################################

# --------------------------
# 16 PRIMARY VEDIC SUTRAS
# --------------------------


def sutra1_Ekadhikena(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        new_val = p + 0.001 * math.sin(p)
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra2_Nikhilam(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        new_val = p - 0.002 * (1.0 - p)
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra3_Urdhva_Tiryagbhyam(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        new_val = p * (1.0 + 0.003 * math.cos(p))
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra4_Urdhva_Veerya(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        new_val = p * math.exp(0.0005 * p)
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra5_Paravartya(params: np.ndarray) -> np.ndarray:
    reversed_array = params[::-1]
    updated = []
    for val in reversed_array:
        new_val = val + 0.0008
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra6_Shunyam_Sampurna(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        if abs(p) <= 0.1:
            new_val = p + 0.1
        else:
            new_val = p
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra7_Anurupyena(params: np.ndarray) -> np.ndarray:
    avg = np.mean(params)
    updated = []
    for p in params:
        new_val = p * (1.0 + 0.0003 * (p - avg))
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra8_Sopantyadvayamantyam(params: np.ndarray) -> np.ndarray:
    updated = []
    i = 0
    while i < len(params) - 1:
        avg_val = 0.5 * (params[i] + params[i + 1])
        updated.append(avg_val)
        updated.append(avg_val)
        i += 2
    if len(params) % 2 == 1:
        updated.append(params[-1])
    return np.array(updated, dtype=float)


def sutra9_Ekanyunena(params: np.ndarray) -> np.ndarray:
    half_size = len(params) // 2
    if half_size == 0:
        return params
    half_values = params[:half_size]
    factor = np.mean(half_values)
    updated = []
    for p in params:
        new_val = p + 0.0007 * factor
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra10_Dvitiya(params: np.ndarray) -> np.ndarray:
    half_start = len(params) // 2
    if half_start == len(params):
        return params
    factor = np.mean(params[half_start:])
    updated = []
    for p in params:
        new_val = p * (1.0 + 0.0004 * factor)
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra11_Virahata(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        new_val = p + 0.0015 * math.sin(2.0 * p)
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra12_Ayur(params: np.ndarray) -> np.ndarray:
    updated = []
    for p in params:
        new_val = p * (1.0 + 0.0006 * abs(p))
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra13_Samuchchhayo(params: np.ndarray) -> np.ndarray:
    s = np.sum(params)
    updated = []
    for p in params:
        new_val = p + 0.0002 * s
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra14_Alankara(params: np.ndarray) -> np.ndarray:
    updated = []
    for i, p in enumerate(params):
        new_val = p + 0.0005 * math.sin(float(i))
        updated.append(new_val)
    return np.array(updated, dtype=float)


def sutra15_Sandhya(params: np.ndarray) -> np.ndarray:
    updated = []
    for i in range(len(params) - 1):
        mid_val = 0.5 * (params[i] + params[i + 1])
        updated.append(mid_val)
    if len(params) > 0:
        updated.append(params[-1])
    return np.array(updated, dtype=float)


def sutra16_Sandhya_Samuccaya(params: np.ndarray) -> np.ndarray:
    indices = np.linspace(1.0, float(len(params)), len(params))
    total_indices = np.sum(indices)
    weighted_sum = 0.0
    for i, p in enumerate(params):
        weighted_sum += p * indices[i]
    w_avg = weighted_sum / total_indices if total_indices != 0 else 0.0
    updated = []
    for p in params:
        new_val = p + 0.0003 * w_avg
        updated.append(new_val)
    return np.array(updated, dtype=float)


def apply_main_sutras(params: np.ndarray) -> np.ndarray:
    functions = [
        sutra1_Ekadhikena,
        sutra2_Nikhilam,
        sutra3_Urdhva_Tiryagbhyam,
        sutra4_Urdhva_Veerya,
        sutra5_Paravartya,
        sutra6_Shunyam_Sampurna,
        sutra7_Anurupyena,
        sutra8_Sopantyadvayamantyam,
        sutra9_Ekanyunena,
        sutra10_Dvitiya,
        sutra11_Virahata,
        sutra12_Ayur,
        sutra13_Samuchchhayo,
        sutra14_Alankara,
        sutra15_Sandhya,
        sutra16_Sandhya_Samuccaya,
    ]
    updated = params
    for func in functions:
        updated = func(updated)
    return updated


# --------------------------
# 13 SUB-SUTRAS (PARALLEL)
# --------------------------


def subsutra1_Refinement(params: np.ndarray) -> np.ndarray:
    out = []
    for p in params:
        out.append(p + 0.0001 * p**2)
    return np.array(out, dtype=float)


def subsutra2_Correction(params: np.ndarray) -> np.ndarray:
    out = []
    for p in params:
        out.append(p - 0.0002 * (p - 0.5))
    return np.array(out, dtype=float)


def subsutra3_Recursion(params: np.ndarray) -> np.ndarray:
    shifted = np.roll(params, 1)
    return 0.5 * (params + shifted)


def subsutra4_Convergence(params: np.ndarray) -> np.ndarray:
    out = []
    for p in params:
        out.append(0.9 * p)
    return np.array(out, dtype=float)


def subsutra5_Stabilization(params: np.ndarray) -> np.ndarray:
    return np.clip(params, 0.0, 1.0)


def subsutra6_Simplification(params: np.ndarray) -> np.ndarray:
    out = []
    for p in params:
        out.append(round(p, 4))
    return np.array(out, dtype=float)


def subsutra7_Interpolation(params: np.ndarray) -> np.ndarray:
    out = [p + 0.00005 for p in params]
    return np.array(out, dtype=float)


def subsutra8_Extrapolation(params: np.ndarray) -> np.ndarray:
    if len(params) < 2:
        return params
    xvals = np.arange(len(params), dtype=float)
    poly = np.polyfit(xvals, params, 1)
    correction = np.polyval(poly, float(len(params)))
    out = []
    for p in params:
        out.append(p + 0.0001 * correction)
    return np.array(out, dtype=float)


def subsutra9_ErrorReduction(params: np.ndarray) -> np.ndarray:
    sd = float(np.std(params))
    out = [p - 0.0001 * sd for p in params]
    return np.array(out, dtype=float)


def subsutra10_Optimization(params: np.ndarray) -> np.ndarray:
    mean_val = float(np.mean(params))
    out = []
    for p in params:
        out.append(p + 0.0002 * (mean_val - p))
    return np.array(out, dtype=float)


def subsutra11_Adjustment(params: np.ndarray) -> np.ndarray:
    out = []
    for p in params:
        out.append(p + 0.0003 * math.cos(p))
    return np.array(out, dtype=float)


def subsutra12_Modulation(params: np.ndarray) -> np.ndarray:
    out = []
    for i, p in enumerate(params):
        out.append(p * (1.0 + 0.00005 * float(i)))
    return np.array(out, dtype=float)


def subsutra13_Differentiation(params: np.ndarray) -> np.ndarray:
    if len(params) < 2:
        return params
    gradient = np.gradient(params)
    out = []
    for p, g in zip(params, gradient):
        out.append(p + 0.0001 * g)
    return np.array(out, dtype=float)


_SUBSUTRA_FUNCS: Tuple = (
    subsutra1_Refinement,
    subsutra2_Correction,
    subsutra3_Recursion,
    subsutra4_Convergence,
    subsutra5_Stabilization,
    subsutra6_Simplification,
    subsutra7_Interpolation,
    subsutra8_Extrapolation,
    subsutra9_ErrorReduction,
    subsutra10_Optimization,
    subsutra11_Adjustment,
    subsutra12_Modulation,
    subsutra13_Differentiation,
)


def _run_parallel_subsutras(params: np.ndarray, max_workers: int = 8) -> List[np.ndarray]:
    def _apply(func) -> np.ndarray:
        return func(params)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_apply, func) for func in _SUBSUTRA_FUNCS]
        return [f.result() for f in futures]


def apply_subsutras_parallel(params: np.ndarray, max_workers: int = 8) -> np.ndarray:
    parallel_outputs = _run_parallel_subsutras(params, max_workers=max_workers)
    stacked = np.vstack(parallel_outputs)
    combined = np.mean(stacked, axis=0)
    return combined


def update_29_sutras(params: np.ndarray, max_workers: int = 8) -> np.ndarray:
    intermediate = apply_main_sutras(params)
    final = apply_subsutras_parallel(intermediate, max_workers=max_workers)
    return final


########################################################################
# SECTION B: FULL CONFIGURATION INTERACTION (FCI) FOR A SMALL SYSTEM
########################################################################


def generate_slater_determinants(n_spin_orbs: int, n_elec: int) -> List[Tuple[int, ...]]:
    from itertools import combinations

    dets = []
    for combo in combinations(range(n_spin_orbs), n_elec):
        dets.append(combo)
    return dets


def compute_single_excitation_sign(det_occ: Sequence[int], p_out: int, p_in: int) -> int:
    occ_list = sorted(det_occ)
    idx_out = occ_list.index(p_out)
    sign = (-1) ** idx_out
    occ_list.pop(idx_out)
    insert_pos = 0
    for orb in occ_list:
        if orb < p_in:
            insert_pos += 1
    sign *= (-1) ** insert_pos
    return sign


def compute_double_excitation_sign(det_occ: Sequence[int], out_list: Sequence[int], in_list: Sequence[int]) -> int:
    occ_list = sorted(det_occ)
    sign = 1
    idx0 = occ_list.index(out_list[0])
    sign *= (-1) ** idx0
    occ_list.pop(idx0)
    idx1 = occ_list.index(out_list[1])
    sign *= (-1) ** idx1
    occ_list.pop(idx1)
    insert0 = 0
    for x in occ_list:
        if x < in_list[0]:
            insert0 += 1
    sign *= (-1) ** insert0
    occ_list.insert(insert0, in_list[0])
    insert1 = 0
    for x in occ_list:
        if x < in_list[1]:
            insert1 += 1
    sign *= (-1) ** insert1
    occ_list.insert(insert1, in_list[1])
    return sign


def compute_offdiag_element(det_i: Sequence[int], det_j: Sequence[int], h_1e: np.ndarray, g_2e: np.ndarray) -> float:
    occ_i = set(det_i)
    occ_j = set(det_j)
    diff_i = occ_i - occ_j
    diff_j = occ_j - occ_i
    n_diff = len(diff_i) + len(diff_j)

    if n_diff == 0:
        return 0.0

    if n_diff == 2:
        p_out = list(diff_i)[0]
        p_in = list(diff_j)[0]
        sign = compute_single_excitation_sign(det_i, p_out, p_in)
        val_1e = sign * h_1e[p_in, p_out]
        remain = list(occ_i)
        remain.remove(p_out)
        val_2e = 0.0
        for q_ in remain:
            val_2e += sign * (g_2e[p_in, q_, p_out, q_] - g_2e[p_in, q_, q_, p_out])
        return val_1e + val_2e

    if n_diff == 4:
        p_out_list = sorted(list(diff_i))
        p_in_list = sorted(list(diff_j))
        sign = compute_double_excitation_sign(det_i, p_out_list, p_in_list)
        (p, q) = p_out_list
        (r, s) = p_in_list
        val_2e = sign * (g_2e[r, s, p, q] - g_2e[r, s, q, p])
        return val_2e

    return 0.0


def build_fci_hamiltonian(slater_dets: Iterable[Sequence[int]], h_1e: np.ndarray, g_2e: np.ndarray) -> np.ndarray:
    dets = list(slater_dets)
    ndets = len(dets)
    H = np.zeros((ndets, ndets), dtype=float)

    for i, det_i in enumerate(dets):
        diag_val = 0.0
        for p in det_i:
            diag_val += h_1e[p, p]
        from itertools import combinations

        for (p, q) in combinations(det_i, 2):
            diag_val += g_2e[p, q, p, q]
        H[i, i] = diag_val

    for i, det_i in enumerate(dets):
        occ_i = set(det_i)
        for j, det_j in enumerate(dets):
            if j <= i:
                continue
            occ_j = set(det_j)
            diff_i = occ_i - occ_j
            diff_j = occ_j - occ_i
            n_diff = len(diff_i) + len(diff_j)
            if n_diff in (2, 4):
                val = compute_offdiag_element(det_i, det_j, h_1e, g_2e)
                H[i, j] = val
                H[j, i] = val

    return H


########################################################################
# SECTION C: COSMOLOGICAL UNIVERSE AGE (NUMERIC INTEGRATION)
########################################################################


def expansion_function(a: float, omega_m: float, omega_lambda: float, omega_k: float, omega_r: float, w: float) -> float:
    rad = omega_r * (a ** (-4))
    mat = omega_m * (a ** (-3))
    cur = omega_k * (a ** (-2))
    de = omega_lambda * (a ** (-3 * (1.0 + w)))
    return rad + mat + cur + de


def universe_age(
    omega_m: float,
    omega_lambda: float,
    omega_k: float,
    omega_r: float,
    R0: float,
    w: float = -1.0,
    a_start: float = 1.0e-8,
    a_end: float = 1.0,
    steps: int = 200_000,
) -> float:
    KM_PER_MPC = 3.08567758149137e19
    SECS_PER_YEAR = 3.15576e7
    per_sec = R0 / KM_PER_MPC
    per_year = per_sec * SECS_PER_YEAR

    a_vals = np.linspace(a_start, a_end, steps)

    def integrand(a_val: float) -> float:
        e_val = expansion_function(a_val, omega_m, omega_lambda, omega_k, omega_r, w)
        h_a = per_year * math.sqrt(e_val)
        return 1.0 / (a_val * h_a)

    y_vals = [integrand(x) for x in a_vals]
    integral_years = np.trapezoid(y_vals, a_vals)
    age_gyr = integral_years / 1.0e9
    return age_gyr


########################################################################
# SECTION D: R4 ENTANGLEMENT TOPOLOGY (NUMERIC ONLY)
########################################################################


class GRVQEntityR4:
    """
    Container for a GRVQ R4 entity.

    lambda_vec     : np.ndarray, shape (d_lambda,)
    phase_shells   : np.ndarray, shape (n_shells,)
    psi            : np.ndarray, shape (L,), complex
    """

    def __init__(self, lambda_vec: np.ndarray, phase_shells: np.ndarray, psi: np.ndarray):
        self.lambda_vec = np.array(lambda_vec, dtype=float)
        self.phase_shells = np.array(phase_shells, dtype=float)
        self.psi = np.array(psi, dtype=complex)


def compute_phase_diff(phi_i: np.ndarray, phi_j: np.ndarray) -> float:
    diff = phi_i - phi_j
    return float(np.linalg.norm(diff, ord=2))


def compute_lambda_diff(lambda_i: np.ndarray, lambda_j: np.ndarray) -> np.ndarray:
    return (lambda_i - lambda_j).astype(float)


def build_tensor_link_graph(entities: Sequence[GRVQEntityR4]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(entities)
    if n == 0:
        raise ValueError("No entities in build_tensor_link_graph")
    d_lambda = entities[0].lambda_vec.size
    w = np.zeros((n, n), dtype=float)
    phase_diff = np.zeros((n, n), dtype=float)
    lambda_diff = np.zeros((n, n, d_lambda), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            dphi = compute_phase_diff(entities[i].phase_shells, entities[j].phase_shells)
            dlambda_vec = compute_lambda_diff(entities[i].lambda_vec, entities[j].lambda_vec)
            dlambda_norm = float(np.linalg.norm(dlambda_vec, ord=2))

            phase_diff[i, j] = dphi
            phase_diff[j, i] = dphi
            lambda_diff[i, j, :] = dlambda_vec
            lambda_diff[j, i, :] = -dlambda_vec

            w_ij = dphi + dlambda_norm
            w[i, j] = w_ij
            w[j, i] = w_ij

    return w, phase_diff, lambda_diff


def check_sync_conditions(
    phase_diff: np.ndarray, lambda_diff: np.ndarray, epsilon_phi: float, delta_vec: np.ndarray
) -> np.ndarray:
    n = phase_diff.shape[0]
    d_lambda = lambda_diff.shape[2]
    if d_lambda != delta_vec.size:
        raise ValueError("lambda_diff last dimension and delta_vec size must match")
    violations = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if phase_diff[i, j] >= epsilon_phi:
                violations[i, j] = True
                violations[j, i] = True
                continue
            diff_vec = lambda_diff[i, j, :]
            for k in range(d_lambda):
                if abs(diff_vec[k]) >= delta_vec[k]:
                    violations[i, j] = True
                    violations[j, i] = True
                    break

    return violations


def entangled_subsutra_map(lambda_vec: np.ndarray) -> np.ndarray:
    after_s8 = sutra8_Sopantyadvayamantyam(lambda_vec)
    after_s13 = sutra13_Samuchchhayo(after_s8)
    return after_s13


def harmonic_resync_fft(entity_i: GRVQEntityR4, entity_j: GRVQEntityR4, epsilon: float = 1e-12) -> None:
    psi_i = entity_i.psi
    psi_j = entity_j.psi
    if psi_i.size != psi_j.size:
        raise ValueError("psi arrays must have the same length for harmonic resync")
    F_i = np.fft.fft(psi_i)
    F_j = np.fft.fft(psi_j)
    ratio = F_j / (F_i + epsilon)
    F_i_new = F_i * ratio
    psi_i_new = np.fft.ifft(F_i_new)
    entity_i.psi = psi_i_new


def r4_entanglement_step(
    entities: List[GRVQEntityR4], epsilon_phi: float, delta_vec: np.ndarray, max_workers: int = 8
) -> None:
    if len(entities) == 0:
        return
    w, phase_diff, lambda_diff = build_tensor_link_graph(entities)
    violations = check_sync_conditions(phase_diff, lambda_diff, epsilon_phi, delta_vec)
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            if not violations[i, j]:
                continue
            lambda_i = entities[i].lambda_vec
            lambda_j = entities[j].lambda_vec
            lambda_mean = 0.5 * (lambda_i + lambda_j)
            drift_i = float(np.linalg.norm(lambda_i - lambda_mean, ord=2))
            drift_j = float(np.linalg.norm(lambda_j - lambda_mean, ord=2))
            target_idx = i if drift_i >= drift_j else j
            entangled_sub = entangled_subsutra_map(entities[target_idx].lambda_vec)
            entities[target_idx].lambda_vec = entangled_sub
            harmonic_resync_fft(entities[target_idx], entities[j if target_idx == i else i])


########################################################################
# SECTION E: MAIN RECALCULATION PIPELINE
########################################################################


def run_full_pipeline() -> None:
    print("==============================================================")
    print("  FULL PIPELINE: 29 SUTRAS + FCI + R4 ENTANGLEMENT + AGE")
    print("==============================================================\n")

    # Step 1: Vedic 29-sutra parameter update (serial + concurrent blend)
    initial_params = np.array([0.75, 0.2, 0.91, 0.47, 0.01], dtype=float)
    print("Initial parameters:", initial_params)
    updated_params = update_29_sutras(initial_params)
    print("Updated parameters after 29 sutras:", updated_params)

    # Step 2: Minimal FCI system (4 spin-orbitals, 2 electrons)
    n_spin = 4
    n_elec = 2
    slater_dets = generate_slater_determinants(n_spin, n_elec)

    h_1e = np.array(
        [
            [-1.2, 0.05, 0.02, 0.01],
            [0.05, -1.0, 0.03, 0.02],
            [0.02, 0.03, -0.8, 0.04],
            [0.01, 0.02, 0.04, -0.7],
        ],
        dtype=float,
    )

    g_2e = np.zeros((4, 4, 4, 4), dtype=float)
    g_2e[0, 0, 0, 0] = 1.1
    g_2e[1, 1, 1, 1] = 1.0
    g_2e[2, 2, 2, 2] = 0.9
    g_2e[3, 3, 3, 3] = 0.8

    g_2e[0, 0, 1, 1] = 0.3
    g_2e[1, 1, 0, 0] = 0.3
    g_2e[0, 0, 2, 2] = 0.25
    g_2e[2, 2, 0, 0] = 0.25
    g_2e[0, 0, 3, 3] = 0.2
    g_2e[3, 3, 0, 0] = 0.2
    g_2e[1, 1, 2, 2] = 0.28
    g_2e[2, 2, 1, 1] = 0.28
    g_2e[1, 1, 3, 3] = 0.26
    g_2e[3, 3, 1, 1] = 0.26
    g_2e[2, 2, 3, 3] = 0.27
    g_2e[3, 3, 2, 2] = 0.27

    g_2e[0, 1, 1, 0] = 0.1
    g_2e[1, 0, 0, 1] = 0.1
    g_2e[0, 2, 2, 0] = 0.08
    g_2e[2, 0, 0, 2] = 0.08
    g_2e[0, 3, 3, 0] = 0.07
    g_2e[3, 0, 0, 3] = 0.07
    g_2e[1, 2, 2, 1] = 0.09
    g_2e[2, 1, 1, 2] = 0.09
    g_2e[1, 3, 3, 1] = 0.06
    g_2e[3, 1, 1, 3] = 0.06
    g_2e[2, 3, 3, 2] = 0.05
    g_2e[3, 2, 2, 3] = 0.05

    H_fci = build_fci_hamiltonian(slater_dets, h_1e, g_2e)
    evals, evecs = np.linalg.eigh(H_fci)
    ground_idx = np.argmin(evals)
    ground_e = evals[ground_idx]
    ground_vec = evecs[:, ground_idx]

    print("\nFCI Hamiltonian shape:", H_fci.shape)
    print("Slater determinants:")
    for i, d in enumerate(slater_dets):
        print(f"   Index {i} => {d}")
    print("Eigenvalues:", evals)
    print("Ground state energy:", ground_e)
    print("Ground state wavefunction (lowest eigenvector):", ground_vec)

    quantum_param = float(ground_vec[0] ** 2)
    print(f"\nQuantum parameter (amplitude^2 of determinant 0) = {quantum_param:.6f}")

    # Step 3: R4 entanglement ensemble construction
    rng = np.random.default_rng(123)
    n_entities = 3
    d_lambda = min(len(updated_params), 4)
    psi_len = 64
    entities: List[GRVQEntityR4] = []

    for i in range(n_entities):
        base_lambda = updated_params[:d_lambda]
        scale_factor = 1.0 + 0.05 * (float(quantum_param) - 0.5) * (i + 1)
        lambda_vec = base_lambda * scale_factor
        phase_shells = rng.uniform(low=0.0, high=2.0 * math.pi, size=d_lambda)
        x = np.linspace(0.0, 2.0 * math.pi, psi_len, endpoint=False)
        psi = np.exp(1j * (x * (1.0 + 0.1 * i) + float(quantum_param)))
        entities.append(GRVQEntityR4(lambda_vec, phase_shells, psi))

    epsilon_phi = 0.5
    delta_vec = np.full(d_lambda, 0.15, dtype=float)

    print("\nR4 entanglement: initial lambda vectors for entities:")
    for idx, e in enumerate(entities):
        print(f"  Entity {idx}: lambda_vec = {e.lambda_vec}")

    for _ in range(3):
        r4_entanglement_step(entities, epsilon_phi, delta_vec)

    print("\nR4 entanglement: lambda vectors after synchronization steps:")
    for idx, e in enumerate(entities):
        print(f"  Entity {idx}: lambda_vec = {e.lambda_vec}")

    all_lambda = np.vstack([e.lambda_vec for e in entities])
    mean_lambda_vec = np.mean(all_lambda, axis=0)
    mean_lambda_scalar = float(np.mean(mean_lambda_vec))

    mapped_scalar = 0.5 + 0.5 * math.tanh(mean_lambda_scalar)

    base_om = 0.30
    base_ol = 0.70
    base_ok = 0.00
    base_or = 0.00

    omega_m = base_om * (0.8 + 0.4 * mapped_scalar)
    omega_m = max(0.0, min(omega_m, 1.0))
    omega_lambda = 1.0 - omega_m

    R0 = 69.5
    w = -1.0

    age_estimated = universe_age(omega_m, omega_lambda, base_ok, base_or, R0, w=w)

    print("\nGRVQ/TGCR-style cosmology parameters after entanglement mapping:")
    print("  Effective mean lambda scalar:", mean_lambda_scalar)
    print("  Mapped scalar in [0,1]:", mapped_scalar)
    print("  Omega_m:", omega_m)
    print("  Omega_lambda:", omega_lambda)
    print("  Omega_k, Omega_r:", base_ok, base_or)
    print("  R0 (km/s/Mpc):", R0)
    print("  w (equation-of-state):", w)
    print("Recalculated universe age: %.6f billion years" % age_estimated)

    print("\n[Pipeline complete]")


if __name__ == "__main__":
    run_full_pipeline()
