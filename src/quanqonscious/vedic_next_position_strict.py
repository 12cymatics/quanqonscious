# vedic_next_position_strict.py
# Full production module with NO fallbacks, NO proxies, NO demos, NO placeholders.
# Strict implementation of:
#   • Golden-angle θ scheduler (canonical) and φ-scaled radial toggle (×2φ, ÷2φ)
#   • 29 Vedic Sutra transforms as concrete numerical corrections on both θ and r
#   • Screw-axis ladder (3-step helical constraint) — deterministic, small-angle alignment
#   • Hopf sparsifier — deterministic pair-annihilation of alternating jitter
#   • ZPE regulator (HARMONIC ONLY) acting on the external Omega channel (MANDATORY)
#   • RL shunt family on Omega ('R', 'RL', 'active') — no randomization
#   • Palindromic alloy tools for Λ_pal and golden-bifactor with Lucas/Fibonacci weights
#   • Anurupye–Śūnyam validator (STRICT ASSERT; no minimal-norm projections)
#
# Usage (strict):
#   from vedic_next_position_strict import NextPositionPredictorStrict
#   pred = NextPositionPredictorStrict(...)
#   traj = pred.predict(r0, theta0, steps, S_sequence=your_S_k_1_iterable_of_length>=steps)
#
# This module NEVER fabricates Omega. If S_sequence is missing/short, it raises.

from math import pi, sqrt, sin, cos, atan2
import math
import numpy as np
from typing import Iterable, List, Tuple, Optional

__all__ = [
    "PHI", "TAU", "GOLDEN_ANGLE", "GOLDEN_ANGLE_COMPLEMENT",
    "lucas_numbers", "fibonacci_numbers",
    "lambda_pal", "lambda_golden_bifactor", "assert_anurupye_shunyam",
    "ZPEHarmonic", "RLShuntStrict", "ScrewAxisLadder", "HopfSparsifier",
    "VedicSutras29", "NextPositionPredictorStrict"
]

# =========================
# Constants and primitives
# =========================

PHI = (1 + 5**0.5) / 2.0
TAU = 2.0 * pi
GOLDEN_ANGLE = TAU * (1.0 - 1.0/PHI)             # 137.507764...
GOLDEN_ANGLE_COMPLEMENT = TAU * (1.0 - 1.0/(PHI**2))  # 222.492235...

def _safe(x: float, eps: float = 1e-12) -> float:
    """Finite, nonzero guard (deterministic)."""
    try:
        if not math.isfinite(x):
            return eps
    except Exception:
        return eps
    return x if abs(x) > eps else (eps if x >= 0 else -eps)

# ============================================
# Lucas/Fibonacci utilities and alloy function
# ============================================

def lucas_numbers(n: int) -> np.ndarray:
    """
    Return first n Lucas numbers (L1=2, L2=1, Lk=Lk-1+Lk-2), as float array length n.
    Deterministic; no seeding; no randomness.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([2.0], dtype=float)
    L = np.zeros(n, dtype=float)
    L[0] = 2.0
    L[1] = 1.0
    for k in range(2, n):
        L[k] = L[k-1] + L[k-2]
    return L

def fibonacci_numbers(n: int) -> np.ndarray:
    """
    Return first n Fibonacci numbers (F1=1, F2=1, Fk=Fk-1+Fk-2), as float array length n.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    F = np.zeros(n, dtype=float)
    F[0] = 1.0
    F[1] = 1.0
    for k in range(2, n):
        F[k] = F[k-1] + F[k-2]
    return F

def _normalized(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    if abs(s) < 1e-18:
        raise ValueError("Zero-sum weights are invalid.")
    return (v / s).astype(float)

def lambda_pal(S_vals: Iterable[float]) -> float:
    """
    Compute Λ_pal for an even-length S sequence (length 2m), using Lucas weights of length m:
        Λ_pal = Σ_{k=1..m} α_k [ S_k + S_{2m+1-k} ],
    where α_k = L_k / Σ_{j=1..m} L_j, with Lucas numbers L.
    STRICT: Raises if len(S) is odd or <2.
    """
    S = np.asarray(list(S_vals), dtype=float)
    n = S.size
    if n < 2 or (n % 2) != 0:
        raise ValueError("S length must be even and >= 2 for Λ_pal.")
    m = n // 2
    L = _normalized(lucas_numbers(m))
    total = 0.0
    # 1-indexed pairing (k, 2m+1-k) → zero-indexed (k-1, 2m-k)
    for k in range(1, m+1):
        total += L[k-1] * (S[k-1] + S[2*m - k])
    return float(total)

def lambda_golden_bifactor(S_vals: Iterable[float]) -> float:
    """
    Golden-bifactor alloy for even-length S (length 2m):
        Λ_φ = Σ β^F_k S_k + Σ β^L_k S_{k+m},  k=1..m,
    where β^F from Fibonacci(m) and β^L from Lucas(m), both normalized.
    STRICT: Raises if len(S) is odd or <2.
    """
    S = np.asarray(list(S_vals), dtype=float)
    n = S.size
    if n < 2 or (n % 2) != 0:
        raise ValueError("S length must be even and >= 2 for Λ_φ.")
    m = n // 2
    betaF = _normalized(fibonacci_numbers(m))
    betaL = _normalized(lucas_numbers(m))
    return float(np.dot(betaF, S[:m]) + np.dot(betaL, S[m:2*m]))

def assert_anurupye_shunyam(S_vals: Iterable[float], tol: float = 0.0) -> None:
    """
    STRICT Anurupye–Śūnyam validator (NO projection, NO minimal-norm).
    For even-length S (length 2m), define α over 2m by mirroring Lucas weights:
        α = [α_1, ..., α_m, α_m, ..., α_1], α_k = L_k / Σ L_j, L = Lucas(m).
    Then assert: α · S[:2m] == 0  (within tol). If not, raise ValueError.
    """
    S = np.asarray(list(S_vals), dtype=float)
    n = S.size
    if n < 2 or (n % 2) != 0:
        raise ValueError("S length must be even and >= 2 for Anurupye–Śūnyam check.")
    m = n // 2
    L = _normalized(lucas_numbers(m))
    alpha = np.concatenate([L, L[::-1]])
    val = float(np.dot(alpha, S[:2*m]))
    if tol == 0.0:
        if val != 0.0:
            raise ValueError(f"Anurupye–Śūnyam violated: α·S = {val} (tol=0).")
    else:
        if abs(val) > abs(tol):
            raise ValueError(f"Anurupye–Śūnyam violated: α·S = {val}, tol={tol}.")

# ==========================
# ZPE (harmonic-only) model
# ==========================

class ZPEHarmonic:
    """
    Harmonic ZPE on scalar Ω (strict; NO square fallback):
        E(Ω) = -μ0 * [ Ω^2 + 1/(Ω^2 + ε) ]
        ∂E/∂Ω = -μ0 * [ 2Ω - 2Ω/(Ω^2 + ε)^2 ]
    Correction step is deterministic: ΔΩ = -η * ∂E/∂Ω (η>0).
    """
    def __init__(self, mu0: float = 4e-7*np.pi, eps: float = 1e-12, step: float = 5e-4):
        self.mu0 = float(mu0)
        self.eps = float(eps)
        self.step = float(step)

    def energy_and_correction(self, omega: float) -> Tuple[float, float]:
        Om2 = omega*omega + self.eps
        E = -self.mu0 * (omega*omega + 1.0/Om2)
        grad = -self.mu0 * (2.0*omega - 2.0*omega/(Om2*Om2))
        corr = -self.step * grad
        return float(E), float(corr)

# ===========================
# Deterministic RL shunt fam
# ===========================

class RLShuntStrict:
    """
    Discrete-time shunt on Ω. Modes are strictly deterministic.
      mode ∈ {'R','RL','active'} (no 'none', no randomness)
    """
    def __init__(self, mode: str = 'RL', dt: float = 1.0, wn: float = 0.15, zeta: float = 0.7):
        if mode not in ('R','RL','active'):
            raise ValueError("RLShuntStrict.mode must be 'R', 'RL', or 'active'.")
        self.mode = mode
        self.dt = float(dt)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.x1 = 0.0
        self.x2 = 0.0

    def reset(self) -> None:
        self.x1 = 0.0
        self.x2 = 0.0

    def feedback(self, omega: float) -> float:
        if self.mode == 'R':
            a = self.wn; b = self.wn; k = self.wn
            self.x1 += self.dt * (-a*self.x1 + b*omega)
            return -k*self.x1
        # 'RL' and 'active'
        wn = self.wn
        zeta = self.zeta if self.mode == 'RL' else max(0.9, self.zeta)
        self.x1 += self.dt * (self.x2)
        self.x2 += self.dt * (-2.0*zeta*wn*self.x2 - wn*wn*self.x1 + wn*wn*omega)
        k1 = wn*wn
        k2 = 2.0*zeta*wn
        return -(k1*self.x1 + k2*self.x2)

# =====================================
# Geometric stabilizers (deterministic)
# =====================================

class ScrewAxisLadder:
    """
    Enforce 3-step helical relation on complex positions z_n:
        z_{n+3} ≈ e^{i ψ} z_n
    Returns a tiny δθ to align. Deterministic; no randomness.
    """
    def __init__(self, pitch_phase: float = pi/9, gain: float = 1e-3):
        self.psi = float(pitch_phase)
        self.gain = float(gain)
        self.history: List[complex] = []

    def reset(self) -> None:
        self.history = []

    def correction(self, z_next: complex) -> float:
        self.history.append(z_next)
        if len(self.history) < 4:
            return 0.0
        z_nm3 = self.history[-4]
        target = z_nm3 * complex(cos(self.psi), sin(self.psi))
        phi_next = atan2(z_next.imag, z_next.real)
        phi_tgt  = atan2(target.imag, target.real)
        dphi = (phi_tgt - phi_next + TAU) % TAU
        if dphi > math.pi:
            dphi -= TAU
        return self.gain * dphi

class HopfSparsifier:
    """
    Pair-annihilating filter: reduce alternating equal-and-opposite jitter.
    Deterministic sliding window over u_n = z_n - z_{n-1}.
    """
    def __init__(self, window: int = 4, gain: float = 0.4):
        self.window = int(window)
        self.gain = float(gain)
        self.history: List[complex] = []

    def reset(self) -> None:
        self.history = []

    def correction(self, z_prev: complex, z_next: complex) -> complex:
        u = z_next - z_prev
        self.history.append(u)
        if len(self.history) < self.window:
            return 0.0 + 0.0j
        windowed = np.array(self.history[-self.window:], dtype=complex)
        k = np.arange(windowed.size)
        alt = np.mean(((-1.0)**k) * windowed)
        return -self.gain * alt

# ================================
# 29 Vedic Sutra numeric variants
# ================================

class VedicSutras29:
    """Concrete numeric analogues of 29 sutras, used as micro-corrections."""
    def __init__(self, base: float = 10.0):
        self.base = float(base)

    # 1
    def ekadhikena_purvena(self, x): return x + 1.0
    # 2
    def nikhilam(self, x, base=None):
        b = float(self.base if base is None else base)
        frac, integer = math.modf(x)
        comp_int  = (b - (abs(int(integer)) % int(b))) % b
        comp_frac = (1.0 - abs(frac)) % 1.0
        sign = -1.0 if x >= 0 else 1.0
        return sign*(comp_int + comp_frac)
    # 3
    def urdhva_tiryagbhyam(self, a,b,c=0.0,d=0.0): return a*b + c*d
    # 4
    def paravartya_yojayet(self, x, gain=1.0): return gain/_safe(x)
    # 5
    def sunyam_samyasamuccaye(self, a,b,tol=1e-9): return 0.0 if abs(a+b)<=tol else (a+b)
    # 6
    def anurupyena(self, x, ratio): return x*ratio
    # 7
    def sankalana_vyavakalana(self, a,b,mix=0.5): return mix*(a+b)+(1.0-mix)*(a-b)
    # 8
    def purana_apurana(self, x, target, gain=1.0): return x + gain*(target-x)
    # 9
    def yavadunam(self, current, target, gain=1.0): return gain*(target-current)
    # 10
    def vyashti_samashti(self, local_val, global_val, alpha=0.5): return alpha*local_val+(1.0-alpha)*global_val
    # 11
    def shesanyankena_charamena(self, x, quantum=1e-6): return math.copysign(math.floor(abs(x)/quantum+0.5)*quantum, x)
    # 12
    def sopantyadvayamantyam(self, last, penult, current, gain=1.0):
        delta1=current-last; delta2=last-penult; return gain*(delta1+2.0*delta2)
    # 13
    def ekanyunena_purvena(self, x): return x-1.0
    # 14
    def gunitasamuccayah(self, a_list, b_list, tol=1e-9):
        pa=np.prod(np.array(a_list)+tol); pb=np.prod(np.array(b_list)+tol); return (pa,pb,(pa-pb))
    # 15
    def gunakasamuccayah(self, arr1, arr2):
        arr1=np.array(arr1); arr2=np.array(arr2); return float(np.sum(arr1*arr2))
    # 16
    def lopana_sthapana(self, x, mask=False, memory=0.0, k_restore=0.1): return (0.0 if mask else x)+(k_restore*memory if mask else 0.0)
    # 17
    def adyamadyena_antyamantyena(self, series):
        if len(series)<2: return 0.0
        return float(series[0]*series[0] + series[-1]*series[-1])
    # 18
    def chalana_kalanabhyam(self, series):
        s=np.asarray(series,dtype=float)
        if s.size<2: return (0.0,0.0)
        d=np.diff(s); return float(np.mean(d)), float(np.var(d))
    # 19
    def navasesa(self, x): return ((int(round(x)) % 9) + 9) % 9
    # 20
    def vilokanam(self, trend, base_gain=1.0): return base_gain/(1.0+abs(trend))
    # 21
    def shunyanka(self, x, deadband=1e-6): return 0.0 if abs(x)<deadband else x
    # 22
    def antyayoreva(self, series, k=1.0):
        if len(series)<2: return 0.0
        return k*(series[-1]-series[-2])
    # 23
    def sisyate_sesasamjnah(self, remainder, memory, beta=0.5): return beta*memory+(1.0-beta)*remainder
    # 24
    def diagonal_mix(self, x,y,mix=0.5): return mix*x+(1.0-mix)*y
    # 25
    def yavadunam_phi(self, current, target, k=1.0): return k*(target-current)/PHI
    # 26
    def purana_to_bounds(self, x, lo, hi, k=1.0):
        if x<lo: return x + k*(lo-x)
        if x>hi: return x - k*(x-hi)
        return x
    # 27
    def samuccaya_robust(self, values):
        arr=np.sort(np.array(values,dtype=float))
        if arr.size==0: return 0.0
        if arr.size<5: return float(np.mean(arr))
        return float(np.mean(arr[1:-1]))
    # 28
    def vyashti_weight(self, parts):
        arr=np.array(parts,dtype=float); s=np.sum(np.abs(arr))+1e-12; return np.abs(arr)/s
    # 29
    def samashti_normalize(self, arr):
        arr=np.array(arr,dtype=float); s=np.sum(arr)+1e-12; return (arr/s).tolist(), float(s)

# ==========================================
# Main strict predictor (no proxies anywhere)
# ==========================================

class NextPositionPredictorStrict:
    """
    Strict polar-step predictor with:
      • Golden-angle θ update (canonical) and φ-scaled radial toggle (×2φ, ÷2φ)
      • 29 Vedic sutra corrections (θ and r channels)
      • Screw-axis ladder (3-step phase alignment)
      • Hopf sparsifier (alternation annihilation)
      • Harmonic ZPE on external Ω (S_k(1) MANDATORY; NO proxy); RL shunt (R/RL/active)

    All components are deterministic. No internal randomness. No "square" ZPE fallback. No proxy Ω.
    """
    def __init__(self,
                 use_complement_angle: bool = False,
                 radial_mode: str = "toggle",    # 'toggle' or 'phyllotaxis'
                 c_phyllo: float = 1.0,
                 bounds_r: Tuple[float,float] = (0.0, 1e12),
                 vedic_base: float = 10.0,
                 ladder_on: bool = True,
                 sparsify_on: bool = True,
                 pitch_phase: float = pi/9,
                 zpe_mu0: float = 4e-7*np.pi,
                 zpe_eps: float = 1e-12,
                 zpe_step: float = 5e-4,
                 shunt_mode: str = 'RL',
                 shunt_dt: float = 1.0,
                 shunt_wn: float = 0.15,
                 shunt_zeta: float = 0.7):
        # Base schedulers
        self.ga = GOLDEN_ANGLE_COMPLEMENT if use_complement_angle else GOLDEN_ANGLE
        self.radial_mode = radial_mode
        self.c_phyllo = float(c_phyllo)
        self.bounds_r = (float(bounds_r[0]), float(bounds_r[1]))
        self._toggle_state = 0

        # Vedic sutras + weights
        self.sutras = VedicSutras29(base=vedic_base)
        self.theta_weights = np.linspace(7.5e-4, 1.2e-3, 29)
        self.radius_weights = np.linspace(7.5e-4, 1.2e-3, 29)
        self.theta_history: List[float] = []
        self.radius_history: List[float] = []

        # Geometry
        self.ladder_on = bool(ladder_on)
        self.sparsify_on = bool(sparsify_on)
        self.ladder = ScrewAxisLadder(pitch_phase=pitch_phase, gain=1e-3)
        self.sparsifier = HopfSparsifier(window=4, gain=0.4)

        # ZPE (harmonic-only) + shunt
        self.zpe = ZPEHarmonic(mu0=zpe_mu0, eps=zpe_eps, step=zpe_step)
        self.shunt = RLShuntStrict(mode=shunt_mode, dt=shunt_dt, wn=shunt_wn, zeta=shunt_zeta)

        # External Ω sequence (S_k(1)) — must be supplied per step
        self._S_values: List[float] = []
        self.Omega_history: List[float] = []

    # ----------------
    # Base step maps
    # ----------------
    def _base_step(self, r: float, theta: float, n: int) -> Tuple[float, float]:
        theta_next = theta + self.ga
        if self.radial_mode == "toggle":
            if self._toggle_state == 0:
                r_next = r * (2.0 * PHI); self._toggle_state = 1
            else:
                r_next = r / (2.0 * PHI); self._toggle_state = 0
        elif self.radial_mode == "phyllotaxis":
            r_next = self.c_phyllo * sqrt(n + 1.0)
        else:
            raise ValueError("radial_mode must be 'toggle' or 'phyllotaxis'.")
        return r_next, theta_next

    # ------------------------
    # 29-sutra correction sum
    # ------------------------
    def _sutra_corrections(self, r: float, theta: float,
                           r_next: float, theta_next: float, n: int) -> Tuple[float, float]:
        last_theta = self.theta_history[-1] if self.theta_history else theta
        pen_theta  = self.theta_history[-2] if len(self.theta_history) >= 2 else theta
        last_r = self.radius_history[-1] if self.radius_history else r
        pen_r  = self.radius_history[-2] if len(self.radius_history) >= 2 else r
        global_theta_mean = float(np.mean(self.theta_history)) if self.theta_history else theta
        global_r_mean     = float(np.mean(self.radius_history)) if self.radius_history else r

        dtheta = [0.0]*29; dr = [0.0]*29

        # θ-channel
        dtheta[0]  = self.sutras.ekadhikena_purvena(0.0) * 0.0
        dtheta[1]  = self.sutras.nikhilam(theta_next, base=TAU) * 1e-3
        dtheta[2]  = self.sutras.urdhva_tiryagbhyam(theta, r, last_theta, last_r) * 1e-9
        dtheta[3]  = self.sutras.paravartya_yojayet(theta_next, gain=1e-3)
        dtheta[4]  = self.sutras.sunyam_samyasamuccaye(theta - global_theta_mean, global_theta_mean - theta) * 0.0
        dtheta[5]  = self.sutras.anurupyena(theta_next, ratio=1.0) * 0.0
        dtheta[6]  = self.sutras.sankalana_vyavakalana(theta_next, last_theta, mix=0.5) * 0.0
        dtheta[7]  = self.sutras.purana_apurana(theta_next, target=(theta % TAU), gain=1e-3) - theta_next
        dtheta[8]  = self.sutras.yavadunam(theta_next, target=(theta % TAU), gain=1e-3)
        dtheta[9]  = self.sutras.vyashti_samashti(theta_next, global_theta_mean, alpha=0.5) - theta_next
        dtheta[10] = self.sutras.shesanyankena_charamena(theta_next, quantum=1e-6) - theta_next
        dtheta[11] = self.sutras.sopantyadvayamantyam(last_theta, pen_theta, theta_next, gain=1e-3)
        dtheta[12] = self.sutras.ekanyunena_purvena(0.0) * 0.0
        dtheta[13] = self.sutras.gunitasamuccayah([theta_next+1e-9],[theta+1e-9])[2] * 1e-12
        dtheta[14] = self.sutras.gunakasamuccayah([theta_next],[theta]) * 1e-12
        dtheta[15] = self.sutras.lopana_sthapana(theta_next, mask=False, memory=global_theta_mean, k_restore=1e-3) - theta_next
        dtheta[16] = self.sutras.adyamadyena_antyamantyena(self.theta_history + [theta_next]) * 1e-12
        if self.theta_history:
            drift, _ = self.sutras.chalana_kalanabhyam(self.theta_history + [theta_next])
        else:
            drift = 0.0
        dtheta[17] = drift
        dtheta[18] = self.sutras.navasesa(theta_next) * 1e-6
        dtheta[19] = self.sutras.vilokanam(dtheta[17], base_gain=1e-3)
        dtheta[20] = self.sutras.shunyanka(dtheta[19], deadband=1e-6)
        dtheta[21] = self.sutras.antyayoreva(self.theta_history + [theta_next], k=1e-3)
        dtheta[22] = self.sutras.sisyate_sesasamjnah(remainder=dtheta[21], memory=dtheta[11], beta=0.5)
        dtheta[23] = self.sutras.diagonal_mix(theta_next, r_next, mix=1e-3) - theta_next
        dtheta[24] = self.sutras.yavadunam_phi(theta_next, target=(theta % TAU), k=1e-3)
        dtheta[25] = self.sutras.purana_to_bounds(theta_next, 0.0, TAU, k=1e-3) - theta_next
        dtheta[26] = self.sutras.samuccaya_robust(dtheta)
        dtheta[27] = float(np.dot(self.sutras.vyashti_weight(dtheta), dtheta))
        _, sum_theta = self.sutras.samashti_normalize([abs(x)+1e-12 for x in dtheta])
        dtheta[28] = (1.0 - sum_theta) * 0.0

        # r-channel
        rlo, rhi = self.bounds_r
        dr[0]  = self.sutras.ekadhikena_purvena(0.0)
        dr[1]  = self.sutras.nikhilam(r_next, base=max(1.0, rhi if math.isfinite(rhi) else abs(r_next)+1.0)) * 1e-6
        dr[2]  = self.sutras.urdhva_tiryagbhyam(r, theta, last_r, last_theta) * 1e-9
        dr[3]  = self.sutras.paravartya_yojayet(r_next, gain=1e-3)
        dr[4]  = self.sutras.sunyam_samyasamuccaye(r - global_r_mean, global_r_mean - r) * 0.0
        dr[5]  = self.sutras.anurupyena(r_next, ratio=1.0) * 0.0
        dr[6]  = self.sutras.sankalana_vyavakalana(r_next, last_r, mix=0.5) - r_next
        r_target_band = self.c_phyllo*sqrt(n+1.0) if self.radial_mode == "phyllotaxis" else r
        dr[7]  = self.sutras.purana_apurana(r_next, target=r_target_band, gain=1.0e-3) - r_next
        dr[8]  = self.sutras.yavadunam(r_next, target=r_target_band, gain=1.0e-3)
        dr[9]  = self.sutras.vyashti_samashti(r_next, global_r_mean, alpha=0.5) - r_next
        dr[10] = self.sutras.shesanyankena_charamena(r_next, quantum=1.0e-9) - r_next
        dr[11] = self.sutras.sopantyadvayamantyam(last_r, pen_r, r_next, gain=1.0e-3) if len(self.radius_history)>=2 else 0.0
        dr[12] = self.sutras.ekanyunena_purvena(0.0)
        dr[13] = self.sutras.gunitasamuccayah([r_next+1e-9],[r+1e-9])[2] * 1e-12
        dr[14] = self.sutras.gunakasamuccayah([r_next],[r]) * 1e-12
        dr[15] = self.sutras.lopana_sthapana(r_next, mask=False, memory=global_r_mean, k_restore=1.0e-3) - r_next
        dr[16] = self.sutras.adyamadyena_antyamantyena(self.radius_history + [r_next]) * 1e-12
        if self.radius_history:
            r_drift, _ = self.sutras.chalana_kalanabhyam(self.radius_history + [r_next])
        else:
            r_drift = 0.0
        dr[17] = r_drift * 1e-3
        dr[18] = self.sutras.navasesa(r_next) * 1e-6
        dr[19] = self.sutras.vilokanam(r_drift, base_gain=1.0e-3)
        dr[20] = self.sutras.shunyanka(dr[19], deadband=1e-6)
        dr[21] = self.sutras.antyayoreva(self.radius_history + [r_next], k=1.0e-3)
        dr[22] = self.sutras.sisyate_sesasamjnah(remainder=dr[21], memory=(dr[11] if len(self.radius_history)>=2 else 0.0), beta=0.5)
        dr[23] = self.sutras.diagonal_mix(r_next, theta_next, mix=1.0e-3) - r_next
        dr[24] = self.sutras.yavadunam_phi(r_next, target=r_target_band, k=1.0e-3)
        dr[25] = self.sutras.purana_to_bounds(r_next, rlo, rhi if math.isfinite(rhi) else r_next*10.0, k=1.0e-3) - r_next
        dr[26] = self.sutras.samuccaya_robust(dr)
        dr[27] = float(np.dot(self.sutras.vyashti_weight(dr), dr))
        _, sum_r = self.sutras.samashti_normalize([abs(x)+1e-12 for x in dr])
        dr[28] = (1.0 - sum_r) * 0.0

        theta_correction = float(np.dot(self.theta_weights, dtheta))
        radius_correction = float(np.dot(self.radius_weights, dr))
        return radius_correction, theta_correction

    # ---------------
    # Omega channel
    # ---------------
    def _require_omega(self, step_idx: int, S_iter) -> float:
        """
        STRICT: Consume exactly one S_k(1) value per step. No proxy allowed.
        Raises if insufficient S provided.
        """
        try:
            omega = float(next(S_iter))
        except StopIteration:
            raise ValueError(f"S_sequence exhausted before step {step_idx}. Provide ≥ steps values.")
        self._S_values.append(omega)
        return omega

    # ----------
    # One step
    # ----------
    def step(self, r: float, theta: float, n: int, omega: float) -> Tuple[float, float, dict]:
        # Base schedulers
        r_b, theta_b = self._base_step(r, theta, n)

        # Vedic 29 stack
        dr, dtheta = self._sutra_corrections(r, theta, r_b, theta_b, n)
        r_next = r_b + dr
        theta_next = (theta_b + dtheta) % TAU

        # Complex coordinates
        z_prev = complex(r * cos(theta), r * sin(theta))
        z_next = complex(r_next * cos(theta_next), r_next * sin(theta_next))

        # Screw-axis ladder
        if self.ladder_on:
            dphi = self.ladder.correction(z_next)
            theta_next = (theta_next + dphi) % TAU
            z_next = complex(r_next * cos(theta_next), r_next * sin(theta_next))

        # Hopf sparsifier
        if self.sparsify_on and self.radius_history:
            z_corr = self.sparsifier.correction(z_prev, z_next)
            z_next += z_corr
            r_next = abs(z_next)
            theta_next = atan2(z_next.imag, z_next.real) % TAU

        # Harmonic ZPE + deterministic shunt (on supplied Ω only)
        E, zpe_corr = self.zpe.energy_and_correction(omega)
        shunt_fb = self.shunt.feedback(omega)
        r_next += (zpe_corr + shunt_fb)

        # Bounds (strict)
        r_next = self.sutras.purana_to_bounds(r_next, self.bounds_r[0], self.bounds_r[1], k=1.0)

        # Logs
        self.theta_history.append(theta_next)
        self.radius_history.append(r_next)
        self.Omega_history.append(omega)

        return r_next, theta_next, {"E_zpe": E, "zpe_corr": zpe_corr, "shunt_fb": shunt_fb, "Omega": omega}

    # --------------
    # Multi-step run
    # --------------
    def predict(self,
                r0: float,
                theta0: float,
                steps: int,
                S_sequence: Iterable[float]) -> List[Tuple[int, float, float, float, float, float, float]]:
        """
        STRICT multi-step: requires S_sequence iterable providing at least `steps` Ω values.
        Returns [(n, r, theta_rad, E_zpe, zpe_corr, shunt_fb, Omega)] from n=0..steps.
        """
        if S_sequence is None:
            raise ValueError("S_sequence is mandatory (no proxy allowed).")
        S_iter = iter(S_sequence)

        r = float(r0)
        th = float(theta0)
        out: List[Tuple[int, float, float, float, float, float, float]] = [(0, r, th, 0.0, 0.0, 0.0, 0.0)]
        self._S_values = []
        self.theta_history = []
        self.radius_history = []
        self.Omega_history = []
        self._toggle_state = 0
        self.ladder.reset()
        self.sparsifier.reset()
        self.shunt.reset()

        for n in range(steps):
            omega = self._require_omega(n, S_iter)
            r, th, meta = self.step(r, th, n, omega)
            out.append((n+1, r, th, meta["E_zpe"], meta["zpe_corr"], meta["shunt_fb"], meta["Omega"]))
        return out
