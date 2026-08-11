import numpy as np
from typing import List, Callable

class VedicSutraEngine:
    """Engine implementing all 29 Vedic sutras with safe numerical domains."""

    def __init__(self, base: float = 10.0):
        self.base = float(base)
        # Primary sutras (16)
        self.sutras: List[Callable] = [
            self.ekadhikena_purvena,
            self.nikhilam_navatashcaramam_dashatah,
            self.urdhva_tiryagbhyam,
            self.paravartya_yojayet,
            self.shunyam_samyasamuccaye,
            self.anurupye_sunyamanyat,
            self.sankalana_vyavakalanabhyam,
            self.puranapuranabyham,
            self.chalana_kalanabyham,
            self.yavadunam,
            self.vyashtisamanstih,
            self.shesanyankena_charamena,
            self.sopaantyadvayamantyam,
            self.ekanyunena_purvena,
            self.gunitasamuchyah,
            self.gunakasamuchyah,
        ]
        # Sub-sutras (13)
        self.sub_sutras: List[Callable] = [
            self.anurupyena,
            self.sisyate_sesasamjnah,
            self.adyamadyenantyamantyena,
            self.kevalaih_saptakam_gunyat,
            self.vestanam,
            self.yavadunam_tavadunam,
            self.yavadunam_tavadunikritya,
            self.antyayordashakepi,
            self.antyayoreva,
            self.samuccayagunitah,
            self.lopanasthapanabhyam,
            self.vilokanam,
            self.gunitasamuccayah_samuccayagunitah,
        ]

    # -------- Primary sutras --------
    def ekadhikena_purvena(self, x):
        return np.asarray(x, dtype=np.float64) + 1.0

    def nikhilam_navatashcaramam_dashatah(self, x, base=None):
        b = float(self.base if base is None else base)
        arr = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(np.abs(arr), 1e-12)
        nearest_base = np.power(b, np.ceil(np.log(x_safe) / np.log(b)))
        return nearest_base - x_safe

    def urdhva_tiryagbhyam(self, a, b):
        return np.matmul(a, b)

    def paravartya_yojayet(self, x, divisor=1.0):
        d = np.asarray(divisor, dtype=np.float64)
        return np.divide(x.T, d)

    def shunyam_samyasamuccaye(self, x, y, tol=1e-9):
        sx = np.sum(x)
        sy = np.sum(y)
        return 0.0 if np.isclose(sx, sy, atol=tol) else (sx - sy)

    def anurupye_sunyamanyat(self, x, y, ratio):
        return x - ratio * y

    def sankalana_vyavakalanabhyam(self, a, b):
        return a + b, a - b

    def puranapuranabyham(self, x, complement_base=None):
        b = float(self.base if complement_base is None else complement_base)
        return b - x

    def chalana_kalanabyham(self, x, steps=1, direction=1):
        return np.roll(x, shift=steps * direction, axis=-1)

    def yavadunam(self, x, deficit):
        return x - deficit

    def vyashtisamanstih(self, whole, parts):
        return np.isclose(whole, np.sum(parts))

    def shesanyankena_charamena(self, coeffs, m):
        last_digit = np.mod(m, 10)
        return np.polyval(coeffs, last_digit)

    def sopaantyadvayamantyam(self, x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.size >= 2:
            arr[-1] = arr[-1] + arr[-2]
        return arr

    def ekanyunena_purvena(self, x):
        return np.asarray(x, dtype=np.float64) - 1.0

    def gunitasamuchyah(self, x, y):
        return np.sum(x) * np.sum(y)

    def gunakasamuchyah(self, arrays):
        sums = [np.sum(np.asarray(a, dtype=np.float64)) for a in arrays]
        prod = 1.0
        for s in sums:
            prod *= s
        return prod

    # -------- Sub-sutras --------
    def anurupyena(self, x, y, ratio):
        return x + ratio * y

    def sisyate_sesasamjnah(self, x, modulus):
        return np.mod(x, modulus)

    def adyamadyenantyamantyena(self, series):
        arr = np.asarray(series, dtype=np.float64)
        if arr.size < 2:
            return 0.0
        return arr[0] * arr[0] + arr[-1] * arr[-1]

    def kevalaih_saptakam_gunyat(self, x):
        return np.asarray(x, dtype=np.float64) * 7.0

    def vestanam(self, v):
        return np.cumsum(np.sort(np.asarray(v, dtype=np.float64)))

    def yavadunam_tavadunam(self, x, y):
        return (y - x) * x

    def yavadunam_tavadunikritya(self, x, base):
        return (base - x) * (base - x)

    def antyayordashakepi(self, x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] < 2:
            raise ValueError("Input must have at least two elements")
        denom = np.sum(arr[..., -2:], axis=-1, keepdims=True)
        scale = np.where(np.isclose(denom, 0.0), 1.0, 10.0 / denom)
        arr[..., -2:] = arr[..., -2:] * scale
        return arr

    def antyayoreva(self, x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] < 2:
            raise ValueError("Input must have at least two elements")
        return arr[..., -1] - arr[..., -2]

    def samuccayagunitah(self, arrays):
        sums = [np.sum(np.asarray(a, dtype=np.float64)) for a in arrays]
        prod = 1.0
        for s in sums:
            prod *= s
        return prod

    def lopanasthapanabhyam(self, x, eliminate_index, substitute_value):
        arr = np.asarray(x, dtype=np.float64).copy()
        arr[..., eliminate_index] = substitute_value
        return arr

    def vilokanam(self, x):
        return np.fft.fft2(np.asarray(x, dtype=np.complex128))

    def gunitasamuccayah_samuccayagunitah(self, x, y):
        sum_prod = np.sum(x) * np.sum(y)
        prod_sum = np.sum(x * y)
        return sum_prod, prod_sum
