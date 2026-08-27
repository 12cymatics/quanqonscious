import numpy as np
from typing import List, Callable

class VedicSutraEngine:
    """Engine implementing all 29 Vedic sutras over NumPy float64.

    This is a deliberate float64 sidecar, outside the exact-ℚ path in
    ``vedic.kernel``. Using floats here is fine and intended.

    Domain policy: an input outside an operator's domain raises. Several
    operators used to substitute a value instead -- clamping a magnitude to
    1e-12, reporting a small real difference as exact zero, returning 0.0 for
    an array too short to have the quantity asked for, scaling by 1.0 when a
    divisor was zero. Each produced a finite, plausible number from input the
    operator could not actually evaluate, which is indistinguishable
    downstream from a real result. "Safe numerical domains", which this
    docstring used to claim, meant the caller never learned the domain was
    exceeded.
    """

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
        """Complement to the next power of ``base``. Requires x > 0.

        This read ``x_safe = np.maximum(np.abs(arr), 1e-12)``, which did two
        substitutions at once: ``np.abs`` silently mapped a negative input to
        its positive twin and returned a complement for a number the caller
        never passed, and the 1e-12 floor turned 0 -- and anything under
        1e-12 -- into 1e-12, whose complement is ``base**-11``, a confident
        answer to an undefined question. ``log`` is defined on x > 0, so
        that is the stated domain and anything else raises.
        """
        b = float(self.base if base is None else base)
        if not np.isfinite(b) or b <= 0.0 or b == 1.0:
            raise ValueError(
                f"nikhilam base must be finite, positive and != 1 "
                f"(log base is undefined otherwise); got {b!r}")
        arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("nikhilam input must be finite; got non-finite "
                             "values")
        if np.any(arr <= 0.0):
            bad = np.asarray(arr)[arr <= 0.0]
            raise ValueError(
                f"nikhilam is defined on strictly positive input (it takes "
                f"log(x)); got {bad.size} value(s) <= 0, e.g. "
                f"{float(bad.ravel()[0])!r}")
        nearest_base = np.power(b, np.ceil(np.log(arr) / np.log(b)))
        return nearest_base - arr

    def urdhva_tiryagbhyam(self, a, b):
        return np.matmul(a, b)

    def paravartya_yojayet(self, x, divisor=1.0):
        d = np.asarray(divisor, dtype=np.float64)
        return np.divide(x.T, d)

    def shunyam_samyasamuccaye(self, x, y):
        """Return sum(x) - sum(y). Exactly zero only when it is exactly zero.

        This read ``0.0 if np.isclose(sx, sy, atol=tol) else (sx - sy)``, so a
        real nonzero difference came back as exact 0.0 -- and via isclose's
        default rtol=1e-5, not merely at the advertised 1e-9: two sums near
        1e6 were called equal while differing by 10. The caller could not
        distinguish "the sums balance" from "they differ by less than a
        tolerance I did not choose and cannot see".

        The ``tol`` parameter is gone rather than kept and ignored. Deciding
        what difference counts as zero is the caller's judgement, made on the
        true difference this now returns.
        """
        sx = np.sum(x)
        sy = np.sum(y)
        if not (np.isfinite(sx) and np.isfinite(sy)):
            raise ValueError(
                f"shunyam_samyasamuccaye needs finite sums; got "
                f"sum(x)={sx!r}, sum(y)={sy!r}")
        return sx - sy

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
        """Return ``whole - sum(parts)``: the exact discrepancy, not a verdict.

        This read ``np.isclose(whole, np.sum(parts))``, which returned a Bool
        under numpy's default tolerances — ``rtol=1e-5``, so on values near
        1e6 a discrepancy of 10 was reported as "the parts sum to the whole".
        The caller got a verdict computed to a precision it never chose and
        could not see, and had no way to recover how far off the sum actually
        was.

        Returning the difference moves the decision to the caller and makes
        the exact-zero case exactly representable: ``result == 0`` is a real
        test of the identity, where ``isclose`` could not express one.
        """
        return np.asarray(whole, dtype=np.float64) - np.sum(parts)

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
        """First-squared plus last-squared. Requires at least two elements.

        This returned 0.0 when ``arr.size < 2``, which is a legitimate value
        of this quantity (any series whose first and last are both 0), so the
        caller could not tell it apart from a computed result. The identical
        precondition already raises in ``antyayordashakepi`` and
        ``antyayoreva``; the same precondition now gets the same treatment.
        """
        arr = np.asarray(series, dtype=np.float64)
        if arr.size < 2:
            raise ValueError(
                f"Input must have at least two elements; got size {arr.size}")
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
        if not np.all(np.isfinite(arr)):
            raise ValueError("antyayordashakepi input must be finite")
        denom = np.sum(arr[..., -2:], axis=-1, keepdims=True)
        # A zero denominator raises. This read
        # ``np.where(np.isclose(denom, 0.0), 1.0, 10.0 / denom)``, which
        # substituted a scale of 1.0 -- the identity -- so the row came back
        # unchanged and looked exactly like a row whose last two elements
        # already summed to 10. The sutra scales the last pair to sum to 10;
        # a pair summing to 0 cannot be scaled to sum to 10 by any factor,
        # and that is a fact about the input, not something to paper over.
        if np.any(denom == 0.0):
            n_bad = int(np.count_nonzero(denom == 0.0))
            raise ValueError(
                f"antyayordashakepi scales the last two elements to sum to "
                f"10, which is undefined when they sum to 0: {n_bad} "
                f"row(s) have a zero last-pair sum")
        arr[..., -2:] = arr[..., -2:] * (10.0 / denom)
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
