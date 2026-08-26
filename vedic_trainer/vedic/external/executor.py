"""Serial / threads / processes orchestration for the chainable sutra pipeline.

Adapted from ``codex/locate-runnable-simulations-in-repos:src/quanqonscious/sutra_executor.py``.
The original ``run_full_engine`` dependency is replaced with a sequential
application of the 22 ``vedic.kernel.z2_primitives`` operators that are
Q16 → Q16 (the exact-ℚ Z₂⁴ algebra). This is **not** all 29 sutras: the
other 7 have signatures that cannot be chained, and are listed with their
reasons at ``_PIPELINE`` below. The output type is ``Q16`` (tuple of
Fraction).

Process pool execution requires the worker function to be importable at
module top level — ``_run_unary_pipeline_q`` satisfies that contract.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List

from vedic.kernel.q import Q16
from vedic.kernel.z2_primitives import (
    s1_eka_adhikena,
    s2_nikhilam,
    s4_paravartya,
    s5_shunyam_samya,
    s6_anurupya_shunyam,
    s8_puranapuranabhyam_fill,
    s9_chalana_kalanabhyam,
    s10_yavadunam_tavadunikrtya,
    s11_vyasti_samasti,
    s12_shesanyankena_charamena,
    s13_sopantyadvayamantyam_last2,
    s14_ekanyunena_purvena,
    s15_gunitasamucchaya_product,
    s16_gunaka_samucchaya,
    s19_lopana_sthapanabhyam,
    s20_vilokanam_spect,
    s21_dhvajanka_flag,
    s24_kevalaih_saptakam,
    s25_vestana_circular,
    s26_yavadunam_square,
    s28_lopana_restore,
    s29_mean_drive,
)


class ExecutionMode(str, Enum):
    SERIAL = "serial"
    THREADS = "threads"
    PROCESSES = "processes"


# Ordered pipeline of the 22 operators whose signature is Q16 -> Q16, so each
# one's output can feed the next. The other 7 of the 29 are excluded because
# their signatures do not compose, for three distinct reasons:
#
#   binary -- needs a second Q16 operand, which the pipeline has no source for:
#       S3  s3_urdhva_tiryak(psi, phi)
#       S17 s17_anurupyena_proportion(psi, phi, ref)
#       S23 s23_dwandwa_yoga(psi, phi, mask)
#   scalar output -- returns a single Fraction, not a vector:
#       S18 s18_adyamadyena_antyamantyena(psi, i, j) -> Fraction
#       S27 s27_samuccaya_gunitah(psi) -> Fraction
#   wrong output shape -- unary, but the result is not a Q16:
#       S7  s7_sankalana_vyavakalana(psi, mask) -> (Q16, Q16), a pair
#       S22 s22_parity_complement(psi, mask) -> length-8 tuple, not length 16
#
# An earlier version of this comment named only the first two groups (5 of the
# 7) and left S7 and S22 unaccounted for. The bit-exact gate and the
# simulator-match test cover all 29 operators separately.
_PIPELINE = (
    s1_eka_adhikena,
    s2_nikhilam,
    s4_paravartya,
    s5_shunyam_samya,
    s6_anurupya_shunyam,
    s8_puranapuranabhyam_fill,
    s9_chalana_kalanabhyam,
    s10_yavadunam_tavadunikrtya,
    s11_vyasti_samasti,
    s12_shesanyankena_charamena,
    s13_sopantyadvayamantyam_last2,
    s14_ekanyunena_purvena,
    s15_gunitasamucchaya_product,
    s16_gunaka_samucchaya,
    s19_lopana_sthapanabhyam,
    s20_vilokanam_spect,
    s21_dhvajanka_flag,
    s24_kevalaih_saptakam,
    s25_vestana_circular,
    s26_yavadunam_square,
    s28_lopana_restore,
    s29_mean_drive,
)


def _run_unary_pipeline_q(psi: Q16) -> Q16:
    """Apply the 22 chainable Q16 -> Q16 operators of ``_PIPELINE`` in order.

    Not the full 29-sutra engine — see ``_PIPELINE`` for the 7 exclusions.
    Top-level so it can be pickled by ``ProcessPoolExecutor``.
    """
    out = psi
    for op in _PIPELINE:
        out = op(out)
    return out


@dataclass
class SutraExecutor:
    """Apply the 22-operator chainable ℚ-exact pipeline across many inputs."""

    mode: ExecutionMode = ExecutionMode.SERIAL
    max_workers: int | None = None

    def execute(self, inputs: Iterable[Q16]) -> List[Q16]:
        data = list(inputs)
        if self.mode is ExecutionMode.SERIAL:
            return [_run_unary_pipeline_q(p) for p in data]
        if self.mode is ExecutionMode.THREADS:
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                return list(exe.map(_run_unary_pipeline_q, data))
        if self.mode is ExecutionMode.PROCESSES:
            with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
                return list(exe.map(_run_unary_pipeline_q, data))
        raise ValueError(f"Unsupported execution mode: {self.mode}")
