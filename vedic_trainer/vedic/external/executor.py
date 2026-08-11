"""Serial / threads / processes orchestration for running the full sutra pipeline.

Adapted from ``codex/locate-runnable-simulations-in-repos:src/quanqonscious/sutra_executor.py``.
The original ``run_full_engine`` dependency is replaced with a sequential
application of every operator in ``vedic.kernel.sutras_exact`` (the
exact-ℚ Z₂⁴ algebra). The output type is ``Q16`` (tuple of Fraction).

Process pool execution requires the worker function to be importable at
module top level — ``_run_full_engine_q`` satisfies that contract.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List

from vedic.kernel.q import Q16
from vedic.kernel.sutras_exact import (
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


# Ordered pipeline of unary Q16 → Q16 operators. Binary operators (S3, S17,
# S23) and the two scalar-output operators (S18, S27) are excluded so the
# pipeline is type-uniform; the bit-exact gate and the simulator-match test
# cover those separately.
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


def _run_full_engine_q(psi: Q16) -> Q16:
    """Apply every unary operator in ``_PIPELINE`` in order.

    Top-level so it can be pickled by ``ProcessPoolExecutor``.
    """
    out = psi
    for op in _PIPELINE:
        out = op(out)
    return out


@dataclass
class SutraExecutor:
    """Apply the full unary ℚ-exact pipeline across many inputs."""

    mode: ExecutionMode = ExecutionMode.SERIAL
    max_workers: int | None = None

    def execute(self, inputs: Iterable[Q16]) -> List[Q16]:
        data = list(inputs)
        if self.mode is ExecutionMode.SERIAL:
            return [_run_full_engine_q(p) for p in data]
        if self.mode is ExecutionMode.THREADS:
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                return list(exe.map(_run_full_engine_q, data))
        if self.mode is ExecutionMode.PROCESSES:
            with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
                return list(exe.map(_run_full_engine_q, data))
        raise ValueError(f"Unsupported execution mode: {self.mode}")
