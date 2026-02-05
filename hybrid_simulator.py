#!/usr/bin/env python3
"""
Hybrid quantum-classical simulator orchestrating 29 Vedic sutras
in serial, concurrent, and parallel modes with FM8-style audio control.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from hc_ipc import HcIpcClient
from hypercube_fm8 import HyperCubeFM8
from sutra_repository import SutraContext, SutraMode, SutraRepository


def _optional_import(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


if _optional_import("qiskit") and _optional_import("qiskit_aer"):
    from qiskit_backend import execute_ghz
else:
    execute_ghz = None


PREFERRED_SUTRAS = [
    "ekadhikena_purvena",
    "nikhilam_navatashcaramam_dashatah",
    "urdhva_tiryagbhyam",
    "paravartya_yojayet",
    "shunyam_samyasamuccaye",
    "anurupyena",
    "sankalana_vyavakalanabhyam",
    "purna_apurna_bhyam",
    "chalana_kalana",
    "yavadunam",
    "vyashtisamanstih",
    "sesanyankena_caramena",
    "sopantyadvayamantyam",
    "ekanyunena_purvena",
    "gunitasamuccayah",
    "gunakasamuccayah",
    "anurupye_sunyamanyat",
    "sisyate_sesasamjnah",
    "adyamadyenantyamantyena",
    "antyayordasakepi",
    "antyayoreva",
    "yavadunam_tavadunikrtya",
    "samuccayagunitah",
    "ekadhikena",
    "paravartya",
    "sankalana_samanantara",
    "puranapuranabhyam",
    "vargamula",
    "gunita_samuccaya",
]


@dataclass
class SutraRunResult:
    name: str
    output: Any


@dataclass
class HybridRunSummary:
    mode: str
    results: List[SutraRunResult]
    final_value: float


def _prepare_args(func: Any, value: Any) -> list:
    import inspect

    sig = inspect.signature(func)
    args = []
    for name, param in sig.parameters.items():
        if name in {"self", "ctx"}:
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Parameter.empty:
                if any(key in name for key in ("coeff", "angles", "values", "parts", "list", "vector")):
                    args.append([value, value])
                else:
                    args.append(value)
    return args


def _is_candidate(name: str) -> bool:
    return (
        not name.startswith("_")
        and not name.endswith("_quantum")
        and not name.endswith("_hybrid")
        and not name.endswith("_classical")
    )


def select_sutra_names(repo: SutraRepository, target_count: int = 29) -> List[str]:
    available = repo.list_sutras()
    selected = [name for name in PREFERRED_SUTRAS if name in available]
    remaining = [name for name in available if name not in selected and _is_candidate(name)]
    for name in remaining:
        if len(selected) >= target_count:
            break
        selected.append(name)
    return selected[:target_count]


def _call_sutra_in_process(name: str, value: float, mode: SutraMode) -> Tuple[str, Any]:
    repo = SutraRepository(SutraContext(mode=mode, parallel=False))
    func = repo._methods[name]
    args = _prepare_args(func, value)
    output = repo.call_sutra(name, *args, ctx=repo.context)
    return name, output


def _to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.mean([_to_scalar(v) for v in value]))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_mod_matrix(size: int, base: float) -> List[List[float]]:
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            if i == j:
                row.append(0.0)
            else:
                row.append(base / (1.0 + abs(i - j)))
        matrix.append(row)
    return matrix


def _qiskit_input_matrix(num_ops: int) -> List[List[float]]:
    if execute_ghz is None:
        return [[0.0] for _ in range(num_ops)]
    counts = execute_ghz(num_qubits=29, shots=256)
    total = sum(counts.values()) if counts else 1
    dominant = max(counts.values()) if counts else 0
    ratio = dominant / total if total else 0.0
    return [[ratio * (i + 1) * 0.01] for i in range(num_ops)]


def run_serial(value: float, mode: SutraMode, sutra_names: Sequence[str]) -> HybridRunSummary:
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)
    results = []
    current = value
    for name in sutra_names:
        func = repo._methods[name]
        args = _prepare_args(func, current)
        current = repo.call_sutra(name, *args, ctx=ctx)
        results.append(SutraRunResult(name=name, output=current))
    return HybridRunSummary(mode="serial", results=results, final_value=_to_scalar(current))


def run_concurrent(value: float, mode: SutraMode, sutra_names: Sequence[str]) -> HybridRunSummary:
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)

    def run(name: str) -> SutraRunResult:
        func = repo._methods[name]
        args = _prepare_args(func, value)
        output = repo.call_sutra(name, *args, ctx=ctx)
        return SutraRunResult(name=name, output=output)

    results = []
    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(run, name) for name in sutra_names]
        for fut in futures:
            results.append(fut.result())
    final_value = _to_scalar([_to_scalar(r.output) for r in results])
    return HybridRunSummary(mode="concurrent", results=results, final_value=final_value)


def run_parallel(value: float, mode: SutraMode, sutra_names: Sequence[str]) -> HybridRunSummary:
    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(_call_sutra_in_process, name, value, mode) for name in sutra_names]
        for fut in futures:
            name, output = fut.result()
            results.append(SutraRunResult(name=name, output=output))
    final_value = _to_scalar([_to_scalar(r.output) for r in results])
    return HybridRunSummary(mode="parallel", results=results, final_value=final_value)


def apply_audio_updates(
    cube: HyperCubeFM8,
    results: Sequence[SutraRunResult],
    mix_mode: str,
    ipc: HcIpcClient,
) -> None:
    for result in results:
        cube.apply_sutra_to_operators(result.name, [_to_scalar(result.output)])
    cube.set_mix_mode(mix_mode)
    cube.set_modulation_matrix(_build_mod_matrix(cube.num_ops, base=0.03))
    cube.set_input_matrix(_qiskit_input_matrix(cube.num_ops))
    payload = cube.as_update_payload()
    ipc.send_state(
        payload["base_ops"],
        payload["levels"],
        mod_matrix=payload["mod_matrix"],
        input_matrix=payload["input_matrix"],
        mix_mode=payload["mix_mode"],
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Hybrid sutra simulator")
    parser.add_argument("value", type=float, help="Base input value")
    parser.add_argument(
        "--mode",
        choices=["classical", "quantum", "hybrid", "maya_illusion", "sulba"],
        default="hybrid",
        help="Execution mode",
    )
    parser.add_argument("--run-all-modes", action="store_true", help="Run serial+concurrent+parallel")
    parser.add_argument("--enable-audio", action="store_true", help="Send audio updates")
    args = parser.parse_args(argv)

    mode = SutraMode[args.mode.upper()]
    repo = SutraRepository(SutraContext(mode=mode))
    sutra_names = select_sutra_names(repo)

    audio_enabled = args.enable_audio or os.getenv("QUANQONSCIOUS_AUDIO", "0") == "1"
    ipc = HcIpcClient() if audio_enabled else None
    cube = HyperCubeFM8(num_ops=12, base_frequency=432.0)

    run_modes = ["serial", "concurrent", "parallel"] if args.run_all_modes else ["serial"]

    summaries = []
    for run_mode in run_modes:
        if run_mode == "serial":
            summary = run_serial(args.value, mode, sutra_names)
        elif run_mode == "concurrent":
            summary = run_concurrent(args.value, mode, sutra_names)
        else:
            summary = run_parallel(args.value, mode, sutra_names)
        summaries.append(summary)
        if ipc is not None:
            ipc.start()
            apply_audio_updates(cube, summary.results, run_mode, ipc)

    for summary in summaries:
        print(f"\n[{summary.mode.upper()}] Final value: {summary.final_value:.6f}")
        for result in summary.results:
            scalar = _to_scalar(result.output)
            print(f"  {result.name}: {scalar:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
