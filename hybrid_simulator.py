#!/usr/bin/env python3
"""
Hybrid quantum-classical simulator orchestrating 29 Vedic sutras
in serial, concurrent, and parallel modes with FM8-style audio control.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple
import inspect

import numpy as np

from qiskit_backend import execute_ghz
from sutra_repository import SutraContext, SutraMode, SutraRepository

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


def compute_palindromic_alloy() -> Dict[str, Any]:
    """Compute Λ_pal with exact rational arithmetic from integer sutra coefficients."""
    lucas = (2, 1, 3, 4, 7, 11, 18, 29)
    lucas_total = sum(lucas)

    def d_k(k: int) -> int:
        return (k % 4) + 2

    def s_k_at_one(k: int) -> int:
        degree = d_k(k)
        total = 0
        for i in range(degree + 1):
            sign = -1 if ((i * k) % 2) else 1
            total += sign * math.comb(k + degree, i)
        return total

    terms: List[Dict[str, Any]] = []
    weighted_sum = Fraction(0, 1)
    for idx, lucas_k in enumerate(lucas, start=1):
        s_left = s_k_at_one(idx)
        s_right = s_k_at_one(17 - idx)
        pair_sum = s_left + s_right
        weight = Fraction(lucas_k, lucas_total)
        term_value = weight * pair_sum
        weighted_sum += term_value
        terms.append(
            {
                "k": idx,
                "lucas": lucas_k,
                "weight_fraction": f"{weight.numerator}/{weight.denominator}",
                "s_k_1": s_left,
                "s_17_minus_k_1": s_right,
                "pair_sum": pair_sum,
                "term_fraction": f"{term_value.numerator}/{term_value.denominator}",
            }
        )

    return {
        "fraction": f"{weighted_sum.numerator}/{weighted_sum.denominator}",
        "decimal": float(weighted_sum),
        "lucas_total": lucas_total,
        "terms": terms,
    }


def _arg_value_for_param(name: str, value: Any) -> Any:
    lowered = name.lower()
    if any(key in lowered for key in ("coeff", "angles", "values", "parts", "list", "vector")):
        return [value, value]
    if any(key in lowered for key in ("denominator", "divisor", "modulus", "base")):
        return 1 if value == 0 else value
    if any(key in lowered for key in ("count", "steps", "degree", "order", "index")):
        try:
            parsed = int(abs(float(value)))
            return parsed if parsed > 0 else 1
        except (TypeError, ValueError):
            return 1
    return value


def _prepare_call(func: Any, value: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    sig = inspect.signature(func)
    args = []
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in {"self", "ctx"}:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        generated = _arg_value_for_param(name, value)
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(generated)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = generated
    if not args and not kwargs:
        args = [value]
    return tuple(args), kwargs


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
    args, kwargs = _prepare_call(func, value)
    output = repo.call_sutra(name, *args, ctx=repo.context, **kwargs)
    return name, output


def _to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if np is not None and isinstance(value, np.ndarray):
        return float(np.mean(value))
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(mean(_to_scalar(v) for v in value))
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
        args, kwargs = _prepare_call(func, current)
        current = repo.call_sutra(name, *args, ctx=ctx, **kwargs)
        results.append(SutraRunResult(name=name, output=current))
    return HybridRunSummary(mode="serial", results=results, final_value=_to_scalar(current))


def run_concurrent(value: float, mode: SutraMode, sutra_names: Sequence[str]) -> HybridRunSummary:
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)

    def run(name: str) -> SutraRunResult:
        func = repo._methods[name]
        args, kwargs = _prepare_call(func, value)
        output = repo.call_sutra(name, *args, ctx=ctx, **kwargs)
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


def apply_audio_updates(cube: Any, results: Sequence[SutraRunResult], mix_mode: str, ipc: Any) -> None:
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


def _called_repo_files(repo: SutraRepository, sutra_names: Sequence[str]) -> List[str]:
    files = set()
    for name in sutra_names:
        func = repo._methods.get(name)
        if func is None:
            continue
        source = inspect.getsourcefile(func)
        if source:
            files.add(os.path.relpath(source, os.getcwd()))
    return sorted(files)


def _summary_dict(summary: HybridRunSummary) -> Dict[str, Any]:
    return {
        "mode": summary.mode,
        "final_value": summary.final_value,
        "results": [{"name": r.name, "scalar_output": _to_scalar(r.output)} for r in summary.results],
    }


def _sanitize_ipykernel_value(raw: Any) -> float:
    if raw is None:
        return 1.0
    text = str(raw).strip()
    if text.endswith(".json") and "jupyter/runtime/kernel-" in text:
        return 1.0
    return float(text)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Hybrid sutra simulator")
    parser.add_argument("value", nargs="?", default="1.0", help="Base input value")
    parser.add_argument(
        "--mode",
        choices=["classical", "quantum", "hybrid", "maya_illusion", "sulba"],
        default="hybrid",
        help="Execution mode",
    )
    parser.add_argument("--run-all-modes", action="store_true", help="Run serial+concurrent+parallel")
    parser.add_argument("--enable-audio", action="store_true", help="Send audio updates")
    parser.add_argument("--sutra-count", type=int, default=29, help="Number of sutras to execute")
    parser.add_argument(
        "--inject-palindromic-alloy",
        action="store_true",
        help="Inject the computed Λ_pal value into each run-mode final scalar",
    )
    parser.add_argument(
        "--report-path",
        default="runs/hybrid_run_report.json",
        help="Optional output JSON report path",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        filtered_unknown = []
        skip_next = False
        for token in unknown:
            if skip_next:
                skip_next = False
                continue
            if token == "-f":
                skip_next = True
                continue
            if token.endswith(".json") and "jupyter/runtime/kernel-" in token:
                continue
            filtered_unknown.append(token)
        if filtered_unknown:
            parser.error(f"Unrecognized arguments: {' '.join(filtered_unknown)}")

    input_value = _sanitize_ipykernel_value(args.value)
    mode = SutraMode[args.mode.upper()]
    repo = SutraRepository(SutraContext(mode=mode))
    sutra_names = select_sutra_names(repo, target_count=args.sutra_count)
    if len(sutra_names) < args.sutra_count:
        raise RuntimeError(f"Requested {args.sutra_count} sutras but found {len(sutra_names)}")

    audio_enabled = args.enable_audio or os.getenv("QUANQONSCIOUS_AUDIO", "0") == "1"
    ipc = None
    cube = None
    if audio_enabled:
        from hc_ipc import HcIpcClient
        from hypercube_fm8 import HyperCubeFM8

        ipc = HcIpcClient()
        cube = HyperCubeFM8(num_ops=12, base_frequency=432.0)

    run_modes = ["serial", "concurrent", "parallel"] if args.run_all_modes else ["serial"]

    summaries = []
    alloy_payload = compute_palindromic_alloy()
    alloy_decimal = alloy_payload["decimal"]
    for run_mode in run_modes:
        if run_mode == "serial":
            summary = run_serial(input_value, mode, sutra_names)
        elif run_mode == "concurrent":
            summary = run_concurrent(input_value, mode, sutra_names)
        else:
            summary = run_parallel(input_value, mode, sutra_names)
        if args.inject_palindromic_alloy:
            summary.final_value += alloy_decimal
        summaries.append(summary)
        if ipc is not None:
            ipc.start()
            apply_audio_updates(cube, summary.results, run_mode, ipc)

    for summary in summaries:
        print(f"\n[{summary.mode.upper()}] Final value: {summary.final_value:.6f}")
        for result in summary.results:
            scalar = _to_scalar(result.output)
            print(f"  {result.name}: {scalar:.6f}")

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    report_payload = {
        "input_value": input_value,
        "mode": args.mode,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sutra_count": args.sutra_count,
        "run_modes": run_modes,
        "sutra_names": list(sutra_names),
        "repo_files_called": _called_repo_files(repo, sutra_names),
        "palindromic_alloy": alloy_payload,
        "palindromic_alloy_injected": args.inject_palindromic_alloy,
        "summaries": [_summary_dict(summary) for summary in summaries],
    }
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nReport written to {args.report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
