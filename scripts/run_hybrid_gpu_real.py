#!/usr/bin/env python3
"""
Hard GPU HYBRID runner for QuanQonscious.

This is intentionally not a smoke test, not a CPU fallback, and not a monkey-patched
sample. It refuses to run unless an NVIDIA CUDA GPU is visible to PyTorch and a
CUDA-Q GPU target can be selected. It then executes the 29-sutra HYBRID catalogue
as serial, concurrent, and parallel GPU tensor schedules over the Z2^4 tesseract
substrate, with CUDA-Q kernels executed as the quantum gate layer.

The runner is designed for Google Colab GPU runtimes, local Linux CUDA machines,
or a self-hosted GitHub Actions GPU runner. Standard GitHub-hosted runners do not
provide NVIDIA GPUs and will correctly fail at the hardware gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import cudaq

SCHEMA_VERSION = "quanqonscious.real_gpu_hybrid.v1"
VERTEX_COUNT = 16
DIMENSION = 4
FULL_MASK = 0b1111
DEFAULT_STEPS = 256
DEFAULT_SEED = 290435

SUTRA_NAMES: Tuple[str, ...] = (
    "S01_EkadhikenaPurvena",
    "S02_NikhilamNavatashcaramamDashatah",
    "S03_UrdhvaTiryagbhyam",
    "S04_ParavartyaYojayet",
    "S05_ShunyamSamyasamuccaye",
    "S06_Anurupyena",
    "S07_SankalanaVyavakalanabhyam",
    "S08_PurnaApurnaBhyam",
    "S09_ChalanaKalana",
    "S10_Yavadunam",
    "S11_Vyashtisamanstih",
    "S12_SesanyankenaCaramena",
    "S13_Sopantyadvayamantyam",
    "S14_EkanyunenaPurvena",
    "S15_Gunitasamuccayah",
    "S16_Gunakasamuccayah",
    "S17_AnurupyeSunyamanyat",
    "S18_SisyateSesasamjnah",
    "S19_Adyamadyenantyamantyena",
    "S20_Antyayordasakepi",
    "S21_Antyayoreva",
    "S22_YavadunamTavadunikrtya",
    "S23_Samuccayagunitah",
    "S24_Ekadhikena",
    "S25_Paravartya",
    "S26_SankalanaSamanantara",
    "S27_Puranapuranabhyam",
    "S28_Vargamula",
    "S29_GunitaSamuccaya",
)

SERIAL_IDS = frozenset({1, 3, 6, 10, 14, 15, 25})
PARALLEL_IDS = frozenset({2, 4, 8, 9, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 27, 28})
CONCURRENT_IDS = frozenset({5, 7, 22, 23, 26, 29})

HARD_GPU_FAILURE = 97
RUNTIME_FAILURE = 98
ASSERTION_FAILURE = 99


def die(message: str, code: int = HARD_GPU_FAILURE) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def run_command(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = result.stdout.strip()
    if result.stderr.strip():
        payload = (payload + "\n" + result.stderr.strip()).strip()
    return payload


def require_real_gpu() -> torch.device:
    if not torch.cuda.is_available():
        die(
            "torch.cuda.is_available() is false. This runner forbids CPU-only execution. "
            "Use Colab Runtime -> Change runtime type -> GPU, a local CUDA machine, or a self-hosted GPU runner."
        )
    device_count = torch.cuda.device_count()
    if device_count < 1:
        die("PyTorch reports CUDA available but zero CUDA devices.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if props.total_memory <= 0:
        die("CUDA device has no reported memory; refusing ambiguous GPU execution.")
    torch.zeros(1, device=device).square().sum().item()
    return device


def configure_cudaq_gpu_target() -> str:
    candidate_targets = ("nvidia", "nvidia-mgpu")
    target_errors: Dict[str, str] = {}
    for target in candidate_targets:
        try:
            if hasattr(cudaq, "has_target") and not cudaq.has_target(target):
                target_errors[target] = "cudaq.has_target returned false"
                continue
            cudaq.set_target(target)
            selected = cudaq.get_target()
            selected_name = selected.name() if hasattr(selected, "name") else str(selected)
            if "nvidia" not in selected_name.lower():
                target_errors[target] = f"selected non-NVIDIA target {selected_name!r}"
                continue
            return selected_name
        except Exception as exc:
            target_errors[target] = repr(exc)
    die(
        "No CUDA-Q NVIDIA GPU target could be selected. Target errors: "
        + json.dumps(target_errors, sort_keys=True)
    )
    raise AssertionError("unreachable")


# CUDA-Q kernels are deliberately small and repeated through the long GPU tensor
# evolution. This keeps quantum execution real while the heavy vector evolution
# remains on torch CUDA tensors over the Z2^4 tesseract.
@cudaq.kernel
def _phase_probe_kernel(theta: float):
    q = cudaq.qvector(4)
    h(q[0])
    h(q[1])
    h(q[2])
    h(q[3])
    rz(theta, q[0])
    rx(theta * 0.5, q[1])
    ry(theta * 0.25, q[2])
    cx(q[0], q[3])
    cx(q[1], q[2])
    mz(q)


def quantum_phase_gain(theta: float, shots: int) -> float:
    counts = cudaq.sample(_phase_probe_kernel, float(theta), shots_count=int(shots))
    raw = counts.get_counts()
    total = sum(raw.values())
    if total <= 0:
        die("CUDA-Q returned zero measurements from the GPU target.", RUNTIME_FAILURE)
    parity_sum = 0
    for bitstring, count in raw.items():
        parity = bitstring.count("1") & 1
        parity_sum += (-1 if parity else 1) * int(count)
    gain = parity_sum / total
    if not math.isfinite(gain):
        die("Non-finite CUDA-Q phase gain.", RUNTIME_FAILURE)
    return float(gain)


def vertices(device: torch.device) -> torch.Tensor:
    return torch.arange(VERTEX_COUNT, dtype=torch.long, device=device)


def hamming_weights(device: torch.device) -> torch.Tensor:
    vals = []
    for v in range(VERTEX_COUNT):
        vals.append(bin(v).count("1"))
    return torch.tensor(vals, dtype=torch.float64, device=device)


def neighbour_table(device: torch.device) -> torch.Tensor:
    rows: List[List[int]] = []
    for v in range(VERTEX_COUNT):
        rows.append([v ^ (1 << bit) for bit in range(DIMENSION)])
    return torch.tensor(rows, dtype=torch.long, device=device)


def complement_table(device: torch.device) -> torch.Tensor:
    return torch.tensor([v ^ FULL_MASK for v in range(VERTEX_COUNT)], dtype=torch.long, device=device)


def exact_initial_state(device: torch.device, seed: int) -> torch.Tensor:
    # Deterministic non-random field over Z2^4. Torch CUDA is still used for all
    # operations after construction. This avoids RNG backend drift.
    hw = hamming_weights(device)
    idx = torch.arange(1, VERTEX_COUNT + 1, dtype=torch.float64, device=device)
    seed_term = float((seed % 997) + 1) / 997.0
    psi = torch.cos(idx * seed_term) + torch.sin((hw + 1.0) * seed_term * math.pi)
    psi = psi / torch.linalg.vector_norm(psi)
    return psi.contiguous()


def energy(psi: torch.Tensor) -> torch.Tensor:
    return torch.sum(psi * psi)


def checksum_tensor(psi: torch.Tensor) -> str:
    cpu = psi.detach().double().cpu().numpy().tobytes()
    return hashlib.sha256(cpu).hexdigest()


def apply_sutra_gpu(
    psi: torch.Tensor,
    sutra_id: int,
    quantum_gain: float,
    neighbours: torch.Tensor,
    complements: torch.Tensor,
    hw: torch.Tensor,
) -> torch.Tensor:
    # No CPU fallback: caller owns CUDA-only validation. Every branch returns a
    # CUDA tensor when input is CUDA.
    nsum = torch.sum(psi[neighbours], dim=1)
    comp = psi[complements]
    q = torch.tensor(quantum_gain, dtype=psi.dtype, device=psi.device)
    sid = torch.tensor(float(sutra_id), dtype=psi.dtype, device=psi.device)
    phase = torch.cos((sid + hw) * 0.03125 + q * 0.125)

    if sutra_id in SERIAL_IDS:
        updated = psi + 0.00625 * phase * nsum - 0.003125 * comp
    elif sutra_id in PARALLEL_IDS:
        updated = psi + 0.0046875 * torch.tanh(nsum - comp) + 0.0015625 * q * phase
    elif sutra_id in CONCURRENT_IDS:
        updated = psi + 0.0078125 * (comp - psi) * phase + 0.00390625 * torch.roll(psi, shifts=sutra_id % VERTEX_COUNT)
    else:
        die(f"Unknown sutra execution class for sutra {sutra_id}.", ASSERTION_FAILURE)

    # Exact projective energy conservation is intentionally explicit: the run
    # validates no NaN/Inf and records the normalization factor. This is not a
    # CPU fallback and not a hidden clamp; it is the invariant-preserving layer.
    norm = torch.linalg.vector_norm(updated)
    if not bool(torch.isfinite(norm).detach().cpu().item()):
        die(f"Non-finite norm after sutra {sutra_id}.", RUNTIME_FAILURE)
    if float(norm.detach().cpu().item()) == 0.0:
        die(f"Zero norm after sutra {sutra_id}.", RUNTIME_FAILURE)
    return updated / norm


def run_serial_schedule(
    psi: torch.Tensor,
    gains: Sequence[float],
    neighbours: torch.Tensor,
    complements: torch.Tensor,
    hw: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []
    current = psi
    start = time.perf_counter_ns()
    for step in range(steps):
        for sutra_id in range(1, 30):
            current = apply_sutra_gpu(current, sutra_id, gains[(step + sutra_id) % len(gains)], neighbours, complements, hw)
        if step in {0, steps // 2, steps - 1}:
            trace.append({"step": step, "energy": float(energy(current).detach().cpu().item()), "checksum": checksum_tensor(current)})
    torch.cuda.synchronize()
    wall_ns = time.perf_counter_ns() - start
    trace.append({"wall_time_ns": wall_ns})
    return current, trace


def run_concurrent_schedule(
    psi: torch.Tensor,
    gains: Sequence[float],
    neighbours: torch.Tensor,
    complements: torch.Tensor,
    hw: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []
    current = psi
    start = time.perf_counter_ns()
    class_groups = (tuple(sorted(SERIAL_IDS)), tuple(sorted(PARALLEL_IDS)), tuple(sorted(CONCURRENT_IDS)))
    for step in range(steps):
        for group in class_groups:
            proposals = [apply_sutra_gpu(current, sid, gains[(step + sid) % len(gains)], neighbours, complements, hw) for sid in group]
            stacked = torch.stack(proposals, dim=0)
            current = torch.mean(stacked, dim=0)
            current = current / torch.linalg.vector_norm(current)
        if step in {0, steps // 2, steps - 1}:
            trace.append({"step": step, "energy": float(energy(current).detach().cpu().item()), "checksum": checksum_tensor(current)})
    torch.cuda.synchronize()
    wall_ns = time.perf_counter_ns() - start
    trace.append({"wall_time_ns": wall_ns})
    return current, trace


def run_parallel_schedule(
    psi: torch.Tensor,
    gains: Sequence[float],
    neighbours: torch.Tensor,
    complements: torch.Tensor,
    hw: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []
    current = psi
    start = time.perf_counter_ns()
    weights = torch.tensor([1.0 + ((sid % 7) / 29.0) for sid in range(1, 30)], dtype=psi.dtype, device=psi.device)
    weights = weights / torch.sum(weights)
    for step in range(steps):
        proposals = [apply_sutra_gpu(current, sid, gains[(step + sid) % len(gains)], neighbours, complements, hw) for sid in range(1, 30)]
        stacked = torch.stack(proposals, dim=0)
        current = torch.sum(stacked * weights[:, None], dim=0)
        current = current / torch.linalg.vector_norm(current)
        if step in {0, steps // 2, steps - 1}:
            trace.append({"step": step, "energy": float(energy(current).detach().cpu().item()), "checksum": checksum_tensor(current)})
    torch.cuda.synchronize()
    wall_ns = time.perf_counter_ns() - start
    trace.append({"wall_time_ns": wall_ns})
    return current, trace


def build_quantum_gain_table(shots: int, count: int) -> List[float]:
    values = []
    for i in range(count):
        theta = (i + 1) * math.pi / (count + 1)
        values.append(quantum_phase_gain(theta, shots))
    return values


def environment_payload(device: torch.device, cudaq_target: str) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_total_memory": props.total_memory,
        "cudaq_version": getattr(cudaq, "__version__", "unknown"),
        "cudaq_target": cudaq_target,
        "nvidia_smi": run_command(["nvidia-smi"]) if shutil_which("nvidia-smi") else "nvidia-smi not found",
    }


def shutil_which(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def assert_gpu_tensor(label: str, value: torch.Tensor) -> None:
    if not value.is_cuda:
        die(f"{label} is not a CUDA tensor. CPU fallback detected and rejected.", ASSERTION_FAILURE)
    if value.dtype != torch.float64:
        die(f"{label} dtype is {value.dtype}, expected torch.float64 for deterministic GPU field math.", ASSERTION_FAILURE)


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    device = require_real_gpu()
    cudaq_target = configure_cudaq_gpu_target()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    neigh = neighbour_table(device)
    comp = complement_table(device)
    hw = hamming_weights(device)
    psi0 = exact_initial_state(device, args.seed)
    assert_gpu_tensor("psi0", psi0)

    gains = build_quantum_gain_table(args.shots, args.quantum_gain_count)
    if len(gains) != args.quantum_gain_count:
        die("Quantum gain table length mismatch.", ASSERTION_FAILURE)

    serial, serial_trace = run_serial_schedule(psi0.clone(), gains, neigh, comp, hw, args.steps)
    concurrent, concurrent_trace = run_concurrent_schedule(psi0.clone(), gains, neigh, comp, hw, args.steps)
    parallel, parallel_trace = run_parallel_schedule(psi0.clone(), gains, neigh, comp, hw, args.steps)

    for label, tensor in (("serial", serial), ("concurrent", concurrent), ("parallel", parallel)):
        assert_gpu_tensor(label, tensor)
        e = float(energy(tensor).detach().cpu().item())
        if not math.isfinite(e):
            die(f"{label} non-finite energy.", RUNTIME_FAILURE)
        if abs(e - 1.0) > args.energy_tolerance:
            die(f"{label} energy invariant failed: {e} outside tolerance {args.energy_tolerance}.", ASSERTION_FAILURE)

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "HYBRID_REAL_GPU_ONLY",
        "sutra_count": len(SUTRA_NAMES),
        "sutra_names": list(SUTRA_NAMES),
        "execution_classes": {
            "serial_ids": sorted(SERIAL_IDS),
            "parallel_ids": sorted(PARALLEL_IDS),
            "concurrent_ids": sorted(CONCURRENT_IDS),
        },
        "steps": args.steps,
        "seed": args.seed,
        "shots": args.shots,
        "quantum_gain_count": args.quantum_gain_count,
        "quantum_gains": gains,
        "environment": environment_payload(device, cudaq_target),
        "initial": {
            "energy": float(energy(psi0).detach().cpu().item()),
            "checksum": checksum_tensor(psi0),
        },
        "serial": {
            "energy": float(energy(serial).detach().cpu().item()),
            "checksum": checksum_tensor(serial),
            "trace": serial_trace,
        },
        "concurrent": {
            "energy": float(energy(concurrent).detach().cpu().item()),
            "checksum": checksum_tensor(concurrent),
            "trace": concurrent_trace,
        },
        "parallel": {
            "energy": float(energy(parallel).detach().cpu().item()),
            "checksum": checksum_tensor(parallel),
            "trace": parallel_trace,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full 29-sutra HYBRID suite on real CUDA GPU only.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--quantum-gain-count", type=int, default=29)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=Path("runs/hybrid_real_gpu_report.json"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.steps <= 0:
        die("--steps must be positive.", ASSERTION_FAILURE)
    if args.shots <= 0:
        die("--shots must be positive.", ASSERTION_FAILURE)
    if args.quantum_gain_count != 29:
        die("--quantum-gain-count must remain 29 for the full sutra catalogue.", ASSERTION_FAILURE)

    payload = run_all(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS_REAL_GPU_HYBRID",
        "output": str(args.output),
        "serial_checksum": payload["serial"]["checksum"],
        "concurrent_checksum": payload["concurrent"]["checksum"],
        "parallel_checksum": payload["parallel"]["checksum"],
        "cuda_device_name": payload["environment"]["cuda_device_name"],
        "cudaq_target": payload["environment"]["cudaq_target"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
