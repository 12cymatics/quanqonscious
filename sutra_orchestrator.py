from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Dict

from sutra_repository import SutraRepository, SutraContext, SutraMode
from qiskit_backend import execute_ghz


def serial_run(value: Any, mode: SutraMode = SutraMode.CLASSICAL) -> Any:
    """Run all sutras sequentially in the specified mode."""
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)
    result = value
    for name in repo.list_sutras():
        result = repo.call_sutra(name, result, ctx=ctx)
        print(f"{name} -> {result}")
    return result


def concurrent_run(value: Any, mode: SutraMode = SutraMode.CLASSICAL) -> Dict[str, Any]:
    """Execute all sutras concurrently using threads."""
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)

    def run(name: str) -> tuple[str, Any]:
        return name, repo.call_sutra(name, value, ctx=ctx)

    results: Dict[str, Any] = {}
    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(run, name) for name in repo.list_sutras()]
        for fut in futures:
            name, res = fut.result()
            results[name] = res
            print(f"{name} -> {res}")
    return results


def parallel_hybrid_run(value: Any) -> None:
    """Run all sutras in hybrid mode across multiple processes."""
    repo = SutraRepository()
    names = repo.list_sutras()

    def call(name: str, val: Any) -> tuple[str, Any]:
        ctx = SutraContext(mode=SutraMode.HYBRID, parallel=False)
        inner_repo = SutraRepository(ctx)
        return name, inner_repo.call_sutra(name, val, ctx=ctx)

    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(call, name, value) for name in names]
        for fut in futures:
            name, res = fut.result()
            print(f"{name} (hybrid) -> {res}")


def hybrid_ghz_pipeline(value: Any, num_qubits: int = 29) -> Dict[str, Any]:
    """Run a hybrid sutra workflow and entangle ``num_qubits`` via Qiskit.

    The provided value is first processed through a representative sutra
    (``ekadhikena_purvena``) in hybrid mode.  In parallel, a GHZ circuit
    spanning ``num_qubits`` qubits is constructed and executed using
    :func:`qiskit_backend.execute_ghz`.  The two results are returned together,
    enabling subsequent fusion or analysis steps.

    Parameters
    ----------
    value:
        Initial numeric value supplied to the sutra pipeline.
    num_qubits:
        Number of qubits for the GHZ circuit. Defaults to 29 to mirror the
        count of Vedic sutras.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the final sutra output under ``"sutra_result"``
        and the GHZ measurement counts under ``"quantum_counts"``.
    """

    ctx = SutraContext(mode=SutraMode.HYBRID)
    repo = SutraRepository(ctx)
    sutra_result = repo.call_sutra("ekadhikena_purvena", value, ctx=ctx)
    quantum_counts = execute_ghz(num_qubits=num_qubits)
    return {"sutra_result": sutra_result, "quantum_counts": quantum_counts}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all Vedic sutras")
    parser.add_argument("value", type=float, help="Input value for sutras")
    parser.add_argument(
        "--mode",
        choices=[m.name.lower() for m in SutraMode],
        default="classical",
        help="Execution mode for serial or concurrent runs",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run in parallel hybrid mode across processes",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Execute sutras concurrently using threads",
    )
    parser.add_argument(
        "--ghz",
        action="store_true",
        help="Run hybrid sutra pipeline combined with a multi-qubit GHZ circuit",
    )

    args = parser.parse_args()

    mode = SutraMode[args.mode.upper()]

    if args.ghz:
        result = hybrid_ghz_pipeline(args.value)
        print(result)
    elif args.parallel:
        parallel_hybrid_run(args.value)
    elif args.concurrent:
        concurrent_run(args.value, mode)
    else:
        serial_run(args.value, mode)
