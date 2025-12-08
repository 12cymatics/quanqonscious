from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from sutra_repository import SutraRepository, SutraContext, SutraMode
from qiskit_backend import execute_ghz


def serial_run(value: Any, mode: SutraMode = SutraMode.CLASSICAL) -> Any:
    """Run all sutras sequentially in the specified mode."""
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)
    result = value
    for name in repo.list_sutras():
        func = repo._methods[name]
        args = _prepare_args(func, result)
        try:
            result = repo.call_sutra(name, *args, ctx=ctx)
            print(f"{name} -> {result}")
        except Exception as e:
            print(f"{name} FAILED: {e}")
    return result


def concurrent_run(value: Any, mode: SutraMode = SutraMode.CLASSICAL) -> Dict[str, Any]:
    """Execute all sutras concurrently using threads."""
    ctx = SutraContext(mode=mode)
    repo = SutraRepository(ctx)

    def run(name: str) -> tuple[str, Any]:
        func = repo._methods[name]
        args = _prepare_args(func, value)
        try:
            res = repo.call_sutra(name, *args, ctx=ctx)
        except Exception as e:
            res = f"FAILED: {e}"
        return name, res

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
        func = inner_repo._methods[name]
        args = _prepare_args(func, val)
        try:
            res = inner_repo.call_sutra(name, *args, ctx=ctx)
        except Exception as e:
            res = f"FAILED: {e}"
        return name, res

    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(call, name, value) for name in names]
        for fut in futures:
            name, res = fut.result()
            print(f"{name} (hybrid) -> {res}")


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

    args = parser.parse_args()

    mode = SutraMode[args.mode.upper()]

    if args.parallel:
        parallel_hybrid_run(args.value)
    elif args.concurrent:
        concurrent_run(args.value, mode)
    else:
        serial_run(args.value, mode)
