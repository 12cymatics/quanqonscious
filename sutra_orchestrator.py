from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from sutra_repository import SutraRepository, SutraContext, SutraMode
from qiskit_backend import execute_ghz


def serial_run(value: Any, mode: SutraMode = SutraMode.CLASSICAL) -> Any:
    """Run all sutras sequentially in the specified mode."""

    simulator = _build_simulator(mode)
    report = simulator.run_serial(value)
    for execution in report.executions:
        print(f"{execution.name} -> {execution.output}")
    return report.aggregate


def concurrent_run(value: Any, mode: SutraMode = SutraMode.CLASSICAL) -> Dict[str, Any]:
    """Execute all sutras concurrently using threads."""

    simulator = _build_simulator(mode)
    report = simulator.run_concurrent(value)
    results: Dict[str, Any] = {}
    for execution in report.executions:
        results[execution.name] = execution.output
        print(f"{execution.name} -> {execution.output}")
    return results


def parallel_hybrid_run(value: Any) -> None:
    """Run all sutras in hybrid mode across multiple processes."""

    simulator = _build_simulator(SutraMode.HYBRID)
    report = simulator.run_parallel(value)
    for execution in report.executions:
        print(f"{execution.name} (hybrid) -> {execution.output}")


def hybrid_ghz_pipeline(value: Any, num_qubits: int = 29) -> Dict[str, Any]:
    """Run a hybrid sutra workflow and entangle ``num_qubits`` via local Qiskit.

    The provided value is first processed through a representative sutra
    (``ekadhikena_purvena``) in hybrid mode.  In parallel, a GHZ circuit
    spanning ``num_qubits`` qubits is constructed and executed locally using
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

    from qiskit_backend import execute_ghz

    ctx = SutraContext(mode=SutraMode.HYBRID)
    repo = SutraRepository(ctx)
    sutra_result = repo.call_sutra(
        "ekadhikena_purvena",
        value,
        ctx=ctx,
    )
    quantum_counts = execute_ghz(num_qubits=num_qubits)
    return {"sutra_result": sutra_result, "quantum_counts": quantum_counts}


def maya_cymatic_pipeline(message: str, key: int = 0xDEADBEEF) -> Dict[str, Any]:
    """Encrypt ``message`` with the Maya cipher and render cymatic verification.

    The function leverages :func:`maya_cymatic_simulation.encrypt_with_cymatic`
    to produce both a ciphertext and a GIF animation that encodes the
    timestamp-dependent cymatic signature.

    Parameters
    ----------
    message:
        Textual payload to encrypt and verify.
    key:
        Integer encryption key supplied to :class:`maya_cipher.MayaCipher`.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the ciphertext (hex encoded), the timestamp used
        during encryption and the path to the generated animation.
    """

    result = encrypt_with_cymatic(message.encode("utf-8"), key)
    return {
        "ciphertext": result.ciphertext.hex(),
        "timestamp": result.timestamp,
        "animation_path": result.animation_path,
    }


def hybrid_ghz_ibmq_pipeline(value: Any, num_qubits: int = 29) -> Dict[str, Any]:
    """Hybrid sutra workflow with GHZ circuit on IBM Quantum backend.

    This variant mirrors :func:`hybrid_ghz_pipeline` but delegates circuit
    execution to an IBM Quantum backend through
    :func:`qiskit_backend.execute_ghz_ibmq`, leveraging the embedded API key.

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
        A dictionary containing the final sutra output and the GHZ measurement
        counts observed on the IBM Quantum backend.
    """

    from qiskit_backend import execute_ghz_ibmq

    ctx = SutraContext(mode=SutraMode.HYBRID)
    repo = SutraRepository(ctx)
    sutra_result = repo.call_sutra(
        "ekadhikena_purvena",
        value,
        ctx=ctx,
    )
    quantum_counts = execute_ghz_ibmq(num_qubits=num_qubits)
    return {"sutra_result": sutra_result, "quantum_counts": quantum_counts}


def hybrid_ghz_pipeline(value: Any, num_qubits: int = 29) -> Dict[str, Any]:
    """Run a hybrid sutra workflow and entangle ``num_qubits`` via local Qiskit.

    The provided value is first processed through a representative sutra
    (``ekadhikena_purvena``) in hybrid mode.  In parallel, a GHZ circuit
    spanning ``num_qubits`` qubits is constructed and executed locally using
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

    from qiskit_backend import execute_ghz

    ctx = SutraContext(mode=SutraMode.HYBRID)
    repo = SutraRepository(ctx)
    sutra_result = repo.call_sutra("ekadhikena_purvena", value, ctx=ctx)
    quantum_counts = execute_ghz(num_qubits=num_qubits)
    return {"sutra_result": sutra_result, "quantum_counts": quantum_counts}


def maya_cymatic_pipeline(message: str, key: int = 0xDEADBEEF) -> Dict[str, Any]:
    """Encrypt ``message`` with the Maya cipher and render cymatic verification.

    The function leverages :func:`maya_cymatic_simulation.encrypt_with_cymatic`
    to produce both a ciphertext and a GIF animation that encodes the
    timestamp-dependent cymatic signature.

    Parameters
    ----------
    message:
        Textual payload to encrypt and verify.
    key:
        Integer encryption key supplied to :class:`maya_cipher.MayaCipher`.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the ciphertext (hex encoded), the timestamp used
        during encryption and the path to the generated animation.
    """

    result = encrypt_with_cymatic(message.encode("utf-8"), key)
    return {
        "ciphertext": result.ciphertext.hex(),
        "timestamp": result.timestamp,
        "animation_path": result.animation_path,
    }


def hybrid_ghz_ibmq_pipeline(value: Any, num_qubits: int = 29) -> Dict[str, Any]:
    """Hybrid sutra workflow with GHZ circuit on IBM Quantum backend.

    This variant mirrors :func:`hybrid_ghz_pipeline` but delegates circuit
    execution to an IBM Quantum backend through
    :func:`qiskit_backend.execute_ghz_ibmq`, leveraging the embedded API key.

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
        A dictionary containing the final sutra output and the GHZ measurement
        counts observed on the IBM Quantum backend.
    """

    from qiskit_backend import execute_ghz_ibmq

    ctx = SutraContext(mode=SutraMode.HYBRID)
    repo = SutraRepository(ctx)
    sutra_result = repo.call_sutra("ekadhikena_purvena", value, ctx=ctx)
    quantum_counts = execute_ghz_ibmq(num_qubits=num_qubits)
    return {"sutra_result": sutra_result, "quantum_counts": quantum_counts}


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
