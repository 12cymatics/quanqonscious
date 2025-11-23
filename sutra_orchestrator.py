from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from sutra_repository import SutraRepository, SutraContext, SutraMode
from maya_cymatic_simulation import encrypt_with_cymatic
from sutra_simulator import HybridQuantumClassicalSimulator


@dataclass(frozen=True)
class SutraInvocationState:
    """Immutable snapshot describing the current sutra invocation context."""

    current_value: Any
    initial_value: Any
    context: SutraContext


ArgumentProvider = Callable[[SutraInvocationState], Tuple[Tuple[Any, ...], Dict[str, Any]]]

# Methods that intentionally mutate shared state and should not be part of the
# automated orchestration pipelines.
EXCLUDED_SUTRAS = {"reset_performance_tracking"}


def _safe_number(value: Any, fallback: float) -> float:
    """Extract a numeric value from ``value`` with graceful degradation."""

    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue
        try:
            return float(item)
        except (TypeError, ValueError):
            continue
    return float(fallback)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _paired_scalars(state: SutraInvocationState) -> Tuple[float, float]:
    base = _safe_number(state.current_value, state.context.base)
    delta = max(abs(base) * 0.25, 1.0)
    return base, base + delta


def _positive_scalar(state: SutraInvocationState, *, minimum: float = 1.0) -> float:
    candidate = abs(_safe_number(state.current_value, minimum))
    if candidate < state.context.epsilon:
        candidate = minimum
    return candidate


def _polynomial_coefficients(state: SutraInvocationState) -> Tuple[float, float, float]:
    x = _safe_number(state.current_value, state.context.base)
    delta = max(abs(x) * 0.1, 1.0)
    return x + delta, -delta, state.context.base


def _apply_sutra_sequence_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    seed = _safe_number(state.current_value, state.context.base)
    _, secondary = _paired_scalars(state)
    ratio = _clamp(secondary / (abs(seed) + state.context.epsilon), 0.5, 3.0)
    sequence = [
        ("ekadhikena_purvena", {"iterations": 1}),
        ("anurupyena", {"b": secondary, "ratio": ratio}),
    ]
    return (seed, sequence), {}


def _optimize_sutra_sequence_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    seed = _safe_number(state.current_value, state.context.base)
    _, secondary = _paired_scalars(state)
    initial_sequence = [
        ("ekadhikena_purvena", {"iterations": 1}),
        ("paravartya_yojayet", {"divisor": max(abs(secondary), 1.0)}),
    ]
    target_output = seed * 1.5 + secondary * 0.1
    return (initial_sequence, seed, target_output, 2), {}


def _recommend_sutra_sequence_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    characteristics = {"sparsity": 0.4, "dimensionality": 3, "periodicity": 0.6}
    return ("default", (3,), characteristics), {}


def _visualize_performance_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return tuple(), {"n_top": 5, "output_format": "text"}


def _get_performance_summary_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return tuple(), {}


def _grvq_field_solver_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    radial = _positive_scalar(state, minimum=1.0)
    return (radial, 0.75, 1.047197551, 0.3), {}


def _sesanyankena_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    c0, c1, c2 = _polynomial_coefficients(state)
    coefficients = [c0, c1, c2]
    return (coefficients, x), {}


def _maya_illusion_transform_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    return (x, 0.45, 1.0), {}


def _maya_illusion_phase_cancellation_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    threshold = max(state.context.epsilon * 50.0, 0.05)
    return (x, 0.35, 1.2, threshold), {}


def _maya_illusion_multi_layer_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    return (x, [0.2, 0.4, 0.6], [0.5, 1.0, 1.5]), {}


def _sulba_circle_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    radius = _positive_scalar(state, minimum=1.0) + 0.5
    return (radius,), {}


def _sulba_geometric_mean_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    return (a, b), {}


def _sulba_pythagorean_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    bound = int(max(_positive_scalar(state, minimum=5.0), 5.0)) + 3
    return (bound,), {}


def _sulba_square_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    side = _positive_scalar(state, minimum=1.0) + 1.0
    return (side,), {}


def _two_operand_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    return (a, b), {}


def _anurupyena_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    ratio = _clamp(b / (abs(a) + state.context.epsilon), 0.25, 4.0)
    return (a, b, ratio), {}


def _chalana_kalana_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    steps = max(1, min(5, int(abs(x)) % 5 or 2))
    direction = 1 if x >= 0 else -1
    return (x, steps, direction), {}


def _ekadhikena_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    return (x, 3), {}


def _ekanyunena_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    base = max(state.context.base, 2.0)
    return (x, base), {}


def _nikhilam_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    base = max(state.context.base, 2.0)
    return (x, base), {}


def _paravartya_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    divisor = max(abs(_safe_number(state.initial_value, state.context.base)), 1.0)
    return (x, divisor), {}


def _purna_apurna_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    threshold = x + max(abs(x) * 0.25, 1.0)
    return (x, threshold), {}


def _samuccayagunitah_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    return (a, b, "product_sum"), {}


def _sankalana_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    return (a, b, "both"), {}


def _shunyam_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    return (a, b), {}


def _sunyam_samya_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    a, b = _paired_scalars(state)
    epsilon = max(state.context.epsilon, 1e-6)
    return (a, b, epsilon), {}


def _vyashtisamanstih_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    whole = max(_safe_number(state.current_value, state.context.base), 1.0)
    parts = [whole * 0.6, whole * 0.4]
    return (whole, parts), {}


def _yavadunam_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    x = _safe_number(state.current_value, state.context.base)
    base = max(state.context.base, 2.0)
    return (x, base), {}


def _gunitasamuccayah_arguments(state: SutraInvocationState) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    multiplicand, multiplier = _paired_scalars(state)
    return (multiplicand, multiplier), {}


SUTRA_ARGUMENT_PROVIDERS: Dict[str, ArgumentProvider] = {
    "anurupyena": _anurupyena_arguments,
    "apply_sutra_sequence": _apply_sutra_sequence_arguments,
    "chalana_kalana": _chalana_kalana_arguments,
    "ekadhikena_purvena": _ekadhikena_arguments,
    "ekanyunena_purvena": _ekanyunena_arguments,
    "get_performance_summary": _get_performance_summary_arguments,
    "grvq_field_solver": _grvq_field_solver_arguments,
    "gunakasamuccayah": _two_operand_arguments,
    "gunitasamuccayah": _gunitasamuccayah_arguments,
    "maya_illusion_multi_layer": _maya_illusion_multi_layer_arguments,
    "maya_illusion_phase_cancellation": _maya_illusion_phase_cancellation_arguments,
    "maya_illusion_transform": _maya_illusion_transform_arguments,
    "nikhilam_navatashcaramam_dashatah": _nikhilam_arguments,
    "optimize_sutra_sequence": _optimize_sutra_sequence_arguments,
    "paravartya_yojayet": _paravartya_arguments,
    "purna_apurna_bhyam": _purna_apurna_arguments,
    "recommend_sutra_sequence": _recommend_sutra_sequence_arguments,
    "samuccayagunitah": _samuccayagunitah_arguments,
    "sankalana_vyavakalanabhyam": _sankalana_arguments,
    "sesanyankena_caramena": _sesanyankena_arguments,
    "shunyam_samyasamuccaye": _shunyam_arguments,
    "sulba_circle_construction": _sulba_circle_arguments,
    "sulba_geometric_mean": _sulba_geometric_mean_arguments,
    "sulba_pythagorean_triples": _sulba_pythagorean_arguments,
    "sulba_square_construction": _sulba_square_arguments,
    "sunyam_samya_samuccaye": _sunyam_samya_arguments,
    "visualize_performance": _visualize_performance_arguments,
    "vyashtisamanstih": _vyashtisamanstih_arguments,
    "yavadunam": _yavadunam_arguments,
}


def _should_include_sutra(name: str) -> bool:
    return name not in EXCLUDED_SUTRAS


def _argument_resolver(
    name: str,
    current_value: Any,
    context: SutraContext,
    initial_value: Any,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    state = SutraInvocationState(
        current_value=current_value,
        initial_value=initial_value,
        context=context,
    )
    provider = SUTRA_ARGUMENT_PROVIDERS.get(name)
    if provider is None:
        return (current_value,), {}
    args, kwargs = provider(state)
    return tuple(args), dict(kwargs)


def _build_simulator(mode: SutraMode) -> HybridQuantumClassicalSimulator:
    return HybridQuantumClassicalSimulator(
        SutraContext(mode=mode),
        sutra_filter=_should_include_sutra,
        argument_resolver=_argument_resolver,
    )


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
    parser.add_argument(
        "--ghz-ibmq",
        action="store_true",
        help=(
            "Run hybrid sutra pipeline with GHZ circuit executed on IBM Quantum "
            "backend"
        ),
    )
    parser.add_argument(
        "--maya-cymatic",
        action="store_true",
        help="Encrypt a message using Maya cipher and generate cymatic verification",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="",
        help="Message payload for Maya cipher when using --maya-cymatic",
    )
    parser.add_argument(
        "--key",
        type=lambda x: int(x, 0),
        default=0xDEADBEEF,
        help="Integer key for Maya cipher (e.g., 0xDEADBEEF)",
    )

    args = parser.parse_args()

    mode = SutraMode[args.mode.upper()]

    if args.maya_cymatic:
        result = maya_cymatic_pipeline(args.message, key=args.key)
        print(result)
    elif args.ghz_ibmq:
        result = hybrid_ghz_ibmq_pipeline(args.value)
        print(result)
    elif args.ghz:
        result = hybrid_ghz_pipeline(args.value)
        print(result)
    elif args.parallel:
        parallel_hybrid_run(args.value)
    elif args.concurrent:
        concurrent_run(args.value, mode)
    else:
        serial_run(args.value, mode)
