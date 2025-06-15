"""Final integration utilities for combining engine components."""

from .pcfe_v3_core_engine import PCFECoreEngine, EngineConfig
from .pcfe_mpi_visualization import distributed_average
from .pcfe_validation_deployment import validate_state


def run_full_simulation(config: EngineConfig) -> bool:
    engine = PCFECoreEngine(config)
    state = engine.run()
    avg = distributed_average(state)
    is_valid = validate_state(state)
    print(f"Global average: {avg:.4f}")
    print(f"State valid: {is_valid}")
    return is_valid
