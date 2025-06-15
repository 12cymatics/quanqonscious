import numpy as np
from src.pcfe_v3_core_engine import load_config, PCFECoreEngine


def test_run_simulation():
    config = load_config({"grid_size": 4, "grid_dimensions": 2, "max_iterations": 5})
    engine = PCFECoreEngine(config)
    state = engine.run()
    assert state.shape == (4, 4)
    assert np.isfinite(state).all()
