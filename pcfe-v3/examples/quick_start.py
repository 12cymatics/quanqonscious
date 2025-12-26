"""Quick start example for running a PCFE simulation."""

from src.pcfe_v3_core_engine import load_config, PCFECoreEngine
from src.pcfe_final_integration import run_full_simulation


def main():
    config = load_config({"grid_size": 8, "grid_dimensions": 2, "max_iterations": 10})
    run_full_simulation(config)


if __name__ == "__main__":
    main()
