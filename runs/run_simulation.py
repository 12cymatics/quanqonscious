#!/usr/bin/env python3
"""
GRVQ/MSTVQ/TGCR Cymatic Simulation Runner

Example script demonstrating the complete simulation pipeline.
Follows CODEX specification for:
- Two-lane hybrid execution
- Vedic sutra operators
- Observable computation
- Trace recording
"""

import sys
import os
import json
from datetime import datetime
from fractions import Fraction

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    create_3d_lattice,
    create_gaussian_field,
    HybridPipeline,
    HybridPipelineConfig,
    ClassicalEvolutionConfig,
    QuantumAssistConfig,
    OperatorContext,
)


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'default_config.json')

    with open(config_path, 'r') as f:
        return json.load(f)


def run_simulation(config: dict = None):
    """Run the complete cymatic simulation."""
    if config is None:
        config = load_config()

    print("=" * 60)
    print(f"  {config['simulation']['name']}")
    print(f"  Version: {config['simulation']['version']}")
    print(f"  CODEX: {config['simulation']['codex_version']}")
    print("=" * 60)
    print()

    # Create lattice
    dims = tuple(config['lattice']['dimensions'])
    print(f"Creating {dims[0]}x{dims[1]}x{dims[2]} toroidal lattice...")
    lattice = create_3d_lattice(*dims, r4_radius=config['lattice']['r4_radius'])

    # Create initial field
    ic = config['initial_condition']
    print(f"Initializing Gaussian field (σ={ic['sigma']}, A={ic['amplitude']})...")
    state = create_gaussian_field(
        lattice,
        center=tuple(ic['center']),
        sigma=ic['sigma'],
        amplitude=ic['amplitude']
    )

    # Configure pipeline
    print("Configuring two-lane hybrid pipeline...")

    classical_config = ClassicalEvolutionConfig(
        dt=Fraction(config['evolution']['dt']),
        use_grvq=config['operators']['grvq']['enabled'],
        use_mstvq=config['operators']['mstvq']['enabled'],
        use_r4=config['operators']['r4']['enabled'],
        use_sutras=config['operators']['sutras']['enabled'],
        sutra_sequence=config['operators']['sutras']['sequence'],
        h_m=Fraction(config['operators']['mstvq']['h_m']),
        stress_coupling=Fraction(config['operators']['mstvq']['stress_coupling']),
    )

    quantum_config = QuantumAssistConfig(
        enabled=config['quantum_assist']['enabled'],
        seed=config['quantum_assist']['seed'],
        max_shards=config['quantum_assist']['max_shards'],
        enable_coefficient_tuning=config['quantum_assist']['enable_coefficient_tuning'],
        enable_mode_selection=config['quantum_assist']['enable_mode_selection'],
        enable_phase_template=config['quantum_assist']['enable_phase_template'],
    )

    pipeline_config = HybridPipelineConfig(
        classical=classical_config,
        quantum=quantum_config,
        num_steps=config['evolution']['num_steps'],
        checkpoint_interval=config['evolution']['checkpoint_interval'],
        compute_observables=config['observables']['compute'],
        check_invariants=config['invariants']['check'],
        fail_fast=config['invariants']['fail_fast'],
    )

    pipeline = HybridPipeline(pipeline_config)

    # Run simulation
    print()
    print(f"Running {config['evolution']['num_steps']} evolution steps...")
    print("-" * 40)

    start_time = datetime.now()
    final_state, trace, history = pipeline.run(state)
    end_time = datetime.now()

    elapsed = (end_time - start_time).total_seconds()

    # Report results
    print()
    print("=" * 60)
    print("  Simulation Complete")
    print("=" * 60)
    print()
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Steps completed: {len(history)}")
    print(f"  Trace entries: {len(trace.operator_trace.entries)}")
    print(f"  Checkpoints: {len(trace.checkpoints)}")
    print()

    # Final observables
    if history:
        final_obs = history[-1]
        print("  Final Observables:")
        for key, value in final_obs.items():
            if key == 'timestep':
                continue
            if isinstance(value, dict):
                if 'float' in value:
                    print(f"    {key}: {value['float']:.6f}")
                elif 'mean' in value:
                    print(f"    {key}: mean={value['mean']:.6f}, max={value['max']:.6f}")
            else:
                print(f"    {key}: {value}")

    print()
    print(f"  Final field max amplitude: {final_state.max_amplitude():.6f}")
    print(f"  Final field total norm²: {float(final_state.total_norm_squared()):.6f}")

    # Save outputs
    if config['output']['save_observables']:
        output_dir = config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)

        obs_path = os.path.join(output_dir, 'observables.json')
        with open(obs_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        print(f"\n  Observables saved to: {obs_path}")

    if config['output']['save_trace']:
        output_dir = config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)

        trace_path = os.path.join(output_dir, 'trace.json')
        with open(trace_path, 'w') as f:
            f.write(trace.to_json())
        print(f"  Trace saved to: {trace_path}")

    print()
    return final_state, trace, history


if __name__ == "__main__":
    # Run with default or provided config
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()

    run_simulation(config)
