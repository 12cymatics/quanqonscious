"""Validation routines for PCFE deployments."""

import numpy as np


def validate_state(state: np.ndarray, threshold: float = 0.99) -> bool:
    """Check that the state coherence exceeds the threshold."""
    coherence = np.abs(np.mean(state))
    return coherence > threshold


def deployment_summary(state: np.ndarray) -> str:
    avg_value = np.mean(state)
    return f"Deployment average field value: {avg_value:.4f}"
