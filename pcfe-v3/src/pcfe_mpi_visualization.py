"""MPI utilities and basic visualization for PCFE."""

from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np


def distributed_average(data: np.ndarray) -> float:
    """Compute the global average of the data across MPI ranks."""
    comm = MPI.COMM_WORLD
    local_sum = np.sum(data)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    total_count = comm.allreduce(data.size, op=MPI.SUM)
    return global_sum / total_count


def plot_state(state: np.ndarray, filename: str = "state.png") -> None:
    plt.imshow(state.mean(axis=0))
    plt.colorbar()
    plt.savefig(filename)
    plt.close()
