import os
import sys
import pathlib
import numpy as np

# Ensure the package can be imported from the repository's src directory
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Enable lightweight import without heavy dependencies
os.environ["QUANQONSCIOUS_SKIP_INIT"] = "1"

from quanqonscious.sutra_executor import SutraExecutor, ExecutionMode


def test_execution_modes_consistent():
    rng = np.random.default_rng(0)
    inputs = [rng.random(8) for _ in range(3)]

    serial_exec = SutraExecutor(mode=ExecutionMode.SERIAL)
    thread_exec = SutraExecutor(mode=ExecutionMode.THREADS)
    proc_exec = SutraExecutor(mode=ExecutionMode.PROCESSES)

    serial_results = serial_exec.execute(inputs)
    thread_results = thread_exec.execute(inputs)
    process_results = proc_exec.execute(inputs)

    for s, t, p in zip(serial_results, thread_results, process_results):
        assert np.allclose(s, t)
        assert np.allclose(s, p)
