# SPDX-License-Identifier: Apache-2.0

# Dependency Update: Removal of JAX

The JAX and JAXLIB dependencies have been removed to streamline CUDA deployment and align with the MSTVQ field solver stack. GPU acceleration is now delivered via CuPy and Numba, which interface directly with NVIDIA A100 hardware.

## Benchmark

The following baseline was collected on CPU using NumPy:

| Backend | Operation | Size | Time (ms) |
|---------|-----------|------|-----------|
| NumPy   | elementwise addition x+y | 1e6 | 41.747 |

CuPy kernels executed on an A100 provide additional hardware-level parallelism, enabling order-of-magnitude speed-ups in production deployments.
