# QuanQonscious/performance.py

import time
import functools
import gc
import numpy as np
try:
    import numba
except ImportError:
    numba = None
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import psutil
except ImportError:
    psutil = None

# Performance thresholds (can be adjusted or made configurable)
TIME_THRESHOLD = 1.0       # 1.0 second
MEM_THRESHOLD = 500e6      # 500 MB
CPU_UTIL_THRESHOLD = 0.9   # 90% CPU utilization (if measurable)
GPU_MEM_THRESHOLD = 0.8    # 80% GPU memory usage fraction

def monitor_performance(func):
    """
    Decorator to monitor the performance of the wrapped function. 
    If thresholds are exceeded, attempts to optimize the function for subsequent calls.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record start times (wall-clock and CPU process time)
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        # Memory usage before (RSS and GPU)
        process = psutil.Process() if psutil else None
        mem_before = process.memory_info().rss if process else 0
        gpu_mem_before = 0
        if cp:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                gpu_mem_before = total - free
            except Exception:
                gpu_mem_before = 0
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Record end times and memory
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        mem_after = process.memory_info().rss if process else 0
        gpu_mem_after = 0
        if cp:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                gpu_mem_after = total - free
            except Exception:
                gpu_mem_after = gpu_mem_before  # if we can't get new info, assume unchanged
        
        # Calculate metrics
        wall_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        mem_used = mem_after - mem_before
        gpu_mem_used = gpu_mem_after - gpu_mem_before
        # Determine if thresholds exceeded
        slow = wall_time > TIME_THRESHOLD
        high_mem = mem_used > MEM_THRESHOLD
        gpu_mem_total = None
        gpu_usage_frac = None
        if cp:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                gpu_mem_total = total
                gpu_usage_frac = gpu_mem_after / total if total else None
            except Exception:
                gpu_usage_frac = None
        high_gpu_mem = (gpu_usage_frac is not None and gpu_usage_frac > GPU_MEM_THRESHOLD)
        
        # Report and optimize if needed
        if slow or high_mem or high_gpu_mem:
            print(f"[Performance] Function '{func.__name__}' took {wall_time:.3f}s (CPU time {cpu_time:.3f}s).")
            if high_mem:
                print(f"[Performance] Memory usage increased by {mem_used/1e6:.1f} MB, exceeding threshold.")
            if high_gpu_mem:
                pct = gpu_usage_frac * 100 if gpu_usage_frac is not None else 0
                print(f"[Performance] GPU memory usage at {pct:.1f}% capacity, exceeding threshold.")
            # Attempt optimization measures
            if slow and numba is not None:
                try:
                    optimized = numba.njit(func)
                    wrapper.__wrapped__ = optimized  # replace wrapped function with JIT-compiled version
                    print(f"[Performance] Optimized '{func.__name__}' with Numba JIT for subsequent calls.")
                except Exception as e:
                    print(f"[Performance] Numba JIT optimization failed: {e}")
            # If slow and large data, attempt GPU offload
            if slow and cp is not None:
                # Check if any numpy arrays in args to move to GPU (heuristic)
                if any(isinstance(x, np.ndarray) for x in args) or any(isinstance(v, np.ndarray) for v in kwargs.values()):
                    print(f"[Performance] Suggestion: Offloading heavy NumPy computations to GPU for '{func.__name__}' (if not already).")
            # Free GPU memory if high usage
            if high_gpu_mem and cp is not None:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    print("[Performance] Freed GPU memory pool blocks to reduce memory pressure.")
                except Exception:
                    pass
            # Trigger garbage collection if high memory usage
            if high_mem:
                gc.collect()
        return result
    return wrapper

# Example usage: annotate heavy functions with @monitor_performance
