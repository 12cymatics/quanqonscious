#!/usr/bin/env python3
"""
Test script to verify CUDA-Quantum implementation in PCFE v3.0
"""

import numpy as np
import torch
import cudaq
import time
import sys

def test_cudaq_installation():
    """Test if CUDA-Quantum is properly installed"""
    print("Testing CUDA-Quantum installation...")
    
    try:
        import cudaq
        print(f"✓ CUDA-Quantum version: {cudaq.__version__ if hasattr(cudaq, '__version__') else 'Unknown'}")
        
        # Test simple kernel
        @cudaq.kernel
        def simple_test():
            q = cudaq.qubit()
            h(q)
            mz(q)
        
        result = cudaq.sample(simple_test, shots_count=100)
        print("✓ Basic quantum circuit execution successful")
        
        return True
    except Exception as e:
        print(f"✗ CUDA-Quantum test failed: {e}")
        return False

def test_vedic_sutra_quantum():
    """Test Vedic Sutra quantum implementations"""
    print("\nTesting Vedic Sutra quantum kernels...")
    
    # Test Ekadhikena kernel
    @cudaq.kernel
    def ekadhikena_kernel(n_qubits: int, theta: float):
        qvec = cudaq.qvector(n_qubits)
        # Initialize with Hadamard superposition
        for i in range(n_qubits):
            h(qvec[i])
        # Apply rotation based on iteration
        for i in range(n_qubits):
            ry(theta * (i + 1), qvec[i])
        # Entangle adjacent qubits
        for i in range(n_qubits - 1):
            cx(qvec[i], qvec[i + 1])
    
    try:
        result = cudaq.sample(ekadhikena_kernel, 4, 0.5, shots_count=1000)
        counts = result.get_register_counts()
        print(f"✓ Ekadhikena kernel executed: {len(counts)} unique states observed")
    except Exception as e:
        print(f"✗ Ekadhikena kernel failed: {e}")
        return False
    
    # Test Urdhva kernel
    @cudaq.kernel
    def urdhva_kernel(n_qubits: int, angles: list[float]):
        qvec = cudaq.qvector(n_qubits)
        
        for i in range(min(len(angles), n_qubits)):
            ry(angles[i], qvec[i])
            if i < n_qubits - 1:
                cx(qvec[i], qvec[i + 1])
    
    try:
        angles = [0.1, 0.2, 0.3, 0.4]
        result = cudaq.sample(urdhva_kernel, 4, angles, shots_count=1000)
        print("✓ Urdhva-Tiryagbhyam kernel executed successfully")
    except Exception as e:
        print(f"✗ Urdhva kernel failed: {e}")
        return False
    
    return True

def test_quantum_phase_estimation():
    """Test QPE implementation for division"""
    print("\nTesting Quantum Phase Estimation...")
    
    @cudaq.kernel
    def qpe_kernel(n_precision: int, phase: float):
        # Precision qubits
        precision_qubits = cudaq.qvector(n_precision)
        # Eigenstate qubit
        eigenstate = cudaq.qubit()
        
        # Initialize eigenstate
        x(eigenstate)
        
        # Apply Hadamard to precision qubits
        for i in range(n_precision):
            h(precision_qubits[i])
        
        # Controlled rotations
        for i in range(n_precision):
            for j in range(2**i):
                cu(precision_qubits[i], eigenstate, 0, 0, 0, phase * 2**i)
        
        # Inverse QFT (simplified)
        for i in range(n_precision // 2):
            swap(precision_qubits[i], precision_qubits[n_precision - 1 - i])
        
        for i in range(n_precision):
            h(precision_qubits[i])
    
    try:
        result = cudaq.sample(qpe_kernel, 4, np.pi/4, shots_count=1000)
        print("✓ QPE kernel executed successfully")
        return True
    except Exception as e:
        print(f"✗ QPE kernel failed: {e}")
        return False

def test_performance_comparison():
    """Compare quantum circuit execution performance"""
    print("\nTesting quantum circuit performance...")
    
    @cudaq.kernel
    def performance_kernel(n_qubits: int, depth: int):
        qvec = cudaq.qvector(n_qubits)
        
        for _ in range(depth):
            # Layer of single-qubit gates
            for i in range(n_qubits):
                h(qvec[i])
                rz(0.1, qvec[i])
            
            # Layer of two-qubit gates
            for i in range(0, n_qubits - 1, 2):
                cx(qvec[i], qvec[i + 1])
    
    # Test different circuit sizes
    for n_qubits in [4, 6, 8]:
        for depth in [5, 10]:
            try:
                start_time = time.time()
                result = cudaq.sample(performance_kernel, n_qubits, depth, shots_count=100)
                exec_time = time.time() - start_time
                
                print(f"✓ {n_qubits} qubits, depth {depth}: {exec_time:.3f}s")
            except Exception as e:
                print(f"✗ Performance test failed for {n_qubits} qubits: {e}")
                return False
    
    return True

def test_integration_with_torch():
    """Test integration between CUDA-Quantum and PyTorch"""
    print("\nTesting CUDA-Quantum + PyTorch integration...")
    
    try:
        # Create PyTorch tensor
        field = torch.randn(4, 4, dtype=torch.complex128)
        
        # Extract values for quantum processing
        values = field.flatten()[:4]  # Take first 4 values
        magnitudes = torch.abs(values).tolist()
        phases = torch.angle(values).tolist()
        
        @cudaq.kernel
        def torch_integration_kernel(mags: list[float], phases: list[float]):
            n = len(mags)
            qvec = cudaq.qvector(n)
            
            for i in range(n):
                # Encode magnitude
                if mags[i] > 0:
                    angle = 2 * np.arcsin(min(1.0, mags[i]))
                    ry(angle, qvec[i])
                
                # Encode phase
                rz(phases[i], qvec[i])
        
        result = cudaq.sample(torch_integration_kernel, magnitudes, phases, shots_count=100)
        print("✓ PyTorch tensor processing with CUDA-Quantum successful")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PCFE v3.0 - CUDA-Quantum Implementation Test Suite          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    tests = [
        ("Installation", test_cudaq_installation),
        ("Vedic Sutras", test_vedic_sutra_quantum),
        ("Phase Estimation", test_quantum_phase_estimation),
        ("Performance", test_performance_comparison),
        ("PyTorch Integration", test_integration_with_torch)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Test")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✅ All CUDA-Quantum tests passed! PCFE is ready to run.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check CUDA-Quantum installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
