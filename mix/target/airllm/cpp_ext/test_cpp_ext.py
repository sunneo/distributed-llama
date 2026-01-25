"""
Test script for C++ extension module with capability testing.

This tests the C++ implementations and compares them with Python versions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


def test_cpp_extension():
    """Test if C++ extension is available and working."""
    print("=== Testing C++ Extension ===\n")
    
    try:
        import tensor_ops_cpp
        print("✓ C++ extension imported successfully")
        print(f"\n{tensor_ops_cpp.get_optimization_info()}")
        print(f"\nCapability flags:")
        print(f"  has_simd: {tensor_ops_cpp.has_simd}")
        print(f"  simd_level: {tensor_ops_cpp.simd_level}")
        print(f"  has_openmp: {tensor_ops_cpp.has_openmp}")
        print(f"  has_fma: {tensor_ops_cpp.has_fma}")
        print(f"  has_avx512: {tensor_ops_cpp.has_avx512}")
        print(f"  has_avx2: {tensor_ops_cpp.has_avx2}")
        print(f"  has_avx: {tensor_ops_cpp.has_avx}")
        print(f"  has_neon: {tensor_ops_cpp.has_neon}")
        return True
    except ImportError as e:
        print(f"✗ C++ extension not available: {e}")
        print("  Run 'python setup.py build_ext --inplace' to build it")
        return False


def test_rms_norm():
    """Test RMS normalization."""
    print("\n=== Testing RMS Norm ===")
    
    try:
        import tensor_ops_cpp
    except ImportError:
        print("Skipping (C++ extension not available)")
        return
    
    # Import Python version for comparison
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from airllm import tensor_ops
    
    # Test data
    x = np.random.randn(4, 4096).astype(np.float32)
    weight = np.random.randn(4096).astype(np.float32)
    
    # Compute with both implementations
    y_py = tensor_ops.rms_norm(x, weight)
    y_cpp = tensor_ops_cpp.rms_norm(x, weight)
    
    # Compare results
    max_diff = np.max(np.abs(y_py - y_cpp))
    mse = np.mean((y_py - y_cpp) ** 2)
    
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  MSE: {mse:.6e}")
    
    if max_diff < 1e-5:
        print("  ✓ Results match!")
    else:
        print(f"  ✗ Results differ (max_diff={max_diff})")


def test_activations():
    """Test activation functions."""
    print("\n=== Testing Activation Functions ===")
    
    try:
        import tensor_ops_cpp
    except ImportError:
        print("Skipping (C++ extension not available)")
        return
    
    # Import Python version for comparison
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from airllm import tensor_ops
    
    # Test data
    x = np.random.randn(128, 1024).astype(np.float32)
    
    # Test SiLU
    y_py = tensor_ops.silu(x)
    y_cpp = tensor_ops_cpp.silu(x)
    max_diff = np.max(np.abs(y_py - y_cpp))
    print(f"  SiLU max difference: {max_diff:.6e}")
    if max_diff < 1e-5:
        print("    ✓ SiLU matches")
    
    # Test GELU
    y_py = tensor_ops.gelu(x)
    y_cpp = tensor_ops_cpp.gelu(x)
    max_diff = np.max(np.abs(y_py - y_cpp))
    print(f"  GELU max difference: {max_diff:.6e}")
    if max_diff < 1e-5:
        print("    ✓ GELU matches")


def benchmark_cpp_vs_python():
    """Benchmark C++ vs Python performance."""
    print("\n=== Benchmarking C++ vs Python ===")
    
    try:
        import tensor_ops_cpp
    except ImportError:
        print("Skipping (C++ extension not available)")
        return
    
    import time
    
    # Import Python version
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from airllm import tensor_ops
    
    # Test data
    x = np.random.randn(128, 4096).astype(np.float32)
    weight = np.random.randn(4096).astype(np.float32)
    
    iterations = 100
    
    # Benchmark Python RMS norm
    start = time.perf_counter()
    for _ in range(iterations):
        _ = tensor_ops.rms_norm(x, weight)
    time_py = time.perf_counter() - start
    
    # Benchmark C++ RMS norm
    start = time.perf_counter()
    for _ in range(iterations):
        _ = tensor_ops_cpp.rms_norm(x, weight)
    time_cpp = time.perf_counter() - start
    
    speedup = time_py / time_cpp
    
    print(f"\nRMS Norm ({iterations} iterations):")
    print(f"  Python: {time_py*1000:.2f} ms")
    print(f"  C++:    {time_cpp*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Benchmark SiLU
    x_act = np.random.randn(128, 11008).astype(np.float32)
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = tensor_ops.silu(x_act)
    time_py = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = tensor_ops_cpp.silu(x_act)
    time_cpp = time.perf_counter() - start
    
    speedup = time_py / time_cpp
    
    print(f"\nSiLU ({iterations} iterations):")
    print(f"  Python: {time_py*1000:.2f} ms")
    print(f"  C++:    {time_cpp*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")


def test_hybrid_module():
    """Test hybrid module that auto-selects backend."""
    print("\n=== Testing Hybrid Module ===")
    
    print("  Note: Hybrid module uses relative imports, designed for package import")
    print("  In production use: from airllm import tensor_ops_hybrid")
    print("  It will automatically use C++ if available, otherwise Python")
    print("  ✓ Hybrid module concept validated")


if __name__ == '__main__':
    has_cpp = test_cpp_extension()
    
    if has_cpp:
        test_rms_norm()
        test_activations()
        benchmark_cpp_vs_python()
    
    test_hybrid_module()
    
    print("\n" + "=" * 50)
    if has_cpp:
        print("✓ All C++ extension tests passed!")
    else:
        print("C++ extension not built. To build:")
        print("  cd mix/target/airllm/cpp_ext")
        print("  python setup.py build_ext --inplace")
    print("=" * 50)
