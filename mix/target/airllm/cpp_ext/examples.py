#!/usr/bin/env python3
"""
Example Usage of Optimized Tensor Operations

This demonstrates various ways to use the tensor operations library
with automatic capability detection and backend selection.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


def example_basic_usage():
    """Example 1: Basic usage with CPU backend."""
    print("=" * 70)
    print("Example 1: Basic CPU Backend Usage")
    print("=" * 70)
    
    try:
        import tensor_ops_cpp
        
        # Show what optimizations are enabled
        print("\n" + tensor_ops_cpp.get_optimization_info())
        
        # Create test data
        batch_size = 16
        hidden_dim = 512
        
        x = np.random.randn(batch_size, hidden_dim).astype(np.float32)
        weight = np.random.randn(hidden_dim).astype(np.float32)
        
        # RMS normalization
        normalized = tensor_ops_cpp.rms_norm(x, weight)
        print(f"\n✓ RMS norm: {x.shape} -> {normalized.shape}")
        
        # SiLU activation
        activated = tensor_ops_cpp.silu(x)
        print(f"✓ SiLU: {x.shape} -> {activated.shape}")
        
        # Matrix multiplication
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(64, 128).astype(np.float32)
        c = tensor_ops_cpp.matmul(a, b)
        print(f"✓ Matmul: {a.shape} @ {b.shape} -> {c.shape}")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ CPU backend not available: {e}")
        print("Build it with: python setup.py build_ext --inplace")
        return False


def example_query_capabilities():
    """Example 2: Query optimization capabilities."""
    print("\n" + "=" * 70)
    print("Example 2: Querying Optimization Capabilities")
    print("=" * 70)
    
    try:
        import tensor_ops_cpp
        
        print("\nAvailable optimizations:")
        print(f"  SIMD Level: {tensor_ops_cpp.simd_level}")
        print(f"  OpenMP: {'Yes' if tensor_ops_cpp.has_openmp else 'No'}")
        print(f"  FMA: {'Yes' if tensor_ops_cpp.has_fma else 'No'}")
        
        print("\nDetailed SIMD support:")
        print(f"  AVX-512: {'Yes' if tensor_ops_cpp.has_avx512 else 'No'}")
        print(f"  AVX2: {'Yes' if tensor_ops_cpp.has_avx2 else 'No'}")
        print(f"  AVX: {'Yes' if tensor_ops_cpp.has_avx else 'No'}")
        print(f"  ARM NEON: {'Yes' if tensor_ops_cpp.has_neon else 'No'}")
        
        # Recommendations based on capabilities
        print("\nPerformance recommendations:")
        if tensor_ops_cpp.has_avx2 and tensor_ops_cpp.has_fma and tensor_ops_cpp.has_openmp:
            print("  ✓ Excellent! You have AVX2+FMA+OpenMP - expect 5-15x speedup")
        elif tensor_ops_cpp.has_avx2 and tensor_ops_cpp.has_openmp:
            print("  ✓ Good! You have AVX2+OpenMP - expect 3-10x speedup")
        elif tensor_ops_cpp.has_openmp:
            print("  ⚠ Fair. You have OpenMP but no SIMD - expect 2-4x speedup")
        else:
            print("  ⚠ Limited. No SIMD or OpenMP - speedup will be modest")
        
        return True
        
    except ImportError:
        print("\n✗ Extension not built")
        return False


def example_backend_selection():
    """Example 3: Automatic backend selection."""
    print("\n" + "=" * 70)
    print("Example 3: Automatic Backend Selection")
    print("=" * 70)
    
    backends_available = []
    
    # Try all backends
    print("\nScanning for available backends...")
    
    # CUDA
    try:
        import tensor_ops_cuda
        backends_available.append(('CUDA', tensor_ops_cuda))
        print("  ✓ CUDA backend available")
    except ImportError:
        print("  ✗ CUDA backend not available")
    
    # OpenCL
    try:
        import tensor_ops_opencl
        backends_available.append(('OpenCL', tensor_ops_opencl))
        print("  ✓ OpenCL backend available")
    except ImportError:
        print("  ✗ OpenCL backend not available")
    
    # CPU
    try:
        import tensor_ops_cpp
        backends_available.append(('CPU', tensor_ops_cpp))
        print("  ✓ CPU backend available")
    except ImportError:
        print("  ✗ CPU backend not available")
    
    if not backends_available:
        print("\n✗ No backends available! Build at least the CPU backend.")
        return False
    
    # Select best backend (CUDA > OpenCL > CPU)
    backend_name, backend = backends_available[0]
    print(f"\n✓ Selected backend: {backend_name}")
    
    # Use the backend
    x = np.random.randn(64, 1024).astype(np.float32)
    w = np.random.randn(1024).astype(np.float32)
    
    result = backend.rms_norm(x, w)
    print(f"✓ Processed tensor with shape {x.shape} using {backend_name} backend")
    
    return True


def example_performance_comparison():
    """Example 4: Compare performance across backends."""
    print("\n" + "=" * 70)
    print("Example 4: Performance Comparison")
    print("=" * 70)
    
    import time
    
    # Collect available backends
    backends = []
    
    try:
        import tensor_ops_cpp
        backends.append(('CPU', tensor_ops_cpp))
    except ImportError:
        pass
    
    # Add GPU backends if available
    # (would need to be built separately)
    
    if not backends:
        print("\n✗ No backends available")
        return False
    
    # Test data
    x = np.random.randn(128, 4096).astype(np.float32)
    weight = np.random.randn(4096).astype(np.float32)
    
    print(f"\nBenchmarking RMS norm on {x.shape} tensor:")
    print(f"{'Backend':<15} {'Time (ms)':<15} {'Relative Speed':<15}")
    print("-" * 50)
    
    times = []
    for name, backend in backends:
        # Warmup
        for _ in range(3):
            _ = backend.rms_norm(x, weight)
        
        # Benchmark
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            _ = backend.rms_norm(x, weight)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        times.append((name, avg_time_ms))
    
    # Print results
    baseline = times[0][1]
    for name, avg_time in times:
        speedup = baseline / avg_time
        print(f"{name:<15} {avg_time:<15.3f} {speedup:<15.2f}x")
    
    return True


def main():
    """Run all examples."""
    print("\nTensor Operations - Usage Examples\n")
    
    success = True
    
    # Run examples
    success &= example_basic_usage()
    success &= example_query_capabilities()
    success &= example_backend_selection()
    success &= example_performance_comparison()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
    else:
        print("\n⚠ Some examples failed - check output above")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
