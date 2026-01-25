#!/usr/bin/env python3
"""
Example: Using Tensor Operations with Multiple Backends

This example demonstrates how to use the tensor operations module
with automatic backend detection and how to compare performance
across different backends.
"""

import sys
import os
import numpy as np
import time

# Add the parent directories to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def example_automatic_backend():
    """Example 1: Using automatic backend selection."""
    print_section("Example 1: Automatic Backend Selection")
    
    import tensor_ops
    
    # Check which backend is active
    print(f"\nActive backend: {tensor_ops.get_backend()}")
    print(f"\nBackend details:")
    info = tensor_ops.get_backend_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create sample data
    print("\nCreating sample tensors...")
    x = np.random.randn(128, 4096).astype(np.float32)
    weight = np.random.randn(4096).astype(np.float32)
    
    # Test operations
    print("\nTesting operations:")
    
    # RMS normalization
    start = time.perf_counter()
    y_rms = tensor_ops.rms_norm(x, weight)
    elapsed = time.perf_counter() - start
    print(f"  RMS norm: {elapsed*1000:.3f} ms, output shape: {y_rms.shape}")
    
    # SiLU activation
    start = time.perf_counter()
    y_silu = tensor_ops.silu(x)
    elapsed = time.perf_counter() - start
    print(f"  SiLU:     {elapsed*1000:.3f} ms, output shape: {y_silu.shape}")
    
    # GELU activation
    start = time.perf_counter()
    y_gelu = tensor_ops.gelu(x)
    elapsed = time.perf_counter() - start
    print(f"  GELU:     {elapsed*1000:.3f} ms, output shape: {y_gelu.shape}")


def example_compare_backends():
    """Example 2: Compare performance across available backends."""
    print_section("Example 2: Backend Performance Comparison")
    
    # List of backends to test
    backends = []
    
    # Try CUDA
    try:
        import tensor_ops_cuda
        backends.append(('CUDA', tensor_ops_cuda))
        print("✓ CUDA backend available")
    except ImportError:
        print("✗ CUDA backend not available")
    
    # Try OpenCL
    try:
        import tensor_ops_opencl
        backends.append(('OpenCL', tensor_ops_opencl))
        print("✓ OpenCL backend available")
    except ImportError:
        print("✗ OpenCL backend not available")
    
    # Try C++
    try:
        import tensor_ops_cpp
        backends.append(('C++', tensor_ops_cpp))
        print("✓ C++ backend available")
    except ImportError:
        print("✗ C++ backend not available")
    
    # Python is always available
    import tensor_ops as tensor_ops_py
    # Make sure we're using Python backend
    if tensor_ops_py.get_backend() == 'python':
        backends.append(('Python', tensor_ops_py))
        print("✓ Python backend available")
    
    if len(backends) < 2:
        print("\nNeed at least 2 backends to compare. Skipping comparison.")
        return
    
    # Benchmark parameters
    print("\n\nBenchmarking RMS Normalization...")
    print(f"{'Backend':<12} {'Time (ms)':<12} {'Relative Speed':<15}")
    print("-" * 45)
    
    x = np.random.randn(256, 4096).astype(np.float32)
    weight = np.random.randn(4096).astype(np.float32)
    iterations = 100
    
    results = []
    
    for name, backend in backends:
        # Warmup
        for _ in range(5):
            _ = backend.rms_norm(x, weight)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = backend.rms_norm(x, weight)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        results.append((name, avg_time_ms))
    
    # Print results with relative speeds
    baseline_time = results[-1][1]  # Use last backend as baseline
    for name, avg_time in results:
        relative_speed = baseline_time / avg_time
        print(f"{name:<12} {avg_time:<12.3f} {relative_speed:<15.2f}x")


def example_specific_backend():
    """Example 3: Using a specific backend directly."""
    print_section("Example 3: Using Specific Backend Directly")
    
    # Try to use C++ backend
    try:
        import tensor_ops_cpp
        
        print("\nUsing C++ Backend")
        print("\nOptimization info:")
        print(tensor_ops_cpp.get_optimization_info())
        
        print("\nCapability flags:")
        print(f"  SIMD Level:  {tensor_ops_cpp.simd_level}")
        print(f"  Has OpenMP:  {tensor_ops_cpp.has_openmp}")
        print(f"  Has FMA:     {tensor_ops_cpp.has_fma}")
        print(f"  Has AVX2:    {tensor_ops_cpp.has_avx2}")
        print(f"  Has AVX512:  {tensor_ops_cpp.has_avx512}")
        
        # Use the backend
        x = np.random.randn(64, 2048).astype(np.float32)
        weight = np.random.randn(2048).astype(np.float32)
        
        result = tensor_ops_cpp.rms_norm(x, weight)
        print(f"\nRMS norm executed successfully, output shape: {result.shape}")
        
    except ImportError as e:
        print(f"\nC++ backend not available: {e}")
        print("Build it with: python setup.py build_ext --inplace")


def example_batch_processing():
    """Example 4: Batch processing with tensor operations."""
    print_section("Example 4: Batch Processing")
    
    import tensor_ops
    
    print(f"\nUsing {tensor_ops.get_backend()} backend for batch processing")
    
    # Simulate processing multiple batches
    batch_size = 32
    seq_len = 128
    hidden_dim = 768
    num_batches = 10
    
    print(f"\nProcessing {num_batches} batches of shape ({batch_size}, {seq_len}, {hidden_dim})")
    
    weight = np.random.randn(hidden_dim).astype(np.float32)
    
    total_time = 0
    for i in range(num_batches):
        # Generate batch
        batch = np.random.randn(batch_size, hidden_dim).astype(np.float32)
        
        # Process
        start = time.perf_counter()
        normalized = tensor_ops.rms_norm(batch, weight)
        activated = tensor_ops.silu(normalized)
        elapsed = time.perf_counter() - start
        
        total_time += elapsed
    
    avg_time = total_time / num_batches
    print(f"\nAverage time per batch: {avg_time*1000:.3f} ms")
    print(f"Total processing time: {total_time:.3f} s")
    print(f"Throughput: {num_batches/total_time:.2f} batches/sec")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Tensor Operations Backend Examples")
    print("=" * 70)
    
    try:
        example_automatic_backend()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_compare_backends()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_specific_backend()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Build additional backends: python setup.py build_cuda")
    print("  - Run benchmarks: python detect_and_benchmark.py")
    print("  - Read documentation: BACKEND_GUIDE.md")


if __name__ == '__main__':
    main()
