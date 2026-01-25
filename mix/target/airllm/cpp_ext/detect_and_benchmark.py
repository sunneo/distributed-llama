#!/usr/bin/env python3
"""
Capability Detection and Benchmarking Script

This script demonstrates:
1. How to detect available CPU/GPU capabilities
2. How to build the extension with detected optimizations
3. How to benchmark different backends

Usage:
    python detect_and_benchmark.py
"""

import sys
import os
import subprocess
import time
import numpy as np

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

def detect_capabilities():
    """Run capability detection."""
    print_header("Step 1: Detecting System Capabilities")
    
    result = subprocess.run(
        [sys.executable, 'setup.py', 'test_capabilities'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def build_extension():
    """Build the C++ extension."""
    print_header("Step 2: Building C++ Extension with Auto-Detected Optimizations")
    
    # Clean previous builds
    subprocess.run([sys.executable, 'setup.py', 'clean', '--all'], 
                   capture_output=True)
    
    # Build with auto-detection
    result = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        capture_output=True,
        text=True
    )
    
    # Print relevant build info
    for line in result.stdout.split('\n'):
        if 'Available' in line or 'Building' in line or 'Flags:' in line or 'Defines:' in line:
            print(line)
    
    if result.returncode == 0:
        print("\n✓ Build successful!")
        return True
    else:
        print("\n✗ Build failed!")
        print(result.stderr)
        return False

def test_extension():
    """Test the built extension."""
    print_header("Step 3: Testing Extension")
    
    # Test all available backends
    backends_tested = []
    
    # Test CUDA backend
    try:
        import tensor_ops_cuda
        print("\n--- CUDA Backend ---")
        print(tensor_ops_cuda.get_cuda_info())
        
        # Test basic functionality
        x = np.random.randn(8, 128).astype(np.float32)
        w = np.random.randn(128).astype(np.float32)
        
        result = tensor_ops_cuda.rms_norm(x, w)
        print("✓ CUDA RMS norm: PASSED")
        
        result = tensor_ops_cuda.silu(x)
        print("✓ CUDA SiLU: PASSED")
        
        result = tensor_ops_cuda.gelu(x)
        print("✓ CUDA GELU: PASSED")
        
        backends_tested.append('cuda')
    except Exception as e:
        print(f"✗ CUDA backend not available: {e}")
    
    # Test OpenCL backend
    try:
        import tensor_ops_opencl
        print("\n--- OpenCL Backend ---")
        print(tensor_ops_opencl.get_opencl_info())
        
        # Test basic functionality
        x = np.random.randn(8, 128).astype(np.float32)
        w = np.random.randn(128).astype(np.float32)
        
        result = tensor_ops_opencl.rms_norm(x, w)
        print("✓ OpenCL RMS norm: PASSED")
        
        result = tensor_ops_opencl.silu(x)
        print("✓ OpenCL SiLU: PASSED")
        
        result = tensor_ops_opencl.gelu(x)
        print("✓ OpenCL GELU: PASSED")
        
        backends_tested.append('opencl')
    except Exception as e:
        print(f"✗ OpenCL backend not available: {e}")
    
    # Test C++ backend
    try:
        import tensor_ops_cpp
        print("\n--- C++ Backend ---")
        print(tensor_ops_cpp.get_optimization_info())
        print()
        
        # Test basic functionality
        x = np.random.randn(8, 128).astype(np.float32)
        w = np.random.randn(128).astype(np.float32)
        
        result = tensor_ops_cpp.rms_norm(x, w)
        print("✓ C++ RMS norm: PASSED")
        
        result = tensor_ops_cpp.silu(x)
        print("✓ C++ SiLU: PASSED")
        
        result = tensor_ops_cpp.gelu(x)
        print("✓ C++ GELU: PASSED")
        
        a = np.random.randn(16, 32).astype(np.float32)
        b = np.random.randn(32, 24).astype(np.float32)
        result = tensor_ops_cpp.matmul(a, b)
        print("✓ C++ Matmul: PASSED")
        
        backends_tested.append('cpp')
    except Exception as e:
        print(f"✗ C++ backend not available: {e}")
    
    if not backends_tested:
        print("\n✗ No accelerated backends available")
        return False
    
    print(f"\n✓ Tested backends: {', '.join(backends_tested)}")
    return True

def benchmark_extension():
    """Benchmark the extension."""
    print_header("Step 4: Benchmarking Performance")
    
    # Benchmark parameters
    iterations = 100
    
    # Test different sizes
    test_sizes = [
        (32, 512),    # Small
        (128, 2048),  # Medium
        (256, 4096),  # Large
    ]
    
    # Try to benchmark all available backends
    backends = []
    
    # Check CUDA
    try:
        import tensor_ops_cuda
        backends.append(('CUDA', tensor_ops_cuda))
    except ImportError:
        pass
    
    # Check OpenCL
    try:
        import tensor_ops_opencl
        backends.append(('OpenCL', tensor_ops_opencl))
    except ImportError:
        pass
    
    # Check C++
    try:
        import tensor_ops_cpp
        backends.append(('C++', tensor_ops_cpp))
    except ImportError:
        pass
    
    if not backends:
        print("✗ No accelerated backends available for benchmarking")
        return False
    
    for backend_name, backend_module in backends:
        print(f"\n=== {backend_name} Backend Benchmark ===")
        
        try:
            print("\nRMS Normalization Benchmark:")
            print(f"{'Size':<15} {'Time (ms)':<12} {'Throughput (GFLOPS)':<20}")
            print("-" * 50)
            
            for rows, dim in test_sizes:
                x = np.random.randn(rows, dim).astype(np.float32)
                w = np.random.randn(dim).astype(np.float32)
                
                # Warmup
                for _ in range(5):
                    _ = backend_module.rms_norm(x, w)
                
                # Benchmark
                start = time.perf_counter()
                for _ in range(iterations):
                    _ = backend_module.rms_norm(x, w)
                elapsed = time.perf_counter() - start
                
                # Calculate throughput (approximate)
                ops = rows * dim * 3 * iterations
                gflops = (ops / elapsed) / 1e9
                
                avg_time_ms = (elapsed / iterations) * 1000
                print(f"{rows}x{dim:<10} {avg_time_ms:<12.3f} {gflops:<20.2f}")
            
            print("\nActivation Functions Benchmark (SiLU):")
            print(f"{'Size':<15} {'Time (ms)':<12} {'Throughput (GFLOPS)':<20}")
            print("-" * 50)
            
            for rows, dim in test_sizes:
                x = np.random.randn(rows, dim).astype(np.float32)
                
                # Warmup
                for _ in range(5):
                    _ = backend_module.silu(x)
                
                # Benchmark
                start = time.perf_counter()
                for _ in range(iterations):
                    _ = backend_module.silu(x)
                elapsed = time.perf_counter() - start
                
                # SiLU: exp + division + multiplication per element (~5 ops)
                ops = rows * dim * 5 * iterations
                gflops = (ops / elapsed) / 1e9
                
                avg_time_ms = (elapsed / iterations) * 1000
                print(f"{rows}x{dim:<10} {avg_time_ms:<12.3f} {gflops:<20.2f}")
        
        except Exception as e:
            print(f"✗ {backend_name} benchmark failed: {e}")
            continue
    
    return True

def main():
    """Main function."""
    print_header("Tensor Operations Capability Detection and Benchmark")
    print(f"\nPython: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Change to the cpp_ext directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run detection
    if not detect_capabilities():
        print("\n✗ Capability detection failed")
        return 1
    
    # Build extension
    if not build_extension():
        print("\n✗ Build failed")
        return 1
    
    # Test extension
    if not test_extension():
        print("\n✗ Tests failed")
        return 1
    
    # Benchmark
    if not benchmark_extension():
        print("\n✗ Benchmark failed")
        return 1
    
    print_header("All Steps Completed Successfully!")
    print("\nNext steps:")
    print("  - Use 'import tensor_ops_cpp' in your Python code")
    print("  - Check tensor_ops_cpp.get_optimization_info() for details")
    print("  - GPU backends available:")
    print("    * CUDA: python setup.py build_cuda")
    print("    * OpenCL: python setup.py build_opencl")
    print("  - Unified API: Use airllm.tensor_ops for automatic backend selection")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
