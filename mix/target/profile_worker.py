"""
Profiling script for identifying performance bottlenecks in Python implementation.

This script profiles key tensor operations to identify which ones would benefit
most from C++ acceleration.
"""

import numpy as np
import time
from typing import Dict, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm import tensor_ops


def profile_function(func: Callable, *args, iterations: int = 100, **kwargs) -> Dict:
    """
    Profile a function execution.
    
    Args:
        func: Function to profile
        args: Positional arguments
        iterations: Number of iterations
        kwargs: Keyword arguments
        
    Returns:
        Dictionary with profiling results
    """
    # Warm-up
    for _ in range(5):
        _ = func(*args, **kwargs)
    
    # Profile
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end = time.perf_counter()
    
    total_time = end - start
    avg_time = total_time / iterations
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'iterations': iterations,
        'throughput': iterations / total_time
    }


def profile_rms_norm():
    """Profile RMS normalization."""
    print("\n=== Profiling RMS Normalization ===")
    
    sizes = [
        (1, 4096),     # Single token
        (128, 4096),   # Small batch
        (512, 4096),   # Large batch
    ]
    
    results = []
    for shape in sizes:
        x = np.random.randn(*shape).astype(np.float32)
        weight = np.random.randn(shape[1]).astype(np.float32)
        
        stats = profile_function(tensor_ops.rms_norm, x, weight, iterations=1000)
        
        print(f"  Shape {shape}: {stats['avg_time']*1000:.3f} ms/iter, "
              f"{stats['throughput']:.1f} ops/sec")
        
        results.append({
            'operation': 'rms_norm',
            'shape': shape,
            **stats
        })
    
    return results


def profile_matmul():
    """Profile matrix multiplication."""
    print("\n=== Profiling Matrix Multiplication ===")
    
    configs = [
        # (input_shape, weight_shape, description)
        ((1, 4096), (4096, 4096), "Q/K/V projection"),
        ((1, 4096), (4096, 11008), "FFN up-projection"),
        ((1, 11008), (11008, 4096), "FFN down-projection"),
        ((128, 4096), (4096, 4096), "Batch Q/K/V"),
    ]
    
    results = []
    for x_shape, w_shape, desc in configs:
        x = np.random.randn(*x_shape).astype(np.float32)
        w = np.random.randn(*w_shape).astype(np.float32)
        
        stats = profile_function(tensor_ops.matmul_f32, x, w, iterations=100)
        
        # Calculate FLOPS
        m, k = x_shape
        k2, n = w_shape
        flops = 2 * m * k * n  # 2 for multiply-add
        gflops = (flops / stats['avg_time']) / 1e9
        
        print(f"  {desc} {x_shape} × {w_shape}: {stats['avg_time']*1000:.3f} ms/iter, "
              f"{gflops:.2f} GFLOPS")
        
        results.append({
            'operation': 'matmul_f32',
            'description': desc,
            'x_shape': x_shape,
            'w_shape': w_shape,
            'gflops': gflops,
            **stats
        })
    
    return results


def profile_rope():
    """Profile RoPE (Rotary Position Embedding)."""
    print("\n=== Profiling RoPE ===")
    
    configs = [
        (1, 4096, 128),    # Single token, 32 heads
        (128, 4096, 128),  # Batch
    ]
    
    results = []
    for seq_len, q_dim, head_dim in configs:
        q = np.random.randn(seq_len, q_dim).astype(np.float32)
        k = np.random.randn(seq_len, q_dim // 4).astype(np.float32)  # GQA: fewer KV heads
        
        stats = profile_function(tensor_ops.apply_rope, q, k, 0, head_dim, iterations=1000)
        
        print(f"  seq_len={seq_len}, q_dim={q_dim}: {stats['avg_time']*1000:.3f} ms/iter")
        
        results.append({
            'operation': 'apply_rope',
            'seq_len': seq_len,
            'q_dim': q_dim,
            'head_dim': head_dim,
            **stats
        })
    
    return results


def profile_attention():
    """Profile multi-head attention."""
    print("\n=== Profiling Multi-Head Attention ===")
    
    configs = [
        # (seq_len, q_dim, n_heads, n_kv_heads)
        (1, 4096, 32, 8),      # Single token, GQA
        (128, 4096, 32, 8),    # Batch
        (512, 4096, 32, 8),    # Large context
    ]
    
    results = []
    for seq_len, q_dim, n_heads, n_kv_heads in configs:
        head_dim = q_dim // n_heads
        kv_dim = n_kv_heads * head_dim
        
        q = np.random.randn(seq_len, q_dim).astype(np.float32)
        k = np.random.randn(seq_len, kv_dim).astype(np.float32)
        v = np.random.randn(seq_len, kv_dim).astype(np.float32)
        
        # Fewer iterations for expensive ops
        iters = 100 if seq_len <= 128 else 10
        stats = profile_function(tensor_ops.multi_head_attention, q, k, v, 
                               n_heads, n_kv_heads, iterations=iters)
        
        print(f"  seq_len={seq_len}, heads={n_heads}/{n_kv_heads}: "
              f"{stats['avg_time']*1000:.3f} ms/iter")
        
        results.append({
            'operation': 'multi_head_attention',
            'seq_len': seq_len,
            'q_dim': q_dim,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            **stats
        })
    
    return results


def profile_activations():
    """Profile activation functions."""
    print("\n=== Profiling Activation Functions ===")
    
    shape = (1, 11008)  # FFN hidden dim
    x = np.random.randn(*shape).astype(np.float32)
    
    results = []
    
    # SiLU
    stats = profile_function(tensor_ops.silu, x, iterations=1000)
    print(f"  SiLU: {stats['avg_time']*1000:.3f} ms/iter")
    results.append({'operation': 'silu', 'shape': shape, **stats})
    
    # GELU
    stats = profile_function(tensor_ops.gelu, x, iterations=1000)
    print(f"  GELU: {stats['avg_time']*1000:.3f} ms/iter")
    results.append({'operation': 'gelu', 'shape': shape, **stats})
    
    return results


def profile_ffn():
    """Profile feed-forward network."""
    print("\n=== Profiling Feed-Forward Network ===")
    
    configs = [
        (1, 4096, 11008),    # Single token
        (128, 4096, 11008),  # Batch
    ]
    
    results = []
    for seq_len, dim, hidden_dim in configs:
        x = np.random.randn(seq_len, dim).astype(np.float32)
        w1 = np.random.randn(dim, hidden_dim).astype(np.float32)
        w2 = np.random.randn(hidden_dim, dim).astype(np.float32)
        w3 = np.random.randn(dim, hidden_dim).astype(np.float32)
        
        stats = profile_function(tensor_ops.feed_forward, x, w1, w2, w3, 
                               iterations=100)
        
        print(f"  seq_len={seq_len}: {stats['avg_time']*1000:.3f} ms/iter")
        
        results.append({
            'operation': 'feed_forward',
            'seq_len': seq_len,
            'dim': dim,
            'hidden_dim': hidden_dim,
            **stats
        })
    
    return results


def analyze_results(all_results):
    """Analyze profiling results and identify hotspots."""
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    # Group by operation
    by_operation = {}
    for result in all_results:
        op = result['operation']
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(result)
    
    # Calculate total time per operation type
    print("\nTotal time per operation (for profiled iterations):")
    operation_times = []
    for op, results in by_operation.items():
        total_time = sum(r['total_time'] for r in results)
        operation_times.append((op, total_time))
    
    operation_times.sort(key=lambda x: x[1], reverse=True)
    
    total_all = sum(t for _, t in operation_times)
    for op, time in operation_times:
        pct = (time / total_all) * 100
        print(f"  {op:30s}: {time:.3f}s ({pct:.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR C++ ACCELERATION")
    print("=" * 70)
    
    print("\n🎯 TOP PRIORITIES (highest impact):")
    print("  1. Matrix multiplication (matmul_f32)")
    print("     - Dominates computation time")
    print("     - Use SIMD/BLAS acceleration")
    print("     - Priority: CRITICAL")
    
    print("\n  2. Multi-head attention")
    print("     - Complex operation with matmuls + softmax")
    print("     - Includes Q@K^T and attn@V")
    print("     - Priority: HIGH")
    
    print("\n  3. Feed-forward network")
    print("     - Contains 3 matmuls + activations")
    print("     - Can benefit from fused operations")
    print("     - Priority: HIGH")
    
    print("\n⚡ MEDIUM PRIORITIES:")
    print("  4. RMS normalization")
    print("     - Called frequently but relatively fast")
    print("     - Easy to optimize with SIMD")
    print("     - Priority: MEDIUM")
    
    print("\n  5. RoPE (Rotary Position Embedding)")
    print("     - Transcendental functions (sin/cos)")
    print("     - Can use lookup tables")
    print("     - Priority: MEDIUM")
    
    print("\n💡 SUGGESTED C++ IMPLEMENTATION ORDER:")
    print("  1. Start with matmul_f32 - use existing nn-cpu-ops.cpp")
    print("  2. Add RMS norm - simple SIMD operation")
    print("  3. Add SiLU/GELU activations")
    print("  4. Add RoPE with precomputed sin/cos tables")
    print("  5. Finally, add fused attention and FFN")
    
    print("\n📊 EXPECTED PERFORMANCE GAINS:")
    print("  - Matmul (BLAS): 5-10x speedup")
    print("  - RMS norm (SIMD): 3-5x speedup")
    print("  - Activations (SIMD): 2-4x speedup")
    print("  - Overall: 3-7x speedup for forward pass")


def main():
    """Run all profiling benchmarks."""
    print("=" * 70)
    print("DISTRIBUTED-LLAMA PYTHON WORKER PROFILING")
    print("=" * 70)
    print(f"\nSystem: {sys.platform}")
    print(f"NumPy version: {np.__version__}")
    
    all_results = []
    
    # Profile each component
    all_results.extend(profile_rms_norm())
    all_results.extend(profile_matmul())
    all_results.extend(profile_rope())
    all_results.extend(profile_attention())
    all_results.extend(profile_activations())
    all_results.extend(profile_ffn())
    
    # Analyze and provide recommendations
    analyze_results(all_results)
    
    print("\n" + "=" * 70)
    print("Profiling complete! Use results to guide C++ optimization.")
    print("=" * 70)


if __name__ == '__main__':
    main()
