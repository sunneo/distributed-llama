"""
Hybrid tensor operations module.

This module automatically uses C++ implementations when available,
falling back to Python implementations otherwise.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import C++ extensions
try:
    import tensor_ops_cpp
    HAS_CPP_EXT = True
    HAS_SIMD = tensor_ops_cpp.has_simd
except ImportError:
    HAS_CPP_EXT = False
    HAS_SIMD = False

# Import Python fallback
try:
    from . import tensor_ops as tensor_ops_py
except ImportError:
    # Fallback for standalone usage
    import tensor_ops as tensor_ops_py


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    RMS normalization (auto-selects C++ or Python).
    """
    if HAS_CPP_EXT:
        return tensor_ops_cpp.rms_norm(x, weight, eps)
    else:
        return tensor_ops_py.rms_norm(x, weight, eps)


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU activation (auto-selects C++ or Python).
    """
    if HAS_CPP_EXT:
        return tensor_ops_cpp.silu(x)
    else:
        return tensor_ops_py.silu(x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (auto-selects C++ or Python).
    """
    if HAS_CPP_EXT:
        return tensor_ops_cpp.gelu(x)
    else:
        return tensor_ops_py.gelu(x)


def matmul_f32(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication (auto-selects C++ or Python).
    
    Note: For production use, NumPy's matmul (which uses BLAS) is typically
    faster than our simple C++ implementation. C++ extension is mainly
    for demonstration and can be improved with BLAS integration.
    """
    # Always use NumPy for matmul as it's already optimized with BLAS
    return tensor_ops_py.matmul_f32(x, weight)


# Re-export other operations from Python module
apply_rope = tensor_ops_py.apply_rope
multi_head_attention = tensor_ops_py.multi_head_attention
feed_forward = tensor_ops_py.feed_forward
matmul_quantized = tensor_ops_py.matmul_quantized
dequantize_q40 = tensor_ops_py.dequantize_q40
dequantize_q80 = tensor_ops_py.dequantize_q80


def get_backend_info() -> dict:
    """
    Get information about which backend is being used.
    
    Returns:
        Dictionary with backend information
    """
    info = {
        'has_cpp_ext': HAS_CPP_EXT,
        'has_simd': HAS_SIMD,
        'rms_norm': 'C++' if HAS_CPP_EXT else 'Python',
        'silu': 'C++' if HAS_CPP_EXT else 'Python',
        'gelu': 'C++' if HAS_CPP_EXT else 'Python',
        'matmul': 'NumPy (BLAS)',
        'other_ops': 'Python'
    }
    
    # Add detailed optimization info if C++ extension is available
    if HAS_CPP_EXT:
        info['simd_level'] = getattr(tensor_ops_cpp, 'simd_level', 'Unknown')
        info['has_openmp'] = getattr(tensor_ops_cpp, 'has_openmp', False)
        info['has_fma'] = getattr(tensor_ops_cpp, 'has_fma', False)
        info['has_avx2'] = getattr(tensor_ops_cpp, 'has_avx2', False)
        info['has_avx512'] = getattr(tensor_ops_cpp, 'has_avx512', False)
        info['has_neon'] = getattr(tensor_ops_cpp, 'has_neon', False)
    
    return info


def print_backend_info():
    """Print backend information."""
    info = get_backend_info()
    print("Tensor Operations Backend:")
    print(f"  C++ extensions: {'✓' if info['has_cpp_ext'] else '✗'}")
    print(f"  SIMD support:   {'✓' if info['has_simd'] else '✗'}")
    
    if info['has_cpp_ext']:
        print(f"  SIMD level:     {info.get('simd_level', 'Unknown')}")
        print(f"  OpenMP:         {'✓' if info.get('has_openmp') else '✗'}")
        print(f"  FMA:            {'✓' if info.get('has_fma') else '✗'}")
    
    print(f"\nOperation implementations:")
    print(f"  RMS norm:       {info['rms_norm']}")
    print(f"  SiLU:           {info['silu']}")
    print(f"  GELU:           {info['gelu']}")
    print(f"  Matmul:         {info['matmul']}")
    print(f"  Other ops:      {info['other_ops']}")
    
    if HAS_CPP_EXT:
        print(f"\nDetailed optimization info:")
        try:
            print(tensor_ops_cpp.get_optimization_info())
        except:
            pass


# Print info on import
if __name__ != '__main__':
    # Silently check for C++ extensions on import
    pass
