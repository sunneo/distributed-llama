# C++ Extension Module for Distributed-Llama Python Worker

This directory contains C++ implementations of critical tensor operations with Python bindings via pybind11.

## Features

- **RMS Normalization**: SIMD-optimized (AVX2/NEON)
- **SiLU Activation**: Optimized sigmoid linear unit
- **GELU Activation**: Gaussian error linear unit
- **Matrix Multiplication**: Simple implementation (use NumPy/BLAS for production)

## Building

### Prerequisites

```bash
pip install pybind11 numpy
```

### Build Commands

```bash
cd mix/target/airllm/cpp_ext

# Option 1: Using setup.py (recommended)
python setup.py build_ext --inplace

# Option 2: Manual compilation
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python3 -m pybind11 --includes) \
    -o tensor_ops_cpp$(python3-config --extension-suffix) \
    tensor_ops_cpp.cpp
```

### Verify Installation

```python
import tensor_ops_cpp
print(f"SIMD support: {tensor_ops_cpp.has_simd}")
```

## Usage

### Direct Usage

```python
import numpy as np
import tensor_ops_cpp

# RMS normalization
x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)
y = tensor_ops_cpp.rms_norm(x, weight, eps=1e-6)

# SiLU activation
x = np.random.randn(128, 4096).astype(np.float32)
y = tensor_ops_cpp.silu(x)
```

### Hybrid Module (Automatic Fallback)

The hybrid module automatically uses C++ when available:

```python
from airllm import tensor_ops_hybrid

# Will use C++ if available, otherwise Python
y = tensor_ops_hybrid.rms_norm(x, weight)

# Check which backend is being used
tensor_ops_hybrid.print_backend_info()
```

## Performance

Expected speedups compared to Python/NumPy:

| Operation | Speedup | Notes |
|-----------|---------|-------|
| RMS Norm  | 3-5x    | SIMD-optimized |
| SiLU      | 2-3x    | Reduced function call overhead |
| GELU      | 2-3x    | Reduced function call overhead |
| Matmul    | ~1x     | NumPy already uses BLAS |

## SIMD Support

The module automatically detects and uses SIMD instructions:

- **AVX2**: On Intel/AMD x86_64 CPUs with AVX2
- **NEON**: On ARM CPUs with NEON
- **Scalar**: Fallback for other architectures

Check SIMD support:

```python
import tensor_ops_cpp
print(tensor_ops_cpp.has_simd)  # True if SIMD is available
```

## Architecture-Specific Optimization

### Intel/AMD (x86_64)

Build with AVX2:
```bash
c++ -O3 -mavx2 -mfma ... tensor_ops_cpp.cpp
```

### ARM (Apple Silicon, Raspberry Pi)

Build with NEON:
```bash
c++ -O3 -march=armv8-a+simd ... tensor_ops_cpp.cpp
```

## Testing

```bash
cd mix/target/airllm/cpp_ext
python test_cpp_ext.py
```

## Benchmarking

Compare Python vs C++ performance:

```bash
cd mix/target
python profile_worker.py  # Baseline
# Then build C++ extension and run again to see improvement
```

## Troubleshooting

### "ImportError: No module named 'tensor_ops_cpp'"

The extension wasn't built. Run:
```bash
python setup.py build_ext --inplace
```

### Build Errors

Make sure you have:
- C++ compiler (g++, clang)
- Python development headers
- pybind11

On Ubuntu/Debian:
```bash
sudo apt-get install python3-dev g++
pip install pybind11
```

### Runtime Errors

Check that NumPy arrays are:
- dtype=np.float32 (not float64)
- C-contiguous (use .ascontiguous() if needed)

## Integration with Worker

The C++ extensions are automatically used by the layer engine when available.

To enable in your worker:

```python
from airllm.layer_engine import LayerWiseInferenceEngine
from airllm import tensor_ops_hybrid

# Engine will automatically use hybrid ops
engine = LayerWiseInferenceEngine(model_path)
```

## Future Improvements

Potential areas for further optimization:

1. **Fused Operations**: Combine multiple ops (e.g., matmul + activation)
2. **Better Matmul**: Integrate with BLAS or use optimized kernels
3. **Quantized Operations**: Q8_0/Q4_0 matmul in C++
4. **Multi-threading**: Parallel execution for batch operations
5. **GPU Support**: Vulkan/CUDA kernels via C++ interface

## References

- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
