# C++ Extension Module for Distributed-Llama Python Worker

This directory contains C++ implementations of critical tensor operations with Python bindings via pybind11.

## Features

- **RMS Normalization**: Multi-level SIMD optimization (AVX-512/AVX2/AVX/NEON)
- **SiLU Activation**: Optimized sigmoid linear unit with OpenMP
- **GELU Activation**: Gaussian error linear unit with OpenMP
- **Matrix Multiplication**: Blocked implementation with SIMD and OpenMP
- **Automatic Capability Detection**: Detects and uses best available CPU features

## Optimization Levels

The module automatically detects and uses the best available optimizations:

| Level | Technologies | Description |
|-------|-------------|-------------|
| **AVX-512** | AVX-512F, FMA, OpenMP | Highest performance (16-wide SIMD vectors) |
| **AVX2** | AVX2, FMA, OpenMP | High performance (8-wide SIMD vectors) |
| **AVX** | AVX, OpenMP | Good performance (8-wide SIMD vectors) |
| **NEON** | ARM NEON, OpenMP | ARM CPU optimization (4-wide SIMD vectors) |
| **Scalar** | OpenMP only | Fallback for all architectures |

## Building

### Prerequisites

```bash
pip install pybind11 numpy
```

### Test Available Capabilities

Before building, test what optimizations your system supports:

```bash
cd mix/target/airllm/cpp_ext
python setup.py test_capabilities
```

This will output something like:

```
Detecting CPU and compiler capabilities...

Platform: Linux x86_64
Compiler: g++

OpenMP          ✓ Available
AVX             ✓ Available
AVX2            ✓ Available
AVX-512         ✗ Not available
FMA             ✓ Available
ARM NEON        ✗ Not available
CUDA            ✗ Not available
OpenCL          ✗ Not available
Vulkan          ✗ Not available

Recommended build configuration:
  Compile flags: -O3 -std=c++11 -march=native -fopenmp -mavx2 -mfma
```

### Build with Auto-Detected Optimizations

The setup.py automatically detects and uses the best optimizations:

```bash
cd mix/target/airllm/cpp_ext
python setup.py build_ext --inplace
```

The build will:
1. Detect available CPU features by checking /proc/cpuinfo (Linux)
2. Test if compiler supports various instruction sets
3. Run test programs to verify features work at runtime
4. Build with the best available optimizations

### Verify Installation

```python
import tensor_ops_cpp

# Check what optimizations are enabled
print(tensor_ops_cpp.get_optimization_info())

# Check individual capabilities
print(f"SIMD Level: {tensor_ops_cpp.simd_level}")
print(f"OpenMP: {tensor_ops_cpp.has_openmp}")
print(f"AVX2: {tensor_ops_cpp.has_avx2}")
print(f"FMA: {tensor_ops_cpp.has_fma}")
```
## Usage

### Quick Test

After building, test the extension:

```bash
cd mix/target/airllm/cpp_ext

# Quick functionality test
python -c "import tensor_ops_cpp; print(tensor_ops_cpp.get_optimization_info())"

# Run comprehensive examples
python examples.py

# Run detection and benchmark
python detect_and_benchmark.py
```

### Direct Usage

```python
import numpy as np
import tensor_ops_cpp

# Check optimization info
print(tensor_ops_cpp.get_optimization_info())

# RMS normalization
x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)
y = tensor_ops_cpp.rms_norm(x, weight, eps=1e-6)

# SiLU activation
x = np.random.randn(128, 4096).astype(np.float32)
y = tensor_ops_cpp.silu(x)

# GELU activation
y = tensor_ops_cpp.gelu(x)

# Matrix multiplication
a = np.random.randn(128, 256).astype(np.float32)
b = np.random.randn(256, 512).astype(np.float32)
c = tensor_ops_cpp.matmul(a, b)
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

**Note**: The hybrid module is part of the airllm package and requires additional dependencies (psutil, etc.). For standalone usage of the C++ extension, use the direct import method above.

## Performance

Expected speedups compared to Python/NumPy (with AVX2+FMA+OpenMP):

| Operation | Speedup | Notes |
|-----------|---------|-------|
| RMS Norm  | 5-15x   | SIMD + OpenMP parallelization |
| SiLU      | 2-4x    | OpenMP parallelization |
| GELU      | 2-4x    | OpenMP parallelization |
| Matmul    | 2-5x    | SIMD + blocking + OpenMP |

Actual speedup depends on:
- CPU architecture and capabilities
- Problem size (larger tensors benefit more from parallelization)
- Number of CPU cores available

## Optimization Details

### SIMD Optimization

The module uses different SIMD instruction sets based on CPU support:

- **AVX-512**: 16 floats per operation (Intel Skylake-X and newer)
- **AVX2**: 8 floats per operation with FMA (Intel Haswell and newer)
- **AVX**: 8 floats per operation (Intel Sandy Bridge and newer)
- **NEON**: 4 floats per operation (ARM Cortex-A series)
- **Scalar**: Fallback for all architectures

Check your SIMD level:

```python
import tensor_ops_cpp
print(f"SIMD Level: {tensor_ops_cpp.simd_level}")
```

### OpenMP Parallelization

Operations are automatically parallelized across CPU cores when:
- OpenMP is available (detected at build time)
- Problem size is large enough to benefit from parallelization
- Multiple CPU cores are available

OpenMP is used for:
- RMS Norm: Parallel processing of independent rows
- SiLU/GELU: Parallel processing when tensor size > 8192 elements
- Matmul: Parallel outer loop for row-wise computation

Control thread count:

```python
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Set before importing the module
import tensor_ops_cpp
```

## Capability Detection API

### Runtime Detection

Query optimization capabilities at runtime:

```python
import tensor_ops_cpp

# Get detailed info string
print(tensor_ops_cpp.get_optimization_info())

# Check individual capabilities
if tensor_ops_cpp.has_avx2 and tensor_ops_cpp.has_fma:
    print("Using AVX2 with FMA - excellent performance!")
elif tensor_ops_cpp.has_avx:
    print("Using AVX - good performance")
else:
    print("Using scalar code - consider upgrading CPU")

if tensor_ops_cpp.has_openmp:
    print("OpenMP parallelization enabled")
```

### Build-Time Detection

Test capabilities before building:

```bash
python setup.py test_capabilities
```

This tests:
1. **Compiler availability**: Checks for g++, clang++, or c++
2. **CPU features**: Reads /proc/cpuinfo on Linux
3. **Compile-time support**: Tests if compiler supports flags
4. **Runtime support**: Compiles and runs test programs to verify features work

## Advanced Build Options

### Manual Feature Selection

If you want to disable certain optimizations:

```bash
# Build without OpenMP (single-threaded)
CC=gcc CXX=g++ python setup.py build_ext --inplace

# Then manually edit the built extension's compile flags
```

### Architecture-Specific Builds

The build system uses `-march=native` which optimizes for the build machine's CPU. For portable builds targeting older CPUs:

```bash
# Build for generic x86-64 (no SIMD)
CFLAGS="-O3 -mtune=generic" python setup.py build_ext --inplace

# Build with AVX2 only (no AVX-512)
CFLAGS="-O3 -mavx2 -mfma" python setup.py build_ext --inplace
```

## Testing

### Basic Functionality Test

```bash
cd mix/target/airllm/cpp_ext
python test_cpp_ext.py
```

This tests:
- Extension import and capability reporting
- RMS normalization correctness
- Activation function correctness
- Performance benchmarking

### Quick Test

```python
import tensor_ops_cpp
import numpy as np

# Check what's enabled
print(tensor_ops_cpp.get_optimization_info())

# Simple test
x = np.random.randn(10, 100).astype(np.float32)
w = np.random.randn(100).astype(np.float32)
result = tensor_ops_cpp.rms_norm(x, w)
print(f"Test passed! Shape: {result.shape}")
```

## Benchmarking

Compare performance across different optimization levels:

```bash
# Build with maximum optimizations
python setup.py build_ext --inplace

# Run benchmarks
cd mix/target
python profile_worker.py  # If available
```

## Troubleshooting

### "ImportError: No module named 'tensor_ops_cpp'"

The extension wasn't built. Run:
```bash
python setup.py build_ext --inplace
```

### "Illegal instruction" or Segfault

The binary was built with CPU features not supported at runtime. This can happen if:
- You copied the .so file from another machine
- You used `-march=native` on a newer CPU and ran on older CPU

Solution: Rebuild on the target machine:
```bash
rm -rf build *.so
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

On macOS:
```bash
xcode-select --install
pip install pybind11
```

### Low Performance / No Speedup

Check what optimizations are enabled:

```python
import tensor_ops_cpp
print(tensor_ops_cpp.get_optimization_info())
```

If SIMD or OpenMP are disabled:
1. Verify your CPU supports these features (check /proc/cpuinfo)
2. Verify your compiler supports these features (gcc 4.9+ for AVX2)
3. Rebuild the extension

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

## Future Improvements and GPU Support

The capability detection system is ready to support additional backends:

### GPU Backends (Experimental)

GPU backend implementations are provided but require specific hardware and drivers:

#### 1. CUDA Backend (NVIDIA GPUs)

**Status**: Source code provided (`tensor_ops_cuda.cu`)  
**Requirements**: 
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- cuDNN (optional, for additional optimizations)

**Detection**:
```bash
python setup.py test_capabilities
```

**Build** (when CUDA is available):
```bash
python setup.py build_cuda
```

Or manually:
```bash
nvcc -O3 -shared -Xcompiler -fPIC \
    $(python3 -m pybind11 --includes) \
    -o tensor_ops_cuda$(python3-config --extension-suffix) \
    tensor_ops_cuda.cu
```

**Usage**:
```python
import tensor_ops_cuda
print(tensor_ops_cuda.get_cuda_info())
result = tensor_ops_cuda.rms_norm(x, weight)
```

#### 2. OpenCL Backend (Cross-Platform GPU)

**Status**: Source code provided (`tensor_ops_opencl.cpp`)  
**Requirements**:
- OpenCL-compatible GPU (NVIDIA, AMD, Intel)
- OpenCL runtime and drivers
- OpenCL headers

**Detection**:
```bash
python setup.py test_capabilities
```

**Build** (when OpenCL is available):
```bash
python setup.py build_opencl
```

Or manually:
```bash
g++ -O3 -shared -std=c++11 -fPIC \
    $(python3 -m pybind11 --includes) \
    -lOpenCL \
    -o tensor_ops_opencl$(python3-config --extension-suffix) \
    tensor_ops_opencl.cpp
```

**Usage**:
```python
import tensor_ops_opencl
print(tensor_ops_opencl.get_opencl_info())
result = tensor_ops_opencl.rms_norm(x, weight)
```

#### 3. Vulkan Backend (Modern GPU API)

**Status**: Planned (detection implemented)  
**Requirements**:
- Vulkan-compatible GPU
- Vulkan SDK
- Compute shader support

**Note**: Vulkan backend implementation is planned but not yet available. The main distributed-llama project already has Vulkan support in the C++ codebase (see `src/nn/nn-vulkan.cpp`).

## Automated Backend Selection

The `tensor_ops` module provides automatic backend selection that picks the fastest available backend:

```python
from airllm import tensor_ops

# Automatically uses: CUDA > OpenCL > C++ > Python
print(f"Using backend: {tensor_ops.get_backend()}")
print("Backend info:", tensor_ops.get_backend_info())

# Use tensor operations - automatically accelerated
import numpy as np
x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)

# These automatically use the best available backend
y = tensor_ops.rms_norm(x, weight)
y_silu = tensor_ops.silu(x)
y_gelu = tensor_ops.gelu(x)
```

**Backend priority order**:
1. CUDA (if available) - Best for NVIDIA GPUs
2. OpenCL (if available) - Good for AMD/Intel GPUs
3. C++ (if built) - Fast CPU operations with SIMD+OpenMP
4. Python (always available) - NumPy fallback

**See also**: `BACKEND_GUIDE.md` and `backend_examples.py` for detailed usage examples.

## Choosing the Right Backend

**For CPU-only systems**:
- Use the default CPU backend (automatically optimized)
- Best for: Development, small models, systems without GPU

**For NVIDIA GPUs**:
- Use CUDA backend for best performance
- Best for: Large models, production inference on NVIDIA hardware

**For AMD/Intel GPUs**:
- Use OpenCL backend for cross-platform GPU support
- Best for: Non-NVIDIA GPUs, portable code

**For maximum portability**:
- Use CPU backend - works everywhere with automatic optimization
- Good performance on modern CPUs with AVX2+OpenMP

### Automated Backend Selection

You can create a wrapper that automatically selects the best available backend:

```python
def get_best_backend():
    """Auto-select best available backend."""
    backends = []
    
    # Try CUDA first (fastest for NVIDIA)
    try:
        import tensor_ops_cuda
        backends.append(('CUDA', tensor_ops_cuda))
    except ImportError:
        pass
    
    # Try OpenCL (good for AMD/Intel GPUs)
    try:
        import tensor_ops_opencl
        backends.append(('OpenCL', tensor_ops_opencl))
    except ImportError:
        pass
    
    # Fall back to CPU (always available)
    try:
        import tensor_ops_cpp
        backends.append(('CPU', tensor_ops_cpp))
    except ImportError:
        pass
    
    if backends:
        name, backend = backends[0]
        print(f"Using {name} backend")
        return backend
    else:
        raise ImportError("No tensor ops backend available")

# Use in your code
tensor_ops = get_best_backend()
result = tensor_ops.rms_norm(x, weight)
```

### Future CPU Optimizations

1. **Fused Operations**: Combine multiple ops (e.g., matmul + activation)
2. **Better Matmul**: Integrate with BLAS or use optimized kernels
3. **Quantized Operations**: Q8_0/Q4_0 matmul in C++
4. **ARM SVE**: Scalable Vector Extension for newer ARM CPUs
5. **RISC-V Vector**: Vector extension support

## References

- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [OpenMP Documentation](https://www.openmp.org/)
- [AVX-512 Programming](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
