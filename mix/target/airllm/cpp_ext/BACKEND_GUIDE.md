# Backend Guide for Tensor Operations

This guide explains how to use different acceleration backends (CPU, CUDA, OpenCL) for tensor operations in the distributed-llama project.

## Available Backends

The tensor operations module supports multiple backends with automatic fallback:

1. **CUDA** - NVIDIA GPU acceleration (highest performance for NVIDIA GPUs)
2. **OpenCL** - Cross-platform GPU acceleration (works on NVIDIA, AMD, Intel GPUs)
3. **C++** - CPU optimization with SIMD (AVX, AVX2, AVX-512) and OpenMP
4. **Python** - Pure NumPy fallback (always available)

## Quick Start

### Automatic Backend Selection

The `tensor_ops` module automatically detects and uses the best available backend:

```python
from airllm import tensor_ops
import numpy as np

# Check which backend is active
print("Active backend:", tensor_ops.get_backend())
print("Backend info:", tensor_ops.get_backend_info())

# Use tensor operations (automatically accelerated)
x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)

# RMS normalization
y = tensor_ops.rms_norm(x, weight)

# Activation functions
y_silu = tensor_ops.silu(x)
y_gelu = tensor_ops.gelu(x)
```

### Configurable Backend Selection

You can configure which backend to use via `backend.json` or setup.py:

```bash
# View current configuration
cd mix/target/airllm/cpp_ext
python setup.py configure_backend --show

# Set preferred backend
python setup.py configure_backend --backend=cpp

# Set backend priority
python setup.py configure_backend --priority="cpp,cuda,opencl,python"
```

**See `BACKEND_CONFIG.md` for detailed configuration options.**

## Building Backends

### 1. Detect Available Capabilities

First, check what hardware capabilities are available on your system:

```bash
cd mix/target/airllm/cpp_ext
python setup.py test_capabilities
```

This will show which backends can be built on your system.

### 2. Build C++ Backend (CPU Optimization)

The C++ backend provides CPU optimization with SIMD and OpenMP:

```bash
cd mix/target/airllm/cpp_ext
python setup.py build_ext --inplace
```

This automatically detects and enables:
- AVX, AVX2, AVX-512 (if supported)
- FMA (Fused Multiply-Add)
- OpenMP (multi-threading)

### 3. Build CUDA Backend (NVIDIA GPU)

Requirements:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (https://developer.nvidia.com/cuda-downloads)

Build command:
```bash
cd mix/target/airllm/cpp_ext
python setup.py build_cuda
```

**Note:** The CUDA build now uses C++14 to fix template parameter pack issues in older CUDA versions.

### 4. Build OpenCL Backend (Multi-GPU Support)

Requirements:
- OpenCL-capable GPU (NVIDIA, AMD, Intel)
- OpenCL drivers installed

Build command:
```bash
cd mix/target/airllm/cpp_ext
python setup.py build_opencl
```

## Testing and Benchmarking

### Comprehensive Test and Benchmark

Run the complete detection, build, test, and benchmark workflow:

```bash
cd mix/target/airllm/cpp_ext
python detect_and_benchmark.py
```

This script will:
1. Detect system capabilities
2. Build C++ extension with optimal flags
3. Test all available backends
4. Benchmark performance

### Unit Tests

Test specific backends:

```bash
cd mix/target/airllm/cpp_ext
python test_cpp_ext.py
```

## Using Specific Backends

If you want to use a specific backend instead of automatic selection:

### CUDA Backend
```python
import tensor_ops_cuda
import numpy as np

x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)

# Get CUDA device info
print(tensor_ops_cuda.get_cuda_info())

# Use CUDA operations
y = tensor_ops_cuda.rms_norm(x, weight)
```

### OpenCL Backend
```python
import tensor_ops_opencl
import numpy as np

x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)

# Get OpenCL device info
print(tensor_ops_opencl.get_opencl_info())

# Use OpenCL operations
y = tensor_ops_opencl.rms_norm(x, weight)
```

### C++ Backend
```python
import tensor_ops_cpp
import numpy as np

x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)

# Get optimization info
print(tensor_ops_cpp.get_optimization_info())

# Use C++ operations
y = tensor_ops_cpp.rms_norm(x, weight)
```

## Backend Priority

When using automatic backend selection (`from airllm import tensor_ops`), the priority is:

1. CUDA (if available)
2. OpenCL (if available)
3. C++ (if built)
4. Python (always available)

## Troubleshooting

### CUDA Build Fails

**Error:** `parameter packs not expanded with '...'`

**Solution:** This error occurs with older CUDA versions. The setup.py has been updated to use C++14 which fixes this issue.

If you still encounter issues, try:
```bash
# Manually specify CUDA version
nvcc --version
# Use appropriate C++ standard
```

### OpenCL Not Detected

**Check if OpenCL is installed:**
```bash
# On Ubuntu/Debian
apt-get install opencl-headers ocl-icd-opencl-dev

# Check available OpenCL platforms
clinfo
```

### C++ Extension Import Error

**Error:** `ModuleNotFoundError: No module named 'tensor_ops_cpp'`

**Solution:** Make sure the extension is built and the directory is in your Python path:
```python
import sys
sys.path.insert(0, 'path/to/cpp_ext')
import tensor_ops_cpp
```

## Performance Comparison

Typical speedups compared to pure Python (NumPy):

- **RMS Norm:**
  - C++ (AVX2+OpenMP): 1.5-2x faster
  - OpenCL (GPU): 5-10x faster
  - CUDA (GPU): 10-20x faster

- **Activation Functions (SiLU, GELU):**
  - C++ (AVX2+OpenMP): 1-2x faster
  - OpenCL (GPU): 5-15x faster
  - CUDA (GPU): 10-30x faster

Note: Actual performance depends on:
- Tensor size (GPUs excel with larger tensors)
- Hardware capabilities
- Memory bandwidth
- CPU vs GPU data transfer overhead

## Best Practices

1. **Use automatic backend selection** for most cases - it will choose the best available option
2. **Build all available backends** to maximize performance across different hardware
3. **Profile your specific workload** - performance varies by tensor size and operation
4. **Consider data transfer overhead** - for small tensors, CPU backends may be faster than GPU due to transfer costs
5. **Keep backends updated** - rebuild after system updates or driver changes

## API Reference

All backends implement the same interface:

### Functions

- `rms_norm(x, weight, eps=1e-6)` - RMS normalization
- `silu(x)` - SiLU activation function
- `gelu(x)` - GELU activation function

### Info Functions

- `get_backend()` - Get active backend name (in tensor_ops)
- `get_backend_info()` - Get detailed backend information (in tensor_ops)
- `get_cuda_info()` - Get CUDA device info (in tensor_ops_cuda)
- `get_opencl_info()` - Get OpenCL device info (in tensor_ops_opencl)
- `get_optimization_info()` - Get C++ optimization info (in tensor_ops_cpp)

## Contributing

When adding new operations:

1. Implement in Python (tensor_ops.py) as baseline
2. Add C++ implementation (tensor_ops_cpp.cpp)
3. Add CUDA kernel (tensor_ops_cuda.cu)
4. Add OpenCL kernel (tensor_ops_opencl.cpp)
5. Update backend detection in tensor_ops.py
6. Add tests in test_cpp_ext.py
7. Update benchmarks in detect_and_benchmark.py
