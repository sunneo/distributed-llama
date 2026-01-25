# Quick Start Guide - Optimized Tensor Operations

This guide helps you quickly get started with the optimized tensor operations library.

## 🚀 Quick Start (3 Steps)

### Step 1: Test Your System Capabilities

```bash
cd mix/target/airllm/cpp_ext
python setup.py test_capabilities
```

This will show what optimizations your system supports (AVX2, OpenMP, CUDA, etc.)

### Step 2: Build the Extension

```bash
python setup.py build_ext --inplace
```

The build automatically uses the best optimizations detected in Step 1.

### Step 3: Use It!

```python
import numpy as np
import tensor_ops_cpp

# Check what's enabled
print(tensor_ops_cpp.get_optimization_info())

# Use optimized operations
x = np.random.randn(128, 4096).astype(np.float32)
weight = np.random.randn(4096).astype(np.float32)

result = tensor_ops_cpp.rms_norm(x, weight)  # 5-15x faster!
result = tensor_ops_cpp.silu(x)              # 2-4x faster!
```

## 📊 Run the Detection & Benchmark Script

For automated testing and benchmarking:

```bash
python detect_and_benchmark.py
```

This will:
1. Detect all available capabilities
2. Build the extension automatically
3. Run functionality tests
4. Benchmark performance

## 🎯 See Usage Examples

```bash
python examples.py
```

Shows:
- Basic usage
- Querying capabilities
- Backend selection
- Performance comparison

## ⚡ Expected Performance

With AVX2 + FMA + OpenMP (typical modern x86 CPU):

| Operation | Speedup vs Pure Python |
|-----------|------------------------|
| RMS Norm  | **5-15x faster** |
| SiLU      | **2-4x faster** |
| GELU      | **2-4x faster** |
| Matmul    | **2-5x faster** |

## 🔧 Troubleshooting

**"Illegal instruction"** → CPU doesn't support compiled features. Rebuild on target machine.

**"No module named tensor_ops_cpp"** → Not built yet. Run `python setup.py build_ext --inplace`

**Low speedup** → Check `get_optimization_info()` to see what's enabled.

## 🎮 GPU Backends (Optional)

If you have a GPU:

```bash
# For NVIDIA GPUs
python setup.py build_cuda      # Requires CUDA toolkit

# For AMD/Intel GPUs
python setup.py build_opencl    # Requires OpenCL drivers
```

Then use:
```python
import tensor_ops_cuda  # or tensor_ops_opencl
result = tensor_ops_cuda.rms_norm(x, weight)
```

## 📚 More Information

- Full documentation: See [README.md](README.md)
- Architecture details: See source code comments in `tensor_ops_cpp.cpp`
- Build system: See `setup.py` for capability detection implementation
