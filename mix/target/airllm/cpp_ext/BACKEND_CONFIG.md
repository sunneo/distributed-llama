# Backend Configuration Guide

## Overview

The tensor operations module supports configurable backend selection through a JSON configuration file. This allows you to specify which backend to use instead of relying solely on automatic detection.

## Configuration File

The backend configuration is stored in `backend.json` located in the `mix/target/airllm/cpp_ext/` directory.

### Default Configuration

```json
{
  "preferred_backend": "auto",
  "backend_priority": [
    "cuda",
    "opencl",
    "cpp",
    "python"
  ],
  "force_backend": null
}
```

### Configuration Options

- **`preferred_backend`**: Set to `"auto"` for automatic detection, or specify a specific backend: `"cuda"`, `"opencl"`, `"cpp"`, or `"python"`
- **`backend_priority`**: Order of backends to try when `preferred_backend` is `"auto"` (only used in auto mode)
- **`force_backend`**: Set to a specific backend name to force its use (overrides availability checks). Set to `null` for normal behavior.

## Using setup.py to Configure Backend

### View Current Configuration

```bash
cd mix/target/airllm/cpp_ext
python setup.py configure_backend --show
```

Output:
```
Current Backend Configuration:
  Preferred backend: auto
  Force backend: None
  Priority order: cuda, opencl, cpp, python

Configuration file: /path/to/backend.json
```

### Set Preferred Backend

```bash
# Set preferred backend to C++ (CPU optimization)
python setup.py configure_backend --backend=cpp

# Set to CUDA (NVIDIA GPU)
python setup.py configure_backend --backend=cuda

# Set to OpenCL (cross-platform GPU)
python setup.py configure_backend --backend=opencl

# Set to Python (pure NumPy)
python setup.py configure_backend --backend=python

# Set back to auto-detection
python setup.py configure_backend --backend=auto
```

### Force a Specific Backend

Use `--force` to fail if the specified backend is not available (instead of falling back):

```bash
# Force CUDA - will fail if CUDA backend is not built
python setup.py configure_backend --backend=cuda --force
```

### Change Backend Priority

```bash
# Prioritize CPU backend over GPU backends
python setup.py configure_backend --priority="cpp,cuda,opencl,python"

# Prioritize OpenCL over CUDA
python setup.py configure_backend --priority="opencl,cuda,cpp,python"
```

### Combined Configuration

```bash
# Set to auto with custom priority
python setup.py configure_backend --backend=auto --priority="cpp,cuda,opencl,python"
```

## Manual Configuration

You can also manually edit the `backend.json` file:

```json
{
  "preferred_backend": "cpp",
  "backend_priority": [
    "cpp",
    "cuda",
    "opencl",
    "python"
  ],
  "force_backend": null
}
```

After editing, the changes take effect on the next import of `tensor_ops`.

## Usage Examples

### Example 1: Force CPU Backend

```bash
# Configure to use C++ backend only
cd mix/target/airllm/cpp_ext
python setup.py configure_backend --backend=cpp
```

Then in Python:
```python
from airllm import tensor_ops

# Will use C++ backend (even if CUDA is available)
print(tensor_ops.get_backend())  # Output: 'cpp'
```

### Example 2: Prefer GPU with Fallback

```bash
# Set CUDA as preferred, but allow fallback if not available
cd mix/target/airllm/cpp_ext
python setup.py configure_backend --backend=cuda
```

```python
from airllm import tensor_ops

# Will use CUDA if available, otherwise falls back to next available
print(tensor_ops.get_backend())
```

### Example 3: Custom Priority Order

```bash
# Prefer CPU over GPU backends
cd mix/target/airllm/cpp_ext
python setup.py configure_backend --backend=auto --priority="cpp,opencl,cuda,python"
```

```python
from airllm import tensor_ops

# Will try backends in order: cpp → opencl → cuda → python
print(tensor_ops.get_backend())
```

### Example 4: Force Backend (Strict Mode)

```bash
# Force CUDA - will fail if not available
cd mix/target/airllm/cpp_ext
python setup.py configure_backend --backend=cuda --force
```

```python
from airllm import tensor_ops

# Will use CUDA or fail during import if not available
print(tensor_ops.get_backend())
```

## Checking Current Configuration

In Python, you can check the current configuration:

```python
from airllm import tensor_ops

info = tensor_ops.get_backend_info()
print("Active backend:", info['backend'])
print("Configuration:", info['config'])
print("Available backends:", info['available_backends'])
```

Output:
```
Active backend: cpp
Configuration: {
  'preferred_backend': 'cpp',
  'backend_priority': ['cuda', 'opencl', 'cpp', 'python'],
  'force_backend': None
}
Available backends: ['cpp', 'python']
```

## Use Cases

### Development vs Production

**Development** (want immediate errors):
```bash
python setup.py configure_backend --backend=cuda --force
```

**Production** (want fallback):
```bash
python setup.py configure_backend --backend=auto
```

### Benchmarking

Test different backends:
```bash
# Test C++ backend
python setup.py configure_backend --backend=cpp
python benchmark.py

# Test CUDA backend
python setup.py configure_backend --backend=cuda
python benchmark.py

# Compare results
```

### Server Deployment

Pin to specific backend for consistency:
```bash
# Production server with NVIDIA GPU
python setup.py configure_backend --backend=cuda --force

# Production server with AMD GPU
python setup.py configure_backend --backend=opencl --force

# CPU-only server
python setup.py configure_backend --backend=cpp
```

## Troubleshooting

### Configuration Not Taking Effect

If changes don't take effect:
1. Restart your Python interpreter
2. Clear any cached imports
3. Check the configuration file was saved correctly

### Backend Not Available

If you configure a backend that's not built:
```bash
python setup.py configure_backend --backend=cuda
```

But you see:
```python
print(tensor_ops.get_backend())  # Output: 'cpp' (fallback)
```

This means CUDA backend is not built. Build it first:
```bash
python setup.py build_cuda
```

Then the configuration will take effect.

### Force Mode Failures

If using `--force` and getting errors:
```
ImportError: No module named 'tensor_ops_cuda'
```

This means the forced backend is not available. Either:
1. Build the backend: `python setup.py build_cuda`
2. Remove force mode: `python setup.py configure_backend --backend=auto`

## Best Practices

1. **Use `auto` in most cases** - automatic detection works well
2. **Use `force` for critical deployments** - ensures expected backend is used
3. **Customize priority for your hardware** - if you have specific GPU preferences
4. **Document your configuration** - include backend.json in version control for consistency
5. **Test after configuration changes** - verify the correct backend is active

## Integration with CI/CD

In CI/CD pipelines:

```bash
# In deployment script
cd mix/target/airllm/cpp_ext

# Configure backend based on environment
if [ "$GPU_TYPE" = "nvidia" ]; then
    python setup.py configure_backend --backend=cuda --force
elif [ "$GPU_TYPE" = "amd" ]; then
    python setup.py configure_backend --backend=opencl --force
else
    python setup.py configure_backend --backend=cpp
fi
```

## See Also

- `BACKEND_GUIDE.md` - Detailed backend usage guide
- `backend_examples.py` - Working code examples
- `README.md` - Build instructions for each backend
