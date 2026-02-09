# install/

Scripts in this folder provision a minimal **Distributed-Llama + AirLLM** environment, distribute Python workers to other nodes, verify connectivity, and run a lightweight benchmark.

## Quick start

```bash
# 1) Build C++ root binary and install Python package with optimized C++ extensions
./install/setup_root.sh

# 2) Package Python worker + AirLLM helpers into a deployable tarball
./install/package_worker.sh

# 3) Deploy the bundle to remote nodes listed in a file (one SSH target per line)
./install/deploy_workers.sh install/nodes.example /opt/distributed-llama

# 4) From the root node, verify worker TCP ports (default 9999)
./install/test_connection.sh install/nodes.example 9999

# 5) Run a synthetic benchmark to confirm the Python worker stack is ready
./install/run_benchmark.sh
```

## What's New

The installation process now properly packages both `mix/target/airllm` and `mix/target/distributed-llama.python` as Python modules with the following improvements:

- **✨ Unified Package**: Both modules are installed as a single `distributed-llama` package
- **🚀 Optimized C++ Extensions**: Automatically builds and includes optimized tensor operations with AVX2/AVX-512/NEON support
- **📦 Proper Packaging**: Uses modern Python packaging (setup.py + pyproject.toml)
- **🔧 Easy Installation**: Simple pip install with automatic dependency management
- **⚡ Performance**: C++ extensions provide 5-15x speedup for tensor operations

## Installation Methods

### Method 1: Automated (Recommended)

Uses `setup_root.sh` to handle everything automatically:

```bash
# Standard installation
./install/setup_root.sh

# Development/editable installation (changes reflect immediately)
EDITABLE=1 ./install/setup_root.sh

# With virtualenv
VENV_PATH=~/dllama-venv ./install/setup_root.sh

# Skip Python installation (only build C++ binary)
SKIP_PYTHON=1 ./install/setup_root.sh
```

### Method 2: Manual Installation

For more control over the installation process:

```bash
# 1. Build C++ root binary
make dllama

# 2. Install Python package
pip install .                    # Regular installation
pip install -e .                 # Development mode
pip install .[dev]               # With development dependencies
pip install .[cpp]               # Ensure pybind11 is installed

# 3. Verify installation
dllama-worker --help            # Check if worker command is available
python -c "import airllm; import distributed_llama_python"  # Check imports
```

### Method 3: Build C++ Extensions Only

If you only need to rebuild the C++ extensions:

```bash
# Automatic detection and build
python setup.py build_ext --inplace

# Or build directly in cpp_ext directory
cd mix/target/airllm/cpp_ext
python setup.py test_capabilities  # Test what optimizations are available
python setup.py build_ext --inplace
```

## Environment Variables

- `PYTHON`: Python executable to use (default: `python3`)
- `VENV_PATH`: Path to create/use virtualenv
- `SKIP_PYTHON=1`: Skip Python package installation
- `SKIP_CPP_BUILD=1`: Skip building C++ extensions (use pure Python fallback)
- `EDITABLE=1`: Install in editable/development mode
- `DEBUG=1`: Build C++ extensions with debug symbols
- `SCP_OPTS`/`SSH_OPTS`: SSH options for deployment (e.g., `-P 2222`)
- `WORKER_PORT`: Override default worker port (default: 9999)

## Files

- `setup_root.sh`: Builds `dllama` and installs the unified Python package with optimized C++ extensions (supports `VENV_PATH`, `PYTHON`, `EDITABLE`, and `SKIP_PYTHON` variables).
- `package_worker.sh`: Creates `install/dist-worker-bundle.tar.gz` with the Python worker and AirLLM modules including built C++ extensions.
- `deploy_workers.sh`: Copies the bundle to each node, extracts it to a target path, and installs requirements.
- `test_connection.sh`: Quick TCP reachability check to worker ports from the root node.
- `run_benchmark.sh`: Executes `mix/target/profile_worker.py` as a sanity/throughput check without needing model files.
- `nodes.example`: Sample inventory file for SSH targets (one per line, e.g., `user@10.0.0.2`).

## Usage After Installation

Once installed, you can use the worker in multiple ways:

```bash
# Method 1: Using installed command
dllama-worker --host 0.0.0.0 --port 9999 --model /path/to/model.m

# Method 2: Using Python module
python -m distributed_llama_python.worker --host 0.0.0.0 --port 9999 --model /path/to/model.m

# Method 3: Direct import in Python code
from distributed_llama_python import Worker
worker = Worker('127.0.0.1', 9999, '/path/to/model.m')
worker.connect()
worker.load_weights()
worker.run()
```

## Package Structure

After installation, the following modules are available:

- `airllm`: AirLLM optimization module
  - Layer-wise inference engine
  - Memory-mapped weight loading
  - Tensor operations (Python + C++ optimized)
  - Activation compression
  
- `distributed_llama_python`: Distributed-Llama Python worker
  - Network client for root node communication
  - Worker implementation
  - Configuration management

## Troubleshooting

### C++ Extensions Not Building

If C++ extensions fail to build:
```bash
# Install build dependencies
pip install pybind11

# Check compiler availability
python -c "import subprocess; subprocess.run(['g++', '--version'])"

# Try manual build
cd mix/target/airllm/cpp_ext
python setup.py test_capabilities  # See what's available
python setup.py build_ext --inplace

# Use pure Python fallback
SKIP_CPP_BUILD=1 pip install .
```

### Import Errors

If imports fail:
```bash
# Verify installation
pip show distributed-llama

# Reinstall
pip uninstall distributed-llama
pip install -e .  # Development mode recommended during testing
```

> Tip: Set `SCP_OPTS`/`SSH_OPTS` (e.g., `-P 2222`) if your SSH port is non-default. Set `WORKER_PORT` to override the default 9999 port for connection checks.
