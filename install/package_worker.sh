#!/usr/bin/env bash
set -euo pipefail

# Package the Python worker + AirLLM helpers into a portable tarball.
# This creates a bundle with the optimized installed packages.
# Env vars:
#   BUNDLE_PATH - output path (default: install/dist-worker-bundle.tar.gz)
#   PYTHON - python executable to use (default: python3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUNDLE_PATH="${BUNDLE_PATH:-${SCRIPT_DIR}/dist-worker-bundle.tar.gz}"
PYTHON_BIN="${PYTHON:-python3}"

# Check if required paths exist
for path in "${REPO_ROOT}/mix/target/distributed-llama.python" "${REPO_ROOT}/mix/target/airllm" "${REPO_ROOT}/mix/target/profile_worker.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "[package_worker] Missing required path: ${path}"
    exit 1
  fi
done

# Check if C++ extensions are built
CPP_EXT_DIR="${REPO_ROOT}/mix/target/airllm/cpp_ext"
if [[ -d "${CPP_EXT_DIR}" ]]; then
  CPP_BUILT=$(find "${CPP_EXT_DIR}" -name "tensor_ops_cpp*.so" -o -name "tensor_ops_cpp*.pyd" -o -name "tensor_ops_cpp*.dll" | head -1)
  if [[ -n "${CPP_BUILT}" ]]; then
    echo "[package_worker] Found built C++ extensions: ${CPP_BUILT}"
  else
    echo "[package_worker] Warning: No built C++ extensions found in ${CPP_EXT_DIR}"
    echo "[package_worker] Run 'cd ${CPP_EXT_DIR} && python setup.py build_ext --inplace' to build them"
  fi
fi

echo "[package_worker] Creating bundle at ${BUNDLE_PATH}"
echo "[package_worker] This bundle includes:"
echo "  - mix/target/distributed-llama.python (worker module)"
echo "  - mix/target/airllm (optimization module with C++ extensions)"
echo "  - mix/target/profile_worker.py (benchmark tool)"

tar -czf "${BUNDLE_PATH}" -C "${REPO_ROOT}" \
  mix/target/distributed-llama.python \
  mix/target/airllm \
  mix/target/profile_worker.py

echo "[package_worker] Bundle ready -> ${BUNDLE_PATH}"
echo "[package_worker] To deploy this bundle to a worker node:"
echo "  1. Copy bundle: scp ${BUNDLE_PATH} user@worker:/tmp/"
echo "  2. Extract: ssh user@worker 'cd /opt && tar -xzf /tmp/dist-worker-bundle.tar.gz'"
echo "  3. Install deps: ssh user@worker 'cd /opt && pip install -r mix/target/distributed-llama.python/requirements.txt'"
echo "  4. Build C++ (optional): ssh user@worker 'cd /opt/mix/target/airllm/cpp_ext && python setup.py build_ext --inplace'"
echo "  5. Start worker: ssh user@worker 'cd /opt/mix/target/distributed-llama.python && python -m worker --host 0.0.0.0 --port 9999 --model /path/to/model.m'"
