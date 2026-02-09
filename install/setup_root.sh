#!/usr/bin/env bash
set -euo pipefail

# Build the C++ root binary and install Python deps for the AirLLM-style worker.
# Env vars:
#   PYTHON   - python executable to use (default: python3)
#   VENV_PATH- optional virtualenv path to create/use
#   SKIP_PYTHON=1 to skip pip install
#   EDITABLE=1 to install in editable/development mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python3}"

if [[ -n "${VENV_PATH:-}" ]]; then
  echo "[setup_root] Creating virtualenv at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  PYTHON_BIN="${VENV_PATH}/bin/python"
fi

echo "[setup_root] Building dllama (C++ root binary)"
make -C "${REPO_ROOT}" dllama

if [[ "${SKIP_PYTHON:-0}" != "1" ]]; then
  echo "[setup_root] Installing Python package with optimized C++ extensions"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  
  # Install pybind11 first for C++ extensions
  "${PYTHON_BIN}" -m pip install pybind11
  
  # Install the distributed-llama package
  # This will automatically build and include the optimized airllm C++ extensions
  if [[ "${EDITABLE:-0}" == "1" ]]; then
    echo "[setup_root] Installing in editable/development mode"
    "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
  else
    echo "[setup_root] Installing in regular mode"
    "${PYTHON_BIN}" -m pip install "${REPO_ROOT}"
  fi
fi

echo "[setup_root] Done."
echo "  - Binary: ${REPO_ROOT}/dllama"
echo "  - Python packages installed: distributed-llama (includes airllm + distributed_llama_python)"
echo "  - Worker command available: dllama-worker"
if [[ -n "${VENV_PATH:-}" ]]; then
  echo "  - Virtualenv: ${VENV_PATH} (activate to run Python tools)"
fi
