#!/usr/bin/env bash
set -euo pipefail

# Build the C++ root binary and install Python deps for the AirLLM-style worker.
# Env vars:
#   PYTHON   - python executable to use (default: python3)
#   VENV_PATH- optional virtualenv path to create/use
#   SKIP_PYTHON=1 to skip pip install

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
  echo "[setup_root] Installing Python requirements for workers"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/mix/target/distributed-llama.python/requirements.txt"
fi

echo "[setup_root] Done."
echo "  - Binary: ${REPO_ROOT}/dllama"
if [[ -n "${VENV_PATH:-}" ]]; then
  echo "  - Virtualenv: ${VENV_PATH} (activate to run Python tools)"
fi
