#!/usr/bin/env bash
set -euo pipefail

# Run a lightweight Python-side benchmark to verify the AirLLM-style worker stack.
# Uses mix/target/profile_worker.py (no model required).
# Env vars:
#   PYTHON - python executable/venv (default: python3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

BENCH_SCRIPT="${REPO_ROOT}/mix/target/profile_worker.py"

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "[run_benchmark] Benchmark script not found: ${BENCH_SCRIPT}"
  exit 1
fi

echo "[run_benchmark] Running profile_worker.py with ${PYTHON_BIN}"
"${PYTHON_BIN}" "${BENCH_SCRIPT}"
