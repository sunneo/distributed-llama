#!/usr/bin/env bash
set -euo pipefail

# Package the Python worker + AirLLM helpers into a portable tarball.
# Env vars:
#   BUNDLE_PATH - output path (default: install/dist-worker-bundle.tar.gz)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUNDLE_PATH="${BUNDLE_PATH:-${SCRIPT_DIR}/dist-worker-bundle.tar.gz}"

for path in "${REPO_ROOT}/mix/target/distributed-llama.python" "${REPO_ROOT}/mix/target/airllm" "${REPO_ROOT}/mix/target/profile_worker.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "[package_worker] Missing required path: ${path}"
    exit 1
  fi
done

echo "[package_worker] Creating bundle at ${BUNDLE_PATH}"
tar -czf "${BUNDLE_PATH}" -C "${REPO_ROOT}" \
  mix/target/distributed-llama.python \
  mix/target/airllm \
  mix/target/profile_worker.py

echo "[package_worker] Bundle ready -> ${BUNDLE_PATH}"
