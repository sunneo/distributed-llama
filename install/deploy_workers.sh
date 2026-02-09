#!/usr/bin/env bash
set -euo pipefail

# Distribute the packaged Python worker bundle to remote nodes and install deps.
# Usage: deploy_workers.sh <nodes_file> <remote_dir>
#   nodes_file: one SSH target per line (e.g., user@10.0.0.2)
#   remote_dir: destination directory on the remote hosts
# Env vars:
#   BUNDLE_PATH - path to bundle (default: install/dist-worker-bundle.tar.gz)
#   PYTHON      - remote python executable (default: python3)
#   SCP_OPTS    - extra flags for scp (e.g., -P 2222)
#   SSH_OPTS    - extra flags for ssh (e.g., -p 2222)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <nodes_file> <remote_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_HELPERS="${SCRIPT_DIR}/common.sh"
if [[ ! -f "${COMMON_HELPERS}" ]]; then
  echo "[deploy_workers] Missing common helpers at ${COMMON_HELPERS}"
  exit 1
fi
# shellcheck source=install/common.sh
. "${COMMON_HELPERS}"

NODES_FILE="$1"
REMOTE_DIR="$2"
PYTHON_BIN="${PYTHON:-python3}"
BUNDLE_PATH="${BUNDLE_PATH:-${SCRIPT_DIR}/dist-worker-bundle.tar.gz}"
REMOTE_BUNDLE_NAME="$(basename "${BUNDLE_PATH}")"
REMOTE_BUNDLE_PATH="${REMOTE_DIR}/${REMOTE_BUNDLE_NAME}"

SCP_FLAGS=()
SSH_FLAGS=()
validate_flag() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[-A-Za-z0-9/._@+]+$ || "${value}" =~ ProxyCommand ]]; then
    echo "[deploy_workers] ${name} contains unsupported characters: ${value}"
    exit 1
  fi
}
validate_path() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._/+\-~]+$ ]]; then
    echo "[deploy_workers] ${name} contains invalid characters"
    exit 1
  fi
}
validate_cmd() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._+/-]+$ ]]; then
    echo "[deploy_workers] ${name} contains invalid characters"
    exit 1
  fi
}

if [[ -n "${SCP_OPTS:-}" ]]; then
  read -r -a SCP_FLAGS <<< "${SCP_OPTS}"
  for flag in "${SCP_FLAGS[@]}"; do
    validate_flag "SCP_OPTS" "${flag}"
  done
fi
if [[ -n "${SSH_OPTS:-}" ]]; then
  read -r -a SSH_FLAGS <<< "${SSH_OPTS}"
  for flag in "${SSH_FLAGS[@]}"; do
    validate_flag "SSH_OPTS" "${flag}"
  done
fi

if [[ ! -f "${NODES_FILE}" ]]; then
  echo "[deploy_workers] Nodes file not found: ${NODES_FILE}"
  exit 1
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "[deploy_workers] Bundle missing at ${BUNDLE_PATH}. Run package_worker.sh first."
  exit 1
fi

validate_path "REMOTE_DIR" "${REMOTE_DIR}"
validate_cmd "PYTHON_BIN" "${PYTHON_BIN}"
SAFE_REMOTE_DIR=$(printf "%q" "${REMOTE_DIR}")
SAFE_REMOTE_BUNDLE_PATH=$(printf "%q" "${REMOTE_BUNDLE_PATH}")
SAFE_PYTHON_BIN=$(printf "%q" "${PYTHON_BIN}")

while IFS= read -r NODE; do
  [[ -z "${NODE}" || "${NODE}" =~ ^# ]] && continue
  validate_node "${NODE}"

  echo "[deploy_workers] -> ${NODE}"
  ssh "${SSH_FLAGS[@]}" "${NODE}" "mkdir -p ${SAFE_REMOTE_DIR}" \
    || { echo "[deploy_workers] Failed to create remote dir on ${NODE}"; exit 1; }

  scp "${SCP_FLAGS[@]}" "${BUNDLE_PATH}" "${NODE}:${SAFE_REMOTE_BUNDLE_PATH}" \
    || { echo "[deploy_workers] Failed to copy bundle to ${NODE}"; exit 1; }

  ssh "${SSH_FLAGS[@]}" "${NODE}" "tar -xzf ${SAFE_REMOTE_BUNDLE_PATH} -C ${SAFE_REMOTE_DIR}" \
    || { echo "[deploy_workers] Failed to extract bundle on ${NODE}"; exit 1; }

  ssh "${SSH_FLAGS[@]}" "${NODE}" "cd ${SAFE_REMOTE_DIR}/mix/target/distributed-llama.python && ${SAFE_PYTHON_BIN} -m pip install -r requirements.txt" \
    || { echo "[deploy_workers] Failed to install Python requirements on ${NODE}"; exit 1; }
done < "${NODES_FILE}"

echo "[deploy_workers] Completed deployment to all nodes listed in ${NODES_FILE}"
