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

NODES_FILE="$1"
REMOTE_DIR="$2"
PYTHON_BIN="${PYTHON:-python3}"
BUNDLE_PATH="${BUNDLE_PATH:-${SCRIPT_DIR}/dist-worker-bundle.tar.gz}"
REMOTE_BUNDLE_NAME="$(basename "${BUNDLE_PATH}")"
REMOTE_BUNDLE_PATH="${REMOTE_DIR}/${REMOTE_BUNDLE_NAME}"

SCP_FLAGS=()
SSH_FLAGS=()
validate_opts() {
  local name="$1"
  local value="$2"
  if [[ "${value}" =~ [^[:alnum:][:space:]=/._:@+-] ]]; then
    echo "[deploy_workers] ${name} contains unsupported characters"
    exit 1
  fi
}

if [[ -n "${SCP_OPTS:-}" ]]; then
  validate_opts "SCP_OPTS" "${SCP_OPTS}"
  read -r -a SCP_FLAGS <<< "${SCP_OPTS}"
fi
if [[ -n "${SSH_OPTS:-}" ]]; then
  validate_opts "SSH_OPTS" "${SSH_OPTS}"
  read -r -a SSH_FLAGS <<< "${SSH_OPTS}"
fi

if [[ ! -f "${NODES_FILE}" ]]; then
  echo "[deploy_workers] Nodes file not found: ${NODES_FILE}"
  exit 1
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "[deploy_workers] Bundle missing at ${BUNDLE_PATH}. Run package_worker.sh first."
  exit 1
fi

while IFS= read -r NODE; do
  [[ -z "${NODE}" || "${NODE}" =~ ^# ]] && continue

  echo "[deploy_workers] -> ${NODE}"
  ssh "${SSH_FLAGS[@]}" "${NODE}" "mkdir -p '${REMOTE_DIR}'" \
    || { echo "[deploy_workers] Failed to create remote dir on ${NODE}"; exit 1; }

  scp "${SCP_FLAGS[@]}" "${BUNDLE_PATH}" "${NODE}:${REMOTE_BUNDLE_PATH}" \
    || { echo "[deploy_workers] Failed to copy bundle to ${NODE}"; exit 1; }

  ssh "${SSH_FLAGS[@]}" "${NODE}" "tar -xzf '${REMOTE_BUNDLE_PATH}' -C '${REMOTE_DIR}'" \
    || { echo "[deploy_workers] Failed to extract bundle on ${NODE}"; exit 1; }

  ssh "${SSH_FLAGS[@]}" "${NODE}" "cd '${REMOTE_DIR}/mix/target/distributed-llama.python' && '${PYTHON_BIN}' -m pip install -r requirements.txt" \
    || { echo "[deploy_workers] Failed to install Python requirements on ${NODE}"; exit 1; }
done < "${NODES_FILE}"

echo "[deploy_workers] Completed deployment to all nodes listed in ${NODES_FILE}"
