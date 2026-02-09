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
  scp ${SCP_OPTS:-} "${BUNDLE_PATH}" "${NODE}:${REMOTE_DIR}/worker-bundle.tgz"
  ssh ${SSH_OPTS:-} "${NODE}" "mkdir -p ${REMOTE_DIR} && tar -xzf ${REMOTE_DIR}/worker-bundle.tgz -C ${REMOTE_DIR}"
  ssh ${SSH_OPTS:-} "${NODE}" "cd ${REMOTE_DIR}/mix/target/distributed-llama.python && ${PYTHON_BIN} -m pip install -r requirements.txt"
done < "${NODES_FILE}"

echo "[deploy_workers] Completed deployment to all nodes listed in ${NODES_FILE}"
