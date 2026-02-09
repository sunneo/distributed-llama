#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_HELPERS="${SCRIPT_DIR}/common.sh"
if [[ ! -f "${COMMON_HELPERS}" ]]; then
  echo "[test_connection] Missing common helpers at ${COMMON_HELPERS}"
  exit 1
fi
# shellcheck source=install/common.sh
. "${COMMON_HELPERS}"

# Simple TCP reachability test from the root node to workers.
# Usage: test_connection.sh <nodes_file> [port]
#   nodes_file: matches deploy_workers.sh (SSH targets; format: [user@]host)
#   port: worker listening port (default from WORKER_PORT or 9999)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <nodes_file> [port]"
  exit 1
fi

NODES_FILE="$1"
PORT="${2:-${WORKER_PORT:-9999}}"

if ! command -v nc >/dev/null 2>&1; then
  echo "[test_connection] Error: netcat (nc) is not installed. Please install it to use this script."
  exit 1
fi

validate_port "${PORT}"

if [[ ! -f "${NODES_FILE}" ]]; then
  echo "[test_connection] Nodes file not found: ${NODES_FILE}"
  exit 1
fi

ok=0
fail=0

while IFS= read -r NODE; do
  [[ -z "${NODE}" || "${NODE}" =~ ^# ]] && continue
  validate_node "${NODE}"

  # Extract host part (strip user@ and optional :sshport)
  HOST_WITH_PORT="${NODE##*@}"
  HOST="${HOST_WITH_PORT%%:*}"
  validate_host "${HOST}"

  if nc -z -w3 "${HOST}" "${PORT}"; then
    echo "[test_connection] ${HOST}:${PORT} reachable"
    ok=$((ok + 1))
  else
    echo "[test_connection] ${HOST}:${PORT} FAILED"
    fail=$((fail + 1))
  fi
done < "${NODES_FILE}"

echo "[test_connection] Summary: ok=${ok}, fail=${fail}, port=${PORT}"

if [[ ${fail} -ne 0 ]]; then
  exit 1
fi
