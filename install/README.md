# install/

Scripts in this folder provision a minimal **Distributed-Llama + AirLLM** environment, distribute Python workers to other nodes, verify connectivity, and run a lightweight benchmark.

## Quick start

```bash
# 1) Build C++ root binary and install Python deps (optionally set VENV_PATH)
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

### Files
- `nodes.example`: sample inventory file for SSH targets (one per line, e.g., `user@10.0.0.2`).
- `setup_root.sh`: builds `dllama` and installs Python requirements (supports `VENV_PATH` and `PYTHON` overrides).
- `package_worker.sh`: creates `install/dist-worker-bundle.tar.gz` with the Python worker and AirLLM modules.
- `deploy_workers.sh`: copies the bundle to each node, extracts it to a target path, and installs requirements.
- `test_connection.sh`: quick TCP reachability check to worker ports from the root node.
- `run_benchmark.sh`: executes `mix/target/profile_worker.py` as a sanity/throughput check without needing model files.

> Tip: Set `SCP_OPTS`/`SSH_OPTS` (e.g., `-P 2222`) if your SSH port is non-default. Set `WORKER_PORT` to override the default 9999 port for connection checks.
