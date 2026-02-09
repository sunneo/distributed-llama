#!/usr/bin/env bash

validate_node() {
  local value="$1"
  if [[ ! "${value}" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+(:[0-9]+)?$ ]]; then
    echo "[common] Invalid node entry: ${value}"
    exit 1
  fi
}

validate_host() {
  local value="$1"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[common] Invalid host entry: ${value}"
    exit 1
  fi
}

validate_port() {
  local value="$1"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    echo "[common] Invalid port: ${value}"
    exit 1
  fi
}

validate_flag() {
  local name="$1"
  local value="$2"
  local lower="${value,,}"
  if [[ ! "${value}" =~ ^[-A-Za-z0-9/._@+]+$ || "${lower}" =~ proxy ]]; then
    echo "[common] ${name} contains unsupported characters: ${value}"
    exit 1
  fi
}

validate_path() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._/+\-~]+$ ]]; then
    echo "[common] ${name} contains invalid characters"
    exit 1
  fi
}

validate_cmd() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._+/-]+$ ]]; then
    echo "[common] ${name} contains invalid characters"
    exit 1
  fi
}
