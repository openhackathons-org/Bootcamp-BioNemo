#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_nim_container.sh molmim [port] [gpu_id]
  scripts/run_nim_container.sh boltz2 [port] [gpu_id]

Environment:
  OPENHACKATHON_CONTAINER_RUNTIME  auto, apptainer, singularity, or docker. Default: auto.

The auto mode prefers Apptainer/Singularity when available, and otherwise uses
Docker. Set OPENHACKATHON_CONTAINER_RUNTIME=docker for GB300/ARM workstation
testing.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# MolMIM has no aarch64 NIM image. When the ARM-local path is selected, route
# MolMIM to the pure-PyTorch NIM-like runner regardless of container runtime.
if [ "${1:-}" = "molmim" ] && [ "${OPENHACKATHON_MOLMIM_ARM:-0}" = "1" ]; then
    exec "$repo_root/scripts/run_molmim_arm.sh" "$@"
fi

runtime="${OPENHACKATHON_CONTAINER_RUNTIME:-auto}"

case "$runtime" in
    auto)
        if command -v apptainer >/dev/null 2>&1; then
            exec "$repo_root/scripts/run_nim_apptainer.sh" "$@"
        elif command -v singularity >/dev/null 2>&1; then
            APPTAINER_BIN=singularity exec "$repo_root/scripts/run_nim_apptainer.sh" "$@"
        elif command -v docker >/dev/null 2>&1; then
            exec "$repo_root/scripts/run_nim_docker.sh" "$@"
        else
            echo "Error: no supported container runtime found. Install Apptainer/Singularity or Docker." >&2
            exit 1
        fi
        ;;
    apptainer)
        APPTAINER_BIN="${APPTAINER_BIN:-apptainer}" exec "$repo_root/scripts/run_nim_apptainer.sh" "$@"
        ;;
    singularity)
        APPTAINER_BIN="${APPTAINER_BIN:-singularity}" exec "$repo_root/scripts/run_nim_apptainer.sh" "$@"
        ;;
    docker)
        exec "$repo_root/scripts/run_nim_docker.sh" "$@"
        ;;
    *)
        echo "Error: OPENHACKATHON_CONTAINER_RUNTIME must be auto, apptainer, singularity, or docker." >&2
        exit 1
        ;;
esac
