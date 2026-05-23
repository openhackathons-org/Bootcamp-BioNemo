#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_nim_apptainer.sh molmim [port] [gpu_id]
  scripts/run_nim_apptainer.sh boltz2 [port] [gpu_id]

Environment:
  NGC_API_KEY              Required for model download and private NGC pulls.
  LOCAL_NIM_CACHE          Host cache directory. Default: ~/.cache/nim
  LOCAL_NIM_WORKSPACE      Host workspace directory. Default: LOCAL_NIM_CACHE/workspace
  SIF_DIR                  Directory for pulled .sif images. Default: ./.sif
  MOLMIM_IMAGE             Default: nvcr.io/nim/nvidia/molmim:1.0.0
  BOLTZ2_IMAGE             Default: nvcr.io/nim/mit/boltz2:1.7.0
  APPTAINER_BIN            Optional explicit runtime binary.
  APPTAINER_GPU_MODE       GPU setup mode: auto, nvccli, or nv. Default: auto.
  APPTAINER_PULL_ARGS      Optional extra args for apptainer/singularity pull.
  APPTAINER_RUN_ARGS       Optional extra args for apptainer/singularity run.

Examples:
  export NGC_API_KEY=<your-ngc-key>
  scripts/run_nim_apptainer.sh molmim 8001 0
  scripts/run_nim_apptainer.sh boltz2 8000 1
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

if [ -z "${NGC_API_KEY:-}" ]; then
    echo "Error: NGC_API_KEY is required." >&2
    exit 1
fi

if [ -n "${APPTAINER_BIN:-}" ]; then
    runtime="$APPTAINER_BIN"
elif command -v apptainer >/dev/null 2>&1; then
    runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    runtime="singularity"
else
    echo "Error: neither apptainer nor singularity was found in PATH." >&2
    exit 1
fi

service="$1"
port="${2:-}"
gpu_id="${3:-0}"

case "$service" in
    molmim)
        image="${MOLMIM_IMAGE:-nvcr.io/nim/nvidia/molmim:1.0.0}"
        port="${port:-8001}"
        sif_name="molmim_${image##*:}.sif"
        ;;
    boltz2)
        image="${BOLTZ2_IMAGE:-nvcr.io/nim/mit/boltz2:1.7.0}"
        port="${port:-8000}"
        sif_name="boltz2_${image##*:}.sif"
        ;;
    *)
        echo "Error: service must be 'molmim' or 'boltz2'." >&2
        usage >&2
        exit 1
        ;;
esac

default_home="${HOME:-}"
account_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)"
if [ -n "$account_home" ] && { [ -z "$default_home" ] || [ ! -w "$default_home" ]; }; then
    default_home="$account_home"
fi

sif_dir="${SIF_DIR:-$PWD/.sif}"
cache_dir="${LOCAL_NIM_CACHE:-$default_home/.cache/nim}"
workspace_dir="${LOCAL_NIM_WORKSPACE:-$cache_dir/workspace}"
mkdir -p "$sif_dir" "$cache_dir" "$workspace_dir"

sif_path="$sif_dir/$sif_name"

if [ ! -f "$sif_path" ]; then
    echo "Pulling $image into $sif_path"
    export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-\$oauthtoken}"
    export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:-$NGC_API_KEY}"
    # shellcheck disable=SC2086
    "$runtime" pull ${APPTAINER_PULL_ARGS:-} "$sif_path" "docker://$image"
fi

echo "Starting $service NIM on port $port with GPU $gpu_id"
echo "Cache: $cache_dir"
echo "Workspace: $workspace_dir"
echo "Image: $sif_path"

no_mount_args=()
if [ ! -e /etc/localtime ] && "$runtime" run --help 2>/dev/null | grep -q -- "--no-mount"; then
    no_mount_args=(--no-mount /etc/localtime)
fi

export CUDA_VISIBLE_DEVICES="$gpu_id"
export NVIDIA_VISIBLE_DEVICES="$gpu_id"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="$gpu_id"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="$gpu_id"

nvccli_available() {
    "$runtime" run --help 2>/dev/null | grep -q -- "--nvccli" && command -v nvidia-container-cli >/dev/null 2>&1
}

nvccli_smoke_test() {
    local err_file
    err_file="$(mktemp "${TMPDIR:-/tmp}/openhackathon-nvccli.XXXXXX")"
    if "$runtime" exec --nv --nvccli --contain --writable-tmpfs --cleanenv "${no_mount_args[@]}" "$sif_path" /bin/true >/dev/null 2>"$err_file"; then
        rm -f "$err_file"
        return 0
    fi

    echo "nvccli probe failed; falling back to standard --nv:" >&2
    sed -n '1,4p' "$err_file" >&2
    rm -f "$err_file"
    return 1
}

gpu_args=(--nv)
gpu_mode="${APPTAINER_GPU_MODE:-auto}"
case "$gpu_mode" in
    auto)
        if nvccli_available && nvccli_smoke_test; then
            gpu_args=(--nv --nvccli --contain --writable-tmpfs)
            echo "GPU mode: nvccli isolated device set ($gpu_id)"
        else
            echo "GPU mode: standard --nv; CUDA_VISIBLE_DEVICES=$gpu_id limits CUDA-visible devices"
        fi
        ;;
    nvccli)
        if ! nvccli_available; then
            echo "Error: APPTAINER_GPU_MODE=nvccli was requested, but $runtime --nvccli or nvidia-container-cli is unavailable." >&2
            exit 1
        fi
        if ! nvccli_smoke_test; then
            echo "Error: APPTAINER_GPU_MODE=nvccli was requested, but this Apptainer install rejected nvidia-container-cli." >&2
            echo "Ask the cluster admins for non-setuid Apptainer/user namespaces, or run with APPTAINER_GPU_MODE=nv and one NIM per allocation." >&2
            exit 1
        fi
        gpu_args=(--nv --nvccli --contain --writable-tmpfs)
        echo "GPU mode: nvccli isolated device set ($gpu_id)"
        ;;
    nv)
        echo "GPU mode: standard --nv; CUDA_VISIBLE_DEVICES=$gpu_id limits CUDA-visible devices"
        ;;
    *)
        echo "Error: APPTAINER_GPU_MODE must be auto, nvccli, or nv." >&2
        exit 1
        ;;
esac

# Apptainer/Singularity share the host network namespace by default. Setting
# NIM_HTTP_API_PORT changes the port that the service binds inside that shared
# namespace, so Docker-style -p port mapping is not needed for the common HPC case.
# shellcheck disable=SC2086
exec "$runtime" run \
    "${gpu_args[@]}" \
    --cleanenv \
    "${no_mount_args[@]}" \
    --env "NGC_API_KEY=$NGC_API_KEY" \
    --env "NGC_CLI_API_KEY=${NGC_CLI_API_KEY:-$NGC_API_KEY}" \
    --env "NVIDIA_VISIBLE_DEVICES=$gpu_id" \
    --env "CUDA_VISIBLE_DEVICES=$gpu_id" \
    --env "NIM_CACHE_PATH=/opt/nim/.cache" \
    --env "NIM_HTTP_API_PORT=$port" \
    --bind "$cache_dir:/opt/nim/.cache" \
    --bind "$workspace_dir:/opt/nim/workspace" \
    ${APPTAINER_RUN_ARGS:-} \
    "$sif_path"
