#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Build and run a NIM-like MolMIM service on aarch64 (Grace / GB200 / GB300).
#
# There is no arm64 MolMIM NIM and the BioNeMo Framework v1 (which has MolMIM)
# is amd64/pre-Blackwell, so this serves MolMIM via a pure-PyTorch
# reimplementation (molmim_arm/molmim_torch.py) that loads the official
# molmim_70m_24_3 checkpoint on a stock NGC PyTorch base. It exposes the MolMIM
# NIM REST surface (/hidden, /decode, /sampling, /generate, /v1/health/ready)
# plus OpenAPI docs at /docs, so the bootcamp client + CMA-ES run unmodified
# against MOLMIM_URL=http://localhost:<port>.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_context="$repo_root/molmim_arm"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_molmim_arm.sh molmim [port] [gpu_id]

Environment:
  NGC_API_KEY              Required (pull the PyTorch base image + download weights).
  BIONEMO_PYTORCH_BASE     PyTorch base image (arm64 + Blackwell).
                          Default: nvcr.io/nvidia/pytorch:25.01-py3
  MOLMIM_ARM_IMAGE         Image tag to build/run. Default: molmim-arm-nim:local
  MOLMIM_ARM_REBUILD       Set to 1 to force an image rebuild.
  MOLMIM_WEIGHTS_DIR       Host dir holding molmim_70m_24_3.nemo (mounted at /models).
                          Default: LOCAL_NIM_CACHE/molmim
  MOLMIM_NGC_MODEL         NGC model to fetch weights from. Default: nvidia/clara/molmim:1.3
  MOLMIM_SAMPLING_METHOD   greedy or topkp (decode for /sampling). Default: greedy
  LOCAL_NIM_CACHE          Base cache dir. Default: ~/.cache/nim
  DOCKER_BIN               Docker command. Default: docker.
  DOCKER_PLATFORM          Default: linux/arm64 on aarch64.
  OPENHACKATHON_DOCKER_GPU_MODE   device or all. Default: device.

Examples:
  export NGC_API_KEY=<your-ngc-key>
  scripts/run_molmim_arm.sh molmim 8001 0
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

service="$1"; port="${2:-8001}"; gpu_id="${3:-0}"
[ "$service" = "molmim" ] || { echo "Error: only the 'molmim' service is supported." >&2; exit 1; }
[ -n "${NGC_API_KEY:-}" ] || { echo "Error: NGC_API_KEY is required." >&2; exit 1; }

if [ -n "${DOCKER_BIN:-}" ]; then
    # shellcheck disable=SC2206
    docker_cmd=($DOCKER_BIN)
else
    docker_cmd=(docker)
fi
command -v "${docker_cmd[0]}" >/dev/null 2>&1 || { echo "Error: Docker not found." >&2; exit 1; }
"${docker_cmd[@]}" ps >/dev/null 2>&1 || { echo "Error: cannot access the Docker daemon." >&2; exit 1; }

host_arch="$(uname -m)"
platform="${DOCKER_PLATFORM:-}"
if [ -z "$platform" ]; then
    case "$host_arch" in aarch64|arm64) platform="linux/arm64" ;; *) platform="" ;; esac
fi

base_image="${BIONEMO_PYTORCH_BASE:-nvcr.io/nvidia/pytorch:25.01-py3}"
shim_image="${MOLMIM_ARM_IMAGE:-molmim-arm-nim:local}"
ngc_model="${MOLMIM_NGC_MODEL:-nvidia/clara/molmim:1.3}"

default_home="${HOME:-}"
account_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)"
if [ -n "$account_home" ] && { [ -z "$default_home" ] || [ ! -w "$default_home" ]; }; then
    default_home="$account_home"
fi
cache_root="${LOCAL_NIM_CACHE:-$default_home/.cache/nim}"
weights_dir="${MOLMIM_WEIGHTS_DIR:-$cache_root/molmim}"
mkdir -p "$weights_dir"

printf '%s' "$NGC_API_KEY" | "${docker_cmd[@]}" login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null

# --- ensure weights present -------------------------------------------------
nemo_file="$weights_dir/molmim_70m_24_3.nemo"
if [ ! -f "$nemo_file" ]; then
    echo "MolMIM weights not found; downloading $ngc_model via NGC CLI..."
    if ! command -v ngc >/dev/null 2>&1 && [ ! -x "$default_home/ngc-cli/ngc" ]; then
        ( cd "$default_home" && \
          wget -q -O ngccli_arm64.zip https://ngc.nvidia.com/downloads/ngccli_arm64.zip && \
          unzip -q -o ngccli_arm64.zip )
    fi
    export PATH="$default_home/ngc-cli:$PATH"
    NGC_CLI_API_KEY="$NGC_API_KEY" NGC_CLI_ORG=nvidia NGC_CLI_TEAM=clara \
        ngc registry model download-version "$ngc_model" --dest "$weights_dir" >/dev/null
    found="$(find "$weights_dir" -name 'molmim_70m_24_3.nemo' | head -1)"
    [ -n "$found" ] || { echo "Error: could not download MolMIM weights." >&2; exit 1; }
    [ "$found" = "$nemo_file" ] || ln -sf "$found" "$nemo_file"
fi

# --- build image ------------------------------------------------------------
build_args=()
[ -n "$platform" ] && build_args+=(--platform "$platform")
if [ "${MOLMIM_ARM_REBUILD:-0}" = "1" ] || ! "${docker_cmd[@]}" image inspect "$shim_image" >/dev/null 2>&1; then
    echo "Building $shim_image (base: $base_image)"
    "${docker_cmd[@]}" build "${build_args[@]}" --build-arg "BASE_IMAGE=$base_image" -t "$shim_image" "$build_context"
fi

# --- run --------------------------------------------------------------------
container_name="openhackathon-molmim-arm-${port}"
"${docker_cmd[@]}" rm -f "$container_name" >/dev/null 2>&1 || true

gpu_args=()
case "${OPENHACKATHON_DOCKER_GPU_MODE:-device}" in
    all) gpu_args=(--gpus all) ;;
    device) gpu_args=(--gpus "device=$gpu_id") ;;
    *) echo "Error: OPENHACKATHON_DOCKER_GPU_MODE must be 'device' or 'all'." >&2; exit 1 ;;
esac
run_platform_args=()
[ -n "$platform" ] && run_platform_args+=(--platform "$platform")

echo "Starting MolMIM ARM NIM on port $port (GPU $gpu_id); weights: $(dirname "$nemo_file")"
exec "${docker_cmd[@]}" run --rm \
    --name "$container_name" \
    "${run_platform_args[@]}" \
    "${gpu_args[@]}" \
    --ipc=host \
    -e "NIM_HTTP_API_PORT=$port" \
    -e "MOLMIM_CHECKPOINT=/models/molmim_70m_24_3.nemo" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$gpu_id}" \
    -e "MOLMIM_SAMPLING_METHOD=${MOLMIM_SAMPLING_METHOD:-greedy}" \
    -v "$(dirname "$nemo_file"):/models:ro" \
    -p "127.0.0.1:${port}:${port}" \
    "$shim_image"
