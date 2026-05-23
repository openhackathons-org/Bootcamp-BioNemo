#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_nim_docker.sh molmim [port] [gpu_id]
  scripts/run_nim_docker.sh boltz2 [port] [gpu_id]

Environment:
  NGC_API_KEY              Required for model download and private NGC pulls.
  LOCAL_NIM_CACHE          Host cache directory. Default: ~/.cache/nim
  LOCAL_NIM_WORKSPACE      Host workspace directory. Default: LOCAL_NIM_CACHE/workspace
  MOLMIM_IMAGE             Default: nvcr.io/nim/nvidia/molmim:1.0.0
  BOLTZ2_IMAGE             Default: nvcr.io/nim/mit/boltz2:1.7.0, or
                          nvcr.io/nim/mit/boltz2:1.4.0 on ARM hosts with
                          pre-590 NVIDIA drivers.
  DOCKER_BIN               Docker command. Default: docker. May be "sudo docker".
  DOCKER_PLATFORM          Docker platform. Default: linux/arm64 on aarch64, otherwise unset.
  DOCKER_GPU_ARGS          GPU args. Default: --gpus device=<gpu_id>.
  DOCKER_SHM_SIZE          Shared memory size. Default: 16G.
  DOCKER_RUN_ARGS          Optional extra docker run args.
  OPENHACKATHON_DOCKER_GPU_MODE
                          Set to all to expose all GPUs. Default: device.
  OPENHACKATHON_DOCKER_USE_NVIDIA_RUNTIME
                          Add --runtime=nvidia when the Docker host has a
                          runtime named nvidia. Default: 0.
  OPENHACKATHON_SKIP_GPU_RECOVERY_CHECK
                          Set to 1 to skip the nvidia-smi GPU recovery-state
                          preflight. Default: 0.
  OPENHACKATHON_DOCKER_LOGIN  Set to 0 to skip docker login. Default: 1.
  OPENHACKATHON_SKIP_PLATFORM_CHECK
                          Set to 1 to skip Docker image platform checks.
  OPENHACKATHON_RELAX_CACHE_PERMISSIONS
                          Chmod cache/workspace dirs to 0777 for container
                          users that do not match the host UID. Default: 1.

Examples:
  export NGC_API_KEY=<your-ngc-key>
  scripts/run_nim_docker.sh molmim 8001 0
  scripts/run_nim_docker.sh boltz2 8000 0
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

if [ -n "${DOCKER_BIN:-}" ]; then
    # shellcheck disable=SC2206
    docker_cmd=($DOCKER_BIN)
else
    docker_cmd=(docker)
fi

if ! command -v "${docker_cmd[0]}" >/dev/null 2>&1; then
    echo "Error: Docker command not found: ${docker_cmd[*]}" >&2
    exit 1
fi

if ! "${docker_cmd[@]}" ps >/dev/null 2>&1; then
    cat >&2 <<EOF
Error: Docker is installed, but the current user cannot access the Docker daemon.

Options:
  - Add this user to the docker group and start a new login session.
  - Set DOCKER_BIN='sudo docker' when sudo can run non-interactively.
  - Use Apptainer/Singularity on clusters that do not provide Docker.
EOF
    exit 1
fi

host_arch="$(uname -m)"

nvidia_driver_major() {
    local driver_version
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
    [ -n "$driver_version" ] || return 1
    printf '%s\n' "${driver_version%%.*}"
}

default_boltz2_image() {
    local driver_major
    case "$host_arch" in
        aarch64|arm64)
            driver_major="$(nvidia_driver_major || true)"
            if [ -n "$driver_major" ] && [ "$driver_major" -lt 590 ]; then
                echo "Info: ARM host has NVIDIA driver major $driver_major; defaulting Boltz-2 to 1.4.0 for the 580/CUDA 13 support matrix. Set BOLTZ2_IMAGE to override." >&2
                printf '%s\n' "nvcr.io/nim/mit/boltz2:1.4.0"
                return 0
            fi
            ;;
    esac

    printf '%s\n' "nvcr.io/nim/mit/boltz2:1.7.0"
}

service="$1"
port="${2:-}"
gpu_id="${3:-0}"

case "$service" in
    molmim)
        image="${MOLMIM_IMAGE:-nvcr.io/nim/nvidia/molmim:1.0.0}"
        port="${port:-8001}"
        ;;
    boltz2)
        image="${BOLTZ2_IMAGE:-$(default_boltz2_image)}"
        port="${port:-8000}"
        ;;
    *)
        echo "Error: service must be 'molmim' or 'boltz2'." >&2
        usage >&2
        exit 1
        ;;
esac

platform="${DOCKER_PLATFORM:-}"
if [ -z "$platform" ]; then
    case "$host_arch" in
        aarch64|arm64)
            platform="linux/arm64"
            ;;
        x86_64|amd64)
            platform=""
            ;;
        *)
            echo "Warning: unknown host architecture '$host_arch'; Docker will choose the default image platform." >&2
            ;;
    esac
fi

default_home="${HOME:-}"
account_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)"
if [ -n "$account_home" ] && { [ -z "$default_home" ] || [ ! -w "$default_home" ]; }; then
    default_home="$account_home"
fi

cache_dir="${LOCAL_NIM_CACHE:-$default_home/.cache/nim}"
workspace_dir="${LOCAL_NIM_WORKSPACE:-$cache_dir/workspace}"
mkdir -p "$cache_dir" "$workspace_dir"
if [ "${OPENHACKATHON_RELAX_CACHE_PERMISSIONS:-1}" = "1" ]; then
    chmod 0777 "$cache_dir" "$workspace_dir"
fi

if [ "${OPENHACKATHON_DOCKER_LOGIN:-1}" = "1" ]; then
    printf '%s' "$NGC_API_KEY" | "${docker_cmd[@]}" login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
fi

check_image_platform() {
    local image_ref="$1"
    local expected_platform="$2"
    local inspect_output
    local expected_os
    local expected_arch

    [ -n "$expected_platform" ] || return 0
    [ "${OPENHACKATHON_SKIP_PLATFORM_CHECK:-0}" != "1" ] || return 0

    if "${docker_cmd[@]}" buildx imagetools inspect "$image_ref" >/dev/null 2>&1; then
        inspect_output="$("${docker_cmd[@]}" buildx imagetools inspect "$image_ref")"
        if grep -q '^  Platform:' <<< "$inspect_output"; then
            if grep -q "Platform:  $expected_platform$" <<< "$inspect_output"; then
                return 0
            fi
            echo "Error: $image_ref does not advertise platform $expected_platform." >&2
            printf '%s\n' "$inspect_output" | sed -n '/^Manifests:/,$p' >&2
            exit 1
        fi

        cat >&2 <<EOF
Error: $image_ref does not advertise a multi-architecture manifest.
This usually means Docker will pull the image's native/default architecture,
which is not safe to run on $expected_platform hosts with GPU containers.

Set MOLMIM_IMAGE or BOLTZ2_IMAGE to an image tag that supports $expected_platform,
or set OPENHACKATHON_SKIP_PLATFORM_CHECK=1 if you are intentionally testing an
emulated/non-native path.
EOF
        exit 1
    fi

    if "${docker_cmd[@]}" manifest inspect "$image_ref" >/dev/null 2>&1; then
        inspect_output="$("${docker_cmd[@]}" manifest inspect "$image_ref")"
        expected_os="${expected_platform%%/*}"
        expected_arch="${expected_platform##*/}"
        if grep -Eq "\"os\"[[:space:]]*:[[:space:]]*\"$expected_os\"" <<< "$inspect_output" &&
            grep -Eq "\"architecture\"[[:space:]]*:[[:space:]]*\"$expected_arch\"" <<< "$inspect_output"; then
            return 0
        fi

        echo "Error: $image_ref does not appear to advertise platform $expected_platform." >&2
        printf '%s\n' "$inspect_output" >&2
        exit 1
    fi

    echo "Warning: could not inspect image platform for $image_ref; continuing." >&2
}

check_gpu_recovery_state() {
    local recovery_action

    [ "${OPENHACKATHON_SKIP_GPU_RECOVERY_CHECK:-0}" != "1" ] || return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0

    case "$gpu_id" in
        ''|*[!0-9]*)
            return 0
            ;;
    esac

    recovery_action="$(
        nvidia-smi -q -i "$gpu_id" 2>/dev/null |
            awk -F: '/GPU Recovery Action/ && !found {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; found=1}'
    )"

    if [ "$recovery_action" = "Reset" ]; then
        cat >&2 <<EOF
Error: GPU $gpu_id reports 'GPU Recovery Action: Reset'.
CUDA kernels are unlikely to launch until the GPU or node is reset.

Try:
  sudo systemctl stop nvidia-dcgm nvidia-persistenced
  sudo nvidia-smi --gpu-reset -i $gpu_id
  sudo systemctl start nvidia-persistenced nvidia-dcgm

If the reset is rejected, reboot the node through the BMC or scheduler.
Set OPENHACKATHON_SKIP_GPU_RECOVERY_CHECK=1 only if you intentionally want to
try the launch anyway.
EOF
        exit 1
    fi
}

pull_args=()
run_platform_args=()
if [ -n "$platform" ]; then
    pull_args+=(--platform "$platform")
    run_platform_args+=(--platform "$platform")
fi

check_image_platform "$image" "$platform"
check_gpu_recovery_state

echo "Pulling $image"
"${docker_cmd[@]}" pull "${pull_args[@]}" "$image"

container_name="openhackathon-${service}-${port}"
if "${docker_cmd[@]}" container inspect "$container_name" >/dev/null 2>&1; then
    echo "Removing existing stopped/running container named $container_name"
    "${docker_cmd[@]}" rm -f "$container_name" >/dev/null
fi

gpu_args=()
if [ -n "${DOCKER_GPU_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    gpu_args=($DOCKER_GPU_ARGS)
else
    runtime_args=()
    if [ "${OPENHACKATHON_DOCKER_USE_NVIDIA_RUNTIME:-0}" = "1" ]; then
        runtime_args=(--runtime=nvidia)
    fi

    case "${OPENHACKATHON_DOCKER_GPU_MODE:-device}" in
        all)
            gpu_args=("${runtime_args[@]}" --gpus all)
            ;;
        device)
            gpu_args=("${runtime_args[@]}" --gpus "device=$gpu_id")
            ;;
        *)
            echo "Error: OPENHACKATHON_DOCKER_GPU_MODE must be 'device' or 'all'." >&2
            exit 1
            ;;
    esac
fi

run_extra_args=()
if [ -n "${DOCKER_RUN_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    run_extra_args=($DOCKER_RUN_ARGS)
fi

echo "Starting $service NIM on port $port with GPU $gpu_id"
echo "Cache: $cache_dir"
echo "Workspace: $workspace_dir"
echo "Image: $image"
if [ -n "$platform" ]; then
    echo "Docker platform: $platform"
fi

export NGC_API_KEY
export NGC_CLI_API_KEY="${NGC_CLI_API_KEY:-$NGC_API_KEY}"

env_args=(
    -e NGC_API_KEY
    -e NGC_CLI_API_KEY
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-$gpu_id}"
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$gpu_id}"
    -e "NIM_CACHE_PATH=/opt/nim/.cache"
    -e "NIM_HTTP_API_PORT=$port"
)

for optional_env in \
    NIM_MODEL_PROFILE \
    NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND \
    NIM_BOLTZ_ENABLE_DIFFUSION_TF32 \
    NIM_LOG \
    NIM_LOG_LEVEL \
    TLLM_LOG_LEVEL \
    NIM_TELEMETRY_MODE \
    NIM_EXPOSE_CONFIDENCE_SCORES; do
    if [ -n "${!optional_env:-}" ]; then
        env_args+=(-e "$optional_env=${!optional_env}")
    fi
done

exec "${docker_cmd[@]}" run --rm \
    --name "$container_name" \
    "${run_platform_args[@]}" \
    "${gpu_args[@]}" \
    --shm-size="${DOCKER_SHM_SIZE:-16G}" \
    "${env_args[@]}" \
    -v "$cache_dir:/opt/nim/.cache" \
    -v "$workspace_dir:/opt/nim/workspace" \
    -p "127.0.0.1:${port}:${port}" \
    "${run_extra_args[@]}" \
    "$image"
