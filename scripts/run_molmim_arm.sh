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
#
# Works with Docker (workstation/GB300 ARM) and Apptainer/Singularity (HPC).
# The container runtime is selected from OPENHACKATHON_CONTAINER_RUNTIME (auto
# prefers Apptainer/Singularity when present, otherwise Docker). In every case
# the molmim_70m_24_3.nemo checkpoint is mounted at /models at runtime.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_context="$repo_root/molmim_arm"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_molmim_arm.sh molmim [port] [gpu_id]

Environment:
  NGC_API_KEY              Required (pull the PyTorch base image + download weights).
  OPENHACKATHON_CONTAINER_RUNTIME  auto, apptainer, singularity, or docker. Default: auto.
  BIONEMO_PYTORCH_BASE     PyTorch base image (arm64 + Blackwell).
                          Default: nvcr.io/nvidia/pytorch:25.01-py3
  MOLMIM_ARM_IMAGE         Docker image tag to build/run. Default: molmim-arm-nim:local
  MOLMIM_ARM_SIF           Apptainer .sif path. Default: SIF_DIR/molmim-arm-nim.sif
  MOLMIM_ARM_REBUILD       Set to 1 to force an image/sif rebuild.
  MOLMIM_WEIGHTS_DIR       Host dir holding molmim_70m_24_3.nemo (mounted at /models).
                          Default: LOCAL_NIM_CACHE/molmim
  MOLMIM_NGC_MODEL         NGC model to fetch weights from. Default: nvidia/clara/molmim:1.3
  MOLMIM_SAMPLING_METHOD   greedy or topkp (decode for /sampling). Default: greedy
  LOCAL_NIM_CACHE          Base cache dir. Default: ~/.cache/nim
  SIF_DIR                  Directory for built .sif images. Default: ./.sif
  DOCKER_BIN               Docker command. Default: docker.
  APPTAINER_BIN            Apptainer/Singularity binary. Default: auto-detected.
  DOCKER_PLATFORM          Default: linux/arm64 on aarch64 (Docker only).
  OPENHACKATHON_DOCKER_GPU_MODE   device or all (Docker only). Default: device.
  APPTAINER_GPU_MODE       auto, nvccli, or nv (Apptainer only). Default: auto.
  MOLMIM_ARM_MKSQUASHFS_PROCS  mksquashfs -processors for the Apptainer build
                          (caps CPUs to avoid mksquashfs segfaults on many-core
                          ARM). Default: 4.

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

# --- runtime selection ------------------------------------------------------
select_runtime() {
    local rt="${OPENHACKATHON_CONTAINER_RUNTIME:-auto}"
    case "$rt" in
        apptainer)   echo "apptainer ${APPTAINER_BIN:-apptainer}"; return ;;
        singularity) echo "apptainer ${APPTAINER_BIN:-singularity}"; return ;;
        docker)      echo "docker ${DOCKER_BIN:-docker}"; return ;;
        auto) : ;;
        *) echo "Error: OPENHACKATHON_CONTAINER_RUNTIME must be auto, apptainer, singularity, or docker." >&2; exit 1 ;;
    esac
    if [ -n "${APPTAINER_BIN:-}" ]; then echo "apptainer $APPTAINER_BIN"; return; fi
    if command -v apptainer >/dev/null 2>&1; then echo "apptainer apptainer"; return; fi
    if command -v singularity >/dev/null 2>&1; then echo "apptainer singularity"; return; fi
    if command -v "${DOCKER_BIN:-docker}" >/dev/null 2>&1; then echo "docker ${DOCKER_BIN:-docker}"; return; fi
    echo "Error: no supported container runtime found (Apptainer/Singularity or Docker)." >&2
    exit 1
}
read -r runtime_kind runtime_bin <<<"$(select_runtime)"

base_image="${BIONEMO_PYTORCH_BASE:-nvcr.io/nvidia/pytorch:25.01-py3}"
ngc_model="${MOLMIM_NGC_MODEL:-nvidia/clara/molmim:1.3}"

default_home="${HOME:-}"
account_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)"
if [ -n "$account_home" ] && { [ -z "$default_home" ] || [ ! -w "$default_home" ]; }; then
    default_home="$account_home"
fi
cache_root="${LOCAL_NIM_CACHE:-$default_home/.cache/nim}"
weights_dir="${MOLMIM_WEIGHTS_DIR:-$cache_root/molmim}"
mkdir -p "$weights_dir"

# --- ensure weights present (shared by both runtimes) -----------------------
ensure_weights() {
    local nemo_file="$weights_dir/molmim_70m_24_3.nemo"
    if [ -f "$nemo_file" ]; then return 0; fi
    echo "MolMIM weights not found; downloading $ngc_model via NGC CLI..."
    if ! command -v ngc >/dev/null 2>&1 && [ ! -x "$default_home/ngc-cli/ngc" ]; then
        ( cd "$default_home" && \
          wget -q -O ngccli_arm64.zip https://ngc.nvidia.com/downloads/ngccli_arm64.zip && \
          unzip -q -o ngccli_arm64.zip )
    fi
    export PATH="$default_home/ngc-cli:$PATH"
    NGC_CLI_API_KEY="$NGC_API_KEY" NGC_CLI_ORG=nvidia NGC_CLI_TEAM=clara \
        ngc registry model download-version "$ngc_model" --dest "$weights_dir" >/dev/null
    local found
    found="$(find "$weights_dir" -name 'molmim_70m_24_3.nemo' | head -1)"
    [ -n "$found" ] || { echo "Error: could not download MolMIM weights." >&2; exit 1; }
    [ "$found" = "$nemo_file" ] || ln -sf "$found" "$nemo_file"
}
ensure_weights

container_cuda_checkpoint="/models/molmim_70m_24_3.nemo"
sampling_method="${MOLMIM_SAMPLING_METHOD:-greedy}"

# ===========================================================================
# Docker path
# ===========================================================================
run_docker() {
    local docker_cmd=("$runtime_bin")
    command -v "${docker_cmd[0]}" >/dev/null 2>&1 || { echo "Error: Docker not found." >&2; exit 1; }
    "${docker_cmd[@]}" ps >/dev/null 2>&1 || { echo "Error: cannot access the Docker daemon." >&2; exit 1; }

    local host_arch platform
    host_arch="$(uname -m)"
    platform="${DOCKER_PLATFORM:-}"
    if [ -z "$platform" ]; then
        case "$host_arch" in aarch64|arm64) platform="linux/arm64" ;; *) platform="" ;; esac
    fi
    local shim_image="${MOLMIM_ARM_IMAGE:-molmim-arm-nim:local}"

    printf '%s' "$NGC_API_KEY" | "${docker_cmd[@]}" login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null

    local build_args=()
    [ -n "$platform" ] && build_args+=(--platform "$platform")
    if [ "${MOLMIM_ARM_REBUILD:-0}" = "1" ] || ! "${docker_cmd[@]}" image inspect "$shim_image" >/dev/null 2>&1; then
        echo "Building $shim_image (base: $base_image)"
        "${docker_cmd[@]}" build "${build_args[@]}" --build-arg "BASE_IMAGE=$base_image" -t "$shim_image" "$build_context"
    fi

    local container_name="openhackathon-molmim-arm-${port}"
    "${docker_cmd[@]}" rm -f "$container_name" >/dev/null 2>&1 || true

    local gpu_args=()
    case "${OPENHACKATHON_DOCKER_GPU_MODE:-device}" in
        all) gpu_args=(--gpus all) ;;
        device) gpu_args=(--gpus "device=$gpu_id") ;;
        *) echo "Error: OPENHACKATHON_DOCKER_GPU_MODE must be 'device' or 'all'." >&2; exit 1 ;;
    esac
    local run_platform_args=()
    [ -n "$platform" ] && run_platform_args+=(--platform "$platform")

    echo "Starting MolMIM ARM NIM (docker) on port $port (GPU $gpu_id); weights: $weights_dir"
    exec "${docker_cmd[@]}" run --rm \
        --name "$container_name" \
        "${run_platform_args[@]}" \
        "${gpu_args[@]}" \
        --ipc=host \
        -e "NIM_HTTP_API_PORT=$port" \
        -e "MOLMIM_CHECKPOINT=$container_cuda_checkpoint" \
        -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$gpu_id}" \
        -e "MOLMIM_SAMPLING_METHOD=$sampling_method" \
        -v "$weights_dir:/models:ro" \
        -p "127.0.0.1:${port}:${port}" \
        "$shim_image"
}

# ===========================================================================
# Apptainer / Singularity path (HPC)
# ===========================================================================
run_apptainer() {
    local rt="$runtime_bin"
    command -v "$rt" >/dev/null 2>&1 || { echo "Error: $rt not found." >&2; exit 1; }

    local sif_dir sif_path
    sif_dir="${SIF_DIR:-$PWD/.sif}"
    mkdir -p "$sif_dir"
    sif_path="${MOLMIM_ARM_SIF:-$sif_dir/molmim-arm-nim.sif}"

    # Build the .sif from the PyTorch base + molmim_arm/ code (no prebuilt NIM).
    if [ "${MOLMIM_ARM_REBUILD:-0}" = "1" ] || [ ! -f "$sif_path" ]; then
        echo "Building $sif_path (base: $base_image) via $rt build"
        export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-\$oauthtoken}"
        export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:-$NGC_API_KEY}"
        export SINGULARITY_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME:-\$oauthtoken}"
        export SINGULARITY_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD:-$NGC_API_KEY}"
        # Cap mksquashfs processors. On many-core ARM (Grace, 100+ CPUs) the
        # default (all CPUs) can segfault mksquashfs on a large image; -processors
        # keeps it stable without needing admin edits to apptainer.conf.
        local mksq_procs="${MOLMIM_ARM_MKSQUASHFS_PROCS:-4}"
        local mksq_args=()
        if "$rt" build --help 2>/dev/null | grep -q -- "--mksquashfs-args"; then
            mksq_args=(--mksquashfs-args "-processors $mksq_procs")
        fi
        ( cd "$build_context" && \
          "$rt" build --force "${mksq_args[@]}" --build-arg "BASE_IMAGE=$base_image" "$sif_path" molmim_arm.def )
    fi

    # GPU selection (mirror run_nim_apptainer.sh).
    local no_mount_args=()
    if [ ! -e /etc/localtime ] && "$rt" run --help 2>/dev/null | grep -q -- "--no-mount"; then
        no_mount_args=(--no-mount /etc/localtime)
    fi
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu_id"
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="$gpu_id"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_id"
    export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="$gpu_id"

    nvccli_available() {
        "$rt" run --help 2>/dev/null | grep -q -- "--nvccli" && command -v nvidia-container-cli >/dev/null 2>&1
    }
    nvccli_smoke_test() {
        "$rt" exec --nv --nvccli --contain --writable-tmpfs --cleanenv "${no_mount_args[@]}" "$sif_path" /bin/true >/dev/null 2>&1
    }
    local gpu_args=(--nv)
    case "${APPTAINER_GPU_MODE:-auto}" in
        auto)
            if nvccli_available && nvccli_smoke_test; then
                gpu_args=(--nv --nvccli --contain --writable-tmpfs)
                echo "GPU mode: nvccli isolated device ($gpu_id)"
            else
                echo "GPU mode: standard --nv; CUDA_VISIBLE_DEVICES=$gpu_id"
            fi
            ;;
        nvccli)
            nvccli_available && nvccli_smoke_test || { echo "Error: nvccli requested but unavailable." >&2; exit 1; }
            gpu_args=(--nv --nvccli --contain --writable-tmpfs)
            ;;
        nv) : ;;
        *) echo "Error: APPTAINER_GPU_MODE must be auto, nvccli, or nv." >&2; exit 1 ;;
    esac

    echo "Starting MolMIM ARM NIM ($rt) on port $port (GPU $gpu_id); weights: $weights_dir"
    # Apptainer shares the host network namespace, so NIM_HTTP_API_PORT (not -p)
    # selects the listen port.
    exec "$rt" run \
        "${gpu_args[@]}" \
        --cleanenv \
        "${no_mount_args[@]}" \
        --env "NIM_HTTP_API_PORT=$port" \
        --env "MOLMIM_CHECKPOINT=$container_cuda_checkpoint" \
        --env "MOLMIM_DEVICE=cuda" \
        --env "MOLMIM_SAMPLING_METHOD=$sampling_method" \
        --env "CUDA_VISIBLE_DEVICES=$gpu_id" \
        --bind "$weights_dir:/models:ro" \
        "$sif_path"
}

case "$runtime_kind" in
    docker)    run_docker ;;
    apptainer) run_apptainer ;;
    *) echo "Error: unknown runtime kind '$runtime_kind'." >&2; exit 1 ;;
esac
