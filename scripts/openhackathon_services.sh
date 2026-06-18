#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${OPENHACKATHON_LOG_DIR:-$repo_root/logs/nims}"
env_file="${OPENHACKATHON_ENV_FILE:-$repo_root/.openhackathon-nims.env}"

molmim_port="${MOLMIM_PORT:-8001}"
boltz2_port="${BOLTZ2_PORT:-8000}"
extra_boltz2_start_port="${EXTRA_BOLTZ2_START_PORT:-8010}"
boltz2_count=1
start_molmim=1
start_boltz2=1
molmim_url_override=""
boltz2_url_override=""
external_molmim_url=""
external_boltz2_url=""
auto_ports="${OPENHACKATHON_AUTO_PORTS:-1}"
detected_arch="${OPENHACKATHON_ARCH:-$(uname -m)}"
hosted_molmim_url="${OPENHACKATHON_HOSTED_MOLMIM_URL:-https://health.api.nvidia.com/v1/biology/nvidia/molmim}"
hosted_boltz2_url="${OPENHACKATHON_HOSTED_BOLTZ2_URL:-https://health.api.nvidia.com/v1/biology/mit/boltz2}"
molmim_mode="${OPENHACKATHON_MOLMIM_MODE:-auto}"
boltz2_mode="${OPENHACKATHON_BOLTZ2_MODE:-auto}"
wait_for_ready="${OPENHACKATHON_WAIT_FOR_READY:-1}"
ready_poll_seconds="${OPENHACKATHON_READY_POLL_SECONDS:-10}"
molmim_ready_timeout="${OPENHACKATHON_MOLMIM_READY_TIMEOUT:-900}"
boltz2_ready_timeout="${OPENHACKATHON_BOLTZ2_READY_TIMEOUT:-1800}"
boltz2_predict_timeout="${OPENHACKATHON_BOLTZ2_PREDICT_TIMEOUT:-300}"
boltz2_smoke_test="${OPENHACKATHON_BOLTZ2_SMOKE_TEST:-1}"
resolved_boltz2_ports=()
reserved_ports=()

usage() {
    cat <<'EOF'
Usage:
  scripts/openhackathon_services.sh start [--boltz2 N] [--boltz2-mode auto|local|hosted|none] [--boltz2-url URL] [--molmim auto|local|local-arm|hosted|none] [--molmim-url URL]
  scripts/openhackathon_services.sh stop
  scripts/openhackathon_services.sh status
  scripts/openhackathon_services.sh env

Common workflow:
  export NGC_API_KEY=<your-ngc-key>
  scripts/openhackathon_services.sh start --boltz2 2
  source .openhackathon-nims.env
  jupyter-lab

Defaults:
  MolMIM:        http://localhost:8001
  Boltz-2 first: http://localhost:8000
  Boltz-2 extra: starts at http://localhost:8010

MolMIM selection:
  auto       x86_64/amd64 tries local MolMIM first, then hosted fallback.
             aarch64/arm64 tries the local-arm MolMIM NIM first, then hosted fallback.
  local      require local MolMIM NIM (amd64 image; not available on ARM).
  local-arm  run local MolMIM on aarch64 (Grace/GB200/GB300) via a pure-PyTorch
             NIM-like service, with hosted fallback. Enables /hidden, /decode,
             and CMA-ES guided optimization. Builds/runs via Docker or
             Apptainer/Singularity; requires NGC_API_KEY.
  hosted     use hosted MolMIM and do not launch local MolMIM.
  none       do not configure or launch MolMIM.

Boltz-2 selection:
  auto    try local Boltz-2 first, then hosted fallback if the prediction smoke test fails.
  local   require local Boltz-2 NIM.
  hosted  use hosted Boltz-2 and do not launch local Boltz-2.
  none    do not configure or launch Boltz-2.

Environment overrides:
  MOLMIM_PORT, BOLTZ2_PORT, EXTRA_BOLTZ2_START_PORT
  OPENHACKATHON_LOG_DIR, OPENHACKATHON_ENV_FILE
  OPENHACKATHON_AUTO_PORTS=0 to fail instead of picking a free port
  OPENHACKATHON_CONTAINER_RUNTIME=auto|apptainer|singularity|docker
  OPENHACKATHON_DOCKER_GPU_MODE=device|all
  OPENHACKATHON_DOCKER_USE_NVIDIA_RUNTIME=0|1
  OPENHACKATHON_MOLMIM_MODE=auto|local|local-arm|hosted|none
  OPENHACKATHON_BOLTZ2_MODE=auto|local|hosted|none
  BIONEMO_PYTORCH_BASE, MOLMIM_ARM_IMAGE, MOLMIM_WEIGHTS_DIR (for --molmim local-arm)
  OPENHACKATHON_HOSTED_MOLMIM_URL=https://health.api.nvidia.com/v1/biology/nvidia/molmim
  OPENHACKATHON_HOSTED_BOLTZ2_URL=https://health.api.nvidia.com/v1/biology/mit/boltz2
  OPENHACKATHON_WAIT_FOR_READY=0 to return immediately after launching
  OPENHACKATHON_MOLMIM_READY_TIMEOUT, OPENHACKATHON_BOLTZ2_READY_TIMEOUT
  OPENHACKATHON_BOLTZ2_SMOKE_TEST=0 to skip the prediction-level Boltz-2 check
  OPENHACKATHON_BOLTZ2_PREDICT_TIMEOUT controls the smoke-test timeout
  NGC_API_KEY, NVIDIA_API_KEY, MOLMIM_API_KEY, LOCAL_NIM_CACHE, LOCAL_NIM_WORKSPACE, SIF_DIR
  MOLMIM_IMAGE, BOLTZ2_IMAGE
EOF
}

pid_file() {
    echo "$log_dir/$1.pid"
}

port_for_boltz2() {
    local idx="$1"
    if [ "${#resolved_boltz2_ports[@]}" -gt "$idx" ]; then
        echo "${resolved_boltz2_ports[$idx]}"
        return 0
    fi

    if [ "$idx" -eq 0 ]; then
        echo "$boltz2_port"
    else
        echo $((extra_boltz2_start_port + idx - 1))
    fi
}

port_is_free() {
    local port="$1"
    local reserved
    for reserved in "${reserved_ports[@]}"; do
        if [ "$reserved" = "$port" ]; then
            return 1
        fi
    done

    if command -v ss >/dev/null 2>&1; then
        ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"
    else
        ! (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
    fi
}

find_free_port() {
    local port="$1"
    local limit=$((port + 200))
    while [ "$port" -le "$limit" ]; do
        if port_is_free "$port"; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
    done
    return 1
}

resolve_port() {
    local name="$1"
    local desired="$2"
    local actual

    if port_is_free "$desired"; then
        echo "$desired"
        return 0
    fi

    if [ "$auto_ports" != "1" ]; then
        echo "Error: port $desired for $name is already in use. Choose another port or enable OPENHACKATHON_AUTO_PORTS=1." >&2
        exit 1
    fi

    actual="$(find_free_port "$((desired + 1))")" || {
        echo "Error: could not find a free port for $name after $desired." >&2
        exit 1
    }

    echo "Port $desired for $name is in use; using $actual instead." >&2
    echo "$actual"
}

service_is_running() {
    local name="$1"
    local pid_path
    pid_path="$(pid_file "$name")"

    [ -f "$pid_path" ] && kill -0 "$(cat "$pid_path")" >/dev/null 2>&1
}

stop_service_by_name() {
    local name="$1"
    local pid_path
    pid_path="$(pid_file "$name")"

    [ -f "$pid_path" ] || return 0
    local pid
    pid="$(cat "$pid_path")"
    if kill -0 "$pid" >/dev/null 2>&1; then
        echo "Stopping $name (pid $pid)"
        kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
        sleep 1
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
        fi
    fi
    rm -f "$pid_path"
}

env_url_port() {
    local url="$1"
    echo "${url##*:}"
}

is_arm_arch() {
    case "$detected_arch" in
        aarch64|arm64)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# The ARM-local MolMIM service (--molmim local-arm) builds and runs via either
# Docker or Apptainer/Singularity (see run_molmim_arm.sh), so it is the auto-mode
# default on ARM whenever a container runtime is available. If none is usable,
# auto mode falls back to hosted MolMIM.
arm_local_molmim_supported() {
    case "${OPENHACKATHON_CONTAINER_RUNTIME:-auto}" in
        docker)
            command -v "${DOCKER_BIN:-docker}" >/dev/null 2>&1
            ;;
        apptainer)
            command -v "${APPTAINER_BIN:-apptainer}" >/dev/null 2>&1
            ;;
        singularity)
            command -v "${APPTAINER_BIN:-singularity}" >/dev/null 2>&1
            ;;
        *)
            command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1 || \
                { command -v "${DOCKER_BIN:-docker}" >/dev/null 2>&1 && "${DOCKER_BIN:-docker}" ps >/dev/null 2>&1; }
            ;;
    esac
}

is_external_url() {
    local url="$1"
    [ -n "$url" ] || return 1
    case "$url" in
        http://localhost:*|http://127.0.0.1:*|"")
            return 1
            ;;
        http://*|https://*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_hosted_api_url() {
    printf '%s\n' "$1" | grep -q 'api\.nvidia\.com'
}

has_molmim_api_key() {
    [ -n "${MOLMIM_API_KEY:-${NVIDIA_API_KEY:-${NGC_API_KEY:-}}}" ]
}

has_boltz2_api_key() {
    [ -n "${BOLTZ2_API_KEY:-${NVIDIA_API_KEY:-${NGC_API_KEY:-}}}" ]
}

require_hosted_api_key() {
    local service="$1"
    local endpoint="$2"

    is_hosted_api_url "$endpoint" || return 0
    case "$service" in
        molmim)
            has_molmim_api_key && return 0
            echo "Error: hosted MolMIM requires MOLMIM_API_KEY, NVIDIA_API_KEY, or NGC_API_KEY." >&2
            ;;
        boltz2)
            has_boltz2_api_key && return 0
            echo "Error: hosted Boltz-2 requires BOLTZ2_API_KEY, NVIDIA_API_KEY, or NGC_API_KEY." >&2
            ;;
    esac
    return 1
}

show_service_log_tail() {
    local name="$1"
    local log_file="$log_dir/$name.log"
    if [ -f "$log_file" ]; then
        echo "Last lines from $log_file:" >&2
        tail -40 "$log_file" >&2 || true
    fi
}

wait_until_ready() {
    local service="$1"
    local endpoint="$2"
    local timeout="$3"
    local name="${4:-}"
    local deadline=$((SECONDS + timeout))

    if [ "$wait_for_ready" != "1" ]; then
        return 0
    fi

    echo "Waiting up to ${timeout}s for $service at $endpoint"
    while [ "$SECONDS" -le "$deadline" ]; do
        if "$repo_root/scripts/check_nim_health.sh" "$service" "$endpoint" 1 >/dev/null 2>&1; then
            echo "$service is ready: $endpoint"
            return 0
        fi

        if [ -n "$name" ] && ! service_is_running "$name"; then
            echo "$service process $name exited before becoming ready." >&2
            show_service_log_tail "$name"
            return 1
        fi

        sleep "$ready_poll_seconds"
    done

    echo "$service did not become ready within ${timeout}s: $endpoint" >&2
    if [ -n "$name" ]; then
        show_service_log_tail "$name"
    fi
    return 1
}

run_boltz2_smoke_test() {
    local endpoint="$1"
    local name="${2:-}"

    if [ "$wait_for_ready" != "1" ] || [ "$boltz2_smoke_test" != "1" ]; then
        return 0
    fi

    if "$repo_root/scripts/check_boltz2_prediction.sh" "$endpoint" "$boltz2_predict_timeout"; then
        return 0
    fi

    echo "Boltz-2 prediction smoke test failed for $endpoint" >&2
    if [ -n "$name" ]; then
        show_service_log_tail "$name"
    fi
    return 1
}

validate_molmim_mode() {
    case "$molmim_mode" in
        auto|local|local-arm|hosted|none)
            ;;
        *)
            echo "Error: MolMIM mode must be auto, local, local-arm, hosted, or none." >&2
            exit 1
            ;;
    esac
}

validate_boltz2_mode() {
    case "$boltz2_mode" in
        auto|local|hosted|none)
            ;;
        *)
            echo "Error: Boltz-2 mode must be auto, local, hosted, or none." >&2
            exit 1
            ;;
    esac
}

boltz2_endpoint_list() {
    if [ "$start_boltz2" -eq 0 ] && [ -n "$external_boltz2_url" ]; then
        printf '%s\n' "$external_boltz2_url"
        return 0
    fi
    if [ "$start_boltz2" -eq 0 ]; then
        return 0
    fi

    if [ "${#resolved_boltz2_ports[@]}" -gt 0 ]; then
        local i
        for i in $(seq 0 $((${#resolved_boltz2_ports[@]} - 1))); do
            printf 'http://localhost:%s\n' "${resolved_boltz2_ports[$i]}"
        done
        return 0
    fi

    if is_external_url "${BOLTZ2_URL:-}"; then
        printf '%s\n' "$BOLTZ2_URL"
        return 0
    fi

    local i
    for i in $(seq 0 $((boltz2_count - 1))); do
        printf 'http://localhost:%s\n' "$(port_for_boltz2 "$i")"
    done
}

write_env_file() {
    local endpoints=()
    local endpoint
    while IFS= read -r endpoint; do
        [ -n "$endpoint" ] && endpoints+=("$endpoint")
    done < <(boltz2_endpoint_list)

    local output_molmim_url="$external_molmim_url"
    if [ -z "$output_molmim_url" ] && [ "$start_molmim" -eq 1 ]; then
        output_molmim_url="${molmim_url_override:-${MOLMIM_URL:-}}"
    fi

    {
        if [ "$start_molmim" -eq 1 ]; then
            echo "export MOLMIM_URL=\"http://localhost:$molmim_port\""
        elif [ -n "$output_molmim_url" ]; then
            echo "export MOLMIM_URL=\"$output_molmim_url\""
        else
            echo "# MOLMIM_URL is not set. Export MOLMIM_URL to a hosted MolMIM endpoint before running MolMIM notebooks."
        fi
        if [ "$start_molmim" -eq 1 ]; then
            echo "export OPENHACKATHON_ACTIVE_MOLMIM_MODE=\"local\""
            echo "export OPENHACKATHON_USE_CMA=\"1\""
        elif [ -n "$output_molmim_url" ]; then
            echo "export OPENHACKATHON_ACTIVE_MOLMIM_MODE=\"hosted\""
            echo "export OPENHACKATHON_USE_CMA=\"0\""
        else
            echo "export OPENHACKATHON_ACTIVE_MOLMIM_MODE=\"none\""
            echo "export OPENHACKATHON_USE_CMA=\"0\""
        fi
        if [ "${#endpoints[@]}" -gt 0 ]; then
            echo "export BOLTZ2_URL=\"${endpoints[0]}\""
            local joined
            joined="$(IFS=,; echo "${endpoints[*]}")"
            echo "export BOLTZ2_ENDPOINTS=\"$joined\""
            if [ "$start_boltz2" -eq 0 ] && [ -n "$external_boltz2_url" ]; then
                echo "export OPENHACKATHON_ACTIVE_BOLTZ2_MODE=\"hosted\""
            else
                echo "export OPENHACKATHON_ACTIVE_BOLTZ2_MODE=\"local\""
            fi
        else
            echo "# BOLTZ2_URL is not set. Export BOLTZ2_URL to a hosted Boltz-2 endpoint or start local Boltz-2."
            echo "export OPENHACKATHON_ACTIVE_BOLTZ2_MODE=\"none\""
        fi
    } > "$env_file"
}

start_service() {
    local name="$1"
    local service="$2"
    local port="$3"
    local gpu="$4"
    local log_file="$log_dir/$name.log"
    local pid_path
    pid_path="$(pid_file "$name")"

    if [ -f "$pid_path" ] && kill -0 "$(cat "$pid_path")" >/dev/null 2>&1; then
        echo "$name already appears to be running (pid $(cat "$pid_path"))."
        return 0
    fi

    echo "Starting $name on port $port using GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    NVIDIA_VISIBLE_DEVICES="$gpu" \
    APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" \
    APPTAINERENV_NVIDIA_VISIBLE_DEVICES="$gpu" \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu" \
    SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="$gpu" \
    nohup setsid "$repo_root/scripts/run_nim_container.sh" "$service" "$port" "$gpu" \
        >"$log_file" 2>&1 &
    echo "$!" > "$pid_path"
}

cmd_start() {
    mkdir -p "$log_dir"
    local requested_molmim_url="${MOLMIM_URL:-}"
    local requested_boltz2_url="${BOLTZ2_URL:-}"
    validate_molmim_mode
    validate_boltz2_mode

    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi

    local env_molmim_url="${MOLMIM_URL:-}"
    local env_boltz2_url="${BOLTZ2_URL:-}"
    case "$molmim_mode" in
        auto)
            if [ -n "$molmim_url_override" ]; then
                start_molmim=0
                external_molmim_url="$molmim_url_override"
            elif is_external_url "$requested_molmim_url"; then
                start_molmim=0
                external_molmim_url="$requested_molmim_url"
            elif is_arm_arch && arm_local_molmim_supported; then
                # ARM (GB200/GB300): default to the local pure-PyTorch MolMIM NIM
                # (Docker or Apptainer/Singularity) so /hidden, /decode, and CMA-ES
                # work out of the box. Falls back to hosted MolMIM if the local
                # build/health check fails.
                start_molmim=1
                export OPENHACKATHON_MOLMIM_ARM=1
                molmim_ready_timeout="${OPENHACKATHON_MOLMIM_READY_TIMEOUT:-2400}"
                if is_external_url "$env_molmim_url"; then
                    external_molmim_url="$env_molmim_url"
                else
                    external_molmim_url="$hosted_molmim_url"
                fi
            elif is_arm_arch; then
                # ARM with no usable container runtime: use hosted MolMIM.
                start_molmim=0
                if is_external_url "$env_molmim_url"; then
                    external_molmim_url="$env_molmim_url"
                else
                    external_molmim_url="$hosted_molmim_url"
                fi
            else
                start_molmim=1
                if is_external_url "$env_molmim_url"; then
                    external_molmim_url="$env_molmim_url"
                else
                    external_molmim_url="${molmim_url_override:-$hosted_molmim_url}"
                fi
            fi
            ;;
        local)
            start_molmim=1
            external_molmim_url="${molmim_url_override:-${requested_molmim_url:-$hosted_molmim_url}}"
            ;;
        local-arm)
            # Local MolMIM on aarch64 via the pure-PyTorch NIM-like service
            # (molmim_arm/; no prebuilt NIM image required).
            start_molmim=1
            export OPENHACKATHON_MOLMIM_ARM=1
            # First launch builds the molmim_arm image and downloads the
            # checkpoint, so allow a longer readiness window unless the user set
            # one explicitly.
            molmim_ready_timeout="${OPENHACKATHON_MOLMIM_READY_TIMEOUT:-2400}"
            if ! is_arm_arch; then
                echo "Warning: --molmim local-arm selected on non-ARM host ($detected_arch)." >&2
                echo "         Set DOCKER_PLATFORM/BIONEMO_PYTORCH_BASE appropriately if this is intentional." >&2
            fi
            external_molmim_url="${molmim_url_override:-${requested_molmim_url:-$hosted_molmim_url}}"
            ;;
        hosted)
            start_molmim=0
            if [ -n "$molmim_url_override" ]; then
                external_molmim_url="$molmim_url_override"
            elif is_external_url "$requested_molmim_url"; then
                external_molmim_url="$requested_molmim_url"
            elif is_external_url "$env_molmim_url"; then
                external_molmim_url="$env_molmim_url"
            else
                external_molmim_url="$hosted_molmim_url"
            fi
            ;;
        none)
            start_molmim=0
            external_molmim_url="${molmim_url_override:-}"
            ;;
    esac

    case "$boltz2_mode" in
        auto)
            if [ -n "$boltz2_url_override" ]; then
                start_boltz2=0
                external_boltz2_url="$boltz2_url_override"
            elif is_external_url "$requested_boltz2_url"; then
                start_boltz2=0
                external_boltz2_url="$requested_boltz2_url"
            else
                start_boltz2=1
                if is_external_url "$env_boltz2_url"; then
                    external_boltz2_url="$env_boltz2_url"
                else
                    external_boltz2_url="$hosted_boltz2_url"
                fi
            fi
            ;;
        local)
            start_boltz2=1
            external_boltz2_url="${boltz2_url_override:-${requested_boltz2_url:-$hosted_boltz2_url}}"
            ;;
        hosted)
            start_boltz2=0
            if [ -n "$boltz2_url_override" ]; then
                external_boltz2_url="$boltz2_url_override"
            elif is_external_url "$requested_boltz2_url"; then
                external_boltz2_url="$requested_boltz2_url"
            elif is_external_url "$env_boltz2_url"; then
                external_boltz2_url="$env_boltz2_url"
            else
                external_boltz2_url="$hosted_boltz2_url"
            fi
            ;;
        none)
            start_boltz2=0
            external_boltz2_url="${boltz2_url_override:-}"
            ;;
    esac

    if [ "$start_molmim" -eq 0 ] && [ -z "$external_molmim_url" ]; then
        echo "Warning: MolMIM will not be launched locally and no hosted MOLMIM_URL was provided." >&2
        echo "         Set MOLMIM_URL, pass --molmim-url, or use --molmim auto/local/hosted." >&2
    fi

    if [ "$start_boltz2" -eq 0 ] && [ -z "$external_boltz2_url" ]; then
        echo "Warning: Boltz-2 will not be launched locally and no hosted BOLTZ2_URL was provided." >&2
        echo "         Set BOLTZ2_URL, pass --boltz2-url, or use --boltz2-mode auto/local/hosted." >&2
    fi

    if { [ "$start_molmim" -eq 1 ] || [ "$start_boltz2" -eq 1 ]; } && [ -z "${NGC_API_KEY:-}" ]; then
        echo "Error: NGC_API_KEY is required to launch local NIM containers." >&2
        echo "       For hosted-only mode, set --molmim hosted --boltz2-mode hosted and export NVIDIA_API_KEY or service-specific API keys." >&2
        exit 1
    fi

    if [ "$start_molmim" -eq 0 ] && [ -n "$external_molmim_url" ]; then
        require_hosted_api_key "molmim" "$external_molmim_url" || exit 1
    fi
    if [ "$start_boltz2" -eq 0 ] && [ -n "$external_boltz2_url" ]; then
        require_hosted_api_key "boltz2" "$external_boltz2_url" || exit 1
    fi

    echo "Detected architecture: $detected_arch"
    if [ "$start_molmim" -eq 1 ] && [ "${OPENHACKATHON_MOLMIM_ARM:-0}" = "1" ]; then
        echo "MolMIM mode: local-arm (pure-PyTorch NIM-like service) with hosted fallback ($external_molmim_url)"
    elif [ "$start_molmim" -eq 1 ]; then
        echo "MolMIM mode: local with hosted fallback ($external_molmim_url)"
    elif [ -n "$external_molmim_url" ]; then
        echo "MolMIM mode: hosted/external ($external_molmim_url)"
    else
        echo "MolMIM mode: none"
    fi
    if [ "$start_boltz2" -eq 1 ]; then
        echo "Boltz-2 mode: local with hosted fallback ($external_boltz2_url)"
    elif [ -n "$external_boltz2_url" ]; then
        echo "Boltz-2 mode: hosted/external ($external_boltz2_url)"
    else
        echo "Boltz-2 mode: none"
    fi

    local gpu_count=1
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(nvidia-smi -L | wc -l)"
        if [ "$gpu_count" -lt 1 ]; then
            gpu_count=1
        fi
    fi

    if [ "$start_molmim" -eq 1 ]; then
        if service_is_running "molmim" && [ -n "${MOLMIM_URL:-}" ]; then
            molmim_port="$(env_url_port "$MOLMIM_URL")"
        else
            molmim_port="$(resolve_port "molmim" "$molmim_port")"
        fi
        reserved_ports+=("$molmim_port")
        start_service "molmim" "molmim" "$molmim_port" 0
        if ! wait_until_ready "molmim" "http://localhost:$molmim_port" "$molmim_ready_timeout" "molmim"; then
            if { [ "$molmim_mode" = "auto" ] || [ "$molmim_mode" = "local-arm" ]; } && [ -n "$external_molmim_url" ]; then
                echo "Local MolMIM did not become healthy; falling back to hosted/external MolMIM."
                stop_service_by_name "molmim"
                start_molmim=0
                unset OPENHACKATHON_MOLMIM_ARM
            else
                echo "Error: local MolMIM did not become ready." >&2
                exit 1
            fi
        fi
    fi

    if [ "$start_boltz2" -eq 1 ]; then
        local existing_boltz2_endpoints="${BOLTZ2_ENDPOINTS:-}"
        local existing_boltz2_urls=()
        if [ -n "$existing_boltz2_endpoints" ]; then
            IFS=',' read -r -a existing_boltz2_urls <<< "$existing_boltz2_endpoints"
        fi

        for i in $(seq 0 $((boltz2_count - 1))); do
            local desired_port
            local actual_port
            if service_is_running "boltz2-$i" && [ "${#existing_boltz2_urls[@]}" -gt "$i" ]; then
                actual_port="$(env_url_port "${existing_boltz2_urls[$i]}")"
            else
                desired_port="$(port_for_boltz2 "$i")"
                actual_port="$(resolve_port "boltz2-$i" "$desired_port")"
            fi
            resolved_boltz2_ports+=("$actual_port")
            reserved_ports+=("$actual_port")
            start_service "boltz2-$i" "boltz2" "$actual_port" "$((i % gpu_count))"
            sleep 2
        done
    fi

    if [ "$start_molmim" -eq 0 ] && [ -n "$external_molmim_url" ]; then
        require_hosted_api_key "molmim" "$external_molmim_url" || exit 1
        wait_until_ready "molmim" "$external_molmim_url" "$molmim_ready_timeout" || {
            echo "Error: hosted/external MolMIM is not ready." >&2
            exit 1
        }
    fi

    if [ "$start_boltz2" -eq 1 ]; then
        local boltz2_failed=0
        for i in $(seq 0 $((boltz2_count - 1))); do
            local boltz2_endpoint="http://localhost:$(port_for_boltz2 "$i")"
            wait_until_ready "boltz2" "$boltz2_endpoint" "$boltz2_ready_timeout" "boltz2-$i" || {
                echo "Error: Boltz-2 endpoint $i is not ready." >&2
                boltz2_failed=1
                break
            }
            run_boltz2_smoke_test "$boltz2_endpoint" "boltz2-$i" || {
                boltz2_failed=1
                break
            }
        done

        if [ "$boltz2_failed" -eq 1 ]; then
            if [ "$boltz2_mode" = "auto" ] && [ -n "$external_boltz2_url" ]; then
                echo "Local Boltz-2 failed the readiness/prediction check; falling back to hosted/external Boltz-2."
                for i in $(seq 0 $((boltz2_count - 1))); do
                    stop_service_by_name "boltz2-$i"
                done
                start_boltz2=0
                resolved_boltz2_ports=()
                require_hosted_api_key "boltz2" "$external_boltz2_url" || exit 1
                wait_until_ready "boltz2" "$external_boltz2_url" "$boltz2_ready_timeout" || {
                    echo "Error: hosted/external Boltz-2 is not configured." >&2
                    exit 1
                }
                run_boltz2_smoke_test "$external_boltz2_url" || {
                    echo "Error: hosted/external Boltz-2 did not pass the prediction smoke test." >&2
                    exit 1
                }
            else
                echo "Error: local Boltz-2 did not pass readiness checks." >&2
                exit 1
            fi
        fi
    elif [ -n "$external_boltz2_url" ]; then
        require_hosted_api_key "boltz2" "$external_boltz2_url" || exit 1
        wait_until_ready "boltz2" "$external_boltz2_url" "$boltz2_ready_timeout" || {
            echo "Error: hosted/external Boltz-2 is not configured." >&2
            exit 1
        }
        run_boltz2_smoke_test "$external_boltz2_url" || {
            echo "Error: hosted/external Boltz-2 did not pass the prediction smoke test." >&2
            exit 1
        }
    fi

    write_env_file

    echo ""
    echo "Environment written to: $env_file"
    echo "Use:"
    echo "  source $env_file"
    echo "  scripts/openhackathon_services.sh status"
}

cmd_stop() {
    if [ ! -d "$log_dir" ]; then
        echo "No log directory found: $log_dir"
        return 0
    fi

    local found=0
    for pid_path in "$log_dir"/*.pid; do
        [ -e "$pid_path" ] || continue
        found=1
        local pid
        pid="$(cat "$pid_path")"
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "Stopping $(basename "$pid_path" .pid) (pid $pid)"
            kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
            sleep 1
            if kill -0 "$pid" >/dev/null 2>&1; then
                kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
            fi
        else
            echo "$(basename "$pid_path" .pid) is not running"
        fi
        rm -f "$pid_path"
    done

    if [ "$found" -eq 0 ]; then
        echo "No service pid files found in $log_dir"
    fi
}

cmd_status() {
    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi

    echo "Architecture: $detected_arch"
    echo "MolMIM mode: ${OPENHACKATHON_ACTIVE_MOLMIM_MODE:-$molmim_mode}"
    echo "Boltz-2 mode: ${OPENHACKATHON_ACTIVE_BOLTZ2_MODE:-$boltz2_mode}"
    echo "Logs: $log_dir"
    if [ -n "${MOLMIM_URL:-}" ]; then
        "$repo_root/scripts/check_nim_health.sh" molmim "$MOLMIM_URL" 1 || true
    else
        echo "MolMIM URL is not configured. Set MOLMIM_URL to a hosted endpoint or start local MolMIM."
    fi

    if [ -n "${BOLTZ2_ENDPOINTS:-${BOLTZ2_URL:-}}" ]; then
        local endpoints="${BOLTZ2_ENDPOINTS:-$BOLTZ2_URL}"
        IFS=',' read -r -a urls <<< "$endpoints"
        for url in "${urls[@]}"; do
            "$repo_root/scripts/check_nim_health.sh" boltz2 "$url" 1 || true
        done
    else
        echo "Boltz-2 URL is not configured. Set BOLTZ2_URL to a hosted endpoint or start local Boltz-2."
    fi
}

cmd_env() {
    validate_molmim_mode
    validate_boltz2_mode

    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi

    case "$molmim_mode" in
        hosted)
            start_molmim=0
            if [ -n "$molmim_url_override" ]; then
                external_molmim_url="$molmim_url_override"
            elif is_external_url "${MOLMIM_URL:-}"; then
                external_molmim_url="$MOLMIM_URL"
            else
                external_molmim_url="$hosted_molmim_url"
            fi
            ;;
        none)
            start_molmim=0
            external_molmim_url="${molmim_url_override:-}"
            ;;
        local|local-arm)
            start_molmim=1
            ;;
        auto)
            if [ -n "$molmim_url_override" ]; then
                start_molmim=0
                external_molmim_url="$molmim_url_override"
            elif is_external_url "${MOLMIM_URL:-}"; then
                start_molmim=0
                external_molmim_url="$MOLMIM_URL"
            elif is_arm_arch; then
                start_molmim=0
                external_molmim_url="$hosted_molmim_url"
            else
                start_molmim=1
            fi
            ;;
    esac

    case "$boltz2_mode" in
        hosted)
            start_boltz2=0
            if [ -n "$boltz2_url_override" ]; then
                external_boltz2_url="$boltz2_url_override"
            elif is_external_url "${BOLTZ2_URL:-}"; then
                external_boltz2_url="$BOLTZ2_URL"
            else
                external_boltz2_url="$hosted_boltz2_url"
            fi
            ;;
        none)
            start_boltz2=0
            external_boltz2_url="${boltz2_url_override:-}"
            ;;
        local)
            start_boltz2=1
            ;;
        auto)
            if [ -n "$boltz2_url_override" ]; then
                start_boltz2=0
                external_boltz2_url="$boltz2_url_override"
            elif is_external_url "${BOLTZ2_URL:-}"; then
                start_boltz2=0
                external_boltz2_url="$BOLTZ2_URL"
            else
                start_boltz2=1
            fi
            ;;
    esac

    write_env_file
    cat "$env_file"
}

cmd="${1:-}"
shift || true

while [ "$#" -gt 0 ]; do
    case "$1" in
        --boltz2)
            boltz2_count="$2"
            shift 2
            ;;
        --boltz2-mode)
            boltz2_mode="$2"
            shift 2
            ;;
        --no-boltz2)
            boltz2_mode="none"
            start_boltz2=0
            shift
            ;;
        --boltz2-url|--hosted-boltz2-url)
            boltz2_url_override="$2"
            external_boltz2_url="$2"
            boltz2_mode="hosted"
            start_boltz2=0
            shift 2
            ;;
        --molmim)
            molmim_mode="$2"
            shift 2
            ;;
        --no-molmim)
            molmim_mode="none"
            start_molmim=0
            shift
            ;;
        --molmim-url|--hosted-molmim-url)
            molmim_url_override="$2"
            external_molmim_url="$2"
            molmim_mode="hosted"
            start_molmim=0
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$cmd" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    env) cmd_env ;;
    help|--help|-h|"") usage ;;
    *)
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
