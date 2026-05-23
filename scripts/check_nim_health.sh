#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

service="${1:-boltz2}"
endpoint="${2:-8000}"
num_instances="${3:-1}"
all_ready=1

case "$service" in
    molmim)
        path="/v1/health/ready"
        ;;
    boltz2)
        path="/v1/health/ready"
        ;;
    *)
        echo "Error: service must be 'molmim' or 'boltz2'." >&2
        exit 1
        ;;
esac

auth_header=()
case "$service" in
    molmim)
        api_key="${MOLMIM_API_KEY:-${NVIDIA_API_KEY:-${NGC_API_KEY:-}}}"
        ;;
    boltz2)
        api_key="${BOLTZ2_API_KEY:-${NVIDIA_API_KEY:-${NGC_API_KEY:-}}}"
        ;;
esac

if [ -n "${api_key:-}" ]; then
    auth_header=(-H "Authorization: Bearer $api_key")
fi

molmim_generate_url() {
    local value="${1%/}"

    case "$value" in
        *://integrate.api.nvidia.com/*)
            value="${value/integrate.api.nvidia.com/health.api.nvidia.com}"
            ;;
    esac

    case "$value" in
        */generate)
            printf '%s\n' "$value"
            ;;
        */biology/nvidia/molmim)
            printf '%s/generate\n' "$value"
            ;;
        */v1)
            printf '%s/biology/nvidia/molmim/generate\n' "$value"
            ;;
        https://*.api.nvidia.com)
            printf '%s/v1/biology/nvidia/molmim/generate\n' "$value"
            ;;
        *)
            printf '%s/biology/nvidia/molmim/generate\n' "$value"
            ;;
    esac
}

is_hosted_api_url() {
    printf '%s\n' "$1" | grep -q 'api\.nvidia\.com'
}

health_url() {
    local value="$1"
    local offset="$2"

    case "$value" in
        http://*|https://*)
            if [ "$offset" -gt 0 ]; then
                return 1
            fi
            printf '%s%s\n' "${value%/}" "$path"
            ;;
        *:*)
            if [ "$offset" -gt 0 ]; then
                return 1
            fi
            printf 'http://%s%s\n' "$value" "$path"
            ;;
        *)
            if ! printf '%s\n' "$value" | grep -Eq '^[0-9]+$'; then
                if [ "$offset" -gt 0 ]; then
                    return 1
                fi
                printf 'http://%s%s\n' "$value" "$path"
                return 0
            fi
            port=$((value + offset))
            printf 'http://localhost:%s%s\n' "$port" "$path"
            ;;
    esac
}

for i in $(seq 0 $((num_instances - 1))); do
    if [ "$service" = "molmim" ] && is_hosted_api_url "$endpoint"; then
        url="$(molmim_generate_url "$endpoint")"
        printf "%s " "$url"
        if [ -z "${api_key:-}" ]; then
            printf "not ready (set MOLMIM_API_KEY, NVIDIA_API_KEY, or NGC_API_KEY)\n"
            all_ready=0
            break
        fi
        if curl -fsS --max-time 30 "${auth_header[@]}" \
            -H "accept: application/json" \
            -H "Content-Type: application/json" \
            -d '{"smi":"CCO","algorithm":"none","num_molecules":1,"particles":2,"scaled_radius":1.0}' \
            "$url" >/dev/null; then
            printf "ready\n"
        else
            printf "not ready\n"
            all_ready=0
        fi
        break
    fi

    if [ "$service" = "boltz2" ] && is_hosted_api_url "$endpoint"; then
        printf "%s " "$endpoint"
        if [ -z "${api_key:-}" ]; then
            printf "not ready (set BOLTZ2_API_KEY, NVIDIA_API_KEY, or NGC_API_KEY)\n"
            all_ready=0
        else
            printf "configured (hosted endpoint; readiness is checked by prediction smoke tests)\n"
        fi
        break
    fi

    url="$(health_url "$endpoint" "$i")" || continue
    printf "%s " "$url"
    if curl -fsS --max-time 5 "${auth_header[@]}" "$url"; then
        printf "\n"
    else
        printf "not ready\n"
        all_ready=0
    fi
done

[ "$all_ready" = "1" ]
