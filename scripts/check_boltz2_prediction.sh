#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

endpoint="${1:-http://localhost:8000}"
timeout_seconds="${2:-${OPENHACKATHON_BOLTZ2_PREDICT_TIMEOUT:-300}}"
api_key="${BOLTZ2_API_KEY:-${NVIDIA_API_KEY:-${NGC_API_KEY:-}}}"

predict_url() {
    local value="${1%/}"

    case "$value" in
        */predict)
            printf '%s\n' "$value"
            ;;
        */biology/mit/boltz2)
            printf '%s/predict\n' "$value"
            ;;
        */v1)
            printf '%s/biology/mit/boltz2/predict\n' "$value"
            ;;
        https://*.api.nvidia.com)
            printf '%s/v1/biology/mit/boltz2/predict\n' "$value"
            ;;
        *)
            printf '%s/biology/mit/boltz2/predict\n' "$value"
            ;;
    esac
}

url="$(predict_url "$endpoint")"

headers=(-H "accept: application/json" -H "Content-Type: application/json")
if [ -n "$api_key" ]; then
    headers+=(-H "Authorization: Bearer $api_key")
fi

payload='{
  "polymers": [
    {
      "id": "A",
      "molecule_type": "protein",
      "sequence": "MTEYKLVVVGACGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVID"
    }
  ],
  "recycling_steps": 1,
  "sampling_steps": 5,
  "diffusion_samples": 1,
  "step_scale": 1.638,
  "output_format": "mmcif"
}'

printf "Boltz-2 prediction smoke test: %s\n" "$url"
curl -fsS --max-time "$timeout_seconds" "${headers[@]}" -d "$payload" "$url" >/dev/null
printf "Boltz-2 prediction smoke test passed: %s\n" "$url"
