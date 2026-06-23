#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Launch the RFdiffusion + ProteinMPNN NIMs locally (Docker) for the optional
# protein-binder-design stream. Both have arm64 images, so this works on GB200/
# GB300. Boltz-2 is provided separately (reuse the bootcamp's local Boltz-2, or
# point BOLTZ2_URL at a hosted endpoint).
#
# Usage:
#   export NGC_API_KEY=nvapi-...
#   protein-binder-design/launch_binder_nims.sh
# then in the notebook env:
#   export RFDIFFUSION_URL=http://localhost:${RFD_PORT:-8101}
#   export PROTEINMPNN_URL=http://localhost:${PMPNN_PORT:-8102}
set -euo pipefail

: "${NGC_API_KEY:?Set NGC_API_KEY (NGC API key) first}"
RFD_PORT="${RFD_PORT:-8101}"
PMPNN_PORT="${PMPNN_PORT:-8102}"
GPU="${PBD_GPU:-0}"
RFD_IMAGE="${RFDIFFUSION_IMAGE:-nvcr.io/nim/ipd/rfdiffusion:latest}"
PMPNN_IMAGE="${PROTEINMPNN_IMAGE:-nvcr.io/nim/ipd/proteinmpnn:latest}"
docker_bin="${DOCKER_BIN:-docker}"

printf '%s' "$NGC_API_KEY" | "$docker_bin" login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
mkdir -p "$HOME/nimcache_rfd" "$HOME/nimcache_pmpnn"
chmod 777 "$HOME/nimcache_rfd" "$HOME/nimcache_pmpnn" || true

"$docker_bin" rm -f pbd-rfdiffusion pbd-proteinmpnn >/dev/null 2>&1 || true
echo "Starting RFdiffusion on :$RFD_PORT (GPU $GPU)"
"$docker_bin" run -d --name pbd-rfdiffusion --gpus "device=$GPU" --shm-size=4g \
    -e NGC_API_KEY -e CUDA_VISIBLE_DEVICES=0 \
    -v "$HOME/nimcache_rfd:/opt/nim/.cache" -p "127.0.0.1:${RFD_PORT}:8000" "$RFD_IMAGE"
echo "Starting ProteinMPNN on :$PMPNN_PORT (GPU $GPU)"
"$docker_bin" run -d --name pbd-proteinmpnn --gpus "device=$GPU" --shm-size=4g \
    -e NGC_API_KEY -e CUDA_VISIBLE_DEVICES=0 \
    -v "$HOME/nimcache_pmpnn:/opt/nim/.cache" -p "127.0.0.1:${PMPNN_PORT}:8000" "$PMPNN_IMAGE"

echo "Waiting for readiness (first start downloads weights)..."
for url in "http://localhost:${RFD_PORT}" "http://localhost:${PMPNN_PORT}"; do
    for _ in $(seq 1 60); do
        if curl -fsS "$url/v1/health/ready" >/dev/null 2>&1; then echo "  ready: $url"; break; fi
        sleep 10
    done
done

cat <<EOF

RFdiffusion: http://localhost:${RFD_PORT}
ProteinMPNN: http://localhost:${PMPNN_PORT}

In your notebook shell:
  export RFDIFFUSION_URL=http://localhost:${RFD_PORT}
  export PROTEINMPNN_URL=http://localhost:${PMPNN_PORT}
  export BOLTZ2_URL=http://localhost:8000   # bootcamp local Boltz-2 (or a hosted endpoint)
Stop with: ${docker_bin} rm -f pbd-rfdiffusion pbd-proteinmpnn
EOF
