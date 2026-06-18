#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

export NIM_HTTP_API_PORT="${NIM_HTTP_API_PORT:-8001}"
export MOLMIM_CHECKPOINT="${MOLMIM_CHECKPOINT:-/models/molmim_70m_24_3.nemo}"
export MOLMIM_DEVICE="${MOLMIM_DEVICE:-cuda}"
export MOLMIM_SAMPLING_METHOD="${MOLMIM_SAMPLING_METHOD:-greedy}"
export PYTHONPATH="/opt/molmim-shim:${PYTHONPATH:-}"

echo "MolMIM ARM (pure-torch, NIM-like) service"
echo "  port=${NIM_HTTP_API_PORT}  device=${MOLMIM_DEVICE}  checkpoint=${MOLMIM_CHECKPOINT}"
if [ ! -f "$MOLMIM_CHECKPOINT" ]; then
  echo "Warning: checkpoint not found at $MOLMIM_CHECKPOINT; mount the .nemo into /models." >&2
fi

cd /opt/molmim-shim
exec python server.py
