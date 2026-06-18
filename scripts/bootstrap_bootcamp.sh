#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${OPENHACKATHON_VENV:-$repo_root/.venv}"
boltz2_count=1
molmim_mode="${OPENHACKATHON_MOLMIM_MODE:-auto}"
boltz2_mode="${OPENHACKATHON_BOLTZ2_MODE:-auto}"
setup_python=1
start_services=1
execute_challenge=0
launch_jupyter=0
extra_service_args=()

usage() {
    cat <<'EOF'
Usage:
  scripts/bootstrap_bootcamp.sh [options]

One-command bootcamp setup:
  1. Create/update a repository-local Python virtual environment.
  2. Install deployment-requirements.txt.
  3. Launch architecture-aware NIM services.
  4. Write .openhackathon-nims.env.
  5. Run dependency and endpoint checks.

Options:
  --boltz2 N              Number of local Boltz-2 endpoints to launch. Default: 1.
  --boltz2-mode MODE      auto, local, hosted, or none. Default: auto.
  --boltz2-url URL        Hosted/external Boltz-2 URL.
  --molmim MODE          auto, local, local-arm, hosted, or none. Default: auto.
                         local-arm runs local MolMIM on aarch64 (Grace/GB200/
                         GB300) via a pure-PyTorch NIM-like service (enables CMA-ES).
  --molmim-url URL       Hosted/external MolMIM URL.
  --container-runtime R  auto, apptainer, singularity, or docker.
  --skip-python          Do not create/install the Python virtual environment.
  --skip-services        Do not start NIM services; only set up/check Python.
  --execute-challenge    Execute challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb with nbconvert.
  --jupyter              Launch JupyterLab after setup.
  --help                 Show this help.

Environment:
  NGC_API_KEY is required when services are started.
  OPENHACKATHON_HOSTED_MOLMIM_URL overrides the NVIDIA-hosted MolMIM URL.
  OPENHACKATHON_HOSTED_BOLTZ2_URL overrides the NVIDIA-hosted Boltz-2 URL.
  OPENHACKATHON_VENV overrides the virtual environment directory.

Architecture behavior:
  x86_64/amd64: start local MolMIM plus local Boltz-2; fall back to hosted endpoints.
  aarch64/arm64: start the local-arm MolMIM NIM (enables /hidden, /decode, CMA-ES)
                 with hosted MolMIM fallback; try local Boltz-2, then fall back to
                 hosted Boltz-2. Pass --molmim hosted to skip the local MolMIM build.
EOF
}

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
        --boltz2-url|--hosted-boltz2-url)
            extra_service_args+=(--boltz2-url "$2")
            shift 2
            ;;
        --molmim)
            molmim_mode="$2"
            shift 2
            ;;
        --molmim-url|--hosted-molmim-url)
            extra_service_args+=(--molmim-url "$2")
            shift 2
            ;;
        --container-runtime)
            export OPENHACKATHON_CONTAINER_RUNTIME="$2"
            shift 2
            ;;
        --skip-python)
            setup_python=0
            shift
            ;;
        --skip-services)
            start_services=0
            shift
            ;;
        --execute-challenge)
            execute_challenge=1
            shift
            ;;
        --jupyter)
            launch_jupyter=1
            shift
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

cd "$repo_root"

if [ "$setup_python" = "1" ]; then
    python_cmd="${PYTHON:-python3}"
    if ! command -v "$python_cmd" >/dev/null 2>&1; then
        echo "Error: Python command not found: $python_cmd" >&2
        exit 1
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "Creating Python virtual environment: $venv_dir"
        "$python_cmd" -m venv "$venv_dir"
    fi

    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -r deployment-requirements.txt
elif [ -f "$venv_dir/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
fi

if [ "$start_services" = "1" ]; then
    OPENHACKATHON_MOLMIM_MODE="$molmim_mode" \
    OPENHACKATHON_BOLTZ2_MODE="$boltz2_mode" \
        "$repo_root/scripts/openhackathon_services.sh" start \
        --boltz2 "$boltz2_count" \
        --boltz2-mode "$boltz2_mode" \
        --molmim "$molmim_mode" \
        "${extra_service_args[@]}"
fi

if [ -f "$repo_root/.openhackathon-nims.env" ]; then
    # shellcheck disable=SC1091
    source "$repo_root/.openhackathon-nims.env"
fi

python scoring/check_dependencies.py

if [ "$execute_challenge" = "1" ]; then
    mkdir -p executed-notebooks
    python -m jupyter nbconvert \
        --to notebook \
        --execute challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb \
        --output-dir executed-notebooks \
        --output 03_Hands-On_CDK_Inhibitor_Design.executed.ipynb \
        --ExecutePreprocessor.timeout="${OPENHACKATHON_NOTEBOOK_TIMEOUT:-7200}"
fi

if [ "$launch_jupyter" = "1" ]; then
    exec jupyter-lab Start_Here.ipynb
fi

cat <<EOF

Bootcamp setup complete.

Use:
  source "$venv_dir/bin/activate"
  source "$repo_root/.openhackathon-nims.env"
  jupyter-lab Start_Here.ipynb

Check endpoints any time with:
  scripts/openhackathon_services.sh status
EOF
