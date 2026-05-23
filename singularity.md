# Singularity/Apptainer Deployment

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


This bootcamp should run on clusters that do not provide Docker or the NVIDIA
Container Runtime. The recommended HPC path is to run the NIM containers with
Apptainer or Singularity and `--nv`.

For Docker-only GB200/GB300 ARM systems, use the Docker path in
[`deployment.md`](deployment.md) with
`OPENHACKATHON_CONTAINER_RUNTIME=docker`. The service wrapper uses
`scripts/run_nim_container.sh`, which chooses Apptainer/Singularity when
available and Docker when explicitly requested or when it is the only available
runtime. The Docker launcher checks requested image tags for native platform
support before pulling; Boltz-2 `1.7.0` is the current default local image and
MolMIM `1.0.0` is not a native ARM container. ARM hosts that still run a
pre-590 driver default to Boltz-2 `1.4.0` unless `BOLTZ2_IMAGE` is set. For ARM
deployments, use a NVIDIA-hosted or x86-hosted MolMIM endpoint by setting
`MOLMIM_URL` plus `MOLMIM_API_KEY` or `NVIDIA_API_KEY` when authentication is
required. The wrapper tries local Boltz-2 and then falls back to hosted Boltz-2
if a prediction smoke test fails.

The Docker commands in NVIDIA NIM documentation map host ports with `-p`.
Apptainer normally shares the host network namespace, so these scripts set
`NIM_HTTP_API_PORT` instead of using Docker-style port mapping.

The Boltz-2 configuration page is the source of truth for NIM-specific
environment variables:

https://docs.nvidia.com/nim/bionemo/boltz2/latest/configure.html

## Prerequisites

- A GPU allocation on a compute node.
- `apptainer` or `singularity` available in `PATH`.
- An NGC API key exported as `NGC_API_KEY`.
- Enough local or shared storage for NIM model caches.

```bash
export NGC_API_KEY=<your-ngc-key>
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"
```

## Quick Start

For most learners, use the service wrapper. It hides image pulls, PIDs, logs,
ports, and endpoint environment variables:

```bash
export NGC_API_KEY=<your-ngc-key>
scripts/bootstrap_bootcamp.sh --boltz2 1
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
jupyter-lab
```

This installs Python dependencies and starts:

- MolMIM locally on x86_64/amd64, with hosted fallback
- Boltz-2 at `http://localhost:8000`, or the next free port if `8000` is busy,
  with hosted fallback when the local prediction smoke test fails

The selected endpoints are written to `.openhackathon-nims.env`. Always source
that file before running notebooks or scoring jobs:

```bash
source .openhackathon-nims.env
```

Stop services with:

```bash
scripts/openhackathon_services.sh stop
```

Logs are written to `logs/nims/` by default.

To add more Boltz-2 endpoints, pass a larger count, for example
`scripts/openhackathon_services.sh start --boltz2 4`. On clusters where
Apptainer falls back to plain `--nv`, each Boltz-2 service may see every GPU, so
start with one endpoint unless you have confirmed the local GPU isolation mode.

## GPU Isolation

Boltz-2 honors `NVIDIA_VISIBLE_DEVICES`, and the launcher sets it from the GPU
index argument. With Apptainer, plain `--nv` can still expose all host NVIDIA
device files. When available, the launcher automatically uses Apptainer's
`--nvccli --contain` mode so `NVIDIA_VISIBLE_DEVICES` controls which GPU device
is bound into the container. Some setuid Apptainer installs advertise
`--nvccli` but reject it at runtime; the launcher probes this path and falls
back to plain `--nv` when the probe fails.

To require this stricter mode and fail fast if the cluster lacks
`nvidia-container-cli`, set:

```bash
export APPTAINER_GPU_MODE=nvccli
```

To force the legacy Apptainer behavior, set:

```bash
export APPTAINER_GPU_MODE=nv
```

If `--nvccli` is not available, run one Boltz-2 service per allocation or ask
the cluster admins to enable non-setuid Apptainer/user namespace support for
NVIDIA Container CLI integration.

On the deployment test cluster, `--nvccli` was present but rejected by the
setuid Apptainer install. The launcher fell back to plain `--nv`, so use one
local Boltz-2 endpoint per allocation unless the site confirms GPU isolation.

## Port Conflicts

On shared nodes another service may already own `8000`. By default the wrapper
detects busy ports and picks the next free port, then writes the actual
endpoint values to `.openhackathon-nims.env`.

To fail instead of auto-selecting ports:

```bash
export OPENHACKATHON_AUTO_PORTS=0
```

If `8000` is busy, do not hard-code ports in notebooks. Use:

```bash
source .openhackathon-nims.env
```

The challenge and scoring code read `MOLMIM_URL`, `BOLTZ2_URL`, and
`BOLTZ2_ENDPOINTS` from that file.

When using hosted MolMIM with local Boltz-2 explicitly, start the wrapper with:

```bash
export NGC_API_KEY=<your-ngc-key>
scripts/openhackathon_services.sh start --molmim hosted --boltz2 1
source .openhackathon-nims.env
```

By default, `--molmim auto` uses hosted MolMIM on aarch64/arm64 and local
MolMIM with hosted fallback on x86_64/amd64. The NVIDIA-hosted MolMIM endpoint
supports molecule generation. Workflows that need MolMIM latent `/hidden` and
`/decode` endpoints for CMA-ES should use a local or x86-hosted MolMIM NIM.

By default, `--boltz2-mode auto` launches local Boltz-2 and checks both
`/v1/health/ready` and a tiny prediction request. If local Boltz-2 reports ready
but prediction workers cannot get a GPU, the wrapper falls back to
`https://health.api.nvidia.com/v1/biology/mit/boltz2`. Set
`OPENHACKATHON_BOLTZ2_SMOKE_TEST=0` only when you intentionally want to skip
that prediction-level check.

## Manual Start: MolMIM

The notebooks expect MolMIM at `http://localhost:8001` by default.

```bash
scripts/run_nim_apptainer.sh molmim 8001 0
```

In another terminal on the same node:

```bash
scripts/check_nim_health.sh molmim 8001 1
curl -X POST http://localhost:8001/embedding \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"sequences": ["CC(Cc1ccc(cc1)C(C(=O)O)C)C"]}'
```

## Manual Start: Boltz-2

For a single endpoint:

```bash
scripts/run_nim_apptainer.sh boltz2 8000 0
```

For multiple endpoints on a multi-GPU node, source the generated environment or
set `BOLTZ2_ENDPOINTS` explicitly:

```bash
scripts/run_nim_apptainer.sh boltz2 8000 0
scripts/launch_multiple_boltz2_apptainer.sh 3 8010
scripts/check_nim_health.sh boltz2 8000 1
scripts/check_boltz2_prediction.sh http://localhost:8000
scripts/check_nim_health.sh boltz2 8010 3
export BOLTZ2_ENDPOINTS="http://localhost:8000,http://localhost:8010,http://localhost:8011,http://localhost:8012"
```

## Run the Evaluation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deployment-requirements.txt

cd scoring
source ../.openhackathon-nims.env
python evaluate_submission_parallel_no_mock.py cdk_test_compounds.csv CDK_Validation \
  --max-workers 8 \
  --skip-toxicity --skip-novelty \
  --verbose
```

When `BOLTZ2_ENDPOINTS` is present, the evaluator uses those endpoints by
default. Use `--endpoints` only when you need to override the generated service
environment.

## Operational Notes

- Use `module load python312` or the site-provided Python module before creating
  a virtual environment if system Python lacks `ensurepip`.
- Keep NIM caches on fast storage when possible. Model download and package
  installation can be slow or appear stuck on heavily shared filesystems.
- If several users share a node, choose non-conflicting ports and set
  `MOLMIM_URL`, `BOLTZ2_URL`, or `BOLTZ2_ENDPOINTS` accordingly.
- To override image tags, set `MOLMIM_IMAGE` or `BOLTZ2_IMAGE`.
- To force a runtime for the wrapper, set
  `OPENHACKATHON_CONTAINER_RUNTIME=apptainer`, `singularity`, or `docker`.
- If `scoring/chembl_data/chembl_fingerprints.pkl` is missing or is only a Git
  LFS pointer, the hands-on CDK notebook falls back to seed/reference novelty
  scoring. For full ChEMBL novelty scoring, run `git lfs pull` or rebuild the
  cache with `cd scoring && python create_chembl_database.py`.
- Troubleshooting starts with `scripts/openhackathon_services.sh status`, then
  the per-service logs under `logs/nims/`.
