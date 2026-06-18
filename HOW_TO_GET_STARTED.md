# How To Get Started

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

This guide summarizes the dependencies and first-run workflow for implementing the AI-Powered-Drug-Discovery-Bootcamp from this branch. The recommended deployment path for clusters is Apptainer/Singularity because many shared GPU systems do not provide Docker runtime support.

## What You Need

### System Runtime

- A Linux GPU workstation or GPU compute-node allocation.
- NVIDIA driver access on the node where the NIM services run.
- `apptainer` or `singularity` available in `PATH` for the cluster workflow, or
  Docker with NVIDIA Container Toolkit for workstation/GB200/GB300 ARM testing.
- An NGC API key with access to the BioNeMo NIM images.
- Enough local or shared storage for NIM image files and model caches.

Docker with NVIDIA Container Toolkit is supported as an alternate workstation
path and as the recommended local Boltz-2 validation path on GB200/GB300 ARM
systems that do not have Apptainer installed. See [`deployment.md`](deployment.md)
for Docker details and current image architecture notes.

### NIM Services

The bootcamp notebooks expect BioNeMo NIM endpoints:

- MolMIM NIM for molecular generation, hidden-state encoding, and decoding.
- Boltz-2 NIM for protein-ligand affinity prediction.

On x86 GPU clusters, both services can be self-hosted. On GB200, GB300, and
other ARM64 nodes, you can run local MolMIM with `--molmim local-arm` (a
pure-PyTorch MolMIM NIM that exposes `/hidden` and `/decode`) or use a
NVIDIA-hosted/x86-hosted MolMIM endpoint; either way, try local Boltz-2 with a
hosted Boltz-2 fallback if prediction requests do not complete.

The service wrapper starts these services and writes endpoint values to `.openhackathon-nims.env`:

- `MOLMIM_URL`
- `BOLTZ2_URL`
- `BOLTZ2_ENDPOINTS`

For NVIDIA-hosted MolMIM, set `MOLMIM_URL` to
`https://health.api.nvidia.com/v1/biology/nvidia/molmim` and export
`MOLMIM_API_KEY` or `NVIDIA_API_KEY` for bearer-token authentication. Hosted
MolMIM supports generation, while local MolMIM NIMs — including the ARM
`--molmim local-arm` service — also expose the latent `/hidden` and `/decode`
endpoints used by CMA-ES notebooks.

For NVIDIA-hosted Boltz-2 fallback, the wrapper uses
`https://health.api.nvidia.com/v1/biology/mit/boltz2` by default and exports it
as `BOLTZ2_URL`/`BOLTZ2_ENDPOINTS` when local Boltz-2 fails the prediction smoke
test. Override it with `OPENHACKATHON_HOSTED_BOLTZ2_URL` or `--boltz2-url`.

Always source `.openhackathon-nims.env` before running notebooks or scoring scripts.

### Python Packages

Install the Python dependencies from [`deployment-requirements.txt`](deployment-requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deployment-requirements.txt
```

The current dependency set includes:

```text
numpy
pandas
matplotlib
seaborn
cma
rdkit
biopython
tqdm
requests
nest_asyncio
sqlalchemy
boltz2-python-client>=0.5.2.post1
jupyterlab
scipy
```

`nest_asyncio` lets the notebooks run the Boltz-2 affinity client's async calls
inside the Jupyter kernel; without it, the affinity cells raise
`RuntimeError: This event loop is already running`.

## Quick Start On A Cluster

From the repository root:

```bash
export NGC_API_KEY=<your-ngc-key>
scripts/bootstrap_bootcamp.sh --boltz2 1
source .openhackathon-nims.env

jupyter-lab Start_Here.ipynb
```

This creates `.venv`, installs the Python dependencies, starts one Boltz-2
endpoint, configures MolMIM, waits for endpoint health checks, runs a small
Boltz-2 prediction smoke test, and writes `.openhackathon-nims.env`.

The launch path is architecture-aware:

- `x86_64`/`amd64`: try local MolMIM plus local Boltz-2, then fall back to
  hosted endpoints if local services do not pass their checks.
- `aarch64`/`arm64`: default to hosted MolMIM, try local Boltz-2, then fall back
  to hosted Boltz-2 if local prediction requests hang or fail. Pass
  `--molmim local-arm` to run local MolMIM on ARM (enables `/hidden`, `/decode`,
  and CMA-ES).

For Docker-only GB200/GB300 ARM nodes, set the runtime and run the same
bootstrap command:

```bash
export NGC_API_KEY=<your-ngc-key>
export OPENHACKATHON_CONTAINER_RUNTIME=docker
scripts/bootstrap_bootcamp.sh --molmim local-arm --boltz2 1
```

The Docker launcher selects `linux/arm64` automatically on `aarch64` hosts and
checks that the image tag advertises that platform before pulling. The
`--molmim local-arm` option builds and runs a pure-PyTorch MolMIM NIM (see
[`deployment.md`](deployment.md)); omit it to use hosted MolMIM (generation
only). Boltz-2 `1.7.0` is the default ARM-capable tag for current local testing. On ARM hosts
with a pre-590 NVIDIA driver, the launcher defaults Boltz-2 to
`nvcr.io/nim/mit/boltz2:1.4.0`. The launching user
must be able to run
`docker ps` without an interactive sudo prompt; otherwise add the user to the
Docker group or use an administrator managed launch shell.

The Docker path uses `--gpus` by default. Set
`OPENHACKATHON_DOCKER_USE_NVIDIA_RUNTIME=1` only on hosts where
`docker run --runtime=nvidia ...` is supported.

The generated `.openhackathon-nims.env` records the selected MolMIM and Boltz-2
URLs, plus notebook generation mode, so hosted MolMIM can be combined with local
or hosted Boltz-2 without editing notebook code.

## Multi-GPU Notes

If you have multiple GPUs allocated, the wrapper can start more than one Boltz-2 endpoint:

```bash
scripts/openhackathon_services.sh start --boltz2 4
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

On some Apptainer/Singularity installations, plain `--nv` exposes all GPUs to every container even when `NVIDIA_VISIBLE_DEVICES` is set. Start with one Boltz-2 endpoint unless you have confirmed that the cluster supports strict GPU isolation. See [`singularity.md`](singularity.md) for details.

## Notebook Paths

Start here:

- [`Start_Here.ipynb`](Start_Here.ipynb)

Full path:

- [`tutorials/00_Container_Setup.ipynb`](tutorials/00_Container_Setup.ipynb)
- [`challenge/02_The_Challenge-Designing_CDK4_Inhibitors.ipynb`](challenge/02_The_Challenge-Designing_CDK4_Inhibitors.ipynb)
- [`challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb`](challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb)

Compact path:

- [`mini-hands-on/00_Introduction.ipynb`](mini-hands-on/00_Introduction.ipynb)

Optional deeper dives:

- [`tutorials/01_MolMIMGeneration.ipynb`](tutorials/01_MolMIMGeneration.ipynb)
- [`tutorials/02_ClusterMolMIMEmbeddings.ipynb`](tutorials/02_ClusterMolMIMEmbeddings.ipynb)
- [`tutorials/03_MolMIMInterpolation.ipynb`](tutorials/03_MolMIMInterpolation.ipynb)
- [`tutorials/04_MolMIMOracleControlledGeneration.ipynb`](tutorials/04_MolMIMOracleControlledGeneration.ipynb)
- [`tutorials/05_Suggested_Tools_for_Scoring_Oracles.ipynb`](tutorials/05_Suggested_Tools_for_Scoring_Oracles.ipynb)
- [`tutorials/06_Boltz2_Validation.ipynb`](tutorials/06_Boltz2_Validation.ipynb)

## Running The CDK Challenge

The hands-on CDK notebook defaults to demo mode so it can finish in a live bootcamp session with one Boltz-2 endpoint. For a larger exploratory run, set:

```bash
export OPENHACKATHON_DEMO_MODE=0
```

You can also tune individual search knobs:

```bash
export OPENHACKATHON_CMA_ITERATIONS=10
export OPENHACKATHON_CMA_POPSIZE=20
export OPENHACKATHON_TOP_K_FOR_BOLTZ2=10
```

Then launch Jupyter from the same shell:

```bash
source .openhackathon-nims.env
jupyter-lab challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb
```

## Optional Components

- **ChEMBL novelty cache**: Full novelty scoring uses `scoring/chembl_data/chembl_fingerprints.pkl`. If that file is missing or only present as a Git LFS pointer, the challenge notebook falls back to seed/reference novelty scoring. For full novelty scoring, run `git lfs pull` or rebuild the cache with `cd scoring && python create_chembl_database.py`.
- **ReaSyn synthesis pathway prediction**: ReaSyn is optional. The notebook cells skip cleanly unless a ReaSyn MCP server is running and `REASYN_URL` is set when needed.
- **Docker workstation or GB200/GB300 ARM path**: Use
  [`deployment.md`](deployment.md) if running on a workstation or ARM node with
  Docker and NVIDIA Container Toolkit.

## Stop Services

When finished:

```bash
scripts/openhackathon_services.sh stop
```

Logs are written under `logs/nims/` by default.

## Troubleshooting

Check service status first:

```bash
scripts/openhackathon_services.sh status
```

Then verify Python and endpoint dependencies:

```bash
source .openhackathon-nims.env
python scoring/check_dependencies.py
```

If a port is busy on a shared node, the wrapper selects the next free port and records the actual endpoint in `.openhackathon-nims.env`. Do not hard-code `localhost:8000` in notebooks or scoring scripts.

If Docker can see the GPU but Boltz-2 fails with `CUDA-capable device(s) is/are
busy or unavailable`, check the GPU recovery state:

```bash
nvidia-smi -q | grep -E "GPU Recovery Action|GPU Fabric GUID|Compute Mode"
```

When `GPU Recovery Action` reports `Reset`, reset the GPU or reboot the node
before launching NIMs. On a dedicated test node, the reset path is:

```bash
sudo systemctl stop nvidia-dcgm nvidia-persistenced
sudo nvidia-smi --gpu-reset -i 0
sudo systemctl start nvidia-persistenced nvidia-dcgm
```

If Apptainer/Singularity cannot isolate GPUs, run one Boltz-2 endpoint per allocation or ask the cluster administrators about NVIDIA Container CLI support for Apptainer.
