# BioNeMo NIM Deployment Guide

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


This guide provides instructions for deploying or configuring the NVIDIA MolMIM
and Boltz-2 NIM services used by the BioNeMo bootcamp tutorials and challenge.

## Prerequisites

Before starting, ensure you have:
- NVIDIA GPU allocation on a compute node
- NGC API key from [NVIDIA NGC](https://org.ngc.nvidia.com/setup/api-key)
- Apptainer/Singularity for HPC clusters, or Docker for local workstations

## Getting Started

1. Start or configure a MolMIM NIM endpoint for molecule generation
2. Start one or more Boltz-2 NIM endpoints for affinity prediction
3. Verify service health before launching notebooks
4. Work through the tutorials and challenge


## Setup Instructions

Please refer to the [NVIDIA MolMIM NIM docs](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html) and [QuickStart guide](https://docs.nvidia.com/nim/bionemo/molmim/latest/quickstart-guide.html) for comprehensive information. Additional examples showcasing MolMIM capabilities like clustering molecules and interpolating between molecular structures are available in the [endpoint documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html#notebooks).

### Option A: Apptainer/Singularity on HPC

Use this path on clusters where Docker and the NVIDIA Container Runtime are not
available. The scripts pull the NIM images into local `.sif` files and run them
with `--nv`.

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}

# Install Python dependencies and start the architecture-aware endpoint stack
scripts/bootstrap_bootcamp.sh --boltz2 1
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

See [`singularity.md`](singularity.md) for the full cluster workflow, cache
notes, image overrides, generated endpoint environment, and multi-endpoint
evaluation commands.

### Option B: Docker on Local GPU Workstations and GB200/GB300 ARM Systems

Use the same service wrapper on Docker-based systems. On GB200, GB300, and
other ARM64 hosts, set `OPENHACKATHON_CONTAINER_RUNTIME=docker`; the Docker
launcher selects `linux/arm64` automatically when the host reports `aarch64`
and checks that each NIM image advertises that platform before pulling. The
check uses `docker buildx imagetools inspect` when available and falls back to
`docker manifest inspect` on minimal Docker installations.

`nvcr.io/nim/mit/boltz2:1.7.0` is the default local Boltz-2 image for current
testing and advertises current local deployment support. On ARM hosts with a
pre-590 driver, the Docker launcher defaults Boltz-2 to
`nvcr.io/nim/mit/boltz2:1.4.0`. Set `BOLTZ2_IMAGE` to override this selection
for a specific validation environment. The service wrapper now verifies local
Boltz-2 with a small prediction request after `/v1/health/ready`; if that smoke
test fails in `auto` mode, `.openhackathon-nims.env` is written with the hosted
Boltz-2 fallback URL instead of a local endpoint that may hang.

The MolMIM `nvcr.io/nim/nvidia/molmim:1.0.0` image currently resolves as
`linux/amd64`, so it is not a native ARM container for these nodes. There are
two ways to get MolMIM on ARM:

1. **Local MolMIM on ARM via a pure-PyTorch NIM-like service (`--molmim local-arm`, the default on ARM with a container runtime).**
   The bootcamp ships a pure-PyTorch MolMIM (loading the official
   `molmim_70m_24_3` weights) wrapped in a FastAPI service that implements the
   MolMIM NIM REST contract — `/hidden`, `/decode`, `/sampling`, `/generate` —
   and runs natively on aarch64 + Blackwell with no NeMo/Megatron/apex/TE. It
   builds and runs via **either Docker or Apptainer/Singularity**, so on ARM
   hosts with a container runtime this is the auto-mode default and enables CMA-ES
   guided optimization out of the box where the amd64-only NIM cannot run. See
   [ARM-Local MolMIM (pure-PyTorch NIM-like service)](#arm-local-molmim-pure-pytorch-nim-like-service)
   below.

2. **Hosted MolMIM (fallback / generation-only).** Use a NVIDIA-hosted MolMIM
   endpoint or another x86-hosted MolMIM endpoint and set `MOLMIM_URL` to that
   service (or pass `--molmim hosted`). The wrapper also uses this automatically
   if the local-arm build/health check fails, and it is the auto-mode default if
   no container runtime is available. If the endpoint requires authentication,
   set `MOLMIM_API_KEY` or `NVIDIA_API_KEY`; the notebooks and shared client add
   the bearer token automatically. For NVIDIA-hosted MolMIM, use
   `https://health.api.nvidia.com/v1/biology/nvidia/molmim` as the base URL.
   The hosted endpoint supports molecule generation only — it does not expose the
   latent-space `/hidden` and `/decode` endpoints that CMA-ES guided
   optimization needs. Override `MOLMIM_IMAGE` only if NVIDIA publishes an ARM64
   MolMIM NIM tag.

For NVIDIA-hosted Boltz-2 fallback, the default endpoint root is
`https://health.api.nvidia.com/v1/biology/mit/boltz2`. Override it with
`OPENHACKATHON_HOSTED_BOLTZ2_URL` or `scripts/openhackathon_services.sh start
--boltz2-url ...`.

The user running the wrapper must be able to access the Docker daemon. For
interactive testing, `docker ps` should succeed without `sudo`. For unattended
bootcamp launches, avoid relying on an interactive sudo password prompt.
The Docker launcher relaxes permissions on the selected NIM cache/workspace
directories by default because NIM containers can run as a UID that differs from
the host user. To disable that behavior, set
`OPENHACKATHON_RELAX_CACHE_PERMISSIONS=0` and pre-create writable cache paths.

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export OPENHACKATHON_CONTAINER_RUNTIME=docker
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}

scripts/bootstrap_bootcamp.sh --boltz2 1
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

`scripts/openhackathon_services.sh start --boltz2 1` can still be used when
Python dependencies are already installed. Its default `--molmim auto` mode
uses local MolMIM on x86_64/amd64 with hosted fallback; on aarch64/arm64 with a
container runtime (Docker or Apptainer/Singularity) it defaults to the local-arm
MolMIM NIM (latent `/hidden`, `/decode`, and CMA-ES) with hosted fallback, and
falls back to hosted MolMIM only when no container runtime is available. Use
`--molmim local-arm` to force the local ARM build, or `--molmim hosted` to force
hosted (see
[ARM-Local MolMIM (pure-PyTorch NIM-like service)](#arm-local-molmim-pure-pytorch-nim-like-service)).
Its default `--boltz2-mode auto` mode starts local Boltz-2, then falls back to
hosted Boltz-2 if the prediction smoke test fails.

If you need to force a platform or image tag for validation, set:

```bash
export DOCKER_PLATFORM=linux/arm64
export BOLTZ2_IMAGE=nvcr.io/nim/mit/boltz2:1.7.0
```

On single-GPU ARM nodes, the default Docker GPU mode exposes that GPU with
`--gpus device=0`. On multi-GPU ARM nodes, the service wrapper starts one
Boltz-2 endpoint per requested GPU. If a site image expects the all-GPU path,
set:

```bash
export OPENHACKATHON_DOCKER_GPU_MODE=all
```

Some Docker hosts also define a runtime named `nvidia`, while others rely on
the `--gpus` integration without that runtime name. If `docker run
--runtime=nvidia ...` works on your host and you want to match NVIDIA's
documentation exactly, set `OPENHACKATHON_DOCKER_USE_NVIDIA_RUNTIME=1`.

Manual Docker starts are still useful for debugging.

First login to the `nvcr.io` Docker registry with your NGC API key:

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export NGC_CLI_API_KEY=$NGC_API_KEY
docker login nvcr.io
```

Start MolMIM:

```bash
docker run --rm -it --name molmim --gpus device=0 \
     -e CUDA_VISIBLE_DEVICES=0 \
     -e NGC_CLI_API_KEY \
     -p 8001:8000 \
     nvcr.io/nim/nvidia/molmim:1.0.0
```

Start Boltz-2:

```bash
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 "$LOCAL_NIM_CACHE"

docker run --rm -it --name boltz2 --gpus device=0 \
     --shm-size=16G \
     -e NGC_API_KEY \
     -e NIM_LOG=INFO \
     -e NIM_LOG_LEVEL=INFO \
     -e TLLM_LOG_LEVEL=INFO \
     -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
     -p 8000:8000 \
     nvcr.io/nim/mit/boltz2:1.7.0
```

### ARM-Local MolMIM (pure-PyTorch NIM-like service)

On GB200/GB300 (aarch64 + Blackwell) there is **no** MolMIM that runs as a
prebuilt container: the MolMIM NIM is amd64-only, the arm64 BioNeMo Framework
images (`2.x-arm`) do not contain MolMIM, and the only MolMIM-bearing framework
(v1) is pinned to a pre-Blackwell CUDA 12.3 stack that won't execute on these
GPUs.

To make MolMIM work anyway, the bootcamp ships `molmim_arm/`, a small
**pure-PyTorch reimplementation** (`molmim_torch.py`) that loads the official
`molmim_70m_24_3` checkpoint and runs entirely on stock PyTorch (no NeMo /
Megatron / apex / TransformerEngine), so it runs natively on aarch64 + Blackwell.
It is wrapped in a **FastAPI service** (`server.py`) that mirrors the MolMIM NIM
REST surface — `/hidden`, `/decode`, `/sampling`, `/generate`,
`/v1/health/ready`, `/v1/metadata`, and OpenAPI docs at `/docs` — so the bootcamp
client and CMA-ES guided optimization run unmodified against `MOLMIM_URL`.

Validated on a GB200: 9/10 exact drug-like reconstruction and 100% valid,
on-target latent sampling.

It is the auto-mode default on ARM with a container runtime, and you can also
select it explicitly. It builds and runs via either Docker or
Apptainer/Singularity:

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
# Docker workstation:
export OPENHACKATHON_CONTAINER_RUNTIME=docker
# or, on an Apptainer/Singularity HPC cluster:
# export OPENHACKATHON_CONTAINER_RUNTIME=apptainer

scripts/openhackathon_services.sh start --molmim local-arm --boltz2 1
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

The first launch builds the image (Docker: `molmim_arm/Dockerfile`; Apptainer:
a `.sif` from `molmim_arm/molmim_arm.def`) on a stock NGC PyTorch base
(`nvcr.io/nvidia/pytorch:25.01-py3`, which provides arm64 + Blackwell), and
downloads the `molmim_70m_24_3` weights from NGC (`nvidia/clara/molmim:1.3`)
into the weights cache, so allow extra time before the endpoint reports ready
(`OPENHACKATHON_MOLMIM_READY_TIMEOUT` defaults to 2400s in this mode). The
weights are mounted at `/models` at runtime (not baked into the image). The
generated `.openhackathon-nims.env` sets `OPENHACKATHON_ACTIVE_MOLMIM_MODE=local`
and `OPENHACKATHON_USE_CMA=1`, which re-enables the guided-optimization notebook
section. If the local ARM MolMIM does not become healthy, the wrapper falls back
to the hosted MolMIM endpoint (generation only).

Relevant overrides:

```bash
export BIONEMO_PYTORCH_BASE=nvcr.io/nvidia/pytorch:25.01-py3  # arm64 + Blackwell base
export MOLMIM_ARM_IMAGE=molmim-arm-nim:local                 # Docker built tag
export MOLMIM_ARM_SIF=$PWD/.sif/molmim-arm-nim.sif           # Apptainer .sif path
export SIF_DIR=$PWD/.sif                                     # Apptainer .sif directory
export MOLMIM_WEIGHTS_DIR=$HOME/.cache/nim/molmim            # holds molmim_70m_24_3.nemo
export MOLMIM_NGC_MODEL=nvidia/clara/molmim:1.3              # weights source
export MOLMIM_ARM_REBUILD=1                                  # force a rebuild
export MOLMIM_SAMPLING_METHOD=greedy                         # greedy|topkp for /sampling
export APPTAINER_GPU_MODE=auto                               # auto|nvccli|nv (Apptainer GPU)
```

You can also run it directly for debugging, then browse `http://localhost:8001/docs`:

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
scripts/run_molmim_arm.sh molmim 8001 0
```

### ARM CUDA Readiness Check

If `nvidia-smi` works but a Docker CUDA workload or Boltz-2 startup fails with
`CUDA-capable device(s) is/are busy or unavailable`, inspect the host recovery
state:

```bash
nvidia-smi -q | grep -E "GPU Recovery Action|GPU Fabric GUID|Compute Mode"
```

When `GPU Recovery Action` is `Reset`, the GPU is visible but not ready for CUDA
kernel launches. Reset the GPU on a dedicated node:

```bash
sudo systemctl stop nvidia-dcgm nvidia-persistenced
sudo nvidia-smi --gpu-reset -i 0
sudo systemctl start nvidia-persistenced nvidia-dcgm
```

If the reset is rejected or the state returns to `Reset`, reboot through the BMC
or scheduler before starting the NIMs.

### Python Environment

Clone this repository; optionally set up a python virtual environment, and install dependencies:

```bash
git clone https://github.com/openhackathons-org/AI-Powered-Drug-Discovery-Bootcamp.git
cd AI-Powered-Drug-Discovery-Bootcamp
python3 -m venv venv
source venv/bin/activate
pip install -r deployment-requirements.txt
```

### Launch Tutorials

After the NIM services are running and dependencies are installed, start Jupyter
Lab to explore the tutorials and challenge:

```bash
source .openhackathon-nims.env
jupyter-lab
```

## Structure

- **Tutorials - Container Setup**: [`tutorials/00_Container_Setup.ipynb`](tutorials/00_Container_Setup.ipynb) - Detailed container deployment guide
- **Tutorials - Lab 1**: Basic MolMIM operations (clustering, generation, interpolation)
- **Tutorials - Lab 2**: Advanced techniques with custom oracles and optimization
- **Challenge**: Apply your knowledge to solve drug discovery problems


## Support

For technical issues or questions:
- Check the [NVIDIA MolMIM documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html)
- Review the tutorials for step-by-step guidance
- Consult the challenge folder for specific hackathon requirements

## License

Please see [LICENSE.txt](LICENSE.txt) for licensing information.
