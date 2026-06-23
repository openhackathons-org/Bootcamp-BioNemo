# Optional Stream: De Novo Protein Binder Design

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


An **optional**, self-contained stream (independent of the MolMIM / CDK inhibitor track) that
designs a brand-new protein **binder** against a target by composing three NVIDIA BioNeMo NIMs:

**RFdiffusion** (backbones) → **ProteinMPNN** (sequences) → **Boltz-2** (co-fold + score).

The single notebook [`Protein_Binder_Design.ipynb`](Protein_Binder_Design.ipynb) walks through the
biology, the NIMs, the architecture, an interactive **Mol\*** view of the designed complexes, and
assessment plots, and finishes in **well under an hour** on a GB200 in demo mode.

Adapted in part from the NVIDIA BioNeMo
[agent-toolkit `protein-binder-design` skill](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/tree/main/workflows/generative_protein_binder_design)
(Apache-2.0 / CC-BY-4.0).

## Target

SARS-CoV-2 spike **RBD** (PDB `6M0J`, chain `E`), designing binders to the ACE2-binding interface
(hotspots `453, 455, 456, 486, 489, 493, 501`). See the notebook for the biology and therapeutic
relevance.

## Layout

```
protein-binder-design/
├── Protein_Binder_Design.ipynb   # the workflow notebook (start here)
├── requirements.txt              # extra deps: ipymolstar (Mol*), py3Dmol
├── binder_design/                # helper package
│   ├── nim_clients.py            # RFdiffusion / ProteinMPNN / Boltz-2 HTTP clients
│   ├── pdb_utils.py              # PDB + mmCIF parsing, chain/residue helpers
│   ├── metrics.py                # Kabsch CA-RMSD (self-consistency)
│   ├── manifest.py               # campaign manifest (scores / filter / rank / CSV)
│   ├── controls.py               # scrambled negative controls
│   └── viz.py                    # Mol* (ipymolstar) + py3Dmol + matplotlib
└── runs/                         # created at run time (manifest.json, candidates.csv, structures)
```

## Prerequisites

- Access to the **RFdiffusion**, **ProteinMPNN**, and **Boltz-2** NIMs — hosted at
  [build.nvidia.com](https://build.nvidia.com) or self-hosted via NGC.
- Python deps: `pip install -r protein-binder-design/requirements.txt` (the notebook also installs
  `ipymolstar` / `py3Dmol` on first run if missing).

## Configure the NIM endpoints

The notebook reads endpoints from the environment (defaults are the NVIDIA-hosted endpoints):

| Variable | Default |
|---|---|
| `RFDIFFUSION_URL` | `https://health.api.nvidia.com` |
| `PROTEINMPNN_URL` | `https://health.api.nvidia.com` |
| `BOLTZ2_URL` | `https://health.api.nvidia.com` |
| `NVIDIA_API_KEY` | (required for hosted) |

### Hosted

```bash
export NVIDIA_API_KEY=nvapi-...
jupyter-lab protein-binder-design/Protein_Binder_Design.ipynb
```

### Local on GB200 / GB300 (all three NIMs have arm64 images)

```bash
export NGC_API_KEY=nvapi-...
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
mkdir -p ~/nimcache_rfd ~/nimcache_pmpnn && chmod 777 ~/nimcache_*

docker run -d --name rfdiffusion --gpus device=0 --shm-size=4g -e NGC_API_KEY \
  -v ~/nimcache_rfd:/opt/nim/.cache -p 127.0.0.1:8101:8000 nvcr.io/nim/ipd/rfdiffusion:latest
docker run -d --name proteinmpnn --gpus device=0 --shm-size=4g -e NGC_API_KEY \
  -v ~/nimcache_pmpnn:/opt/nim/.cache -p 127.0.0.1:8102:8000 nvcr.io/nim/ipd/proteinmpnn:latest
# Boltz-2: reuse the bootcamp's local Boltz-2 (scripts/openhackathon_services.sh) or run it here.

export RFDIFFUSION_URL=http://localhost:8101
export PROTEINMPNN_URL=http://localhost:8102
export BOLTZ2_URL=http://localhost:8000     # the bootcamp's local Boltz-2 NIM
jupyter-lab protein-binder-design/Protein_Binder_Design.ipynb
```

Local NIMs serve `/biology/ipd/rfdiffusion/generate`, `/biology/ipd/proteinmpnn/predict`, and
`/biology/mit/boltz2/predict` (no `/v1`); the hosted endpoints use `/v1/biology/...`. The clients
resolve both automatically from the base URL.

## Runtime budget

`OPENHACKATHON_DEMO_MODE=1` (default) keeps the campaign small (a dozen backbones, a short co-fold
shortlist) so it finishes in **< 1 hour**. Tune with `PBD_N_BACKBONES`, `PBD_SEQS_PER_BACKBONE`,
`PBD_N_COFOLD`, `PBD_DIFFUSION_STEPS`, `PBD_COFOLD_STEPS`, or set `OPENHACKATHON_DEMO_MODE=0`.

## Outputs

Each run writes `runs/<target>_<timestamp>/` with `manifest.json`, `candidates.csv`, the
RFdiffusion `backbones/`, and the Boltz-2 `complexes/` (mmCIF). Designs are ranked by interface
confidence and compared against scrambled negative controls (report a **success rate**, not just
top scores).

## Responsible use

De novo binder design is dual-use. Use it only for legitimate research and therapeutic intent.
