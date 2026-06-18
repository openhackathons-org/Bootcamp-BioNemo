# AI-Powered-Drug-Discovery-Bootcamp

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Welcome to AI-Powered-Drug-Discovery-Bootcamp

NVIDIA BioNeMo is a computational drug discovery platform built for developers and data scientists. BioNeMo provides an open-source machine learning framework for building and training deep learning models, and a set of optimized and easy-to-use NIM microservices for deploying AI workflows at scale.  BioNeMo NIM microservices are constructed as containers with everything needed for efficient, portable deployment and easy API integration into enterprise-grade AI applications.

This bootcamp combines comprehensive tutorials with a cutting-edge hackathon challenge, giving you hands-on experience with NVIDIA BioNeMo's tools for AI-enabled drug discovery. You'll learn to harness the power of MolMIM, a generative model for small-molecule drug discovery.  MolMIM is a probabilistic auto-encoder trained with Mutual Information Machine (MIM) learning designed to learn an informative and clustered latent space.  By leveraging targeted optimization with scoring oracles, you will learn how to explore MolMIM's dense latent-space representation of chemical space to generate and optimize novel molecular structures.

## Repository Structure

At the top level, [`HOW_TO_GET_STARTED.md`](HOW_TO_GET_STARTED.md) provides the shortest implementation checklist. You'll also find detailed instructions for deploying or configuring the MolMIM and Boltz-2 NIM endpoints in [`deployment.md`](deployment.md), an HPC-focused Apptainer/Singularity workflow in [`singularity.md`](singularity.md), and all required dependencies in [`deployment-requirements.txt`](deployment-requirements.txt). Once the services are healthy, you can follow along with the Tutorials and Challenge. On ARM nodes (GB200/GB300), the wrapper defaults to a local pure-PyTorch MolMIM NIM (built and run via Docker or Apptainer/Singularity, exposing `/hidden` and `/decode`, enabling CMA-ES guided optimization), with hosted MolMIM as a fallback; pass `--molmim hosted` for generation-only hosted MolMIM. The wrapper also tries local Boltz-2 and falls back to hosted Boltz-2 when local prediction requests fail.

### 📚 Tutorials
The [`tutorials/`](tutorials/) folder contains everything you need to get started and background on the models and techniques used in the Challenge:
- **00_Container_Setup.ipynb**: Step-by-step deployment guide for setting up MolMIM NIM with NGC API key configuration
- **01_MolMIMGeneration.ipynb**: Fundamental MolMIM operations including unguided sampling and guided optimization using CMA-ES algorithm
- **02_ClusterMolMIMEmbeddings.ipynb**: Clustering molecular embeddings to identify molecular families and functional relationships
- **03_MolMIMInterpolation.ipynb**: Interpolating between molecules by manipulating hidden states to generate novel structures
- **04_MolMIMOracleControlledGeneration.ipynb**: Advanced controlled generation using custom oracle scoring functions with CMA-ES optimization
- **05_Suggested_Tools_for_Scoring_Oracles.ipynb**: Comprehensive guide to tools and resources for building custom scoring oracles
- **06_Boltz2_Validation.ipynb**: Binding affinity prediction validation using Boltz-2 with MSA for CDK inhibitor assessment

### 🏆 Challenge
The [`challenge/`](challenge/) folder contains the hackathon challenge where you'll apply your knowledge to solve real drug discovery problems:
- **01_Challenge_Overview.ipynb**: Complete hackathon challenge introduction with objectives, scoring methods, and evaluation criteria
- **02_The_Challenge-Designing_CDK4_Inhibitors.ipynb**: Detailed challenge specification for designing selective CDK4 inhibitors while avoiding CDK11 binding
- **03_Hands-On_CDK_Inhibitor_Design.ipynb**: End-to-end pipeline for CDK4 inhibitor design including generation, affinity prediction, and composite scoring

### 🧪 Mini Hands-On
The [`mini-hands-on/`](mini-hands-on/) folder contains a compact hands-on track integrated into this workshop. It uses the same MolMIM and Boltz-2 endpoint environment plus shared CDK oracle code as the main bootcamp, but presents the material as a shorter guided sequence:
- **00_Introduction.ipynb**: Mini-track entry point and notebook order
- **01_NIM_Setup.ipynb**: Apptainer/Singularity setup notes for AI-Powered-Drug-Discovery-Bootcamp
- **02_Overview-Designing_CDK4_Inhibitors.ipynb**: CDK4 inhibitor design background
- **03_Hands-On_CDK_Inhibitor_Design.ipynb**: End-to-end CDK inhibitor generation and scoring workflow
- **04_Optional_MolMIM_CMA-ES_Controlled_Generation.ipynb**: Focused MolMIM CMA-ES optimization exercise
- **05_Optional_Boltz2_validation.ipynb**: Boltz-2 validation on known actives and decoys


## Bootcamp Objectives

By the end of this workshop, participants will:

* Understand NVIDIA BioNeMo architecture and key functionalities
* Gain deep insights into MolMIM for generative small molecule design
* Develop hands-on skills to apply BioNeMo workflows using real-world, complex datasets
* Integrate advanced scoring and optimization methods to refine molecular designs


## Getting Started

1. **Setup Environment**: Follow the deployment guide in [`deployment.md`](deployment.md), or use [`singularity.md`](singularity.md) when Docker is not available on the cluster.
3. **Complete Tutorials**: Work through the set of introductory notebooks in the [`tutorials/`](tutorials/) folder
4. **Try the Mini Track**: Use [`mini-hands-on/`](mini-hands-on/) for a shorter guided path
5. **Take the Challenge**: Apply your skills using the examples in the [`challenge/`](challenge/) folder

## Quick Start

```bash
# Clone and navigate to the repository
git clone https://github.com/openhackathons-org/AI-Powered-Drug-Discovery-Bootcamp.git
cd AI-Powered-Drug-Discovery-Bootcamp

# Install dependencies and start the architecture-aware endpoint stack
export NGC_API_KEY=<PASTE_API_KEY_HERE>
scripts/bootstrap_bootcamp.sh --boltz2 1
source .openhackathon-nims.env

# Start with the overview Start_Here.ipynb notebook
jupyter-lab Start_Here.ipynb
```

`bootstrap_bootcamp.sh` creates a local `.venv`, installs dependencies, starts
the NIM services, waits for health checks, and writes `.openhackathon-nims.env`.
The service wrapper is architecture-aware: x86_64/amd64 tries local MolMIM plus
local Boltz-2 and falls back to hosted endpoints when local services do not pass
their checks; aarch64/arm64 defaults to the local-arm MolMIM NIM (a pure-PyTorch
service built and run via Docker or Apptainer/Singularity that exposes `/hidden`
and `/decode`, enabling CMA-ES guided optimization) with hosted MolMIM fallback,
and tries local Boltz-2 with a hosted Boltz-2 fallback. Boltz-2 readiness
includes a small prediction smoke test, because `/v1/health/ready` alone can be
true before worker GPUs are usable.

For GB200/GB300 ARM nodes, the local-arm MolMIM NIM is the auto-mode default and
builds/runs via either Docker or Apptainer/Singularity (enabling `/hidden`,
`/decode`, and CMA-ES guided optimization). On a Docker workstation, select the
Docker runtime:

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export OPENHACKATHON_CONTAINER_RUNTIME=docker
scripts/bootstrap_bootcamp.sh --boltz2 1
source .openhackathon-nims.env
```

The first launch builds a small pure-PyTorch MolMIM image from `molmim_arm/` and
downloads the `molmim_70m_24_3` weights, so allow extra time before the endpoint
reports ready. On Apptainer/Singularity HPC clusters the same
`scripts/bootstrap_bootcamp.sh --boltz2 1` builds a `.sif` from `molmim_arm/molmim_arm.def`
instead (see [`singularity.md`](singularity.md)). Pass `--molmim hosted` to use
hosted MolMIM (generation only). See [`deployment.md`](deployment.md) for details.

The service wrapper writes the actual MolMIM and Boltz-2 URLs to
`.openhackathon-nims.env`. Source that file in every shell before running
notebooks or scoring scripts, especially on shared nodes where port `8000` may
already be in use.

The hands-on CDK challenge notebook defaults to a workshop demo mode tuned for
one Boltz-2 endpoint. For a larger exploratory run, set
`OPENHACKATHON_DEMO_MODE=0` before launching Jupyter, or override individual
knobs such as `OPENHACKATHON_CMA_ITERATIONS`, `OPENHACKATHON_CMA_POPSIZE`, and
`OPENHACKATHON_TOP_K_FOR_BOLTZ2`.

## The Challenge

The core of this bootcamp culminates in an exciting challenge: **Accelerating Drug Discovery with NVIDIA MolMIM**. You'll harness cutting-edge AI to revolutionize drug discovery by generating and optimizing novel molecular structures with potential as therapeutic agents.

### What You'll Do:
- Generate diverse molecular structures using MolMIM
- Optimize molecules for drug-like characteristics using custom scoring oracles
- Evaluate drug potential through property prediction and binding assessment

### Key Properties to Explore:
- **Drug-likeness (QED)**: Overall suitability as a drug candidate
- **Synthesizability**: Ease of laboratory synthesis
- **Solubility**: Dissolution characteristics
- **Toxicity**: Safety assessment
- **Tanimoto Similarity**: Chemical similarity to known therapeutics

## Why Participate?

This bootcamp with challenge offers a unique opportunity to:

* Gain hands-on experience with state-of-the-art AI models for molecular design
* Deepen your understanding of computational drug discovery workflows
* Collaborate with fellow innovators and experts in the field
* Contribute to the advancement of medical science by accelerating the identification of new drugs

Prepare to innovate, experiment, and push the boundaries of what's possible in the quest for life-changing medicines!
