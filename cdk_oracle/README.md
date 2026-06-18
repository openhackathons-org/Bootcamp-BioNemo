# CDK Oracle - Modular CDK Inhibitor Design Package

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


A modular Python package for designing selective CDK4/6 inhibitors while avoiding CDK11 binding. This package integrates NVIDIA NIMs (MolMIM for molecule generation, Boltz2 for affinity prediction) with physicochemical property calculations and multi-objective scoring.

## Package Structure

```
cdk_oracle/
├── __init__.py          # Package exports
├── config.py            # Central configuration
├── nim_client.py        # NIM service clients (MolMIM, Boltz2)
├── physicochemical.py   # Drug-likeness calculations
├── scoring.py           # Composite scoring system
├── visualization.py     # Plots and reports
├── pipeline.py          # End-to-end workflow orchestration
└── README.md            # This file
```

---

## Module Reference

### 1. `config.py` - Configuration

**Purpose:** Centralizes all configurable parameters for the entire pipeline.

**Key Class:** `CDKConfig`

#### Customizable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| **NIM URLs** | | | |
| `molmim_url` | Line 68 | `http://localhost:8001` | MolMIM service endpoint |
| `boltz2_url` | Line 69 | `http://localhost:8000` | Primary Boltz2 endpoint |
| `boltz2_endpoints` | Line 74 | Auto-parsed from env | Multiple Boltz2 endpoints for parallel predictions |
| **API Keys** | | | |
| `nvidia_api_key` | Line 77 | From `NVIDIA_API_KEY` env | NVIDIA API key |
| `molmim_api_key` | Line 78 | From `MOLMIM_API_KEY` or `NVIDIA_API_KEY` env | MolMIM hosted endpoint authentication |
| `boltz2_api_key` | Line 79 | From `BOLTZ2_API_KEY` or `NVIDIA_API_KEY` env | Boltz2 authentication |
| **Target Proteins** | | | |
| `on_target` | Line 86 | `"CDK4"` | Primary target protein |
| `anti_target` | Line 87 | `"CDK11"` | Off-target to avoid |
| `binding_sites` | Line 90-93 | Residue indices | Binding pocket residues |
| **MolMIM Parameters** | | | |
| `molmim_latent_dims` | Line 96 | `512` | Latent space dimensions |
| `cma_popsize` | Line 97 | `20` | CMA-ES population size |
| `cma_sigma` | Line 98 | `1.0` | CMA-ES step size |
| `cma_iterations` | Line 99 | `10` | Number of optimization iterations |
| **Scoring Weights** | | | |
| `weights` | Line 102-110 | See below | Multi-objective scoring weights |
| **Thresholds** | | | |
| `ic50_potent_threshold` | Line 113 | `100.0` nM | "Potent" IC50 cutoff |
| `novelty_similarity_cutoff` | Line 127 | `0.85` | Tanimoto cutoff for novelty |
| **Output Settings** | | | |
| `save_boltz2_structures` | Line 133 | `True` | Save Boltz2 predicted complex structures (CIF) |
| `top_n_compounds` | Line 130 | `25` | Number of top compounds to output |

#### Default Scoring Weights

```python
weights = {
    "binding_affinity": 0.25,   # CDK4 binding potency
    "selectivity": 0.20,        # CDK11/CDK4 ratio
    "cdk11_avoidance": 0.15,    # Penalty for CDK11 binding
    "qed": 0.15,                # Drug-likeness
    "sa": 0.10,                 # Synthetic accessibility
    "pains": 0.10,              # PAINS filter (penalty)
    "novelty": 0.05             # Structural novelty
}
```

#### How to Customize

**Option 1: Environment Variables**
```bash
export MOLMIM_URL="http://my-molmim-server:8001"
export MOLMIM_API_KEY="your-hosted-molmim-key"
export BOLTZ2_URL="http://my-boltz2-server:8000"
export BOLTZ2_ENDPOINTS="http://gpu1:8000,http://gpu2:8000"
export NVIDIA_API_KEY="your-api-key"
```

For NVIDIA-hosted fallback, use `https://health.api.nvidia.com/v1/biology/nvidia/molmim`
for MolMIM and `https://health.api.nvidia.com/v1/biology/mit/boltz2` for
Boltz-2. Hosted MolMIM supports generation but not latent `/hidden` or
`/decode`, so the pipeline automatically uses sampling instead of CMA-ES in
that mode. On ARM (GB200/GB300), point `MOLMIM_URL` at the local `--molmim
local-arm` NIM (`http://localhost:8001`, a pure-PyTorch MolMIM that exposes
`/hidden` and `/decode`) to run full CMA-ES optimization.

**Option 2: Constructor Arguments**
```python
from cdk_oracle import CDKConfig

config = CDKConfig(
    molmim_url="http://custom-server:8001",
    cma_iterations=20,
    cma_popsize=30,
    weights={
        "binding_affinity": 0.30,
        "selectivity": 0.25,
        # ... customize weights
    }
)
```

**Option 3: Direct File Edit**
Edit lines 102-110 in `config.py` to change default scoring weights.

---

### 2. `nim_client.py` - NIM Service Clients

**Purpose:** Handles all communication with NVIDIA NIMs (MolMIM and Boltz2).

**Key Classes:**

#### `NIMHealthChecker`
- **Purpose:** Check service availability
- **Key Methods:**
  - `check_molmim()` - Check MolMIM health
  - `check_boltz2()` - Check Boltz2 health
  - `print_status()` - Print formatted status

#### `MolMIMClient`
- **Purpose:** Molecule encoding/decoding/sampling
- **Key Methods:**
  - `encode(smiles)` - Encode SMILES to latent space (Line 114)
  - `decode(latent)` - Decode latent vectors to SMILES (Line 134)
  - `sample(smiles, num_samples)` - Sample similar molecules (Line 163)

**Customizable:** Modify timeout values on lines 127, 158, 175 (default: 60s)

#### `Boltz2AffinityClient`
- **Purpose:** Binding affinity prediction with multi-endpoint parallel support
- **Key Methods:**
  - `predict_affinity(smiles, protein)` - Single prediction (Line 357)
  - `predict_batch(smiles_list, proteins)` - Batch prediction (Line 375)
  - `predict_batch_parallel(...)` - Parallel multi-endpoint prediction (Line 426)
  - `check_endpoints_health()` - Health check all endpoints (Line 531)

**Customizable:**

| Parameter | Location | Description |
|-----------|----------|-------------|
| `max_concurrent` | Line 382, 432 | Max parallel requests (default: `num_endpoints * 2`) |
| MSA usage | `use_msa` parameter | Enable/disable MSA for predictions |

**How to use multiple endpoints:**
```python
from cdk_oracle import Boltz2AffinityClient, CDKConfig

endpoints = [
    {"url": "http://gpu1:8000", "api_key": "key1"},
    {"url": "http://gpu2:8000", "api_key": "key2"},
]

client = Boltz2AffinityClient(CDKConfig(), endpoints=endpoints)
results = client.predict_batch(smiles_list, parallel=True)
```

---

### 3. `physicochemical.py` - Physicochemical Properties

**Purpose:** Calculate drug-likeness metrics, filters, and novelty scores.

**Key Classes/Functions:**

#### `PhysicochemCalculator`
- **Purpose:** Calculate molecular properties
- **Key Methods:**
  - `calculate_properties(smiles)` - All properties for one molecule (Line 56)
  - `calculate_batch(smiles_list)` - Batch calculation (Line 150)
  - `passes_filters(smiles)` - Check filter compliance (Line 170)

**Properties Calculated:**
- Molecular Weight (MW)
- LogP
- H-bond Donors/Acceptors (HBD, HBA)
- TPSA
- Rotatable Bonds
- Ring Count / Aromatic Rings
- QED (Quantitative Estimate of Drug-likeness)
- SA Score (Synthetic Accessibility, 1-10, lower=easier)
- Lipinski Violations
- PAINS Alerts
- Murcko Scaffold

**Customizable Thresholds (in `config.py`):**
```python
mw_max: float = 500.0          # Line 119
logp_max: float = 5.0          # Line 120
hbd_max: int = 5               # Line 121
hba_max: int = 10              # Line 122
tpsa_max: float = 140.0        # Line 123
rotatable_bonds_max: int = 10  # Line 124
```

#### Novelty Functions

**`calculate_novelty_score_chembl(smiles, cutoff, chembl_path, additional_refs)`** (Line 343)

Calculates novelty by comparing to ChEMBL database using Tanimoto similarity.

**Algorithm:**
1. Generate Morgan fingerprint (radius=2, 2048 bits) for query
2. Compare to all ChEMBL fingerprints
3. Find maximum similarity
4. If `max_sim >= cutoff`: not novel (score = 0)
5. If `max_sim < cutoff`: novel (score = `1 - max_sim/cutoff`)

**Customizable:**
- `cutoff` parameter (default: 0.85) - adjust novelty threshold
- `additional_refs` - add extra reference molecules (e.g., seed molecules)

**How to modify:**
```python
# More strict novelty (require more dissimilar)
novelty, max_sim, is_novel = calculate_novelty_score_chembl(
    smiles, cutoff=0.70  # Stricter threshold
)

# Include generated compounds as references
novelty, max_sim, is_novel = calculate_novelty_score_chembl(
    smiles, 
    cutoff=0.85,
    additional_refs=previously_generated_smiles
)
```

---

### 4. `scoring.py` - Composite Scoring

**Purpose:** Combine multiple objectives into a single score for compound ranking.

**Key Class:** `CDKScorer`

#### Score Components

| Component | Score Name | Range | Higher = Better |
|-----------|------------|-------|-----------------|
| CDK4 Binding | `binding_score` | 0-1 | Yes (lower IC50) |
| CDK11/CDK4 Selectivity | `selectivity_score` | 0-1 | Yes (higher ratio) |
| CDK11 Avoidance | `avoidance_score` | 0-1 | Yes (higher CDK11 IC50) |
| QED | `qed_score` | 0-1 | Yes |
| Synthetic Accessibility | `sa_score_norm` | 0-1 | Yes (lower SA) |
| PAINS | `pains_score` | 0 or 1 | Yes (no alerts) |
| Novelty | `novelty_score_norm` | 0-1 | Yes |

#### Scoring Methods (Lines to Customize)

**Binding Score Normalization** (Line 66-77):
```python
def _normalize_ic50_to_score(self, ic50_nm, best=1.0, worst=10000.0):
    # Log-scale normalization
    # Modify best/worst to change sensitivity
```

**Selectivity Score** (Line 79-96):
```python
def _normalize_selectivity_to_score(self, ratio):
    # Thresholds from config:
    # - excellent_threshold: 100x (score = 1.0)
    # - good_threshold: 10x (score >= 0.7)
```

**Avoidance Score** (Line 98-111):
```python
def _normalize_avoidance_to_score(self, cdk11_ic50):
    # Higher CDK11 IC50 = better avoidance
    # >= 10000 nM: score = 1.0
    # >= 1000 nM: score = 0.8+
    # >= 100 nM: score = 0.5+
```

#### How to Customize Scoring

**Change scoring weights:**
```python
config = CDKConfig(
    weights={
        "binding_affinity": 0.35,  # Emphasize potency
        "selectivity": 0.30,       # Emphasize selectivity
        "cdk11_avoidance": 0.10,
        "qed": 0.10,
        "sa": 0.05,
        "pains": 0.05,
        "novelty": 0.05
    }
)
```

**Modify scoring functions directly:**
Edit `_normalize_*_to_score()` methods in `scoring.py` (Lines 66-125).

---

### 5. `visualization.py` - Plots and Reports

**Purpose:** Generate publication-quality visualizations and HTML reports.

**Key Class:** `CDKVisualizer`

#### Available Plots

| Method | Description | Output |
|--------|-------------|--------|
| `plot_affinity_scatter()` | CDK4 vs CDK11 IC50 scatter | Shows selectivity regions |
| `plot_score_distribution()` | Histograms of all score components | 8-panel figure |
| `plot_top_compounds()` | Bar charts of top compounds | Score and IC50 comparison |
| `plot_optimization_progress()` | Training curves | Score, validity, diversity vs iteration |
| `generate_report()` | Full HTML report | All plots + tables |

#### Customizable

**Color Scheme (Line 26-33):**
```python
self.colors = {
    "cdk4": "#27ae60",      # Green - modify for CDK4 color
    "cdk11": "#9b59b6",     # Purple - modify for CDK11 color
    "good": "#2ecc71",      # Bright green
    "warning": "#f39c12",   # Orange
    "bad": "#e74c3c",       # Red
    "neutral": "#3498db",   # Blue
}
```

**Plot Style (Line 25):**
```python
plt.style.use('seaborn-v0_8-whitegrid')  # Change matplotlib style
```

**Selectivity Region Thresholds** (Line 66-70 in `plot_affinity_scatter`):
```python
# Excellent selectivity (>100x)
ax.fill_between(x_range, x_range * 100, ...)
# Good selectivity (>10x)
ax.fill_between(x_range, x_range * 10, x_range * 100, ...)
```

---

### 6. `pipeline.py` - Workflow Orchestration

**Purpose:** End-to-end pipeline combining all modules.

**Key Class:** `CDKDesignPipeline`

#### Optimization Loop (Line 154-472)

The `generate_molecules()` method implements the CDK-aware CMA-ES optimization:

```
For each iteration:
1. Generate `popsize` candidates from CMA-ES latent space
2. Decode to SMILES via MolMIM
3. Fast filter: validity, QED, SA, PAINS, novelty (all compounds)
4. Select top `top_k_for_boltz2` candidates
5. Run Boltz2 predictions (PARALLEL across endpoints)
6. Compute full composite score
7. Feed scores back to CMA-ES (minimization)
8. Track diversity and history
```

#### Key Parameters to Customize

| Parameter | Method Argument | Config Default | Description |
|-----------|-----------------|----------------|-------------|
| Iterations | `num_iterations` | `cma_iterations=10` | Number of CMA-ES iterations |
| Population | `popsize` | `cma_popsize=20` | Molecules per iteration |
| Top-K for Boltz2 | `top_k_for_boltz2` | `popsize//2` | How many get expensive affinity predictions |
| Use MSA | `use_msa` | `True` | Enable MSA for better Boltz2 predictions |
| Use CMA-ES | `use_cma` | `True` | Use optimization (False = random sampling) |

#### Molecular Diversity Calculation (Line 112-152)

**What it measures:** Average pairwise Tanimoto distance within the population.

**Formula:** `Diversity = 1 - average_pairwise_similarity`

- **1.0** = maximally diverse (all molecules very different)
- **0.0** = no diversity (all molecules identical)

**How to interpret:**
- Diversity should stay moderate (0.3-0.7) during optimization
- Very low diversity = stuck in local optimum
- Very high diversity = exploration, not exploitation

#### Caching Mechanism (Line 490-549)

The `_get_affinities_with_cache()` method reuses Boltz2 predictions from the optimization loop:
- Compounds scored during optimization are cached
- Final evaluation skips redundant Boltz2 calls
- Only cached compounds appear in final results

**To force fresh predictions:** Set `skip_generation=True` and use `evaluate_existing()`.

---

## Usage Examples

### Basic Usage

```python
from cdk_oracle import CDKDesignPipeline, CDKConfig

# Create pipeline with defaults
pipeline = CDKDesignPipeline()

# Check services
pipeline.check_services()

# Run optimization
results = pipeline.run(
    seed_smiles=["CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)F"],
    num_iterations=10,
    popsize=20
)

# Access results
print(results.summary)
top_compounds = results.get_top_compounds(10)
```

### Custom Configuration

```python
# Custom scoring weights emphasizing selectivity
config = CDKConfig(
    cma_iterations=20,
    cma_popsize=30,
    weights={
        "binding_affinity": 0.20,
        "selectivity": 0.35,       # Increased weight
        "cdk11_avoidance": 0.20,   # Increased weight
        "qed": 0.10,
        "sa": 0.05,
        "pains": 0.05,
        "novelty": 0.05
    }
)

pipeline = CDKDesignPipeline(config=config)
```

### Multi-Endpoint Parallel Predictions

```python
# Use multiple Boltz2 servers
boltz2_endpoints = [
    {"url": "http://gpu1:8000"},
    {"url": "http://gpu2:8000"},
    {"url": "http://gpu3:8000"},
]

pipeline = CDKDesignPipeline(
    boltz2_endpoints=boltz2_endpoints
)

# Predictions will be distributed across all endpoints
results = pipeline.run(seed_smiles=[...], num_iterations=10)
```

### Evaluate Existing Compounds

```python
# Skip generation, just score provided compounds
results = pipeline.evaluate_existing(
    smiles_list=["CCO", "CCN", "CCCO"],
    use_msa=True
)
```

---

## Output Files

When `pipeline.run()` completes, it saves to a timestamped folder:

```
output/run_20260115_143052/
├── all_compounds_scores.csv    # All scored compounds
├── top_compounds.csv           # Top 25 by score
├── run_summary.json            # Summary statistics
├── generated_smiles.txt        # All generated SMILES
├── seed_molecules.txt          # Input seed molecules
├── design_report.html          # HTML report with plots
├── affinity_scatter.png        # CDK4 vs CDK11 plot
├── score_distribution.png      # Score histograms
├── top_compounds.png           # Top compounds bar chart
└── boltz2_structures/          # Predicted complex structures (CIF files)
    ├── seed0_iter0/            # Structures from seed 0, iteration 0
    │   ├── compound1_CDK4_struct0.cif
    │   ├── compound1_CDK11_struct0.cif
    │   └── ...
    ├── seed0_iter1/            # Structures from seed 0, iteration 1
    └── ...
```

### Structure Files

When `save_boltz2_structures=True` (default), the pipeline saves Boltz2-predicted 
protein-ligand complex structures in mmCIF format:

- **Organization:** Structures are organized by seed molecule and iteration
- **Naming:** `{compound_smiles}_{protein}_{structure_idx}.cif`
- **Usage:** Can be visualized in PyMOL, ChimeraX, or other molecular viewers

To disable structure saving:
```python
results = pipeline.run(seed_smiles=[...], save_structures=False)
```

Or in config:
```python
config = CDKConfig(save_boltz2_structures=False)
```

---

## Common Customizations

### 1. Change Target Proteins

Edit `config.py` lines 86-93:
```python
on_target: str = "CDK6"         # Change primary target
anti_target: str = "CDK2"       # Change off-target
binding_sites: Dict = {
    "CDK6": [35, 71, ...],      # Update residue indices
    "CDK2": [41, 87, ...],
}
```

### 2. Adjust Optimization Aggressiveness

For **more exploration** (diverse molecules):
```python
config = CDKConfig(
    cma_sigma=2.0,          # Larger step size
    cma_popsize=50,         # Larger population
    cma_iterations=5,       # Fewer iterations
)
```

For **more exploitation** (refine best molecules):
```python
config = CDKConfig(
    cma_sigma=0.5,          # Smaller step size
    cma_popsize=10,         # Smaller population
    cma_iterations=30,      # More iterations
)
```

### 3. Stricter Novelty Requirements

In `config.py` line 127:
```python
novelty_similarity_cutoff: float = 0.70  # Stricter (default 0.85)
```

### 4. Custom Fast Filtering

Edit `pipeline.py` lines 280-288 to change fast score weights:
```python
fast_score = (
    0.40 * qed_score +      # Increase QED importance
    0.25 * sa_score +       # Increase SA importance
    0.15 * pains_score +
    0.10 * lipinski_score +
    0.10 * novelty
)
```

---

## Dependencies

- `rdkit` - Molecular property calculations
- `numpy`, `pandas` - Data handling
- `matplotlib`, `seaborn` - Visualization
- `requests` - HTTP client
- `asyncio`, `nest_asyncio` - Async operations
- `cma` - CMA-ES optimization
- `boltz2_client` - Boltz2 NIM client

---

## Version

`1.0.0`

## Authors

NVIDIA BioNeMo Hackathon Team
