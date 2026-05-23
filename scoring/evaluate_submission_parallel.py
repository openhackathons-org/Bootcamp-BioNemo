#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
Parallel AI-Powered-Drug-Discovery-Bootcamp Evaluation Script with Multi-Endpoint Boltz2 Support

This script evaluates chemical compound submissions against multiple criteria
using parallel processing with multiple Boltz2 NIM endpoints for faster predictions.

Usage:
    python evaluate_submission_parallel.py <smiles_file> <team_name> [options]

Example:
    python evaluate_submission_parallel.py demo_compounds.csv TeamA --endpoints 8000,8001,8002 --max-workers 6
"""

import pandas as pd
import numpy as np
import time
import json
import argparse
import os
import sys
import asyncio
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from shutil import copy2
from Bio.PDB import MMCIFParser, PDBIO
from string import ascii_uppercase, digits
import re
import traceback
import warnings
warnings.filterwarnings('ignore')

from endpoint_env import boltz2_client_kwargs, boltz2_endpoint_urls, load_openhackathon_env, normalize_endpoint_urls

load_openhackathon_env()

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import AllChem, DataStructs, FilterCatalog
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False

# Boltz2 client import
try:
    from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest, PocketConstraint
    BOLTZ2_AVAILABLE = True
except ImportError:
    print("Warning: boltz2-python-client not available. Install with: pip install boltz2-python-client")
    BOLTZ2_AVAILABLE = False

# Other imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Configuration
CONFIG = {
    "endpoints": boltz2_endpoint_urls("8000,8001,8002"),
    "boltz2_url": os.environ.get("BOLTZ2_URL", "http://localhost:8000"),
    "max_workers": 6,  # Maximum concurrent workers
    "api_timeout": 300,  # Timeout for API calls in seconds
    "confidence_threshold": 0.7,
    "weights": {
        "binding_affinity": 0.35,
        "cdk11_avoidance": 0.25,
        "qed": 0.15,
        "sa_score": 0.15,
        "pains": 0.05,
        "novelty": 0.05
    },
    "chembl_data_path": str(Path(__file__).resolve().parent / "chembl_data"),
    "novelty_cutoff": 0.85
}

if RDKIT_AVAILABLE:
    ExplicitBitVect = DataStructs.ExplicitBitVect  # type: ignore[attr-defined]
else:
    ExplicitBitVect = object

_CHEMBL_FP_CACHE: Optional[Dict[str, ExplicitBitVect]] = None


def get_novelty_cutoff() -> float:
    return CONFIG.get("novelty_cutoff", 0.85)


def load_chembl_fingerprints(chembl_path: str) -> Dict[str, ExplicitBitVect]:
    global _CHEMBL_FP_CACHE

    if _CHEMBL_FP_CACHE is not None:
        return _CHEMBL_FP_CACHE

    fp_path = os.path.join(chembl_path, "chembl_fingerprints.pkl")

    if os.path.exists(fp_path):
        print(f"Loading ChEMBL fingerprints from {fp_path}...")
        with open(fp_path, "rb") as f:
            _CHEMBL_FP_CACHE = pickle.load(f)
    else:
        print("Warning: ChEMBL fingerprints not found. Using empty reference set.")
        _CHEMBL_FP_CACHE = {}

    return _CHEMBL_FP_CACHE


def calculate_novelty_scores(
    df: pd.DataFrame,
    chembl_path: str,
    similarity_cutoff: Optional[float] = None,
) -> pd.DataFrame:
    if not RDKIT_AVAILABLE:
        df['max_chembl_similarity'] = np.nan
        df['is_novel'] = True
        df['novelty_normalized'] = 0.5
        return df

    cutoff = similarity_cutoff if similarity_cutoff is not None else get_novelty_cutoff()
    reference_fps = load_chembl_fingerprints(chembl_path)

    if not reference_fps:
        print("No reference compounds available. Marking all compounds as novel.")
        df['max_chembl_similarity'] = 0.0
        df['is_novel'] = True
        df['novelty_normalized'] = 1.0
        return df

    query_fps: List[Optional[ExplicitBitVect]] = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            query_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        else:
            query_fps.append(None)

    max_similarities: List[float] = []
    reference_fp_values = list(reference_fps.values())

    for query_fp in tqdm(query_fps, desc="Novelty scoring") if TQDM_AVAILABLE else query_fps:
        if query_fp is None:
            max_similarities.append(np.nan)
            continue

        max_sim = 0.0
        for ref_fp in reference_fp_values:
            sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            if sim > max_sim:
                max_sim = sim
                if max_sim >= cutoff:
                    break

        max_similarities.append(max_sim)

    df['max_chembl_similarity'] = max_similarities
    df['is_novel'] = df['max_chembl_similarity'] < cutoff
    df['novelty_normalized'] = 1.0 - df['max_chembl_similarity'].fillna(cutoff).clip(upper=cutoff) / cutoff
    df.loc[~df['is_novel'], 'novelty_normalized'] = 0.0

    n_novel = df['is_novel'].sum()
    print(f"Novel compounds: {n_novel}/{len(df)} ({(n_novel/len(df))*100:.1f}%) using cutoff {cutoff:.2f}")

    return df


def apply_pains_filter_parallel(df: pd.DataFrame) -> pd.DataFrame:
    if not RDKIT_AVAILABLE:
        df['is_pains'] = False
        df['pains_score'] = 1.0
        df['pains_score_norm'] = 1.0
        df['pains_alerts'] = [[] for _ in range(len(df))]
        return df

    print("Applying PAINS filters...")

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)

    pains_flags = []
    pains_matches = []

    mols = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

    for mol in tqdm(mols, desc="PAINS") if TQDM_AVAILABLE else mols:
        if mol is None:
            pains_flags.append(np.nan)
            pains_matches.append([])
            continue

        if not catalog.HasMatch(mol):
            pains_flags.append(False)
            pains_matches.append([])
        else:
            matches = [entry.GetDescription() for entry in catalog.GetMatches(mol)]
            pains_flags.append(True)
            pains_matches.append(matches)

    df['is_pains'] = pains_flags
    df['pains_alerts'] = [list(filter(None, alerts)) for alerts in pains_matches]
    df['pains_score'] = np.where(df['is_pains'] == True, 0.0, 1.0)
    df['pains_score_norm'] = df['pains_score']

    n_pains = (df['is_pains'] == True).sum()
    print_rt(f"PAINS-positive compounds: {n_pains}/{len(df)} ({n_pains/len(df)*100:.1f}%)")
    if n_pains > 0:
        print_rt("Top PAINS alerts:")
        alert_counts: Dict[str, int] = {}
        for alerts in df.loc[df['is_pains'] == True, 'pains_alerts']:
            for alert in alerts:
                alert_counts[alert] = alert_counts.get(alert, 0) + 1
        for alert, count in sorted(alert_counts.items(), key=lambda item: item[1], reverse=True)[:5]:
            print_rt(f"  {alert}: {count}")

    return df

# Protein sequences for CDK targets
CDK_PROTEIN_INFO = {
    "CDK4": {
        "sequence": "MATSRYEPVAEIGVGAYGTVYKARDPHSGHFVALKSVRVPNGGGGGGGLPISTVREVALLRRLEAFEHPNVVRLMDVCATSRTDREIKVTLVFEHVDQDLRTYLDKAPPPGLPAETIKDLMRQFLRGLDFLHANCIVHRDLKPENILVTSGGTVKLADFGLARIYSYQMALTPVVVTLWYRAPEVLLQSTYATPVDMWSVGCIFAEMFRRKPLFCGNSEADQLGKIFDLIGLPPEDDWPRDVSLPRGAFPPRGPRPVQSVVPEMEESGAQLLLEMLTFNPHKRISAFRALQHSYLHKDEGNPE",
        "binding_site_residues": [
            {"residue": "Lys35", "position": 35},
            {"residue": "Glu71", "position": 71},
            {"residue": "Val96", "position": 96},
            {"residue": "Lys112", "position": 112},
            {"residue": "Asp158", "position": 158},
            {"residue": "Phe164", "position": 164},
            {"residue": "Leu196", "position": 196}
        ],
    },
    "CDK11": {
        "sequence": "ALQGCRSVEEFQCLNRIEEGTYGVVYRAKDKKTDEIVALKRLKMEKEKEGFPITSLREINTILKAQHPNIVTVREIVVGSNMDKIYIVMNYVEHDLKSLMETMKQPFLPGEVKTLMIQLLRGVKHLHDNWILHRDLKTSNLLLSHAGILKVGDFGLAREYGSPLKAYTPVVVTLWYRAPELLLGAKEYSTAVDMWSVGCIFAEMFRRKPLFPGKSEIDQINKVFKDLGTPSEKIWPGYSELPAVKKMTFSEHPYNNLRKRFGALLSDQGFDLMNKFLTYFPGRRISAEDGLKHEYFRETPLPIDPSMFPKLVEKY",
        "binding_site_residues": [
            {"residue": "Lys41", "position": 41},
            {"residue": "Glu87", "position": 87},
            {"residue": "Val113", "position": 113},
            {"residue": "Lys128", "position": 128},
            {"residue": "Asp175", "position": 175},
            {"residue": "Phe182", "position": 182},
            {"residue": "Asp206", "position": 206}
        ],
    }
}

# Endpoint management


class EndpointPool:
    """Manages a pool of Boltz2 endpoints with health checking and load balancing"""
    
    def __init__(self, endpoints: List[str], timeout: int = 300):
        self.endpoints = endpoints
        self.timeout = timeout
        self.healthy_endpoints = []
        self.clients = {}
        self.current_index = 0
        
    async def check_health(self, endpoint: str) -> bool:
        """Check if an endpoint is healthy using boltz2-python-client"""
        try:
            if BOLTZ2_AVAILABLE:
                # Create client and test with a simple request
                client = Boltz2Client(**boltz2_client_kwargs(endpoint, timeout=10))
                
                # Test with a minimal prediction request to verify the endpoint works
                test_protein = Polymer(id="test_protein", sequence="MKLLKWAWLLLSKASSAHDKA")  # Short test sequence
                test_ligand = Ligand(id="test_ligand", smiles="CCO", predict_affinity=True)  # Simple ethanol
                
                test_request = PredictionRequest(
                    polymers=[test_protein],
                    ligands=[test_ligand],
                    recycling_steps=1,      # Minimal settings for health check
                    sampling_steps=10,      # Minimum required
                    diffusion_samples=1,
                    sampling_steps_affinity=20,  # Reduced for health check
                    diffusion_samples_affinity=1,
                    affinity_mw_correction=True
                )
                
                print(f"Testing endpoint {endpoint} with minimal prediction...")
                
                # Try to make a prediction - if this works, the endpoint is healthy
                test_prediction = await client.predict(test_request)
                
                # If we get here without exception, the endpoint is working
                self.clients[endpoint] = client
                print(f"✓ Endpoint {endpoint} is healthy and responsive")
                return True
                
            return False
        except Exception as e:
            print(f"Health check failed for {endpoint}: {e}")
            return False
    
    async def initialize(self):
        """Initialize the endpoint pool by checking health of all endpoints"""
        print(f"Initializing endpoint pool with {len(self.endpoints)} endpoints...")
        for endpoint in self.endpoints:
            is_healthy = await self.check_health(endpoint)
            if is_healthy:
                self.healthy_endpoints.append(endpoint)
                print(f"✓ {endpoint} is healthy")
            else:
                print(f"✗ {endpoint} is not available")
        
        if not self.healthy_endpoints:
            raise Exception("No healthy Boltz2 endpoints found!")
        
        print(f"Endpoint pool initialized with {len(self.healthy_endpoints)} healthy endpoints")
    
    def get_next_endpoint(self) -> Tuple[str, Optional[object]]:
        """Get the next available endpoint using round-robin"""
        if not self.healthy_endpoints:
            return None, None
            
        endpoint = self.healthy_endpoints[self.current_index]
        client = self.clients.get(endpoint)
        
        self.current_index = (self.current_index + 1) % len(self.healthy_endpoints)
        return endpoint, client

def print_rt(message: str):
    """Print message in real-time with immediate flush"""
    print(message, flush=True)


class TeeIO:
    """Duplicate writes to multiple streams (e.g., stdout and log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        for stream in self.streams:
            encoding = getattr(stream, "encoding", None)
            if encoding:
                return encoding
        return "utf-8"


def sanitize_filename(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe_value or "structure"


def get_output_dir() -> Optional[Path]:
    output_dir = CONFIG.get("current_output_dir")
    return Path(output_dir) if output_dir else None


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    """Convert an mmCIF file to PDB format using Biopython."""
    parser = MMCIFParser(QUIET=True)
    structure_id = cif_path.stem
    structure = parser.get_structure(structure_id, str(cif_path))

    available_ids = list(ascii_uppercase + digits)
    used_ids = set()

    for model in structure:
        for chain in model:
            cid = (chain.id or "").strip()
            if len(cid) == 1 and cid in available_ids and cid not in used_ids:
                used_ids.add(cid)
                continue

            for candidate in available_ids:
                if candidate not in used_ids:
                    chain.id = candidate
                    used_ids.add(candidate)
                    break
            else:
                raise ValueError("Unable to assign unique chain ID within PDB format limits.")

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def save_prediction_structures(
    prediction: Any,
    compound_label: str,
    protein_target: str,
    prediction_start_time: Optional[float] = None
) -> List[Path]:
    output_dir = get_output_dir()
    if output_dir is None:
        return []

    structures = getattr(prediction, "structures", None)
    if not structures:
        return []

    structures_dir = output_dir / "boltz2_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    compound_slug = sanitize_filename(compound_label or "compound")
    saved_cif_paths: List[Path] = []

    for idx, structure in enumerate(structures, start=1):
        cif_data = None
        structure_path = None

        if hasattr(structure, "mmcif") and structure.mmcif:
            cif_data = structure.mmcif
        if hasattr(structure, "path") and structure.path:
            structure_path = structure.path
        if hasattr(structure, "file_path") and structure.file_path:
            structure_path = structure.file_path

        if hasattr(structure, "model_dump"):
            structure_dict = structure.model_dump()
            cif_data = cif_data or structure_dict.get("mmcif") or structure_dict.get("mmCIF")
            structure_path = structure_path or structure_dict.get("path") or structure_dict.get("file_path")
        elif isinstance(structure, dict):
            cif_data = cif_data or structure.get("mmcif") or structure.get("mmCIF")
            structure_path = structure_path or structure.get("path") or structure.get("file_path")

        filename_base = f"{compound_slug}_{protein_target}_structure{idx}"

        if cif_data:
            cif_path = structures_dir / f"{filename_base}.cif"
            with open(cif_path, "w") as cif_file:
                cif_file.write(cif_data)
            saved_cif_paths.append(cif_path)
            continue

        if structure_path:
            src = Path(structure_path)
            if src.exists():
                extension = src.suffix or ".cif"
                cif_path = structures_dir / f"{filename_base}{extension}"
                copy2(src, cif_path)
                saved_cif_paths.append(cif_path)

    if prediction_start_time is not None:
        existing_names = {path.name for path in saved_cif_paths}
        for cif_file in Path.cwd().glob("structure*.cif"):
            try:
                if prediction_start_time is not None and cif_file.stat().st_mtime < prediction_start_time - 1:
                    continue
            except OSError:
                continue

            target_name = f"{compound_slug}_{protein_target}_{cif_file.name}"
            if target_name in existing_names:
                try:
                    cif_file.unlink()
                except OSError:
                    pass
                continue

            cif_path = structures_dir / target_name
            try:
                copy2(cif_file, cif_path)
                saved_cif_paths.append(cif_path)
                existing_names.add(target_name)
            except OSError:
                continue
            finally:
                try:
                    cif_file.unlink()
                except OSError:
                    pass

    for cif_path in saved_cif_paths:
        pdb_path = cif_path.with_suffix(".pdb")
        try:
            convert_cif_to_pdb(cif_path, pdb_path)
            try:
                cif_path.unlink()
            except OSError:
                pass
        except Exception:
            continue

    return saved_cif_paths


def format_affinity_result(row: pd.Series, target: str) -> Dict[str, Any]:
    ic50 = row.get(f"{target}_ic50_nm")
    pic50 = row.get(f"{target}_pic50")
    ic50_raw = row.get(f"{target}_ic50_nm_raw")
    pic50_raw = row.get(f"{target}_pic50_raw")
    confidence = row.get(f"{target}_confidence")
    accepted = row.get(f"{target}_accepted")

    display_ic50 = ic50 if pd.notna(ic50) else ic50_raw
    display_pic50 = pic50 if pd.notna(pic50) else pic50_raw

    suffix = ""
    if accepted is False and pd.notna(display_ic50):
        suffix = " (low confidence)"

    ic50_str = f"{display_ic50:.1f}" if display_ic50 is not None and not pd.isna(display_ic50) else "N/A"
    pic50_str = f"{display_pic50:.2f}" if display_pic50 is not None and not pd.isna(display_pic50) else "N/A"
    confidence_str = f"{confidence:.2f}" if confidence is not None and not pd.isna(confidence) else "N/A"

    return {
        "ic50_str": ic50_str,
        "pic50_str": pic50_str,
        "confidence_str": confidence_str,
        "suffix": suffix,
        "ic50_value": display_ic50,
        "pic50_value": display_pic50,
        "confidence": confidence,
        "accepted": bool(accepted) if accepted is True else False
    }

async def predict_binding_affinity_boltz2_async(smiles: str, protein_target: str, 
                                               endpoint_pool: EndpointPool,
                                               confidence_threshold: float = CONFIG["confidence_threshold"],
                                               verbose: bool = True,
                                               worker_id: int = 0,
                                               compound_id: Optional[str] = None) -> Dict:
    """Async predict IC50 values using Boltz2 Python client with endpoint pool"""
    
    start_time = time.time()
    api_time = 0.0
    ic50_nm = None
    pic50 = None
    confidence = 0.8
    endpoint, client = endpoint_pool.get_next_endpoint()
    compound_label = compound_id or f"compound_{abs(hash(smiles)) % 10**8}"
    binding_sites = CDK_PROTEIN_INFO[protein_target].get("binding_site_residues", [])
    binding_site_summary = ", ".join(
        f"{site['residue']} (pos {site['position']})" for site in binding_sites
    ) if binding_sites else ""
    binding_site_positions = [
        int(site["position"]) for site in binding_sites if "position" in site
    ]
    
    if verbose:
        print_rt(f"\n[Worker {worker_id}] {'='*60}")
        print_rt(f"[Worker {worker_id}] BOLTZ2 PREDICTION REQUEST - {protein_target}")
        print_rt(f"[Worker {worker_id}] Endpoint: {endpoint}")
        print_rt(f"[Worker {worker_id}] {'='*60}")
        print_rt(f"[Worker {worker_id}] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print_rt(f"[Worker {worker_id}] SMILES: {smiles}")
        print_rt(f"[Worker {worker_id}] Protein: {protein_target}")
        print_rt(f"[Worker {worker_id}] Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} aa")
        if compound_id:
            print_rt(f"[Worker {worker_id}] Compound ID: {compound_id}")
        if binding_site_positions:
            print_rt(f"[Worker {worker_id}] Binding site residue indices: {binding_site_positions}")
    
    try:
        if BOLTZ2_AVAILABLE and client and endpoint:
            if verbose:
                print_rt(f"[Worker {worker_id}] Mode: ACTUAL Boltz2 Python Client")
                print_rt(f"[Worker {worker_id}] Endpoint: {endpoint}")
                print_rt(f"[Worker {worker_id}] Client type: {type(client)}")
                print_rt(f"[Worker {worker_id}] BOLTZ2_AVAILABLE: {BOLTZ2_AVAILABLE}")
            
            protein_sequence = CDK_PROTEIN_INFO[protein_target]["sequence"]
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nINPUT DETAILS:")
                print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
                print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
                print_rt(f"[Worker {worker_id}]   Sequence length: {len(protein_sequence)} residues")
            
            # Create prediction request
            protein = Polymer(id=f"protein_{protein_target}", sequence=protein_sequence)
            ligand = Ligand(
                id=f"ligand_{protein_target}_{hash(smiles) % 10000}",
                smiles=smiles,
                predict_affinity=True
            )
            
            constraints = None
            if binding_site_positions:
                constraints = [
                    PocketConstraint(
                        ligand_id=ligand.id,
                        polymer_id=protein.id,
                        residue_ids=binding_site_positions,
                        binder=ligand.id
                    )
                ]
            
            request = PredictionRequest(
                polymers=[protein],
                ligands=[ligand],
                constraints=constraints,
                recycling_steps=3,
                sampling_steps=10,
                diffusion_samples=1,
                sampling_steps_affinity=50,
                diffusion_samples_affinity=2,
                affinity_mw_correction=True
            )
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nREQUEST PAYLOAD:")
                try:
                    print(json.dumps(request.model_dump(), indent=2, default=str))
                except Exception:
                    print_rt(f"[Worker {worker_id}] {request}")
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nBOLTZ2 REQUEST PARAMETERS:")
                print_rt(f"[Worker {worker_id}]   Protein ID: {protein.id}")
                print_rt(f"[Worker {worker_id}]   Ligand ID: {ligand.id}")
                print_rt(f"[Worker {worker_id}]   Affinity prediction: {ligand.predict_affinity}")
                print_rt(f"[Worker {worker_id}] \nSending request to Boltz2...")
            
            # Make async prediction
            try:
                prediction = await client.predict(request)
                api_time = time.time() - start_time
                
                if verbose:
                    print_rt(f"[Worker {worker_id}] Prediction successful in {api_time:.2f} seconds")
                saved_cif_paths = save_prediction_structures(
                    prediction,
                    compound_label,
                    protein_target,
                    prediction_start_time=start_time
                )
            except Exception as pred_error:
                if verbose:
                    print_rt(f"[Worker {worker_id}] Prediction failed: {pred_error}")
                raise pred_error
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nBOLTZ2 PREDICTION RESPONSE")
                print_rt(f"[Worker {worker_id}] {'-'*50}")
                print_rt(f"[Worker {worker_id}] Status: SUCCESS")
                print_rt(f"[Worker {worker_id}] Response received in {api_time:.2f} seconds")
            
            # Extract affinity data
            pic50 = 5.0  # Default
            confidence = 0.5  # Default
            
            if hasattr(prediction, 'affinities') and prediction.affinities:
                ligand_id = f"ligand_{protein_target}_{hash(smiles) % 10000}"
                if ligand_id in prediction.affinities:
                    affinity = prediction.affinities[ligand_id]
                    if hasattr(affinity, 'affinity_pic50') and affinity.affinity_pic50:
                        pic50 = affinity.affinity_pic50[0]
                    
                    if hasattr(affinity, 'affinity_probability_binary') and affinity.affinity_probability_binary:
                        confidence = affinity.affinity_probability_binary[0]
                    
                    ic50_nm = 10**(9 - pic50)
                    
                    if verbose:
                        print_rt(f"[Worker {worker_id}] \nAFFINITY PREDICTION RESULTS:")
                        print_rt(f"[Worker {worker_id}]   pIC50: {pic50:.2f}")
                        print_rt(f"[Worker {worker_id}]   IC50: {ic50_nm:.2f} nM")
                        print_rt(f"[Worker {worker_id}]   Binding probability: {confidence:.1%}")
                        
                        binding_strength = "STRONG" if pic50 > 7.0 else "MODERATE" if pic50 > 5.0 else "WEAK"
                        print_rt(f"[Worker {worker_id}]   Binding strength: {binding_strength}")
                else:
                    if verbose:
                        print_rt(f"[Worker {worker_id}]   Warning: No pIC50 data in affinity response")
            else:
                if verbose:
                    print_rt(f"[Worker {worker_id}]   Warning: No affinity data in response, using estimated values")
                
                # Use mock values based on target
                if protein_target in ["CDK4"]:
                    ic50_nm = np.random.lognormal(np.log(50), 1.5)
                else:
                    ic50_nm = np.random.lognormal(np.log(5000), 1.5)
                pic50 = -np.log10(ic50_nm * 1e-9)
                
                if verbose:
                    print_rt(f"[Worker {worker_id}] \nESTIMATED AFFINITY (MOCK):")
                    print_rt(f"[Worker {worker_id}]   pIC50: {pic50:.2f}")
                    print_rt(f"[Worker {worker_id}]   IC50: {ic50_nm:.2f} nM")
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nConfidence: {confidence:.3f}")
                print_rt(f"[Worker {worker_id}] Accepted: {confidence >= confidence_threshold}")
                if 'saved_cif_paths' in locals() and saved_cif_paths:
                    print_rt(f"[Worker {worker_id}] Saved {len(saved_cif_paths)} CIF file(s):")
                    for cif_path in saved_cif_paths:
                        print_rt(f"[Worker {worker_id}]   - {cif_path}")
            
        else:
            # No mock fallback - raise error if no valid client/endpoint
            error_msg = f"No valid Boltz2 client available for prediction"
            if verbose:
                print_rt(f"[Worker {worker_id}] ❌ PREDICTION FAILED")
                print_rt(f"[Worker {worker_id}] Error: {error_msg}")
                print_rt(f"[Worker {worker_id}] Reason: BOLTZ2_AVAILABLE={BOLTZ2_AVAILABLE}, client={client is not None}, endpoint={endpoint}")
                print_rt(f"[Worker {worker_id}] \nINPUT DETAILS:")
                print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
                print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
                print_rt(f"[Worker {worker_id}]   Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")

            raise RuntimeError(error_msg)

        total_time = time.time() - start_time

        ic50_raw_value = float(ic50_nm) if ic50_nm is not None else np.nan
        pic50_raw_value = float(pic50) if pic50 is not None else np.nan
        prediction_accepted = confidence >= confidence_threshold

        return {
            'ic50_nm': ic50_nm if prediction_accepted else np.nan,
            'pic50': pic50 if prediction_accepted else np.nan,
            'confidence': confidence,
            'accepted': prediction_accepted,
            'api_time': api_time,
            'total_time': total_time,
            'endpoint': endpoint,
            'worker_id': worker_id,
            'ic50_nm_raw': ic50_raw_value,
            'pic50_raw': pic50_raw_value,
            'binding_site_residues': binding_site_summary
        }
    
    except Exception as e:
        if verbose:
            print_rt(f"[Worker {worker_id}] \nBOLTZ2 PREDICTION RESPONSE")
            print_rt(f"[Worker {worker_id}] {'-'*50}")
            print_rt(f"[Worker {worker_id}] ERROR: {type(e).__name__}: {str(e)}")
            print_rt(f"[Worker {worker_id}] \nRequest that caused error:")
            print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
            print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
            print_rt(f"[Worker {worker_id}]   Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
            print_rt(f"[Worker {worker_id}] \n❌ NO MOCK FALLBACK - PREDICTION FAILED")
        
        # Re-raise the exception instead of using fallback values
        raise e

def predict_binding_affinity_boltz2_sync(smiles: str, protein_target: str, 
                                        endpoint_pool: EndpointPool,
                                        confidence_threshold: float = CONFIG["confidence_threshold"],
                                        verbose: bool = True,
                                        worker_id: int = 0) -> Dict:
    """Synchronous wrapper for async prediction"""
    return asyncio.run(predict_binding_affinity_boltz2_async(
        smiles, protein_target, endpoint_pool, confidence_threshold, verbose, worker_id
    ))

async def calculate_all_binding_affinities_parallel(df: pd.DataFrame, 
                                                   endpoint_pool: EndpointPool,
                                                   max_workers: int = 6,
                                                   verbose: bool = True) -> pd.DataFrame:
    """Calculate binding affinities in parallel using multiple endpoints"""
    print_rt(f"\nCalculating binding affinities using {len(endpoint_pool.healthy_endpoints)} endpoints with {max_workers} workers...")
    
    for target in ["CDK4", "CDK11"]:
        df[f"{target}_ic50_nm"] = np.nan
        df[f"{target}_pic50"] = np.nan
        df[f"{target}_confidence"] = np.nan
        df[f"{target}_accepted"] = False
        df[f"{target}_ic50_nm_raw"] = np.nan
        df[f"{target}_pic50_raw"] = np.nan
        df[f"{target}_binding_site_residues"] = ""
        df[f"{target}_endpoint"] = ""
        df[f"{target}_api_time"] = np.nan
    
    # Prepare prediction tasks
    tasks = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        compound_id = row.get('compound_id', f"compound_{idx+1}")
        for target in ["CDK4", "CDK11"]:
            tasks.append((idx, smiles, target, compound_id))
    
    results = {}
    completed_tasks = 0
    total_tasks = len(tasks)
    
    # Use semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(max_workers)
    
    async def worker_task(task_data, worker_id):
        async with semaphore:
            idx, smiles, target, compound_id = task_data
            result = await predict_binding_affinity_boltz2_async(
                smiles, target, endpoint_pool, CONFIG["confidence_threshold"], verbose, worker_id, compound_id
            )
            return (idx, target, result)
    
    # Start all tasks
    worker_tasks = []
    for i, task_data in enumerate(tasks):
        worker_id = i % max_workers
        worker_tasks.append(worker_task(task_data, worker_id))
    
    # Process results as they complete
    for task in asyncio.as_completed(worker_tasks):
        idx, target, result = await task
        
        if idx not in results:
            results[idx] = {}
        results[idx][target] = result
        
        completed_tasks += 1
        if verbose:
            progress = (completed_tasks / total_tasks) * 100
            print_rt(f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) - Latest: {target} for compound {idx}")
    
    # Apply results to dataframe
    for idx in results:
        for target in ["CDK4", "CDK11"]:
            if target in results[idx]:
                result = results[idx][target]
                df.loc[idx, f'{target}_ic50_nm'] = result.get('ic50_nm', np.nan)
                df.loc[idx, f'{target}_pic50'] = result.get('pic50', np.nan)
                df.loc[idx, f'{target}_confidence'] = result['confidence']
                df.loc[idx, f'{target}_accepted'] = result['accepted']
                df.loc[idx, f'{target}_api_time'] = result.get('api_time', np.nan)
                df.loc[idx, f'{target}_endpoint'] = result.get('endpoint', '')
                df.loc[idx, f'{target}_ic50_nm_raw'] = result.get('ic50_nm_raw', np.nan)
                df.loc[idx, f'{target}_pic50_raw'] = result.get('pic50_raw', np.nan)
                df.loc[idx, f'{target}_binding_site_residues'] = result.get('binding_site_residues', '')
    
    # Calculate selectivity metrics
    # Use raw fallback values when accepted predictions missing
    df['CDK4_pic50_eff'] = df.apply(
        lambda row: row['CDK4_pic50'] if pd.notna(row['CDK4_pic50']) else row['CDK4_pic50_raw'], axis=1
    )
    df['CDK11_pic50_eff'] = df.apply(
        lambda row: row['CDK11_pic50'] if pd.notna(row['CDK11_pic50']) else row['CDK11_pic50_raw'], axis=1
    )
    df['CDK4_ic50_nm_eff'] = df.apply(
        lambda row: row['CDK4_ic50_nm'] if pd.notna(row['CDK4_ic50_nm']) else row['CDK4_ic50_nm_raw'], axis=1
    )
    df['CDK11_ic50_nm_eff'] = df.apply(
        lambda row: row['CDK11_ic50_nm'] if pd.notna(row['CDK11_ic50_nm']) else row['CDK11_ic50_nm_raw'], axis=1
    )

    df['cdk11_avoidance'] = np.maximum(0, df['on_target_pic50'] - df['CDK11_pic50_eff'])
    df['selectivity_ratio'] = df['on_target_pic50'] / (df['CDK11_pic50_eff'] + 1e-6)
    
    if verbose:
        print_rt(f"\nAFFINITY PREDICTION RESULTS SUMMARY:")
        print_rt(f"Total predictions: {completed_tasks}")
        print_rt(f"Endpoints used: {len(endpoint_pool.healthy_endpoints)}")
        for idx, row in df.iterrows():
            print_rt(f"Compound {idx + 1}:")
            cdk4 = format_affinity_result(row, "CDK4")
            cdk11 = format_affinity_result(row, "CDK11")
            print_rt(f"  CDK4:  IC50 = {cdk4['ic50_str']} nM{cdk4['suffix']}, pIC50 = {cdk4['pic50_str']}, confidence = {cdk4['confidence_str']}")
            print_rt(f"  CDK11: IC50 = {cdk11['ic50_str']} nM{cdk11['suffix']}, pIC50 = {cdk11['pic50_str']}, confidence = {cdk11['confidence_str']}")
            print_rt(f"  Selectivity: {row.get('selectivity_ratio', np.nan):.2f}, CDK11 avoidance: {row.get('cdk11_avoidance', np.nan):.2f}")
    
    return df

def calculate_qed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate QED scores for compounds"""
    if not RDKIT_AVAILABLE:
        df['qed'] = 0.5
        df['qed_normalized'] = 0.5
        return df
    
    print("Calculating QED scores...")
    qed_scores = []
    
    for smiles in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                qed_score = QED.qed(mol)
            else:
                qed_score = 0.0
        except:
            qed_score = 0.0
        qed_scores.append(qed_score)
    
    df['qed'] = qed_scores
    df['qed_normalized'] = df['qed']  # QED is already normalized 0-1
    
    return df

def calculate_sa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Synthetic Accessibility scores"""
    if not RDKIT_AVAILABLE:
        df['sa_score'] = 5.0
        df['sa_score_normalized'] = 0.5
        return df
    
    print("Calculating Synthetic Accessibility scores...")
    sa_scores = []
    
    for smiles in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Simplified SA score calculation
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                rotbonds = CalcNumRotatableBonds(mol)
                heteroatoms = Descriptors.NumHeteroatoms(mol)
                
                # Simple heuristic (lower is better, range 1-10)
                sa_score = min(10, max(1, 
                    1 + (mw / 100) + abs(logp) + (rotbonds / 3) + (heteroatoms / 2)
                ))
            else:
                sa_score = 10.0
        except:
            sa_score = 10.0
        sa_scores.append(sa_score)
    
    df['sa_score'] = sa_scores
    df['sa_score_normalized'] = 1 - (df['sa_score'] / 10)  # Normalize and invert
    
    return df

def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all scores to 0-1 range"""
    print("Normalizing scores...")
    
    # Binding affinity: Higher pIC50 is better
    if 'on_target_pic50' in df.columns:
        df['binding_affinity_normalized'] = (df['on_target_pic50'] - df['on_target_pic50'].min()) / \
                                          (df['on_target_pic50'].max() - df['on_target_pic50'].min() + 1e-6)
    else:
        df['binding_affinity_normalized'] = 0.5
    
    # CDK11 avoidance: Higher is better
    if 'cdk11_avoidance' in df.columns:
        df['cdk11_avoidance_normalized'] = (df['cdk11_avoidance'] - df['cdk11_avoidance'].min()) / \
                                         (df['cdk11_avoidance'].max() - df['cdk11_avoidance'].min() + 1e-6)
    else:
        df['cdk11_avoidance_normalized'] = 0.5
    
    # Other scores are already normalized or set defaults
    for score_type in ['qed', 'sa_score', 'pains_score', 'novelty']:
        norm_col = f'{score_type}_normalized'
        if norm_col not in df.columns:
            df[norm_col] = 0.5
    
    return df

def calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted composite scores"""
    print("Calculating composite scores...")
    
    weights = CONFIG["weights"]
    
    # Ensure all normalized columns exist
    required_cols = [
        'binding_affinity_normalized', 'cdk11_avoidance_normalized',
        'qed_normalized', 'sa_score_normalized', 'pains_score_norm', 'novelty_normalized'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.5  # Default neutral score
    
    df['composite_score'] = (
        df['binding_affinity_normalized'] * weights['binding_affinity'] +
        df['cdk11_avoidance_normalized'] * weights['cdk11_avoidance'] +
        df['qed_normalized'] * weights['qed'] +
        df['sa_score_normalized'] * weights['sa_score'] +
        df['pains_score_norm'] * weights['pains'] +
        df['novelty_normalized'] * weights['novelty']
    )
    
    return df

def generate_html_report(df: pd.DataFrame, team_name: str, output_dir: str):
    """Generate HTML report"""
    print("Generating HTML report...")
    
    # Sort by composite score
    df_sorted = df.sort_values('composite_score', ascending=False)
    top_25 = df_sorted.head(25)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Powered-Drug-Discovery-Bootcamp Evaluation Report - {team_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; }}
            .table-container {{ overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .score-high {{ background-color: #d4edda; }}
            .score-medium {{ background-color: #fff3cd; }}
            .score-low {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI-Powered-Drug-Discovery-Bootcamp Evaluation Report</h1>
            <h2>Team: {team_name}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Evaluated using parallel processing with multiple Boltz2 endpoints</p>
        </div>
        
        <div class="summary">
            <h3>Summary Statistics</h3>
            <p>Total compounds evaluated: {len(df)}</p>
            <p>Top 25 compounds average score: {top_25['composite_score'].mean():.3f}</p>
            <p>Best compound score: {df['composite_score'].max():.3f}</p>
            <p>Evaluation method: Parallel processing with {len(set(df.get('CDK4_endpoint', ['localhost:8000'])))} Boltz2 endpoints</p>
        </div>
        
        <div class="table-container">
            <h3>Top 25 Compounds</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>SMILES</th>
                        <th>Composite Score</th>
                        <th>CDK4 pIC50</th>
                        <th>CDK11 pIC50</th>
                        <th>Selectivity Ratio</th>
                        <th>QED</th>
                        <th>SA Score</th>
                        <th>PAINS Alerts</th>
                        <th>Max ChEMBL Similarity</th>
                        <th>Novel</th>
                        <th>Processing Endpoint</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i, (_, row) in enumerate(top_25.iterrows(), 1):
        score_class = "score-high" if row['composite_score'] > 0.7 else "score-medium" if row['composite_score'] > 0.5 else "score-low"
        endpoint = row.get('CDK4_endpoint', 'Unknown')
        cdk4 = format_affinity_result(row, "CDK4")
        cdk11 = format_affinity_result(row, "CDK11")
        
        html_content += f"""
                    <tr class="{score_class}">
                        <td>{i}</td>
                        <td style="font-family: monospace; text-align: left;">{row['smiles'][:50]}{'...' if len(row['smiles']) > 50 else ''}</td>
                        <td>{row['composite_score']:.3f}</td>
                        <td>{cdk4['pic50_str']}</td>
                        <td>{cdk6['pic50_str']}</td>
                        <td>{cdk11['pic50_str']}</td>
                        <td>{row.get('selectivity_ratio', 0):.2f}</td>
                        <td>{row.get('qed', 0):.3f}</td>
                        <td>{row.get('sa_score', 0):.2f}</td>
                        <td>{'; '.join(row.get('pains_alerts', [])) if row.get('is_pains', False) else 'None'}</td>
                        <td>{row.get('max_chembl_similarity', float('nan')):.2f}</td>
                        <td>{'Yes' if row.get('is_novel', False) else 'No'}</td>
                        <td>{endpoint}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, f"{team_name}_evaluation_report.html"), 'w') as f:
        f.write(html_content)

async def main():
    parser = argparse.ArgumentParser(description='Parallel AI-Powered-Drug-Discovery-Bootcamp Compound Evaluation')
    parser.add_argument('smiles_file', help='CSV file containing SMILES')
    parser.add_argument('team_name', help='Team name for the submission')
    parser.add_argument('--endpoints', default=os.environ.get("BOLTZ2_ENDPOINTS", os.environ.get("BOLTZ2_URL", "8000,8001,8002")),
                       help='Comma-separated Boltz2 endpoint URLs or ports (default: BOLTZ2_ENDPOINTS/BOLTZ2_URL or 8000,8001,8002)')
    parser.add_argument('--max-workers', type=int, default=6,
                       help='Maximum number of concurrent workers (default: 6)')
    parser.add_argument('--output-dir', default='evaluation_output',
                       help='Output directory for results')
    parser.add_argument('--skip-pains', action='store_true',
                       help='Skip PAINS filtering')
    parser.add_argument('--skip-novelty', action='store_true',
                       help='Skip novelty calculations')
    parser.add_argument('--novelty-cutoff', type=float, default=CONFIG["novelty_cutoff"],
                       help='Maximum Tanimoto similarity allowed before a compound is considered non-novel')
    parser.add_argument('--boltz2-url', default=CONFIG["boltz2_url"],
                       help='Base URL for the Boltz2 NIM endpoint (default: http://localhost:8000)')
    parser.add_argument('--skip-boltz2', action='store_true',
                       help='Skip Boltz2 predictions and use placeholder affinity values')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    endpoint_urls = normalize_endpoint_urls(args.endpoints, args.boltz2_url)
    CONFIG["endpoints"] = endpoint_urls
    CONFIG["max_workers"] = args.max_workers
    CONFIG["novelty_cutoff"] = args.novelty_cutoff
    CONFIG["boltz2_url"] = args.boltz2_url
    
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    team_slug = sanitize_filename(args.team_name)
    output_dir = base_output_dir / f"{team_slug}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["current_output_dir"] = str(output_dir)
    
    log_path = output_dir / f"{team_slug}_parallel_evaluation.log"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeIO(original_stdout, log_file)
    sys.stderr = TeeIO(original_stderr, log_file)
    
    try:
        print_rt(f"Evaluation log will be saved to: {log_path}")
        print_rt(f"Starting parallel evaluation for team: {args.team_name}")
        print_rt(f"Boltz2 base URL: {CONFIG['boltz2_url']}")
        print_rt(f"Using endpoints: {endpoint_urls}")
        print_rt(f"Max workers: {args.max_workers}")

        smiles_path = Path(args.smiles_file)
        if not smiles_path.exists():
            print_rt(f"Error: Input file '{smiles_path}' not found.")
            return

        if not args.skip_novelty:
            chembl_cache = Path(CONFIG["chembl_data_path"]) / "chembl_fingerprints.pkl"
            if not chembl_cache.exists():
                print_rt(f"Error: Required ChEMBL fingerprint cache not found at {chembl_cache}.")
                print_rt("Run the ChEMBL fingerprint preparation script or supply --skip-novelty.")
                return

        try:
            df = pd.read_csv(smiles_path)
            if 'smiles' not in df.columns:
                if 'SMILES' in df.columns:
                    df = df.rename(columns={'SMILES': 'smiles'})
                else:
                    print_rt("Error: No 'smiles' or 'SMILES' column found in the input file.")
                    return
            if 'compound_id' not in df.columns:
                df['compound_id'] = [f"COMP_{i+1:04d}" for i in range(len(df))]

            print_rt(f"Loaded {len(df)} compounds from {smiles_path}")
        except Exception as load_exc:
            print_rt(f"Error loading SMILES file: {load_exc}")
            traceback.print_exc()
            return

        start_time = time.time()

        if args.skip_boltz2:
            print_rt("Skipping Boltz2 predictions (using placeholder affinity values)")
            for target in ["CDK4", "CDK11"]:
                df[f"{target}_ic50_nm"] = np.nan
                df[f"{target}_pic50"] = np.nan
                df[f"{target}_confidence"] = np.nan
                df[f"{target}_accepted"] = False
                df[f"{target}_ic50_nm_raw"] = np.nan
                df[f"{target}_pic50_raw"] = np.nan
                binding_sites = CDK_PROTEIN_INFO[target].get("binding_site_residues", [])
                binding_site_summary = ", ".join(
                    f"{site['residue']} (pos {site['position']})" for site in binding_sites
                ) if binding_sites else ""
                df[f"{target}_binding_site_residues"] = binding_site_summary
            df['on_target_pic50'] = np.nan
            df['cdk11_avoidance'] = np.nan
            df['selectivity_ratio'] = np.nan
        else:
            endpoint_pool = EndpointPool(endpoint_urls, timeout=CONFIG["api_timeout"])
            await endpoint_pool.initialize()
            df = await calculate_all_binding_affinities_parallel(
                df, endpoint_pool, args.max_workers, args.verbose
            )

        df = calculate_qed_scores(df)
        df = calculate_sa_scores(df)

        if not args.skip_pains:
            df = apply_pains_filter_parallel(df)
        else:
            df['is_pains'] = False
            df['pains_score'] = 1.0
            df['pains_score_norm'] = 1.0
            print_rt("Skipping PAINS filtering (set to non-PAINS)")

        if not args.skip_novelty:
            df = calculate_novelty_scores(df, CONFIG["chembl_data_path"], CONFIG["novelty_cutoff"])
        else:
            df['max_chembl_similarity'] = np.nan
            df['is_novel'] = True
            df['novelty_normalized'] = 0.5
            print_rt("Skipping novelty calculations (set to neutral score)")

        df = normalize_scores(df)
        df = calculate_composite_scores(df)

        generate_html_report(df, args.team_name, str(output_dir))

        csv_path = output_dir / f"{team_slug}_detailed_results.csv"
        df.to_csv(csv_path, index=False)

        total_time = time.time() - start_time

        print_rt(f"\nEvaluation completed successfully!")
        print_rt(f"Total time: {total_time:.2f} seconds")
        print_rt(f"Average time per compound: {total_time/len(df):.2f} seconds")
        print_rt(f"Results saved to: {output_dir}")
        print_rt(f"HTML report: {(output_dir / f'{args.team_name}_evaluation_report.html').name}")
        print_rt(f"Detailed CSV: {csv_path.name}")

        df_sorted = df.sort_values('composite_score', ascending=False)
        print_rt(f"\nTop 5 compounds:")
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            cdk4 = format_affinity_result(row, "CDK4")
            print_rt(
                f"{i}. Score: {row['composite_score']:.3f}, "
                f"CDK4 pIC50={cdk4['pic50_str']} (conf {cdk4['confidence_str']}), "
                f"SMILES: {row['smiles'][:60]}..."
            )

        print_rt(f"\nLog saved to: {log_path}")

    except Exception as exc:
        print_rt(f"Unexpected error during evaluation: {exc}")
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

if __name__ == "__main__":
    asyncio.run(main())
