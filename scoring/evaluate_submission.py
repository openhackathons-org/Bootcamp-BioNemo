#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
AI-Powered-Drug-Discovery-Bootcamp CDK Inhibitor Evaluation Script

This script evaluates chemical compound submissions for CDK4/6 selectivity
with minimal CDK11 binding. It processes SMILES files and generates
comprehensive evaluation reports.

Usage:
    python evaluate_submission.py <smiles_file> <team_name> [options]

Example:
    python evaluate_submission.py compounds.csv TeamAlpha --output-dir results/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copy2
from Bio.PDB import MMCIFParser, PDBIO
from string import ascii_uppercase, digits
from datetime import datetime
import time
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from endpoint_env import boltz2_api_key, boltz2_client_kwargs, load_openhackathon_env, normalize_boltz2_endpoint

load_openhackathon_env()

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
import sqlite3
import pickle
from tqdm import tqdm
import json
import re
import traceback
from datetime import timezone

# Try to import boltz2 client (optional)
BOLTZ2_AVAILABLE = False
BOLTZ2_IMPORT_ERROR = None

try:
    from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest, PocketConstraint
    BOLTZ2_AVAILABLE = True
    print("✓ Successfully loaded boltz2-python-client")
except ImportError as e:
    BOLTZ2_AVAILABLE = False
    BOLTZ2_IMPORT_ERROR = str(e)
    print("\n" + "="*80)
    print("NOTICE: boltz2-python-client is not installed or cannot be imported.")
    print("The script will use mock predictions for IC50 values.")
    print("\nTo enable actual Boltz2 predictions, install it with:")
    print("  pip install boltz2-python-client")
    print("="*80 + "\n")

# Configuration
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "boltz2_url": normalize_boltz2_endpoint(os.environ.get("BOLTZ2_URL", "http://localhost:8000")),
    "boltz2_api_key": boltz2_api_key(),
    "chembl_data_path": str(BASE_DIR / "chembl_data"),
    "novelty_cutoff": 0.85,
    "top_n_compounds": 25,
    "confidence_threshold": 0.2,
    "weights": {
        "binding_affinity": 0.25,
        "selectivity": 0.15,
        "cdk11_avoidance": 0.20,
        "qed": 0.15,
        "sa": 0.10,
        "pains": 0.10,
        "novelty": 0.05
    }
}

_CHEMBL_FP_CACHE = None


def get_novelty_cutoff() -> float:
    """Return the configured novelty similarity cutoff."""
    return CONFIG.get("novelty_cutoff", 0.85)

# CDK Protein Information
CDK_PROTEIN_INFO = {
    "CDK4": {
        "sequence": "MATSRYEPVAEIGVGAYGTVYKARDPHSGHFVALKSVRVPNGGGGGGGLPISTVREVALLRRLEAF"
                    "EHPNVVRLMDVCATSRTDREIKVTLVFEHVDQDLRTYLDKAPPPGLPAETIKDLMRQFLRGLDFLH"
                    "ANCIVHRDLKPENILVTSGGTVKLADFGLARIYSYQMALTPVVVTLWYRAPEVLLQSTYATPVDMW"
                    "SVGCIFAEMFRRKPLFCGNSEADQLGKIFDLIGLPPEDDWPRDVSLPRGAFPPRGPRPVQSVVPEM"
                    "EESGAQLLLEMLTFNPHKRISAFRALQHSYLHKDEGNPE",
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
        "sequence": "ALQGCRSVEEFQCLNRIEEGTYGVVYRAKDKKTDEIVALKRLKMEKEKEGFPITSLREINTILKAQ"
                    "HPNIVTVREIVVGSNMDKIYIVMNYVEHDLKSLMETMKQPFLPGEVKTLMIQLLRGVKHLHDNWIL"
                    "HRDLKTSNLLLSHAGILKVGDFGLAREYGSPLKAYTPVVVTLWYRAPELLLGAKEYSTAVDMWSVG"
                    "CIFGELLTQKPLFPGKSEIDQINKVFKDLGTPSEKIWPGYSELPAVKKMTFSEHPYNNLRKRFGAL"
                    "LSDQGFDLMNKFLTYFPGRRISAEDGLKHEYFRETPLPIDPSMFPKLVEKY",
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


def load_smiles_file(file_path: str) -> pd.DataFrame:
    """Load SMILES from various file formats"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # Find SMILES column
        smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if smiles_cols:
            df['smiles'] = df[smiles_cols[0]]
        elif 'SMILES' in df.columns:
            df['smiles'] = df['SMILES']
    elif file_path.suffix in ['.txt', '.smi']:
        df = pd.read_csv(file_path, sep='\t', names=['smiles'])
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Add compound IDs if not present
    if 'compound_id' not in df.columns:
        df['compound_id'] = [f"COMP_{i+1:04d}" for i in range(len(df))]
    
    return df


def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate SMILES and add molecular objects"""
    print("Validating SMILES...")
    
    valid_mols = []
    canonical_smiles = []
    is_valid = []
    
    for smiles in tqdm(df['smiles'], desc="Validating"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                is_valid.append(True)
            else:
                valid_mols.append(None)
                canonical_smiles.append(None)
                is_valid.append(False)
        except:
            valid_mols.append(None)
            canonical_smiles.append(None)
            is_valid.append(False)
    
    df['mol'] = valid_mols
    df['canonical_smiles'] = canonical_smiles
    df['is_valid'] = is_valid
    
    print(f"Valid SMILES: {df['is_valid'].sum()}/{len(df)}")
    return df


def calculate_qed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate QED drug-likeness scores"""
    print("Calculating QED scores...")
    
    qed_scores = []
    for mol in tqdm(df['mol'], desc="Calculating QED"):
        if mol is not None:
            qed_scores.append(QED.qed(mol))
        else:
            qed_scores.append(np.nan)
    
    df['qed_score'] = qed_scores
    df['qed_category'] = pd.cut(df['qed_score'], 
                                 bins=[0, 0.25, 0.5, 0.75, 1.0],
                                 labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    return df


def calculate_sa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate synthetic accessibility scores"""
    print("Calculating SA scores...")
    
    # Simple SA score approximation
    sa_scores = []
    for mol in tqdm(df['mol'], desc="Calculating SA"):
        if mol is not None:
            # Simplified SA score based on molecular complexity
            score = 0
            score += min(mol.GetNumHeavyAtoms() / 50, 1) * 2
            score += min(len(Chem.GetSymmSSSR(mol)) / 6, 1) * 2
            score += min(Descriptors.BertzCT(mol) / 1000, 1) * 3
            # Calculate heteroatoms (N, O, S, halogens, etc.)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            score += min(num_heteroatoms / 10, 1) * 1
            score += min(Descriptors.NumRotatableBonds(mol) / 10, 1) * 2
            sa_scores.append(score)
        else:
            sa_scores.append(np.nan)
    
    df['sa_score'] = sa_scores
    df['sa_score_normalized'] = 1 - (df['sa_score'] / 10)  # Normalize and invert
    
    return df


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
    """Return a filesystem-safe version of the provided value."""
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
    """Persist returned structures (if any) as CIF files under the output directory."""
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

    # Fallback: capture any CIF files generated in working directory during prediction
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
    """Return formatted affinity readouts with raw fallbacks and confidence."""
    ic50 = row.get(f"{target}_ic50_nm")
    pic50 = row.get(f"{target}_pic50")
    ic50_raw = row.get(f"{target}_ic50_nm_raw")
    pic50_raw = row.get(f"{target}_pic50_raw")
    confidence = row.get(f"{target}_confidence")
    accepted = row.get(f"{target}_prediction_accepted")

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


def predict_binding_affinity_boltz2(smiles: str, protein_target: str, 
                                   confidence_threshold: float = CONFIG["confidence_threshold"],
                                   verbose: bool = True,
                                   compound_id: Optional[str] = None) -> Dict:
    """Predict IC50 values using Boltz2 Python client"""
    
    start_time = time.time()
    compound_label = compound_id or f"compound_{abs(hash(smiles)) % 10**8}"
    
    print_rt(f"\n{'='*80}")
    print_rt(f"BOLTZ2 PREDICTION REQUEST - {protein_target}")
    print_rt(f"{'='*80}")
    print_rt(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print_rt(f"SMILES: {smiles}")
    print_rt(f"Protein: {protein_target}")
    print_rt(f"Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} aa")
    binding_sites = CDK_PROTEIN_INFO[protein_target].get("binding_site_residues", [])
    if binding_sites:
        print_rt("Binding site residues (≤9 residues):")
        for site in binding_sites:
            print_rt(f"  - {site['residue']} (pos {site['position']})")
    binding_site_summary = ", ".join(
        f"{site['residue']} (pos {site['position']})" for site in binding_sites
    ) if binding_sites else ""
    binding_site_positions = [
        int(site["position"]) for site in binding_sites if "position" in site
    ]
    
    try:
        if BOLTZ2_AVAILABLE:
            print_rt(f"Mode: ACTUAL Boltz2 Python Client")
            print_rt(f"Endpoint: {CONFIG['boltz2_url']}")
            
            # Initialize Boltz2 client
            client = Boltz2Client(**boltz2_client_kwargs(
                CONFIG["boltz2_url"],
                api_key=CONFIG.get("boltz2_api_key", "")
            ))
            
            # Create prediction input
            protein_sequence = CDK_PROTEIN_INFO[protein_target]["sequence"]
            
            # Print input information
            print_rt(f"\nINPUT DETAILS:")
            print_rt(f"  SMILES: {smiles}")
            print_rt(f"  Target: {protein_target}")
            print_rt(f"  Sequence length: {len(protein_sequence)} residues")
            if compound_id:
                print_rt(f"  Compound ID: {compound_id}")
            if binding_site_positions:
                print_rt(f"  Binding site residue indices: {binding_site_positions}")
            
            # Create protein polymer with full sequence
            protein = Polymer(
                id="A",
                molecule_type="protein",
                sequence=protein_sequence
            )
            
            # Create ligand with affinity prediction enabled
            ligand = Ligand(
                id="LIG",
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
            
            # Create prediction request with minimal parameters for speed
            request = PredictionRequest(
                polymers=[protein],
                ligands=[ligand],
                constraints=constraints,
                # Minimal structure parameters for faster execution
                recycling_steps=1,
                sampling_steps=10,  # Minimum allowed value
                diffusion_samples=1,
                # Affinity prediction parameters
                sampling_steps_affinity=50,
                diffusion_samples_affinity=2,
                affinity_mw_correction=True
            )
            
            print_rt(f"\nREQUEST PAYLOAD:")
            try:
                print(json.dumps(request.model_dump(), indent=2, default=str))
            except Exception:
                print_rt(str(request))
            
            # Print detailed request information
            print_rt(f"\nBOLTZ2 REQUEST PARAMETERS:")
            print(f"  Protein ID: {protein.id}")
            print(f"  Protein sequence: {protein_sequence[:50]}...{protein_sequence[-50:]}")
            print(f"  Full sequence length: {len(protein_sequence)} residues")
            print(f"  Ligand ID: {ligand.id}")
            print(f"  Ligand SMILES: {smiles}")
            print(f"  Affinity prediction: {ligand.predict_affinity}")
            print(f"\n  Structure parameters:")
            print(f"    - recycling_steps: {request.recycling_steps}")
            print(f"    - sampling_steps: {request.sampling_steps}")
            print(f"    - diffusion_samples: {request.diffusion_samples}")
            print(f"\n  Affinity parameters:")
            print(f"    - sampling_steps_affinity: {request.sampling_steps_affinity}")
            print(f"    - diffusion_samples_affinity: {request.diffusion_samples_affinity}")
            print(f"    - affinity_mw_correction: {request.affinity_mw_correction}")
            
            # Print full sequence if verbose
            if verbose:
                print(f"\n  Full protein sequence:")
                print(f"    {protein_sequence}")
            
            # Run prediction
            print_rt(f"\nSending request to Boltz2...")
            api_start = time.time()
            
            # Define async function to run prediction
            async def run_prediction():
                return await client.predict(request)
            
            # Run async function in sync context
            prediction = asyncio.run(run_prediction())
            
            api_time = time.time() - api_start
            
            saved_cif_paths = save_prediction_structures(
                prediction,
                compound_label,
                protein_target,
                prediction_start_time=start_time
            )
            
            # Extract results
            print_rt(f"\nBOLTZ2 PREDICTION RESPONSE")
            print_rt(f"{'-'*80}")
            print_rt(f"Status: SUCCESS")
            
            # Print raw response information
            print_rt(f"Response received in {api_time:.2f} seconds")
            print(f"\nRAW BOLTZ2 RESPONSE:")
            print(f"  Response type: {type(prediction)}")
            
            # Print all attributes of the response
            attrs = [attr for attr in dir(prediction) if not attr.startswith('_') and not callable(getattr(prediction, attr))]
            print(f"  Response attributes: {attrs}")
            
            # Try to dump the full response
            if hasattr(prediction, 'model_dump'):
                try:
                    response_dict = prediction.model_dump()
                    print(f"\n  Full response data:")
                    # Print key response fields
                    for key, value in response_dict.items():
                        if key == 'structures':
                            print(f"    - {key}: {len(value) if isinstance(value, list) else type(value)} structure(s)")
                        elif key == 'affinities':
                            print(f"    - {key}: {value}")
                        elif key == 'confidence_scores':
                            print(f"    - {key}: {value}")
                        else:
                            print(f"    - {key}: {str(value)[:100]}...")
                except Exception as e:
                    print(f"  Could not parse response: {e}")
            
            # Print affinity details if available
            if hasattr(prediction, 'affinities') and prediction.affinities:
                print(f"\n  Affinity predictions found for: {list(prediction.affinities.keys())}")
                for ligand_id, affinity in prediction.affinities.items():
                    print(f"\n  Detailed affinity data for {ligand_id}:")
                    if hasattr(affinity, 'model_dump'):
                        affinity_dict = affinity.model_dump()
                        for key, value in affinity_dict.items():
                            print(f"    - {key}: {value}")
                    else:
                        if hasattr(affinity, 'affinity_pic50'):
                            print(f"    - affinity_pic50: {affinity.affinity_pic50}")
                        if hasattr(affinity, 'affinity_probability_binary'):
                            print(f"    - affinity_probability_binary: {affinity.affinity_probability_binary}")
                        if hasattr(affinity, 'affinity_model_predictions'):
                            print(f"    - affinity_model_predictions: {affinity.affinity_model_predictions}")
            
            if saved_cif_paths:
                print_rt(f"\nSaved {len(saved_cif_paths)} CIF file(s):")
                for cif_path in saved_cif_paths:
                    print_rt(f"  - {cif_path}")
            
            # Extract affinity data
            ic50_nm = None
            pic50 = None
            confidence = 0.8  # Default confidence
            
            if hasattr(prediction, 'affinities') and prediction.affinities and "LIG" in prediction.affinities:
                affinity = prediction.affinities["LIG"]
                
                if hasattr(affinity, 'affinity_pic50') and affinity.affinity_pic50:
                    pic50 = affinity.affinity_pic50[0]
                    # Convert pIC50 to IC50 in nM: IC50 (M) = 10^(-pIC50), then convert to nM
                    ic50_nm = 10 ** (-pic50) * 1e9
                    
                    # Use binding probability as confidence
                    if hasattr(affinity, 'affinity_probability_binary') and affinity.affinity_probability_binary:
                        confidence = affinity.affinity_probability_binary[0]
                    
                    # Print affinity results
                    print_rt(f"\nAFFINITY PREDICTION RESULTS:")
                    print_rt(f"  pIC50: {pic50:.2f}")
                    print_rt(f"  IC50: {ic50_nm:.2f} nM")
                    print_rt(f"  Binding probability: {confidence:.1%}")
                    
                    # Interpret the results
                    if pic50 > 7.0:
                        binding_strength = "STRONG"
                    elif pic50 > 5.0:
                        binding_strength = "MODERATE"
                    else:
                        binding_strength = "WEAK"
                    print_rt(f"  Binding strength: {binding_strength}")
                else:
                    print_rt(f"Warning: No pIC50 data in affinity response")
            else:
                print_rt(f"Warning: No affinity data in response, using estimated values")
                # Use mock values based on target
                if protein_target == "CDK4":
                    ic50_nm = np.random.lognormal(np.log(50), 1.5)
                else:  # CDK11
                    ic50_nm = np.random.lognormal(np.log(5000), 1.5)
                pic50 = -np.log10(ic50_nm * 1e-9)
                
                print_rt(f"\nESTIMATED AFFINITY (MOCK):")
                print_rt(f"  pIC50: {pic50:.2f}")
                print_rt(f"  IC50: {ic50_nm:.2f} nM")
            
            print_rt(f"\nConfidence: {confidence:.3f}")
            print_rt(f"Accepted: {confidence >= 0.2}")
            
        else:
            # Fallback to mock if client not available
            print_rt(f"Mode: MOCK predictions (boltz2-python-client not available)")
            
            # Print input information
            print_rt(f"\nINPUT DETAILS:")
            print_rt(f"  SMILES: {smiles}")
            print_rt(f"  Target: {protein_target}")
            print_rt(f"  Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
            
            if protein_target == "CDK4":
                ic50_nm = np.random.lognormal(np.log(50), 1.5)
            else:  # CDK11
                ic50_nm = np.random.lognormal(np.log(5000), 1.5)
            pic50 = -np.log10(ic50_nm * 1e-9)
            confidence = np.random.uniform(0.65, 0.95)
            api_time = np.random.uniform(0.1, 0.5)
            time.sleep(api_time)
            
            print_rt(f"\nBOLTZ2 PREDICTION RESPONSE")
            print_rt(f"{'-'*80}")
            print_rt(f"Status: MOCK")
            
            print_rt(f"\nAFFINITY PREDICTION RESULTS (MOCK):")
            print_rt(f"  pIC50: {pic50:.2f}")
            print_rt(f"  IC50: {ic50_nm:.2f} nM")
            print_rt(f"  Binding probability: {confidence:.1%}")
            
            # Interpret the results
            if pic50 > 7.0:
                binding_strength = "STRONG"
            elif pic50 > 5.0:
                binding_strength = "MODERATE"
            else:
                binding_strength = "WEAK"
            print_rt(f"  Binding strength: {binding_strength}")
            
            print_rt(f"\nConfidence: {confidence:.3f}")
            print_rt(f"Accepted: {confidence >= confidence_threshold}")
            
    except (ImportError, Exception) as e:
        print(f"\nBOLTZ2 PREDICTION RESPONSE")
        print(f"{'-'*80}")
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        
        # Print request details that caused the error
        print(f"\nRequest that caused error:")
        print(f"  SMILES: {smiles}")
        print(f"  Target: {protein_target}")
        print(f"  Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
        
        # Try to get more details about the error
        if hasattr(e, '__dict__'):
            print(f"\nError details: {e.__dict__}")
        if hasattr(e, 'response'):
            print(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"Response text: {getattr(e.response, 'text', 'N/A')}")
            
        print(f"\nFalling back to mock predictions")
        
        # Fallback to mock predictions
        if protein_target == "CDK4":
            ic50_nm = np.random.lognormal(np.log(50), 1.5)
        else:  # CDK11
            ic50_nm = np.random.lognormal(np.log(5000), 1.5)
        pic50 = -np.log10(ic50_nm * 1e-9)
        confidence = 0.75
        api_time = 0.1
        
        print(f"\nESTIMATED AFFINITY (MOCK DUE TO ERROR):")
        print(f"  pIC50: {pic50:.2f}")
        print(f"  IC50: {ic50_nm:.2f} nM")
    
    elapsed_time = time.time() - start_time
    print(f"\nTiming:")
    print(f"  API/Model time: {api_time*1000:.1f} ms" if 'api_time' in locals() else "  API/Model time: N/A")
    print(f"  Total time: {elapsed_time*1000:.1f} ms")
    print(f"{'='*80}\n")
    
    # Ensure pic50 is calculated if only ic50_nm is available
    if pic50 is None and ic50_nm is not None:
        pic50 = -np.log10(ic50_nm * 1e-9)
    
    ic50_raw_value = float(ic50_nm) if ic50_nm is not None else np.nan
    pic50_raw_value = float(pic50) if pic50 is not None else np.nan
    prediction_accepted = confidence >= confidence_threshold

    result_payload = {
        "ic50_nm": ic50_nm if prediction_accepted else np.nan,
        "pic50": pic50 if prediction_accepted else np.nan,
        "confidence": confidence,
        "prediction_accepted": prediction_accepted,
        "binding_site_residues": binding_site_summary,
        "ic50_nm_raw": ic50_raw_value,
        "pic50_raw": pic50_raw_value
    }

    if not prediction_accepted:
        result_payload["reason"] = f"Low confidence ({confidence:.3f} < {confidence_threshold})"

    return result_payload


def calculate_all_binding_affinities(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Calculate IC50 values for all compounds"""
    print("Calculating IC50 values...")
    
    # Initialize statistics
    prediction_stats = {
        "total_predictions": 0,
        "accepted_predictions": 0,
        "rejected_predictions": 0,
        "total_time": 0,
        "by_target": {target: {"accepted": 0, "rejected": 0} for target in ["CDK4", "CDK11"]}
    }
    
    for target in ["CDK4", "CDK11"]:
        df[f"{target}_ic50_nm"] = np.nan
        df[f"{target}_pic50"] = np.nan
        df[f"{target}_confidence"] = np.nan
        df[f"{target}_prediction_accepted"] = False
        df[f"{target}_ic50_nm_raw"] = np.nan
        df[f"{target}_pic50_raw"] = np.nan
        df[f"{target}_binding_site_residues"] = ""
    
    # Temporarily disable verbose output if not requested
    overall_start = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting IC50", disable=verbose):
        if row['mol'] is not None:
            for target in ["CDK4", "CDK11"]:
                # Always show Boltz2 predictions for transparency
                try:
                    result = predict_binding_affinity_boltz2(
                        row['canonical_smiles'], 
                        target,
                        CONFIG["confidence_threshold"],
                        verbose=verbose,
                        compound_id=row.get('compound_id', f"compound_{idx+1}")
                    )
                    
                    prediction_stats["total_predictions"] += 1
                    df.at[idx, f"{target}_ic50_nm_raw"] = result.get("ic50_nm_raw", np.nan)
                    df.at[idx, f"{target}_pic50_raw"] = result.get("pic50_raw", np.nan)
                    df.at[idx, f"{target}_binding_site_residues"] = result.get("binding_site_residues", "")
                    
                    if result["prediction_accepted"]:
                        df.at[idx, f"{target}_ic50_nm"] = result["ic50_nm"]
                        df.at[idx, f"{target}_pic50"] = result["pic50"]
                        df.at[idx, f"{target}_confidence"] = result["confidence"]
                        df.at[idx, f"{target}_prediction_accepted"] = True
                        prediction_stats["accepted_predictions"] += 1
                        prediction_stats["by_target"][target]["accepted"] += 1
                    else:
                        prediction_stats["rejected_predictions"] += 1
                        prediction_stats["by_target"][target]["rejected"] += 1
                except Exception as e:
                    print(f"Error during prediction: {e}")
    
    prediction_stats["total_time"] = time.time() - overall_start
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"BOLTZ2 PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total predictions: {prediction_stats['total_predictions']}")
    print(f"Accepted: {prediction_stats['accepted_predictions']} ({prediction_stats['accepted_predictions']/prediction_stats['total_predictions']*100:.1f}%)")
    print(f"Rejected: {prediction_stats['rejected_predictions']} ({prediction_stats['rejected_predictions']/prediction_stats['total_predictions']*100:.1f}%)")
    print(f"\nBy target:")
    for target in ["CDK4", "CDK11"]:
        stats = prediction_stats["by_target"][target]
        total = stats["accepted"] + stats["rejected"]
        percentage = stats['accepted']/total*100 if total > 0 else 0
        print(f"  {target}: {stats['accepted']}/{total} accepted ({percentage:.1f}%)")
    print(f"\nTotal time: {prediction_stats['total_time']:.2f} seconds")
    print(f"Average time per prediction: {prediction_stats['total_time']/prediction_stats['total_predictions']*1000:.1f} ms")
    print(f"{'='*80}\n")
    
    # Print detailed affinity results for each compound
    print(f"\n{'='*80}")
    print(f"AFFINITY PREDICTION RESULTS PER COMPOUND")
    print(f"{'='*80}")
    for idx, row in df.iterrows():
        if pd.notna(row.get('canonical_smiles')):
            print(f"\nCompound {idx + 1}: {row['canonical_smiles'][:50]}...")
            cdk4 = format_affinity_result(row, "CDK4")
            cdk11 = format_affinity_result(row, "CDK11")
            print(f"  CDK4:  IC50 = {cdk4['ic50_str']} nM{cdk4['suffix']}, pIC50 = {cdk4['pic50_str']}, confidence = {cdk4['confidence_str']}")
            print(f"  CDK11: IC50 = {cdk11['ic50_str']} nM{cdk11['suffix']}, pIC50 = {cdk11['pic50_str']}, confidence = {cdk11['confidence_str']}")
            if pd.notna(row.get('selectivity_ratio')):
                print(f"  Selectivity (CDK11/CDK4,6): {row['selectivity_ratio']:.1f}x")
    print(f"{'='*80}\n")
    
    # Calculate selectivity metrics (CDK4 only, CDK6 removed for simplified hackathon)
    df['on_target_pic50'] = df['CDK4_pic50']
    df['on_target_ic50_nm'] = df['CDK4_ic50_nm']
    df['selectivity_ratio'] = df['CDK11_ic50_nm'] / df['on_target_ic50_nm']
    df['selectivity_ratio'] = df['selectivity_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # CDK11 avoidance score
    def calculate_cdk11_avoidance(ic50):
        if pd.isna(ic50):
            return np.nan
        elif ic50 >= 10000:
            return 1.0
        elif ic50 >= 1000:
            return 0.5 + 0.5 * (ic50 - 1000) / 9000
        else:
            return 0.5 * ic50 / 1000
    
    df['cdk11_avoidance'] = df['CDK11_ic50_nm'].apply(calculate_cdk11_avoidance)
    
    return df


def apply_pains_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PAINS filters and annotate compounds."""
    print("Applying PAINS filters...")

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)

    pains_flags = []
    pains_matches = []

    for mol in tqdm(df['mol'], desc="PAINS"):
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

    n_pains = (df['is_pains'] == True).sum()
    print(f"PAINS-positive compounds: {n_pains}/{len(df)} ({n_pains/len(df)*100:.1f}%)")
    if n_pains > 0:
        print("Top PAINS alerts:")
        alert_counts: Dict[str, int] = {}
        for alerts in df.loc[df['is_pains'] == True, 'pains_alerts']:
            for alert in alerts:
                alert_counts[alert] = alert_counts.get(alert, 0) + 1
        for alert, count in sorted(alert_counts.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"  {alert}: {count}")

    return df


def load_chembl_fingerprints(chembl_path: str) -> Dict[str, object]:
    """Load ChEMBL fingerprints from pickle file with caching."""
    global _CHEMBL_FP_CACHE

    if _CHEMBL_FP_CACHE is not None:
        return _CHEMBL_FP_CACHE

    fp_path = os.path.join(chembl_path, "chembl_fingerprints.pkl")

    if os.path.exists(fp_path):
        print(f"Loading ChEMBL fingerprints from {fp_path}...")
        with open(fp_path, 'rb') as f:
            _CHEMBL_FP_CACHE = pickle.load(f)
    else:
        print("Warning: ChEMBL fingerprints not found. Using empty reference set.")
        _CHEMBL_FP_CACHE = {}

    return _CHEMBL_FP_CACHE


def calculate_novelty_scores(
    df: pd.DataFrame,
    chembl_path: str = CONFIG["chembl_data_path"],
    similarity_cutoff: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate novelty scores by comparing to ChEMBL database."""

    print("Calculating novelty scores...")

    cutoff = similarity_cutoff if similarity_cutoff is not None else get_novelty_cutoff()

    reference_fps = load_chembl_fingerprints(chembl_path)

    if not reference_fps:
        print("No reference compounds available. Setting all compounds as novel.")
        df['max_chembl_similarity'] = 0.0
        df['is_novel'] = True
        df['novelty_score'] = 1.0
        return df

    query_fps: List[Optional[DataStructs.ExplicitBitVect]] = []
    for mol in df['mol']:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            query_fps.append(fp)
        else:
            query_fps.append(None)

    max_similarities: List[float] = []
    reference_fp_values = list(reference_fps.values())

    for query_fp in tqdm(query_fps, desc="Computing novelty"):
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
    df['novelty_score'] = 1.0 - df['max_chembl_similarity'].fillna(cutoff).clip(upper=cutoff) / cutoff
    df.loc[~df['is_novel'], 'novelty_score'] = 0.0

    n_novel = df['is_novel'].sum()
    print(f"Novel compounds: {n_novel}/{len(df)} ({n_novel/len(df)*100:.1f}%) using cutoff {cutoff:.2f}")

    return df


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all scores to 0-1 range"""
    score_mappings = {
        'on_target_pic50': lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x.fillna(0.5),
        'selectivity_ratio': lambda x: np.minimum(x / 50, 1.0),
        'cdk11_avoidance': lambda x: x,
        'qed_score': lambda x: x,
        'sa_score_normalized': lambda x: x,
        'pains_score': lambda x: x,
        'novelty_score': lambda x: x
    }
    
    for col, norm_func in score_mappings.items():
        if col in df.columns:
            df[f'{col}_norm'] = norm_func(df[col].fillna(df[col].median()))
    
    return df


def calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted composite scores"""
    weights = CONFIG["weights"]
    
    weight_mapping = {
        'binding_affinity': 'on_target_pic50_norm',
        'selectivity': 'selectivity_ratio_norm',
        'cdk11_avoidance': 'cdk11_avoidance_norm',
        'qed': 'qed_score_norm',
        'sa': 'sa_score_normalized_norm',
        'pains': 'pains_score_norm',
        'novelty': 'novelty_score_norm'
    }
    
    df['composite_score'] = 0
    for weight_key, col_name in weight_mapping.items():
        if col_name in df.columns and weight_key in weights:
            df['composite_score'] += weights[weight_key] * df[col_name]
    
    df['rank'] = df['composite_score'].rank(ascending=False, method='min')
    
    return df


# def create_evaluation_dashboard(df: pd.DataFrame, team_name: str, output_dir: Path):
#     """Generate evaluation dashboard"""
#     print("Creating evaluation dashboard...")
    
#     fig = plt.figure(figsize=(20, 24))
    
#     # 1. Score distributions
#     ax1 = plt.subplot(4, 2, 1)
#     score_cols = ['qed_score', 'sa_score_normalized', 'pains_score', 'novelty_score']
#     valid_cols = [col for col in score_cols if col in df.columns]
#     if valid_cols:
#         df[valid_cols].boxplot(ax=ax1)
#         ax1.set_title('Score Distributions', fontsize=14)
#         ax1.set_ylabel('Score (0-1)')
#         plt.xticks(rotation=45)
    
    # 2. IC50 comparison
    ax2 = plt.subplot(4, 2, 2)
    ic50_cols = ['CDK4_ic50_nm', 'CDK11_ic50_nm']
    valid_ic50_cols = [col for col in ic50_cols if col in df.columns]
    if valid_ic50_cols:
        ic50_data = df[valid_ic50_cols].median()
        ic50_data.plot(kind='bar', ax=ax2, color=['green', 'blue', 'red'], logy=True)
        ax2.set_title('Median IC50 Values', fontsize=14)
        ax2.set_ylabel('IC50 (nM, log scale)')
        ax2.set_xlabel('Target')
        plt.xticks(rotation=45)
    
#     # 3. Top compounds scatter
#     ax3 = plt.subplot(4, 2, 3)
#     top_25 = df.nsmallest(25, 'rank')
#     if 'qed_score' in df.columns and 'on_target_pic50' in df.columns:
#         valid = top_25[['qed_score', 'on_target_pic50', 'composite_score']].dropna()
#         if not valid.empty:
#             scatter = ax3.scatter(valid['qed_score'], valid['on_target_pic50'], 
#                                  c=valid['composite_score'], s=100, cmap='viridis')
#             ax3.set_xlabel('QED Score')
#             ax3.set_ylabel('On-target pIC50')
#             ax3.set_title('Top 25 Compounds: QED vs Potency', fontsize=14)
#             plt.colorbar(scatter, ax=ax3, label='Composite Score')
#         else:
#             ax3.set_visible(False)
    
#     # 4. Novelty distribution
#     ax4 = plt.subplot(4, 2, 4)
#     if 'max_chembl_similarity' in df.columns:
#         valid_novelty = df['max_chembl_similarity'].dropna()
#         if not valid_novelty.empty:
#             valid_novelty.hist(bins=30, ax=ax4, alpha=0.7)
#             novelty_cutoff = get_novelty_cutoff()
#             ax4.axvline(novelty_cutoff, color='red', linestyle='--', 
#                         label=f'Cutoff ({novelty_cutoff:.2f})')
#             ax4.set_xlabel('Max ChEMBL Similarity')
#             ax4.set_ylabel('Count')
#             ax4.set_title('Novelty Distribution', fontsize=14)
#             ax4.legend()
#         else:
#             ax4.set_visible(False)
    
#     # 5. Composite score distribution
#     ax5 = plt.subplot(4, 2, 5)
#     if 'composite_score' in df.columns:
#         df['composite_score'].hist(bins=50, ax=ax5, alpha=0.7, color='purple')
#         ax5.axvline(df['composite_score'].quantile(0.975), color='green', linestyle='--', 
#                     label='Top 2.5%')
#         ax5.set_xlabel('Composite Score')
#         ax5.set_ylabel('Count')
#         ax5.set_title('Composite Score Distribution', fontsize=14)
#         ax5.legend()
    
#     # 6. Selectivity vs CDK11 avoidance
#     ax6 = plt.subplot(4, 2, 6)
#     if 'selectivity_ratio' in df.columns and 'cdk11_avoidance' in df.columns:
#         ax6.scatter(df['selectivity_ratio'], df['cdk11_avoidance'], alpha=0.5)
#         ax6.set_xlabel('Selectivity Ratio')
#         ax6.set_ylabel('CDK11 Avoidance Score')
#         ax6.set_title('Selectivity vs CDK11 Avoidance', fontsize=14)
#         ax6.set_xlim(0, 100)
    
#     # 7. Drug properties
#     ax7 = plt.subplot(4, 2, 7)
#     if 'mol' in df.columns:
#         mw = [Descriptors.MolWt(mol) if mol else np.nan for mol in df['mol']]
#         logp = [Descriptors.MolLogP(mol) if mol else np.nan for mol in df['mol']]
#         valid_idx = ~(pd.isna(mw) | pd.isna(logp))
#         if valid_idx.sum() > 0:
#             ax7.scatter(np.array(mw)[valid_idx], np.array(logp)[valid_idx], alpha=0.5)
#             ax7.set_xlabel('Molecular Weight')
#             ax7.set_ylabel('LogP')
#             ax7.set_title('Drug-like Properties', fontsize=14)
#             ax7.axvline(500, color='red', linestyle='--', alpha=0.5)
#             ax7.axhline(5, color='red', linestyle='--', alpha=0.5)
    
#     # 8. Score breakdown for top compounds
#     ax8 = plt.subplot(4, 2, 8)
#     top_10 = df.nsmallest(10, 'rank')
#     if len(top_10) > 0:
#         score_components = []
#         for weight_key in CONFIG["weights"]:
#             col_map = {
#                 'binding_affinity': 'on_target_pic50_norm',
#                 'selectivity': 'selectivity_ratio_norm',
#                 'cdk11_avoidance': 'cdk11_avoidance_norm',
#                 'qed': 'qed_score_norm',
#                 'sa': 'sa_score_normalized_norm',
#                 'pains': 'pains_score_norm',
#                 'novelty': 'novelty_score_norm'
#             }
#             if col_map.get(weight_key) in df.columns:
#                 score_components.append(
#                     top_10[col_map[weight_key]] * CONFIG["weights"][weight_key]
#                 )
        
#         if score_components:
#             score_df = pd.DataFrame(score_components).T
#             score_df.columns = list(CONFIG["weights"].keys())[:len(score_components)]
#             score_df.plot(kind='bar', stacked=True, ax=ax8)
#             ax8.set_title('Score Breakdown - Top 10 Compounds', fontsize=14)
#             ax8.set_xlabel('Compound Rank')
#             ax8.set_ylabel('Score Contribution')
#             ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     dashboard_path = output_dir / f"{team_name}_dashboard.png"
#     plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Dashboard saved to {dashboard_path}")


# def generate_report(df: pd.DataFrame, team_name: str, output_dir: Path):
#     """Generate evaluation report files"""
#     print("Generating evaluation report...")
    
#     # Save top compounds
#     top_25 = df.nsmallest(CONFIG['top_n_compounds'], 'rank')
#     summary_cols = [
#         'rank', 'compound_id', 'canonical_smiles', 'composite_score',
#         'on_target_pic50', 'on_target_ic50_nm', 'selectivity_ratio', 
#         'cdk11_avoidance', 'qed_score', 'sa_score', 'pains_score', 'novelty_score',
#         'pains_alerts'
#     ]
#     valid_cols = [col for col in summary_cols if col in top_25.columns]
    
#     top_compounds_path = output_dir / f"{team_name}_top_compounds.csv"
#     top_25[valid_cols].to_csv(top_compounds_path, index=False)
    
#     # Save full results
#     full_results_path = output_dir / f"{team_name}_full_results.csv"
#     df.to_csv(full_results_path, index=False)
    
#     # Generate summary statistics
#     summary_path = output_dir / f"{team_name}_summary.txt"
#     with open(summary_path, 'w') as f:
#         f.write(f"Evaluation Summary for {team_name}\n")
#         f.write(f"{'='*50}\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
#         f.write(f"Total compounds submitted: {len(df)}\n")
#         f.write(f"Valid compounds: {df['is_valid'].sum()}\n")
#         f.write(f"Novel compounds: {df['is_novel'].sum() if 'is_novel' in df.columns else 'N/A'}\n\n")
        
        f.write("Top 5 Compounds:\n")
        for idx, row in top_25.head(5).iterrows():
            rank_value = row['rank']
            rank_display = int(rank_value) if pd.notna(rank_value) else 'N/A'
            f.write(f"{rank_display}. {row['compound_id']} - Score: {row['composite_score']:.3f}\n")
            cdk4 = format_affinity_result(row, "CDK4")
            cdk11 = format_affinity_result(row, "CDK11")
            f.write(
                f"   CDK4: IC50={cdk4['ic50_str']} nM{cdk4['suffix']}, "
                f"pIC50={cdk4['pic50_str']}, confidence={cdk4['confidence_str']}\n"
            )
            f.write(
                f"   CDK11: IC50={cdk11['ic50_str']} nM{cdk11['suffix']}, "
                f"pIC50={cdk11['pic50_str']}, confidence={cdk11['confidence_str']}\n"
            )
        
#         f.write(f"\nMedian Scores:\n")
#         for score_name in ['qed_score', 'sa_score_normalized', 'pains_score', 
#                           'novelty_score', 'composite_score']:
#             if score_name in df.columns:
#                 f.write(f"  {score_name}: {df[score_name].median():.3f}\n")
    
#     print(f"Report files saved to {output_dir}")


def generate_json_output(df: pd.DataFrame, team_name: str, output_dir: Path) -> None:
    """Generate JSON output with top compound score and median score"""
    
    # Calculate metrics
    top_compound_score = float(df['composite_score'].max())
    median_score = float(df['composite_score'].median())
    
    # Create result JSON
    result = {
        "team_name": team_name,
        "Top_compound_score": top_compound_score,
        "Median_score": median_score,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_compounds": len(df)
    }
    
    # Save JSON output
    json_path = output_dir / f"{team_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"JSON results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI-Powered-Drug-Discovery-Bootcamp CDK inhibitor submissions"
    )
    parser.add_argument("smiles_file", help="Path to SMILES file (CSV, TXT, or SMI)")
    parser.add_argument("team_name", help="Team name for identification")
    parser.add_argument("--output-dir", default="evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--chembl-path", default="./chembl_data",
                        help="Path to ChEMBL database")
    parser.add_argument("--skip-pains", action="store_true",
                        help="Skip PAINS filtering")
    parser.add_argument("--skip-novelty", action="store_true",
                        help="Skip novelty assessment")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Minimum confidence for IC50 predictions")
    parser.add_argument("--boltz2-url", default=CONFIG["boltz2_url"],
                        help="Base URL for the Boltz2 NIM endpoint")
    parser.add_argument("--novelty-cutoff", type=float, default=CONFIG["novelty_cutoff"],
                        help="Maximum Tanimoto similarity to ChEMBL allowed before a compound is considered non-novel")
    parser.add_argument("--skip-boltz2", action="store_true",
                        help="Skip Boltz2 affinity predictions and use placeholder values")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed Boltz2 prediction requests and responses")
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    CONFIG["chembl_data_path"] = args.chembl_path
    CONFIG["confidence_threshold"] = args.confidence_threshold
    CONFIG["novelty_cutoff"] = args.novelty_cutoff
    CONFIG["boltz2_url"] = args.boltz2_url
    
    # Create output directory
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    team_slug = sanitize_filename(args.team_name)
    output_dir = base_output_dir / f"{team_slug}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["current_output_dir"] = str(output_dir)
    
    log_path = output_dir / f"{args.team_name}_evaluation.log"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeIO(original_stdout, log_file)
    sys.stderr = TeeIO(original_stderr, log_file)
    
    try:
        print(f"Evaluation log will be saved to: {log_path}")
        print(f"Evaluating submission for {args.team_name}")
        print(f"Input file: {args.smiles_file}")
        print(f"Output directory: {output_dir}")
        
        smiles_path = Path(args.smiles_file)
        if not smiles_path.exists():
            print(f"Error: Input file '{smiles_path}' not found.")
            return 1
        
        if not args.skip_novelty:
            chembl_cache = Path(args.chembl_path) / "chembl_fingerprints.pkl"
            if not chembl_cache.exists():
                print(f"Error: Required ChEMBL fingerprint cache not found at {chembl_cache}.")
                print("Run the ChEMBL fingerprint preparation script or supply --skip-novelty.")
                return 1
        
        # Load and validate SMILES
        df = load_smiles_file(str(smiles_path))
        df = validate_smiles(df)
        
        if df['is_valid'].sum() == 0:
            print("Error: No valid SMILES found in submission!")
            return 1
        
        # Filter to valid compounds only
        df = df[df['is_valid']].copy()
        
        # Calculate properties
        df = calculate_qed_scores(df)
        df = calculate_sa_scores(df)
        
        # Calculate binding affinities
        if args.skip_boltz2:
            for target in ["CDK4", "CDK11"]:
                df[f"{target}_ic50_nm"] = np.nan
                df[f"{target}_pic50"] = np.nan
                df[f"{target}_confidence"] = np.nan
                df[f"{target}_prediction_accepted"] = False
                df[f"{target}_ic50_nm_raw"] = np.nan
                df[f"{target}_pic50_raw"] = np.nan
                binding_sites = CDK_PROTEIN_INFO[target].get("binding_site_residues", [])
                binding_site_summary = ", ".join(
                    f"{site['residue']} (pos {site['position']})" for site in binding_sites
                ) if binding_sites else ""
                df[f"{target}_binding_site_residues"] = binding_site_summary
            df['on_target_pic50'] = np.nan
            df['on_target_ic50_nm'] = np.nan
            df['selectivity_ratio'] = np.nan
            df['cdk11_avoidance'] = np.nan
        else:
            df = calculate_all_binding_affinities(df, verbose=args.verbose)
        
        # Apply PAINS filtering
        if not args.skip_pains:
            df = apply_pains_filter(df)
        else:
            df['is_pains'] = False
            df['pains_score'] = 1.0
        
        # Keep only non-PAINS compounds for scoring
        df = df[df['is_pains'] != True].copy()
        if len(df) == 0:
            print("Error: All compounds flagged as PAINS. No compounds to score.")
            return 1
    
        # Calculate novelty
        if not args.skip_novelty:
            df = calculate_novelty_scores(df, similarity_cutoff=CONFIG["novelty_cutoff"])
        else:
            df['novelty_score'] = 0.5  # Neutral score
        
        # Normalize and calculate composite scores
        df = normalize_scores(df)
        df = calculate_composite_scores(df)
        
        # Generate outputs
        create_evaluation_dashboard(df, args.team_name, output_dir)
        generate_report(df, args.team_name, output_dir)
        
        # Print summary
        print("\nEvaluation Complete!")
        print(f"Total valid compounds: {len(df)}")
        print(f"Top compound score: {df['composite_score'].max():.3f}")
        print(f"Median score: {df['composite_score'].median():.3f}")
        
        if 'selectivity_ratio' in df.columns:
            good_selectivity = (df['selectivity_ratio'] > 10).sum()
            print(f"Compounds with >10-fold selectivity: {good_selectivity}")
        
        print(f"\nResults saved to {output_dir}")
        return 0
    except Exception as exc:
        print(f"Unexpected error during evaluation: {exc}")
        traceback.print_exc()
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    return 0  # Should not be reached

if __name__ == "__main__":
    sys.exit(main())
