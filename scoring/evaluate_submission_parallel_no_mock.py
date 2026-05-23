#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
Parallel AI-Powered-Drug-Discovery-Bootcamp Evaluation Script - NO MOCK PREDICTIONS
Uses only real Boltz2 predictions from multiple endpoints

This script evaluates chemical compound submissions using parallel processing
with multiple Boltz2 NIM endpoints. It NEVER uses mock predictions and will
fail if no valid Boltz2 endpoints are available.

Usage:
    python evaluate_submission_parallel_no_mock.py <smiles_file> <team_name> [options]
"""

import pandas as pd
import numpy as np
import time
import json
import argparse
import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
    RDKIT_AVAILABLE = True
except ImportError:
    print("ERROR: RDKit not available. Install with: pip install rdkit-pypi")
    sys.exit(1)

# Boltz2 client import - REQUIRED
try:
    from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest
    BOLTZ2_AVAILABLE = True
    print("✓ boltz2-python-client is available")
except ImportError:
    print("ERROR: boltz2-python-client not available. Install with: pip install boltz2-python-client")
    print("This script requires actual Boltz2 predictions and cannot run without the client.")
    sys.exit(1)

# Other imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Configuration
CONFIG = {
    "endpoints": boltz2_endpoint_urls("8000,8001,8002"),
    "max_workers": 6,  # Maximum concurrent workers
    "api_timeout": 300,  # Timeout for API calls in seconds
    "confidence_threshold": 0.5,
    "weights": {
        "binding_affinity": 0.35,
        "cdk11_avoidance": 0.25,
        "qed": 0.15,
        "sa_score": 0.15,
        "toxicity": 0.05,
        "novelty": 0.05
    },
    "chembl_data_path": "./openhackathon/chembl_data"
}

# Protein sequences for CDK targets
CDK_PROTEIN_INFO = {
    "CDK4": {
        "sequence": "MATSRYEPVAEIGVGAYGTVYKARDPHSGHFVALKSVRVPNGGGGGGGLPISTVREVALLRRLEAFEHPNVVRLMDVCATSRTDREIKVTLVFEHVDQDLRTYLDKAPPPGLPAETIKDLMRQFLRGLDFLHANCIVHRDLKPENILVTSGGTVKLADFGLARIYSYQMALTPVVVTLWYRAPEVLLQSTYATPVDMWSVGCIFAEMFRRKPLFCGNSEADQLGKIFDLIGLPPEDDWPRDVSLPRGAFPPRGPRPVQSVVPEMEESGAQLLLEMLTFNPHKRISAFRALQHSYLHKDEGNPE"
    },
    "CDK11": {
        "sequence": "ALQGCRSVEEFQCLNRIEEGTYGVVYRAKDKKTDEIVALKRLKMEKEKEGFPITSLREINTILKAQHPNIVTVREIVVGSNMDKIYIVMNYVEHDLKSLMETMKQPFLPGEVKTLMIQLLRGVKHLHDNWILHRDLKTSNLLLSHAGILKVGDFGLAREYGSPLKAYTPVVVTLWYRAPELLLGAKEYSTAVDMWSVGCIFAEMFRRKPLFPGKSEIDQINKVFKDLGTPSEKIWPGYSELPAVKKMTFSEHPYNNLRKRFGALLSDQGFDLMNKFLTYFPGRRISAEDGLKHEYFRETPLPIDPSMFPKLVEKY"
    }
}

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
            # Create client and test with a simple request
            client = Boltz2Client(**boltz2_client_kwargs(endpoint, timeout=30))
            
            # Test with a minimal prediction request to verify the endpoint works
            test_protein = Polymer(id="A", sequence="MKLLKWAWLLLSKASSAHDKA", molecule_type="protein")  # Short test sequence
            test_ligand = Ligand(id="LIG1", smiles="C", predict_affinity=True)  # Simple methane
            
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
                
        except Exception as e:
            print(f"✗ Health check failed for {endpoint}: {e}")
            return False
    
    async def initialize(self):
        """Initialize the endpoint pool by checking health of all endpoints"""
        print(f"Initializing endpoint pool with {len(self.endpoints)} endpoints...")
        for endpoint in self.endpoints:
            is_healthy = await self.check_health(endpoint)
            if is_healthy:
                self.healthy_endpoints.append(endpoint)
        
        if not self.healthy_endpoints:
            raise Exception("❌ NO HEALTHY BOLTZ2 ENDPOINTS FOUND! Cannot proceed without actual predictions.")
        
        print(f"✓ Endpoint pool initialized with {len(self.healthy_endpoints)} healthy endpoints")
    
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

async def predict_binding_affinity_boltz2_real_only(smiles: str, protein_target: str, 
                                                   endpoint_pool: EndpointPool,
                                                   confidence_threshold: float = CONFIG["confidence_threshold"],
                                                   verbose: bool = True,
                                                   worker_id: int = 0) -> Dict:
    """REAL Boltz2 predictions only - NO MOCK FALLBACKS"""
    
    start_time = time.time()
    endpoint, client = endpoint_pool.get_next_endpoint()
    
    if verbose:
        print_rt(f"\n{'='*60}")
        print_rt(f"[Worker {worker_id}] REAL BOLTZ2 PREDICTION - {protein_target}")
        print_rt(f"[Worker {worker_id}] Endpoint: {endpoint}")
        print_rt(f"[Worker {worker_id}] {'='*60}")
        print_rt(f"[Worker {worker_id}] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print_rt(f"[Worker {worker_id}] SMILES: {smiles}")
        print_rt(f"[Worker {worker_id}] Protein: {protein_target}")
        print_rt(f"[Worker {worker_id}] Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} aa")
    
    # Check if we have a valid endpoint and client
    if not client or not endpoint:
        error_msg = f"No valid Boltz2 client/endpoint available"
        if verbose:
            print_rt(f"[Worker {worker_id}] ❌ PREDICTION FAILED - NO VALID ENDPOINT")
            print_rt(f"[Worker {worker_id}] Error: {error_msg}")
        raise RuntimeError(error_msg)
    
    try:
        if verbose:
            print_rt(f"[Worker {worker_id}] Mode: REAL Boltz2 Prediction (NO MOCK)")
            print_rt(f"[Worker {worker_id}] Client type: {type(client)}")
        
        protein_sequence = CDK_PROTEIN_INFO[protein_target]["sequence"]
        
        if verbose:
            print_rt(f"[Worker {worker_id}] INPUT:")
            print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
            print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
            print_rt(f"[Worker {worker_id}]   Sequence: {len(protein_sequence)} residues")
        
        # Create prediction request with proper ID formats
        # Polymer ID must be single letter (A-Z) or 4 alphanumeric chars
        protein_id = protein_target[0]  # Use first letter of target (C for CDK4/CDK11)
        ligand_id = f"L{hash(smiles) % 1000:03d}"  # L + 3-digit number = 4 chars
        
        protein = Polymer(id=protein_id, sequence=protein_sequence, molecule_type="protein")
        ligand = Ligand(
            id=ligand_id,
            smiles=smiles,
            predict_affinity=True
        )
        
        request = PredictionRequest(
            polymers=[protein],
            ligands=[ligand],
            recycling_steps=3,
            sampling_steps=10,
            diffusion_samples=1,
            sampling_steps_affinity=50,
            diffusion_samples_affinity=2,
            affinity_mw_correction=True
        )
        
        if verbose:
            print_rt(f"[Worker {worker_id}] REQUEST PARAMS:")
            print_rt(f"[Worker {worker_id}]   Protein ID: {protein.id}")
            print_rt(f"[Worker {worker_id}]   Ligand ID: {ligand.id}")
            print_rt(f"[Worker {worker_id}]   Sampling steps: {request.sampling_steps}")
            print_rt(f"[Worker {worker_id}]   Affinity steps: {request.sampling_steps_affinity}")
            print_rt(f"[Worker {worker_id}] Sending to Boltz2...")
        
        # Make async prediction
        prediction = await client.predict(request)
        api_time = time.time() - start_time
        
        if verbose:
            print_rt(f"[Worker {worker_id}] ✅ SUCCESS in {api_time:.2f} seconds")
        
        # Extract affinity data - MUST have real data
        pic50 = None
        confidence = None
        ic50_nm = None
        
        if hasattr(prediction, 'affinities') and prediction.affinities:
            # Use the same ligand_id we created above
            if ligand_id in prediction.affinities:
                affinity = prediction.affinities[ligand_id]
                if hasattr(affinity, 'affinity_pic50') and affinity.affinity_pic50:
                    pic50 = affinity.affinity_pic50[0]
                    ic50_nm = 10**(9 - pic50)
                
                if hasattr(affinity, 'affinity_probability_binary') and affinity.affinity_probability_binary:
                    confidence = affinity.affinity_probability_binary[0]
        
        # Validate we got real prediction data
        if pic50 is None or confidence is None or ic50_nm is None:
            error_msg = f"Boltz2 returned incomplete affinity data"
            if verbose:
                print_rt(f"[Worker {worker_id}] ❌ INCOMPLETE RESPONSE")
                print_rt(f"[Worker {worker_id}] pic50: {pic50}, confidence: {confidence}, ic50_nm: {ic50_nm}")
                if hasattr(prediction, 'affinities'):
                    print_rt(f"[Worker {worker_id}] Affinity keys: {list(prediction.affinities.keys()) if prediction.affinities else 'None'}")
            raise RuntimeError(error_msg)
        
        # We have valid data
        if verbose:
            print_rt(f"[Worker {worker_id}] REAL AFFINITY RESULTS:")
            print_rt(f"[Worker {worker_id}]   pIC50: {pic50:.2f}")
            print_rt(f"[Worker {worker_id}]   IC50: {ic50_nm:.2f} nM")
            print_rt(f"[Worker {worker_id}]   Confidence: {confidence:.3f}")
            print_rt(f"[Worker {worker_id}]   Accepted: {confidence >= confidence_threshold}")
            
            binding_strength = "STRONG" if pic50 > 7.0 else "MODERATE" if pic50 > 5.0 else "WEAK"
            print_rt(f"[Worker {worker_id}]   Strength: {binding_strength}")
        
        total_time = time.time() - start_time
        
        return {
            'ic50_nm': ic50_nm,
            'pic50': pic50,
            'confidence': confidence,
            'accepted': confidence >= confidence_threshold,
            'api_time': api_time,
            'total_time': total_time,
            'endpoint': endpoint,
            'worker_id': worker_id,
            'real_prediction': True  # Flag to confirm this is real data
        }
    
    except Exception as e:
        if verbose:
            print_rt(f"[Worker {worker_id}] ❌ PREDICTION FAILED")
            print_rt(f"[Worker {worker_id}] Error: {type(e).__name__}: {str(e)}")
            print_rt(f"[Worker {worker_id}] SMILES: {smiles}")
            print_rt(f"[Worker {worker_id}] Target: {protein_target}")
            print_rt(f"[Worker {worker_id}] NO MOCK FALLBACK - FAILING")
        
        # Re-raise - no fallback
        raise e

async def calculate_all_binding_affinities_real_only(df: pd.DataFrame, 
                                                    endpoint_pool: EndpointPool,
                                                    max_workers: int = 6,
                                                    verbose: bool = True) -> pd.DataFrame:
    """Calculate binding affinities using only real Boltz2 predictions"""
    print_rt(f"\n🔬 REAL BOLTZ2 PREDICTIONS ONLY")
    print_rt(f"Using {len(endpoint_pool.healthy_endpoints)} endpoints with {max_workers} workers")
    print_rt(f"NO MOCK PREDICTIONS - WILL FAIL IF ENDPOINTS UNAVAILABLE")
    
    # Prepare prediction tasks
    tasks = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        for target in ["CDK4", "CDK11"]:
            tasks.append((idx, smiles, target))
    
    results = {}
    completed_tasks = 0
    total_tasks = len(tasks)
    failed_tasks = 0
    
    print_rt(f"Total predictions to make: {total_tasks}")
    
    # Use semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(max_workers)
    
    async def worker_task(task_data, worker_id):
        async with semaphore:
            idx, smiles, target = task_data
            try:
                result = await predict_binding_affinity_boltz2_real_only(
                    smiles, target, endpoint_pool, CONFIG["confidence_threshold"], verbose, worker_id
                )
                return (idx, target, result, None)
            except Exception as e:
                return (idx, target, None, str(e))
    
    # Start all tasks
    worker_tasks = []
    for i, task_data in enumerate(tasks):
        worker_id = i % max_workers
        worker_tasks.append(worker_task(task_data, worker_id))
    
    # Process results as they complete
    for task in asyncio.as_completed(worker_tasks):
        idx, target, result, error = await task
        
        if idx not in results:
            results[idx] = {}
        
        if result is not None:
            results[idx][target] = result
            completed_tasks += 1
            if verbose:
                progress = (completed_tasks / total_tasks) * 100
                print_rt(f"✅ {completed_tasks}/{total_tasks} ({progress:.1f}%) - {target} for compound {idx}")
        else:
            failed_tasks += 1
            print_rt(f"❌ FAILED: {target} for compound {idx} - {error}")
            # Since we don't use mock predictions, we'll leave this data missing
    
    print_rt(f"\n📊 PREDICTION SUMMARY:")
    print_rt(f"Successful: {completed_tasks}/{total_tasks}")
    print_rt(f"Failed: {failed_tasks}/{total_tasks}")
    
    if failed_tasks > 0:
        print_rt(f"⚠️  {failed_tasks} predictions failed - NO MOCK DATA SUBSTITUTED")
        print_rt(f"Consider checking endpoint health or reducing load")
    
    # Apply results to dataframe
    for idx in results:
        for target in ["CDK4", "CDK11"]:
            if target in results[idx]:
                result = results[idx][target]
                df.loc[idx, f'{target}_ic50_nm'] = result['ic50_nm']
                df.loc[idx, f'{target}_pic50'] = result['pic50']
                df.loc[idx, f'{target}_confidence'] = result['confidence']
                df.loc[idx, f'{target}_accepted'] = result['accepted']
                df.loc[idx, f'{target}_api_time'] = result['api_time']
                df.loc[idx, f'{target}_endpoint'] = result['endpoint']
                df.loc[idx, f'{target}_real_prediction'] = result['real_prediction']
            else:
                # Mark missing data explicitly
                df.loc[idx, f'{target}_ic50_nm'] = np.nan
                df.loc[idx, f'{target}_pic50'] = np.nan
                df.loc[idx, f'{target}_confidence'] = np.nan
                df.loc[idx, f'{target}_accepted'] = False
                df.loc[idx, f'{target}_real_prediction'] = False
    
    # Calculate selectivity metrics (only for rows with complete data)
    valid_rows = df.dropna(subset=['CDK4_pic50', 'CDK11_pic50'])
    
    for idx in valid_rows.index:
        df.loc[idx, 'on_target_pic50'] = df.loc[idx, 'CDK4_pic50']
        df.loc[idx, 'cdk11_avoidance'] = max(0, df.loc[idx, 'on_target_pic50'] - df.loc[idx, 'CDK11_pic50'])
        df.loc[idx, 'selectivity_ratio'] = df.loc[idx, 'on_target_pic50'] / (df.loc[idx, 'CDK11_pic50'] + 1e-6)
    
    print_rt(f"\n🎯 REAL PREDICTION RESULTS:")
    valid_count = len(valid_rows)
    print_rt(f"Compounds with complete real predictions: {valid_count}/{len(df)}")
    
    if valid_count > 0:
        for idx in valid_rows.index:
            print_rt(f"Compound {idx + 1} (REAL DATA):")
            print_rt(f"  CDK4:  IC50 = {df.loc[idx, 'CDK4_ic50_nm']:.1f} nM, pIC50 = {df.loc[idx, 'CDK4_pic50']:.2f}")
            print_rt(f"  CDK11: IC50 = {df.loc[idx, 'CDK11_ic50_nm']:.1f} nM, pIC50 = {df.loc[idx, 'CDK11_pic50']:.2f}")
            print_rt(f"  Selectivity: {df.loc[idx, 'selectivity_ratio']:.2f}")
    
    return df

def calculate_qed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate QED scores for compounds"""
    print_rt("Calculating QED scores...")
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
    print_rt("Calculating Synthetic Accessibility scores...")
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
    """Normalize all scores to 0-1 range (only for compounds with real predictions)"""
    print_rt("Normalizing scores...")
    
    # Only normalize for rows with complete binding affinity data
    valid_rows = df.dropna(subset=['on_target_pic50', 'cdk11_avoidance'])
    
    if len(valid_rows) > 0:
        # Binding affinity: Higher pIC50 is better
        df.loc[valid_rows.index, 'binding_affinity_normalized'] = (
            (valid_rows['on_target_pic50'] - valid_rows['on_target_pic50'].min()) / 
            (valid_rows['on_target_pic50'].max() - valid_rows['on_target_pic50'].min() + 1e-6)
        )
        
        # CDK11 avoidance: Higher is better
        df.loc[valid_rows.index, 'cdk11_avoidance_normalized'] = (
            (valid_rows['cdk11_avoidance'] - valid_rows['cdk11_avoidance'].min()) / 
            (valid_rows['cdk11_avoidance'].max() - valid_rows['cdk11_avoidance'].min() + 1e-6)
        )
    
    # Set defaults for compounds without real predictions
    missing_rows = df.index.difference(valid_rows.index)
    df.loc[missing_rows, 'binding_affinity_normalized'] = 0.0  # Penalize missing data
    df.loc[missing_rows, 'cdk11_avoidance_normalized'] = 0.0
    
    # Other scores
    for score_type in ['qed', 'sa_score', 'toxicity', 'novelty']:
        norm_col = f'{score_type}_normalized'
        if norm_col not in df.columns:
            df[norm_col] = 0.5
    
    return df

def calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted composite scores"""
    print_rt("Calculating composite scores...")
    
    weights = CONFIG["weights"]
    
    # Ensure all normalized columns exist
    required_cols = [
        'binding_affinity_normalized', 'cdk11_avoidance_normalized',
        'qed_normalized', 'sa_score_normalized', 'toxicity_normalized', 'novelty_normalized'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0  # Penalize missing data
    
    df['composite_score'] = (
        df['binding_affinity_normalized'] * weights['binding_affinity'] +
        df['cdk11_avoidance_normalized'] * weights['cdk11_avoidance'] +
        df['qed_normalized'] * weights['qed'] +
        df['sa_score_normalized'] * weights['sa_score'] +
        df['toxicity_normalized'] * weights['toxicity'] +
        df['novelty_normalized'] * weights['novelty']
    )
    
    return df

def generate_html_report(df: pd.DataFrame, team_name: str, output_dir: str):
    """Generate HTML report highlighting real vs missing predictions"""
    print_rt("Generating HTML report...")
    
    # Sort by composite score
    df_sorted = df.sort_values('composite_score', ascending=False)
    top_25 = df_sorted.head(25)
    
    # Count real predictions
    real_predictions = df[df.get('CDK4_real_prediction', False) == True]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Powered-Drug-Discovery-Bootcamp Evaluation Report - {team_name} (REAL PREDICTIONS ONLY)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #e8f4fd; padding: 20px; border-radius: 5px; border-left: 5px solid #2196F3; }}
            .summary {{ margin: 20px 0; }}
            .table-container {{ overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .real-data {{ background-color: #d4edda; }}
            .missing-data {{ background-color: #f8d7da; }}
            .warning {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 AI-Powered-Drug-Discovery-Bootcamp Evaluation Report - REAL PREDICTIONS ONLY</h1>
            <h2>Team: {team_name}</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Mode:</strong> NO MOCK PREDICTIONS - Real Boltz2 data only</p>
            <p><strong>Endpoints Used:</strong> Multiple parallel Boltz2 NIM instances</p>
        </div>
        
        <div class="warning">
            <h3>⚠️ Important Note</h3>
            <p>This evaluation uses ONLY real Boltz2 predictions. Any compounds with failed predictions 
            receive zero scores for binding affinity metrics. This ensures complete data integrity but 
            may result in lower overall scores if endpoints were unavailable.</p>
        </div>
        
        <div class="summary">
            <h3>📊 Summary Statistics</h3>
            <p><strong>Total compounds:</strong> {len(df)}</p>
            <p><strong>Compounds with real predictions:</strong> {len(real_predictions)} 
               ({len(real_predictions)/len(df)*100:.1f}%)</p>
            <p><strong>Top 25 average score:</strong> {top_25['composite_score'].mean():.3f}</p>
            <p><strong>Best compound score:</strong> {df['composite_score'].max():.3f}</p>
            <p><strong>Real data integrity:</strong> 100% (no mock predictions used)</p>
        </div>
        
        <div class="table-container">
            <h3>Top 25 Compounds</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>SMILES</th>
                        <th>Score</th>
                        <th>CDK4 pIC50</th>
                        <th>CDK11 pIC50</th>
                        <th>Selectivity</th>
                        <th>QED</th>
                        <th>Data Quality</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i, (_, row) in enumerate(top_25.iterrows(), 1):
        has_real_data = row.get('CDK4_real_prediction', False)
        row_class = "real-data" if has_real_data else "missing-data"
        data_quality = "✅ Real" if has_real_data else "❌ Missing"
        
        html_content += f"""
                    <tr class="{row_class}">
                        <td>{i}</td>
                        <td style="font-family: monospace; text-align: left;">{row['smiles'][:50]}{'...' if len(row['smiles']) > 50 else ''}</td>
                        <td>{row['composite_score']:.3f}</td>
                        <td>{row.get('CDK4_pic50', 'N/A') if pd.notna(row.get('CDK4_pic50')) else 'N/A'}</td>
                        <td>{row.get('CDK11_pic50', 'N/A') if pd.notna(row.get('CDK11_pic50')) else 'N/A'}</td>
                        <td>{row.get('selectivity_ratio', 'N/A') if pd.notna(row.get('selectivity_ratio')) else 'N/A'}</td>
                        <td>{row.get('qed', 0):.3f}</td>
                        <td>{data_quality}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="summary">
            <h3>🎯 Data Quality Legend</h3>
            <p><span style="background-color: #d4edda; padding: 2px 8px;">✅ Real</span> - 
               Compound evaluated using actual Boltz2 predictions</p>
            <p><span style="background-color: #f8d7da; padding: 2px 8px;">❌ Missing</span> - 
               Boltz2 prediction failed, compound penalized in scoring</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, f"{team_name}_real_evaluation_report.html"), 'w') as f:
        f.write(html_content)

async def main():
    parser = argparse.ArgumentParser(description='Parallel AI-Powered-Drug-Discovery-Bootcamp Evaluation - REAL PREDICTIONS ONLY')
    parser.add_argument('smiles_file', help='CSV file containing SMILES')
    parser.add_argument('team_name', help='Team name for the submission')
    parser.add_argument('--endpoints', default=os.environ.get("BOLTZ2_ENDPOINTS", os.environ.get("BOLTZ2_URL", "8000,8001,8002")),
                       help='Comma-separated Boltz2 endpoint URLs or ports (default: BOLTZ2_ENDPOINTS/BOLTZ2_URL or 8000,8001,8002)')
    parser.add_argument('--max-workers', type=int, default=6,
                       help='Maximum number of concurrent workers (default: 6)')
    parser.add_argument('--output-dir', default='evaluation_output_real',
                       help='Output directory for results')
    parser.add_argument('--skip-toxicity', action='store_true',
                       help='Skip toxicity calculations')
    parser.add_argument('--skip-novelty', action='store_true',
                       help='Skip novelty calculations')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    endpoint_urls = normalize_endpoint_urls(args.endpoints)
    CONFIG["endpoints"] = endpoint_urls
    CONFIG["max_workers"] = args.max_workers
    
    print_rt(f"🔬 STARTING REAL-ONLY EVALUATION for team: {args.team_name}")
    print_rt(f"Mode: NO MOCK PREDICTIONS - Real Boltz2 only")
    print_rt(f"Endpoints: {endpoint_urls}")
    print_rt(f"Max workers: {args.max_workers}")
    
    # Load SMILES data
    try:
        df = pd.read_csv(args.smiles_file)
        if 'smiles' not in df.columns:
            if 'SMILES' in df.columns:
                df = df.rename(columns={'SMILES': 'smiles'})
            else:
                print("ERROR: No 'smiles' or 'SMILES' column found")
                return
        
        print_rt(f"✓ Loaded {len(df)} compounds from {args.smiles_file}")
    except Exception as e:
        print(f"ERROR loading SMILES file: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize endpoint pool - REQUIRED
    try:
        endpoint_pool = EndpointPool(endpoint_urls, timeout=CONFIG["api_timeout"])
        await endpoint_pool.initialize()
    except Exception as e:
        print_rt(f"❌ CRITICAL ERROR: {e}")
        print_rt(f"Cannot proceed without healthy Boltz2 endpoints.")
        print_rt(f"Please ensure Boltz2 NIM instances are running on the specified ports.")
        return
    
    start_time = time.time()
    
    # Calculate binding affinities - REAL ONLY
    try:
        df = await calculate_all_binding_affinities_real_only(
            df, endpoint_pool, args.max_workers, args.verbose
        )
    except Exception as e:
        print_rt(f"❌ BINDING AFFINITY CALCULATION FAILED: {e}")
        print_rt(f"Cannot proceed without real Boltz2 predictions.")
        return
    
    # Calculate other scores
    df = calculate_qed_scores(df)
    df = calculate_sa_scores(df)
    
    # Set scores for skipped calculations
    if args.skip_toxicity:
        df['toxicity_normalized'] = 0.5
        print_rt("Skipping toxicity calculations")
    
    if args.skip_novelty:
        df['novelty_normalized'] = 0.5
        print_rt("Skipping novelty calculations")
    
    # Normalize and calculate composite scores
    df = normalize_scores(df)
    df = calculate_composite_scores(df)
    
    # Generate outputs
    generate_html_report(df, args.team_name, args.output_dir)
    
    # Save detailed CSV
    csv_path = os.path.join(args.output_dir, f"{args.team_name}_real_results.csv")
    df.to_csv(csv_path, index=False)
    
    total_time = time.time() - start_time
    
    # Final summary
    real_count = len(df[df.get('CDK4_real_prediction', False) == True])
    
    print_rt(f"\n🎉 REAL-ONLY EVALUATION COMPLETED!")
    print_rt(f"Total time: {total_time:.2f} seconds")
    print_rt(f"Real predictions: {real_count}/{len(df)} compounds")
    print_rt(f"Data integrity: 100% (no mock data)")
    print_rt(f"Results saved to: {args.output_dir}")
    print_rt(f"HTML report: {args.team_name}_real_evaluation_report.html")
    print_rt(f"CSV data: {args.team_name}_real_results.csv")
    
    # Show top results
    valid_df = df[df.get('CDK4_real_prediction', False) == True]
    if len(valid_df) > 0:
        df_sorted = valid_df.sort_values('composite_score', ascending=False)
        print_rt(f"\n🏆 TOP 5 COMPOUNDS (Real Data Only):")
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            print_rt(f"{i}. Score: {row['composite_score']:.3f}, SMILES: {row['smiles'][:60]}...")
    else:
        print_rt(f"\n⚠️  NO COMPOUNDS WITH COMPLETE REAL PREDICTIONS")

if __name__ == "__main__":
    asyncio.run(main())
