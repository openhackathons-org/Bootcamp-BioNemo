# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
Pipeline Module - End-to-end CDK inhibitor design workflow

Orchestrates molecule generation, affinity prediction, scoring, and visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import json

from .config import CDKConfig
from .nim_client import NIMHealthChecker, MolMIMClient, Boltz2AffinityClient
from .physicochemical import PhysicochemCalculator
from .scoring import CDKScorer
from .visualization import CDKVisualizer


def _top_compounds_by_rank(scores_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return top compounds even when CSV/object dtypes leak into rank columns."""
    if scores_df.empty:
        return scores_df.copy()

    ranked = scores_df.copy()
    if "rank" in ranked.columns:
        rank = pd.to_numeric(ranked["rank"], errors="coerce")
        if rank.notna().any():
            return (
                ranked.assign(_rank_sort=rank)
                .sort_values("_rank_sort", na_position="last")
                .drop(columns="_rank_sort")
                .head(n)
            )

    if "total_score" in ranked.columns:
        score = pd.to_numeric(ranked["total_score"], errors="coerce")
        if score.notna().any():
            return (
                ranked.assign(_score_sort=score)
                .sort_values("_score_sort", ascending=False, na_position="last")
                .drop(columns="_score_sort")
                .head(n)
            )

    return ranked.head(n)


@dataclass
class PipelineResults:
    """Container for pipeline results."""
    seed_smiles: List[str]
    generated_smiles: List[str]
    scores_df: pd.DataFrame
    summary: Dict[str, Any]
    history: List[Dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    
    def get_top_compounds(self, n: int = 10) -> pd.DataFrame:
        """Get top N compounds by score."""
        return _top_compounds_by_rank(self.scores_df, n)
    
    def save(self, output_dir: str):
        """Save results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scores
        self.scores_df.to_csv(output_dir / "all_scores.csv", index=False)
        self.get_top_compounds(25).to_csv(output_dir / "top_compounds.csv", index=False)
        
        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            json.dump(self.summary, f, indent=2, default=str)
        
        # Save generated SMILES
        with open(output_dir / "generated_smiles.txt", "w") as f:
            for smi in self.generated_smiles:
                f.write(smi + "\n")
        
        print(f"Results saved to: {output_dir}")


class CDKDesignPipeline:
    """End-to-end pipeline for CDK inhibitor design.
    
    Combines:
    - MolMIM for molecule generation
    - Boltz2 for binding affinity prediction
    - Physicochemical property calculation
    - Composite scoring
    - Visualization and reporting
    
    Usage:
        pipeline = CDKDesignPipeline()
        results = pipeline.run(seed_smiles=["CCO", "CCN"])
        results.save("./output")
    """
    
    def __init__(self, config: CDKConfig = None, boltz2_endpoints: List[Dict[str, str]] = None):
        """Initialize the CDK design pipeline.
        
        Args:
            config: CDK configuration (or created from env vars)
            boltz2_endpoints: List of Boltz2 endpoint configs for parallel predictions
                              e.g., [{"url": "http://gpu1:8000"}, {"url": "http://gpu2:8000"}]
                              If None, uses config.boltz2_endpoints (from env vars)
        """
        self.config = config or CDKConfig()
        
        # Get Boltz2 endpoints from args, config, or default
        endpoints = boltz2_endpoints or self.config.boltz2_endpoints
        
        # Initialize components
        self.health_checker = NIMHealthChecker(self.config)
        self.molmim = MolMIMClient(self.config)
        self.boltz2 = Boltz2AffinityClient(self.config, endpoints=endpoints)
        self.physchem = PhysicochemCalculator(self.config)
        
        # Log endpoint configuration
        if len(endpoints) > 1:
            print(f"⚡ Multi-endpoint mode: {len(endpoints)} Boltz2 endpoints configured")
            for i, ep in enumerate(endpoints):
                print(f"   Endpoint {i+1}: {ep.get('url')}")
        self.scorer = CDKScorer(self.config)
        self.visualizer = CDKVisualizer(self.config)
        
        # Optimization state
        self._history = []
    
    def check_services(self) -> Dict[str, bool]:
        """Check all NIM services are available.
        
        Returns:
            Dict with service availability status
        """
        return self.health_checker.print_status()
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Sanitize text for use as filename."""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized[:max_length]
    
    def _save_structures(
        self,
        structures_data: List[Dict[str, Any]],
        seed_idx: int,
        iteration: int,
        verbose: bool = True
    ) -> int:
        """Save Boltz2 predicted structures to output folder.
        
        Naming convention: {run_name}_seed_{seed}_iter_{iter}_design_{idx}_{target}.cif
        Example: run_20260115_095027_seed_0_iter_2_design_1_cdk4.cif
        
        Args:
            structures_data: List of dicts with smiles and structure data
            seed_idx: Seed molecule index
            iteration: Iteration number
            verbose: Print progress
            
        Returns:
            Number of structures saved
        """
        if not structures_data:
            return 0
        
        # Create structures subfolder
        structures_dir = self.config.output_dir / "boltz2_structures"
        structures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get run name from output_dir (e.g., "run_20260115_095027")
        run_name = self.config.output_dir.name
        
        saved_count = 0
        
        for compound_idx, data in enumerate(structures_data):
            # Save structures for each protein target
            for protein in [self.config.on_target, self.config.anti_target]:
                structures = data.get(f"{protein}_structures", [])
                
                for struct in structures:
                    mmcif_data = struct.get("mmcif")
                    
                    if mmcif_data:
                        # Format: run_20260115_095027_seed_0_iter_2_design_1_cdk4.cif
                        target_lower = protein.lower()
                        filename = f"{run_name}_seed_{seed_idx}_iter_{iteration}_design_{compound_idx}_{target_lower}.cif"
                        filepath = structures_dir / filename
                        
                        with open(filepath, "w") as f:
                            f.write(mmcif_data)
                        saved_count += 1
        
        if verbose and saved_count > 0:
            print(f"    Saved {saved_count} structure files to {structures_dir}")
        
        return saved_count
    
    def _calculate_diversity(self, smiles_list: List[str]) -> float:
        """Calculate molecular diversity as average pairwise Tanimoto distance.

        Diversity = 1 - average_pairwise_similarity

        Higher values = more diverse set of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Diversity score (0-1), where 1 = maximally diverse
        """
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import TanimotoSimilarity

        if len(smiles_list) < 2:
            return 1.0  # Single molecule = trivially diverse

        # Create fingerprint generator using new rdFingerprintGenerator API
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        # Calculate fingerprints
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = gen.GetFingerprint(mol)
                fps.append(fp)
        
        if len(fps) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def generate_molecules(
        self,
        seed_smiles: List[str],
        num_iterations: int = None,
        popsize: int = None,
        top_k_for_boltz2: int = None,
        use_cma: bool = True,
        use_msa: bool = True,
        reference_smiles: List[str] = None,
        verbose: bool = True,
        save_structures: bool = None
    ) -> List[str]:
        """Generate molecules using MolMIM with full CDK scoring in the loop.
        
        Each iteration:
        1. Generate `popsize` candidates from CMA-ES
        2. Fast filter: validity, QED, SA, PAINS, novelty
        3. Select top `top_k_for_boltz2` for expensive Boltz2 predictions
        4. Compute full CDK score (affinity + selectivity + drug-likeness)
        5. Feed combined scores back to CMA-ES
        6. Optionally save Boltz2 predicted complex structures
        
        Args:
            seed_smiles: Starting molecules
            num_iterations: CMA-ES iterations (default: config.cma_iterations)
            popsize: Population size (default: config.cma_popsize)
            top_k_for_boltz2: Top K molecules to run Boltz2 on (default: popsize//2)
            use_cma: Use CMA-ES optimization (if False, just sample)
            use_msa: Use MSA for Boltz2 predictions
            reference_smiles: Reference compounds for novelty scoring
            verbose: Print progress
            save_structures: Save Boltz2 predicted structures (default: config.save_boltz2_structures)
            
        Returns:
            List of generated SMILES
        """
        num_iterations = num_iterations or self.config.cma_iterations
        popsize = popsize or self.config.cma_popsize
        top_k_for_boltz2 = top_k_for_boltz2 or max(1, popsize // 2)
        save_structures = save_structures if save_structures is not None else self.config.save_boltz2_structures

        if use_cma and getattr(self.molmim, "hosted", False):
            if verbose:
                print("Hosted MolMIM does not expose latent /hidden or /decode endpoints; using generation sampling instead of CMA-ES.")
            use_cma = False
        
        # Set reference for novelty scoring
        if reference_smiles:
            self.scorer.set_reference_compounds(reference_smiles)
        else:
            self.scorer.set_reference_compounds(seed_smiles)
        
        if verbose:
            print(f"Generating molecules from {len(seed_smiles)} seeds...")
            print(f"  Iterations: {num_iterations}")
            print(f"  Population size: {popsize}")
            print(f"  Top-K for Boltz2: {top_k_for_boltz2}")
            print(f"  Using MSA: {use_msa}")
            print(f"  Save structures: {save_structures}")
        
        if not use_cma:
            # Simple sampling
            all_generated = []
            for seed in seed_smiles:
                samples = self.molmim.sample(seed, num_samples=popsize * num_iterations)
                all_generated.extend(samples)
            return all_generated
        
        # CMA-ES optimization
        try:
            import cma
        except ImportError:
            print("CMA library not available, falling back to sampling")
            return self.generate_molecules(seed_smiles, num_iterations, popsize, use_cma=False, verbose=verbose)
        
        # Encode seeds
        encodings = self.molmim.encode(seed_smiles)
        
        # Handle different array dimensions (single vs multiple seeds)
        # encodings shape: (1, n_seeds, latent_dims) for multiple seeds
        #                  (1, latent_dims) for single seed after squeeze
        if len(encodings.shape) == 2:
            # Single seed case - reshape to (1, 1, latent_dims)
            encodings = np.expand_dims(encodings, 1)
        
        all_generated = []
        self._history = []
        self._novelty_cache = {}
        best_compounds_overall = []
        
        for seed_idx, seed in enumerate(seed_smiles):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Optimizing from seed {seed_idx + 1}/{len(seed_smiles)}")
                print(f"  Seed: {seed[:60]}...")
                print(f"{'='*60}")
            
            # Initialize CMA-ES
            seed_encoding = encodings[0, seed_idx, :]
            optimizer = cma.CMAEvolutionStrategy(
                seed_encoding,
                self.config.cma_sigma,
                {'popsize': popsize}
            )
            
            for iteration in range(num_iterations):
                if verbose:
                    print(f"\n  Iteration {iteration + 1}/{num_iterations}")
                
                # === Step 1: Generate candidates from CMA-ES ===
                candidates = optimizer.ask(popsize)
                candidates_array = np.array(candidates)
                
                # Decode to SMILES
                if len(candidates_array.shape) == 1:
                    candidates_array = candidates_array.reshape(1, -1)
                
                decoded = self.molmim.decode(np.expand_dims(candidates_array, 0))
                
                # === Step 2: Fast physicochemical filtering (all candidates) ===
                fast_scores = []
                candidate_data = []
                
                for i, smi in enumerate(decoded):
                    props = self.physchem.calculate_properties(smi)
                    
                    if not props.valid:
                        fast_scores.append(0.0)  # Invalid = worst score
                        candidate_data.append(None)
                        continue
                    
                    # Compute fast score components (0-1 each)
                    qed_score = props.qed or 0.0
                    sa_score = max(0.0, 1.0 - (props.sa_score - 1.0) / 9.0) if props.sa_score else 0.5
                    pains_score = 1.0 if props.pains_alerts == 0 else 0.0
                    lipinski_score = 1.0 if props.lipinski_violations <= 1 else 0.5
                    
                    # Novelty score using ChEMBL database (same as evaluate_submission.py)
                    from .physicochemical import calculate_novelty_score_chembl
                    novelty, max_sim, is_novel = calculate_novelty_score_chembl(
                        smi, 
                        cutoff=self.config.novelty_similarity_cutoff,
                        additional_refs=self.scorer.reference_smiles + all_generated[-100:]
                    )
                    self._novelty_cache[smi] = (novelty, max_sim, is_novel)
                    
                    # Combined fast score (weighted)
                    fast_score = (
                        0.35 * qed_score +
                        0.20 * sa_score +
                        0.20 * pains_score +
                        0.15 * lipinski_score +
                        0.10 * novelty
                    )
                    
                    fast_scores.append(fast_score)
                    candidate_data.append({
                        "smiles": smi,
                        "idx": i,
                        "props": props,
                        "fast_score": fast_score,
                        "qed": qed_score,
                        "sa": sa_score,
                        "novelty": novelty
                    })
                
                valid_candidates = [c for c in candidate_data if c is not None]
                valid_count = len(valid_candidates)
                
                if verbose:
                    print(f"    Valid molecules: {valid_count}/{popsize}")
                
                if valid_count == 0:
                    # All invalid - use zeros for CMA-ES
                    optimizer.tell(candidates, [0.0] * popsize)
                    self._history.append({
                        "seed_idx": seed_idx, "iteration": iteration,
                        "valid_count": 0, "best_score": 0.0, "mean_score": 0.0
                    })
                    continue
                
                # === Step 3: Select top-K for Boltz2 predictions ===
                # Sort by fast score (descending)
                valid_candidates.sort(key=lambda x: x["fast_score"], reverse=True)
                top_k_candidates = valid_candidates[:top_k_for_boltz2]
                
                if verbose:
                    print(f"    Running Boltz2 on top {len(top_k_candidates)} candidates (parallel)...")
                
                # === Step 4: Boltz2 affinity predictions for top-K (PARALLEL) ===
                top_k_smiles = [cand["smiles"] for cand in top_k_candidates]
                top_k_idx = [cand["idx"] for cand in top_k_candidates]
                
                # Use parallel batch prediction (with structures if enabled)
                batch_results = self.boltz2.predict_batch(
                    top_k_smiles,
                    proteins=[self.config.on_target, self.config.anti_target],
                    use_msa=use_msa,
                    verbose=False,  # Suppress per-compound output during iteration
                    parallel=True,
                    return_structures=save_structures
                )
                
                # Save structures if enabled
                if save_structures and batch_results:
                    saved = self._save_structures(batch_results, seed_idx, iteration, verbose=verbose)
                
                # Process results and compute scores
                boltz2_scores = {}
                for result, cand in zip(batch_results, top_k_candidates):
                    idx = cand["idx"]
                    smi = cand["smiles"]
                    
                    cdk4_ic50 = result.get(f"{self.config.on_target}_IC50_pred")
                    cdk11_ic50 = result.get(f"{self.config.anti_target}_IC50_pred")
                    
                    # Compute affinity score components
                    if cdk4_ic50 and cdk4_ic50 > 0:
                        binding_score = max(0.0, min(1.0, 1.0 - (np.log10(cdk4_ic50) - 0) / 4))
                    else:
                        binding_score = 0.3
                    
                    if cdk11_ic50 and cdk11_ic50 > 0:
                        avoidance_score = max(0.0, min(1.0, (np.log10(cdk11_ic50) - 1) / 4))
                    else:
                        avoidance_score = 0.5
                    
                    # Selectivity score
                    selectivity = None
                    if cdk4_ic50 and cdk11_ic50 and cdk4_ic50 > 0:
                        selectivity = cdk11_ic50 / cdk4_ic50
                        if selectivity >= 100:
                            selectivity_score = 1.0
                        elif selectivity >= 10:
                            selectivity_score = 0.7 + 0.3 * (selectivity - 10) / 90
                        elif selectivity >= 1:
                            selectivity_score = 0.3 + 0.4 * (selectivity - 1) / 9
                        else:
                            selectivity_score = 0.3 * selectivity
                    else:
                        selectivity_score = 0.5
                    
                    boltz2_scores[idx] = {
                        "cdk4_ic50": cdk4_ic50,
                        "cdk11_ic50": cdk11_ic50,
                        "binding_score": binding_score,
                        "avoidance_score": avoidance_score,
                        "selectivity_score": selectivity_score
                    }
                    
                    if verbose:
                        sel_str = f"{selectivity:.1f}x" if selectivity else "N/A"
                        cdk4_str = f"{cdk4_ic50:.1f}" if cdk4_ic50 else "N/A"
                        on = self.config.on_target
                        anti = self.config.anti_target
                        ep4 = result.get(f"{on}_endpoint", "")
                        ep11 = result.get(f"{anti}_endpoint", "")
                        t4 = result.get(f"{on}_time_s")
                        t11 = result.get(f"{anti}_time_s")
                        ep_tag = ""
                        if ep4 or ep11:
                            ep4_short = ep4.split("//")[-1] if ep4 else "?"
                            ep11_short = ep11.split("//")[-1] if ep11 else "?"
                            t4_str = f" {t4:.1f}s" if t4 is not None else ""
                            t11_str = f" {t11:.1f}s" if t11 is not None else ""
                            ep_tag = f"  [{on}→{ep4_short}{t4_str}, {anti}→{ep11_short}{t11_str}]"
                        print(f"      {smi[:40]}... CDK4={cdk4_str}nM, sel={sel_str}{ep_tag}")
                
                # === Step 5: Compute final scores for CMA-ES ===
                final_scores = []
                for i, smi in enumerate(decoded):
                    cand = candidate_data[i]
                    
                    if cand is None:
                        # Invalid molecule
                        final_scores.append(0.0)
                        continue
                    
                    if i in boltz2_scores:
                        # Has Boltz2 predictions - use full score
                        b2 = boltz2_scores[i]
                        weights = self.config.weights
                        
                        total_score = (
                            weights["binding_affinity"] * b2["binding_score"] +
                            weights["selectivity"] * b2["selectivity_score"] +
                            weights["cdk11_avoidance"] * b2["avoidance_score"] +
                            weights["qed"] * cand["qed"] +
                            weights["sa"] * cand["sa"] +
                            weights["pains"] * (1.0 if cand["props"].pains_alerts == 0 else 0.0) +
                            weights["novelty"] * cand["novelty"]
                        )
                        
                        # Store best compounds
                        best_compounds_overall.append({
                            "smiles": smi,
                            "cdk4_ic50": b2["cdk4_ic50"],
                            "cdk11_ic50": b2["cdk11_ic50"],
                            "selectivity": b2["cdk11_ic50"] / b2["cdk4_ic50"] if b2["cdk4_ic50"] and b2["cdk11_ic50"] else None,
                            "total_score": total_score,
                            "qed": cand["qed"],
                            "seed_idx": seed_idx,
                            "iteration": iteration
                        })
                    else:
                        # No Boltz2 - use fast score (penalized)
                        total_score = cand["fast_score"] * 0.5  # Penalty for no affinity data
                    
                    final_scores.append(total_score)
                
                # CMA-ES minimizes, so negate scores
                cma_scores = [-s for s in final_scores]
                optimizer.tell(candidates, cma_scores)
                
                # Collect valid molecules
                for cand in valid_candidates:
                    all_generated.append(cand["smiles"])
                
                # Track history
                best_score = max(final_scores)
                mean_score = np.mean([s for s in final_scores if s > 0])
                
                # Calculate molecular diversity (average pairwise Tanimoto distance)
                diversity = self._calculate_diversity([c["smiles"] for c in valid_candidates])
                
                self._history.append({
                    "seed_idx": seed_idx,
                    "iteration": iteration,
                    "valid_count": valid_count,
                    "best_score": best_score,
                    "mean_score": mean_score,
                    "boltz2_count": len(boltz2_scores),
                    "diversity": diversity,
                })
                
                if verbose:
                    print(f"    Best score: {best_score:.3f}, Mean: {mean_score:.3f}")
        
        # Deduplicate
        all_generated = list(set(all_generated))
        
        # Store best compounds for later retrieval
        self._best_compounds = sorted(best_compounds_overall, key=lambda x: x["total_score"], reverse=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation complete!")
            print(f"  Total unique molecules: {len(all_generated)}")
            print(f"  Boltz2-scored compounds: {len(self._best_compounds)}")
            if self._best_compounds:
                best = self._best_compounds[0]
                cdk4_best = f"{best['cdk4_ic50']:.1f}" if best.get('cdk4_ic50') else "N/A"
                print(f"  Best compound: score={best['total_score']:.3f}, CDK4={cdk4_best}nM")
            print(f"{'='*60}")
        
        return all_generated
    
    def get_best_compounds_from_generation(self, n: int = 25) -> pd.DataFrame:
        """Get best compounds found during generation (with Boltz2 scores).
        
        These are compounds that were scored with Boltz2 during the optimization loop.
        
        Args:
            n: Number of top compounds to return
            
        Returns:
            DataFrame with top compounds
        """
        if not hasattr(self, '_best_compounds') or not self._best_compounds:
            return pd.DataFrame()
        
        return pd.DataFrame(self._best_compounds[:n])
    
    def _get_affinities_with_cache(
        self,
        smiles_list: List[str],
        use_msa: bool = True,
        skip_generation: bool = False,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Get affinities, reusing cached predictions from optimization loop.
        
        This avoids redundant Boltz-2 calls for molecules already scored during generation.
        
        Args:
            smiles_list: List of SMILES to get affinities for
            use_msa: Use MSA for new predictions
            skip_generation: If True, no cache available (direct evaluation mode)
            verbose: Print progress
            
        Returns:
            DataFrame with affinity predictions
        """
        on_target = self.config.on_target
        anti_target = self.config.anti_target
        
        # Build cache from optimization loop results
        affinity_cache = {}
        if not skip_generation and hasattr(self, '_best_compounds') and self._best_compounds:
            for comp in self._best_compounds:
                smi = comp.get("smiles")
                if smi and comp.get("cdk4_ic50") is not None:
                    affinity_cache[smi] = {
                        "smiles": smi,
                        f"{on_target}_IC50_pred": comp.get("cdk4_ic50"),
                        f"{anti_target}_IC50_pred": comp.get("cdk11_ic50"),
                        f"{on_target}_success": True,
                        f"{anti_target}_success": True,
                    }
        
        # Identify cached vs uncached molecules
        cached_smiles = set(affinity_cache.keys())
        cached_in_list = [smi for smi in smiles_list if smi in cached_smiles]
        uncached = [smi for smi in smiles_list if smi not in cached_smiles]
        
        if verbose:
            print(f"\nAffinity predictions:")
            print(f"  Total molecules: {len(smiles_list)}")
            print(f"  Cached from optimization: {len(cached_in_list)}")
            print(f"  Skipping uncached: {len(uncached)} (no Boltz-2 scores)")
        
        # Use cached results where available
        all_results = []
        for smi in smiles_list:
            if smi in affinity_cache:
                all_results.append(affinity_cache[smi])

        # When skip_generation=True (evaluate_existing mode), predict uncached directly.
        # Hosted MolMIM runs use sampling rather than CMA-ES, so there is no optimization
        # cache; in that case, predict the generated molecules directly as well.
        predict_uncached = skip_generation or (uncached and not all_results)
        if predict_uncached and uncached:
            if verbose:
                if not skip_generation and not all_results:
                    print("  No cached Boltz2 scores from generation; predicting generated compounds directly")
                print(f"  Predicting affinities for {len(uncached)} compounds via Boltz2...")
            batch = self.boltz2.predict_batch(
                uncached,
                proteins=[self.config.on_target, self.config.anti_target],
                use_msa=use_msa,
                verbose=verbose,
                parallel=True
            )
            all_results.extend(batch)
        elif uncached and verbose:
            print(f"  → Using only {len(all_results)} Boltz2-scored compounds for final ranking")
        
        return pd.DataFrame(all_results)
    
    def predict_affinities(
        self,
        smiles_list: List[str],
        use_msa: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Predict binding affinities using Boltz2 (direct, no caching).
        
        Args:
            smiles_list: List of SMILES to predict
            use_msa: Use MSA for better predictions
            verbose: Print progress
            
        Returns:
            DataFrame with affinity predictions
        """
        if verbose:
            print(f"Predicting affinities for {len(smiles_list)} compounds...")
            print(f"  Using MSA: {use_msa}")
        
        results = self.boltz2.predict_batch(
            smiles_list,
            proteins=[self.config.on_target, self.config.anti_target],
            use_msa=use_msa,
            verbose=verbose
        )
        
        return pd.DataFrame(results)
    
    def score_compounds(
        self,
        affinity_df: pd.DataFrame,
        reference_smiles: List[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Score compounds using composite scoring.
        
        Args:
            affinity_df: DataFrame with affinity predictions
            reference_smiles: Reference compounds for novelty scoring
            verbose: Print progress
            
        Returns:
            DataFrame with all scores
        """
        if reference_smiles:
            self.scorer.set_reference_compounds(reference_smiles)
        
        if verbose:
            print(f"Scoring {len(affinity_df)} compounds...")
        
        novelty_cache = getattr(self, '_novelty_cache', None)
        return self.scorer.score_batch(affinity_df, verbose=verbose, novelty_cache=novelty_cache)
    
    def run(
        self,
        seed_smiles: List[str],
        num_iterations: int = None,
        popsize: int = None,
        top_k_for_boltz2: int = None,
        use_cma: bool = True,
        use_msa: bool = True,
        reference_smiles: List[str] = None,
        skip_generation: bool = False,
        smiles_to_evaluate: List[str] = None,
        verbose: bool = True,
        generate_report: bool = True,
        save_structures: bool = None
    ) -> PipelineResults:
        """Run the complete design pipeline.
        
        Args:
            seed_smiles: Starting molecules for generation
            num_iterations: CMA-ES iterations
            popsize: Population size per seed
            top_k_for_boltz2: Top K compounds per iteration to run Boltz2 on
            use_cma: Use CMA-ES optimization
            use_msa: Use MSA for Boltz2 predictions
            reference_smiles: Reference compounds for novelty
            skip_generation: Skip generation, use provided smiles_to_evaluate
            smiles_to_evaluate: Direct SMILES list if skip_generation=True
            verbose: Print progress
            generate_report: Generate HTML report
            save_structures: Save Boltz2 predicted structures (default: config.save_boltz2_structures)
            
        Returns:
            PipelineResults with all results
        """
        start_time = time.time()
        
        if verbose:
            print("=" * 60)
            print("CDK Inhibitor Design Pipeline")
            print("=" * 60)
            print(f"On-target: {self.config.on_target}")
            print(f"Anti-target: {self.config.anti_target}")
            print("=" * 60)
        
        # Step 1: Generate molecules with CDK-aware optimization loop
        if skip_generation and smiles_to_evaluate:
            generated = smiles_to_evaluate
            if verbose:
                print(f"\nUsing {len(generated)} provided compounds")
        else:
            generated = self.generate_molecules(
                seed_smiles,
                num_iterations=num_iterations,
                popsize=popsize,
                top_k_for_boltz2=top_k_for_boltz2,
                use_cma=use_cma,
                use_msa=use_msa,
                reference_smiles=reference_smiles,
                verbose=verbose,
                save_structures=save_structures
            )
        
        if not generated:
            raise ValueError("No molecules generated!")
        
        # Step 2: Get affinities - REUSE cached predictions from optimization loop
        affinity_df = self._get_affinities_with_cache(
            generated, 
            use_msa=use_msa, 
            skip_generation=skip_generation,
            verbose=verbose
        )
        
        # Step 3: Score compounds
        scores_df = self.score_compounds(
            affinity_df,
            reference_smiles=reference_smiles or seed_smiles,
            verbose=verbose
        )
        
        # Step 4: Generate summary
        summary = self.scorer.get_scoring_summary(scores_df)
        summary["seeds"] = len(seed_smiles)
        summary["generated"] = len(generated)
        summary["runtime_seconds"] = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 60)
            print("Results Summary")
            print("=" * 60)
            print(f"Total compounds: {summary['total_compounds']}")
            print(f"Best score: {summary['max_total_score']:.3f}")
            print(f"Potent CDK4 (<100nM): {summary['potent_cdk4_count']}")
            print(f"Selective (>10x): {summary['selective_count']}")
            print(f"Runtime: {summary['runtime_seconds']:.1f}s")
            print("=" * 60)
        
        # Create results
        results = PipelineResults(
            seed_smiles=seed_smiles,
            generated_smiles=generated,
            scores_df=scores_df,
            summary=summary,
            history=self._history,
            runtime_seconds=time.time() - start_time
        )
        
        # Step 5: Save all results to timestamped output folder
        output_dir = self.config.output_dir
        if verbose:
            print(f"\nSaving results to: {output_dir}")
        
        # Save all compounds with scores to CSV
        scores_df.to_csv(output_dir / "all_compounds_scores.csv", index=False)
        
        # Save top compounds separately
        top_df = _top_compounds_by_rank(scores_df, self.config.top_n_compounds)
        top_df.to_csv(output_dir / "top_compounds.csv", index=False)
        
        # Save summary as JSON
        import json
        with open(output_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save generated SMILES list
        with open(output_dir / "generated_smiles.txt", "w") as f:
            for smi in generated:
                f.write(smi + "\n")
        
        # Save seed molecules
        with open(output_dir / "seed_molecules.txt", "w") as f:
            for smi in seed_smiles:
                f.write(smi + "\n")
        
        if verbose:
            print(f"  ✓ all_compounds_scores.csv ({len(scores_df)} compounds)")
            print(f"  ✓ top_compounds.csv (top {len(top_df)})")
            print(f"  ✓ run_summary.json")
            print(f"  ✓ generated_smiles.txt")
            
            # Check if structures were saved
            structures_dir = output_dir / "boltz2_structures"
            if structures_dir.exists():
                num_struct_files = sum(1 for _ in structures_dir.rglob("*.cif"))
                print(f"  ✓ boltz2_structures/ ({num_struct_files} CIF files)")
        
        # Step 6: Generate HTML report
        if generate_report:
            self.visualizer.generate_report(
                scores_df,
                summary,
                output_dir=output_dir
            )
            if verbose:
                print(f"  ✓ design_report.html")
        
        return results
    
    def evaluate_existing(
        self,
        smiles_list: List[str],
        compound_ids: List[str] = None,
        use_msa: bool = True,
        verbose: bool = True
    ) -> PipelineResults:
        """Evaluate existing compounds without generation.
        
        Convenience method for evaluating user-provided compounds.
        
        Args:
            smiles_list: List of SMILES to evaluate
            compound_ids: Optional compound identifiers
            use_msa: Use MSA for predictions
            verbose: Print progress
            
        Returns:
            PipelineResults
        """
        return self.run(
            seed_smiles=[],
            skip_generation=True,
            smiles_to_evaluate=smiles_list,
            use_msa=use_msa,
            verbose=verbose
        )
