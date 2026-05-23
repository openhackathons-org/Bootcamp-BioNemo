# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
NIM Client Module - Connections to MolMIM and Boltz2 services

Provides health checking and client wrappers for NVIDIA NIMs.
"""

import os
import json
import asyncio
import requests
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from .config import CDKConfig


def _is_hosted_molmim_url(url: str) -> bool:
    """Return True for NVIDIA hosted API endpoints."""
    return "api.nvidia.com" in urlparse(url).netloc


def _hosted_molmim_generate_url(url: str) -> str:
    """Normalize hosted MolMIM URLs to the generation endpoint."""
    normalized = url.rstrip("/")
    parsed = urlparse(normalized)

    # The hosted BioNeMo inference routes currently live on health.api.nvidia.com.
    if parsed.netloc == "integrate.api.nvidia.com":
        parsed = parsed._replace(netloc="health.api.nvidia.com")
        normalized = urlunparse(parsed)

    if normalized.endswith("/generate"):
        return normalized

    if "/biology/nvidia/molmim" in parsed.path:
        return f"{normalized}/generate"

    if _is_hosted_molmim_url(normalized):
        if parsed.path.rstrip("/") == "/v1":
            return f"{normalized}/biology/nvidia/molmim/generate"
        if parsed.path in ("", "/"):
            return f"{normalized}/v1/biology/nvidia/molmim/generate"

    return f"{normalized}/biology/nvidia/molmim/generate"


def _is_hosted_boltz2_url(url: str) -> bool:
    """Return True for NVIDIA hosted Boltz-2 API endpoints."""
    return "api.nvidia.com" in urlparse(url).netloc


def _normalize_boltz2_base_url(url: str) -> str:
    """Normalize Boltz-2 endpoint roots for the Python client."""
    normalized = url.rstrip("/")
    if normalized.endswith("/biology/mit/boltz2/predict"):
        return normalized[:-len("/predict")]
    return normalized


def _boltz2_client_base_url(url: str) -> str:
    """Return the base URL shape expected by boltz2-python-client."""
    normalized = _normalize_boltz2_base_url(url)
    if not _is_hosted_boltz2_url(normalized):
        return normalized

    parsed = urlparse(normalized)
    if parsed.netloc == "integrate.api.nvidia.com":
        parsed = parsed._replace(netloc="health.api.nvidia.com")
    return urlunparse(parsed._replace(path="", params="", query="", fragment="")).rstrip("/")


def _boltz2_client_kwargs(url: str, api_key: str = None) -> Dict[str, Any]:
    """Build Boltz2Client kwargs for local NIMs or NVIDIA-hosted API catalog."""
    normalized = _normalize_boltz2_base_url(url)
    kwargs = {
        "base_url": _boltz2_client_base_url(normalized),
        "api_key": api_key,
    }
    if _is_hosted_boltz2_url(normalized):
        kwargs["endpoint_type"] = "nvidia_hosted"
    return kwargs


def is_hosted_boltz2_url(url: str) -> bool:
    """Public helper for notebooks and scripts that need hosted-mode branching."""
    return _is_hosted_boltz2_url(url)


def normalize_boltz2_url(url: str) -> str:
    """Public helper that normalizes Boltz-2 REST endpoint roots."""
    return _normalize_boltz2_base_url(url)


def boltz2_client_kwargs(url: str, api_key: str = None) -> Dict[str, Any]:
    """Public helper for constructing Boltz2Client arguments."""
    return _boltz2_client_kwargs(url, api_key=api_key)


@dataclass
class NIMStatus:
    """Status of a NIM service."""
    name: str
    url: str
    available: bool
    message: str


class NIMHealthChecker:
    """Check health status of NIM services."""

    def __init__(self, config: CDKConfig = None, boltz2_endpoints: List[Dict[str, str]] = None):
        self._boltz2_endpoints = boltz2_endpoints
        self.config = config or CDKConfig()

    def check_molmim(self) -> NIMStatus:
        """Check MolMIM NIM health."""
        url = self.config.molmim_url
        headers = {}
        if self.config.molmim_api_key:
            headers["Authorization"] = f"Bearer {self.config.molmim_api_key}"
        try:
            if _is_hosted_molmim_url(url):
                hosted_headers = {
                    **headers,
                    "accept": "application/json",
                    "Content-Type": "application/json",
                }
                response = requests.post(
                    _hosted_molmim_generate_url(url),
                    headers=hosted_headers,
                    json={
                        "smi": "CCO",
                        "algorithm": "none",
                        "num_molecules": 1,
                        "particles": 2,
                        "scaled_radius": 1.0,
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    return NIMStatus("MolMIM", url, True, "Hosted endpoint reachable")
                return NIMStatus("MolMIM", url, False, f"HTTP {response.status_code}")

            # Try health endpoint
            response = requests.get(f"{url}/v1/health/ready", headers=headers, timeout=5)
            if response.status_code == 200:
                return NIMStatus("MolMIM", url, True, "Service is healthy")

            # Try root endpoint
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code in [200, 404]:
                return NIMStatus("MolMIM", url, True, "Service reachable")

            return NIMStatus("MolMIM", url, False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            return NIMStatus("MolMIM", url, False, "Connection refused")
        except requests.exceptions.Timeout:
            return NIMStatus("MolMIM", url, False, "Connection timeout")
        except Exception as e:
            return NIMStatus("MolMIM", url, False, str(e))

    def _check_boltz2_url(self, url: str, api_key: str = None, label: str = "Boltz2") -> NIMStatus:
        """Check a single Boltz2 endpoint."""
        try:
            headers = {}
            ak = api_key or self.config.boltz2_api_key
            if ak:
                headers["Authorization"] = f"Bearer {ak}"

            if _is_hosted_boltz2_url(url):
                if ak:
                    return NIMStatus(label, url, True, "Hosted endpoint configured")
                return NIMStatus(label, url, False, "Hosted endpoint requires BOLTZ2_API_KEY, NVIDIA_API_KEY, or NGC_API_KEY")

            response = requests.get(f"{_normalize_boltz2_base_url(url)}/v1/health/ready", headers=headers, timeout=5)
            if response.status_code == 200:
                return NIMStatus(label, url, True, "Service is healthy")

            return NIMStatus(label, url, False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            return NIMStatus(label, url, False, "Connection refused")
        except requests.exceptions.Timeout:
            return NIMStatus(label, url, False, "Connection timeout")
        except Exception as e:
            return NIMStatus(label, url, False, str(e))

    def check_boltz2(self) -> NIMStatus:
        """Check primary Boltz2 NIM health."""
        return self._check_boltz2_url(self.config.boltz2_url)

    def check_boltz2_all(self) -> List[NIMStatus]:
        """Check all Boltz2 endpoints (primary + any extras)."""
        urls_seen = set()
        results = []

        primary_url = self.config.boltz2_url
        results.append(self._check_boltz2_url(primary_url, label="Boltz2 [1]"))
        urls_seen.add(primary_url)

        if self._boltz2_endpoints:
            for i, ep in enumerate(self._boltz2_endpoints):
                ep_url = ep.get("url", "")
                if ep_url and ep_url not in urls_seen:
                    urls_seen.add(ep_url)
                    results.append(self._check_boltz2_url(
                        ep_url, api_key=ep.get("api_key"),
                        label=f"Boltz2 [{len(results)+1}]"))

        return results

    def check_all(self) -> Dict[str, NIMStatus]:
        """Check all NIM services (backward-compatible, primary Boltz2 only)."""
        return {
            "molmim": self.check_molmim(),
            "boltz2": self.check_boltz2()
        }

    def print_status(self) -> Dict[str, bool]:
        """Print formatted status of all services, including all Boltz2 endpoints."""
        print("=" * 60)
        print("NIM Service Health Check")
        print("=" * 60)

        results = {}

        molmim = self.check_molmim()
        icon = "✓" if molmim.available else "✗"
        print(f"{icon} {molmim.name}")
        print(f"    URL: {molmim.url}")
        print(f"    Status: {molmim.message}")
        results["molmim"] = molmim.available

        boltz2_statuses = self.check_boltz2_all()
        for bs in boltz2_statuses:
            icon = "✓" if bs.available else "✗"
            print(f"{icon} {bs.name}")
            print(f"    URL: {bs.url}")
            print(f"    Status: {bs.message}")

        results["boltz2"] = any(bs.available for bs in boltz2_statuses)

        print("=" * 60)
        return results


class MolMIMClient:
    """Client for MolMIM NIM - molecule encoding/decoding."""

    def __init__(self, config: CDKConfig = None):
        self.config = config or CDKConfig()
        self.base_url = self.config.molmim_url
        self.api_key = self.config.molmim_api_key
        self.latent_dims = self.config.molmim_latent_dims
        self.hosted = _is_hosted_molmim_url(self.base_url)

    @property
    def num_latent_dims(self) -> int:
        return self.latent_dims

    def _headers(self) -> Dict[str, str]:
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def encode(self, smiles: List[str]) -> np.ndarray:
        """Encode SMILES strings into latent space.

        Args:
            smiles: List of SMILES strings

        Returns:
            np.ndarray of shape (1, len(smiles), latent_dims)
        """
        if self.hosted:
            raise RuntimeError(
                "Hosted MolMIM supports molecule generation but does not expose "
                "latent-space /hidden. Set OPENHACKATHON_USE_CMA=0 or use a local MolMIM NIM."
            )

        url = f"{self.base_url}/hidden"
        data = json.dumps({"sequences": smiles})

        response = requests.post(url, headers=self._headers(), data=data, timeout=60)
        response.raise_for_status()

        embeddings = response.json()["hiddens"]
        embeddings_array = np.squeeze(np.array(embeddings))
        return np.expand_dims(embeddings_array, 0)

    def decode(self, latent_features: np.ndarray) -> List[str]:
        """Decode latent features into SMILES strings.

        Args:
            latent_features: np.ndarray of shape (1, n, latent_dims)

        Returns:
            List of generated SMILES strings
        """
        if self.hosted:
            raise RuntimeError(
                "Hosted MolMIM supports molecule generation but does not expose "
                "latent-space /decode. Set OPENHACKATHON_USE_CMA=0 or use a local MolMIM NIM."
            )

        dims = list(latent_features.shape)
        if len(dims) == 2:
            latent_features = np.expand_dims(latent_features, 0)

        url = f"{self.base_url}/decode"

        latent_squeezed = np.squeeze(latent_features)
        latent_array = np.expand_dims(np.array(latent_squeezed), axis=1)

        payload = {
            "hiddens": latent_array.tolist(),
            "mask": [[True] for _ in range(latent_array.shape[0])]
        }

        response = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()

        return response.json()["generated"]

    def _sample_hosted(self, smiles: str, num_samples: int = 10, scaled_radius: float = 1.0) -> List[str]:
        """Sample molecules from NVIDIA hosted MolMIM."""
        payload = {
            "smi": smiles,
            "algorithm": os.environ.get("MOLMIM_HOSTED_ALGORITHM", "none"),
            "num_molecules": int(num_samples),
            "particles": max(2, int(num_samples)),
            "scaled_radius": float(scaled_radius),
        }

        response = requests.post(
            _hosted_molmim_generate_url(self.base_url),
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        body = response.json()
        molecules = body.get("molecules", body.get("samples", body.get("generated", [])))
        if isinstance(molecules, str):
            try:
                molecules = json.loads(molecules)
            except json.JSONDecodeError:
                molecules = [molecules]
        if molecules and isinstance(molecules[0], list):
            molecules = molecules[0]

        samples = []
        for entry in molecules:
            if isinstance(entry, str):
                smi = entry
            elif isinstance(entry, dict):
                smi = entry.get("sample") or entry.get("smiles") or entry.get("smi")
            else:
                smi = None
            if smi:
                samples.append(smi)

        return samples

    def sample(self, smiles: str, num_samples: int = 10) -> List[str]:
        """Sample similar molecules to a seed SMILES.

        Args:
            smiles: Seed SMILES string
            num_samples: Number of samples to generate

        Returns:
            List of generated SMILES strings
        """
        if self.hosted:
            return self._sample_hosted(smiles, num_samples=num_samples)

        url = f"{self.base_url}/sampling"

        payload = {
            "smiles": smiles,
            "num_samples": num_samples,
            "temperature": 1.0
        }

        response = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()

        body = response.json()
        return body.get("samples", body.get("generated", []))


class Boltz2AffinityClient:
    """Client for Boltz2 NIM - binding affinity prediction.

    Supports multiple endpoints for parallel predictions.
    """

    def __init__(self, config: CDKConfig = None, endpoints: List[Dict[str, str]] = None):
        """Initialize Boltz2 client.

        Args:
            config: CDK configuration
            endpoints: List of endpoint configs, each with 'url' and optional 'api_key'
                       e.g., [{"url": "http://gpu1:8000"}, {"url": "http://gpu2:8000"}]
                       If None, uses single endpoint from config.
        """
        self.config = config or CDKConfig()

        # Set up endpoint pool
        if endpoints:
            self.endpoints = [
                {**ep, "url": _normalize_boltz2_base_url(ep.get("url", self.config.boltz2_url))}
                for ep in endpoints
            ]
        else:
            self.endpoints = [{
                "url": _normalize_boltz2_base_url(self.config.boltz2_url),
                "api_key": self.config.boltz2_api_key
            }]

        # For single-endpoint backward compatibility
        self.base_url = self.endpoints[0]["url"]
        self.api_key = self.endpoints[0].get("api_key", self.config.boltz2_api_key)

        # Endpoint tracking for load balancing
        self._endpoint_idx = 0
        self._endpoint_busy = {i: 0 for i in range(len(self.endpoints))}

        # Lazy import boltz2_client
        self._boltz2_available = False
        self._client_module = None
        self._load_client()

        # Concurrency control
        self._semaphore = None  # Initialized when needed

    def _load_client(self):
        """Load boltz2_client module."""
        try:
            from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest, PocketConstraint
            from boltz2_client.models import AlignmentFileRecord
            self._boltz2_available = True
            self._client_module = {
                "Boltz2Client": Boltz2Client,
                "Polymer": Polymer,
                "Ligand": Ligand,
                "PredictionRequest": PredictionRequest,
                "PocketConstraint": PocketConstraint,
                "AlignmentFileRecord": AlignmentFileRecord
            }
        except ImportError:
            self._boltz2_available = False

    @property
    def available(self) -> bool:
        return self._boltz2_available

    def _get_next_endpoint(self) -> Dict[str, str]:
        """Get next endpoint using round-robin load balancing."""
        endpoint = self.endpoints[self._endpoint_idx]
        self._endpoint_idx = (self._endpoint_idx + 1) % len(self.endpoints)
        return endpoint

    def _get_least_busy_endpoint(self) -> tuple:
        """Get endpoint with least active requests. Breaks ties with round-robin."""
        min_busy = min(self._endpoint_busy.values())
        candidates = [idx for idx, busy in self._endpoint_busy.items() if busy == min_busy]
        # Round-robin among tied candidates to avoid always picking the first
        start = self._endpoint_idx % len(self.endpoints)
        for offset in range(len(self.endpoints)):
            idx = (start + offset) % len(self.endpoints)
            if idx in candidates:
                self._endpoint_idx = idx + 1
                return idx, self.endpoints[idx]
        return candidates[0], self.endpoints[candidates[0]]

    async def predict_affinity_async(
        self,
        smiles: str,
        protein: str,
        use_msa: bool = True,
        endpoint: Dict[str, str] = None,
        return_structures: bool = False
    ) -> Dict[str, Any]:
        """Predict binding affinity asynchronously.

        Args:
            smiles: Ligand SMILES string
            protein: Target protein name ("CDK4" or "CDK11")
            use_msa: Whether to use MSA for better predictions
            endpoint: Specific endpoint to use (optional, uses load balancing if None)
            return_structures: Whether to include structure data (mmCIF) in response

        Returns:
            Dict with ic50_nm, pic50, confidence, success, and optionally structures
        """
        if not self._boltz2_available:
            return {"error": "Boltz2 client not available", "success": False}

        # Select endpoint using least-busy strategy
        ep_idx = None
        if endpoint is None:
            ep_idx, endpoint = self._get_least_busy_endpoint()
            self._endpoint_busy[ep_idx] += 1

        base_url = endpoint.get("url", self.base_url)
        api_key = endpoint.get("api_key", self.api_key)

        # Get protein data
        sequence = self.config.load_sequence(protein)
        binding_sites = self.config.binding_sites.get(protein, [])
        msa_content = self.config.load_msa(protein) if use_msa else None

        # Build request
        Boltz2Client = self._client_module["Boltz2Client"]
        Polymer = self._client_module["Polymer"]
        Ligand = self._client_module["Ligand"]
        PredictionRequest = self._client_module["PredictionRequest"]
        PocketConstraint = self._client_module["PocketConstraint"]
        AlignmentFileRecord = self._client_module["AlignmentFileRecord"]

        client = Boltz2Client(**_boltz2_client_kwargs(base_url, api_key=api_key))

        try:
            polymer_kwargs = {"id": "A", "molecule_type": "protein", "sequence": sequence}
            if msa_content:
                polymer_kwargs["msa"] = {
                    "Uniref30_2302": {
                        "a3m": AlignmentFileRecord(alignment=msa_content, format="a3m")
                    }
                }
            polymer = Polymer(**polymer_kwargs)

            ligand = Ligand(id="B", smiles=smiles, predict_affinity=True)

            pocket_fields = getattr(PocketConstraint, "model_fields", {})
            if "contacts" in pocket_fields:
                # boltz2-python-client>=0.5.2 uses Contact records instead of
                # the older ligand_id/polymer_id/residue_ids pocket shape.
                pocket = PocketConstraint(
                    binder="B",
                    contacts=[
                        {"id": "A", "residue_index": int(residue_id)}
                        for residue_id in binding_sites
                    ]
                )
            else:
                pocket = PocketConstraint(
                    ligand_id="B",
                    polymer_id="A",
                    residue_ids=binding_sites,
                    binder="B"
                )

            request = PredictionRequest(
                polymers=[polymer],
                ligands=[ligand],
                pocket_constraints=[pocket]
            )

            response = await client.predict(request)

            # Parse response
            if hasattr(response, "affinities") and response.affinities and "B" in response.affinities:
                affinity = response.affinities["B"]
                pic50_list = getattr(affinity, "affinity_pic50", None)
                pic50 = pic50_list[0] if pic50_list else None
                ic50_nm = (10 ** (-pic50)) * 1e9 if pic50 else None
                conf_list = getattr(affinity, "affinity_probability_binary", None)
                confidence = conf_list[0] if conf_list else None

                result = {
                    "ic50_nm": ic50_nm,
                    "pic50": pic50,
                    "confidence": confidence,
                    "endpoint": base_url,
                    "success": True
                }

                # Extract structure data if requested
                if return_structures:
                    structures = self._extract_structures(response, verbose=False)
                    result["structures"] = structures
                    result["_has_structures"] = len(structures) > 0

                return result
            else:
                return {"error": f"No affinity data in response from {base_url}", "endpoint": base_url, "success": False}

        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "endpoint": base_url, "success": False}
        finally:
            if ep_idx is not None:
                self._endpoint_busy[ep_idx] -= 1

    def _extract_structures(self, response: Any, verbose: bool = False) -> List[Dict[str, str]]:
        """Extract structure data (mmCIF) from Boltz2 response.

        Handles multiple formats:
        - Direct mmCIF string in response
        - File paths to CIF files
        - Nested structure objects

        Args:
            response: Boltz2 prediction response
            verbose: Print debug info

        Returns:
            List of dicts with structure data: [{"mmcif": str, "idx": int}, ...]
        """
        from pathlib import Path
        structures = []

        # Debug: print response attributes
        if verbose:
            attrs = [attr for attr in dir(response) if not attr.startswith('_')]
            print(f"    [DEBUG] Response attributes: {attrs}")

        raw_structures = getattr(response, "structures", None)
        if not raw_structures:
            if verbose:
                print(f"    [DEBUG] No 'structures' attribute in response")
            return structures

        if verbose:
            print(f"    [DEBUG] Found {len(raw_structures)} raw structures")

        for idx, structure in enumerate(raw_structures):
            cif_data = None
            structure_path = None

            # Debug: print structure type and attributes
            if verbose and idx == 0:
                print(f"    [DEBUG] Structure type: {type(structure)}")
                if hasattr(structure, '__dict__'):
                    print(f"    [DEBUG] Structure attrs: {list(structure.__dict__.keys())}")
                if hasattr(structure, 'model_dump'):
                    sd = structure.model_dump()
                    print(f"    [DEBUG] Structure dict keys: {list(sd.keys())}")
                    # Show format and whether structure has content
                    print(f"    [DEBUG] format={sd.get('format')}, structure_len={len(sd.get('structure', '') or '')}")

            # Try to get mmCIF data directly
            # boltz2_client.models.StructureData has: format, structure, name, source
            if hasattr(structure, "structure") and structure.structure:
                cif_data = structure.structure  # This is where boltz2_client stores CIF data
            elif hasattr(structure, "mmcif") and structure.mmcif:
                cif_data = structure.mmcif
            elif hasattr(structure, "mmCIF") and structure.mmCIF:
                cif_data = structure.mmCIF

            # Try to get file path
            if hasattr(structure, "path") and structure.path:
                structure_path = structure.path
            elif hasattr(structure, "file_path") and structure.file_path:
                structure_path = structure.file_path
            elif hasattr(structure, "source") and structure.source:
                # boltz2_client might use 'source' as file path
                structure_path = structure.source

            # Try model_dump() for Pydantic models
            if hasattr(structure, "model_dump"):
                structure_dict = structure.model_dump()
                cif_data = cif_data or structure_dict.get("structure") or structure_dict.get("mmcif") or structure_dict.get("mmCIF") or structure_dict.get("cif")
                structure_path = structure_path or structure_dict.get("path") or structure_dict.get("file_path") or structure_dict.get("source")
            elif isinstance(structure, dict):
                cif_data = cif_data or structure.get("structure") or structure.get("mmcif") or structure.get("mmCIF") or structure.get("cif")
                structure_path = structure_path or structure.get("path") or structure.get("file_path") or structure.get("source")

            # Also try direct string (some APIs return CIF as string directly)
            if not cif_data and isinstance(structure, str):
                if structure.startswith("data_"):
                    cif_data = structure
                elif Path(structure).exists():
                    structure_path = structure

            # If we have a file path, read the content
            if not cif_data and structure_path:
                try:
                    path_obj = Path(structure_path)
                    if path_obj.exists():
                        cif_data = path_obj.read_text()
                        if verbose:
                            print(f"    [DEBUG] Read structure from file: {structure_path}")
                except Exception as e:
                    if verbose:
                        print(f"    [DEBUG] Failed to read structure file {structure_path}: {e}")

            if cif_data:
                structures.append({
                    "mmcif": cif_data,
                    "idx": idx
                })
                if verbose:
                    print(f"    [DEBUG] Extracted structure {idx} ({len(cif_data)} chars)")
            elif verbose:
                print(f"    [DEBUG] Structure {idx} has no mmCIF data (path={structure_path})")

        return structures

    def predict_affinity(
        self,
        smiles: str,
        protein: str,
        use_msa: bool = True,
        return_structures: bool = False
    ) -> Dict[str, Any]:
        """Synchronous wrapper for predict_affinity_async."""
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.predict_affinity_async(smiles, protein, use_msa, return_structures=return_structures)
        )

    def predict_batch(
        self,
        smiles_list: List[str],
        proteins: List[str] = None,
        use_msa: bool = True,
        verbose: bool = True,
        parallel: bool = True,
        max_concurrent: int = None,
        return_structures: bool = False
    ) -> List[Dict[str, Any]]:
        """Predict affinities for multiple compounds.

        Supports parallel predictions across multiple endpoints.

        Args:
            smiles_list: List of SMILES strings
            proteins: List of proteins (default: [on_target, anti_target])
            use_msa: Whether to use MSA
            verbose: Print progress
            parallel: Enable parallel predictions (uses multiple endpoints)
            max_concurrent: Max concurrent requests (default: num_endpoints * 2)
            return_structures: Whether to include structure data (mmCIF) in results

        Returns:
            List of result dictionaries
        """
        if proteins is None:
            proteins = [self.config.on_target, self.config.anti_target]

        # Use parallel if multiple endpoints available
        if parallel and len(self.endpoints) > 1:
            return self.predict_batch_parallel(
                smiles_list, proteins, use_msa, verbose, max_concurrent, return_structures
            )

        # Sequential fallback
        results = []
        for i, smiles in enumerate(smiles_list):
            if verbose:
                print(f"Predicting {i+1}/{len(smiles_list)}: {smiles[:30]}...")

            result = {"smiles": smiles}
            for protein in proteins:
                pred = self.predict_affinity(smiles, protein, use_msa, return_structures)
                result[f"{protein}_IC50_pred"] = pred.get("ic50_nm")
                result[f"{protein}_pIC50_pred"] = pred.get("pic50")
                result[f"{protein}_confidence"] = pred.get("confidence")
                result[f"{protein}_success"] = pred.get("success", False)
                if return_structures and pred.get("structures"):
                    result[f"{protein}_structures"] = pred["structures"]

            results.append(result)

        return results

    def predict_batch_parallel(
        self,
        smiles_list: List[str],
        proteins: List[str] = None,
        use_msa: bool = True,
        verbose: bool = True,
        max_concurrent: int = None,
        return_structures: bool = False
    ) -> List[Dict[str, Any]]:
        """Predict affinities in parallel using multiple endpoints.

        Distributes predictions across all available endpoints for faster throughput.

        Args:
            smiles_list: List of SMILES strings
            proteins: List of proteins
            use_msa: Whether to use MSA
            verbose: Print progress
            max_concurrent: Max concurrent requests
            return_structures: Whether to include structure data (mmCIF) in results

        Returns:
            List of result dictionaries (preserves order)
        """
        if proteins is None:
            proteins = [self.config.on_target, self.config.anti_target]

        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

        import time
        start_time = time.time()

        async def run_parallel():
            # Shared work queue: each item is (compound_idx, smiles, protein)
            task_queue = asyncio.Queue()
            for i, smi in enumerate(smiles_list):
                for protein in proteins:
                    task_queue.put_nowait((i, smi, protein))

            total_tasks = task_queue.qsize()
            all_results = []
            completed = [0]  # mutable counter for progress

            if verbose:
                print(f"Running parallel predictions with {len(self.endpoints)} endpoint(s)")
                print(f"  Strategy: 1 dedicated worker per endpoint (work-stealing queue)")
                print(f"  Total tasks: {total_tasks} ({len(smiles_list)} compounds × {len(proteins)} proteins)")

            async def endpoint_worker(ep_idx: int):
                """Dedicated worker for one endpoint — pulls tasks until queue is empty."""
                endpoint = self.endpoints[ep_idx]
                ep_url = endpoint.get("url", self.base_url)
                ep_short = ep_url.split("//")[-1]
                task_count = 0

                while True:
                    try:
                        cidx, smi, protein = task_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    prot_start = time.time()
                    try:
                        pred = await self.predict_affinity_async(
                            smi, protein, use_msa,
                            return_structures=return_structures,
                            endpoint=endpoint
                        )
                    except Exception as e:
                        pred = {"error": str(e), "success": False, "endpoint": ep_url}

                    pred["_time_s"] = time.time() - prot_start
                    pred["_compound_idx"] = cidx
                    pred["_protein"] = protein
                    pred["_smiles"] = smi
                    all_results.append(pred)
                    task_count += 1
                    completed[0] += 1

                    err = pred.get("error")
                    if verbose:
                        elapsed = time.time() - start_time
                        if err:
                            print(f"    [{completed[0]}/{total_tasks}] {protein}({smi[:20]}...) → {ep_short} ERROR: {err[:80]}")
                        else:
                            print(f"    [{completed[0]}/{total_tasks}] {protein}({smi[:20]}...) → {ep_short} {pred['_time_s']:.1f}s @{elapsed:.0f}s")

                    if err:
                        await asyncio.sleep(0)  # yield to other workers on error

                return ep_short, task_count

            # One worker per endpoint — they race to drain the queue
            worker_results = await asyncio.gather(
                *[endpoint_worker(i) for i in range(len(self.endpoints))]
            )
            return all_results, worker_results

        loop = asyncio.get_event_loop()
        all_results, worker_stats = loop.run_until_complete(run_parallel())

        total_time = time.time() - start_time

        # Reassemble flat (compound, protein) results into per-compound dicts
        compound_results = {}
        for r in all_results:
            cidx = r.get("_compound_idx", 0)
            protein = r.get("_protein", "")
            if cidx not in compound_results:
                compound_results[cidx] = {"smiles": r.get("_smiles", ""), "_idx": cidx}
            cr = compound_results[cidx]
            cr[f"{protein}_IC50_pred"] = r.get("ic50_nm")
            cr[f"{protein}_pIC50_pred"] = r.get("pic50")
            cr[f"{protein}_confidence"] = r.get("confidence")
            cr[f"{protein}_success"] = r.get("success", False)
            if return_structures and r.get("structures"):
                cr[f"{protein}_structures"] = r["structures"]
            cr[f"{protein}_endpoint"] = r.get("endpoint", "?")
            cr[f"{protein}_time_s"] = r.get("_time_s", 0)

        processed = [compound_results[i] for i in sorted(compound_results.keys())]

        if verbose:
            ep_counts = {}
            for cr in processed:
                endpoint_parts = []
                for protein in proteins:
                    ep = cr.get(f"{protein}_endpoint", "?")
                    ep_short = ep.split("//")[-1] if ep else "?"
                    t = cr.get(f"{protein}_time_s", 0)
                    endpoint_parts.append(f"{protein}→{ep_short} {t:.1f}s")
                    ep_counts[ep_short] = ep_counts.get(ep_short, 0) + 1
                smi = cr.get("smiles", "")

            print(f"\n  ⚡ Parallel execution complete:")
            print(f"     Total time: {total_time:.1f}s for {len(smiles_list)} compounds ({len(smiles_list)*len(proteins)} requests)")
            for ep_name, count in worker_stats:
                print(f"     {ep_name}: {count} tasks")

        for r in processed:
            r.pop("_idx", None)

        return processed

    def check_endpoints_health(self, verbose: bool = True) -> Dict[int, bool]:
        """Check health of all configured endpoints.

        Args:
            verbose: Print status

        Returns:
            Dict mapping endpoint index to health status
        """
        status = {}

        if verbose:
            print(f"Checking {len(self.endpoints)} Boltz2 endpoint(s)...")

        for i, endpoint in enumerate(self.endpoints):
            url = endpoint.get("url")
            api_key = endpoint.get("api_key", self.api_key)

            try:
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                response = requests.get(f"{url}/v1/health/ready", headers=headers, timeout=5)
                healthy = response.status_code == 200
                status[i] = healthy

                if verbose:
                    icon = "✓" if healthy else "✗"
                    print(f"  {icon} Endpoint {i+1}: {url} - {'healthy' if healthy else 'unreachable'}")
            except Exception as e:
                status[i] = False
                if verbose:
                    print(f"  ✗ Endpoint {i+1}: {url} - error: {e}")

        return status
