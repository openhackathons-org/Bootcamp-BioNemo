# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""
HTTP clients for the BioNeMo NIMs used by the protein-binder-design workflow:
RFdiffusion (backbones), ProteinMPNN (sequences), and Boltz-2 (co-fold).

Each NIM is reached over HTTP and works with either a hosted endpoint
(``https://health.api.nvidia.com/...`` with ``NVIDIA_API_KEY``) or a local
self-hosted NIM (``http://host:port/...``, no auth). Choose once via the
endpoint URLs and reuse for the whole campaign. Request/response shapes follow
the BioNeMo NIM REST contracts.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

HOSTED_ROOT = "https://health.api.nvidia.com"


def _api_key() -> Optional[str]:
    return (
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NGC_API_KEY")
        or os.environ.get("BOLTZ2_API_KEY")
    )


def _is_hosted(url: str) -> bool:
    return "api.nvidia.com" in url


def _headers(url: str) -> Dict[str, str]:
    h = {"accept": "application/json", "content-type": "application/json"}
    if _is_hosted(url):
        key = _api_key()
        if key:
            h["Authorization"] = f"Bearer {key}"
    return h


def _resolve(base: str, hosted_suffix: str, local_suffix: str, leaf: str) -> str:
    """Resolve a NIM endpoint from a base URL.

    hosted_suffix/local_suffix are the path prefixes (e.g. "/v1/biology/...");
    leaf is the operation (e.g. "generate"/"predict").
    """
    b = base.rstrip("/")
    if b.endswith(leaf):
        return b
    if b.endswith(hosted_suffix) or b.endswith(local_suffix):
        return f"{b}/{leaf}"
    if _is_hosted(b):
        # bare hosted root or .../v1
        if b.endswith("/v1"):
            return f"{b}{hosted_suffix[len('/v1'):]}/{leaf}"
        return f"{b}{hosted_suffix}/{leaf}"
    # local base like http://host:port (optionally .../v1)
    if b.endswith("/v1"):
        return f"{b}{local_suffix[len('/v1'):] if local_suffix.startswith('/v1') else local_suffix}/{leaf}"
    return f"{b}{local_suffix}/{leaf}"


def _post(url: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(url, headers=_headers(url), json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"{url} -> HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def health_ready(base: str, timeout: int = 8) -> bool:
    b = base.rstrip("/")
    # strip any known leaf/path to reach the host root for /v1/health/ready
    root = re.sub(r"/v1.*$|/biology.*$", "", b)
    for path in ("/v1/health/ready", "/health/ready"):
        try:
            r = requests.get(root + path, headers=_headers(base), timeout=timeout)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
    return False


# ---------------------------------------------------------------------------
# RFdiffusion
# ---------------------------------------------------------------------------
class RFdiffusionClient:
    """Generate binder backbones for a target + hotspots."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 300):
        self.base = (base_url or os.environ.get("RFDIFFUSION_URL") or HOSTED_ROOT).rstrip("/")
        self.url = _resolve(
            self.base, "/v1/biology/ipd/rfdiffusion", "/biology/ipd/rfdiffusion", "generate"
        )
        self.timeout = timeout

    def generate(
        self,
        input_pdb: str,
        contigs: str,
        hotspot_res: List[str],
        diffusion_steps: int = 15,
    ) -> str:
        payload = {
            "input_pdb": input_pdb,
            "contigs": contigs,
            "hotspot_res": hotspot_res,
            "diffusion_steps": int(diffusion_steps),
        }
        data = _post(self.url, payload, self.timeout)
        pdb = data.get("output_pdb") or data.get("pdb") or data.get("structure")
        if not pdb:
            raise RuntimeError(f"RFdiffusion: no output_pdb in response keys={list(data)}")
        return pdb


# ---------------------------------------------------------------------------
# ProteinMPNN
# ---------------------------------------------------------------------------
class ProteinMPNNClient:
    """Design sequences for a backbone (redesign the binder chain only)."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 300):
        self.base = (base_url or os.environ.get("PROTEINMPNN_URL") or HOSTED_ROOT).rstrip("/")
        self.url = _resolve(
            self.base, "/v1/biology/ipd/proteinmpnn", "/biology/ipd/proteinmpnn", "predict"
        )
        self.timeout = timeout

    def predict(
        self,
        input_pdb: str,
        input_pdb_chains: List[str],
        num_seq_per_target: int = 2,
        sampling_temp: Optional[List[float]] = None,
        use_soluble_model: bool = True,
    ) -> List[Tuple[str, str, Optional[float]]]:
        """Return designed (header, sequence, score) rows (native/WT row dropped)."""
        payload = {
            "input_pdb": input_pdb,
            "input_pdb_chains": input_pdb_chains,
            "num_seq_per_target": int(num_seq_per_target),
            "sampling_temp": sampling_temp or [0.1],
            "use_soluble_model": bool(use_soluble_model),
        }
        data = _post(self.url, payload, self.timeout)
        mfasta = data.get("mfasta") or data.get("fasta") or ""
        records = _parse_fasta(mfasta)
        designed = []
        for i, (header, seq) in enumerate(records):
            # ProteinMPNN: the first record is the native/input sequence; designs
            # carry "sample=" in the header. Drop native/WT.
            hl = header.lower()
            if i == 0 and "sample=" not in hl:
                continue
            if "native" in hl or "wt" in hl.split(","):
                continue
            score = _parse_score(header)
            designed.append((header, seq, score))  # keep raw (chains joined by '/')
        if not designed:  # fallback: keep all but the first
            designed = [(h, s, _parse_score(h)) for h, s in records[1:]]
        return designed


def binder_subsequence(raw_seq: str, chain_order: List[str], binder_chain: str) -> str:
    """Extract the binder chain's sequence from a ProteinMPNN multi-chain row.

    ProteinMPNN joins chains with '/'. Pick the segment whose position matches
    the binder chain's index in ``chain_order``; fall back to the whole string.
    """
    if "/" not in raw_seq:
        return raw_seq
    segs = raw_seq.split("/")
    try:
        idx = chain_order.index(binder_chain)
    except ValueError:
        idx = 0
    if idx < len(segs):
        return segs[idx]
    return max(segs, key=len)


# ---------------------------------------------------------------------------
# Boltz-2 (co-fold a binder + target complex)
# ---------------------------------------------------------------------------
class Boltz2StructureClient:
    """Predict the 3D structure of a protein complex and report confidence."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 600):
        self.base = (base_url or os.environ.get("BOLTZ2_URL") or HOSTED_ROOT).rstrip("/")
        self.url = _resolve(self.base, "/v1/biology/mit/boltz2", "/biology/mit/boltz2", "predict")
        self.timeout = timeout

    def predict_complex(
        self,
        chains: List[Tuple[str, str]],
        recycling_steps: int = 3,
        sampling_steps: int = 25,
        diffusion_samples: int = 1,
        step_scale: float = 1.638,
    ) -> Dict[str, Any]:
        """chains: list of (chain_id, sequence). Returns dict with cif + scores."""
        polymers = [
            {"id": cid, "molecule_type": "protein", "sequence": seq} for cid, seq in chains
        ]
        payload = {
            "polymers": polymers,
            "recycling_steps": int(recycling_steps),
            "sampling_steps": int(sampling_steps),
            "diffusion_samples": int(diffusion_samples),
            "step_scale": float(step_scale),
            "output_format": "mmcif",
        }
        data = _post(self.url, payload, self.timeout)
        cif = _first_structure(data)
        conf = _first_number(data, ("confidence_scores", "confidence", "complex_confidence"))
        # Boltz-2 reports protein-protein interface PTM as protein_iptm_scores /
        # iptm_scores; prefer the protein-protein interface for binder design.
        iptm = _first_number(
            data,
            ("protein_iptm_scores", "iptm_scores", "iptm", "iptm_score", "interface_ptm"),
        )
        ptm = _first_number(data, ("ptm_scores", "ptm", "ptm_score"))
        plddt = _first_number(data, ("complex_plddt_scores", "plddt"))
        return {
            "cif": cif, "confidence": conf, "iptm": iptm, "ptm": ptm,
            "complex_plddt": plddt, "raw_keys": list(data),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _parse_fasta(text: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    seq_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        elif line.strip():
            seq_lines.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _parse_score(header: str) -> Optional[float]:
    m = re.search(r"global_score=\s*([0-9.eE+-]+)", header) or re.search(
        r"score=\s*([0-9.eE+-]+)", header
    )
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _first_structure(data: dict) -> Optional[str]:
    for key in ("structures", "structure", "cif", "output_cif", "predictions"):
        v = data.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            return v
        if isinstance(v, list) and v:
            item = v[0]
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                for sk in ("structure", "cif", "mmcif", "content", "data"):
                    if isinstance(item.get(sk), str):
                        return item[sk]
        if isinstance(v, dict):
            for sk in ("structure", "cif", "mmcif", "content"):
                if isinstance(v.get(sk), str):
                    return v[sk]
    return None


def _first_number(data: dict, keys) -> Optional[float]:
    for key in keys:
        if key not in data:
            continue
        v = data[key]
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return float(v[0])
    return None
