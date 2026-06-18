# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
MolMIM inference engine for the ARM (Grace/GB200/GB300) NIM-like service.

This wraps the pure-PyTorch MolMIM reimplementation (``molmim_torch``) which
loads the official ``molmim_70m_24_3`` checkpoint and runs entirely on stock
PyTorch -- so it works natively on aarch64 + Blackwell, where the MolMIM NIM
(amd64-only) and the BioNeMo Framework v1 stack cannot run.

It produces the same JSON shapes the MolMIM NIM REST API uses, so the bootcamp
client (``cdk_oracle/nim_client.py``) and the CMA-ES guided-optimization
workflow run unmodified against it via ``MOLMIM_URL``.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("molmim_arm.engine")


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


class MolMIMEngine:
    """Lazy wrapper around the pure-torch MolMIMRunner."""

    def __init__(self) -> None:
        self.checkpoint = _env("MOLMIM_CHECKPOINT", "/models/molmim_70m_24_3.nemo")
        self.device = _env("MOLMIM_DEVICE", "cuda")
        # decode method for /sampling and /generate: greedy or topkp
        self.sampling_method = _env("MOLMIM_SAMPLING_METHOD", "greedy")
        self._runner = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._runner is not None:
            return
        from molmim_torch import MolMIMRunner

        if not os.path.exists(self.checkpoint):
            raise RuntimeError(
                f"MolMIM checkpoint not found at {self.checkpoint}. Mount the "
                "molmim_70m_24_3.nemo there or set MOLMIM_CHECKPOINT."
            )
        logger.info("Loading MolMIM checkpoint %s on %s", self.checkpoint, self.device)
        self._runner = MolMIMRunner(self.checkpoint, device=self.device)
        logger.info("MolMIM model is ready (device=%s).", self._runner.device)

    @property
    def ready(self) -> bool:
        return self._runner is not None

    # -- operations --------------------------------------------------------

    def embed(self, sequences: Sequence[str]) -> dict:
        """Encode SMILES -> latent z_mean. Returns NIM-style /hidden payload."""
        self.load()
        z = self._runner.seq_to_hiddens(list(sequences))  # [n, 1, H]
        z_np = z.detach().cpu().float().numpy()
        mask = [[True] for _ in range(z_np.shape[0])]
        return {"hiddens": z_np.tolist(), "mask": mask}

    def decode(self, hiddens, mask: Optional[list] = None) -> List[str]:
        """Decode latent representations back into SMILES strings."""
        self.load()
        import torch

        arr = np.asarray(hiddens, dtype=np.float32)
        if arr.ndim == 2:  # (n, H) -> (n, 1, H)
            arr = np.expand_dims(arr, axis=1)
        return self._runner.hiddens_to_seq(torch.from_numpy(arr), method="greedy")

    def sample(self, smiles: str, num_samples: int = 10, scaled_radius: float = 1.0) -> List[str]:
        """Generate molecules near a seed by perturbing its latent code."""
        self.load()
        return self._runner.sample(
            smiles,
            num_samples=int(num_samples),
            scaled_radius=float(scaled_radius),
            method=self.sampling_method,
        )
