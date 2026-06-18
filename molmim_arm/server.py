# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
FastAPI MolMIM service (NIM-like) for aarch64 + Blackwell.

Mirrors the MolMIM NIM REST surface so the bootcamp client and CMA-ES workflow
run unmodified against it via MOLMIM_URL:

    GET  /v1/health/live    -> 200 once the process is up
    GET  /v1/health/ready   -> 200 when the model is loaded, else 503
    GET  /v1/metadata       -> basic model/service info
    POST /hidden            -> {"sequences": [...]} -> {"hiddens": [...], "mask": [...]}
    POST /decode            -> {"hiddens": [...], "mask": [...]} -> {"generated": [...]}
    POST /sampling          -> {"smiles": "...", "num_samples": N} -> {"generated": [...], "samples": [...]}
    POST /generate          -> {"smi": "...", "num_molecules": N} -> {"molecules": [...]} (hosted-style)

Interactive OpenAPI docs are served at /docs (like the BioNeMo NIMs).
Backed by the pure-PyTorch MolMIM (molmim_torch) so no NeMo/Megatron/apex/TE.
"""

from __future__ import annotations

import os
import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engine import MolMIMEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molmim_arm.server")

app = FastAPI(
    title="MolMIM ARM NIM",
    version="1.0.0",
    description="NIM-like MolMIM microservice for aarch64 + Blackwell (pure-PyTorch).",
)

_STATE: Dict[str, Any] = {"engine": None, "ready": False, "error": None}


def _load_engine() -> None:
    try:
        engine = MolMIMEngine()
        engine.load()
        _STATE["engine"] = engine
        _STATE["ready"] = True
        logger.info("MolMIM engine loaded; service is ready.")
    except Exception as exc:  # pragma: no cover - depends on runtime
        _STATE["error"] = str(exc)
        logger.exception("Failed to load MolMIM engine: %s", exc)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_engine, name="molmim-load", daemon=True).start()


def _engine_or_503():
    engine = _STATE.get("engine")
    if engine is None or not _STATE.get("ready"):
        return None
    return engine


# -- request models --------------------------------------------------------


class HiddenRequest(BaseModel):
    sequences: List[str] = Field(..., description="SMILES strings to encode.")


class DecodeRequest(BaseModel):
    hiddens: List[Any] = Field(..., description="Latent array, shape (n, 1, 512) or (n, 512).")
    mask: Optional[List[Any]] = None


class SamplingRequest(BaseModel):
    smiles: Optional[str] = None
    sequences: Optional[List[str]] = None
    num_samples: int = 10
    scaled_radius: float = 1.0
    temperature: float = 1.0


class GenerateRequest(BaseModel):
    smi: Optional[str] = None
    sequences: Optional[List[str]] = None
    num_molecules: int = 10
    scaled_radius: float = 1.0
    algorithm: Optional[str] = None


# -- health / metadata -----------------------------------------------------


@app.get("/v1/health/live")
def health_live() -> JSONResponse:
    return JSONResponse({"status": "live"}, status_code=200)


@app.get("/v1/health/ready")
def health_ready() -> JSONResponse:
    if _STATE.get("ready"):
        return JSONResponse({"status": "ready"}, status_code=200)
    return JSONResponse({"status": "not ready", "detail": _STATE.get("error") or "loading"}, status_code=503)


@app.get("/v1/metadata")
def metadata() -> Dict[str, Any]:
    eng = _STATE.get("engine")
    return {
        "model": "molmim_70m_24_3",
        "backend": "pytorch (pure)",
        "arch": "aarch64",
        "latent_dim": 512,
        "ready": bool(_STATE.get("ready")),
        "device": str(getattr(getattr(eng, "_runner", None), "device", "unknown")),
    }


# -- inference -------------------------------------------------------------


@app.post("/hidden")
def hidden(req: HiddenRequest) -> JSONResponse:
    if not req.sequences:
        return JSONResponse({"detail": "sequences must be non-empty"}, status_code=400)
    engine = _engine_or_503()
    if engine is None:
        return JSONResponse({"detail": _STATE.get("error") or "model not ready"}, status_code=503)
    return JSONResponse(engine.embed(req.sequences))


@app.post("/decode")
def decode(req: DecodeRequest) -> JSONResponse:
    if not req.hiddens:
        return JSONResponse({"detail": "hiddens must be non-empty"}, status_code=400)
    engine = _engine_or_503()
    if engine is None:
        return JSONResponse({"detail": _STATE.get("error") or "model not ready"}, status_code=503)
    return JSONResponse({"generated": engine.decode(req.hiddens, req.mask)})


@app.post("/sampling")
def sampling(req: SamplingRequest) -> JSONResponse:
    seed = req.smiles or (req.sequences[0] if req.sequences else None)
    if not seed:
        return JSONResponse({"detail": "smiles (or sequences) is required"}, status_code=400)
    engine = _engine_or_503()
    if engine is None:
        return JSONResponse({"detail": _STATE.get("error") or "model not ready"}, status_code=503)
    samples = engine.sample(seed, num_samples=req.num_samples, scaled_radius=req.scaled_radius)
    return JSONResponse({"generated": samples, "samples": samples})


@app.post("/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    seed = req.smi or (req.sequences[0] if req.sequences else None)
    if not seed:
        return JSONResponse({"detail": "smi (or sequences) is required"}, status_code=400)
    engine = _engine_or_503()
    if engine is None:
        return JSONResponse({"detail": _STATE.get("error") or "model not ready"}, status_code=503)
    samples = engine.sample(seed, num_samples=req.num_molecules, scaled_radius=req.scaled_radius)
    return JSONResponse(
        {
            "molecules": [{"sample": s, "smiles": s} for s in samples],
            "generated": samples,
            "samples": samples,
        }
    )


def main() -> None:
    import uvicorn

    port = int(os.environ.get("NIM_HTTP_API_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
