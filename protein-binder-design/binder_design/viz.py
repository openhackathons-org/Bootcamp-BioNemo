# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""Visualization helpers for the protein-binder-design workflow.

3D molecular views use py3Dmol (interactive in JupyterLab); assessment charts
use matplotlib. All functions degrade gracefully if py3Dmol is unavailable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 3D complex views (Mol* via ipymolstar) -- preferred for binder complexes
# ---------------------------------------------------------------------------
_MOLSTAR_UI = dict(
    hide_water=True,
    hide_carbs=True,
    hide_controls_icon=True,
    hide_expand_icon=True,
    hide_settings_icon=True,
    hide_animation_icon=True,
)


def molstar_structure(data, fmt="cif", binary=False, color_chains=None, **ui):
    """Mol* (PDBeMolstar) view of inline structure data.

    data: structure text (mmCIF/PDB string) or bytes (bcif).
    color_chains: optional {struct_asym_id: '#hex'} mapping.
    """
    from ipymolstar import PDBeMolstar

    if isinstance(data, str) and binary:
        data = data.encode()
    custom_data = {"data": data, "format": fmt, "binary": bool(binary)}
    flags = dict(_MOLSTAR_UI)
    flags.update(ui)
    view = PDBeMolstar(custom_data=custom_data, **flags)
    if color_chains:
        view.color_data = {
            "data": [{"struct_asym_id": c, "color": col} for c, col in color_chains.items()],
            "nonSelectedColor": None,
        }
    return view


def molstar_complex(
    cif_text,
    binder_chain="A",
    target_chain="B",
    hotspot_seq_idx: Optional[Sequence[int]] = None,
    binder_color="#ff8c00",
    target_color="#cfd8dc",
    hotspot_color="#e53935",
):
    """Mol* view of a co-folded binder+target complex.

    Binder is highlighted in orange, target in light gray, and (optionally) the
    target hotspot residues (given as 1-based sequence indices in the predicted
    structure) in red.
    """
    view = molstar_structure(cif_text, fmt="cif", binary=False)
    data = [
        {"struct_asym_id": target_chain, "color": target_color},
        {"struct_asym_id": binder_chain, "color": binder_color},
    ]
    if hotspot_seq_idx:
        for i in hotspot_seq_idx:
            data.append(
                {"struct_asym_id": target_chain, "residue_number": int(i), "color": hotspot_color}
            )
    view.color_data = {"data": data, "nonSelectedColor": None}
    return view


# ---------------------------------------------------------------------------
# 3D structure views (py3Dmol) -- simple target / backbone views
# ---------------------------------------------------------------------------
def _view(width=720, height=480):
    import py3Dmol

    return py3Dmol.view(width=width, height=height)


def show_target_with_hotspots(pdb_text, chain, hotspot_resnums, width=720, height=480):
    """Cartoon of the target with hotspot residues highlighted as red sticks."""
    v = _view(width, height)
    v.addModel(pdb_text, "pdb")
    v.setStyle({"chain": chain}, {"cartoon": {"color": "lightblue"}})
    for r in hotspot_resnums:
        sel = {"chain": chain, "resi": int(r)}
        v.addStyle(sel, {"stick": {"colorscheme": "redCarbon", "radius": 0.3}})
        v.addStyle(sel, {"cartoon": {"color": "red"}})
    v.zoomTo({"chain": chain})
    return v


def show_complex(cif_text, binder_chain="A", target_chain="B", width=720, height=480):
    """Cartoon of a co-folded complex: binder (orange) + target (light gray)."""
    v = _view(width, height)
    v.addModel(cif_text, "mmcif")
    v.setStyle({"chain": target_chain}, {"cartoon": {"color": "lightgray"}})
    v.setStyle({"chain": binder_chain}, {"cartoon": {"color": "orange"}})
    v.zoomTo()
    return v


def show_backbone(pdb_text, width=720, height=480):
    """Cartoon of an RFdiffusion backbone, colored by chain."""
    v = _view(width, height)
    v.addModel(pdb_text, "pdb")
    v.setStyle({}, {"cartoon": {"colorscheme": "chainHetatm"}})
    v.zoomTo()
    return v


# ---------------------------------------------------------------------------
# assessment charts (matplotlib)
# ---------------------------------------------------------------------------
def plot_design_funnel(counts: dict, ax=None):
    """Bar chart of how many candidates survive each stage of the funnel."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    stages = list(counts.keys())
    vals = [counts[s] for s in stages]
    bars = ax.bar(stages, vals, color="#76b900")
    ax.set_ylabel("count")
    ax.set_title("Binder design funnel")
    ax.bar_label(bars)
    ax.tick_params(axis="x", rotation=20)
    return ax


def plot_iptm_vs_plddt(designs, controls=None, iptm_cut=0.8, plddt_cut=80.0, ax=None):
    """Scatter of interface confidence (ipTM/complex confidence) vs binder pLDDT."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    dx = [d["iptm"] for d in designs]
    dy = [d["binder_plddt"] for d in designs]
    ax.scatter(dx, dy, c="#76b900", edgecolor="k", s=70, label="designs", zorder=3)
    if controls:
        cx = [c["iptm"] for c in controls]
        cy = [c["binder_plddt"] for c in controls]
        ax.scatter(cx, cy, c="#bbbbbb", edgecolor="k", marker="^", s=70,
                   label="scrambled controls", zorder=2)
    ax.axvline(iptm_cut, ls="--", c="r", alpha=0.6)
    ax.axhline(plddt_cut, ls="--", c="r", alpha=0.6)
    ax.set_xlabel("interface confidence (ipTM / Boltz-2 confidence)")
    ax.set_ylabel("binder pLDDT")
    ax.set_title("Interface confidence vs binder pLDDT")
    ax.legend()
    return ax


def plot_rmsd_hist(designs, controls=None, rmsd_cut=2.0, ax=None):
    """Histogram of self-consistency CA-RMSD (backbone vs predicted binder)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    d = [x["self_consistency_rmsd"] for x in designs if x.get("self_consistency_rmsd") is not None]
    if d:
        ax.hist(d, bins=min(12, max(3, len(d))), color="#76b900", alpha=0.85, label="designs")
    if controls:
        c = [x["self_consistency_rmsd"] for x in controls if x.get("self_consistency_rmsd") is not None]
        if c:
            ax.hist(c, bins=min(12, max(3, len(c))), color="#bbbbbb", alpha=0.7,
                    label="scrambled controls")
    ax.axvline(rmsd_cut, ls="--", c="r", alpha=0.6, label=f"cutoff {rmsd_cut} A")
    ax.set_xlabel("self-consistency CA-RMSD (A)")
    ax.set_ylabel("count")
    ax.set_title("Self-consistency RMSD")
    ax.legend()
    return ax


def plot_ranked(designs, by="iptm", top=15, ax=None):
    """Horizontal bar of the top designs by a chosen score."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ranked = sorted(
        [d for d in designs if d.get(by) is not None], key=lambda d: d[by], reverse=True
    )[:top]
    labels = [d["id"] for d in ranked][::-1]
    vals = [d[by] for d in ranked][::-1]
    bars = ax.barh(labels, vals, color="#76b900")
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xlabel(by)
    ax.set_title(f"Top {len(ranked)} designs by {by}")
    return ax
