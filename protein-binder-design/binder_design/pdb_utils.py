# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR CC-BY-4.0
"""Dependency-free PDB/mmCIF parsing helpers for binder-design handoffs.

Adapted for the AI-Powered-Drug-Discovery-Bootcamp from the NVIDIA BioNeMo
agent-toolkit ``protein-binder-design`` skill. Covers the fragile glue between
NIM steps:

- extract a chain, keep ATOM records
- one-letter sequence from CA atoms
- map PDB author residue numbers -> 1-based sequence index (hotspot/pocket remap)
- CA coordinates for RMSD
- list chains; parse CA atoms (coords + pLDDT/B-factor) from Boltz-2 mmCIF output
"""
from __future__ import annotations

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def _iter_atom_lines(pdb_text, chain=None):
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        if chain is not None and line[21] != chain:
            continue
        yield line


def extract_chain(pdb_text, chain):
    """Return PDB text containing only records for ``chain``."""
    keep = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) > 21 and line[21] == chain:
            keep.append(line)
    return "\n".join(keep)


def chain_ids(pdb_text):
    """Ordered unique chain IDs present in ATOM records."""
    seen = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and len(line) > 21:
            c = line[21]
            if c not in seen:
                seen.append(c)
    return seen


def ca_residues(pdb_text, chain=None):
    """Ordered list of (resName, resSeq, iCode, (x, y, z)) for CA atoms."""
    out = []
    for line in _iter_atom_lines(pdb_text, chain):
        if line[12:16].strip() != "CA":
            continue
        res_name = line[17:20].strip()
        res_seq = int(line[22:26])
        icode = line[26].strip()
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        out.append((res_name, res_seq, icode, (x, y, z)))
    return out


def sequence(pdb_text, chain=None):
    """One-letter sequence from CA atoms (unknown residues -> 'X')."""
    return "".join(THREE_TO_ONE.get(r[0], "X") for r in ca_residues(pdb_text, chain))


def residue_index_map(pdb_text, chain=None):
    """Map PDB author residue id -> 1-based sequence index (CA order)."""
    mapping = {}
    for i, (_, res_seq, icode, _) in enumerate(ca_residues(pdb_text, chain), start=1):
        mapping[str(res_seq)] = i
        if icode:
            mapping[f"{res_seq}{icode}"] = i
    return mapping


def remap_to_seq_index(pdb_text, chain, author_resnums):
    """Convert PDB author residue numbers to 1-based sequence indices."""
    mapping = residue_index_map(pdb_text, chain)
    out, missing = [], []
    for a in author_resnums:
        key = str(a)
        if key in mapping:
            out.append(mapping[key])
        else:
            missing.append(key)
    if missing:
        raise KeyError(f"residues not found in chain {chain!r}: {missing}")
    return out


def ca_coords(pdb_text, chain=None):
    """List of (x, y, z) for CA atoms in chain order."""
    return [r[3] for r in ca_residues(pdb_text, chain)]


# ---------------------------------------------------------------------------
# mmCIF (Boltz-2 / OpenFold3 co-fold output)
# ---------------------------------------------------------------------------
def _mmcif_atom_site_rows(cif_text):
    """Yield dict rows of the _atom_site loop from mmCIF text."""
    lines = cif_text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line == "loop_":
            cols = []
            j = i + 1
            while j < n and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols:
                idx = {c: k for k, c in enumerate(cols)}
                k = j
                while k < n:
                    row = lines[k].strip()
                    if row.startswith("_") or row == "loop_" or row.startswith("#") or row == "":
                        break
                    parts = row.split()
                    if len(parts) >= len(cols):
                        yield idx, parts
                    k += 1
                i = k
                continue
        i += 1


def mmcif_ca(cif_text, chain=None, chain_key="auth_asym_id"):
    """CA atoms from mmCIF as list of (chain, resseq, (x,y,z), bfactor/pLDDT)."""
    out = []
    want_keys = (
        f"_atom_site.{chain_key}",
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
    )
    for idx, parts in _mmcif_atom_site_rows(cif_text):
        # atom name
        an_k = idx.get("_atom_site.label_atom_id") or idx.get("_atom_site.auth_atom_id")
        if an_k is None or parts[an_k].strip('"') != "CA":
            continue
        ch_k = None
        for wk in want_keys:
            if wk in idx:
                ch_k = idx[wk]
                break
        ch = parts[ch_k] if ch_k is not None else "?"
        if chain is not None and ch != chain:
            continue
        try:
            x = float(parts[idx["_atom_site.Cartn_x"]])
            y = float(parts[idx["_atom_site.Cartn_y"]])
            z = float(parts[idx["_atom_site.Cartn_z"]])
        except (KeyError, ValueError):
            continue
        b = 0.0
        bk = idx.get("_atom_site.B_iso_or_equiv")
        if bk is not None:
            try:
                b = float(parts[bk])
            except ValueError:
                b = 0.0
        rk = idx.get("_atom_site.auth_seq_id") or idx.get("_atom_site.label_seq_id")
        try:
            rs = int(parts[rk]) if rk is not None else len(out) + 1
        except ValueError:
            rs = len(out) + 1
        out.append((ch, rs, (x, y, z), b))
    return out


def mmcif_chain_ids(cif_text):
    return sorted({r[0] for r in mmcif_ca(cif_text)})


def mmcif_ca_coords(cif_text, chain):
    return [r[2] for r in mmcif_ca(cif_text, chain)]


def mmcif_mean_plddt(cif_text, chain):
    bs = [r[3] for r in mmcif_ca(cif_text, chain)]
    return sum(bs) / len(bs) if bs else 0.0
