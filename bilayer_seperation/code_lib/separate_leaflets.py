#!/usr/bin/env python3
"""
separate_leaflets.py  (original approach + CLI)
------------------------------------------------
Pure, symmetric bilayer -> split into Upper/Lower leaflets:

METHOD (unchanged from your original `separate_by_resid.py`):
- Lipids: split by residue IDs (Upper = resid 1..N/2; Lower = N/2+1..N).
- TIP3 waters: assign by geometric Z mid-plane computed from lipid headgroup P atoms.
- Build each leaflet output with `MDAnalysis.Merge(...)` (this creates a NEW Universe).

Notes:
- This mirrors your original implementation choice (using Merge for waters + final write).
- Like the original, using `Merge` does not carry velocities into the new Universe.
  (If you later want velocities preserved, say the word and I’ll add a minimal copy-back
   step without changing the method.)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import MDAnalysis as mda
import argparse

WATER = "TIP3"  # per your spec

# --------- Helpers (same method, robust implementation) ---------
def _detect_single_lipid_resname(u: mda.Universe, water_name=WATER) -> str:
    """Return the single non-water lipid resname; error if not exactly one."""
    resnames = {r.resname for r in u.residues if r.resname != water_name}
    if len(resnames) != 1:
        raise ValueError(f"Expected exactly one lipid type, found: {sorted(resnames)}")
    return next(iter(resnames))

def _split_lipids_by_resid(u: mda.Universe, lipid_name: str) -> Tuple[mda.AtomGroup, mda.AtomGroup]:
    """Upper/Lower by resid halves (non-geometric)."""
    lip = u.select_atoms(f"resname {lipid_name}")
    n_res = len(lip.residues)
    if n_res < 2:
        raise ValueError("Not enough lipid residues to split into two halves.")
    half = n_res // 2
    upper = lip.residues[:half].atoms
    lower = lip.residues[half:].atoms
    return upper, lower

def _mean_z_of_headgroup_P(ag: mda.AtomGroup) -> float:
    """Mean Z (Å) of headgroup P-like atoms across molecules.
    Uses first residue to find the P* atom index, then strides by atoms_per_mol.
    """
    first = ag.residues[0]
    try:
        ref_idx_local = next(i for i, a in enumerate(first.atoms) if a.name.upper().startswith("P"))
    except StopIteration:
        # Fallback: try a selection by name pattern
        sel = first.atoms.select_atoms("name P*")
        if sel.n_atoms == 0:
            raise ValueError("Cannot locate P atom in lipid headgroup.")
        ref_idx_local = int(np.where(first.atoms.indices == sel.indices[0])[0][0])

    atoms_per_mol = len(first.atoms)
    z = ag.positions[ref_idx_local::atoms_per_mol, 2]  # Å
    return float(z.mean())

def _assign_waters_by_midplane(u: mda.Universe,
                               lip_up: mda.AtomGroup,
                               lip_lo: mda.AtomGroup) -> Tuple[mda.AtomGroup, mda.AtomGroup]:
    """Classify TIP3 residues by mid-plane (original approach: build groups via Merge)."""
    all_w = u.select_atoms(f"resname {WATER}")
    if all_w.n_atoms == 0:
        # Return empty groups via Merge to stay consistent with original style
        return mda.Merge().atoms, mda.Merge().atoms

    z_up = _mean_z_of_headgroup_P(lip_up)
    z_lo = _mean_z_of_headgroup_P(lip_lo)
    mid = 0.5 * (z_up + z_lo)

    up_reslist, lo_reslist = [], []
    for res in all_w.residues:
        # Prefer an oxygen atom to classify; fall back to residue COM if missing
        O = next((a for a in res.atoms if a.name.upper().startswith("O")), None)
        z = (O.position[2] if O is not None else res.atoms.center_of_mass()[2])
        (up_reslist if z >= mid else lo_reslist).append(res.atoms)

    waters_up = mda.Merge(*up_reslist).atoms if up_reslist else mda.Merge().atoms
    waters_lo = mda.Merge(*lo_reslist).atoms if lo_reslist else mda.Merge().atoms
    return waters_up, waters_lo

# --------- Main (same method; only CLI added) ---------
def separate_symmetric_bilayer(input_gro: str,
                               out_upper_gro: str = "out/UPPER.gro",
                               out_lower_gro: str = "out/LOWER.gro") -> None:
    """Split a pure symmetric bilayer into upper/lower leaflet files (original approach)."""
    Path(out_upper_gro).parent.mkdir(parents=True, exist_ok=True)
    Path(out_lower_gro).parent.mkdir(parents=True, exist_ok=True)

    u = mda.Universe(input_gro)
    lipid = _detect_single_lipid_resname(u, water_name=WATER)

    lip_up, lip_lo = _split_lipids_by_resid(u, lipid)
    wat_up, wat_lo = _assign_waters_by_midplane(u, lip_up, lip_lo)

    # ORIGINAL STYLE: build per-leaflet Universes with Merge (no cross-Universe '+')
    up_univ = mda.Merge(lip_up, wat_up)
    lo_univ = mda.Merge(lip_lo, wat_lo)
    up_univ.dimensions = u.dimensions.copy()
    lo_univ.dimensions = u.dimensions.copy()

    with mda.Writer(out_upper_gro, n_atoms=up_univ.atoms.n_atoms) as W:
        W.write(up_univ.atoms)
    with mda.Writer(out_lower_gro, n_atoms=lo_univ.atoms.n_atoms) as W:
        W.write(lo_univ.atoms)

    # Simple report (as before)
    print(f"[separate_by_resid] Lipid={lipid}, Water={WATER}")
    print(f"[separate_by_resid] Upper -> {out_upper_gro}")
    print(f"[separate_by_resid] Lower -> {out_lower_gro}")

def main():
    ap = argparse.ArgumentParser(description="Separate a pure symmetric bilayer into upper/lower leaflets (original approach).")
    ap.add_argument("-i", "--input", required=True, help="Input .gro (pure symmetric bilayer)")
    ap.add_argument("--out-upper", default="out/UPPER.gro", help="Output .gro for upper leaflet (default: out/UPPER.gro)")
    ap.add_argument("--out-lower", default="out/LOWER.gro", help="Output .gro for lower leaflet (default: out/LOWER.gro)")
    args = ap.parse_args()
    separate_symmetric_bilayer(args.input, args.out_upper, args.out_lower)

if __name__ == "__main__":
    main()
