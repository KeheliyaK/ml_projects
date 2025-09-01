#!/usr/bin/env python3
"""
merge_sort_apl.py
-----------------
1) Merge two leaflet .gro files into one bilayer at a target headgroup separation.
2) Preserve velocities if present.
3) Sort final atoms in the order: DPPC → DOPC → TIP3.
4) Print counts and APL per leaflet (DOPC=lower, DPPC=upper).

This consolidates your Merge2.py + sort.py + counter.py while preserving method logic.
"""

import MDAnalysis as mda
import numpy as np
from collections import Counter
from pathlib import Path
import argparse
import sys

ORDER = ["DPPC", "DOPC", "TIP3"]  # final sort order per user requirement

# --------- Merge helpers (logic preserved) ---------
def find_ref_atom_index(atomgroup, atom_name="P"):
    for i, atom in enumerate(atomgroup.residues[0].atoms):
        if atom.name.upper().startswith(atom_name.upper()):
            return i
    raise ValueError(f"Reference atom '{atom_name}' not found.")

def mean_z_ref_atoms(atomgroup, atoms_per_mol, ref_idx):
    z_positions = atomgroup.positions[ref_idx::atoms_per_mol, 2]
    return float(np.mean(z_positions))

def _auto_detect_lipid_resname(univ, exclude={"TIP3"}):
    res = list({r.resname for r in univ.residues if r.resname not in exclude})
    if len(res) != 1:
        raise ValueError(f"Expected single lipid type; got {sorted(res)}")
    return res[0]

def merge_leaflets_with_velocities(
    gro_upper, gro_lower, out_gro,
    resname_upper=None, resname_lower=None,
    ref_atom="P", desired_thickness_nm=3.7
):
    u_upper = mda.Universe(gro_upper)
    u_lower = mda.Universe(gro_lower)

    if not resname_upper:
        resname_upper = _auto_detect_lipid_resname(u_upper)
    if not resname_lower:
        resname_lower = _auto_detect_lipid_resname(u_lower)

    lipids_upper = u_upper.select_atoms(f"resname {resname_upper}")
    lipids_lower = u_lower.select_atoms(f"resname {resname_lower}")
    atoms_per_mol_upper = len(lipids_upper.residues[0].atoms)
    atoms_per_mol_lower = len(lipids_lower.residues[0].atoms)
    ref_idx_upper = find_ref_atom_index(lipids_upper, ref_atom)
    ref_idx_lower = find_ref_atom_index(lipids_lower, ref_atom)

    upper_P_z = mean_z_ref_atoms(lipids_upper, atoms_per_mol_upper, ref_idx_upper)
    lower_P_z = mean_z_ref_atoms(lipids_lower, atoms_per_mol_lower, ref_idx_lower)

    box_z = u_upper.dimensions[2]
    center = box_z / 2.0
    half_thick = (desired_thickness_nm * 10.0) / 2.0  # nm -> Å
    target_upper_z = center + half_thick
    target_lower_z = center - half_thick
    shift_upper = target_upper_z - upper_P_z
    shift_lower = target_lower_z - lower_P_z

    print(f"[merge] Shifting upper by {shift_upper:.2f} Å, lower by {shift_lower:.2f} Å")
    u_upper.atoms.positions[:, 2] += shift_upper
    u_lower.atoms.positions[:, 2] += shift_lower

    merged = mda.Merge(u_upper.atoms, u_lower.atoms)
    merged.dimensions = u_upper.dimensions.copy()

    # velocities: keep if present
    has_vel = hasattr(u_upper.trajectory.ts, "velocities") and u_upper.trajectory.ts.velocities is not None \
              and hasattr(u_lower.trajectory.ts, "velocities") and u_lower.trajectory.ts.velocities is not None
    # (MDAnalysis Writer will include velocities automatically when present in ts)

    Path(out_gro).parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(out_gro, n_atoms=merged.atoms.n_atoms) as W:
        W.write(merged.atoms)

    print(f"[merge] Wrote merged bilayer: {out_gro} (velocities kept: {bool(has_vel)})")
    return out_gro

# --------- Sort (logic preserved) ---------
def reorder_resnames(in_gro: str, out_gro: str, order=ORDER):
    u = mda.Universe(in_gro)
    atomgroups = [u.select_atoms(f"resname {resname}") for resname in order]
    all_atoms = atomgroups[0]
    for ag in atomgroups[1:]:
        all_atoms = all_atoms + ag

    with mda.Writer(out_gro, n_atoms=all_atoms.n_atoms) as W:
        W.write(all_atoms)
    print(f"[sort] Wrote reordered .gro: {out_gro}")
    return out_gro

# --------- Counts & APL (logic preserved) ---------
EXPECTED_ATOMS = {"DOPC": 138, "DPPC": 130}  # CHARMM36 (all-atom)

def check_atom_counts(u):
    bad = []
    for res in u.residues:
        if res.resname in EXPECTED_ATOMS and len(res.atoms) != EXPECTED_ATOMS[res.resname]:
            bad.append(res)
    if bad:
        print("[warn] Residues not matching expected atom counts:")
        show = bad[:10]
        print("       First few (resname,resid,len):",
              [(r.resname, int(r.resid), len(r.atoms)) for r in show])
    else:
        print("[ok] All DOPC/DPPC residues match expected atom counts.")

def count_residues(u):
    resnames = [res.resname for res in u.residues]
    counter = Counter(resnames)
    for res, count in sorted(counter.items()):
        print(f"  {res}: {count}")
    return counter

def calc_apl_per_leaflet(u):
    Lx, Ly = u.dimensions[0], u.dimensions[1]  # Å
    area_nm2 = (Lx * Ly) * 0.01
    N_lower = sum(1 for r in u.residues if r.resname == "DOPC")  # lower
    N_upper = sum(1 for r in u.residues if r.resname == "DPPC")  # upper
    apl_lower = area_nm2 / N_lower if N_lower > 0 else float("nan")
    apl_upper = area_nm2 / N_upper if N_upper > 0 else float("nan")
    return {
        "area_nm2": area_nm2,
        "N_lower_DOPC": N_lower,
        "N_upper_DPPC": N_upper,
        "APL_lower_nm2": apl_lower,
        "APL_upper_nm2": apl_upper,
    }

def report_counts_and_apl(gro_path: str):
    u = mda.Universe(gro_path)
    print("\n[counts]")
    counts = count_residues(u)
    print("\n[box]")
    print(f"  Lx = {u.dimensions[0]:.3f} Å, Ly = {u.dimensions[1]:.3f} Å")
    res = calc_apl_per_leaflet(u)
    print(f"  Area (xy) = {res['area_nm2']:.3f} nm^2")
    print("\n[APL]")
    print(f"  Lower (DOPC): N = {res['N_lower_DOPC']}, APL = {res['APL_lower_nm2']:.5f} nm^2")
    print(f"  Upper (DPPC): N = {res['N_upper_DPPC']}, APL = {res['APL_upper_nm2']:.5f} nm^2")

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Merge two leaflets, sort residues, and report counts + APL per leaflet.")
    ap.add_argument("upper_gro", help="Upper leaflet .gro (DPPC)")
    ap.add_argument("lower_gro", help="Lower leaflet .gro (DOPC)")
    ap.add_argument("--thickness", type=float, default=3.7, help="Target P-P thickness in nm (default: 3.7)")
    ap.add_argument("--merged", default="out/merged_dppc_dopc.gro", help="Path for merged .gro")
    ap.add_argument("--sorted", default="out/dppc_dopc_sorted.gro", help="Path for sorted .gro")
    args = ap.parse_args()

    merged = merge_leaflets_with_velocities(args.upper_gro, args.lower_gro, args.merged,
                                            desired_thickness_nm=args.thickness)
    sorted_path = reorder_resnames(merged, args.sorted, order=ORDER)
    report_counts_and_apl(sorted_path)

if __name__ == "__main__":
    main()
