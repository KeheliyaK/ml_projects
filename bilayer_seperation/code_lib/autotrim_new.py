#!/usr/bin/env python3
"""
autotrim.py
-----------
Trim one leaflet to match the XY box of the other while preserving the source APL.

Method (as per your original logic, plus requested lipid post-filter):
1) Choose the leaflet with the smaller XY area as the reference; trim the other.
2) Preserve source APL by enforcing N_target ≈ (area_ref / APL_source), clamped to available residues.
3) Select lipids by residue COG inside the reference rectangle on the source torus; if not enough,
   fill from nearest outside by toroidal distance.
4) Water keep is TWO-STAGE:
   (a) keep waters whose residue COG lies inside the reference rectangle (on source torus),
   (b) then post-filter to keep only waters whose O-atom is inside the rectangle.
5) NEW (requested): Post-filter selected lipids — if a residue COG lies outside the reference rectangle
   (NO wrapping), drop it.
6) Write: out/trimmed_cog_target_<resname>.gro

Notes:
- Only TIP3 is treated as water.
- Units: coordinates in Å from .gro, converted to nm for area logic.
"""

from pathlib import Path
import argparse
import numpy as np
import MDAnalysis as mda

WATER_RESNAMES = {"TIP3"}  # strict per your spec

# ---------- Utilities ----------
def residue_xy_cog_nm(residue):
    coords_nm = residue.atoms.positions / 10.0  # Å -> nm
    cog = coords_nm.mean(axis=0)
    return float(cog[0]), float(cog[1])

def wrap_points_mod(xy_nm, Lx, Ly):
    x = np.mod(xy_nm[:, 0], Lx)
    y = np.mod(xy_nm[:, 1], Ly)
    return np.column_stack([x, y])

def get_lipid_resname(universe, exclude=WATER_RESNAMES):
    resnames = {res.resname for res in universe.residues if res.resname not in exclude}
    if len(resnames) != 1:
        raise ValueError(f"Expected exactly one lipid type, found: {sorted(resnames)}")
    return next(iter(resnames))

def get_lipid_apl_and_box(universe):
    resname = get_lipid_resname(universe, exclude=WATER_RESNAMES)
    lipids = universe.select_atoms(f"resname {resname}")
    n_lipids = len(lipids.residues)
    boxA = universe.dimensions  # Å
    Lx_nm, Ly_nm = (boxA[0] / 10.0), (boxA[1] / 10.0)
    area_xy = Lx_nm * Ly_nm
    apl = area_xy / float(n_lipids)
    return resname, area_xy, apl, Lx_nm, Ly_nm

def compute_target_count(area_ref_nm2, apl_source_nm2, n_max):
    n_tgt = int(round(area_ref_nm2 / apl_source_nm2))
    return max(1, min(n_tgt, n_max))

# ---------- Torus distances ----------
def torus_distance_to_interval_1d(x, L, a, b):
    x = np.mod(x, L)
    if a <= x <= b:
        return 0.0
    if x < a:
        d1 = a - x
        d2 = x + (L - b)
        return d1 if d1 < d2 else d2
    # x > b
    d1 = x - b
    d2 = (L - x) + a
    return d1 if d1 < d2 else d2

def torus_distance_to_rect(pt, Lx, Ly, Rx, Ry):
    x, y = pt
    dx = torus_distance_to_interval_1d(x, Lx, 0.0, Rx)
    dy = torus_distance_to_interval_1d(y, Ly, 0.0, Ry)
    return float(np.hypot(dx, dy))

# ---------- Lipid selection ----------
def select_lipids_by_cog_with_target(lipid_ag, Lx_ref, Ly_ref, N_target):
    """
    Select residues whose COG is inside [0,Lx_ref)×[0,Ly_ref) on the source torus.
    If fewer than N_target inside, fill by nearest-outside (toroidal metric).
    Returns (selected_atoms, chosen_res_idx).
    """
    u = lipid_ag.universe
    Lx_src, Ly_src = (u.dimensions[0] / 10.0), (u.dimensions[1] / 10.0)

    residues = list(lipid_ag.residues)
    xy = np.array([residue_xy_cog_nm(res) for res in residues])
    xy_wrapped = wrap_points_mod(xy, Lx_src, Ly_src)

    inside_mask = (xy_wrapped[:, 0] >= 0.0) & (xy_wrapped[:, 0] < Lx_ref) & \
                  (xy_wrapped[:, 1] >= 0.0) & (xy_wrapped[:, 1] < Ly_ref)
    inside_idx = np.where(inside_mask)[0]

    if inside_idx.size >= N_target:
        chosen_res_idx = inside_idx[:N_target]
    else:
        outside_idx = np.where(~inside_mask)[0]
        dists = np.array([
            torus_distance_to_rect((xy_wrapped[i, 0], xy_wrapped[i, 1]),
                                   Lx_src, Ly_src, Lx_ref, Ly_ref)
            for i in outside_idx
        ])
        order = np.argsort(dists)
        to_take = outside_idx[order][:max(0, N_target - inside_idx.size)]
        chosen_res_idx = np.concatenate([inside_idx, to_take]) if inside_idx.size else to_take

    selected_residues = [residues[i] for i in chosen_res_idx]
    selected_atoms = mda.Merge(*[res.atoms for res in selected_residues]).atoms
    return selected_atoms, chosen_res_idx

# ---------- Water keep: TWO-STAGE ----------
def select_waters_by_cog_in_rect(water_ag, Lx_ref, Ly_ref):
    """
    Stage (a): keep waters whose RESIDUE COG is inside [0,Lx_ref)×[0,Ly_ref)
    on the SOURCE torus. Returns an AtomGroup (same Universe).
    """
    if water_ag.n_atoms == 0:
        return water_ag.universe.atoms[[]]

    u = water_ag.universe
    Lx_src, Ly_src = (u.dimensions[0] / 10.0), (u.dimensions[1] / 10.0)
    keep_atom_idx = []

    for res in water_ag.residues:
        x_nm, y_nm = residue_xy_cog_nm(res)     # nm
        x_nm = x_nm % Lx_src
        y_nm = y_nm % Ly_src
        if (0.0 <= x_nm < Lx_ref) and (0.0 <= y_nm < Ly_ref):
            keep_atom_idx.extend(res.atoms.indices.tolist())

    return u.atoms[keep_atom_idx]

def filter_waters_inside_rect_by_oxygen(water_ag, Lx_ref_nm, Ly_ref_nm):
    """
    Stage (b): post-filter waters so that ONLY residues whose oxygen atom lies
    inside [0,Lx_ref_nm)×[0,Ly_ref_nm) (nm) are kept. No torus wrapping here.
    """
    if water_ag.n_atoms == 0:
        return water_ag.universe.atoms[[]]

    u = water_ag.universe
    keep_atom_idx = []

    for res in water_ag.residues:
        oxy = res.atoms[0]  # original convention
        x_nm = oxy.position[0] / 10.0
        y_nm = oxy.position[1] / 10.0
        if (0.0 <= x_nm < Lx_ref_nm) and (0.0 <= y_nm < Ly_ref_nm):
            keep_atom_idx.extend(res.atoms.indices.tolist())

    return u.atoms[keep_atom_idx]

# ---------- NEW: Lipid post-filter (no wrapping) ----------
def filter_lipids_inside_rect_by_cog(lipid_atoms, Lx_ref_nm, Ly_ref_nm):
    """
    Drop any LIPID residue whose COG lies outside [0,Lx_ref_nm)×[0,Ly_ref_nm) (nm).
    No torus wrapping (to match your "outside the ref rectangle" criterion).
    """
    if lipid_atoms.n_atoms == 0:
        return lipid_atoms  # already empty, same Universe

    keep_res = []
    for res in lipid_atoms.residues:
        x_nm, y_nm = residue_xy_cog_nm(res)  # from THIS merged Universe's coordinates
        if (0.0 <= x_nm < Lx_ref_nm) and (0.0 <= y_nm < Ly_ref_nm):
            keep_res.append(res.atoms)

    return mda.Merge(*keep_res).atoms if keep_res else mda.Merge().atoms

# ---------- Main trimming procedure ----------
def auto_trim_leaflet_cog_target(gro_file_A: str, gro_file_B: str, output_dir: str = "out") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[load] Leaflet A: {gro_file_A}")
    uA = mda.Universe(gro_file_A)
    print(f"[load] Leaflet B: {gro_file_B}")
    uB = mda.Universe(gro_file_B)

    resA, areaA, aplA, LxA, LyA = get_lipid_apl_and_box(uA)
    resB, areaB, aplB, LxB, LyB = get_lipid_apl_and_box(uB)

    print(f"\n[info] A: resname={resA}, area={areaA:.5f} nm^2, APL={aplA:.5f} nm^2, box={LxA:.3f}×{LyA:.3f} nm^2")
    print(f"[info] B: resname={resB}, area={areaB:.5f} nm^2, APL={aplB:.5f} nm^2, box={LxB:.3f}×{LyB:.3f} nm^2")

    if abs(areaA - areaB) < 1e-3:
        print("[skip] Areas nearly equal; no trimming needed.")
        return

    # Choose reference (smaller XY area) and source-to-trim (preserve source APL)
    if areaA <= areaB:
        ref_u, ref_res, Lx_ref, Ly_ref = uA, resA, LxA, LyA
        trim_u, trim_res, apl_src = uB, resB, aplB
    else:
        ref_u, ref_res, Lx_ref, Ly_ref = uB, resB, LxB, LyB
        trim_u, trim_res, apl_src = uA, resA, aplA

    # Lipids: select N_target by COG-in-rect, fill nearest-outside on torus
    lipids = trim_u.select_atoms(f"resname {trim_res}")
    n_src = len(lipids.residues)
    N_target = compute_target_count(Lx_ref * Ly_ref, apl_src, n_src)
    sel_atoms, chosen_idx = select_lipids_by_cog_with_target(lipids, Lx_ref, Ly_ref, N_target)

    # --- NEW: Post-filter lipids by non-wrapped COG in ref rectangle ---
    n_before_pf = len(sel_atoms.residues)
    sel_atoms = filter_lipids_inside_rect_by_cog(sel_atoms, Lx_ref, Ly_ref)
    n_after_pf = len(sel_atoms.residues)
    dropped = n_before_pf - n_after_pf
    if dropped > 0:
        print(f"[lipid-postfilter] Dropped {dropped} out-of-rectangle residues; N_final (lipids) = {n_after_pf}")
        print("  [warn] APL preservation may drift because N_final < N_target.")

    # Waters: two-stage keep (COG-in-rect on source torus, then oxygen-in-rect)
    all_waters = trim_u.select_atoms("resname TIP3")
    stage_a = select_waters_by_cog_in_rect(all_waters, Lx_ref, Ly_ref)
    trimmed_water = filter_waters_inside_rect_by_oxygen(stage_a, Lx_ref, Ly_ref)

    # Merge selected lipids and kept waters into a new Universe; copy box from reference
    merged = mda.Merge(sel_atoms, trimmed_water)
    merged.dimensions = ref_u.dimensions.copy()
    n_final = len(sel_atoms.residues)

    area_nm2 = (merged.dimensions[0] / 10.0) * (merged.dimensions[1] / 10.0)
    apl_t = area_nm2 / float(n_final) if n_final > 0 else float('nan')

    out_name = f"trimmed_cog_target_{trim_res.lower()}.gro"
    output_file = str(Path(output_dir) / out_name)

    has_vel = hasattr(trim_u.trajectory.ts, "velocities") and (trim_u.trajectory.ts.velocities is not None)

    with mda.Writer(output_file, n_atoms=merged.atoms.n_atoms) as W:
        W.write(merged.atoms)

    print("\n[report]")
    print(f"  Trimmed leaflet: resname={trim_res}")
    print(f"  Selected lipids: {n_final} (requested target {N_target})")
    print(f"  Waters kept     : {len(trimmed_water.residues)}")
    print(f"  Box (nm)        : {(merged.dimensions[0]/10.0):.3f}×{(merged.dimensions[1]/10.0):.3f}")
    print(f"  APL after (nm^2): {apl_t:.5f}  (source APL {apl_src:.5f})")
    if dropped > 0:
        print("  [note] Lipid post-filter removed residues; APL may deviate from source by design.")
    print(f"  Velocities kept : {has_vel}")
    print(f"  Wrote           : {output_file}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Auto-trim one leaflet to the other's XY box, preserving source APL.")
    ap.add_argument("leaflet_A", help="Path to leaflet A .gro")
    ap.add_argument("leaflet_B", help="Path to leaflet B .gro")
    ap.add_argument("--outdir", default="out", help="Output directory")
    args = ap.parse_args()
    auto_trim_leaflet_cog_target(args.leaflet_A, args.leaflet_B, args.outdir)

if __name__ == "__main__":
    main()
