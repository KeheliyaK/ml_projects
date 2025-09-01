#!/usr/bin/env python3
"""
build_asymmetric.py
-------------------
End-to-end builder that:
1) Separates each pure symmetric bilayer into leaflets.
2) Auto-trims the pair to match XY and preserve source APL.
3) Merges and sorts into the final asymmetric bilayer.
4) Prints counts and APL per leaflet.

Assumptions (per user):
- TIP3 is the only water.
- DPPC = upper leaflet, DOPC = lower leaflet.
- Final sort order: DPPC → DOPC → TIP3.
"""

import argparse
from pathlib import Path
import sys

# Import our local modules
import code_lib.separate_leaflets as sep
import code_lib.autotrim_new as at
import code_lib.merge_and_sort as msa
RUN_WITH_DEFAULTS_WHEN_NO_ARGS = True

dppc_gro="heavy_data_no_git/input/dppc.gro"
dopc_gro="heavy_data_no_git/input/dopc_20_bar.gro"
outdir = "heavy_data_no_git/output"

def build_asymmetric(
    dppc_gro: str,
    dopc_gro: str,
    outdir: str = "heavy_data_no_git/output",
    thickness_nm: float = 3.7
):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Separate DPPC (pure symmetric) into leaflets...")
    dppc_upper = str(Path(outdir) / "DPPC_upper.gro")
    dppc_lower = str(Path(outdir) / "DPPC_lower.gro")
    sep.separate_symmetric_bilayer(dppc_gro, dppc_upper, dppc_lower)

    print("\n[2/4] Separate DOPC (pure symmetric) into leaflets...")
    dopc_upper = str(Path(outdir) / "DOPC_upper.gro")
    dopc_lower = str(Path(outdir) / "DOPC_lower.gro")
    sep.separate_symmetric_bilayer(dopc_gro, dopc_upper, dopc_lower)

    # Mapping per user: DPPC = upper, DOPC = lower
    print("\n[3/4] Auto-trim the pair to match XY and preserve source APL...")
    # Feed the DPPC-upper and DOPC-lower into autotrim in any order;
    # autotrim picks the smaller XY as reference and trims the other.
    at.auto_trim_leaflet_cog_target(dppc_upper, dopc_lower, output_dir=outdir)

    # Figure which got trimmed by filename convention from autotrim: "trimmed_cog_target_<res>.gro"
    trimmed_dppc = Path(outdir) / "trimmed_cog_target_dppc.gro"
    trimmed_dopc = Path(outdir) / "trimmed_cog_target_dopc.gro"

    # Choose upper/lower for merge: enforce DPPC=upper, DOPC=lower
    upper_path = str(trimmed_dppc if trimmed_dppc.exists() else dppc_upper)
    lower_path = str(trimmed_dopc if trimmed_dopc.exists() else dopc_lower)

    print("\n[4/4] Merge, sort, and report counts + APL...")
    merged_path = str(Path(outdir) / "merged_dppc_dopc.gro")
    sorted_path = str(Path(outdir) / "dppc_dopc_sorted.gro")
    msa.merge_leaflets_with_velocities(upper_path, lower_path, merged_path, desired_thickness_nm=thickness_nm)
    msa.reorder_resnames(merged_path, sorted_path, order=msa.ORDER)
    msa.report_counts_and_apl(sorted_path)

    print("\n[done] Final sorted bilayer:", sorted_path)

def main():
    ap = argparse.ArgumentParser(description="Builder to create a DPPC↑/DOPC↓ asymmetric bilayer.")
    ap.add_argument("--dppc", default=dppc_gro, help="Path to pure symmetric DPPC .gro")
    ap.add_argument("--dopc", default=dopc_gro, help="Path to pure symmetric DOPC .gro")
    ap.add_argument("--outdir", default=outdir, help="Output directory (default: out)")
    ap.add_argument("--thickness", type=float, default=3.7, help="Target P-P thickness in nm (default: 3.7)")

    if RUN_WITH_DEFAULTS_WHEN_NO_ARGS and len(sys.argv) == 1:
        print("[run] No CLI args detected → using in-file defaults.")
        build_asymmetric(dppc_gro, dopc_gro, outdir, 3.7)
        return
    args = ap.parse_args()
    build_asymmetric(args.dppc, args.dopc, args.outdir, args.thickness)


if __name__ == "__main__":
    main()
