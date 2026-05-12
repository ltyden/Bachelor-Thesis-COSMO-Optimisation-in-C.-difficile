"""
Generates a targeted parameter grid CSV centred on the default COSMO parameters.

CDS_min and IGR_min are fixed at 1 and 2 (agreed upon by both the GP optimum
and the Mtb defaults). FD_CDS-CDS_min and FD_IGR-CDS_min are swept across a
range bracketing the defaults (F=5, f=10).

The output CSV can be passed directly to run_cosmo_lhs.py.

Usage:
    python3 scripts/5.validation/generate_targeted_grid.py [--output PATH]
"""

import argparse
import csv
import itertools
from pathlib import Path

# ── Grid definition ────────────────────────────────────────────────────────────
CDS_MIN_FIXED = 1
IGR_MIN_FIXED = 2

FD_CDS_VALUES = [2, 3, 4, 5, 6, 7, 8, 10]
FD_IGR_VALUES = [2, 4, 6, 8, 10, 12, 14]

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate a targeted FD parameter grid CSV for COSMO validation."
)
parser.add_argument(
    "--output",
    default="parameter_optimisation/raw_datasets/targeted_grid.csv",
    metavar="PATH",
    help="Output CSV path (default: parameter_optimisation/raw_datasets/targeted_grid.csv)",
)
args = parser.parse_args()

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

# ── Generate combinations ──────────────────────────────────────────────────────
combinations = list(itertools.product(FD_CDS_VALUES, FD_IGR_VALUES))

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["combination_number", "CDS_min", "IGR_min",
                     "FD_CDS-CDS_min", "FD_IGR-CDS_min"])
    for n, (fd_cds, fd_igr) in enumerate(combinations, start=1):
        writer.writerow([n, CDS_MIN_FIXED, IGR_MIN_FIXED, fd_cds, fd_igr])

print(f"Generated {len(combinations)} combinations → {out_path}")
print(f"\nFixed parameters:")
print(f"  CDS_min : {CDS_MIN_FIXED}")
print(f"  IGR_min : {IGR_MIN_FIXED}")
print(f"\nFD_CDS-CDS_min values : {FD_CDS_VALUES}")
print(f"FD_IGR-CDS_min values : {FD_IGR_VALUES}")
print(f"\nNext step — run COSMO for all combinations:")
print(f"""
  python3 scripts/2.data_generation/run_cosmo_lhs.py \\
      --lhs-file {out_path} \\
      --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \\
      --gtf /path/to/annotation.gtf \\
      --genome-name "gi|126697566|ref|NC_009089.1|" \\
      --genome-size 4290252
""")
