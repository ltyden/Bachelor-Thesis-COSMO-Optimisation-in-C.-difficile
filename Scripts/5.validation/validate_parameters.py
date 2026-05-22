"""
Parameter Validation Script
============================
Runs COSMO with a single user-defined parameter combination across all BAM
files, unions the predicted operons, evaluates against the EVO reference, and
prints the full metric set to the terminal.

TP  — predicted operon whose gene set exactly matches an EVO gene set
FN  — EVO not exactly matched by any predicted operon

Usage:
    python3 scripts/5.validation/validate_parameters.py \
        --cds 1 --igr 2 --fd-cds 6 --fd-igr 2 \
        --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \
        --gtf /path/to/NC_009089.1_fixed.gtf \
        --genome-name "gi|126697566|ref|NC_009089.1|" \
        --genome-size 4290252

Arguments:
    --cds           Minimum CDS coverage depth (integer)
    --igr           Minimum IGR coverage depth (integer)
    --fd-cds        Max fold difference between adjacent CDSs (float)
    --fd-igr        Max fold difference between IGR and flanking CDSs (float)
    --bam           One or more BAM files (predictions are unioned across all)
    --gtf           GTF annotation file
    --genome-name   Genome name/ID as used in the GTF
    --genome-size   Genome size in base pairs
    --evo-reference Path to EVO reference CSV (default: evo_reference2.csv)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
COSMO_DIR  = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "COSMO"
OUTPUT_SRC = COSMO_DIR / "output"
DEFAULT_EVO = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "evo_reference2.csv"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "2.data_generation"))
from evaluate_cosmo import (    # noqa: E402
    classify_predictions,
    compute_metrics,
    load_cosmo_predictions,
    load_evos,
)

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Validate a single COSMO parameter combination against the EVO reference.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--cds",    required=True, type=int,   help="Min CDS coverage depth (integer)")
parser.add_argument("--igr",    required=True, type=int,   help="Min IGR coverage depth (integer)")
parser.add_argument("--fd-cds", required=True, type=float, help="Max fold difference adjacent CDSs")
parser.add_argument("--fd-igr", required=True, type=float, help="Max fold difference IGR vs CDSs")
parser.add_argument("--bam",    required=True, nargs="+",  help="BAM file path(s)")
parser.add_argument("--gtf",    required=True,             help="GTF annotation file")
parser.add_argument("--genome-name", required=True,        help="Genome name/ID")
parser.add_argument("--genome-size", required=True, type=int, help="Genome size in base pairs")
parser.add_argument("--evo-reference", default=str(DEFAULT_EVO),
                    help=f"EVO reference CSV (default: {DEFAULT_EVO})")
args = parser.parse_args()

bam_files = [Path(b) for b in args.bam]
gtf       = Path(args.gtf)
evo_path  = Path(args.evo_reference)

# ── Validate inputs ────────────────────────────────────────────────────────────
for b in bam_files:
    if not b.exists():
        sys.exit(f"ERROR: BAM file not found: {b}")
if not gtf.exists():
    sys.exit(f"ERROR: GTF file not found: {gtf}")
if not evo_path.exists():
    sys.exit(f"ERROR: EVO reference not found: {evo_path}")
if not COSMO_DIR.exists():
    sys.exit(f"ERROR: COSMO directory not found: {COSMO_DIR}")

# ── Print run header ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COSMO PARAMETER VALIDATION")
print("=" * 60)
print(f"  CDS_min         : {args.cds}")
print(f"  IGR_min         : {args.igr}")
print(f"  FD_CDS-CDS_min  : {args.fd_cds}")
print(f"  FD_IGR-CDS_min  : {args.fd_igr}")
print(f"  BAM files       : {len(bam_files)}")
print("=" * 60)

# ── Run COSMO for each BAM file, collect outputs in a temp directory ──────────
evos   = load_evos(evo_path)
merged: set[frozenset] = set()
temp_dir = tempfile.mkdtemp(prefix="cosmo_validate_")

try:
    for idx, bam in enumerate(bam_files, start=1):
        output_name = f"validate_bam{idx}.csv"
        cmd = [
            sys.executable, "-m", "operon.user_input",
            "-D", str(args.cds),
            "-d", str(args.igr),
            "-F", str(args.fd_cds),
            "-f", str(args.fd_igr),
            "-o", output_name,
            args.genome_name,
            str(args.genome_size),
            str(bam),
            str(gtf),
        ]

        print(f"\nRunning COSMO — BAM {idx}/{len(bam_files)}: {bam.name} ...", flush=True)
        result = subprocess.run(cmd, cwd=COSMO_DIR, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  FAILED\n  stderr: {result.stderr.strip()}", file=sys.stderr)
            continue

        src = OUTPUT_SRC / output_name
        dst = Path(temp_dir) / output_name

        if not src.exists():
            print(f"  FAILED: output file not found at {src}", file=sys.stderr)
            continue

        shutil.move(str(src), str(dst))

        preds = load_cosmo_predictions(dst)
        for pred in preds:
            merged.add(pred)
        print(f"  Done — {len(preds)} operons predicted")

    # ── Evaluate union of predictions against all EVOs ─────────────────────────
    if not merged:
        sys.exit("ERROR: No predictions collected — all COSMO runs failed.")

    counts  = classify_predictions(evos, merged)
    metrics = compute_metrics(counts)
    tp_pct  = metrics["TP"] / metrics["total_EVOs"] * 100 if metrics["total_EVOs"] > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  EVOs evaluated  : {metrics['total_EVOs']}")
    print(f"  Predicted operons (union across {len(bam_files)} BAM files): {metrics['total_predicted']}")
    print()
    print(f"  {'Metric':<20}  {'Value':>10}")
    print(f"  {'─'*20}  {'─'*10}")
    print(f"  {'TP%':<20}  {tp_pct:>9.1f}%  ({metrics['TP']} / {metrics['total_EVOs']} EVOs)")
    print(f"  {'True Positives':<20}  {metrics['TP']:>10}")
    print(f"  {'False Negatives':<20}  {metrics['FN']:>10}")
    print(f"  {'Recall':<20}  {metrics['Recall']*100:>9.1f}%")
    print(f"  {'F1 Score':<20}  {metrics['F1']*100:>9.1f}%")
    print("=" * 60)

finally:
    shutil.rmtree(temp_dir)
