"""
Parameter Validation Script
============================
Runs COSMO with a single user-defined parameter combination across all BAM
files, unions the predicted operons, evaluates against the EVO reference, and
prints the full metric set to the terminal.

Usage:
    python3 scripts/5.validation/validate_parameters.py \
        --cds 1 --igr 2 --fd-cds 3.6 --fd-igr 2.0 \
        --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \
        --gtf /path/to/annotation.gtf \
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
    --evo-reference Path to EVO reference CSV (default: auto-detected)
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COSMO_DIR = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "COSMO"
OUTPUT_SRC = COSMO_DIR / "output"
DEFAULT_EVO = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "evo_reference.csv"

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

# ── Helper functions (inlined from evaluate_cosmo.py) ─────────────────────────
def load_evos(path: Path) -> list[dict]:
    evos = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            evos.append({
                "evo_id": row["evo_id"],
                "genes":  frozenset(row["evo_genes"].split(";")),
            })
    return evos


def load_predictions(path: Path) -> list[frozenset]:
    predictions, current = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].strip():
                if current:
                    predictions.append(frozenset(current))
                current = []
            else:
                raw = row[7].strip() if len(row) > 7 else ""
                if raw:
                    current.append(raw.removeprefix("gene-"))
    if current:
        predictions.append(frozenset(current))
    return predictions


def evaluate(evos: list[dict], predictions: list[frozenset]) -> dict:
    tp = fp = fn = 0
    for evo in evos:
        eg   = evo["genes"]
        best = max(predictions, key=lambda p: len(p & eg), default=frozenset())
        if best == eg:
            tp += 1
        elif eg <= best:
            fp += 1
        else:
            fn += 1
    total     = len(evos)
    recall    = tp / (tp + fn)    if (tp + fn) > 0    else 0.0
    precision = tp / (tp + fp)    if (tp + fp) > 0    else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"TP": tp, "FP": fp, "FN": fn, "total": total,
            "Precision": precision, "Recall": recall, "F1": f1}

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
evos    = load_evos(evo_path)
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

        preds = load_predictions(dst)
        merged.update(preds)
        print(f"  Done — {len(preds)} operons predicted")

    # ── Evaluate union of predictions against all EVOs ─────────────────────────
    if not merged:
        sys.exit("ERROR: No predictions collected — all COSMO runs failed.")

    metrics = evaluate(evos, list(merged))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  EVOs evaluated  : {metrics['total']}")
    print(f"  Predicted operons (union across {len(bam_files)} BAM files): {len(merged)}")
    print()
    print(f"  {'Metric':<20}  {'Value':>10}")
    print(f"  {'─'*20}  {'─'*10}")
    tp_pct = metrics['TP'] / metrics['total'] * 100
    print(f"  {'TP%':<20}  {tp_pct:>9.1f}%  ({metrics['TP']} / {metrics['total']} EVOs)")
    print(f"  {'True Positives':<20}  {metrics['TP']:>10}")
    print(f"  {'False Positives':<20}  {metrics['FP']:>10}")
    print(f"  {'False Negatives':<20}  {metrics['FN']:>10}")
    print(f"  {'Precision':<20}  {metrics['Precision']*100:>9.1f}%")
    print(f"  {'Recall':<20}  {metrics['Recall']*100:>9.1f}%")
    print(f"  {'F1 Score':<20}  {metrics['F1']*100:>9.1f}%")
    print("=" * 60)

finally:
    shutil.rmtree(temp_dir)
