"""
Runs COSMO for every parameter combination in an LHS combinations CSV file.

Usage:
    python3 scripts/run_cosmo_lhs.py \
        --lhs-file scripts/lhs_combinations.csv \
        --bam /path/to/sample.bam \
        --gtf /path/to/annotation.gtf \
        --genome-name "gi|126697566|ref|NC_009089.1|" \
        --genome-size 4290252
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
COSMO_DIR  = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "COSMO"
OUTPUT_SRC = COSMO_DIR / "output"
OUTPUT_DST = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "cosmo_output"


def run_cosmo(combination_number, cds, igr, fd_cds, fd_igr,
              bam, gtf, genome_name, genome_size):
    output_name = f"sample_{combination_number}.csv"
    cmd = [
        sys.executable, "-m", "operon.user_input",
        "-D", str(cds),
        "-d", str(igr),
        "-F", str(fd_cds),
        "-f", str(fd_igr),
        "-o", output_name,
        genome_name,
        str(genome_size),
        str(bam),
        str(gtf),
    ]
    result = subprocess.run(cmd, cwd=COSMO_DIR, capture_output=True, text=True)
    return result, output_name


def main():
    parser = argparse.ArgumentParser(
        description="Run COSMO for every LHS parameter combination."
    )
    parser.add_argument("--lhs-file", required=True,
                        help="Path to the LHS combinations CSV file")
    parser.add_argument("--bam", required=True,
                        help="Path to the BAM input file")
    parser.add_argument("--gtf", required=True,
                        help="Path to the GTF annotation file")
    parser.add_argument("--genome-name", required=True,
                        help="Genome name/ID as it appears in the GTF file")
    parser.add_argument("--genome-size", type=int, required=True,
                        help="Size of the genome in base pairs")
    args = parser.parse_args()

    lhs_file = Path(args.lhs_file)
    bam      = Path(args.bam)
    gtf      = Path(args.gtf)

    if not lhs_file.exists():
        print(f"Error: LHS file not found: {lhs_file}", file=sys.stderr)
        sys.exit(1)
    if not bam.exists():
        print(f"Error: BAM file not found: {bam}", file=sys.stderr)
        sys.exit(1)
    if not gtf.exists():
        print(f"Error: GTF file not found: {gtf}", file=sys.stderr)
        sys.exit(1)
    if not COSMO_DIR.exists():
        print(f"Error: COSMO directory not found: {COSMO_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DST.mkdir(parents=True, exist_ok=True)

    with open(lhs_file, newline="") as f:
        reader = csv.DictReader(f)
        combinations = list(reader)

    total = len(combinations)

    completed = {
        int(p.stem.split("_")[1])
        for p in OUTPUT_DST.glob("sample_*.csv")
    }
    if completed:
        print(f"Found {len(completed)} already completed sample(s), skipping them.")

    remaining = [row for row in combinations if int(row["combination_number"]) not in completed]

    if not remaining:
        print("All combinations already completed.")
        return

    print(f"Running COSMO for {len(remaining)}/{total} remaining combinations...\n")

    failed = []

    for row in remaining:
        n       = int(row["combination_number"])
        cds     = row["CDS_min"]
        igr     = row["IGR_min"]
        fd_cds  = row["FD_CDS-CDS_min"]
        fd_igr  = row["FD_IGR-CDS_min"]

        print(f"[{n}/{total}] CDS={cds}  IGR={igr}  FD_CDS={fd_cds}  FD_IGR={fd_igr}", end="  ", flush=True)

        result, output_name = run_cosmo(
            n, cds, igr, fd_cds, fd_igr,
            bam, gtf, args.genome_name, args.genome_size
        )

        src = OUTPUT_SRC / output_name
        dst = OUTPUT_DST / output_name

        if result.returncode != 0:
            print(f"FAILED")
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
            failed.append(n)
            continue

        if not src.exists():
            print(f"FAILED (output file not found)")
            failed.append(n)
            continue

        shutil.move(str(src), str(dst))
        print(f"→ {dst.name}")

    print(f"\nDone. {len(remaining) - len(failed)}/{len(remaining)} combinations completed successfully.")
    if failed:
        print(f"Failed combinations: {failed}")


if __name__ == "__main__":
    main()
