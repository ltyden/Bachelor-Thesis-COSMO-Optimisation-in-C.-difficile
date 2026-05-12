"""
Runs COSMO for every parameter combination in an LHS combinations CSV,
across one or more BAM files.

For each combination, COSMO is run once per BAM file. Outputs are named
sample_{n}_bam{i}.csv so results from different BAM files don't overwrite
each other. evaluate_cosmo.py then merges them before evaluation.

Usage:
    # All three BAM files from scratch
    python3 scripts/2.data_generation/run_cosmo_lhs.py \
        --lhs-file parameter_optimisation/raw_datasets/lhs_combinations.csv \
        --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \
        --gtf /path/to/annotation.gtf \
        --genome-name "gi|126697566|ref|NC_009089.1|" \
        --genome-size 4290252

    # wt1 already done — run only wt2 and wt3, starting index at 2
    python3 scripts/2.data_generation/run_cosmo_lhs.py \
        --lhs-file parameter_optimisation/raw_datasets/lhs_combinations.csv \
        --bam /path/to/wt2.bam /path/to/wt3.bam \
        --first-bam-index 2 \
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

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
COSMO_DIR  = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "COSMO"
OUTPUT_SRC = COSMO_DIR / "output"
OUTPUT_DST = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "cosmo_output"


def detect_delimiter(path: Path) -> str:
    with open(path, newline="") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","


def run_cosmo(combination_number, bam_idx, cds, igr, fd_cds, fd_igr,
              bam, gtf, genome_name, genome_size):
    output_name = f"sample_{combination_number}_bam{bam_idx}.csv"
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
        description="Run COSMO for every LHS parameter combination across one or more BAM files."
    )
    parser.add_argument("--lhs-file", required=True,
                        help="Path to the LHS combinations CSV file")
    parser.add_argument("--bam", required=True, nargs="+",
                        help="Path(s) to BAM file(s). Pass multiple to run across all of them.")
    parser.add_argument("--first-bam-index", type=int, default=1, metavar="N",
                        help="Index to assign to the first BAM file passed (default: 1). "
                             "Set to 2 if wt1 is already done and you are adding wt2 and wt3.")
    parser.add_argument("--gtf", required=True,
                        help="Path to the GTF annotation file")
    parser.add_argument("--genome-name", required=True,
                        help="Genome name/ID as it appears in the GTF file")
    parser.add_argument("--genome-size", type=int, required=True,
                        help="Size of the genome in base pairs")
    args = parser.parse_args()

    lhs_file  = Path(args.lhs_file)
    bam_files = [Path(b) for b in args.bam]
    gtf       = Path(args.gtf)

    # Validate all inputs before starting any runs
    if not lhs_file.exists():
        sys.exit(f"Error: LHS file not found: {lhs_file}")
    for b in bam_files:
        if not b.exists():
            sys.exit(f"Error: BAM file not found: {b}")
    if not gtf.exists():
        sys.exit(f"Error: GTF file not found: {gtf}")
    if not COSMO_DIR.exists():
        sys.exit(f"Error: COSMO directory not found: {COSMO_DIR}")

    OUTPUT_DST.mkdir(parents=True, exist_ok=True)

    # Load LHS combinations (auto-detect ; or , delimiter)
    delimiter = detect_delimiter(lhs_file)
    with open(lhs_file, newline="") as f:
        combinations = list(csv.DictReader(f, delimiter=delimiter))

    total  = len(combinations)
    n_bams = len(bam_files)

    # Find which (combination_number, bam_idx) pairs are already done
    # so the script can be safely re-run after interruption
    completed_pairs: set[tuple[int, int]] = set()
    for p in OUTPUT_DST.glob("sample_*_bam*.csv"):
        parts = p.stem.split("_bam")
        if len(parts) == 2:
            try:
                combo_num = int(parts[0].replace("sample_", ""))
                bam_idx   = int(parts[1])
                completed_pairs.add((combo_num, bam_idx))
            except ValueError:
                pass

    if completed_pairs:
        print(f"Found {len(completed_pairs)} already completed run(s), skipping them.")

    total_remaining = sum(
        1
        for row in combinations
        for bam_idx in range(args.first_bam_index, args.first_bam_index + n_bams)
        if (int(row["combination_number"]), bam_idx) not in completed_pairs
    )

    print(f"Running COSMO: {total} combinations × {n_bams} BAM file(s) "
          f"= {total * n_bams} total runs ({total_remaining} remaining)\n")

    failed: list[tuple[int, int]] = []

    for row in combinations:
        n      = int(row["combination_number"])
        cds    = row["CDS_min"]
        igr    = row["IGR_min"]
        fd_cds = row["FD_CDS-CDS_min"]
        fd_igr = row["FD_IGR-CDS_min"]

        for bam_idx, bam in enumerate(bam_files, start=args.first_bam_index):
            if (n, bam_idx) in completed_pairs:
                continue

            print(
                f"[combo {n}/{total}, bam {bam_idx}/{n_bams}] "
                f"CDS={cds}  IGR={igr}  FD_CDS={fd_cds}  FD_IGR={fd_igr}  "
                f"BAM={bam.name}",
                end="  ", flush=True
            )

            result, output_name = run_cosmo(
                n, bam_idx, cds, igr, fd_cds, fd_igr,
                bam, gtf, args.genome_name, args.genome_size
            )

            src = OUTPUT_SRC / output_name
            dst = OUTPUT_DST / output_name

            if result.returncode != 0:
                print("FAILED")
                print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
                failed.append((n, bam_idx))
                continue

            if not src.exists():
                print("FAILED (output file not found)")
                failed.append((n, bam_idx))
                continue

            shutil.move(str(src), str(dst))
            print(f"→ {dst.name}")

    completed = total_remaining - len(failed)
    print(f"\nDone. {completed}/{total_remaining} runs completed successfully.")
    if failed:
        print(f"Failed (combo, bam_idx) pairs: {failed}")


if __name__ == "__main__":
    main()
