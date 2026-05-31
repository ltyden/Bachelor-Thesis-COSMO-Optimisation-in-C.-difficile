"""
Find the parameter combination(s) with the highest TP% in a regression dataset.

Usage:
    python3 scripts/4.optimisation/find_best_params.py \
        --input path/to/datagrid
"""

import argparse
import csv
import sys
from pathlib import Path


def detect_delimiter(path: Path) -> str:
    with open(path, newline="") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","


def main():
    parser = argparse.ArgumentParser(
        description="Report parameter combination(s) with the highest TP%."
    )
    parser.add_argument("--input", required=True, help="Path to raw_dataset_regression.csv")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Error: file not found: {path}")

    delim = detect_delimiter(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=delim))

    if not rows:
        sys.exit("Error: file is empty")

    best_tp = max(float(row["TP%"]) for row in rows)
    best = [row for row in rows if float(row["TP%"]) == best_tp]

    print(f"Best TP%: {best_tp}%  ({len(best)} combination(s))\n")
    header = f"{'CDS_min':>8}  {'IGR_min':>8}  {'FD_CDS':>8}  {'FD_IGR':>8}  {'TP%':>6}  {'TP':>4}  {'FP':>6}  {'FN':>4}"
    print(header)
    print("-" * len(header))
    for row in best:
        print(
            f"{row['CDS_min']:>8}  {row['IGR_min']:>8}  "
            f"{row['FD_CDS-CDS_min']:>8}  {row['FD_IGR-CDS_min']:>8}  "
            f"{float(row['TP%']):>6.1f}  {row['TP']:>4}  {row['FP']:>6}  {row['FN']:>4}"
        )


if __name__ == "__main__":
    main()
