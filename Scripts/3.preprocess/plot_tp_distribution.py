"""
Plot the distribution of TP% scores from an evaluation_results.csv file.

Output is saved automatically to:
    parameter_optimisation/analysis/tp_distribution_<dataset>.png

The analysis directory is created if it does not exist.
The filename is derived from the input dataset's parent folder name.

Usage:
    python3 scripts/3.preprocess/plot_tp_distribution.py \
        --input path/to/evaluation_results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent / "parameter_optimisation" / "analysis"


def detect_delimiter(path: Path) -> str:
    with open(path, newline="") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","


def main():
    parser = argparse.ArgumentParser(
        description="Plot TP% distribution from evaluation_results.csv."
    )
    parser.add_argument("--input", required=True, help="Path to evaluation_results.csv")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Error: file not found: {path}")

    delim = detect_delimiter(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=delim))

    if not rows:
        sys.exit("Error: file is empty")

    tp_values = [float(row["TP%"]) for row in rows]
    n = len(tp_values)
    best = max(tp_values)
    worst = min(tp_values)
    mean = sum(tp_values) / n
    sorted_vals = sorted(tp_values)
    median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in tp_values) / n
    std = variance ** 0.5
    n_best = sum(1 for v in tp_values if v == best)

    thresholds = [50, 60, 70, 75]
    above = {t: sum(1 for v in tp_values if v >= t) for t in thresholds}

    print(f"TP% summary  (n={n} combinations)")
    print(f"  Best:    {best:.1f}%  ({n_best} combination(s))")
    print(f"  Worst:   {worst:.1f}%")
    print(f"  Mean:    {mean:.1f}%")
    print(f"  Median:  {median:.1f}%")
    print(f"  Std dev: {std:.1f}%")
    print()
    for t in thresholds:
        print(f"  >= {t}%:  {above[t]} combinations  ({above[t]/n*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(tp_values, bins=20, color="#4c72b0", edgecolor="white", linewidth=0.6)

    ax.axvline(best, color="#dd4444", linestyle="--", linewidth=1.4, label=f"Best: {best:.1f}%")
    ax.axvline(mean, color="#f0a030", linestyle="--", linewidth=1.4, label=f"Mean: {mean:.1f}%")

    ax.set_xlabel("TP%", fontsize=12)
    ax.set_ylabel("Number of parameter combinations", fontsize=12)
    ax.set_title(f"Distribution of TP% scores  (n={n})", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    dataset_name = Path(args.input).parent.name.replace(" ", "_")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / f"tp_distribution_{dataset_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
