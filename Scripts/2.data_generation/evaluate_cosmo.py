"""
Evaluates COSMO operon predictions against experimentally validated operons (EVOs).

When multiple BAM files were used (outputs named sample_{n}_bam{i}.csv), predicted
operons from all BAM files for the same combination are unioned before evaluation.

Definitions:
  TP  — predicted operon whose gene set exactly matches an EVO gene set
  FN  — EVO not exactly matched by any predicted operon

Metrics reported per combination:
  TP, FN, TP%, Recall, F1

EVO reference CSV format (two columns):
  Evo_nr    — integer identifier
  Gene_list — newline-separated gene names, e.g.:
                gene-CD630_RS11950
                gene-CD630_RS11955

Usage:
    python3 scripts/2.data_generation/evaluate_cosmo.py \\
        --cosmo-dir /path/to/cosmo_output \\
        --evo-reference /path/to/evo_reference.csv \\
        --output /path/to/evaluation_results.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COSMO predictions against EVOs, merging across BAM files."
    )
    parser.add_argument(
        "--cosmo-dir",
        default=str(REPO_ROOT / "cosmo_code_&_ supplementary materials" / "cosmo_output"),
        help="Directory containing COSMO output CSV files (default: cosmo_output/)",
    )
    parser.add_argument(
        "--evo-reference",
        default=str(REPO_ROOT / "cosmo_code_&_ supplementary materials" / "evo_reference2.csv"),
        help="Path to the EVO reference CSV (columns: No, EVOs)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "parameter_optimisation" / "raw_datasets" / "evaluation_results.csv"),
        help="Path to write the evaluation results CSV",
    )
    return parser.parse_args()


def load_evos(path: Path) -> list[dict]:
    """Read EVO reference CSV (columns: No, EVOs).

    Genes are semicolon-separated within the EVOs column.
    Rows where EVOs is 'Not expressed' are skipped — those operons are
    undetectable by COSMO and should not count as FN.
    """
    evos = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["EVOs"].strip()
            if raw.lower() == "not expressed":
                continue
            genes = frozenset(g.strip() for g in raw.split(";") if g.strip())
            evos.append({"evo_nr": row["No"], "genes": genes})
    return evos


def load_cosmo_predictions(path: Path) -> list[frozenset]:
    """Return one frozenset of gene names per predicted operon (from column 7)."""
    predictions = []
    current: list[str] = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip():
                if current:
                    predictions.append(frozenset(current))
                current = []
            else:
                gene = row[7].strip() if len(row) > 7 else ""
                if gene:
                    current.append(gene)

    if current:
        predictions.append(frozenset(current))

    return predictions


def classify_predictions(evos: list[dict], predictions: set[frozenset]) -> dict:
    """
    TP: predicted operon exactly matches an EVO gene set
    FN: EVO not matched by any predicted operon
    """
    evo_gene_sets = {evo["genes"] for evo in evos}
    matched_evos: set[frozenset] = set()

    for pred in predictions:
        if pred in evo_gene_sets:
            matched_evos.add(pred)

    tp = len(matched_evos)
    fn = len(evos) - tp

    return {
        "TP": tp, "FN": fn,
        "total_EVOs": len(evos),
        "total_predicted": len(predictions),
    }


def compute_metrics(counts: dict) -> dict:
    tp = counts["TP"]
    fn = counts["FN"]

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * recall / (1 + recall) if (1 + recall) > 0 else 0.0

    return {**counts, "Recall": recall, "F1": f1}


def group_files_by_combination(cosmo_dir: Path) -> dict[int, list[Path]]:
    """Group output files by combination number.

    Handles both naming conventions:
      - New: sample_{n}_bam{i}.csv  (multiple BAM files)
      - Old: sample_{n}.csv         (single BAM file)
    """
    groups: dict[int, list[Path]] = defaultdict(list)

    for p in sorted(cosmo_dir.glob("sample_*.csv")):
        stem = p.stem
        if "_bam" in stem:
            combo_num = int(stem.split("_bam")[0].replace("sample_", ""))
        else:
            combo_num = int(stem.replace("sample_", ""))
        groups[combo_num].append(p)

    return groups


def main():
    args      = parse_args()
    cosmo_dir = Path(args.cosmo_dir)
    evo_path  = Path(args.evo_reference)
    out_path  = Path(args.output)

    if not cosmo_dir.exists():
        sys.exit(f"Error: COSMO output directory not found: {cosmo_dir}")
    if not evo_path.exists():
        sys.exit(f"Error: EVO reference file not found: {evo_path}")

    evos   = load_evos(evo_path)
    groups = group_files_by_combination(cosmo_dir)

    if not groups:
        sys.exit(f"No sample_*.csv files found in {cosmo_dir}")

    max_bams = max(len(files) for files in groups.values())
    print(f"Found {len(groups)} combination(s), up to {max_bams} BAM file(s) each.")
    print(f"Reference EVOs: {len(evos)}\n")

    rows = []
    for combo_num in sorted(groups.keys()):
        files = groups[combo_num]

        # Union predicted operons (as gene frozensets) across all BAM files
        merged: set[frozenset] = set()
        for f in files:
            for pred in load_cosmo_predictions(f):
                merged.add(pred)

        counts  = classify_predictions(evos, merged)
        metrics = compute_metrics(counts)

        tp_pct = round(metrics["TP"] / metrics["total_EVOs"] * 100, 1) if metrics["total_EVOs"] > 0 else 0.0

        row = {
            "run_id":          combo_num,
            "TP%":             tp_pct,
            "TP":              metrics["TP"],
            "FN":              metrics["FN"],
            "total_predicted": metrics["total_predicted"],
            "Recall":          round(metrics["Recall"], 3),
            "F1":              round(metrics["F1"], 3),
            "bam_files":       len(files),
        }
        rows.append(row)

        print(
            f"Combo {combo_num:<4}  BAMs={len(files)}  TP%={row['TP%']:>5}  "
            f"TP={row['TP']}  FN={row['FN']}  "
            f"Recall={row['Recall']}  F1={row['F1']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "TP%", "TP", "FN", "total_predicted",
        "Recall", "F1", "bam_files",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path}")
    print(f"  {len(rows)} combinations evaluated")


if __name__ == "__main__":
    main()
