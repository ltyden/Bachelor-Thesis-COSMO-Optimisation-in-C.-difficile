"""
Evaluates COSMO operon predictions against experimentally validated operons (EVOs).

When multiple BAM files were used (outputs named sample_{n}_bam{i}.csv), the
predicted operons from all BAM files for the same combination are unioned before
evaluation. This ensures all 28 EVOs can be detected even if individual BAM
files only capture a subset.

For each EVO, the best-matching predicted operon is found and classified
according to the chosen scoring mode (default: lenient):

  Lenient  TP  — predicted operon exactly matches or is a superset of the EVO
                 gene set (all EVO genes found, boundary may be too wide)
  Lenient  FN  — no predicted operon covers all EVO genes

  Strict   TP  — predicted operon exactly matches the EVO gene set
  Strict   FP  — predicted operon contains all EVO genes plus extra flanking genes
  Strict   FN  — no predicted operon covers all EVO genes

Note: under lenient scoring FP is always 0 and Precision always 1.0.

Metrics reported per combination:
  TP%, TP, FN, FP, Precision, Recall, F1

Usage:
    # Lenient scoring (default)
    python3 scripts/2.data_generation/evaluate_cosmo.py \
        --cosmo-dir /path/to/cosmo_output \
        --evo-reference /path/to/evo_reference.csv \
        --output /path/to/evaluation_results.csv

    # Strict scoring
    python3 scripts/2.data_generation/evaluate_cosmo.py \
        --cosmo-dir /path/to/cosmo_output \
        --evo-reference /path/to/evo_reference.csv \
        --output /path/to/evaluation_results.csv \
        --strict
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
        default=str(REPO_ROOT / "cosmo_code_&_ supplementary materials" / "evo_reference.csv"),
        help="Path to the EVO reference CSV file",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "parameter_optimisation" / "raw_datasets" / "evaluation_results.csv"),
        help="Path to write the evaluation results CSV",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Use strict TP scoring: superset matches count as FP instead of TP "
             "(default: lenient — superset matches count as TP)",
    )
    return parser.parse_args()


def load_evos(path: Path) -> list[dict]:
    evos = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genes = frozenset(row["evo_genes"].split(";"))
            evos.append({
                "evo_id": row["evo_id"],
                "genes":  genes,
                "start":  row["gene_range_start"],
                "end":    row["gene_range_end"],
            })
    return evos


def load_cosmo_predictions(path: Path) -> list[frozenset]:
    predictions = []
    current_genes: list[str] = []
    current_skip: bool = False

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip():
                # Save previous operon if it was not skipped
                if current_genes and not current_skip:
                    predictions.append(frozenset(current_genes))
                current_genes = []
                # Skip if NOT EXPRESSED or single-gene (num_genes == 1)
                mean_cov  = row[5].strip() if len(row) > 5 else ""
                num_genes = row[6].strip() if len(row) > 6 else ""
                current_skip = (
                    mean_cov == "NOT EXPRESSED"
                    or num_genes == "1"
                )
            else:
                raw = row[7].strip() if len(row) > 7 else ""
                if raw:
                    gene_id = raw.removeprefix("gene-")
                    current_genes.append(gene_id)

    # Save last operon
    if current_genes and not current_skip:
        predictions.append(frozenset(current_genes))

    return predictions


def classify_evos(evos: list[dict], predictions: list[frozenset],
                   strict: bool = False) -> list[dict]:
    results = []
    for evo in evos:
        evo_genes = evo["genes"]
        start     = evo["start"]
        end       = evo["end"]

        # Find the predicted operon that contains both EVO boundary genes
        matched = next(
            (p for p in predictions if start in p and end in p),
            frozenset()
        )

        if not matched:
            # No prediction contains both boundary genes — FN
            outcome = "FN"
        elif matched == evo_genes:
            # Predicted gene set exactly matches EVO gene set — strict TP
            outcome = "TP"
        elif evo_genes <= matched:
            # Predicted operon is a superset of EVO (expansion) — lenient TP or FP
            outcome = "FP" if strict else "TP"
        else:
            # Predicted operon contains both boundaries but not all EVO genes (contraction)
            outcome = "TP"

        results.append({
            "evo_id":       evo["evo_id"],
            "outcome":      outcome,
            "evo_size":     len(evo_genes),
            "matched_size": len(matched),
            "overlap":      len(matched & evo_genes),
        })
    return results


def compute_metrics(results: list[dict]) -> dict:
    # Under lenient scoring FP is always 0 and Precision always 1.0 — present
    # in the return dict for CSV column compatibility but not meaningful.
    tp    = sum(1 for r in results if r["outcome"] == "TP")
    fp    = sum(1 for r in results if r["outcome"] == "FP")
    fn    = sum(1 for r in results if r["outcome"] == "FN")
    total = len(results)

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = tp / total if total > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "total_EVOs": total,
        "Accuracy": accuracy, "Recall": recall,
        "Precision": precision, "F1": f1,
    }


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
    scoring_label = "strict" if args.strict else "lenient"
    print(f"Found {len(groups)} combination(s), up to {max_bams} BAM file(s) each.")
    print(f"Scoring mode: {scoring_label}\n")

    rows = []
    for combo_num in sorted(groups.keys()):
        files = groups[combo_num]

        # Union predicted operons across all BAM files for this combination
        # so EVOs that only appear in one BAM file can still be detected
        merged: set[frozenset] = set()
        for f in files:
            for pred in load_cosmo_predictions(f):
                merged.add(pred)

        results = classify_evos(evos, list(merged), strict=args.strict)
        metrics = compute_metrics(results)

        row = {
            "run_id":    combo_num,
            "TP%":       round(metrics["Accuracy"] * 100, 1),
            "TP":        metrics["TP"],
            "FN":        metrics["FN"],
            "FP":        metrics["FP"],
            "Precision": round(metrics["Precision"], 3),
            "Recall":    round(metrics["Recall"], 3),
            "F1":        round(metrics["F1"], 3),
            "bam_files": len(files),
        }
        rows.append(row)

        print(
            f"Combo {combo_num:<4}  BAMs={len(files)}  TP%={row['TP%']:>5}  "
            f"TP={row['TP']}  FN={row['FN']}  FP={row['FP']}  "
            f"Precision={row['Precision']}  Recall={row['Recall']}  F1={row['F1']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "TP%", "TP", "FN", "FP", "Precision", "Recall", "F1", "bam_files"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path}")
    print(f"  {len(rows)} combinations evaluated")


if __name__ == "__main__":
    main()
