"""
Evaluates COSMO operon predictions against experimentally validated operons (EVOs).

For each EVO, the best-matching predicted operon is found and classified as:
  TP  — predicted operon contains exactly the EVO gene set (full-length match)
  FP  — predicted operon contains all EVO genes plus extra (over-prediction)
  FN  — no predicted operon covers all EVO genes (under-prediction or missed)

Metrics reported:
  Recall    = TP / (TP + FN)
  Precision = TP / (TP + FP)
  F1        = 2 * Precision * Recall / (Precision + Recall)
  Accuracy  = TP / total EVOs  (fraction of EVOs exactly recovered)
"""

import csv
import sys
from pathlib import Path

BASE = Path("/Users/linustyden/Bachelor-Thesis-COSMO-Optimisation-in-C.-difficile")
EVO_PATH     = BASE / "cosmo_code_&_ supplementary materials" / "evo_reference.csv"
COSMO_DIR    = BASE / "cosmo_code_&_ supplementary materials" / "cosmo_output"
OUTPUT_PATH  = BASE / "parameter_optimisation" / "evaluation_results.csv"


def load_evos(path: Path) -> list[dict]:
    evos = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genes = frozenset(row["evo_genes"].split(";"))
            evos.append({"evo_id": row["evo_id"], "genes": genes})
    return evos


def load_cosmo_predictions(path: Path) -> list[frozenset]:
    predictions = []
    current_genes: list[str] = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip():
                if current_genes:
                    predictions.append(frozenset(current_genes))
                current_genes = []
            else:
                raw = row[7].strip() if len(row) > 7 else ""
                if raw:
                    gene_id = raw.removeprefix("gene-")
                    current_genes.append(gene_id)

    if current_genes:
        predictions.append(frozenset(current_genes))

    return predictions


def classify_evos(evos: list[dict], predictions: list[frozenset]) -> list[dict]:
    results = []
    for evo in evos:
        evo_genes = evo["genes"]

        # Find predicted operon with highest overlap
        best = max(predictions, key=lambda p: len(p & evo_genes), default=frozenset())
        overlap = best & evo_genes

        if best == evo_genes:
            outcome = "TP"
        elif evo_genes <= best:
            outcome = "FP"
        else:
            outcome = "FN"

        results.append({
            "evo_id": evo["evo_id"],
            "outcome": outcome,
            "evo_size": len(evo_genes),
            "matched_size": len(best),
            "overlap": len(overlap),
        })
    return results


def compute_metrics(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["outcome"] == "TP")
    fp = sum(1 for r in results if r["outcome"] == "FP")
    fn = sum(1 for r in results if r["outcome"] == "FN")
    total = len(results)

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = tp / total if total > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "total_EVOs": total,
        "Accuracy":  accuracy,
        "Recall":    recall,
        "Precision": precision,
        "F1":        f1,
    }


def evaluate_file(run_id: int, cosmo_path: Path, evos: list[dict]) -> dict:
    predictions = load_cosmo_predictions(cosmo_path)
    results = classify_evos(evos, predictions)
    metrics = compute_metrics(results)
    return {
        "run_id":    run_id,
        "TP%":       round(metrics["Accuracy"] * 100, 1),
        "TP":        metrics["TP"],
        "FN":        metrics["FN"],
        "FP":        metrics["FP"],
        "Precision": round(metrics["Precision"], 3),
        "Recall":    round(metrics["Recall"], 3),
        "F1":        round(metrics["F1"], 3),
    }


def main():
    evos = load_evos(EVO_PATH)
    csv_files = sorted(COSMO_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {COSMO_DIR}")
        sys.exit(1)

    rows = []
    for run_id, path in enumerate(csv_files, start=1):
        row = evaluate_file(run_id, path, evos)
        rows.append(row)
        print(f"Run {row['run_id']:<4}  TP%={row['TP%']:>5}  TP={row['TP']}  "
              f"FN={row['FN']}  FP={row['FP']}  "
              f"Precision={row['Precision']}  Recall={row['Recall']}  F1={row['F1']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "TP%", "TP", "FN", "FP", "Precision", "Recall", "F1"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
