"""
Merges the LHS parameter combinations with the COSMO evaluation results.

Usage:
    python3 scripts/2.data_generation/merge_lhs_evaluation.py \
        <lhs_csv> <evaluation_csv> [--output <path>]

Arguments:
    lhs_csv          Path to the LHS combinations CSV
                     (e.g. parameter_optimisation/raw_datasets/lhs_combinations.csv)
    evaluation_csv   Path to the evaluation results CSV produced by evaluate_cosmo.py
                     (e.g. parameter_optimisation/raw_datasets/evaluation_results.csv)
    --output         Path to write the merged CSV
                     (default: parameter_optimisation/raw_datasets/raw_dataset_regression.csv)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Merge LHS parameter combinations with COSMO evaluation results."
)
parser.add_argument("lhs_csv", help="Path to the LHS combinations CSV")
parser.add_argument("evaluation_csv", help="Path to the evaluation results CSV")
parser.add_argument(
    "--output",
    default="parameter_optimisation/raw_datasets/raw_dataset_regression.csv",
    metavar="PATH",
    help="Output path for the merged CSV (default: parameter_optimisation/raw_datasets/raw_dataset_regression.csv)",
)
args = parser.parse_args()

lhs_path  = Path(args.lhs_csv)
eval_path = Path(args.evaluation_csv)
out_path  = Path(args.output)

if not lhs_path.exists():
    sys.exit(f"ERROR: LHS file not found: {lhs_path}")
if not eval_path.exists():
    sys.exit(f"ERROR: Evaluation file not found: {eval_path}")

lhs     = pd.read_csv(lhs_path,  sep=None, engine="python")
eval_df = pd.read_csv(eval_path, sep=None, engine="python")

merged = eval_df.merge(
    lhs,
    left_on="run_id",
    right_on="combination_number",
    how="left",
)

param_cols  = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
metric_cols = [c for c in eval_df.columns if c != "run_id"]

out_path.parent.mkdir(parents=True, exist_ok=True)
merged[param_cols + metric_cols].to_csv(out_path, sep=";", index=False)
print(f"Written {out_path} ({len(merged)} rows)")
