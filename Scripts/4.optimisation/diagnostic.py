"""
Diagnostic Script — COSMO Parameter Optimisation
=================================================
Loads the merged LHS + evaluation dataset and answers one question:
"What is the smartest route to find the optimal parameters?"

Checks performed:
  1. Metric variation   — is there enough signal to model?
  2. Parameter patterns — do any parameters correlate with the target?
  3. Top-performer clustering — do the best runs share similar parameter values?

Prints a recommendation at the end.

Usage:
    python3 scripts/4.optimisation/diagnostic.py <merged_csv> [--target TP%]

    <merged_csv> is the file produced by merge_lhs_evaluation.py, e.g.
    parameter_optimisation/raw_datasets/raw_dataset_regression.csv
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REGRESSORS = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
OUTPUT_DIR = "parameter_optimisation/analysis"

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Diagnostic tool to decide the best optimisation route."
)
parser.add_argument("merged_csv", help="Merged LHS + evaluation CSV")
parser.add_argument("--target", default="TP%", metavar="COL",
                    help="Metric to optimise (default: TP%%)")
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(args.merged_csv, sep=None, engine="python")

missing = [c for c in REGRESSORS + [args.target] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: column(s) not found: {missing}")

TARGET = args.target
y = df[TARGET]

print("=" * 65)
print(f"DIAGNOSTIC REPORT  (target: {TARGET})")
print("=" * 65)
print(f"  Runs loaded: {len(df)}")

# ── Check 1: Metric variation ─────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("CHECK 1 — METRIC VARIATION")
print(f"{'─'*65}")

y_min, y_max   = y.min(), y.max()
y_std          = y.std()
y_range        = y_max - y_min
n_unique       = y.nunique()

print(f"  Min    : {y_min:.2f}")
print(f"  Max    : {y_max:.2f}")
print(f"  Range  : {y_range:.2f}")
print(f"  Std    : {y_std:.2f}")
print(f"  Unique values: {n_unique}")

# Flag: is there enough variation to model?
variation_ok = y_range > 20 and n_unique >= 8
if variation_ok:
    print(f"\n  ✓ Sufficient variation — modelling approaches are viable.")
else:
    print(f"\n  ✗ Low variation — the response surface is likely too flat to model.")

# ── Check 2: Spearman correlations ────────────────────────────────────────────
print(f"\n{'─'*65}")
print("CHECK 2 — PARAMETER CORRELATIONS WITH TARGET (Spearman)")
print(f"{'─'*65}")
print(f"  {'Parameter':<22}  {'ρ':>6}  {'p-value':>10}  {'Signal'}")
print(f"  {'─'*22}  {'─'*6}  {'─'*10}  {'─'*6}")

correlations = {}
for col in REGRESSORS:
    rho, pval = spearmanr(df[col], y)
    sig = "YES" if pval < 0.05 else "no"
    print(f"  {col:<22}  {rho:>6.3f}  {pval:>10.4f}  {sig}")
    correlations[col] = (rho, pval)

n_significant = sum(1 for rho, p in correlations.values() if p < 0.05)
any_signal = n_significant > 0

if any_signal:
    print(f"\n  ✓ {n_significant} parameter(s) show significant correlation — "
          f"modelling may find a gradient.")
else:
    print(f"\n  ✗ No significant correlations — no detectable monotone relationship.")

# ── Check 3: Top-performer clustering ─────────────────────────────────────────
print(f"\n{'─'*65}")
print("CHECK 3 — TOP-PERFORMER CLUSTERING")
print(f"{'─'*65}")

top_pct   = 0.20
top_n     = max(5, int(len(df) * top_pct))
top_df    = df.nlargest(top_n, TARGET)

print(f"  Comparing parameter ranges: top {top_n} runs vs all {len(df)} runs\n")
print(f"  {'Parameter':<22}  {'Full range':>12}  {'Top range':>12}  {'Ratio':>7}")
print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*7}")

ratios = []
for col in REGRESSORS:
    full_range = df[col].max() - df[col].min()
    top_range  = top_df[col].max() - top_df[col].min()
    ratio      = top_range / full_range if full_range > 0 else 1.0
    ratios.append(ratio)
    print(f"  {col:<22}  {full_range:>12.2f}  {top_range:>12.2f}  {ratio:>7.2f}")

mean_ratio = np.mean(ratios)
clustered  = mean_ratio < 0.55

print(f"\n  Mean range ratio: {mean_ratio:.2f}  "
      f"({'clustered — sweet spot exists' if clustered else 'spread out — no clear sweet spot'})")

# ── Top 10 performers ─────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"TOP 10 RUNS BY {TARGET}")
print(f"{'─'*65}")
top10 = df.nlargest(10, TARGET)[REGRESSORS + [TARGET, "F1", "Precision", "Recall"]]
print(top10.to_string(index=False))

# ── Recommendation ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("RECOMMENDATION")
print(f"{'='*65}")

if not variation_ok:
    print("""
  The target metric has very low variation across your 251 runs.
  Even with the corrected 3-BAM evaluation, the parameters do not
  meaningfully separate good runs from bad ones in this range.

  → Route 1 (empirical selection): pick the top performers directly.
    More modelling will not help without more variation in the data.
    Consider whether running more COSMO combinations (Route 4) across
    a wider parameter range would expose a steeper gradient.
""")
elif clustered and any_signal:
    print(f"""
  Top performers cluster in a sub-region of the parameter space AND
  at least one parameter shows a significant correlation with {TARGET}.

  → Route 3 (coarse-to-fine): run a dense grid within the parameter
    ranges of the top {top_n} runs shown above. This is efficient and
    directly exploits what you already know from the LHS data.
    After that, validate the best combination against the 28 EVOs.
""")
elif any_signal and variation_ok:
    print(f"""
  There is meaningful variation in {TARGET} and significant parameter
  correlations, but the top performers are spread across the space.

  → Route 2 (MLR + GP): re-run both models on the corrected dataset.
    With real variation now accessible, the GP may find a reliable
    optimum. Check the GP test R² — if > 0.4, trust the predicted
    optimum and validate it directly.
""")
else:
    print(f"""
  There is variation in {TARGET} but no detectable parameter patterns
  and no clustering of top performers.

  → Route 1 (empirical selection): validate the top 10 combinations
    above directly against the 28 EVOs using the full metric set.
    Report the best validated combination as the optimum.
    Optionally proceed to Route 4 (Bayesian optimisation) if you
    need a more principled search for the thesis.
""")

# ── Plot: parameter vs target scatter ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), REGRESSORS):
    rho, pval = correlations[col]
    ax.scatter(df[col], y, alpha=0.4, s=15, edgecolors="none", color="steelblue")
    ax.scatter(top_df[col], top_df[TARGET], alpha=0.8, s=25,
               edgecolors="k", linewidths=0.4, color="darkorange", label="Top 20%")
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{col}  (ρ={rho:.2f}, p={pval:.3f})")
    ax.legend(fontsize=7)

plt.suptitle(f"Parameter vs {TARGET} — top 20% highlighted", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_scatter.png"), dpi=150)
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/diagnostic_scatter.png")
