"""
Raw Dataset Visualisation
=========================
Plots distributions and parameter-vs-metric relationships to inform
decisions about transformations before preprocessing.

Saves to parameter_optimisation/analysis/:
  distributions.png   — histograms for all parameters and metrics
  scatter_vs_tp.png   — each parameter vs TP%
  scatter_vs_f1.png   — each parameter vs F1
  correlation.png     — correlation heatmap across all variables

Usage:
    python3 scripts/3.preprocess/visualise_raw.py <merged_csv>

    e.g. python3 scripts/3.preprocess/visualise_raw.py \
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
from scipy.stats import skew

REGRESSORS = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
METRICS    = ["TP%", "F1", "Precision", "Recall"]
OUTPUT_DIR = "parameter_optimisation/analysis"

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Visualise the raw LHS dataset to inform preprocessing decisions."
)
parser.add_argument("merged_csv", help="Merged LHS + evaluation CSV")
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(args.merged_csv, sep=None, engine="python")

available_metrics = [m for m in METRICS if m in df.columns]
missing = [c for c in REGRESSORS if c not in df.columns]
if missing:
    sys.exit(f"ERROR: column(s) not found: {missing}")

all_cols = REGRESSORS + available_metrics

# ── Console: skewness report ───────────────────────────────────────────────────
print(f"{'='*55}")
print("DISTRIBUTION SUMMARY")
print(f"{'='*55}")
print(f"  {'Column':<22}  {'Mean':>7}  {'Std':>7}  {'Skew':>7}  {'Note'}")
print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*20}")

for col in all_cols:
    s     = skew(df[col].dropna())
    note  = ""
    if abs(s) > 1.0:
        note = "⚠ highly skewed — consider log"
    elif abs(s) > 0.5:
        note = "moderately skewed"
    print(f"  {col:<22}  {df[col].mean():>7.2f}  {df[col].std():>7.2f}  {s:>7.2f}  {note}")

# ── Plot 1: Distributions ──────────────────────────────────────────────────────
n_cols = len(all_cols)
fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(14, 7))
axes = axes.ravel()

for ax, col in zip(axes, all_cols):
    ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="white", linewidth=0.4)
    s = skew(df[col].dropna())
    ax.set_title(f"{col}\nskew={s:.2f}", fontsize=9)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[n_cols:]:
    ax.set_visible(False)

plt.suptitle("Variable Distributions (raw data)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distributions.png"), dpi=150)
plt.close()

# ── Plot 2 & 3: Each parameter vs TP% and F1 ──────────────────────────────────
for metric in ["TP%", "F1"]:
    if metric not in df.columns:
        continue

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), REGRESSORS):
        ax.scatter(df[col], df[metric], alpha=0.4, s=14,
                   edgecolors="none", color="steelblue")
        # Overlay a simple trend line
        m, b = np.polyfit(df[col].dropna(), df[metric].dropna(), 1)
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, m * x_line + b, color="red",
                linewidth=1, linestyle="--", label=f"slope={m:.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel(metric)
        ax.set_title(col)
        ax.legend(fontsize=7)

    plt.suptitle(f"Parameter vs {metric} (raw data)", fontsize=11)
    plt.tight_layout()
    fname = f"scatter_vs_{'tp' if metric == 'TP%' else metric.lower()}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()

# ── Plot 4: Correlation heatmap ────────────────────────────────────────────────
corr = df[all_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
fig.colorbar(im, ax=ax, label="Pearson r")

ax.set_xticks(range(len(all_cols)))
ax.set_yticks(range(len(all_cols)))
ax.set_xticklabels(all_cols, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(all_cols, fontsize=8)

for i in range(len(all_cols)):
    for j in range(len(all_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")

ax.set_title("Correlation Matrix (raw data)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation.png"), dpi=150)
plt.close()

print(f"\nPlots saved to {OUTPUT_DIR}/")
print("  distributions.png")
print("  scatter_vs_tp.png")
print("  scatter_vs_f1.png")
print("  correlation.png")
