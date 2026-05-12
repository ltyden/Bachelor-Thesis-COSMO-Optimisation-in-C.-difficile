"""
MLR Fitting Script — COSMO Parameter Optimisation
==================================================
Fits a multiple linear regression model (statsmodels OLS) predicting
operon-prediction accuracy (TP%) from the four COSMO parameters:
  CDS_min, IGR_min, FD_CDS-CDS_min, FD_IGR-CDS_min

Outputs to console:
  - Full OLS model summary
  - 95% confidence intervals on all coefficients
  - 5-fold cross-validation (mean R², std R², mean RMSE)
  - Partial F-tests: each regressor dropped once, compared to full model
  - Test-set RMSE from an 80/20 split

Saves four diagnostic plots to OUTPUT_DIR:
  residuals_vs_fitted.png, qq_plot.png, scale_location.png,
  actual_vs_predicted.png

Usage
-----
  python3 scripts/mlr_fitting.py <input_csv> [--log] [--output-dir <dir>]

Arguments
---------
  input_csv           Path to the CSV file containing the LHS results.
                      Required columns: CDS_min, IGR_min, FD_CDS-CDS_min,
                      FD_IGR-CDS_min, TP%

  --log               Apply log(TP% + 1) transform to the response variable.
                      Omit this flag to fit on raw TP%.

  --output-dir <dir>  Directory where diagnostic plots are saved.
                      Default: parameter_optimisation/analysis

Examples
--------
  # Fit on raw TP%, plots go to the default output directory
  python3 scripts/mlr_fitting.py parameter_optimisation/raw_datasets/lhs_results.csv

  # Fit on log(TP%+1), save plots to a custom directory
  python3 scripts/mlr_fitting.py results.csv --log --output-dir my_analysis/plots
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

# ── Parse command-line arguments ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Fit an MLR model (statsmodels OLS) predicting TP% from "
                "the four COSMO parameters and produce diagnostic plots.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "input_csv",
    help="Path to the CSV containing LHS results "
         "(must have columns: CDS_min, IGR_min, FD_CDS-CDS_min, FD_IGR-CDS_min, TP%%)",
)
parser.add_argument(
    "--log",
    action="store_true",
    default=False,
    help="Apply log(TP%% + 1) transform to the response variable (default: raw TP%%)",
)
parser.add_argument(
    "--output-dir",
    default="parameter_optimisation/analysis",
    metavar="DIR",
    help="Directory where diagnostic plots are saved "
         "(default: parameter_optimisation/analysis)",
)
args = parser.parse_args()

INPUT_CSV       = args.input_csv
LOG_TRANSFORM_Y = args.log
OUTPUT_DIR      = args.output_dir

REGRESSORS = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
TARGET     = "TP%"

# ── Prepare output directory ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and validate data ────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")  # auto-detect , or ;

missing = [c for c in REGRESSORS + [TARGET] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: column(s) not found in CSV: {missing}")

X     = df[REGRESSORS].to_numpy(dtype=float)
y_raw = df[TARGET].to_numpy(dtype=float)

# Apply log transform if requested
if LOG_TRANSFORM_Y:
    y       = np.log1p(y_raw)
    y_label = "log(TP% + 1)"
else:
    y       = y_raw
    y_label = "TP%"

# ── 80/20 train-test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Add constant (intercept) column to training and test sets
X_train_c = sm.add_constant(X_train)
X_test_c  = sm.add_constant(X_test)

# ── Fit full OLS model on training data ───────────────────────────────────────
model   = sm.OLS(y_train, X_train_c)
results = model.fit()

# Attach readable names so they appear in summary tables
param_names = ["const"] + REGRESSORS
results.model.exog_names[:] = param_names   # mutate in-place for display

# ── Console: full model summary ───────────────────────────────────────────────
print("=" * 70)
print(f"FULL MODEL SUMMARY  (y = {y_label})")
print("=" * 70)
print(results.summary())

# ── Console: 95% confidence intervals ────────────────────────────────────────
print("\n95% Confidence Intervals on Coefficients")
print("-" * 50)
ci = pd.DataFrame(results.conf_int(alpha=0.05), index=param_names, columns=["2.5%", "97.5%"])
print(ci.to_string())

# ── 5-fold cross-validation on the full dataset ───────────────────────────────
X_full_c = sm.add_constant(X)   # full design matrix with intercept

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2, cv_rmse = [], []

for train_idx, val_idx in kf.split(X_full_c):
    X_cv_tr, X_cv_val = X_full_c[train_idx], X_full_c[val_idx]
    y_cv_tr, y_cv_val = y[train_idx], y[val_idx]

    cv_res     = sm.OLS(y_cv_tr, X_cv_tr).fit()
    y_pred_val = cv_res.predict(X_cv_val)

    # Compute R² manually so it is consistent across folds
    ss_res = np.sum((y_cv_val - y_pred_val) ** 2)
    ss_tot = np.sum((y_cv_val - np.mean(y_cv_val)) ** 2)
    cv_r2.append(1.0 - ss_res / ss_tot)
    cv_rmse.append(np.sqrt(mean_squared_error(y_cv_val, y_pred_val)))

print("\n5-Fold Cross-Validation  (full dataset)")
print("-" * 50)
print(f"  Mean R²  : {np.mean(cv_r2):.4f}")
print(f"  Std  R²  : {np.std(cv_r2):.4f}")
print(f"  Mean RMSE: {np.mean(cv_rmse):.4f}  ({y_label})")

# ── Partial F-tests: drop each regressor in turn ─────────────────────────────
# Column 0 of X_train_c is the constant; regressors occupy columns 1..k.
print("\nPartial F-Tests  (full model vs. model with one regressor dropped)")
print("-" * 65)
print(f"{'Dropped variable':>22}  {'F-statistic':>12}  {'p-value':>10}")
print("-" * 65)

for i, var in enumerate(REGRESSORS):
    # Keep all columns except the one being dropped (index i+1)
    keep_cols  = [0] + [j for j in range(1, len(REGRESSORS) + 1) if j != i + 1]
    X_reduced  = X_train_c[:, keep_cols]
    res_red    = sm.OLS(y_train, X_reduced).fit()

    # compare_f_test: tests whether full model is a significant improvement
    # over the restricted (reduced) model
    F_stat, p_val, _ = results.compare_f_test(res_red)
    print(f"{var:>22}  {F_stat:>12.4f}  {p_val:>10.6f}")

# ── Test-set RMSE ─────────────────────────────────────────────────────────────
y_pred_test = results.predict(X_test_c)
test_rmse   = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"\nTest-set RMSE ({y_label}): {test_rmse:.4f}")

# ── Residuals and influence statistics for diagnostic plots ───────────────────
y_fitted  = results.fittedvalues
residuals = results.resid
std_resid = results.get_influence().resid_studentized_internal

# ── Plot 1: Residuals vs Fitted ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_fitted, residuals, alpha=0.65, edgecolors="k", linewidths=0.4)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel(f"Fitted values ({y_label})")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_vs_fitted.png"), dpi=150)
plt.close()

# ── Plot 2: Normal Q-Q of Residuals ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
(osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
ax.scatter(osm, osr, alpha=0.65, edgecolors="k", linewidths=0.4)
ax.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "qq_plot.png"), dpi=150)
plt.close()

# ── Plot 3: Scale-Location ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_fitted, np.sqrt(np.abs(std_resid)),
           alpha=0.65, edgecolors="k", linewidths=0.4)
ax.set_xlabel(f"Fitted values ({y_label})")
ax.set_ylabel("√|Standardised Residuals|")
ax.set_title("Scale-Location")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scale_location.png"), dpi=150)
plt.close()

# ── Plot 4: Actual vs Predicted (training set) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_train, y_fitted, alpha=0.65, edgecolors="k", linewidths=0.4)
# 45-degree reference line spanning the full range
lo = min(y_train.min(), y_fitted.min())
hi = max(y_train.max(), y_fitted.max())
ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
ax.set_xlabel(f"Actual {y_label}")
ax.set_ylabel(f"Predicted {y_label}")
ax.set_title("Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

print(f"\nDiagnostic plots saved to: {OUTPUT_DIR}/")
