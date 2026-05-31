"""
MLR Fitting Script — COSMO Parameter Optimisation
==================================================
Fits a multiple linear regression model (statsmodels OLS) predicting
operon-prediction accuracy (TP%) from the four COSMO parameters:
  CDS_min, IGR_min, FD_CDS-CDS_min, FD_IGR-CDS_min

Outputs to console:
  - VIF table for all regressors (multicollinearity check)
  - Full OLS model summary
  - 95% confidence intervals on all coefficients
  - 5-fold cross-validation on training data only (mean R², std R², mean RMSE)
    All RMSE values are reported in both transformed and original TP% units.
  - Partial F-tests: each regressor dropped once, compared to full model
  - Test-set RMSE (both log and raw TP% units when --log is active)

Saves four diagnostic plots to OUTPUT_DIR:
  residuals_vs_fitted.png, qq_plot.png, scale_location.png,
  actual_vs_predicted.png

Usage
-----
  python3 scripts/4.optimisation/mlr_fitting.py path/to/input_file.csv \
      [--transform {none,log,sqrt,cbrt,logit}] \
      [--output-dir path/to/output_directory]

Arguments
---------
  input_csv           Path to the CSV file containing the LHS results.
                      Required columns: CDS_min, IGR_min, FD_CDS-CDS_min,
                      FD_IGR-CDS_min, TP%

  --transform         Transform to apply to the response variable (default: none).
                        none   — raw TP%, no transform
                        log    — log(TP% + 1)              inverse: expm1
                        sqrt   — √TP%                      inverse: square
                        cbrt   — ∛TP%                      inverse: cube
                        logit  — log(TP% / (100 - TP%))   inverse: 100·sigmoid
                                 (0 and 100 are clamped away from the boundary)

  --output-dir        Directory where diagnostic plots are saved.
                      Default: parameter_optimisation/analysis

Examples
--------
  # No transform (default)
  python3 scripts/4.optimisation/mlr_fitting.py path/to/input_file.csv

  # Square-root transform
  python3 scripts/4.optimisation/mlr_fitting.py path/to/input_file.csv \
      --transform sqrt

  # Log transform, custom output directory
  python3 scripts/4.optimisation/mlr_fitting.py path/to/input_file.csv \
      --transform log --output-dir path/to/output_directory
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_text
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
    "--transform",
    choices=["none", "log", "sqrt", "cbrt", "logit"],
    default="none",
    help="Transform to apply to TP%% before fitting "
         "(none|log|sqrt|cbrt|logit, default: none)",
)
parser.add_argument(
    "--output-dir",
    default="parameter_optimisation/analysis",
    metavar="DIR",
    help="Directory where diagnostic plots are saved "
         "(default: parameter_optimisation/analysis)",
)
args = parser.parse_args()

INPUT_CSV  = args.input_csv
TRANSFORM  = args.transform
OUTPUT_DIR = args.output_dir

REGRESSORS = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
TARGET     = "TP%"

# ── Transform / inverse-transform helpers ────────────────────────────────────
_EPS = 1e-6   # guard for logit at 0 and 100

TRANSFORMS = {
    "none":  (lambda y: y,
              lambda y: y,
              "TP%"),
    "log":   (lambda y: np.log1p(y),
              lambda y: np.expm1(y),
              "log(TP% + 1)"),
    "sqrt":  (lambda y: np.sqrt(y),
              lambda y: np.square(y),
              "√TP%"),
    "cbrt":  (lambda y: np.cbrt(y),
              lambda y: np.power(y, 3),
              "∛TP%"),
    "logit": (lambda y: np.log(np.clip(y, _EPS, 100 - _EPS) /
                               (100 - np.clip(y, _EPS, 100 - _EPS))),
              lambda y: 100 / (1 + np.exp(-y)),
              "logit(TP%/100)"),
}

transform_fn, inverse_fn, y_label = TRANSFORMS[TRANSFORM]

# ── Prepare output directory ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and validate data ────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")  # auto-detect , or ;

missing = [c for c in REGRESSORS + [TARGET] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: column(s) not found in CSV: {missing}")

X     = df[REGRESSORS].to_numpy(dtype=float)
y_raw = df[TARGET].to_numpy(dtype=float)
y     = transform_fn(y_raw)

print(f"Transform: {TRANSFORM}  →  y = {y_label}")

# ── FIX 1 — VIF table before any modelling ───────────────────────────────────
# Variance Inflation Factor measures how much the variance of each coefficient
# is inflated due to linear relationships with the other regressors.
# VIF = 1/(1 - R²_j) where R²_j is obtained by regressing regressor j on
# all remaining regressors.  Rule of thumb: VIF > 10 indicates problematic
# multicollinearity; VIF > 5 is worth noting.
print("=" * 70)
print("VARIANCE INFLATION FACTORS  (multicollinearity check)")
print("=" * 70)

X_with_const = sm.add_constant(X)   # shape (n, 5); column 0 is the intercept

vif_data = pd.DataFrame({
    "Variable": ["const"] + REGRESSORS,
    "VIF": [
        variance_inflation_factor(X_with_const, i)
        for i in range(X_with_const.shape[1])
    ],
})
# The VIF for the intercept is not informative; drop it from display
vif_display = vif_data[vif_data["Variable"] != "const"].copy()
vif_display["Flag"] = vif_display["VIF"].apply(
    lambda v: "*** HIGH (> 10)" if v > 10 else ("*  elevated (> 5)" if v > 5 else "")
)
print(vif_display.to_string(index=False))

high_vif = vif_display[vif_display["VIF"] > 10]
if not high_vif.empty:
    print(
        f"\nWARNING: {list(high_vif['Variable'])} have VIF > 10. "
        "Coefficient estimates and individual p-values may be unreliable. "
        "Consider removing or combining correlated predictors."
    )
else:
    print("\nNo severe multicollinearity detected (all VIF ≤ 10).")

# ── 80/20 train-test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# Also keep the corresponding raw TP% test values for back-transformed RMSE
_, _, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.20, random_state=42
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
print("\n" + "=" * 70)
print(f"FULL MODEL SUMMARY  (y = {y_label}, fitted on 80% training data)")
print("=" * 70)
print(results.summary())

# ── Console: 95% confidence intervals ────────────────────────────────────────
print("\n95% Confidence Intervals on Coefficients")
print("-" * 50)
ci = pd.DataFrame(
    results.conf_int(alpha=0.05),
    index=param_names,
    columns=["2.5%", "97.5%"],
)
print(ci.to_string())

# ── FIX 2 — 5-fold CV on training data only ───────────────────────────────────
# Previously CV ran on the full dataset, meaning the held-out 20% test rows
# participated in CV folds.  The test-set RMSE and CV metrics were therefore
# not independent.  Now both CV and model fitting operate exclusively on the
# 80% training split; the test set is touched only once, at the very end.
print("\n5-Fold Cross-Validation  (training data only — test set is untouched)")
print("-" * 65)

X_train_full_c = sm.add_constant(X_train)   # already has intercept column

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2, cv_rmse_log, cv_rmse_raw = [], [], []

for train_idx, val_idx in kf.split(X_train_full_c):
    X_cv_tr,  X_cv_val  = X_train_full_c[train_idx], X_train_full_c[val_idx]
    y_cv_tr,  y_cv_val  = y_train[train_idx],         y_train[val_idx]
    # raw TP% counterpart (needed for back-transformed RMSE when --log is on)
    y_raw_val = y_train_raw[val_idx]

    cv_res     = sm.OLS(y_cv_tr, X_cv_tr).fit()
    y_pred_val = cv_res.predict(X_cv_val)

    # R² computed against the validation fold's own mean (not the training mean)
    # so it is a fair out-of-sample measure
    ss_res = np.sum((y_cv_val - y_pred_val) ** 2)
    ss_tot = np.sum((y_cv_val - np.mean(y_cv_val)) ** 2)
    cv_r2.append(1.0 - ss_res / ss_tot)
    cv_rmse_log.append(np.sqrt(mean_squared_error(y_cv_val, y_pred_val)))

    if TRANSFORM != "none":
        y_pred_raw = inverse_fn(y_pred_val)
        cv_rmse_raw.append(np.sqrt(mean_squared_error(y_raw_val, y_pred_raw)))

print(f"  Mean R²        : {np.mean(cv_r2):.4f}")
print(f"  Std  R²        : {np.std(cv_r2):.4f}")
print(f"  Mean RMSE      : {np.mean(cv_rmse_log):.4f}  ({y_label})")
if TRANSFORM != "none":
    print(f"  Mean RMSE (raw): {np.mean(cv_rmse_raw):.4f}  (TP%  — back-transformed)")

# ── FIX 4 — Partial F-tests with explicit direction comment ───────────────────
# compare_f_test(restricted) is called on the *full* model object (results).
# statsmodels computes: F = [(RSS_restricted - RSS_full) / q] / [RSS_full / df_full]
# where q = number of restrictions (1 per dropped variable).
# A high F / low p-value means dropping that variable significantly worsens
# fit, i.e. the variable contributes meaningful explanatory power.
# The direction is: full model (self) vs. reduced model (argument) — correct.
print("\nPartial F-Tests  (full model vs. model with one regressor dropped)")
print("High F / low p  →  variable carries significant unique explanatory power")
print("-" * 65)
print(f"{'Dropped variable':>22}  {'F-statistic':>12}  {'p-value':>10}  {'Significant?':>12}")
print("-" * 65)

for i, var in enumerate(REGRESSORS):
    # Keep intercept (column 0) plus all regressor columns except the one dropped
    keep_cols  = [0] + [j for j in range(1, len(REGRESSORS) + 1) if j != i + 1]
    X_reduced  = X_train_c[:, keep_cols]
    res_red    = sm.OLS(y_train, X_reduced).fit()

    # results is the full model; res_red is the restricted model (one variable removed)
    # compare_f_test tests: does adding the dropped variable back in help significantly?
    F_stat, p_val, _ = results.compare_f_test(res_red)
    sig = "Yes" if p_val < 0.05 else "No"
    print(f"{var:>22}  {F_stat:>12.4f}  {p_val:>10.6f}  {sig:>12}")

# ── FIX 3 — Test-set RMSE in both units ───────────────────────────────────────
# When --log is active the raw RMSE printed previously was in log(TP%+1) units,
# which are not directly interpretable.  We now back-transform predictions via
# expm1 (the inverse of log1p) and also report RMSE in raw TP% units.
y_pred_test      = results.predict(X_test_c)
test_rmse_log    = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTest-set RMSE ({y_label}): {test_rmse_log:.4f}")

if TRANSFORM != "none":
    y_pred_test_raw = inverse_fn(y_pred_test)
    test_rmse_raw   = np.sqrt(mean_squared_error(y_test_raw, y_pred_test_raw))
    print(f"Test-set RMSE (TP%  — back-transformed): {test_rmse_raw:.4f}")
    print(
        "  Interpretation: on the held-out 20%, predictions are off by "
        f"≈ {test_rmse_raw:.2f} percentage points on average (raw TP% scale)."
    )

# ── Decision Tree — parameter range extraction ────────────────────────────────
# The tree partitions the parameter space into rectangular regions.  We find the
# leaf with the highest predicted mean TP% and trace the path back to the root
# to read off the exact [min, max] bounds for each parameter.  These bounds feed
# directly into a reduced grid search.

print("\n" + "=" * 70)
print("DECISION TREE  (parameter range extraction for reduced grid search)")
print("=" * 70)

# Cross-validate max_depth on training data (depths 1–8)
print("Selecting best max_depth via 5-fold CV on training data...")
best_depth, best_cv_rmse = 1, np.inf
for depth in range(1, 9):
    dt_cv  = DecisionTreeRegressor(max_depth=depth, random_state=42)
    scores = cross_val_score(
        dt_cv, X_train, y_train, cv=5,
        scoring="neg_mean_squared_error",
    )
    rmse = np.sqrt(-scores.mean())
    if rmse < best_cv_rmse:
        best_cv_rmse = rmse
        best_depth   = depth

print(f"  Best max_depth : {best_depth}  (CV RMSE = {best_cv_rmse:.4f} on {y_label})")

# Fit final tree with the selected depth
dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt.fit(X_train, y_train)

# Evaluate on the held-out test set
y_pred_dt = dt.predict(X_test)
dt_rmse   = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print(f"  Test-set RMSE  : {dt_rmse:.4f}  ({y_label})")
if TRANSFORM != "none":
    dt_rmse_raw = np.sqrt(mean_squared_error(y_test_raw, inverse_fn(y_pred_dt)))
    print(f"  Test-set RMSE  : {dt_rmse_raw:.4f}  (TP% — back-transformed)")

# Print the tree structure in text form
print(f"\nTree structure:\n")
print(export_text(dt, feature_names=REGRESSORS))

# ── Find the leaf with the highest predicted mean TP% ────────────────────────
tree_   = dt.tree_
is_leaf = tree_.children_left == -1

leaf_values = {
    node: tree_.value[node][0][0]
    for node in range(tree_.node_count)
    if is_leaf[node]
}
best_leaf = max(leaf_values, key=leaf_values.get)
best_val  = leaf_values[best_leaf]
n_obs     = int(tree_.n_node_samples[best_leaf])

# ── Trace path from root to best leaf, collecting split conditions ────────────
def path_to_node(tree_, target):
    """Return list of (node, direction) pairs along the path root → target."""
    path = []

    def recurse(node):
        if node == target:
            return True
        for child, direction in [
            (tree_.children_left[node],  "left"),
            (tree_.children_right[node], "right"),
        ]:
            if child >= 0:
                path.append((node, direction))
                if recurse(child):
                    return True
                path.pop()
        return False

    recurse(0)
    return path

# Initialise bounds from the full observed data range
lo       = {p: float(df[p].min()) for p in REGRESSORS}
hi       = {p: float(df[p].max()) for p in REGRESSORS}
split_on = set()

for node, direction in path_to_node(tree_, best_leaf):
    feat   = REGRESSORS[tree_.feature[node]]
    thresh = float(tree_.threshold[node])
    split_on.add(feat)
    if direction == "left":    # condition: feature ≤ threshold
        hi[feat] = min(hi[feat], thresh)
    else:                      # condition: feature > threshold
        lo[feat] = max(lo[feat], thresh)

# ── Print optimal ranges ──────────────────────────────────────────────────────
display_val = inverse_fn(best_val) if TRANSFORM != "none" else best_val
val_label   = f"TP% (back-transformed from {y_label})" if TRANSFORM != "none" else "TP%"

print(f"Optimal leaf : node {best_leaf}  |  "
      f"Mean {val_label} = {display_val:.2f}  |  "
      f"n = {n_obs} training observations\n")
print(f"  {'Parameter':<22}  {'Min':>8}  {'Max':>8}  {'Note'}")
print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*30}")
for p in REGRESSORS:
    note = "" if p in split_on else "(not split on — full range retained)"
    print(f"  {p:<22}  {lo[p]:>8.3f}  {hi[p]:>8.3f}  {note}")
print(
    f"\n  Use these bounds to define the reduced grid search.\n"
    f"  Verify n = {n_obs}: a small leaf gives less reliable bounds."
)

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
