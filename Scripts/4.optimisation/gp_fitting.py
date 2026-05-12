"""
Gaussian Process Regression Script — COSMO Parameter Optimisation
=================================================================
Fits a Gaussian Process Regressor (Matérn 5/2 kernel + noise term) to predict
a COSMO performance metric from the four COSMO parameters. Identifies the
predicted optimum across the parameter space and quantifies uncertainty.

Outputs to console:
  - Fitted kernel hyperparameters and log-marginal likelihood
  - Test-set R² and RMSE (80/20 split)
  - 5-fold cross-validation (mean R², std R², mean RMSE)
  - Predicted optimal parameter combination with ± uncertainty

Saves three plot files to OUTPUT_DIR:
  gp_actual_vs_predicted.png    — train and test fit quality
  gp_response_surfaces.png      — GP mean over all 6 pairwise parameter slices
  gp_uncertainty_surfaces.png   — GP std over the same slices

Usage
-----
  python3 scripts/4.optimisation/gp_fitting.py <input_csv> [--log] [--target COL] [--output-dir DIR]

Arguments
---------
  input_csv           Path to CSV (auto-detects , or ; separator).
                      Required columns: CDS_min, IGR_min, FD_CDS-CDS_min,
                      FD_IGR-CDS_min, and the target column.
  --log               Apply log(target + 1) transform to the response variable.
  --target COL        Column to model (default: TP%).
  --output-dir DIR    Directory for saved plots
                      (default: parameter_optimisation/analysis).

Examples
--------
  python3 scripts/4.optimisation/gp_fitting.py \\
      parameter_optimisation/raw_datasets/raw_dataset_regression.csv

  python3 scripts/4.optimisation/gp_fitting.py \\
      raw_dataset_regression.csv --log --target F1
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

# ── Parse command-line arguments ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Fit a GP model predicting a COSMO metric from the four parameters "
                "and identify the predicted optimum.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("input_csv", help="Path to the CSV containing LHS results")
parser.add_argument(
    "--log",
    action="store_true",
    default=False,
    help="Apply log(target + 1) transform to the response variable",
)
parser.add_argument(
    "--target",
    default="TP%",
    metavar="COL",
    help="Column to model (default: TP%%)",
)
parser.add_argument(
    "--output-dir",
    default="parameter_optimisation/analysis",
    metavar="DIR",
    help="Directory where plots are saved (default: parameter_optimisation/analysis)",
)
args = parser.parse_args()

INPUT_CSV       = args.input_csv
LOG_TRANSFORM_Y = args.log
TARGET          = args.target
OUTPUT_DIR      = args.output_dir

REGRESSORS  = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
N_FEATURES  = len(REGRESSORS)
N_GRID      = 30    # grid resolution for continuous parameters in plots and optimum search
N_RESTARTS  = 10    # kernel hyperparameter optimisation restarts (main model)

# CDS_min and IGR_min are coverage-depth thresholds — COSMO requires integers.
# FD parameters are fold-difference ratios and can be continuous.
INTEGER_PARAMS = {"CDS_min", "IGR_min"}

# Known biological search ranges (used for the optimum grid search)
PARAM_BOUNDS = {
    "CDS_min":         (1,  20),
    "IGR_min":         (1,  25),
    "FD_CDS-CDS_min":  (2,  20),
    "FD_IGR-CDS_min":  (2,  15),
}


def make_axis(param_name: str, n_grid: int) -> np.ndarray:
    """Return axis values for a parameter, using integers if required."""
    lo, hi = PARAM_BOUNDS[param_name]
    if param_name in INTEGER_PARAMS:
        return np.arange(lo, hi + 1, dtype=float)
    return np.linspace(lo, hi, n_grid)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and validate data ────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

missing = [c for c in REGRESSORS + [TARGET] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: column(s) not found in CSV: {missing}")

X     = df[REGRESSORS].to_numpy(dtype=float)
y_raw = df[TARGET].to_numpy(dtype=float)

# ── Optional log transform ────────────────────────────────────────────────────
if LOG_TRANSFORM_Y:
    y       = np.log1p(y_raw)
    y_label = f"log({TARGET} + 1)"
else:
    y       = y_raw
    y_label = TARGET

# ── 80/20 train-test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ── Scale features (fit scaler on training set only to prevent data leakage) ──
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_sc       = scaler.transform(X)   # full scaled dataset used for CV

# ── Define GP kernel ──────────────────────────────────────────────────────────
# Matérn 5/2: smooth but not infinitely differentiable — appropriate for
# real experimental data where the response is not perfectly smooth.
# Per-feature length scales let the GP learn which parameters are most influential.
# WhiteKernel accounts for measurement noise / run-to-run variance.
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
    * Matern(length_scale=np.ones(N_FEATURES), length_scale_bounds=(1e-2, 1e2), nu=2.5)
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
)

# ── Fit GP on training data ───────────────────────────────────────────────────
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=N_RESTARTS,
    normalize_y=True,   # centres y internally for numerical stability
    random_state=42,
)
print("Fitting GP model...")
gp.fit(X_train_sc, y_train)

# ── Console: fitted kernel hyperparameters ────────────────────────────────────
print("\n" + "=" * 70)
print("FITTED KERNEL HYPERPARAMETERS")
print("=" * 70)
print(f"  {gp.kernel_}")
print(f"\n  Log-marginal likelihood: {gp.log_marginal_likelihood_value_:.4f}")

# Fitted length scales (one per parameter) — larger = less sensitive to that param
k_params = gp.kernel_.get_params()
length_scales = None
for key, val in k_params.items():
    if "length_scale" in key and hasattr(val, "__len__") and len(val) == N_FEATURES:
        length_scales = val
        break
if length_scales is not None:
    print("\n  Per-parameter length scales (smaller = more influential):")
    for name, ls in zip(REGRESSORS, length_scales):
        print(f"    {name:<22}: {ls:.4f}")

# ── Test-set evaluation ───────────────────────────────────────────────────────
y_pred_test, y_std_test = gp.predict(X_test_sc, return_std=True)
test_r2   = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n" + "=" * 70)
print("TEST-SET PERFORMANCE  (80/20 split)")
print("=" * 70)
print(f"  R²  : {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.4f}  ({y_label})")

# ── 5-fold cross-validation on full dataset ───────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2, cv_rmse = [], []

print("\nRunning 5-fold cross-validation...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sc), 1):
    X_cv_tr,  X_cv_val  = X_sc[tr_idx],  X_sc[val_idx]
    y_cv_tr,  y_cv_val  = y[tr_idx],     y[val_idx]

    # Fewer restarts per fold to keep runtime reasonable
    cv_kernel = (
        ConstantKernel(1.0)
        * Matern(length_scale=np.ones(N_FEATURES), nu=2.5)
        + WhiteKernel(noise_level=0.1)
    )
    cv_gp = GaussianProcessRegressor(
        kernel=cv_kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42
    )
    cv_gp.fit(X_cv_tr, y_cv_tr)
    y_cv_pred = cv_gp.predict(X_cv_val)

    ss_res = np.sum((y_cv_val - y_cv_pred) ** 2)
    ss_tot = np.sum((y_cv_val - np.mean(y_cv_val)) ** 2)
    fold_r2   = 1.0 - ss_res / ss_tot
    fold_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
    cv_r2.append(fold_r2)
    cv_rmse.append(fold_rmse)
    print(f"  Fold {fold}: R²={fold_r2:.4f}, RMSE={fold_rmse:.4f}")

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION  (full dataset)")
print("=" * 70)
print(f"  Mean R²  : {np.mean(cv_r2):.4f}")
print(f"  Std  R²  : {np.std(cv_r2):.4f}")
print(f"  Mean RMSE: {np.mean(cv_rmse):.4f}  ({y_label})")

# ── Find optimum via constrained grid search ──────────────────────────────────
# CDS_min and IGR_min only take integer values — scipy continuous optimisers
# cannot enforce this, so we search a grid instead. The grid uses all valid
# integers for those two parameters and N_GRID points for the FD parameters.
axes_orig = [make_axis(name, N_GRID) for name in REGRESSORS]
n_grid_pts = int(np.prod([len(ax) for ax in axes_orig]))
print(f"\nSearching for optimum over {n_grid_pts:,} grid points "
      f"(integers for CDS_min and IGR_min)...")

grid_orig = np.array(np.meshgrid(*axes_orig, indexing="ij")).reshape(N_FEATURES, -1).T
grid_sc   = scaler.transform(grid_orig)

y_grid_mean, y_grid_std = gp.predict(grid_sc, return_std=True)

best_idx   = np.argmax(y_grid_mean)
best_x_sc  = grid_sc[best_idx]
opt_x_orig = grid_orig[best_idx]

opt_mean_sc = y_grid_mean[best_idx]
opt_std_sc  = y_grid_std[best_idx]

if LOG_TRANSFORM_Y:
    opt_mean = np.expm1(opt_mean_sc)
    opt_std  = np.exp(opt_mean_sc) * opt_std_sc
else:
    opt_mean = opt_mean_sc
    opt_std  = opt_std_sc

print("\n" + "=" * 70)
print("PREDICTED OPTIMAL PARAMETERS")
print("=" * 70)
for name, val in zip(REGRESSORS, opt_x_orig):
    if name in INTEGER_PARAMS:
        print(f"  {name:<22}: {int(round(val))}")
    else:
        print(f"  {name:<22}: {val:.4f}")
print(f"\n  Predicted {TARGET:<10}: {opt_mean:.4f}  ±  {opt_std:.4f}  (mean ± 1 std)")
print("=" * 70)

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
y_pred_train = gp.predict(X_train_sc)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_train, y_pred_train, alpha=0.65, edgecolors="k",
           linewidths=0.4, color="steelblue", label="Train")
ax.scatter(y_test, y_pred_test, alpha=0.65, edgecolors="k",
           linewidths=0.4, color="darkorange", label="Test")
lo = min(y.min(), y_pred_train.min(), y_pred_test.min())
hi = max(y.max(), y_pred_train.max(), y_pred_test.max())
ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
ax.set_xlabel(f"Actual {y_label}")
ax.set_ylabel(f"Predicted {y_label}")
ax.set_title(f"Actual vs Predicted — GP  (test R²={test_r2:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gp_actual_vs_predicted.png"), dpi=150)
plt.close()

# ── Plots 2 & 3: Response surfaces and uncertainty (all 6 pairwise slices) ────
# For each pair of parameters (i, j): vary those two over their training range,
# fix the other two at their optimal (scaled) values.

pairs = list(itertools.combinations(range(N_FEATURES), 2))   # 6 pairs

fig_mean, axes_mean = plt.subplots(2, 3, figsize=(16, 10))
fig_std,  axes_std  = plt.subplots(2, 3, figsize=(16, 10))

for ax_m, ax_s, (i, j) in zip(axes_mean.flat, axes_std.flat, pairs):
    # Build 2D grid; integer params use arange, continuous use linspace
    xi_orig = make_axis(REGRESSORS[i], N_GRID)
    xj_orig = make_axis(REGRESSORS[j], N_GRID)
    xi_sc   = (xi_orig - scaler.mean_[i]) / scaler.scale_[i]
    xj_sc   = (xj_orig - scaler.mean_[j]) / scaler.scale_[j]
    Xi_sc, Xj_sc = np.meshgrid(xi_sc, xj_sc)

    n_i, n_j = len(xi_orig), len(xj_orig)
    X_grid = np.tile(best_x_sc, (n_i * n_j, 1))
    X_grid[:, i] = Xi_sc.ravel()
    X_grid[:, j] = Xj_sc.ravel()

    Z_mean, Z_std = gp.predict(X_grid, return_std=True)
    Z_mean = Z_mean.reshape(n_j, n_i)
    Z_std  = Z_std.reshape(n_j, n_i)

    if LOG_TRANSFORM_Y:
        Z_mean = np.expm1(Z_mean)

    label_i = REGRESSORS[i]
    label_j = REGRESSORS[j]

    # Response surface — GP mean
    cf_m = ax_m.contourf(xi_orig, xj_orig, Z_mean, levels=20, cmap="viridis")
    fig_mean.colorbar(cf_m, ax=ax_m, label=TARGET)
    ax_m.scatter([opt_x_orig[i]], [opt_x_orig[j]],
                 color="red", marker="*", s=200, zorder=5, label="Optimum")
    ax_m.set_xlabel(label_i)
    ax_m.set_ylabel(label_j)
    ax_m.set_title(f"{label_i} × {label_j}")
    ax_m.legend(loc="upper right", fontsize=8)

    # Uncertainty surface — GP std
    cf_s = ax_s.contourf(xi_orig, xj_orig, Z_std, levels=20, cmap="plasma")
    fig_std.colorbar(cf_s, ax=ax_s, label=f"std ({y_label})")
    ax_s.scatter([opt_x_orig[i]], [opt_x_orig[j]],
                 color="cyan", marker="*", s=200, zorder=5, label="Optimum")
    ax_s.set_xlabel(label_i)
    ax_s.set_ylabel(label_j)
    ax_s.set_title(f"{label_i} × {label_j}")
    ax_s.legend(loc="upper right", fontsize=8)

fig_mean.suptitle(f"GP Response Surface — Predicted {TARGET}", fontsize=14)
fig_std.suptitle("GP Prediction Uncertainty (std)", fontsize=14)

fig_mean.tight_layout()
fig_mean.savefig(
    os.path.join(OUTPUT_DIR, "gp_response_surfaces.png"), dpi=150, bbox_inches="tight"
)
plt.close(fig_mean)

fig_std.tight_layout()
fig_std.savefig(
    os.path.join(OUTPUT_DIR, "gp_uncertainty_surfaces.png"), dpi=150, bbox_inches="tight"
)
plt.close(fig_std)

print(f"\nPlots saved to: {OUTPUT_DIR}/")
print("  gp_actual_vs_predicted.png")
print("  gp_response_surfaces.png")
print("  gp_uncertainty_surfaces.png")
