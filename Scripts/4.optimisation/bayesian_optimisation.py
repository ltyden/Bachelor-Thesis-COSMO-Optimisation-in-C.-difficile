"""
Bayesian Optimisation loop for COSMO parameter tuning.

Seeds a Gaussian Process surrogate from an existing evaluation dataset,
then iteratively selects new integer parameter combinations via Expected
Improvement (EI), runs COSMO live, evaluates against EVOs, and refits
the GP. All four parameters are treated as integers.

The script is resume-safe: on restart it reads any existing bo_results.csv
in the output directory and folds those observations back into the GP,
so no completed iteration is re-run.

Usage:
    # Full search space, optimising TP%, lenient scoring (default)
    python3 scripts/4.optimisation/bayesian_optimisation.py \
        --seed-data parameter_optimisation/raw_datasets/merged_dataset.csv \
        --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \
        --gtf /path/to/NC_009089.1_fixed.gtf \
        --evo-reference /path/to/evo_reference.csv \
        --genome-name "gi|126697566|ref|NC_009089.1|" \
        --genome-size 4290252 \
        --n-iter 30 \
        --output-dir parameter_optimisation/analysis/bo_run_1

    # Narrowed ranges around the targeted-grid optimum, optimising F1, strict scoring
    python3 scripts/4.optimisation/bayesian_optimisation.py \
        --seed-data parameter_optimisation/raw_datasets/merged_dataset.csv \
        --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \
        --gtf /path/to/NC_009089.1_fixed.gtf \
        --evo-reference /path/to/evo_reference.csv \
        --genome-name "gi|126697566|ref|NC_009089.1|" \
        --genome-size 4290252 \
        --target F1 \
        --strict \
        --n-iter 30 \
        --cds-range 1 5 --igr-range 1 5 --fd-cds-range 6 12 --fd-igr-range 2 4 \
        --output-dir parameter_optimisation/analysis/bo_run_2

    # Resume an interrupted run (same --output-dir and same arguments as original)
    python3 scripts/4.optimisation/bayesian_optimisation.py \
        --seed-data parameter_optimisation/raw_datasets/merged_dataset.csv \
        --bam /path/to/wt1.bam /path/to/wt2.bam /path/to/wt3.bam \
        --gtf /path/to/NC_009089.1_fixed.gtf \
        --evo-reference /path/to/evo_reference.csv \
        --genome-name "gi|126697566|ref|NC_009089.1|" \
        --genome-size 4290252 \
        --n-iter 30 \
        --output-dir parameter_optimisation/analysis/bo_run_1

Arguments:
    --seed-data       CSV (auto-detects ; or , delimiter) with prior evaluation
                      results. Required columns: CDS_min, IGR_min,
                      FD_CDS-CDS_min, FD_IGR-CDS_min, and the target column.
    --bam             One or more BAM files. Predictions are unioned across all
                      BAM files before evaluation (same logic as evaluate_cosmo.py).
    --gtf             GTF annotation file.
    --evo-reference   CSV listing experimentally validated operons (evo_reference.csv).
    --genome-name     Genome identifier as it appears in the GTF.
    --genome-size     Genome length in base pairs.
    --target          Metric to maximise: TP%, F1, Precision, or Recall
                      (default: TP%).
    --strict          Use strict TP scoring: superset matches count as FP instead
                      of TP (default: lenient).
    --n-iter          Number of BO iterations (default: 30).
    --xi              EI exploration parameter — higher values favour exploration
                      over exploitation (default: 0.01).
    --output-dir      Directory for COSMO outputs and BO result log
                      (default: parameter_optimisation/analysis/bo_results).

Parameter range overrides (optional — omit any to use the full biological range):
    --cds-range     MIN MAX   CDS_min search range        (default: 1 20)
    --igr-range     MIN MAX   IGR_min search range        (default: 1 25)
    --fd-cds-range  MIN MAX   FD_CDS-CDS_min search range (default: 2 20)
    --fd-igr-range  MIN MAX   FD_IGR-CDS_min search range (default: 2 15)
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
COSMO_DIR  = REPO_ROOT / "cosmo_code_&_ supplementary materials" / "COSMO"
OUTPUT_SRC = COSMO_DIR / "output"

REGRESSORS = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]
PARAM_BOUNDS = {
    "CDS_min":        (1,  20),
    "IGR_min":        (1,  25),
    "FD_CDS-CDS_min": (2,  20),
    "FD_IGR-CDS_min": (2,  15),
}
N_FEATURES = len(REGRESSORS)
N_RESTARTS = 10

# Import evaluation helpers from the existing pipeline.
# evaluate_cosmo.py lives two directories up in scripts/2.data_generation/.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "2.data_generation"))
from evaluate_cosmo import (       # noqa: E402
    classify_evos,
    compute_metrics,
    load_cosmo_predictions,
    load_evos,
)


# ── Candidate grid ────────────────────────────────────────────────────────────

def build_integer_grid(bounds: dict) -> np.ndarray:
    """Return every integer parameter combination within bounds."""
    axes = [np.arange(bounds[p][0], bounds[p][1] + 1) for p in REGRESSORS]
    n_pts = int(np.prod([len(a) for a in axes]))
    print(f"Search space: {' × '.join(str(len(a)) for a in axes)} = {n_pts:,} integer combinations")
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(N_FEATURES, -1).T


# ── GP ────────────────────────────────────────────────────────────────────────

def fit_gp(X_sc: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(N_FEATURES),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS,
        normalize_y=True,
        random_state=42,
    )
    gp.fit(X_sc, y)
    return gp


# ── Acquisition function ──────────────────────────────────────────────────────

def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float
) -> np.ndarray:
    """Expected Improvement for maximisation."""
    improvement = mu - f_best - xi
    with np.errstate(divide="ignore"):
        Z  = improvement / np.where(sigma > 1e-9, sigma, 1e-9)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


# ── COSMO runner ──────────────────────────────────────────────────────────────

def run_cosmo_single(
    combo_id: int,
    cds: int, igr: int, fd_cds: int, fd_igr: int,
    bam_files: list[Path],
    gtf: Path,
    genome_name: str,
    genome_size: int,
    cosmo_out_dir: Path,
) -> list[Path]:
    """Run COSMO for one parameter combination across all BAM files.

    Returns output paths (one per BAM) on success, empty list on any failure.
    """
    output_paths = []
    for bam_idx, bam in enumerate(bam_files, start=1):
        output_name = f"sample_{combo_id}_bam{bam_idx}.csv"
        cmd = [
            sys.executable, "-m", "operon.user_input",
            "-D", str(cds),
            "-d", str(igr),
            "-F", str(fd_cds),
            "-f", str(fd_igr),
            "-o", output_name,
            genome_name,
            str(genome_size),
            str(bam),
            str(gtf),
        ]
        result = subprocess.run(cmd, cwd=COSMO_DIR, capture_output=True, text=True)
        src = OUTPUT_SRC / output_name
        dst = cosmo_out_dir / output_name

        if result.returncode != 0 or not src.exists():
            print(
                f"    COSMO failed (BAM {bam_idx}): "
                f"{result.stderr.strip()[:120]}",
                file=sys.stderr,
            )
            return []

        shutil.move(str(src), str(dst))
        output_paths.append(dst)

    return output_paths


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_outputs(output_paths: list[Path], evos: list[dict],
                     strict: bool = False) -> dict:
    """Union predictions across BAM files and compute all metrics."""
    merged: set[frozenset] = set()
    for p in output_paths:
        for pred in load_cosmo_predictions(p):
            merged.add(pred)
    results = classify_evos(evos, list(merged), strict=strict)
    return compute_metrics(results)


def metrics_to_row(metrics: dict) -> dict:
    """Convert compute_metrics output to the flat row format used in the log.

    Under lenient scoring FP is always 0 and Precision always 1.0 — present
    for CSV column compatibility but not meaningful. Under strict scoring both
    are meaningful.
    """
    return {
        "TP%":       round(metrics["Accuracy"] * 100, 1),
        "TP":        metrics["TP"],
        "FN":        metrics["FN"],
        "FP":        metrics["FP"],
        "Precision": round(metrics["Precision"], 3),
        "Recall":    round(metrics["Recall"], 3),
        "F1":        round(metrics["F1"], 3),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_delimiter(path: Path) -> str:
    with open(path, newline="") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","


def target_value(row: dict, target: str) -> float:
    """Extract the optimisation target from a metrics row."""
    if target not in row:
        raise KeyError(
            f"Target '{target}' not in metrics. "
            f"Valid targets: {list(row.keys())}"
        )
    return float(row[target])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimisation loop for COSMO parameter tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed-data", required=True,
                        help="CSV with prior evaluation results")
    parser.add_argument("--bam", required=True, nargs="+",
                        help="BAM file(s) to use for COSMO runs")
    parser.add_argument("--gtf", required=True,
                        help="GTF annotation file")
    parser.add_argument("--evo-reference", required=True,
                        help="EVO reference CSV")
    parser.add_argument("--genome-name", required=True,
                        help="Genome identifier as it appears in the GTF")
    parser.add_argument("--genome-size", type=int, required=True,
                        help="Genome size in base pairs")
    parser.add_argument("--target", default="TP%",
                        metavar="COL",
                        help="Metric to maximise: TP%%, F1, Precision, Recall "
                             "(default: TP%%)")
    parser.add_argument("--n-iter", type=int, default=30,
                        help="Number of BO iterations (default: 30)")
    parser.add_argument("--xi", type=float, default=0.01,
                        help="EI exploration parameter (default: 0.01)")
    parser.add_argument("--cds-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                        help="CDS_min search range (default: 1 20)")
    parser.add_argument("--igr-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                        help="IGR_min search range (default: 1 25)")
    parser.add_argument("--fd-cds-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                        help="FD_CDS-CDS_min search range (default: 2 20)")
    parser.add_argument("--fd-igr-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                        help="FD_IGR-CDS_min search range (default: 2 15)")
    parser.add_argument("--strict",
                        action="store_true",
                        default=False,
                        help="Use strict TP scoring: superset matches count as FP "
                             "instead of TP (default: lenient)")
    parser.add_argument("--output-dir",
                        default="parameter_optimisation/analysis/bo_results",
                        metavar="DIR",
                        help="Directory for outputs "
                             "(default: parameter_optimisation/analysis/bo_results)")
    args = parser.parse_args()

    seed_path  = Path(args.seed_data)
    bam_files  = [Path(b) for b in args.bam]
    gtf        = Path(args.gtf)
    evo_path   = Path(args.evo_reference)
    output_dir = Path(args.output_dir)

    # Resolve active search bounds — use user override if given, else full range
    range_overrides = {
        "CDS_min":        args.cds_range,
        "IGR_min":        args.igr_range,
        "FD_CDS-CDS_min": args.fd_cds_range,
        "FD_IGR-CDS_min": args.fd_igr_range,
    }
    active_bounds = {}
    for param, override in range_overrides.items():
        if override is not None:
            lo, hi = override
            if lo >= hi:
                sys.exit(f"Error: {param} range MIN must be < MAX (got {lo} {hi})")
            active_bounds[param] = (lo, hi)
        else:
            active_bounds[param] = PARAM_BOUNDS[param]

    for p, label in [(seed_path, "seed data"), (gtf, "GTF"), (evo_path, "EVO reference")]:
        if not p.exists():
            sys.exit(f"Error: {label} not found: {p}")
    for b in bam_files:
        if not b.exists():
            sys.exit(f"Error: BAM file not found: {b}")
    if not COSMO_DIR.exists():
        sys.exit(f"Error: COSMO directory not found: {COSMO_DIR}")

    cosmo_out_dir = output_dir / "cosmo_outputs"
    cosmo_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load seed data ────────────────────────────────────────────────────────
    delim = detect_delimiter(seed_path)
    df    = pd.read_csv(seed_path, sep=delim)
    required = REGRESSORS + [args.target]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Error: column(s) missing from seed data: {missing}")

    X_all = df[REGRESSORS].to_numpy(dtype=float)
    y_all = df[args.target].to_numpy(dtype=float)

    # ── Resume: fold in any prior BO iterations ───────────────────────────────
    results_path = output_dir / "bo_results.csv"
    start_iter   = 1
    if results_path.exists():
        bo_df = pd.read_csv(results_path, sep=";")
        if len(bo_df) > 0:
            bo_X = bo_df[REGRESSORS].to_numpy(dtype=float)
            bo_y = bo_df[args.target].to_numpy(dtype=float)
            X_all      = np.vstack([X_all, bo_X])
            y_all      = np.concatenate([y_all, bo_y])
            start_iter = len(bo_df) + 1
            print(f"Resuming: loaded {len(bo_df)} prior BO iteration(s) from {results_path}")
    else:
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "iteration", *REGRESSORS,
                "TP%", "TP", "FN", "FP", "Precision", "Recall", "F1",
                "EI_score", "bam_files",
            ])

    print(f"Seed observations : {len(X_all)}")
    print(f"Best {args.target:<10}: {np.nanmax(y_all):.1f}")
    print(f"Iterations planned: {args.n_iter}  (starting at {start_iter})")
    print(f"Scoring mode      : {'strict' if args.strict else 'lenient'}")
    print("\nParameter search ranges:")
    for param, (lo, hi) in active_bounds.items():
        flag = "  (full range)" if (lo, hi) == PARAM_BOUNDS[param] else "  (narrowed)"
        print(f"  {param:<22}: {lo} – {hi}{flag}")
    print()

    evos           = load_evos(evo_path)
    candidate_grid = build_integer_grid(active_bounds)
    print()

    # Set of all parameter tuples already evaluated (seed + prior BO)
    evaluated_set: set[tuple] = {
        tuple(row.astype(int).tolist()) for row in X_all
    }

    # ── BO loop ───────────────────────────────────────────────────────────────
    for iteration in range(start_iter, start_iter + args.n_iter):
        print(f"{'─' * 62}")
        print(f"Iteration {iteration}  |  best {args.target} so far: {np.nanmax(y_all):.1f}")

        # Fit GP on all valid (non-NaN) observations
        valid  = ~np.isnan(y_all)
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_all[valid])
        print("  Fitting GP...", end=" ", flush=True)
        gp = fit_gp(X_sc, y_all[valid])
        print("done")

        # EI over entire integer grid
        grid_sc   = scaler.transform(candidate_grid.astype(float))
        mu, sigma = gp.predict(grid_sc, return_std=True)
        ei_scores = expected_improvement(mu, sigma, np.nanmax(y_all), args.xi)

        # Zero out already-evaluated points
        for idx, pt in enumerate(candidate_grid):
            if tuple(pt.astype(int).tolist()) in evaluated_set:
                ei_scores[idx] = -1.0

        if (ei_scores <= 0).all():
            print("  No candidates with EI > 0 remain. Search complete.")
            break

        best_idx  = int(np.argmax(ei_scores))
        candidate = candidate_grid[best_idx].astype(int)
        cds, igr, fd_cds, fd_igr = candidate
        ei_val    = float(ei_scores[best_idx])

        evaluated_set.add(tuple(candidate.tolist()))

        print(
            f"  Candidate: CDS={cds}  IGR={igr}  "
            f"FD_CDS={fd_cds}  FD_IGR={fd_igr}  EI={ei_val:.5f}"
        )
        print(f"  Running COSMO ({len(bam_files)} BAM file(s))...", end=" ", flush=True)

        output_paths = run_cosmo_single(
            iteration, int(cds), int(igr), int(fd_cds), int(fd_igr),
            bam_files, gtf, args.genome_name, args.genome_size, cosmo_out_dir,
        )

        if not output_paths:
            print("FAILED — skipping")
            # Still add so we don't re-try this point (NaN in y_all)
            X_all = np.vstack([X_all, candidate.astype(float)])
            y_all = np.append(y_all, np.nan)
            continue

        raw_metrics = evaluate_outputs(output_paths, evos, strict=args.strict)
        row         = metrics_to_row(raw_metrics)
        observed    = target_value(row, args.target)

        print("done")
        print(
            f"  {args.target}={observed}  "
            f"TP={row['TP']}  FN={row['FN']}  FP={row['FP']}  "
            f"Precision={row['Precision']}  F1={row['F1']}"
        )

        X_all = np.vstack([X_all, candidate.astype(float)])
        y_all = np.append(y_all, observed)

        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                iteration,
                int(cds), int(igr), int(fd_cds), int(fd_igr),
                row["TP%"], row["TP"], row["FN"], row["FP"],
                row["Precision"], row["Recall"], row["F1"],
                round(ei_val, 6), len(bam_files),
            ])

    # ── Final summary ─────────────────────────────────────────────────────────
    valid      = ~np.isnan(y_all)
    best_idx   = int(np.argmax(np.where(valid, y_all, -np.inf)))
    best_x     = X_all[best_idx].astype(int)
    best_y     = y_all[best_idx]

    print(f"\n{'═' * 62}")
    print("BAYESIAN OPTIMISATION COMPLETE")
    print(f"{'═' * 62}")
    print(f"  Best {args.target}: {best_y:.1f}")
    for name, val in zip(REGRESSORS, best_x):
        print(f"  {name:<22}: {val}")
    print(f"\n  BO results : {results_path}")
    print(f"  COSMO CSVs : {cosmo_out_dir}")


if __name__ == "__main__":
    main()
