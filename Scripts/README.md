# COSMO Parameter Optimisation Pipeline

Scripts for optimising the four parameters of the [COSMO](https://github.com/SANBI-SA/COSMO) operon prediction tool against a set of experimentally validated operons (EVOs). Developed for *Clostridioides difficile* but adaptable to any organism with a GTF annotation, RNA-seq BAM files, and a validated operon reference.

## Overview

COSMO predicts operons from RNA-seq data using four thresholds:

| Parameter | Flag | Description |
|-----------|------|-------------|
| Min CDS coverage | `-D` | Minimum read depth for a gene to be considered expressed |
| Min IGR coverage | `-d` | Minimum read depth in the intergenic region between two genes |
| Max FD adjacent CDSs | `-F` | Maximum fold difference between two neighbouring genes |
| Max FD IGR vs CDSs | `-f` | Maximum fold difference between an IGR and its flanking genes |

This pipeline finds the combination of these four values that best recovers a set of known operons (EVOs).

## Pipeline overview

```
1. biological_data_analysis/   Analyse GTF + BAM to understand the data and
                                inform parameter search ranges
        ↓
2. data_generation/            Sample parameter combinations (LHS), run COSMO
                                for each, and evaluate predictions against EVOs
        ↓
3. preprocess/                 Visualise and preprocess the results dataset
        ↓
4. optimisation/               Fit regression models and run Bayesian optimisation
                                to find the optimal parameters
        ↓
5. validation/                 Validate the final parameter set against the full
                                EVO reference
```

---

## Requirements

```bash
pip install numpy scipy pandas scikit-learn statsmodels matplotlib seaborn pysam joblib
```

COSMO itself must be installed separately. Clone or download it and ensure the `COSMO/` directory is present in the repository root at:
```
cosmo_code_&_ supplementary materials/COSMO/
```

---

## EVO reference format

All evaluation scripts expect a CSV with two columns:

```
No,EVOs
1,gene-A; gene-B; gene-C
2,gene-D; gene-E
```

Rows where `EVOs` is `Not expressed` are automatically skipped.

---

## 1. Biological data analysis

### `1.biological_data_analyis/analyze_gtf_bam.py`

Analyses your GTF annotation and BAM alignment file to compute distributions of the four COSMO parameters across the genome. Use this to understand your data and choose appropriate search ranges before sampling.

**Outputs:** console summary + `gtf_bam_distributions.png` in the output directory.

```bash
python3 scripts/1.biological_data_analyis/analyze_gtf_bam.py \
    path/to/annotation.gtf \
    path/to/alignment.bam \
    --output-dir path/to/output_directory
```

---

## 2. Data generation

### `2.data_generation/lhs_sampling.py`

Generates a set of parameter combinations using Latin Hypercube Sampling (LHS), which ensures even coverage of the parameter space with far fewer runs than a full grid search.

```bash
python3 scripts/2.data_generation/lhs_sampling.py \
    -n 200 \
    --cds-min 1  --cds-max 20 \
    --igr-min 1  --igr-max 25 \
    --fd-cds-min 2 --fd-cds-max 20 \
    --fd-igr-min 2 --fd-igr-max 15 \
    --output path/to/lhs_combinations.csv
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-n` | Number of parameter combinations to generate | required |
| `--cds-min/max` | Search range for Min CDS coverage | 1 – 20 |
| `--igr-min/max` | Search range for Min IGR coverage | 1 – 25 |
| `--fd-cds-min/max` | Search range for Max FD adjacent CDSs | 2 – 20 |
| `--fd-igr-min/max` | Search range for Max FD IGR vs CDSs | 2 – 15 |
| `--seed` | Random seed for reproducibility | 42 |

---

### `2.data_generation/run_cosmo_lhs.py`

Runs COSMO for every parameter combination in a combinations CSV, across one or more BAM files. Resume-safe: already completed runs are detected and skipped.

```bash
python3 scripts/2.data_generation/run_cosmo_lhs.py \
    --lhs-file path/to/combinations.csv \
    --bam path/to/bam_file1 path/to/bam_file2 path/to/bam_file3 \
    --gtf path/to/annotation.gtf \
    --genome-name "gi|126697566|ref|NC_009089.1|" \
    --genome-size 4290252 \
    --output-dir path/to/cosmo_output_directory
```

| Argument | Description |
|----------|-------------|
| `--lhs-file` | Combinations CSV (from `lhs_sampling.py` or any CSV with a `combination_number` column) |
| `--bam` | One or more BAM files. Predictions are unioned across all files before evaluation |
| `--gtf` | GTF annotation file |
| `--genome-name` | Sequence name exactly as it appears in the BAM header |
| `--genome-size` | Genome length in base pairs |
| `--output-dir` | Where to write COSMO output CSVs |
| `--first-bam-index` | Set to 2 if the first BAM has already been run and you are resuming with the remaining ones |

---

### `2.data_generation/evaluate_cosmo.py`

Evaluates all COSMO output CSVs in a directory against the EVO reference. When multiple BAM files were used, predictions are automatically unioned before scoring.

```bash
python3 scripts/2.data_generation/evaluate_cosmo.py \
    --cosmo-dir path/to/cosmo_output_directory \
    --evo-reference path/to/evo_reference.csv \
    --output path/to/evaluation_results.csv
```

**Output columns:** `run_id`, `TP%`, `TP`, `FP`, `FN`, `total_predicted`, `Precision`, `Recall`, `F1`, `bam_files`

---

### `2.data_generation/merge_lhs_evaluation.py`

Joins the parameter combinations CSV with the evaluation results CSV on `combination_number` / `run_id`, producing a single dataset ready for regression.

```bash
python3 scripts/2.data_generation/merge_lhs_evaluation.py \
    path/to/lhs_combinations.csv \
    path/to/evaluation_results.csv \
    --output path/to/raw_dataset_regression.csv
```

---

## 3. Preprocessing

### `3.preprocess/plot_tp_distribution.py`

Plots the distribution of TP% scores across all evaluated parameter combinations. Useful for a quick sanity check before modelling. Output is saved automatically to `parameter_optimisation/analysis/`.

```bash
python3 scripts/3.preprocess/plot_tp_distribution.py \
    --input path/to/evaluation_results.csv
```

**Terminal output:** best, worst, mean, median, std dev, and counts above common TP% thresholds.

---

### `3.preprocess/preprocess_dataset.py (OPTIONAL)`

Full preprocessing pipeline for the regression dataset: deduplication, missing value handling, variance checking, outlier flagging, multicollinearity analysis (VIF + correlation matrix), train/test split (80/20), and feature scaling.

```bash
python3 scripts/3.preprocess/preprocess_dataset.py \
    path/to/raw_dataset_regression.csv \
    TP% \
    --output-dir path/to/output_directory \
    --scaling minmax
```

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Merged regression dataset | required |
| `outcome_variable` | Target metric: `TP%`, `F1`, `Precision`, `Recall`, `TP`, `FP`, or `FN` | required |
| `--output-dir` | Directory for cleaned CSVs, train/test splits, scaler, and plots | `parameter_optimisation/cleaned_datasets/` |
| `--scaling` | `minmax` (scale to [0,1]) or `standard` (mean=0, std=1) | `minmax` |
| `--no-plots` | Skip generating plots | — |
| `--no-splits` | Skip saving train/test split files | — |

---

## 4. Optimisation

### `4.optimisation/find_best_params.py`

Quickly reports which parameter combination(s) achieved the highest TP% in a results CSV.

```bash
python3 scripts/4.optimisation/find_best_params.py \
    --input path/to/raw_dataset_regression.csv
```

---

### `4.optimisation/mlr_fitting.py`

Fits a Multiple Linear Regression (OLS) model to predict a performance metric from the four COSMO parameters. Also fits a decision tree to extract reduced parameter ranges for a follow-up grid search.

```bash
python3 scripts/4.optimisation/mlr_fitting.py \
    path/to/raw_dataset_regression.csv \
    --transform sqrt \
    --output-dir path/to/output_directory
```

| Argument | Description | Default |
|----------|-------------|---------|
| `input_csv` | Regression dataset | required |
| `--transform` | Response transform: `none`, `log`, `sqrt`, `cbrt`, `logit` | `none` |
| `--output-dir` | Directory for diagnostic plots | `parameter_optimisation/analysis` |

**Output plots:** `residuals_vs_fitted.png`, `qq_plot.png`, `scale_location.png`, `actual_vs_predicted.png`

---

### `4.optimisation/bayesian_optimisation.py`

Iterative Bayesian optimisation loop. Seeds a Gaussian Process (Matérn 5/2 kernel) from an existing evaluation dataset, then repeatedly selects the next most promising parameter combination via Expected Improvement, runs COSMO, evaluates it, and refits the GP. All four parameters are treated as integers.

Resume-safe: re-running with the same `--output-dir` picks up from where it left off.

```bash
python3 scripts/4.optimisation/bayesian_optimisation.py \
    --seed-data path/to/raw_dataset_regression.csv \
    --bam path/to/bam_file1 path/to/bam_file2 path/to/bam_file3 \
    --gtf path/to/annotation.gtf \
    --evo-reference path/to/evo_reference.csv \
    --genome-name "gi|126697566|ref|NC_009089.1|" \
    --genome-size 4290252 \
    --n-iter 30 \
    --output-dir path/to/output_directory
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed-data` | Prior evaluation dataset to seed the GP | required |
| `--bam` | BAM file(s) | required |
| `--gtf` | GTF annotation file | required |
| `--evo-reference` | EVO reference CSV | required |
| `--genome-name` | Genome identifier in GTF/BAM | required |
| `--genome-size` | Genome size in base pairs | required |
| `--target` | Metric to maximise: `TP%`, `F1`, or `Recall` | `TP%` |
| `--n-iter` | Number of BO iterations | 30 |
| `--xi` | Exploration parameter for Expected Improvement | 0.01 |
| `--output-dir` | Directory for results log and COSMO outputs | `parameter_optimisation/analysis/bo_results` |
| `--cds-range MIN MAX` | Narrow the CDS search range | 1 – 20 |
| `--igr-range MIN MAX` | Narrow the IGR search range | 1 – 25 |
| `--fd-cds-range MIN MAX` | Narrow the FD-CDS search range | 2 – 20 |
| `--fd-igr-range MIN MAX` | Narrow the FD-IGR search range | 2 – 15 |

**Output:** `bo_results.csv` (one row per iteration) + individual COSMO output CSVs.

---

## 5. Validation

### `5.validation/validate_parameters.py`

Runs COSMO with a single parameter combination, unions predictions across all BAM files, and prints the full metric set against the EVO reference.

```bash
python3 scripts/5.validation/validate_parameters.py \
    --cds 1 --igr 2 --fd-cds 5 --fd-igr 10 \
    --bam path/to/bam_file1 path/to/bam_file2 path/to/bam_file3 \
    --gtf path/to/annotation.gtf \
    --genome-name "gi|126697566|ref|NC_009089.1|" \
    --genome-size 4290252 \
    --evo-reference path/to/evo_reference.csv
```

**Terminal output:** TP, FP, FN, TP%, Precision, Recall, F1, NPV, Specificity, Accuracy.

---

## Recommended workflow

```
1. analyze_gtf_bam.py          Understand your data, choose search ranges

2. lhs_sampling.py             Generate ~200 parameter combinations

3. run_cosmo_lhs.py            Run COSMO for all combinations

4. evaluate_cosmo.py           Score predictions against EVOs

5. merge_lhs_evaluation.py     Join parameters + scores into one dataset

6. plot_tp_distribution.py     Inspect the score distribution

7a. find_best_params.py        Quick check: what is the best so far?
7b. mlr_fitting.py             Fit MLR + decision tree to understand which
                                parameters matter and narrow the search space

8. bayesian_optimisation.py    Refine with 20–50 targeted COSMO runs

9. validate_parameters.py      Final validation of the optimal combination
```

---

## Adapting to a different organism

1. Replace the GTF and BAM files with your organism's annotation and RNA-seq data.
2. Build an EVO reference CSV listing known operons (two columns: `No`, `EVOs`; genes semicolon-separated).
3. Update `--genome-name` to match the sequence name in your BAM header (`samtools view -H your.bam | grep @SQ`).
4. Update `--genome-size` to your genome length in base pairs.
5. Adjust the parameter search ranges in `lhs_sampling.py` based on the distributions from `analyze_gtf_bam.py`.
