# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis: optimizing the COSMO operon prediction tool for *Clostridioides difficile*. The work involves tuning four parameters of the COSMO algorithm and validating prediction quality against reference operons.

The COSMO package lives in `cosmo_code_&_ supplementary materials/COSMO/` (tracked in `.gitignore` as an external folder — not a git submodule).

## Running COSMO

```bash
# Install dependencies
pip install pysam>=0.15.0

# Run operon detection
cd "cosmo_code_&_ supplementary materials/COSMO"
python operon/user_input.py [-D GDEPTH] [-d IDEPTH] [-F GFACTOR] [-f IFACTOR] <ref_length> <bam_file> <gtf_file>
```

**Parameters:**
| Flag | Meaning | Default |
|------|---------|---------|
| `-D` | Gene coverage depth threshold | 2 |
| `-d` | IGR coverage depth threshold | 1 |
| `-F` | Max fold difference between adjacent genes | 5.0 |
| `-f` | Max fold difference IGR-to-gene | 5.0 |

Output is written to `output/detected-operons.csv`.

## Running Tests

```bash
cd "cosmo_code_&_ supplementary materials/COSMO"
pytest test/
# or a single test file:
pytest test/test_operon.py
```

Note: `test_detect.py` tests are commented out — they require BAM/BAI test files that are not committed.

## Running the Parameter Analysis Script

```bash
# Analyzes GTF+BAM distributions to understand parameter space
python analyze_gtf_bam.py
# or the version with more options:
python scripts/analyze_gtf_bam.py
```

Outputs coverage distribution statistics and PNG plots used to inform parameter selection.

## Architecture

### COSMO Core (`cosmo_code_&_ supplementary materials/COSMO/operon/`)

The detection pipeline is:
1. **`gtf_process.py`** — parses GTF (plain or gzip) → returns gene coordinate dicts; `average_coverage()` queries a BAM file via pysam to compute mean read depth over any genomic interval
2. **`gene.py`** — `Gene` dataclass: gene_id, coordinates, CDS coverage, IGR coverage
3. **`detect.py`** — `Detect` class: iterates over genes, computes coverage for each gene and flanking IGR, applies the four thresholds to decide which genes belong to the same operon
4. **`operon.py`** — `Operon` class: holds a list of `Gene` objects; generates CSV output rows
5. **`user_input.py`** — argparse CLI that wires the above together and writes output CSV

### Parameter Optimization Pipeline

```
COSMO runs (varying -D -d -F -f)
        │
        ├── Algorithm_parameter_testing/Python_scripts/
        │     calc_mean_operons.py               # fold-change between conditions
        │     calculate_total_correct_operons.py # precision vs reference list
        │     ave_operon_genes_and_IGRs.py       # coverage summary stats
        │
        ├── Algorithm_parameter_testing/python_scripts_for_prediction_calls/
        │     dict_no_strains_pred_operon_TP_FP_FN.py   # TP/FP/FN per family
        │     predicted_operons_counts_per_lineage.py    # counts per lineage
        │
        └── Algorithm_parameter_testing/R_script/
              MLR_and_decision_tree.r   # MLR + decision tree to predict optimal params
```

The R script uses `predicted-operons-*.csv` outputs from the prediction scripts as input, fits an MLR and a decision tree on PPV as the response variable, and reports RMSE/MAE.

### Reference Data

- `Algorithm_parameter_testing/GTF_&_other_coordinate_files/Combined_operon_list.txt` — the ground-truth operon list used for TP/FP/FN evaluation
- `Algorithm_parameter_testing/python_scripts_for_prediction_calls/50_combined_operon_list.txt` — subset used in prediction scripts

### Important Conventions

- The prediction scripts in `python_scripts_for_prediction_calls/` are **hardcoded** for specific input file naming patterns — follow the naming instructions in that folder's README before running them.
- Large data files (`.bam`, `.gtf`, `.fastq`, `.csv` outputs) are excluded from git; only code, reference coordinate files, and visualizations are tracked.

---

## 1. What is COSMO?

COSMO (Condition-Specific Mapping of Operons) is a bioinformatics tool that predicts
operons in prokaryotic genomes using RNA sequencing (RNA-seq) data.

An **operon** is a group of neighbouring genes that are co-transcribed as a single
messenger RNA molecule. Identifying operons is important for understanding how an
organism regulates its genes and responds to environmental stress.

COSMO takes two input files:
- A **BAM file** — contains the RNA sequencing reads aligned to the reference genome,
  representing a snapshot of gene expression
- A **GTF file** — contains the structural annotation of the genome (gene coordinates,
  strand information, etc.)

COSMO uses both files together with four user-defined parameters to predict which
neighbouring genes are being co-transcribed as operons.

In this project, COSMO is being applied to **Clostridioides difficile (C. difficile)**,
reference genome **NC_009089.1**, using a single **Wildtype Control 1** sample.
The GTF file used is a manually corrected version: `NC_009089.1_fixed.gtf`.

---

## 2. The Four COSMO Parameters

COSMO requires four user-defined cutoff values. All four were confirmed as statistically
significant predictors of operon prediction accuracy in the original COSMO paper
(Calvert-Joshua et al.).

### Parameter A — Minimum CDS Coverage
- **What it is:** The minimum expression level (read depth) a gene must have to be
  considered active and potentially part of an operon
- **Unit:** reads per base pair (reads/bp)
- **C. difficile recommended range:** 1 – 20
- **Reasoning:** The median CDS coverage in C. difficile is 11.7, with most genes
  expressed below 50x. Setting this too high risks excluding lowly expressed but
  genuine operon genes.

### Parameter B — Minimum IGR Coverage
- **What it is:** The minimum expression level the intergenic region (IGR) between
  two genes must have to suggest those genes may be co-transcribed
- **Unit:** reads per base pair (reads/bp)
- **C. difficile recommended range:** 1 – 25
- **Reasoning:** The median IGR coverage is 16.5, slightly higher than CDS coverage,
  suggesting active intergenic transcription in C. difficile (possibly related to
  non-coding RNA production). Range is set slightly wider than CDS to reflect this.

### Parameter C — Maximum Fold Difference Between Adjacent CDSs
- **What it is:** The maximum allowable ratio of expression levels between two
  neighbouring genes. A small ratio means the genes are expressed at similar levels,
  consistent with co-transcription.
- **Unit:** fold difference (dimensionless ratio)
- **C. difficile recommended range:** 2 – 20
- **Reasoning:** The median fold difference between adjacent CDSs is just 2.03,
  indicating very tight co-expression between neighbouring genes in C. difficile.
  This range is tighter than Mtb (where 5x–7x was recommended), reflecting
  C. difficile's more coordinated transcriptional regulation.

### Parameter D — Maximum Fold Difference Between IGR and Flanking CDSs
- **What it is:** The maximum allowable ratio between the expression level of the
  central portion of an IGR and its neighbouring genes. In real operons, the IGR
  expression tracks closely with flanking gene expression.
- **Unit:** fold difference (dimensionless ratio)
- **C. difficile recommended range:** 2 – 15
- **Reasoning:** The median fold difference is 3.22, with the bulk of values below 10.
  COSMO uses only the central 50% of the IGR for this calculation to avoid ramp-up
  and tail-off effects at gene boundaries.

### Default Cutoffs (from original COSMO paper, as starting reference)
| Parameter | Default Value |
|---|---|
| Min CDS coverage | 1x |
| Min IGR coverage | 2x |
| Max FD adjacent CDSs | 5x |
| Max FD IGR vs CDSs | 10x |

---

## 3. COSMO Output Format

COSMO produces a **CSV file** for each run. Each row represents one predicted operon
and contains the following fields:

| Field | Description |
|---|---|
| Operon name | Identifier for the predicted operon |
| Genomic coordinates | Start and end position on the genome |
| Operon length | Number of base pairs spanning the operon |
| Average coverage | Mean expression level across the entire operon |
| CDS names | Names of each gene within the operon |
| CDS coverages | Expression level of each individual gene |
| IGR names | Names of each intergenic region within the operon |
| IGR coverages | Expression level of each individual IGR |

---

## 4. Performance Metrics

COSMO predictions are evaluated by comparing against the **28 experimentally
validated operons (EVOs)** for C. difficile.

> **Note:** The original COSMO paper used 51 EVOs for Mtb. With only 28 EVOs
> available for C. difficile, performance metrics may show higher variance and
> should be interpreted with this limitation in mind.

### Primary Metrics Recorded Per Run
| Metric | Description |
|---|---|
| TP% | Percentage of EVOs correctly predicted as full-length operons |
| FP count | Number of operons predicted longer than in literature |
| FN count | Number of operons predicted shorter than in literature |

### Full Validation Metrics (used for final parameter evaluation)
| Metric | Formula |
|---|---|
| Sensitivity / Recall | TP / (TP + FN) × 100 |
| Precision / PPV | TP / (TP + FP) × 100 |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) |
| NPV | TN / (TN + FN) × 100 |
| Specificity | TN / (TN + FP) × 100 |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) × 100 |

### Benchmark (from original COSMO paper on Mtb, for reference)
| Algorithm | TP% | Sensitivity | PPV | F1 Score |
|---|---|---|---|---|
| COSMO | 61% | 78% | 74% | 76% |
| REMap | 49% | 68% | 64% | 66% |
| Rockhopper | 41% | 51% | 68% | 58% |

---

## 5. Optimisation Goal

The goal of this project is to find the optimal combination of COSMO's four
parameters for **C. difficile** as efficiently as possible, without running an
exhaustive grid search like the original paper.

### Optimisation Strategy
1. **Latin Hypercube Sampling (LHS)** — generate an initial set of ~100–200
   parameter combinations spread evenly across the search space
2. **Run COSMO** for each combination and record TP%, FP count, FN count
3. **Fit a Gaussian Process model** to the results (fallback: polynomial regression,
   or MLR as simplest option)
4. **Identify initial optimum** from the fitted model
5. **Bayesian Optimisation** — iteratively refine by intelligently selecting new
   parameter combinations to test (fallback: manual refinement around optimum)
6. **Validate** final parameters against the 28 EVOs using full performance metrics

### Parameter Search Ranges for LHS
| Parameter | Min | Max |
|---|---|---|
| A — Min CDS Coverage | 1 | 20 |
| B — Min IGR Coverage | 1 | 25 |
| C — Max FD Adjacent CDSs | 2 | 20 |
| D — Max FD IGR vs CDSs | 2 | 15 |

---

## 6. Environment and Tooling

| Item | Detail |
|---|---|
| Language | Python |
| Reference genome | NC_009089.1 (C. difficile) |
| GTF file | NC_009089.1_fixed.gtf (manually corrected) |
| Sample used | Wildtype Control 1 only |
| Version control | Git / GitHub |
| Running environment | Local (may move to cluster later) |
| EVO count | 28 experimentally validated C. difficile operons |

---

## 7. Key Biological Notes for C. difficile

- C. difficile shows **tighter co-expression** between adjacent genes than Mtb
  (median FD = 2.03 vs ~5–7x in Mtb), suggesting more coordinated operon regulation
- IGR coverage is **higher than CDS coverage** on average (median 16.5 vs 11.7),
  possibly reflecting active non-coding RNA transcription, a known feature of
  C. difficile
- Unlike Mtb, **no RIF-resistant strains** are being analysed in this project —
  all predictions are for wildtype under control conditions
- IGR length exclusion (as used in many other operon predictors) is likely also
  **not appropriate for C. difficile**, consistent with COSMO's design philosophy
  of not relying on IGR distance as a feature
