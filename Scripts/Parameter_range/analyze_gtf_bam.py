"""
================================================================================
Operon Parameter Analysis — C. difficile (NC_009089.1)
================================================================================
Input    : (1) GTF annotation file   — defines CDS coordinates
           (2) BAM alignment file    — RNA-seq reads mapped to the genome
Output   : Console summary of statistics + distribution plots (PNG)

BACKGROUND
----------
This script evaluates four parameters used to identify operons (groups of
co-transcribed genes) in bacterial RNA-seq data. An operon is characterised by:
  - Genes being actively transcribed (sufficient coverage)
  - Neighbouring genes having similar expression levels (low fold difference)
  - The intergenic region between them also being transcribed (read-through)

All four parameters are expression-based: they require knowing how many RNA-seq
reads mapped to each genomic position, which comes from the BAM file. The GTF
file only provides the coordinates of each CDS.

PARAMETERS COMPUTED
-------------------
1. CDS Coverage
   The average number of RNA-seq reads per base pair across each CDS.
   Formula:  sum of read depth at every position in the CDS
             ─────────────────────────────────────────────
                         length of CDS (bp)
   Interpretation: A higher value means the gene is more actively transcribed.
   A minimum threshold is set so that only sufficiently expressed genes are
   considered as operon candidates.

2. IGR Coverage
   The same average depth calculation, but applied to the intergenic region
   (IGR) — the gap in base pairs between the end of one CDS and the start of
   the next CDS.
   Formula:  sum of read depth at every position in the IGR
             ──────────────────────────────────────────────
                         length of IGR (bp)
   Interpretation: If reads "read through" from one gene into the next, the IGR
   will have non-zero coverage. A minimum IGR coverage threshold filters for
   regions that show this read-through signal.

   IGR DEFINITION USED HERE:
   IGRs are defined strictly as the gap between two consecutive CDS features.
   Non-coding RNA features (tRNA, rRNA) that may lie in those gaps are ignored
   for boundary purposes, consistent with focusing on protein-coding operons.

3. Fold Difference Between Adjacent CDSs
   The ratio of expression levels between two neighbouring genes. Genes in the
   same operon are expected to be transcribed from the same promoter and
   therefore expressed at similar levels.
   Formula:  max(coverage_gene_A, coverage_gene_B)
             ───────────────────────────────────────
             min(coverage_gene_A, coverage_gene_B)
   Interpretation: A ratio of 1.0 means identical expression. A maximum
   threshold (e.g. <=10) ensures only similarly-expressed gene pairs are
   considered as potential operon members.

4. Fold Difference Between IGR and Flanking CDSs
   The ratio between the expression level in the CENTRAL PORTION of an IGR
   and the expression level of each of its flanking genes. A low ratio means
   the IGR is expressed at a similar level to its neighbours, consistent with
   continuous transcription across the region.

   CENTRAL PORTION DEFINITION:
   The middle 50% of the IGR is used (positions from the 25th to 75th
   percentile of the IGR length). This avoids edge artefacts where coverage
   near gene boundaries may be elevated due to reads overlapping from the
   flanking CDS.

   Formula (computed separately for left and right flanking gene):
     ratio_left  = max(IGR_mid_cov, left_CDS_cov)  / min(IGR_mid_cov, left_CDS_cov)
     ratio_right = max(IGR_mid_cov, right_CDS_cov) / min(IGR_mid_cov, right_CDS_cov)
     recorded value = max(ratio_left, ratio_right)   <- most conservative choice

   Interpretation: A low fold difference suggests the IGR is being transcribed
   at a level consistent with its flanking genes, supporting co-transcription.

USAGE
-----
    python analyze_gtf_bam.py <annotations.gtf> <alignments.bam>

    The BAM file must be sorted and indexed (.bai file in the same directory).
    If not already sorted/indexed, the script will do this automatically.

DEPENDENCIES
------------
    pip install pysam matplotlib numpy
================================================================================
"""

import sys
import os
import statistics
import numpy as np
import pysam
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: GTF PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_gtf(path):
    """
    Reads the GTF annotation file and extracts the genomic coordinates of
    every CDS (protein-coding sequence).

    GTF format (tab-separated columns):
        col 0: chromosome / sequence name
        col 1: source (e.g. RefSeq)
        col 2: feature type (transcript, CDS, exon, ...)
        col 3: start position (1-based, inclusive)
        col 4: end position   (1-based, inclusive)
        col 6: strand (+ or -)

    We keep ONLY rows where feature == 'CDS' and discard 'transcript' rows,
    which are coordinate-duplicates of their child CDS and would falsely
    inflate the gene count.

    Returns
    -------
    cds_list : list of (start, end) tuples, sorted by genomic start position.
               Coordinates are 1-based and inclusive, matching the GTF standard.
    """
    cds_list = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue                      # skip comment / header lines
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue                      # skip malformed lines
            feature = parts[2]
            if feature != "CDS":
                continue                      # ignore transcript, exon, etc.
            start = int(parts[3])
            end   = int(parts[4])
            cds_list.append((start, end))

    # Sort by genomic start so consecutive pairs are truly neighbours
    cds_list.sort(key=lambda x: x[0])
    print(f"  Loaded {len(cds_list):,} CDS features from GTF.")
    return cds_list


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BAM LOADING AND INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def load_bam(path):
    """
    Opens the BAM alignment file and ensures it is sorted and indexed.

    A BAM index (.bai file) is required for random-access queries (i.e.
    fetching coverage at specific genomic coordinates). If the index does
    not exist, this function creates it automatically using pysam.

    pysam.AlignmentFile requires the BAM to be coordinate-sorted before
    indexing. If the BAM is unsorted, it is sorted first.

    Returns
    -------
    bam   : open pysam.AlignmentFile object (caller must close it)
    chrom : the chromosome/reference name string to use in coverage queries
    """
    bai_path = path + ".bai"

    # Check if index exists; if not, sort and index automatically
    if not os.path.exists(bai_path):
        print("  BAM index not found — sorting and indexing (may take a moment)...")
        sorted_path = path.replace(".bam", "_sorted.bam")
        pysam.sort("-o", sorted_path, path)   # produces coordinate-sorted BAM
        pysam.index(sorted_path)              # produces sorted_path + ".bai"
        path = sorted_path
        print("  Sorting and indexing complete.")

    bam = pysam.AlignmentFile(path, "rb")

    # Extract the reference sequence name from the BAM header.
    # The header SQ (sequence dictionary) lists all reference sequences.
    # We search for a name containing 'NC_009089' to match C. difficile 630.
    chrom = None
    for sq in bam.header.to_dict().get("SQ", []):
        if "NC_009089" in sq["SN"]:
            chrom = sq["SN"]
            break
    if chrom is None:
        # Fall back to the first reference if name-matching fails
        chrom = bam.header.to_dict()["SQ"][0]["SN"]

    print(f"  BAM reference sequence : '{chrom}'")
    print(f"  Total mapped reads     : {bam.mapped:,}")
    return bam, chrom


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COVERAGE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def mean_coverage(bam, chrom, start_1based, end_1based):
    """
    Computes the mean per-base read depth over a genomic interval.

    pysam.count_coverage() returns four arrays (one per nucleotide A, C, G, T),
    each of length equal to the queried region. Each value is the number of
    reads covering that exact position. We sum across all four nucleotides at
    each position to get total depth, then take the mean across all positions.

    COORDINATE CONVERSION:
    GTF uses 1-based inclusive coordinates.  [start=1, end=5] covers 5 bases.
    pysam uses 0-based half-open coordinates. [start=0, end=5] covers 5 bases.
    Therefore: pysam_start = gtf_start - 1
               pysam_end   = gtf_end        (no change needed for end)

    Parameters
    ----------
    bam          : open pysam.AlignmentFile
    chrom        : reference sequence name (must match BAM header exactly)
    start_1based : 1-based inclusive start (as in GTF)
    end_1based   : 1-based inclusive end   (as in GTF)

    Returns
    -------
    float : mean read depth across the region; 0.0 if region is empty
    """
    length = end_1based - start_1based + 1
    if length <= 0:
        return 0.0

    try:
        # count_coverage returns (A_counts, C_counts, G_counts, T_counts)
        # each an array of length = queried region length.
        # quality_threshold=0 includes all reads regardless of base quality.
        counts = bam.count_coverage(
            chrom,
            start_1based - 1,   # convert GTF 1-based start -> pysam 0-based
            end_1based,         # pysam half-open end == GTF inclusive end
            quality_threshold=0
        )
    except ValueError:
        return 0.0

    # Sum the four nucleotide arrays to get total depth per position,
    # then average across all positions in the region.
    total_depth = np.sum(counts, axis=0)   # shape: (length,)
    return float(np.mean(total_depth))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PARAMETER COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_parameters(cds_list, bam, chrom):
    """
    Iterates over all CDSs and their consecutive IGRs to compute the four
    operon-detection parameters.

    WORKFLOW
    --------
    Step A — For every CDS:
        Compute mean coverage -> stored in cds_coverages[i]

    Step B — For every consecutive CDS pair (i, i+1):
        1. Derive the IGR boundaries from CDS coordinates
        2. Compute full IGR coverage           (Parameter 2)
        3. Compute adjacent-CDS fold diff      (Parameter 3)
        4. Compute middle-50% IGR fold diff    (Parameter 4)

    MIDDLE 50% OF IGR
    -----------------
    For an IGR spanning positions [igr_start, igr_end] with length L:
        quarter   = L / 4
        mid_start = igr_start + round(L/4)   <- trim 25% from left
        mid_end   = igr_end   - round(L/4)   <- trim 25% from right
    This leaves the central 50% of the IGR, avoiding boundary artefacts
    from reads that originate in flanking CDSs and bleed into the IGR edges.
    If the IGR is too short to trim (mid_end < mid_start), the full IGR is used.

    FOLD DIFFERENCES
    ----------------
    Always computed as max/min so the result is >= 1.0, making it easier to
    interpret as "how many times higher is the larger value".
    Pairs where either value is zero are excluded to avoid division-by-zero
    and to discard uninformative results from unexpressed regions.

    Returns
    -------
    cds_coverages  : list of mean coverage per CDS (Parameter 1)
    igr_coverages  : list of mean coverage per IGR (Parameter 2)
    fold_adj       : list of fold diffs between adjacent CDS pairs (Parameter 3)
    fold_igr_flank : list of fold diffs between mid-IGR and flanking CDSs (Parameter 4)
    """
    n = len(cds_list)

    # ── Step A: CDS coverages (Parameter 1) ───────────────────────────────
    print("  Computing CDS coverages (Parameter 1)...")
    cds_coverages = []
    for i, (s, e) in enumerate(cds_list):
        cov = mean_coverage(bam, chrom, s, e)
        cds_coverages.append(cov)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,} / {n:,} CDSs processed...")
    print(f"    All {n:,} CDS coverages computed.")

    # ── Step B: IGR-based parameters (Parameters 2, 3, 4) ─────────────────
    igr_coverages  = []
    fold_adj       = []
    fold_igr_flank = []

    print("  Computing IGR coverages and fold differences (Parameters 2-4)...")

    for i in range(n - 1):

        # Coordinates of the left CDS (gene i) and right CDS (gene i+1)
        s_left,  e_left  = cds_list[i]
        s_right, e_right = cds_list[i + 1]

        # Retrieve precomputed mean coverages for both flanking CDSs
        cov_left  = cds_coverages[i]
        cov_right = cds_coverages[i + 1]

        # ── Parameter 3: fold difference between adjacent CDSs ────────
        # Only computed when both genes have non-zero expression, to avoid
        # artefactual infinite or undefined ratios from silent genes.
        if cov_left > 0 and cov_right > 0:
            fold = max(cov_left, cov_right) / min(cov_left, cov_right)
            fold_adj.append(fold)

        # ── IGR boundaries ─────────────────────────────────────────────
        # The IGR is the gap between the end of gene i and the start of
        # gene i+1. Using 1-based coordinates throughout:
        #   igr_start = e_left  + 1  (first base after end of left CDS)
        #   igr_end   = s_right - 1  (last base before start of right CDS)
        igr_start = e_left  + 1
        igr_end   = s_right - 1
        igr_len   = igr_end - igr_start + 1

        # Skip pairs where CDSs overlap (no IGR exists)
        if igr_len <= 0:
            continue

        # ── Parameter 2: full IGR mean coverage ───────────────────────
        igr_cov = mean_coverage(bam, chrom, igr_start, igr_end)
        igr_coverages.append(igr_cov)

        # ── Parameter 4: middle 50% of IGR vs flanking CDSs ──────────
        # Trim outer 25% from each end of the IGR
        quarter   = igr_len / 4
        mid_start = igr_start + int(round(quarter))
        mid_end   = igr_end   - int(round(quarter))

        # If IGR is too short to trim meaningfully, fall back to full IGR
        if mid_end < mid_start:
            mid_start = igr_start
            mid_end   = igr_end

        mid_cov = mean_coverage(bam, chrom, mid_start, mid_end)

        # Compute fold ratio vs each flanking CDS; record the larger (more
        # conservative) of the two. Exclude if any value is zero.
        if mid_cov > 0 and cov_left > 0 and cov_right > 0:
            ratio_left  = max(mid_cov, cov_left)  / min(mid_cov, cov_left)
            ratio_right = max(mid_cov, cov_right) / min(mid_cov, cov_right)
            fold_igr_flank.append(max(ratio_left, ratio_right))

    print(f"    Done. "
          f"{len(igr_coverages):,} IGRs | "
          f"{len(fold_adj):,} adj. CDS pairs | "
          f"{len(fold_igr_flank):,} IGR/flank pairs.")

    return cds_coverages, igr_coverages, fold_adj, fold_igr_flank


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summarise(name, data):
    """
    Prints range, mean, median, and std dev for a parameter to the console.
    Zero-coverage values are excluded from summary statistics because they
    represent unexpressed or unannotated regions, not meaningful signal.
    """
    nonzero = [x for x in data if x > 0]
    if not nonzero:
        print(f"\n  {name}: no non-zero values found.")
        return

    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
    print(f"  N (non-zero) : {len(nonzero):,}  (of {len(data):,} total)")
    print(f"  Range        : {min(nonzero):.4g}  ->  {max(nonzero):.4g}")
    print(f"  Mean         : {statistics.mean(nonzero):.4g}")
    print(f"  Median       : {statistics.median(nonzero):.4g}")
    print(f"  Std dev      : {statistics.stdev(nonzero):.4g}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DISTRIBUTION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# One distinct colour per parameter for visual consistency across plots
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]


def add_stats_box(ax, data):
    """
    Overlays a small annotation box in the top-right corner of a plot axes,
    showing key summary statistics for quick visual reference.
    """
    txt = (f"n = {len(data):,}\n"
           f"min = {min(data):.3g}\n"
           f"max = {max(data):.3g}\n"
           f"mean = {statistics.mean(data):.3g}\n"
           f"median = {statistics.median(data):.3g}")
    ax.text(0.97, 0.97, txt,
            transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#cccccc", alpha=0.85))


def plot_distribution(ax, data, title, xlabel, color, log_x=False):
    """
    Draws a histogram of the parameter distribution on the given axes,
    with vertical dashed/dotted lines marking the mean and median.

    LOG SCALE RATIONALE:
    Coverage and fold-difference data are typically right-skewed — most
    values cluster near the low end, with a long tail of high values.
    A log-scale x-axis spreads out the crowded low end so the full shape
    of the distribution is clearly visible, rather than being compressed
    against the left axis.

    Parameters
    ----------
    ax     : matplotlib Axes object to draw on
    data   : list of numeric values (zeros filtered out inside this function)
    title  : string for the plot title
    xlabel : string for the x-axis label
    color  : hex colour string for histogram bars
    log_x  : if True, apply log scale to x-axis and use log-spaced bins
    """
    # Remove zeros before plotting — they are not informative signal
    data_arr = np.array([x for x in data if x > 0])
    if len(data_arr) == 0:
        ax.set_title(title + "\n(no data)", fontsize=10)
        return

    if log_x:
        # Log-spaced bins give equal visual width to each order of magnitude
        bins = np.logspace(np.log10(data_arr.min()),
                           np.log10(data_arr.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60

    ax.hist(data_arr, bins=bins, color=color, alpha=0.82,
            edgecolor="white", linewidth=0.4)

    # Vertical reference lines for mean (dashed) and median (dotted)
    mean_val   = statistics.mean(data_arr)
    median_val = statistics.median(data_arr)
    ax.axvline(mean_val,   color="#333333", linestyle="--",
               linewidth=1.4, label=f"Mean = {mean_val:.3g}")
    ax.axvline(median_val, color="#888888", linestyle=":",
               linewidth=1.4, label=f"Median = {median_val:.3g}")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    # Format y-axis tick labels with thousands separator for readability
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    add_stats_box(ax, data_arr)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("Usage: python analyze_gtf_bam.py <annotations.gtf> <alignments.bam>")
        sys.exit(1)

    gtf_path = sys.argv[1]
    bam_path = sys.argv[2]

    # ── Load inputs ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  LOADING INPUT FILES")
    print(f"{'='*65}")
    print(f"\nParsing GTF : {gtf_path}")
    cds_list = parse_gtf(gtf_path)

    print(f"\nOpening BAM : {bam_path}")
    bam, chrom = load_bam(bam_path)

    # ── Compute parameters ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  COMPUTING PARAMETERS")
    print(f"{'='*65}\n")
    cds_cov, igr_cov, fold_adj, fold_igr = compute_parameters(
        cds_list, bam, chrom)
    bam.close()

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY  —  C. difficile NC_009089.1")
    print(f"{'='*65}")
    summarise("1. CDS Coverage  (mean reads/bp per gene)",          cds_cov)
    summarise("2. IGR Coverage  (mean reads/bp per IGR)",           igr_cov)
    summarise("3. Fold Diff — Adjacent CDSs",                       fold_adj)
    summarise("4. Fold Diff — Middle-50% IGR vs Flanking CDSs",     fold_igr)

    # ── Plots ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#F7F7F7")
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    datasets = [
        (cds_cov,  "1. CDS Coverage Distribution",
                   "Mean Coverage (reads / bp)",             True),
        (igr_cov,  "2. IGR Coverage Distribution",
                   "Mean Coverage (reads / bp)",             True),
        (fold_adj, "3. Fold Difference — Adjacent CDSs",
                   "Fold Difference (log scale)",            True),
        (fold_igr, "4. Fold Difference — Middle-50% IGR vs Flanking CDSs",
                   "Fold Difference (log scale)",            True),
    ]

    for idx, (data, title, xlabel, log_x) in enumerate(datasets):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#FAFAFA")
        plot_distribution(ax, data, title, xlabel, PALETTE[idx], log_x)

    fig.suptitle(
        "C. difficile (NC_009089.1) — Expression-Based Operon Parameters\n"
        "Sample: Wildtype Control 1",
        fontsize=13, fontweight="bold", y=0.99)

    out_path = "gtf_bam_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved -> {out_path}\n")
    plt.show()


if __name__ == "__main__":
    main()
