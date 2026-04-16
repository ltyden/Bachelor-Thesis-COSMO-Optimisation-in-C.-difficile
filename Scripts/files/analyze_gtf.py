"""
GTF Parameter Analysis for C. difficile (NC_009089.1)
=====================================================
Computes range, mean, median, and distribution plots for:
  1. CDS lengths (Minimum CDS Coverage)
  2. IGR lengths (Minimum IGR Coverage)
  3. Fold difference between adjacent CDSs
  4. Fold difference between an IGR and its flanking CDSs
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import statistics


# ── 1. Parse GTF ─────────────────────────────────────────────────────────────

def parse_gtf(path):
    cds_list = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            feature = parts[2]
            start   = int(parts[3])
            end     = int(parts[4])
            if feature == "CDS":
                cds_list.append((start, end, end - start + 1))
    return sorted(cds_list, key=lambda x: x[0])   # sort by genomic start

# ── 2. Compute parameters ────────────────────────────────────────────────────

def compute_parameters(cds_list):
    # --- Parameter 1: CDS lengths ---
    cds_lengths = [length for _, _, length in cds_list]

    # --- Parameter 2: IGR lengths (positive gaps between consecutive CDSs) ---
    igr_lengths = []
    for i in range(len(cds_list) - 1):
        gap = cds_list[i + 1][0] - cds_list[i][1] - 1
        if gap > 0:
            igr_lengths.append(gap)

    # --- Parameter 3: fold difference between adjacent CDSs ---
    fold_adj = []
    for i in range(len(cds_list) - 1):
        l1 = cds_list[i][2]
        l2 = cds_list[i + 1][2]
        fold_adj.append(max(l1, l2) / min(l1, l2))

    # --- Parameter 4: max fold difference between each IGR and its flanking CDSs ---
    fold_igr = []
    for i in range(len(cds_list) - 1):
        gap = cds_list[i + 1][0] - cds_list[i][1] - 1
        if gap > 0:
            l_left  = cds_list[i][2]
            l_right = cds_list[i + 1][2]
            fold_left  = max(gap, l_left)  / min(gap, l_left)
            fold_right = max(gap, l_right) / min(gap, l_right)
            fold_igr.append(max(fold_left, fold_right))

    return cds_lengths, igr_lengths, fold_adj, fold_igr

# ── 3. Summary statistics ────────────────────────────────────────────────────

def summarise(name, data):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  N          : {len(data):,}")
    print(f"  Min        : {min(data):.4g}")
    print(f"  Max        : {max(data):.4g}")
    print(f"  Range      : {min(data):.4g}  →  {max(data):.4g}")
    print(f"  Mean       : {statistics.mean(data):.4g}")
    print(f"  Median     : {statistics.median(data):.4g}")
    print(f"  Std dev    : {statistics.stdev(data):.4g}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

def add_stats_box(ax, data):
    """Overlay a small stats annotation box."""
    txt = (f"n={len(data):,}\n"
           f"min={min(data):.3g}\n"
           f"max={max(data):.3g}\n"
           f"mean={statistics.mean(data):.3g}\n"
           f"median={statistics.median(data):.3g}")
    ax.text(0.97, 0.97, txt,
            transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#cccccc", alpha=0.85))

def plot_distribution(ax, data, title, xlabel, color, log_x=False):
    data_arr = np.array(data)

    if log_x:
        data_arr = data_arr[data_arr > 0]
        bins = np.logspace(np.log10(data_arr.min()),
                           np.log10(data_arr.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60

    ax.hist(data_arr, bins=bins, color=color, alpha=0.82,
            edgecolor="white", linewidth=0.4)

    # Mean & median lines
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
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    add_stats_box(ax, data_arr)

# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    filepath = input("Enter GTF file path (or press Enter to use default): ").strip()
    gtf_path = sys.argv[1] if len(sys.argv) > 1 else (filepath if filepath else "/Users/linustyden/Bachelor-Thesis-COSMO-Optimisation-in-C.-difficile/cosmo_code_&_ supplementary materials/cosmo_Input/NC_009089.1.gtf")

    print(f"\nParsing: {gtf_path}")
    cds_list = parse_gtf(gtf_path)
    print(f"  → {len(cds_list):,} CDS features loaded.")


    cds_len, igr_len, fold_adj, fold_igr = compute_parameters(cds_list)

    # ── Console summary ──
    print("\n" + "═"*55)
    print("  PARAMETER SUMMARY  —  C. difficile NC_009089.1")
    print("═"*55)
    summarise("1. CDS Lengths (bp)",                             cds_len)
    summarise("2. IGR Lengths (bp)",                             igr_len)
    summarise("3. Fold Diff — Adjacent CDSs",                   fold_adj)
    summarise("4. Fold Diff — IGR vs Flanking CDSs",            fold_igr)
    print()

    # ── Figure ──
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#F7F7F7")
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    datasets = [
        (cds_len,  "1. CDS Length Distribution",
                   "CDS Length (bp)",                   False),
        (igr_len,  "2. IGR Length Distribution",
                   "IGR Length (bp)",                   True),
        (fold_adj, "3. Fold Difference — Adjacent CDSs",
                   "Fold Difference",                   True),
        (fold_igr, "4. Fold Difference — IGR vs Flanking CDSs",
                   "Fold Difference",                   True),
    ]

    for idx, (data, title, xlabel, log_x) in enumerate(datasets):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#FAFAFA")
        plot_distribution(ax, data, title, xlabel, PALETTE[idx], log_x)

    fig.suptitle("C. difficile (NC_009089.1) — GTF Parameter Distributions",
                 fontsize=13, fontweight="bold", y=0.98)

    out_path = "gtf_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved → {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
