"""
Latin Hypercube Sampling for COSMO parameter combinations.

Usage:
    python lhs_sampling.py -n 100 \
        --cds-min 1 --cds-max 20 \
        --igr-min 1 --igr-max 25 \
        --fd-cds-min 2 --fd-cds-max 20 \
        --fd-igr-min 2 --fd-igr-max 15 \
        --output lhs_combinations.csv
"""

import argparse
import csv
import sys

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale


def generate_lhs(n, bounds):
    sampler = LatinHypercube(d=4, seed=42)
    unit_sample = sampler.random(n=n)
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    scaled = scale(unit_sample, lower, upper)
    return scaled


def main():
    parser = argparse.ArgumentParser(
        description="Generate Latin Hypercube Sampling combinations for COSMO parameters."
    )
    parser.add_argument("-n", "--num-samples", type=int, required=True,
                        help="Number of parameter combinations to generate")
    parser.add_argument("--cds-min", type=float, default=1.0,
                        help="Min CDS coverage lower bound (default: 1)")
    parser.add_argument("--cds-max", type=float, default=20.0,
                        help="Min CDS coverage upper bound (default: 20)")
    parser.add_argument("--igr-min", type=float, default=1.0,
                        help="Min IGR coverage lower bound (default: 1)")
    parser.add_argument("--igr-max", type=float, default=25.0,
                        help="Min IGR coverage upper bound (default: 25)")
    parser.add_argument("--fd-cds-min", type=float, default=2.0,
                        help="Max FD adjacent CDSs lower bound (default: 2)")
    parser.add_argument("--fd-cds-max", type=float, default=20.0,
                        help="Max FD adjacent CDSs upper bound (default: 20)")
    parser.add_argument("--fd-igr-min", type=float, default=2.0,
                        help="Max FD IGR vs CDSs lower bound (default: 2)")
    parser.add_argument("--fd-igr-max", type=float, default=15.0,
                        help="Max FD IGR vs CDSs upper bound (default: 15)")
    parser.add_argument("--output", type=str, default="lhs_combinations.csv",
                        help="Output CSV filename (default: lhs_combinations.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    bounds = [
        (args.cds_min, args.cds_max),
        (args.igr_min, args.igr_max),
        (args.fd_cds_min, args.fd_cds_max),
        (args.fd_igr_min, args.fd_igr_max),
    ]

    for name, (lo, hi) in zip(["CDS", "IGR", "FD-CDS", "FD-IGR"], bounds):
        if lo >= hi:
            print(f"Error: {name} min ({lo}) must be less than max ({hi})", file=sys.stderr)
            sys.exit(1)

    sampler = LatinHypercube(d=4, seed=args.seed)
    unit_sample = sampler.random(n=args.num_samples)
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    samples = scale(unit_sample, lower, upper)

    header = [
        "combination_number",
        "CDS_min",
        "IGR_min",
        "FD_CDS-CDS_min",
        "FD_IGR-CDS_min",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(samples, start=1):
            cds  = int(round(row[0]))
            igr  = int(round(row[1]))
            fd_cds = round(row[2] * 2) / 2
            fd_igr = round(row[3] * 2) / 2
            writer.writerow([i, cds, igr, fd_cds, fd_igr])

    print(f"Generated {args.num_samples} LHS combinations → {args.output}")
    print(f"Seed: {args.seed}")
    print(f"\nParameter ranges used:")
    print(f"  Min CDS coverage:       {args.cds_min} – {args.cds_max}")
    print(f"  Min IGR coverage:       {args.igr_min} – {args.igr_max}")
    print(f"  Max FD adjacent CDSs:   {args.fd_cds_min} – {args.fd_cds_max}")
    print(f"  Max FD IGR vs CDSs:     {args.fd_igr_min} – {args.fd_igr_max}")


if __name__ == "__main__":
    main()
