#!/usr/bin/env python3
"""
Divergence pipeline.

Reads the scraper output (munich_restaurants.csv), classifies each restaurant
into a cuisine from its Google Places `types`, builds a star-rating distribution
per cuisine, computes pairwise Jensen-Shannon divergence between those
distributions, and writes a heatmap PNG.

Usage:
    python divergence_pipeline.py [input_csv] [output_png]
Defaults:
    input_csv  = munich_restaurants.csv
    output_png = munich_jsd_heatmap.png
"""

import sys
import csv
import math

import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Minimum reviews to include. The scraper already filters, but enforcing it here
# keeps the pipeline correct even if fed an unfiltered CSV.
MIN_REVIEWS = 100

# Rating bins (half-star bands from 3.5 to 5.0). Edges are left-inclusive.
BIN_EDGES = [3.5, 3.8, 4.1, 4.4, 4.7, 5.01]
BIN_LABELS = ["3.5-3.7", "3.8-4.0", "4.1-4.3", "4.4-4.6", "4.7-5.0"]

# Cuisine classification from Google place types, in priority order. The first
# matching specific type wins; broad fallbacks (asian, mediterranean) sit last
# so a vietnamese_restaurant becomes "Asian" only if nothing more specific hit.
CUISINE_RULES = [
    ("Italian",  ["italian_restaurant", "pizza_restaurant"]),
    ("Bavarian", ["bavarian_restaurant", "german_restaurant"]),
    ("Turkish",  ["turkish_restaurant"]),
    ("Indian",   ["indian_restaurant", "north_indian_restaurant"]),
    ("Greek",    ["greek_restaurant"]),
    ("Japanese", ["japanese_restaurant", "sushi_restaurant"]),
    ("Thai",     ["thai_restaurant"]),
    ("Chinese",  ["chinese_restaurant"]),
    ("Vietnamese", ["vietnamese_restaurant"]),
    ("Asian",    ["asian_restaurant", "asian_fusion_restaurant"]),
    ("Mediterranean", ["mediterranean_restaurant"]),
]


def classify(types_field):
    """Map a pipe-delimited Google types string to a cuisine label, or None."""
    types = set(types_field.split("|")) if types_field else set()
    for label, keys in CUISINE_RULES:
        if any(k in types for k in keys):
            return label
    return None


def load(csv_path):
    """Read the CSV into (cuisine, rating) rows, filtering by reviews and validity."""
    rows = []
    skipped_unclassified = 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                count = int(float(r["user_rating_count"] or 0))
                rating = float(r["rating"]) if r["rating"] not in ("", None) else None
            except (ValueError, KeyError):
                continue
            if rating is None or count <= MIN_REVIEWS:
                continue
            cuisine = classify(r.get("types", ""))
            if cuisine is None:
                skipped_unclassified += 1
                continue
            rows.append((cuisine, rating))
    return rows, skipped_unclassified


def build_distributions(rows, min_group=3):
    """One smoothed rating distribution per cuisine that has >= min_group places."""
    cuisines = sorted(set(c for c, _ in rows))
    dists, kept = {}, []
    for c in cuisines:
        ratings = [r for cc, r in rows if cc == c]
        if len(ratings) < min_group:
            continue
        hist, _ = np.histogram(ratings, bins=BIN_EDGES)
        p = hist.astype(float) + 0.5      # Laplace smoothing keeps JS finite
        p /= p.sum()
        dists[c] = p
        kept.append(c)
    return kept, dists


def jsd_matrix(cuisines, dists):
    """Pairwise Jensen-Shannon divergence (base 2), squaring scipy's distance."""
    n = len(cuisines)
    M = np.zeros((n, n))
    for i, a in enumerate(cuisines):
        for j, b in enumerate(cuisines):
            M[i, j] = jensenshannon(dists[a], dists[b], base=2) ** 2
    return M


def plot(M, cuisines, out_png, n_places):
    n = len(cuisines)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 * n + 2.5), max(5.5, 0.9 * n + 2)))
    im = ax.imshow(M, cmap="magma_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cuisines, rotation=45, ha="right")
    ax.set_yticklabels(cuisines)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > M.max() * 0.55 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jensen-Shannon divergence (base 2)", fontsize=10)
    ax.set_title("Pairwise JS divergence of star-rating distributions\n"
                 f"Munich restaurants with >{MIN_REVIEWS} reviews, by cuisine "
                 f"(n={n_places})", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "munich_jsd_heatmap.png"

    rows, skipped = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}. Did the scrape run and produce cuisines?")
        sys.exit(1)

    cuisines, dists = build_distributions(rows)

    print(f"Loaded {len(rows)} classified restaurants from {in_csv} "
          f"({skipped} unclassified, dropped)")
    print(f"Cuisines with enough places: {', '.join(cuisines)}\n")

    print("Per-cuisine rating distributions (rows sum to 1):")
    print(f"{'cuisine':<13} " + "  ".join(f"{b:>8}" for b in BIN_LABELS))
    for c in cuisines:
        print(f"{c:<13} " + "  ".join(f"{x:8.3f}" for x in dists[c]))

    M = jsd_matrix(cuisines, dists)
    print("\nPairwise Jensen-Shannon divergence matrix:")
    print(f"{'':<13} " + "  ".join(f"{c[:6]:>7}" for c in cuisines))
    for i, c in enumerate(cuisines):
        print(f"{c:<13} " + "  ".join(f"{M[i,j]:7.4f}" for j in range(len(cuisines))))

    pairs = sorted((M[i, j], cuisines[i], cuisines[j])
                   for i in range(len(cuisines)) for j in range(i + 1, len(cuisines)))
    if pairs:
        print(f"\nMost similar pair:   {pairs[0][1]} vs {pairs[0][2]}  ({pairs[0][0]:.4f})")
        print(f"Most divergent pair: {pairs[-1][1]} vs {pairs[-1][2]}  ({pairs[-1][0]:.4f})")

    plot(M, cuisines, out_png, len(rows))


if __name__ == "__main__":
    main()
