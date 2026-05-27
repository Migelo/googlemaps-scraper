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
from prettytable import PrettyTable

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
    """Map a pipe-delimited Google types string to a cuisine label, or None.

    Order-dependent: the first matching CUISINE_RULES entry wins. A place
    tagged both `pizza_restaurant` and `mediterranean_restaurant` becomes
    "Italian" because Italian is listed earlier. Broad fallbacks (Asian,
    Mediterranean) intentionally sit last.
    """
    types = set(types_field.split("|")) if types_field else set()
    for label, keys in CUISINE_RULES:
        if any(k in types for k in keys):
            return label
    return None


def load(csv_path):
    """Read the CSV into (cuisine, rating) rows, filtering by reviews, rating range, and classification.

    Ratings outside [BIN_EDGES[0], BIN_EDGES[-1]] are dropped here rather than
    silently zeroed by np.histogram later, so they don't inflate a cuisine's
    apparent group size (the min_group eligibility check) while contributing
    nothing to its histogram.
    """
    lo, hi = BIN_EDGES[0], BIN_EDGES[-1]
    rows = []
    skipped_unclassified = 0
    skipped_out_of_range = 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                count = int(float(r["user_rating_count"] or 0))
                rating = float(r["rating"]) if r["rating"] not in ("", None) else None
            except (ValueError, KeyError):
                continue
            if rating is None or count <= MIN_REVIEWS:
                continue
            if not (lo <= rating < hi):
                skipped_out_of_range += 1
                continue
            cuisine = classify(r.get("types", ""))
            if cuisine is None:
                skipped_unclassified += 1
                continue
            rows.append((cuisine, rating))
    return rows, skipped_unclassified, skipped_out_of_range


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
    """Pairwise Jensen-Shannon divergence (base 2), squaring scipy's distance.

    Exploits symmetry: computes the upper triangle once and mirrors.
    """
    n = len(cuisines)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(dists[cuisines[i]], dists[cuisines[j]], base=2) ** 2
            M[i, j] = M[j, i] = d
    return M


def bootstrap_jsd(rows, cuisines, n_boot=1000, rng=None):
    """Resample within-cuisine with replacement n_boot times, return 95% CIs.

    Returns (M_mean, M_lo, M_hi) where each is (k, k) — k = len(cuisines).
    Smoothed distributions are rebuilt each iteration; smoothing keeps JSD
    finite even when a tiny resample lands all-in-one-bin.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    by_cuisine = {c: np.array([r for cc, r in rows if cc == c]) for c in cuisines}
    k = len(cuisines)
    samples = np.zeros((n_boot, k, k))
    for b in range(n_boot):
        # Resample each cuisine's ratings, with replacement, to its original size.
        boot_dists = {}
        for c in cuisines:
            base = by_cuisine[c]
            draw = rng.choice(base, size=len(base), replace=True)
            hist, _ = np.histogram(draw, bins=BIN_EDGES)
            p = hist.astype(float) + 0.5
            p /= p.sum()
            boot_dists[c] = p
        samples[b] = jsd_matrix(cuisines, boot_dists)
    mean = samples.mean(axis=0)
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return mean, lo, hi


def plot(M, cuisines, out_png, n_places, M_lo=None, M_hi=None):
    """Heatmap of JSD (lower triangle only) with optional CI annotations.

    JSD is symmetric, so we render only the lower triangle (i >= j) — the
    upper triangle is masked and rendered transparent, eliminating the
    redundant copy. Diagonal entries are always 0 by definition; we keep
    them so the cuisine labels stay aligned.
    """
    n = len(cuisines)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * n + 2.5), max(6, n + 2)))

    # Mask the strict upper triangle (j > i) so it doesn't render.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    M_disp = np.ma.array(M, mask=mask)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="white", alpha=0.0)

    im = ax.imshow(M_disp, cmap=cmap)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cuisines, rotation=45, ha="right")
    ax.set_yticklabels(cuisines)
    # Hide the now-empty top/right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    vmax = float(M_disp.max())  # for picking text contrast on the visible cells
    for i in range(n):
        for j in range(i + 1):  # lower triangle including diagonal
            v = M[i, j]
            label = f"{v:.3f}"
            if M_lo is not None and i != j:
                half = (M_hi[i, j] - M_lo[i, j]) / 2
                label = f"{v:.3f}\n±{half:.3f}"
                # Mark cells whose CI lower bound is below 0.005 — pairs of
                # JSDs that small (with smoothing) are effectively zero, so
                # we can't reject "same distribution" at 95%.
                if M_lo[i, j] <= 0.005:
                    label += "\n(ns)"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if v > vmax * 0.55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jensen-Shannon divergence (base 2)", fontsize=10)
    subtitle = (
        f"with 95% bootstrap CIs (1000 resamples); (ns) = CI touches 0"
        if M_lo is not None else ""
    )
    ax.set_title("Pairwise JS divergence of star-rating distributions\n"
                 f"Munich restaurants with >{MIN_REVIEWS} reviews, by cuisine "
                 f"(n={n_places})\n{subtitle}", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "munich_jsd_heatmap.png"

    rows, skipped_unclassified, skipped_oor = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}. Did the scrape run and produce cuisines?")
        sys.exit(1)

    cuisines, dists = build_distributions(rows)

    print(f"Loaded {len(rows)} classified restaurants from {in_csv} "
          f"({skipped_unclassified} unclassified, "
          f"{skipped_oor} outside rating range [{BIN_EDGES[0]}, {BIN_EDGES[-1]}); dropped)")
    print(f"Cuisines with enough places: {', '.join(cuisines)}\n")

    print("Per-cuisine rating distributions (rows sum to 1):")
    print(f"{'cuisine':<13} " + "  ".join(f"{b:>8}" for b in BIN_LABELS))
    for c in cuisines:
        print(f"{c:<13} " + "  ".join(f"{x:8.3f}" for x in dists[c]))

    M = jsd_matrix(cuisines, dists)

    # Bootstrap 95% CIs on every pair. Cheap at this scale (~1.5 s for k=11).
    print("\nBootstrapping JSD (1000 resamples) ...")
    M_mean, M_lo, M_hi = bootstrap_jsd(rows, cuisines, n_boot=1000)

    print("\nPairwise Jensen-Shannon divergence matrix:")
    print(f"{'':<13} " + "  ".join(f"{c[:6]:>7}" for c in cuisines))
    for i, c in enumerate(cuisines):
        print(f"{c:<13} " + "  ".join(f"{M[i,j]:7.4f}" for j in range(len(cuisines))))

    pairs = sorted((M[i, j], cuisines[i], cuisines[j])
                   for i in range(len(cuisines)) for j in range(i + 1, len(cuisines)))
    if pairs:
        print(f"\nMost similar pair:   {pairs[0][1]} vs {pairs[0][2]}  ({pairs[0][0]:.4f})")
        print(f"Most divergent pair: {pairs[-1][1]} vs {pairs[-1][2]}  ({pairs[-1][0]:.4f})")

    # Surface pairs whose 95% CI touches zero — statistically indistinguishable.
    ns_pairs = [
        (cuisines[i], cuisines[j], M[i, j], M_lo[i, j], M_hi[i, j])
        for i in range(len(cuisines))
        for j in range(i + 1, len(cuisines))
        if M_lo[i, j] <= 0.005
    ]
    if ns_pairs:
        print(f"\nStatistically indistinguishable pairs at 95% (CI touches 0):")
        for a, b, m, lo, hi in ns_pairs:
            print(f"  {a:<13} ~ {b:<13}  JSD = {m:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    plot(M, cuisines, out_png, len(rows), M_lo=M_lo, M_hi=M_hi)
    print_summary_table(cuisines, M, M_lo, M_hi)


def print_summary_table(cuisines, M, M_lo, M_hi):
    """Lower-triangle JSD pairs sorted by divergence, mirroring what the PNG shows."""
    pairs = []
    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            half = (M_hi[i, j] - M_lo[i, j]) / 2
            pairs.append((cuisines[i], cuisines[j], M[i, j],
                          M_lo[i, j], M_hi[i, j], half, M_lo[i, j] <= 0.005))
    pairs.sort(key=lambda p: p[2], reverse=True)

    t = PrettyTable()
    t.field_names = ["Pair", "JSD", "± half-CI", "95% CI", "Note"]
    t.align["Pair"] = "l"
    t.align["JSD"] = "r"
    t.align["± half-CI"] = "r"
    t.align["95% CI"] = "c"
    t.align["Note"] = "l"
    for a, b, m, lo, hi, half, ns in pairs:
        t.add_row([
            f"{a} ~ {b}",
            f"{m:.3f}",
            f"±{half:.3f}",
            f"[{lo:.3f}, {hi:.3f}]",
            "(ns)" if ns else "",
        ])
    print(
        "\nJensen-Shannon divergence (base 2) measures how different two probability\n"
        "distributions are. It is symmetric (JSD(a,b) = JSD(b,a)) and bounded in [0, 1]:\n"
        "  0   = the two cuisines have identical rating distributions\n"
        "  ~0.05 and below — practically indistinguishable at this sample size (see (ns))\n"
        "  0.1-0.2 — clearly different shapes (e.g. one is right-skewed, the other flat)\n"
        "  1   = no overlap at all (would require disjoint rating bands; doesn't happen here)"
    )
    print(f"\nAll {len(pairs)} pairs by JSD (descending) — summary of the heatmap:")
    print(t)


if __name__ == "__main__":
    main()
