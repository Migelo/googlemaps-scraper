#!/usr/bin/env python3
"""
Price × cuisine contingency table with Bayesian-shrunk mean ratings.

Builds a 2D grid: cuisine (rows) × price_level (columns). Each cell shows:

    n        — number of restaurants in that bucket
    mean     — raw mean rating
    shrunk   — Bayesian-shrunk mean toward the global mean (prior k=8)

Cells are flagged when their shrunk rating deviates strongly (|z|>1) from a
naive additive model:

    expected_cell = cuisine_marginal + price_marginal - global_mean

Output: price_cuisine_grid.png

Usage:
    python price_cuisine_grid.py [input_csv] [output_png]
"""

import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from divergence_pipeline import classify, MIN_REVIEWS

PRIOR_STRENGTH = 8     # equivalent reviews of pull toward the global mean
PRICE_ORDER = ["INEXPENSIVE", "MODERATE", "EXPENSIVE", "VERY_EXPENSIVE", "(missing)"]


def load(csv_path):
    rows = []
    missing_price = 0
    for r in csv.DictReader(open(csv_path, newline="")):
        try:
            rating = float(r["rating"])
            reviews = int(float(r["user_rating_count"] or 0))
        except (ValueError, TypeError, KeyError):
            continue
        if reviews <= MIN_REVIEWS or not (0 < rating <= 5):
            continue
        cuisine = classify(r.get("types", ""))
        if cuisine is None:
            continue
        raw_price = (r.get("price_level") or "").replace("PRICE_LEVEL_", "")
        if not raw_price:
            missing_price += 1
            raw_price = "(missing)"
        rows.append((cuisine, raw_price, rating))
    return rows, missing_price


def pivot(rows):
    """Return cuisines (sorted by count desc), prices (fixed order), and a count+mean cube."""
    cuisines = sorted({c for c, _, _ in rows},
                      key=lambda c: -sum(1 for cc, _, _ in rows if cc == c))
    cube_n = np.zeros((len(cuisines), len(PRICE_ORDER)), dtype=int)
    cube_sum = np.zeros((len(cuisines), len(PRICE_ORDER)))
    for c, p, r in rows:
        if c not in cuisines or p not in PRICE_ORDER:
            continue
        i = cuisines.index(c)
        j = PRICE_ORDER.index(p)
        cube_n[i, j] += 1
        cube_sum[i, j] += r
    with np.errstate(invalid="ignore", divide="ignore"):
        cube_mean = np.where(cube_n > 0, cube_sum / cube_n, np.nan)
    return cuisines, cube_n, cube_mean


def shrunk(cube_n, cube_mean, global_mean):
    """Bayesian shrinkage toward the global mean with prior strength PRIOR_STRENGTH."""
    k = PRIOR_STRENGTH
    return (cube_n * np.where(np.isnan(cube_mean), 0, cube_mean) + k * global_mean) / (cube_n + k)


def deviation(cube_shrunk, row_marg, col_marg, global_mean):
    """Cell value minus additive expectation; flags strong interaction effects."""
    expected = row_marg[:, None] + col_marg[None, :] - global_mean
    return cube_shrunk - expected


def plot(cuisines, cube_n, cube_mean, cube_shrunk, dev, out_png, missing_price, total):
    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(cuisines) + 2)))
    im = ax.imshow(cube_shrunk, cmap="RdYlGn", vmin=4.0, vmax=4.8, aspect="auto")

    ax.set_xticks(range(len(PRICE_ORDER)))
    ax.set_xticklabels([p.replace("_", " ").title() for p in PRICE_ORDER],
                       rotation=20, ha="right")
    ax.set_yticks(range(len(cuisines)))
    ax.set_yticklabels(cuisines)

    for i in range(len(cuisines)):
        for j in range(len(PRICE_ORDER)):
            n = cube_n[i, j]
            if n == 0:
                ax.text(j, i, "—", ha="center", va="center", color="grey", fontsize=9)
                continue
            mean = cube_mean[i, j]
            shr = cube_shrunk[i, j]
            d = dev[i, j]
            # Show shrunk rating, count, and a marker if the cell interacts strongly.
            flag = ""
            if abs(d) >= 0.15 and n >= 5:
                flag = "  ★" if d > 0 else "  ✗"
            ax.text(j, i, f"{shr:.2f}\nn={n}{flag}",
                    ha="center", va="center", fontsize=8,
                    color="black" if 4.2 < shr < 4.6 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Bayesian-shrunk mean rating (k=8 prior toward global mean)")

    pct_missing = 100 * missing_price / total if total else 0
    ax.set_title(
        f"Munich price × cuisine: shrunk mean rating  "
        f"(n={total} > {MIN_REVIEWS} reviews; "
        f"price_level missing on {missing_price} = {pct_missing:.0f}%)\n"
        f"★ = positive interaction, ✗ = negative interaction "
        f"(|shrunk − additive expectation| ≥ 0.15, cell n ≥ 5)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "price_cuisine_grid.png"

    rows, missing_price = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}.")
        sys.exit(1)

    cuisines, cube_n, cube_mean = pivot(rows)
    global_mean = float(np.mean([r[2] for r in rows]))
    cube_shrunk = shrunk(cube_n, cube_mean, global_mean)

    # Row and column marginals on the shrunk grid (weighted by counts).
    with np.errstate(invalid="ignore", divide="ignore"):
        row_n = cube_n.sum(axis=1)
        col_n = cube_n.sum(axis=0)
        row_marg = np.where(row_n > 0,
                            (cube_n * cube_shrunk).sum(axis=1) / np.where(row_n > 0, row_n, 1),
                            global_mean)
        col_marg = np.where(col_n > 0,
                            (cube_n * cube_shrunk).sum(axis=0) / np.where(col_n > 0, col_n, 1),
                            global_mean)
    dev = deviation(cube_shrunk, row_marg, col_marg, global_mean)

    print(f"Loaded {len(rows)} restaurants from {in_csv}; price_level missing on "
          f"{missing_price} ({100*missing_price/len(rows):.1f}%).")
    print(f"Global mean rating: {global_mean:.3f}")
    print(f"Cuisines: {len(cuisines)}, price tiers: {len(PRICE_ORDER)}")

    plot(cuisines, cube_n, cube_mean, cube_shrunk, dev, out_png,
         missing_price, len(rows))


if __name__ == "__main__":
    main()
