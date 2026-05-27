#!/usr/bin/env python3
"""
2D histogram of restaurants: review count (log x) vs average rating (linear y).

Reads munich_restaurants.csv and writes a PNG showing where places cluster in
the (popularity, quality) plane. Log-scaled review-count axis because counts
span ~3 orders of magnitude (100 to 100k+).

Usage:
    python rating_2d_hist.py [input_csv] [output_png]
Defaults:
    input_csv  = munich_restaurants.csv
    output_png = munich_rating_2d_hist.png
"""

import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from prettytable import PrettyTable

# Bin layout. Log-spaced x over [X_LO, X_HI] (2 decades); linear y over [Y_LO, Y_HI].
# Data outside these ranges is dropped by hist2d, so the colour scale normalizes
# against only what's on screen.
X_LO, X_HI = 100, 10_000
N_X_BINS = 24
Y_LO, Y_HI = 2.5, 5.0
Y_EDGES = np.arange(Y_LO, Y_HI + 0.01, 0.1)


def load(csv_path):
    """Return a list of {name, rating, reviews, price} dicts (one per valid row)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                c = int(float(r["user_rating_count"]))
                v = float(r["rating"])
            except (ValueError, TypeError, KeyError):
                continue
            if c <= 0 or not (0 < v <= 5):
                continue
            rows.append({
                "name": (r.get("name") or "").strip() or "(unnamed)",
                "rating": v,
                "reviews": c,
                "price": (r.get("price_level") or "").replace("PRICE_LEVEL_", "") or "-",
            })
    return rows


SORT_MODES = {
    "rating":  (lambda r: (r["rating"], r["reviews"]),  "by rating, tiebroken by review count"),
    "reviews": (lambda r: (r["reviews"], r["rating"]),  "by review count, tiebroken by rating"),
    "bayes":   (lambda r: (r["bayes"],   r["reviews"]), "by Bayesian-weighted rating "
                                                        "(shrinks small samples toward the global mean)"),
}


def add_bayes(rows):
    """Annotate each row with a Bayesian-weighted rating using an IMDb-style formula.

    WR = (v / (v + m)) * R + (m / (v + m)) * C
        R = the place's average rating
        v = the place's review count
        C = global mean rating across the dataset (the prior mean)
        m = prior strength, in "equivalent reviews" — uses the median review count
            so 100-review places get noticeably shrunk but 5k-review places barely budge.

    Returns (m, C) for display.
    """
    ratings = np.array([r["rating"] for r in rows])
    reviews = np.array([r["reviews"] for r in rows])
    C = float(ratings.mean())
    m = float(np.median(reviews))
    for r in rows:
        v = r["reviews"]
        r["bayes"] = (v / (v + m)) * r["rating"] + (m / (v + m)) * C
    return m, C


def print_top(rows, n=10, *, by="rating"):
    """Print the top-n restaurants ranked according to SORT_MODES[by]."""
    key, heading = SORT_MODES[by]
    top = sorted(rows, key=key, reverse=True)[:n]
    show_bayes = (by == "bayes")
    t = PrettyTable()
    cols = ["#", "Restaurant", "Rating", "Reviews", "Price"]
    if show_bayes:
        cols.insert(3, "Bayes")
    t.field_names = cols
    t.align["Restaurant"] = "l"
    t.align["Rating"] = "r"
    t.align["Reviews"] = "r"
    t.align["Price"] = "l"
    if show_bayes:
        t.align["Bayes"] = "r"
    for i, r in enumerate(top, 1):
        row = [i, r["name"], f"{r['rating']:.1f}"]
        if show_bayes:
            row.append(f"{r['bayes']:.2f}")
        row.extend([f"{r['reviews']:,}", r["price"]])
        t.add_row(row)
    print(f"\nTop {n} restaurants ({heading}):")
    print(t)


def plot(rows, out_png):
    counts = np.array([r["reviews"] for r in rows])
    ratings = np.array([r["rating"] for r in rows])
    x_edges = np.logspace(np.log10(X_LO), np.log10(X_HI), N_X_BINS + 1)

    # Restrict to the visible window so marginals and the 2D color scale agree.
    in_win = (counts >= X_LO) & (counts <= X_HI) & (ratings >= Y_LO) & (ratings <= Y_HI)
    c_win, r_win = counts[in_win], ratings[in_win]

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        wspace=0.04, hspace=0.04,
        left=0.07, right=0.97, top=0.92, bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_xmar = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_ymar = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Top-right corner: thin horizontal colorbar inset, rest of the cell blank.
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    ax_cb = ax_corner.inset_axes([0.05, 0.55, 0.9, 0.18])

    # 2D histogram (LogNorm so single-count cells stay visible next to dense ones).
    _, _, _, im = ax_main.hist2d(
        c_win, r_win, bins=[x_edges, Y_EDGES],
        cmap="magma_r", norm=LogNorm(vmin=1),
    )
    ax_main.set_xscale("log")
    ax_main.set_xlim(X_LO, X_HI)
    ax_main.set_ylim(Y_LO, Y_HI)
    ax_main.set_xlabel("Number of ratings (log scale)")
    ax_main.set_ylabel("Average rating")
    ax_main.grid(True, which="both", alpha=0.2)

    bar_kw = dict(color="#552288", alpha=0.85, edgecolor="white", linewidth=0.4)

    ax_xmar.hist(c_win, bins=x_edges, **bar_kw)
    ax_xmar.set_ylabel("count")
    ax_xmar.tick_params(axis="x", labelbottom=False)
    ax_xmar.grid(True, axis="y", alpha=0.2)

    ax_ymar.hist(r_win, bins=Y_EDGES, orientation="horizontal", **bar_kw)
    ax_ymar.set_xlabel("count")
    ax_ymar.tick_params(axis="y", labelleft=False)
    ax_ymar.grid(True, axis="x", alpha=0.2)

    fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    ax_cb.set_xlabel("Restaurants per 2D bin", fontsize=9)
    ax_cb.tick_params(labelsize=8)

    # Medians on the in-window data, drawn through all three panels.
    med_c = float(np.median(c_win))
    med_r = float(np.median(r_win))
    line_kw = dict(ls="--", lw=1, alpha=0.7)
    ax_main.axvline(med_c, color="steelblue", **line_kw)
    ax_main.axhline(med_r, color="seagreen",  **line_kw)
    ax_xmar.axvline(med_c, color="steelblue", **line_kw)
    ax_ymar.axhline(med_r, color="seagreen",  **line_kw)
    ax_main.text(med_c * 1.05, Y_LO + 0.05, f"median = {int(med_c)}",
                 color="steelblue", fontsize=9, ha="left", va="bottom")
    ax_main.text(X_HI * 0.97, med_r + 0.03, f"median = {med_r:.2f}",
                 color="seagreen", fontsize=9, ha="right", va="bottom")

    fig.suptitle(
        f"Munich restaurants: popularity vs quality  "
        f"(n={in_win.sum()} in window / {len(counts)} total)",
        fontsize=12,
    )
    plt.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "munich_rating_2d_hist.png"

    rows = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}.")
        sys.exit(1)

    counts = np.array([r["reviews"] for r in rows])
    ratings = np.array([r["rating"] for r in rows])
    print(f"Loaded {len(rows)} restaurants from {in_csv}")
    print(f"  rating  range: {ratings.min():.1f} - {ratings.max():.1f}, "
          f"median {np.median(ratings):.2f}")
    print(f"  count   range: {counts.min()} - {counts.max()}, "
          f"median {int(np.median(counts))}")

    m, C = add_bayes(rows)
    print(f"\nBayesian prior: m = {m:.0f} reviews (median), C = {C:.2f} stars (mean)")

    print_top(rows, n=10, by="rating")
    print_top(rows, n=10, by="reviews")
    print_top(rows, n=10, by="bayes")
    plot(rows, out_png)


if __name__ == "__main__":
    main()
