#!/usr/bin/env python3
"""
Cuisine-conditioned outlier ranker.

For each cuisine, fit a rating distribution (mean + std) over its restaurants,
then score every restaurant by a z-score scaled by a review-count shrinkage
factor so 5-review "perfect" places don't dominate.

    surprise = ((rating - cuisine_mean) / cuisine_std) * v / (v + M)

where v is the place's review count and M is a prior strength (median reviews
within the cuisine). Top positive surprises are "great for their kind"; top
negative surprises are "bad for their kind."

Usage:
    python outliers.py [input_csv]
Defaults:
    input_csv = munich_restaurants.csv
"""

import sys
import csv

import numpy as np
from prettytable import PrettyTable

from divergence_pipeline import classify, MIN_REVIEWS

MIN_COHORT = 8   # minimum restaurants per cuisine to compute meaningful stats
TOP_N = 5        # rows printed per cuisine per direction


def load(csv_path):
    """Load CSV rows. Filter by reviews > MIN_REVIEWS and successful cuisine match."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
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
            rows.append({
                "name": (r.get("name") or "").strip() or "(unnamed)",
                "rating": rating,
                "reviews": reviews,
                "cuisine": cuisine,
                "price": (r.get("price_level") or "").replace("PRICE_LEVEL_", "") or "-",
            })
    return rows


def score(rows):
    """Annotate each row with a within-cuisine z-score and a shrunk 'surprise'."""
    # Group rows by cuisine.
    by_cuisine = {}
    for r in rows:
        by_cuisine.setdefault(r["cuisine"], []).append(r)

    stats = {}  # cuisine -> {mu, sigma, M, n}
    for cuisine, group in by_cuisine.items():
        ratings = np.array([r["rating"] for r in group])
        reviews = np.array([r["reviews"] for r in group])
        if len(group) < MIN_COHORT:
            stats[cuisine] = None      # too small to trust
            continue
        sigma = float(ratings.std(ddof=1))
        # If a cuisine has zero spread (everyone tied), z would NaN; force sigma>0.
        sigma = max(sigma, 0.05)
        stats[cuisine] = {
            "mu": float(ratings.mean()),
            "sigma": sigma,
            "M": float(np.median(reviews)),  # shrinkage prior, per cuisine
            "n": len(group),
        }

    for r in rows:
        s = stats.get(r["cuisine"])
        if s is None:
            r["z"] = float("nan")
            r["surprise"] = float("nan")
            continue
        r["z"] = (r["rating"] - s["mu"]) / s["sigma"]
        r["surprise"] = r["z"] * r["reviews"] / (r["reviews"] + s["M"])

    return stats


def print_cuisine_table(rows, stats):
    """Per-cuisine: top TOP_N positive and negative surprises."""
    for cuisine in sorted(stats.keys()):
        s = stats[cuisine]
        if s is None:
            continue
        members = [r for r in rows if r["cuisine"] == cuisine]
        members.sort(key=lambda r: r["surprise"], reverse=True)
        top = members[:TOP_N]
        bot = list(reversed(members[-TOP_N:]))

        print(f"\n{cuisine}  (n={s['n']}, mean={s['mu']:.2f}, sd={s['sigma']:.2f}, "
              f"M={int(s['M'])} reviews)")
        t = PrettyTable()
        t.field_names = ["Rank", "Restaurant", "Rating", "Reviews", "z", "Surprise", "Price"]
        t.align["Restaurant"] = "l"
        t.align["Rating"] = "r"
        t.align["Reviews"] = "r"
        t.align["z"] = "r"
        t.align["Surprise"] = "r"
        for i, r in enumerate(top, 1):
            t.add_row([f"+{i}", r["name"], f"{r['rating']:.1f}",
                       f"{r['reviews']:,}", f"{r['z']:+.2f}",
                       f"{r['surprise']:+.2f}", r["price"]])
        for i, r in enumerate(bot, 1):
            t.add_row([f"-{i}", r["name"], f"{r['rating']:.1f}",
                       f"{r['reviews']:,}", f"{r['z']:+.2f}",
                       f"{r['surprise']:+.2f}", r["price"]])
        print(t)


def print_global_table(rows):
    """Cross-cuisine ranking by absolute surprise — 'most unusual for its kind'."""
    rated = [r for r in rows if not np.isnan(r.get("surprise", float("nan")))]
    rated.sort(key=lambda r: abs(r["surprise"]), reverse=True)
    t = PrettyTable()
    t.field_names = ["#", "Restaurant", "Cuisine", "Rating", "Reviews",
                     "z", "Surprise", "Price"]
    t.align["Restaurant"] = "l"
    t.align["Cuisine"] = "l"
    t.align["Rating"] = "r"
    t.align["Reviews"] = "r"
    t.align["z"] = "r"
    t.align["Surprise"] = "r"
    for i, r in enumerate(rated[:15], 1):
        t.add_row([i, r["name"], r["cuisine"], f"{r['rating']:.1f}",
                   f"{r['reviews']:,}", f"{r['z']:+.2f}",
                   f"{r['surprise']:+.2f}", r["price"]])
    print("\nGlobal top-15 by |surprise| (most unusual for its kind, weighted by reviews):")
    print(t)


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    rows = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}.")
        sys.exit(1)

    stats = score(rows)
    eligible = sum(1 for s in stats.values() if s is not None)
    too_small = [c for c, s in stats.items() if s is None]
    print(f"Loaded {len(rows)} classified restaurants > {MIN_REVIEWS} reviews from {in_csv}")
    print(f"Cuisines analysed: {eligible} (>= {MIN_COHORT} places each)")
    if too_small:
        print(f"Cuisines with too-small cohort, skipped: {', '.join(sorted(too_small))}")

    print_cuisine_table(rows, stats)
    print_global_table(rows)


if __name__ == "__main__":
    main()
