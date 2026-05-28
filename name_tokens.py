#!/usr/bin/env python3
"""
Restaurant-name token regression.

Tokenize each restaurant's name (lowercased, ASCII-folded, length >= 2),
keep tokens that appear in >= MIN_FREQ names, and fit a ridge regression:

    rating ~ alpha * tokens + cuisine_fixed_effects

Sample weights are log10(reviews), so well-evidenced ratings carry more
weight than thinly reviewed ones. Bootstrap resampling (200 draws) gives
each coefficient a 95% CI. Tokens whose CI excludes zero are the
genuinely-predictive words.

Usage:
    python name_tokens.py [input_csv]
"""

import sys
import csv
import re
import unicodedata
import collections

import numpy as np
from sklearn.linear_model import Ridge
from prettytable import PrettyTable

from cuisine import classify, MIN_REVIEWS

MIN_FREQ = 10        # token must appear in at least this many names
RIDGE_ALPHA = 1.0    # L2 penalty
TOP_N = 12           # top positive/negative tokens to print
N_BOOT = 200


def ascii_fold(s):
    """Fold accented characters to plain ASCII (Müller -> Muller)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


WORD_RE = re.compile(r"[a-z][a-z]+")


def tokenize(name):
    """Return a set of lowercased ASCII tokens of length >= 2."""
    s = ascii_fold(name).lower()
    return set(WORD_RE.findall(s))


def load(csv_path):
    rows = []
    for r in csv.DictReader(open(csv_path, newline="")):
        try:
            rating = float(r["rating"])
            reviews = int(float(r["user_rating_count"] or 0))
        except (ValueError, TypeError, KeyError):
            continue
        if reviews <= MIN_REVIEWS or not (0 < rating <= 5):
            continue
        rows.append({
            "name": (r.get("name") or "").strip() or "(unnamed)",
            "rating": rating, "reviews": reviews,
            "cuisine": classify(r.get("types", "")) or "Other",
        })
    return rows


def build_design(rows):
    """Return (X, y, w, token_names, cuisine_names)."""
    # Vocabulary above MIN_FREQ
    counts = collections.Counter()
    per_row_tokens = []
    for r in rows:
        toks = tokenize(r["name"])
        per_row_tokens.append(toks)
        counts.update(toks)
    vocab = sorted(t for t, n in counts.items() if n >= MIN_FREQ)

    # Cuisine fixed effects (dummy-encoded, drop one to avoid colinearity).
    cuisines = sorted({r["cuisine"] for r in rows})
    drop_cuisine = cuisines[0]
    cuisine_cols = [c for c in cuisines if c != drop_cuisine]

    n = len(rows)
    k = len(vocab) + len(cuisine_cols)
    X = np.zeros((n, k), dtype=float)
    y = np.zeros(n)
    w = np.zeros(n)

    vocab_idx = {t: i for i, t in enumerate(vocab)}
    cuisine_idx = {c: len(vocab) + i for i, c in enumerate(cuisine_cols)}

    for i, r in enumerate(rows):
        for t in per_row_tokens[i]:
            if t in vocab_idx:
                X[i, vocab_idx[t]] = 1.0
        if r["cuisine"] in cuisine_idx:
            X[i, cuisine_idx[r["cuisine"]]] = 1.0
        y[i] = r["rating"]
        w[i] = np.log10(r["reviews"])

    return X, y, w, vocab, cuisine_cols, drop_cuisine, counts


def fit_with_ci(X, y, w, n_boot=N_BOOT, rng=None):
    """Bootstrap ridge: returns (mean_coef, lo_coef, hi_coef)."""
    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    samples = np.zeros((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X[idx], y[idx], sample_weight=w[idx])
        samples[b] = model.coef_
    return samples.mean(axis=0), np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def print_results(coef_mean, coef_lo, coef_hi, vocab, cuisine_cols, drop_cuisine, counts):
    n_vocab = len(vocab)
    significant = []
    for i, tok in enumerate(vocab):
        lo, hi = coef_lo[i], coef_hi[i]
        if lo * hi > 0:  # CI on one side of 0 → "significant"
            significant.append((tok, coef_mean[i], lo, hi, counts[tok]))

    significant.sort(key=lambda x: x[1], reverse=True)
    pos = [s for s in significant if s[1] > 0][:TOP_N]
    neg = [s for s in significant if s[1] < 0]
    neg.sort(key=lambda x: x[1])  # ascending: most negative first
    neg = neg[:TOP_N]

    def render(title, items):
        t = PrettyTable()
        t.field_names = ["Token", "Coefficient", "95% CI", "Names with"]
        t.align["Token"] = "l"
        t.align["Coefficient"] = "r"
        t.align["95% CI"] = "c"
        t.align["Names with"] = "r"
        for tok, m, lo, hi, n in items:
            t.add_row([tok, f"{m:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", n])
        print(f"\n{title}")
        print(t)

    render(f"Top {TOP_N} positive tokens (95% CI excludes 0):", pos)
    render(f"Top {TOP_N} negative tokens (95% CI excludes 0):", neg)

    # Cuisine fixed effects, for context.
    print("\nCuisine fixed effects (relative to dropped baseline cuisine "
          f"'{drop_cuisine}'):")
    t = PrettyTable()
    t.field_names = ["Cuisine", "Coefficient", "95% CI"]
    t.align["Cuisine"] = "l"
    t.align["Coefficient"] = "r"
    t.align["95% CI"] = "c"
    for j, c in enumerate(cuisine_cols):
        idx = n_vocab + j
        t.add_row([c, f"{coef_mean[idx]:+.3f}",
                   f"[{coef_lo[idx]:+.3f}, {coef_hi[idx]:+.3f}]"])
    print(t)


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    rows = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}.")
        sys.exit(1)

    X, y, w, vocab, cuisine_cols, drop_cuisine, counts = build_design(rows)
    print(f"Loaded {len(rows)} restaurants > {MIN_REVIEWS} reviews")
    print(f"Token vocabulary: {len(vocab)} tokens (>= {MIN_FREQ} occurrences)")
    print(f"Cuisines (fixed effects, '{drop_cuisine}' as baseline): "
          f"{len(cuisine_cols) + 1}")
    print(f"Ridge alpha={RIDGE_ALPHA}, weighted by log10(reviews); "
          f"bootstrap n={N_BOOT}.")

    coef_mean, coef_lo, coef_hi = fit_with_ci(X, y, w)
    print_results(coef_mean, coef_lo, coef_hi, vocab, cuisine_cols, drop_cuisine, counts)


if __name__ == "__main__":
    main()
