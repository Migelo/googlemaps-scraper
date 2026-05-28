#!/usr/bin/env python3
"""
KDE-based "where is the good food" map for Munich.

Two panels, sharing the same OSM basemap:

    Panel A: density of *every* restaurant (popularity geography)
    Panel B: density weighted by (rating - global_mean) * log(reviews)
             -> positive where there are more well-rated places than average,
                negative where there are more poorly-rated ones.

Output: munich_kde_map.png

Usage:
    python kde_quality_map.py [input_csv] [output_png]
"""

import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from pyproj import Transformer
import contextily as cx

from cuisine import MIN_REVIEWS
from munich_grid_scrape import M_PER_DEG_LAT, m_per_deg_lon

GRID_RES = 220   # samples per side for KDE evaluation
PAD_M = 400      # meters of padding around the data extent for the KDE window

# Bandwidth: KDE auto-bandwidth via Scott's rule is often too tight in geo data;
# multiply by this factor for visually meaningful smoothing.
BW_FACTOR = 0.6

# Web Mercator (what contextily basemaps live in).
to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rating = float(r["rating"])
                reviews = int(float(r["user_rating_count"] or 0))
                lat = float(r["lat"])
                lon = float(r["lon"])
            except (ValueError, TypeError, KeyError):
                continue
            if reviews <= MIN_REVIEWS or not (0 < rating <= 5):
                continue
            rows.append((lat, lon, rating, reviews))
    return rows


def data_window(lats, lons):
    """Padded (lat_lo, lat_hi, lon_lo, lon_hi) box enclosing all points.

    Frames whatever was actually scraped instead of a fixed extent, so a larger
    scrape isn't clipped.
    """
    mlon = m_per_deg_lon((lats.min() + lats.max()) / 2)
    dlat = PAD_M / M_PER_DEG_LAT
    dlon = PAD_M / mlon
    return (lats.min() - dlat, lats.max() + dlat,
            lons.min() - dlon, lons.max() + dlon)


def build_grid_3857(bounds):
    """Return a meshgrid in EPSG:3857 covering the (lat_lo,lat_hi,lon_lo,lon_hi) box."""
    lat_lo, lat_hi, lon_lo, lon_hi = bounds
    xs_lo, ys_lo = to_3857.transform(lon_lo, lat_lo)
    xs_hi, ys_hi = to_3857.transform(lon_hi, lat_hi)
    xs = np.linspace(xs_lo, xs_hi, GRID_RES)
    ys = np.linspace(ys_lo, ys_hi, GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    return X, Y, (xs_lo, xs_hi, ys_lo, ys_hi)


def kde(xs, ys, weights, gridX, gridY):
    """Evaluate a 2D weighted KDE on the meshgrid (returns same-shape array)."""
    data = np.vstack([xs, ys])
    k = gaussian_kde(data, weights=weights, bw_method=BW_FACTOR)
    flat = np.vstack([gridX.ravel(), gridY.ravel()])
    return k(flat).reshape(gridX.shape)


def plot(rows, out_png):
    lats = np.array([r[0] for r in rows])
    lons = np.array([r[1] for r in rows])
    ratings = np.array([r[2] for r in rows])
    reviews = np.array([r[3] for r in rows])

    # Project once.
    xs, ys = to_3857.transform(lons, lats)

    gridX, gridY, extent = build_grid_3857(data_window(lats, lons))

    # Panel A: density (uniform weights).
    A = kde(xs, ys, weights=None, gridX=gridX, gridY=gridY)

    # Panel B: signed-quality density. Weight = (rating - global_mean) * log10(reviews).
    # log10(reviews) downweights small-N places; centering on the mean makes the
    # field negative where bad-rated places concentrate.
    # scipy.stats.gaussian_kde rejects negative weights, so we fit two KDEs —
    # one over above-mean places, one over below-mean — and subtract.
    mu = ratings.mean()
    w_signed = (ratings - mu) * np.log10(reviews)
    pos = w_signed > 0
    neg = w_signed < 0
    B_pos = kde(xs[pos], ys[pos], weights=w_signed[pos], gridX=gridX, gridY=gridY)
    B_neg = kde(xs[neg], ys[neg], weights=-w_signed[neg], gridX=gridX, gridY=gridY)
    B = B_pos - B_neg
    # Renormalize B to its absolute peak so the colormap is symmetric around 0.
    B_abs_max = max(abs(B.min()), abs(B.max()))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)

    panels = [
        # (axes, field, title, cmap_name, vmin, vmax, "sequential"|"diverging")
        (axes[0], A, "Restaurant density", "magma_r",
         None, None, "sequential"),
        (axes[1], B, "Quality density (rating − mean, ×log reviews)", "RdBu_r",
         -B_abs_max, B_abs_max, "diverging"),
    ]

    for ax, field, title, cmap_name, vmin, vmax, mode in panels:
        # Basemap first so the KDE field sits on top of it (and behind the scatter).
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.PositronNoLabels,
                       attribution_size=6, zoom=14)

        # Build an RGBA image with per-pixel alpha proportional to signal
        # magnitude, so the basemap stays visible where the KDE has nothing
        # interesting to say and only the actual signal hides it.
        cmap = matplotlib.colormaps[cmap_name]
        if mode == "sequential":
            norm = mcolors.Normalize(vmin=field.min(), vmax=field.max())
            magnitude = norm(field)                       # 0 (transparent) → 1 (opaque)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            magnitude = np.abs(field) / max(abs(vmin), abs(vmax), 1e-12)
        rgba = cmap(norm(field))
        # gamma 0.6 makes mid-range magnitudes already quite visible while
        # keeping the bottom 10% essentially transparent.
        rgba[..., 3] = np.clip(magnitude ** 0.6 * 0.80, 0, 0.80)

        ax.imshow(rgba, origin="lower", extent=extent, zorder=2)
        ax.scatter(xs, ys, s=5, color="white", edgecolor="black",
                   linewidths=0.3, alpha=0.7, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # The image we drew has bespoke alpha, so the colorbar needs a separate
        # ScalarMappable to expose the colormap+norm without inheriting alpha.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(f"Munich: where is the good food?  (n={len(rows)}, > {MIN_REVIEWS} reviews)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "munich_kde_map.png"

    rows = load(in_csv)
    if len(rows) < 30:
        print(f"Only {len(rows)} usable rows in {in_csv}; KDE needs more data.")
        sys.exit(1)

    print(f"Loaded {len(rows)} restaurants (> {MIN_REVIEWS} reviews)")
    plot(rows, out_png)


if __name__ == "__main__":
    main()
