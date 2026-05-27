#!/usr/bin/env python3
"""Static PNG showing the scan footprint: 5 km box + every scraped place on OSM."""

import sys
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer

from munich_grid_scrape import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON, L, M_PER_DEG_LAT, m_per_deg_lon

CSV_PATH = "munich_restaurants.csv"
OUT_PNG = "munich_scan_coverage.png"

to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def main():
    lats, lons, ratings = [], [], []
    for r in csv.DictReader(open(CSV_PATH, newline="")):
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
            rating = float(r["rating"])
            reviews = int(float(r["user_rating_count"] or 0))
        except (ValueError, TypeError, KeyError):
            continue
        if reviews <= 100:
            continue
        # Append after all four parse, so the three plotted lists stay aligned.
        lats.append(lat); lons.append(lon); ratings.append(rating)
    lats = np.array(lats); lons = np.array(lons); ratings = np.array(ratings)

    # Compute the 5 km box corners (lat/lon), then project to web mercator.
    half = L / 2
    m_lon = m_per_deg_lon(DEFAULT_CENTER_LAT)
    box_lat = [DEFAULT_CENTER_LAT - half / M_PER_DEG_LAT,
               DEFAULT_CENTER_LAT + half / M_PER_DEG_LAT]
    box_lon = [DEFAULT_CENTER_LON - half / m_lon,
               DEFAULT_CENTER_LON + half / m_lon]
    bx, by = to_3857.transform(
        [box_lon[0], box_lon[1], box_lon[1], box_lon[0], box_lon[0]],
        [box_lat[0], box_lat[0], box_lat[1], box_lat[1], box_lat[0]],
    )
    px, py = to_3857.transform(lons, lats)

    fig, ax = plt.subplots(figsize=(9, 9))
    pad = 800  # meters of padding around the box in EPSG:3857 units
    ax.set_xlim(min(bx) - pad, max(bx) + pad)
    ax.set_ylim(min(by) - pad, max(by) + pad)

    cx.add_basemap(ax, crs="EPSG:3857",
                   source=cx.providers.CartoDB.PositronNoLabels, zoom=14)

    sc = ax.scatter(px, py, c=ratings, cmap="RdYlGn", vmin=3.8, vmax=4.9,
                    s=14, edgecolor="black", linewidths=0.25, alpha=0.85,
                    zorder=3)
    ax.plot(bx, by, color="#1f3b66", lw=2.0, zorder=4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Average rating")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"Munich scan coverage: {L/1000:.0f} km box around Marienplatz\n"
        f"{len(lats)} restaurants scraped (>{100} reviews)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
