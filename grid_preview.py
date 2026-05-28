#!/usr/bin/env python3
"""Preview a dry-run seed grid on an OSM basemap before spending any quota.

Reads a grid JSON written by `munich_grid_scrape.py --dry-run` (a list of
{lat, lon, edge, radius} seed tiles) and draws each tile square plus its
searchNearby circle on a CartoDB Positron basemap. Adjacent circles overlap
across shared edges — the no-gap property the adaptive scan relies on.

Only the SEED grid is shown: saturated tiles subdivide during the real run,
so the live scan is finer than this wherever the city is dense.

Usage:
    uv run python munich_grid_scrape.py --city berlin --grid 10 --dry-run
    uv run python grid_preview.py [grid_json] [out_png]

grid_json defaults to munich_grid.json; out_png defaults to the grid name with
_grid.json -> _grid_preview.png.
"""

import sys
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import contextily as cx
from pyproj import Transformer

from munich_grid_scrape import M_PER_DEG_LAT, m_per_deg_lon

to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def main():
    grid_json = sys.argv[1] if len(sys.argv) > 1 else "munich_grid.json"
    out_png = (sys.argv[2] if len(sys.argv) > 2
               else grid_json.replace("_grid.json", "_grid_preview.png"))

    try:
        tiles = json.load(open(grid_json))
    except FileNotFoundError:
        print(f"{grid_json} not found. Generate it first with "
              f"`munich_grid_scrape.py ... --dry-run`.")
        sys.exit(1)
    if not tiles:
        print(f"No tiles in {grid_json}.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 9))
    xs, ys = [], []
    for t in tiles:
        lat, lon, edge, rad = t["lat"], t["lon"], t["edge"], t["radius"]
        half_lat = (edge / 2) / M_PER_DEG_LAT
        half_lon = (edge / 2) / m_per_deg_lon(lat)
        x0, y0 = to_3857.transform(lon - half_lon, lat - half_lat)
        x1, y1 = to_3857.transform(lon + half_lon, lat + half_lat)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#1f3b66", lw=1.1, zorder=4))
        cx0, cy0 = to_3857.transform(lon, lat)
        # Web Mercator stretches distances by sec(lat); scale the circle so it
        # matches the tile square it circumscribes.
        sec = 1.0 / math.cos(math.radians(lat))
        ax.add_patch(Circle((cx0, cy0), rad * sec, fill=False,
                            edgecolor="#cc5500", lw=0.7, alpha=0.55, zorder=3))
        xs += [x0, x1]
        ys += [y0, y1]

    pad = 1500
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    cx.add_basemap(ax, crs="EPSG:3857",
                   source=cx.providers.CartoDB.PositronNoLabels, zoom=12)

    # Title derived from the grid: ground width is the 3857 span * cos(lat).
    lat0 = tiles[0]["lat"]
    side_km = (max(xs) - min(xs)) * math.cos(math.radians(lat0)) / 1000
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Seed grid preview: {len(tiles)} tiles, ~{side_km:.1f} km box, "
        f"{tiles[0]['edge']:.0f} m cells (blue) + search circles (orange)\n"
        f"adaptive subdivision happens during the real run",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
