#!/usr/bin/env python3
"""
Interactive Folium map of Munich restaurants.

Each restaurant is a circle marker, colored by Bayesian-weighted rating and
sized by log(reviews). Tooltip shows name / cuisine / rating / reviews / price.
Layer control toggles per cuisine. The adaptive scrape grid (munich_grid.json)
is overlaid as a hidden layer for debugging coverage.

Output: a single self-contained HTML file (default munich_map.html).

Usage:
    python map_html.py [input_csv] [output_html]
"""

import sys
import csv
import json
import math
import os
import html

import folium

from cuisine import classify, MIN_REVIEWS

CENTER_LAT = 48.1370339      # Marienplatz
CENTER_LON = 11.5758134
ZOOM_START = 13


def load(csv_path):
    """Load CSV rows with classification, restricted to reviews > MIN_REVIEWS."""
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
            cuisine = classify(r.get("types", "")) or "Other"
            rows.append({
                "name": (r.get("name") or "").strip() or "(unnamed)",
                "rating": rating, "reviews": reviews,
                "lat": lat, "lon": lon,
                "cuisine": cuisine,
                "price": (r.get("price_level") or "").replace("PRICE_LEVEL_", "") or "-",
            })
    return rows


def add_bayes(rows):
    """Annotate each row with Bayesian-weighted rating (prior over trustworthy subset)."""
    pool = [r for r in rows if r["reviews"] > MIN_REVIEWS]  # already true here
    ratings = [r["rating"] for r in pool]
    reviews = [r["reviews"] for r in pool]
    C = sum(ratings) / len(ratings)
    m = sorted(reviews)[len(reviews) // 2]
    for r in rows:
        v = r["reviews"]
        r["bayes"] = (v / (v + m)) * r["rating"] + (m / (v + m)) * C
    return m, C


def rating_to_color(bayes, lo=4.0, hi=4.8):
    """Map a Bayesian rating to a HEX color on a red->green ramp.

    Clamps to [lo, hi] so the full range maps to the saturated edges of the ramp.
    Below lo: red. Above hi: green. Middle: yellow/orange.
    """
    t = max(0.0, min(1.0, (bayes - lo) / (hi - lo)))
    # Red (220, 50, 50) -> Yellow (240, 200, 60) -> Green (40, 160, 70)
    if t < 0.5:
        u = t * 2
        r = int(220 + (240 - 220) * u)
        g = int(50 + (200 - 50) * u)
        b = int(50 + (60 - 50) * u)
    else:
        u = (t - 0.5) * 2
        r = int(240 + (40 - 240) * u)
        g = int(200 + (160 - 200) * u)
        b = int(60 + (70 - 60) * u)
    return f"#{r:02x}{g:02x}{b:02x}"


def radius_from_reviews(reviews, lo=4, hi=18):
    """Map log10(reviews) into a pixel radius range."""
    t = (math.log10(max(reviews, 1)) - 2) / (5 - 2)   # 100..100k -> 0..1
    t = max(0.0, min(1.0, t))
    return lo + (hi - lo) * t


def build_map(rows, grid_path=None):
    # prefer_canvas: render 1k+ markers as one canvas element instead of one
    # SVG node per circle. Leaflet's SVG renderer chokes well before 2k markers
    # — pages can appear blank or hang. Canvas keeps it interactive at this size.
    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON], zoom_start=ZOOM_START,
        tiles="cartodbpositron", control_scale=True,
        prefer_canvas=True,
    )

    # One FeatureGroup per cuisine so the LayerControl can toggle them.
    cuisines = sorted({r["cuisine"] for r in rows})
    groups = {c: folium.FeatureGroup(name=c, show=True) for c in cuisines}

    for r in rows:
        # html.escape() handles <, >, &; backticks must also be neutralized
        # because Folium emits tooltips inside JS template literals, and a
        # literal backtick in a name (e.g. "Tapas by Noah`s") closes the
        # literal mid-string and breaks every subsequent script statement.
        # $ likewise needs escaping so "${" can't trigger interpolation.
        def safe(s):
            return html.escape(str(s)).replace("`", "&#96;").replace("$", "&#36;")
        tooltip = (
            f"<b>{safe(r['name'])}</b><br>"
            f"cuisine: {safe(r['cuisine'])}<br>"
            f"rating: {r['rating']:.1f}  (Bayes {r['bayes']:.2f})<br>"
            f"reviews: {r['reviews']:,}<br>"
            f"price: {safe(r['price'])}"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius_from_reviews(r["reviews"]),
            color=rating_to_color(r["bayes"]),
            weight=1, fill=True, fill_opacity=0.75,
            # Tooltip = on-hover preview, popup = click-to-pin persistent panel.
            tooltip=folium.Tooltip(tooltip, sticky=True),
            popup=folium.Popup(tooltip, max_width=300),
        ).add_to(groups[r["cuisine"]])

    for g in groups.values():
        g.add_to(m)

    # Optional: overlay the scrape grid (off by default).
    if grid_path and os.path.exists(grid_path):
        grid_layer = folium.FeatureGroup(name="scrape grid", show=False)
        with open(grid_path) as f:
            tiles = json.load(f)
        for t in tiles:
            # Half-edge expressed in degrees (rough; same approximation as the scraper).
            half_lat = (t["edge"] / 2) / 111_320
            half_lon = (t["edge"] / 2) / (111_320 * math.cos(math.radians(t["lat"])))
            folium.Rectangle(
                bounds=[
                    (t["lat"] - half_lat, t["lon"] - half_lon),
                    (t["lat"] + half_lat, t["lon"] + half_lon),
                ],
                color="#444", weight=0.7, fill=False,
            ).add_to(grid_layer)
        grid_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html()))
    return m


def _legend_html():
    """Floating legend in the bottom-left explaining marker color and size.

    The color stops mirror rating_to_color()'s ramp; the SVG circle radii
    mirror radius_from_reviews() at log10 = 2, 3, 4, 5 so the swatches match
    actual marker sizes on the map.
    """
    return """
    <div style="position: absolute; bottom: 28px; left: 16px; z-index: 1000;
                background: white; padding: 10px 12px; border-radius: 4px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.25);
                font-family: -apple-system, sans-serif; font-size: 12px;
                color: #222;">
      <div style="font-weight: 600; margin-bottom: 4px;">Bayesian rating</div>
      <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 12px;">
        <span style="font-variant-numeric: tabular-nums;">4.0</span>
        <div style="width: 150px; height: 12px;
                    background: linear-gradient(to right,
                        rgb(220, 50, 50),
                        rgb(240, 200, 60),
                        rgb(40, 160, 70));
                    border: 1px solid #888;"></div>
        <span style="font-variant-numeric: tabular-nums;">4.8</span>
      </div>
      <div style="font-weight: 600; margin-bottom: 4px;">Review count</div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="display: inline-flex; flex-direction: column; align-items: center;">
          <svg width="12" height="12" viewBox="0 0 12 12">
            <circle cx="6" cy="6" r="4" fill="#888" stroke="#222" stroke-width="0.5"/>
          </svg>
          <span>10²</span>
        </span>
        <span style="display: inline-flex; flex-direction: column; align-items: center;">
          <svg width="22" height="22" viewBox="0 0 22 22">
            <circle cx="11" cy="11" r="9" fill="#888" stroke="#222" stroke-width="0.5"/>
          </svg>
          <span>10³</span>
        </span>
        <span style="display: inline-flex; flex-direction: column; align-items: center;">
          <svg width="32" height="32" viewBox="0 0 32 32">
            <circle cx="16" cy="16" r="13" fill="#888" stroke="#222" stroke-width="0.5"/>
          </svg>
          <span>10⁴</span>
        </span>
        <span style="display: inline-flex; flex-direction: column; align-items: center;">
          <svg width="40" height="40" viewBox="0 0 40 40"><circle cx="20" cy="20" r="18" fill="#888" stroke="#222" stroke-width="0.5"/></svg>
          <span>10⁵</span>
        </span>
      </div>
    </div>
    """


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_html = sys.argv[2] if len(sys.argv) > 2 else "munich_map.html"
    grid_json = "munich_grid.json"

    rows = load(in_csv)
    if not rows:
        print(f"No usable rows in {in_csv}.")
        sys.exit(1)

    m_prior, C_prior = add_bayes(rows)
    print(f"Loaded {len(rows)} restaurants (> {MIN_REVIEWS} reviews) from {in_csv}")
    print(f"Bayesian prior: m={m_prior} reviews, C={C_prior:.2f} stars")

    fmap = build_map(rows, grid_path=grid_json)
    fmap.save(out_html)
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
