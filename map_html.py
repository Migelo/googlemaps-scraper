#!/usr/bin/env python3
"""
Interactive Folium map of scraped restaurants.

Each restaurant is a circle marker, colored by Bayesian-weighted rating and
sized by log(reviews). Tooltip shows name / cuisine / rating / reviews / price.
Layer control toggles per cuisine. The matching `*_grid.json` (derived from
the input CSV name) is overlaid as a hidden layer for debugging coverage.

City-agnostic: the initial view auto-fits whatever data is in the CSV, so
the same script works for any `{city}_restaurants.csv`.

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


N_CMAP = 64               # LUT resolution for the in-browser colormap switcher
DEFAULT_CMAP = "ryg"      # the original red->yellow->green ramp


def _ramp_color(t):
    """Original red->yellow->green ramp for t in [0, 1] -> HEX."""
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


def bayes_to_t(bayes, lo=4.0, hi=4.8):
    """Normalize a Bayesian rating into [0, 1], clamped to [lo, hi].

    Below lo maps to the cold end of the ramp, above hi to the warm end.
    """
    return max(0.0, min(1.0, (bayes - lo) / (hi - lo)))


def color_index(t, n=N_CMAP):
    """Quantize t in [0, 1] to a LUT index in [0, n-1]."""
    return max(0, min(n - 1, round(t * (n - 1))))


def build_colormaps(n=N_CMAP):
    """Hex-color LUTs for each selectable colormap, keyed by name.

    The custom ramp is sampled from _ramp_color; viridis/magma/jet come straight
    from matplotlib so the in-browser colors match the standard maps exactly.
    """
    import matplotlib
    luts = {DEFAULT_CMAP: [_ramp_color(i / (n - 1)) for i in range(n)]}
    for name in ("viridis", "magma", "jet"):
        cmap = matplotlib.colormaps[name]
        luts[name] = [
            "#{:02x}{:02x}{:02x}".format(
                *(int(round(c * 255)) for c in cmap(i / (n - 1))[:3]))
            for i in range(n)
        ]
    return luts


def radius_from_reviews(reviews, lo=4, hi=18):
    """Map log10(reviews) into a pixel radius range."""
    t = (math.log10(max(reviews, 1)) - 2) / (5 - 2)   # 100..100k -> 0..1
    t = max(0.0, min(1.0, t))
    return lo + (hi - lo) * t


def build_map(rows, grid_path=None):
    # prefer_canvas: render 1k+ markers as one canvas element instead of one
    # SVG node per circle. Leaflet's SVG renderer chokes well before 2k markers
    # — pages can appear blank or hang. Canvas keeps it interactive at this size.
    # tiles=None + a control=False TileLayer adds the basemap without listing it
    # as a (non-removable) radio entry in the LayerControl.
    # Initial location uses the data centroid; fit_bounds at the end pins the
    # actual view to the data bbox, so the map opens framed regardless of city.
    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    m = folium.Map(
        location=[sum(lats) / len(lats), sum(lons) / len(lons)],
        tiles=None, control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("cartodbpositron", control=False).add_to(m)

    # One FeatureGroup per cuisine so the LayerControl can toggle them.
    cuisines = sorted({r["cuisine"] for r in rows})
    groups = {c: folium.FeatureGroup(name=c, show=True) for c in cuisines}

    cmaps = build_colormaps()
    ryg = cmaps[DEFAULT_CMAP]
    marker_idx = {}   # "lat,lon" -> LUT index, so JS can recolor on switch

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
        idx = color_index(bayes_to_t(r["bayes"]))
        marker_idx[f"{r['lat']:.6f},{r['lon']:.6f}"] = idx
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius_from_reviews(r["reviews"]),
            color=ryg[idx],
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
    m.get_root().html.add_child(folium.Element(
        _colormap_script(m.get_name(), cmaps, marker_idx, N_CMAP)))
    # On touch devices (no hover) a tap opens both the sticky tooltip and the
    # popup; with no mouseout the tooltip lingers behind the popup. The popup
    # carries identical content, so hide tooltips where hover is unavailable.
    m.get_root().header.add_child(folium.Element(
        "<style>@media (hover: none) { .leaflet-tooltip { display: none !important; } }</style>"
    ))
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


def _colormap_script(map_name, cmaps, idx_map, n):
    """JS for the legend's colormap <select>: recolor every marker + the legend bar.

    Markers live inside per-cuisine FeatureGroups; they're collected once at load
    (all groups start visible) into a persistent list, so switching a colormap
    recolors even cuisines that are currently toggled off. Each marker keeps its
    LUT index; switching just re-reads the chosen LUT at that index.

    Placeholders are substituted via str.replace to avoid escaping the many JS
    braces through str.format/f-strings.
    """
    tmpl = """
    <script>
    window.__cmap = (function () {
      var N = __N__, CMAPS = __CMAPS__, IDX = __IDX__, markers = [];
      function keyFor(ll) { return ll.lat.toFixed(6) + ',' + ll.lng.toFixed(6); }
      (function collect() {
        // Reference the map global via window[] (not a bare identifier): this
        // script is emitted before Folium's map-init script, so the name isn't
        // declared yet — a bare reference would throw instead of being caught
        // by the guard below, and we'd never retry.
        var m = window["__MAP__"];
        if (!m) { setTimeout(collect, 200); return; }
        var found = [];
        m.eachLayer(function (layer) {
          if (layer.eachLayer) layer.eachLayer(function (child) {
            if (child.setStyle && child.getLatLng) {
              var i = IDX[keyFor(child.getLatLng())];
              if (i !== undefined) found.push([child, i]);
            }
          });
        });
        if (!found.length) { setTimeout(collect, 200); return; }
        markers = found;
      })();
      function apply(name) {
        var lut = CMAPS[name];
        if (!lut) return;
        for (var j = 0; j < markers.length; j++) {
          var c = lut[markers[j][1]];
          markers[j][0].setStyle({ color: c, fillColor: c });
        }
        var bar = document.getElementById('__cmap_bar');
        if (bar) {
          var stops = [];
          for (var k = 0; k <= 6; k++) stops.push(lut[Math.round(k / 6 * (N - 1))]);
          bar.style.background = 'linear-gradient(to right,' + stops.join(',') + ')';
        }
      }
      return { apply: apply };
    })();
    </script>
    """
    return (tmpl
            .replace("__N__", str(n))
            .replace("__CMAPS__", json.dumps(cmaps))
            .replace("__IDX__", json.dumps(idx_map))
            .replace("__MAP__", map_name))


def _legend_html():
    """Floating legend in the bottom-left explaining marker color and size.

    The color stops mirror _ramp_color()'s ramp; the SVG circle radii
    mirror radius_from_reviews() at log10 = 2, 3, 4, 5 so the swatches match
    actual marker sizes on the map.
    """
    return """
    <div style="position: absolute; bottom: 28px; left: 16px; z-index: 1000;
                background: white; padding: 10px 12px; border-radius: 4px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.25);
                font-family: -apple-system, sans-serif; font-size: 12px;
                color: #222;">
      <div style="display: flex; align-items: center; justify-content: space-between;
                  gap: 10px; margin-bottom: 4px;">
        <span style="font-weight: 600;">Bayesian rating</span>
        <select id="__cmap_select" onchange="window.__cmap.apply(this.value)"
                style="font-size: 11px; padding: 1px 2px;">
          <option value="ryg">red→yellow→green</option>
          <option value="viridis">viridis</option>
          <option value="magma">magma</option>
          <option value="jet">jet</option>
        </select>
      </div>
      <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 12px;">
        <span style="font-variant-numeric: tabular-nums;">4.0</span>
        <div id="__cmap_bar" style="width: 150px; height: 12px;
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
    <script>
      // Inject a "Toggle all" button at the top of the LayerControl's overlay
      // list (top-right), right where the per-cuisine checkboxes live. The
      // control renders after this element, so poll until it exists. Folium
      // names its layer vars dynamically, so we drive the DOM checkboxes: if
      // any cuisine is on, turn all off; otherwise all on. The "scrape grid"
      // debug overlay is left alone.
      (function attachToggleAll() {
        var overlays = document.querySelector('.leaflet-control-layers-overlays');
        if (!overlays) { setTimeout(attachToggleAll, 150); return; }
        if (document.getElementById('__toggle_all_btn')) return;
        var btn = document.createElement('button');
        btn.id = '__toggle_all_btn';
        btn.type = 'button';
        btn.textContent = 'Toggle all';
        btn.style.cssText = 'width:100%; margin:2px 0 6px; padding:3px 6px;' +
          'font-size:12px; cursor:pointer; border:1px solid #888;' +
          'border-radius:3px; background:#f4f4f4;';
        btn.onclick = function () {
          var boxes = [];
          overlays.querySelectorAll('label').forEach(function (l) {
            if (l.textContent.trim() === 'scrape grid') return;
            var cb = l.querySelector('input[type=checkbox]');
            if (cb) boxes.push(cb);
          });
          if (!boxes.length) return;
          var anyOn = boxes.some(function (b) { return b.checked; });
          boxes.forEach(function (b) { if (b.checked === anyOn) b.click(); });
        };
        overlays.insertBefore(btn, overlays.firstChild);
      })();
    </script>
    """


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "munich_restaurants.csv"
    out_html = sys.argv[2] if len(sys.argv) > 2 else "munich_map.html"
    # Match the scraper's per-city naming: {city}_restaurants.csv -> {city}_grid.json.
    grid_json = (in_csv.replace("_restaurants.csv", "_grid.json")
                 if in_csv.endswith("_restaurants.csv") else "scrape_grid.json")

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
