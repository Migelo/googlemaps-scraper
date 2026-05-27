# Munich Restaurant Rating-Divergence Pipeline

Collect Munich restaurants from the Google Places API, group them by cuisine,
and measure how different each cuisine's star-rating distribution is from the
others using Jensen-Shannon divergence. The output is a pairwise divergence
heatmap.

The project is two stages joined by a CSV:

```
munich_grid_scrape.py  ->  munich_restaurants.csv  ->  divergence_pipeline.py  ->  munich_jsd_heatmap.png
   (collect)                  (shared contract)            (analyze + plot)
```

## Why it works this way

The Google Places Nearby Search endpoint returns at most 20 results per call and
offers no "minimum reviews" filter. You cannot ask it for "every Munich
restaurant with 100+ reviews" in one query. So the scraper tiles the city into a
grid of small circular searches, queries each, and deduplicates by `place_id`.
Dense areas that would exceed the 20-result cap are detected and recursively
subdivided (an adaptive mesh), so coverage stays complete without wasting calls
on empty outskirts. The review-count filter is applied client-side after
collection.

## Requirements

```
pip install requests scipy numpy matplotlib
```

A Google Maps Platform API key with the **Places API (New)** enabled and billing
active on the project. The scraper calls the v1 `places:searchNearby` endpoint,
which is the new API, not the legacy one.

## Quick start

```bash
# 1. See the grid and a call-count estimate without spending anything
python munich_grid_scrape.py --dry-run

# 2. Run the real scrape with a safety cap on API calls
export GOOGLE_MAPS_API_KEY="your_key_here"
python munich_grid_scrape.py --max-calls 2000

# 3. Analyze the resulting CSV and produce the heatmap
python divergence_pipeline.py
```

## Common usage walkthrough

A start-to-finish path from a fresh machine to a full dataset. Each step de-risks
the next, so you never point a large paid run at an untested setup.

### 1. Set up the API

In the Google Cloud Console: create or select a project, enable billing on it
(the Places API requires an active billing account even while you stay within the
free tier), then enable the **Places API (New)** specifically. Create an API key
under Credentials and restrict it to the Places API (New) so a leaked key cannot
run up charges elsewhere. Then export it:

```bash
export GOOGLE_MAPS_API_KEY="your_key_here"
pip install requests scipy numpy matplotlib
```

### 2. Test that the key and endpoint work

Before running the scraper, confirm the key, the enabled API, and the field mask
with a single hand-made call. One call is effectively free and tells you
immediately if auth or setup is wrong:

```bash
curl -s -X POST 'https://places.googleapis.com/v1/places:searchNearby' \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $GOOGLE_MAPS_API_KEY" \
  -H "X-Goog-FieldMask: places.displayName,places.rating,places.userRatingCount" \
  -d '{
    "includedTypes": ["restaurant"],
    "maxResultCount": 5,
    "locationRestriction": {
      "circle": {
        "center": {"latitude": 48.1370339, "longitude": 11.5758134},
        "radius": 500.0
      }
    }
  }'
```

A JSON list of nearby restaurants means you are ready. A `403`/`PERMISSION_DENIED`
usually means the Places API (New) is not enabled or the key restriction is wrong;
a `400` usually means a malformed field mask.

### 3. Dry run (no spend)

See the grid and the call-count bracket without touching the API:

```bash
python munich_grid_scrape.py --dry-run
```

This prints the floor (seed-grid count) and ceiling (full saturation) estimates
and writes `munich_grid.json`. Use it to sanity-check `L`, `N`, and your intended
`--max-calls` before spending anything.

### 4. Small run WITHOUT adaptive mesh

Do a first real-data pass with refinement disabled via `--max-depth 0`. This makes
exactly one call per seed tile (16 for the defaults), so it is cheap, fast, and
fully predictable (floor equals ceiling). It will undercount dense tiles and print
`still saturated at max depth` warnings, which is expected and is in fact the
signal that those tiles need refinement:

```bash
python munich_grid_scrape.py --max-depth 0 --max-calls 50
python divergence_pipeline.py            # confirm the CSV flows through cleanly
```

The point of this step is to validate the end-to-end plumbing (live API ->
dedup -> CSV -> heatmap) on a tiny, known call count, not to get complete data.

### 5. Full run WITH adaptive mesh

Re-enable refinement (the default `--max-depth 4`), widen the box if you want
greater Munich (for example `L = 12000` at the top of the file), and run with a
budget cap as a backstop. The dense central tiles now subdivide to capture
everything the 20-result cap would otherwise hide:

```bash
# optionally edit munich_grid_scrape.py: set L = 12000
python munich_grid_scrape.py --dry-run            # re-check the new estimate first
python munich_grid_scrape.py --max-calls 2000     # the real, complete scrape
python divergence_pipeline.py                     # final heatmap
```

If the run finishes without a budget-cap message and without
`still saturated at max depth` warnings, coverage is complete. If you see those
warnings, raise `--max-depth` (or `N`) and re-run.

## The scraper: `munich_grid_scrape.py`

Builds an `N x N` grid over a square box of side `L` meters centered on a
lat/lon, runs one Nearby Search per tile, subdivides saturated tiles, dedupes,
filters by review count, and writes `munich_restaurants.csv`.

Parameters live at the top of the file:

- `CENTER_LAT`, `CENTER_LON`: grid center. Default is Marienplatz.
- `L`: bounding-box side length in meters. Default 5000 (5 km). Raise this to
  reach outer districts; roughly 12000 to 15000 covers greater Munich.
- `N`: seed tiles per side. Default 4 (a 16-tile grid). Prefer letting the
  adaptive mesh handle density rather than cranking `N`, since `N` multiplies the
  baseline call count whether or not tiles are dense.
- `MIN_REVIEWS`: keep only places with strictly more than this many reviews.
  Default 100.
- `MAX_DEPTH`: default adaptive-mesh subdivision depth. Default 4. Override per run
  with `--max-depth`; `--max-depth 0` disables the mesh (one call per seed tile).
- `SLEEP_BETWEEN`: spacing between calls in seconds. Raise it if you hit HTTP 429.

Command-line flags:

- `--dry-run`: preview the grid and print a floor/ceiling estimate of API calls,
  then exit without making any calls or spending quota.
- `--max-calls N`: hard stop. The budget is checked before every call, so the
  scrape aborts the instant it would exceed `N`, saves whatever it has collected,
  and labels the CSV as incomplete.
- `--max-depth D`: adaptive-mesh depth for this run, overriding the `MAX_DEPTH`
  default. `--max-depth 0` disables refinement (one call per seed tile).

How the adaptive mesh works: a cell is described by its center and edge length.
If a query returns a full page (20 results) the cell almost certainly holds more
than the API will return, so it splits into four children of half the edge and
each is re-queried. This repeats until cells come back under the cap or
`MAX_DEPTH` is hit. Search radius for a cell is its half-diagonal, so circles
fully cover their square cells with slight overlap that dedup absorbs.

Output CSV columns: `place_id, name, rating, user_rating_count, lat, lon, types,
price_level`.

## The analysis: `divergence_pipeline.py`

Reads the CSV, classifies each restaurant into a cuisine, builds a rating
distribution per cuisine, computes pairwise divergence, and writes the heatmap.

```bash
python divergence_pipeline.py [input_csv] [output_png]
# defaults: munich_restaurants.csv  ->  munich_jsd_heatmap.png
```

Key choices:

- **Cuisine classification** maps Google `types` to a label using priority-ordered
  rules, so specific types win over broad ones. A place tagged
  `japanese_restaurant|sushi_restaurant|asian_restaurant` becomes Japanese, not
  Asian. Unclassifiable places are counted and dropped, not silently discarded.
- **Distributions** are histograms over half-star rating bands (3.5 to 5.0) with
  Laplace smoothing (+0.5 per bin) so no bin is exactly zero. Cuisines with fewer
  than 3 places are dropped so distributions are not built on noise.
- **Jensen-Shannon over KL**: JS is symmetric (clean distance-like matrix) and
  stays finite even when a rating band is empty, which KL does not. The code
  squares SciPy's JS distance to report divergence.

The rating bins were tuned for a small sample spanning 3.5 to 5.0. On a fuller,
denser dataset you will likely want to widen them; see the open items below.

## Cost and the free tier

Google replaced its old universal $200 monthly credit (as of March 2025) with
per-SKU free monthly usage caps under Essentials / Pro / Enterprise tiers. The
field mask in this scraper requests location, rating, review count, types, and
price level, which puts Nearby Search in the **Pro** tier. You are billed at the
highest tier your requested fields touch. The Pro free cap is on the order of a
few thousand calls per month.

A one-off 5 km Munich run will likely make only a few hundred calls and stay free,
but a larger box or repeated runs in the same month can cross into paid usage.
Exact dollar rates change, so verify current numbers on Google's pricing page
before a big run, and set a billing budget alert plus a daily quota cap on the key
in the Cloud Console. The `--dry-run` and `--max-calls` flags are your in-script
guardrails on top of that.

## Files

- `munich_grid_scrape.py` - the scraper (grid, adaptive mesh, dry-run, budget cap).
- `divergence_pipeline.py` - cuisine classification, divergence, heatmap.
- `make_sample_csv.py` - builds a `munich_restaurants.csv` from a real 60-restaurant
  sample collected earlier, so the analysis stage can be demonstrated without a
  live key. See the note below.
- `munich_restaurants.csv` - the shared data contract between the two stages.
- `munich_jsd_heatmap.png` - the final output.

## Important caveats

- **The included heatmap is from a sample, not a full scrape.** The sample CSV was
  assembled from ~60 real Munich restaurants collected through a separate tool, not
  by running the scraper against Google. The collection geometry, adaptive
  subdivision, dedup, and budget logic are all tested offline (including a
  simulation with a synthetic 1,500-point density field), but the live HTTP round
  trip to Google has not been exercised. On your first real run, confirm the auth
  header, the response shape, and that the field mask bills as described.
- **A capped run is partial and not resumable.** Re-running after a `--max-calls`
  abort is safe (dedup absorbs overlap) but starts over from the first tile rather
  than resuming, re-spending calls on already-covered tiles. Fine for a one-off;
  for repeated or very large runs, persist the seen-set and tile frontier first.
- **The 100+ review threshold is client-side**, applied after collection, because
  the API has no minimum-reviews parameter.

## Possible next steps

- Auto-adapt the rating bin edges to the data range for larger datasets.
- Add a price-level divergence variant as a second output.
- Add a resumable checkpoint (persist seen-set and frontier) for large runs.
- Add a category-faceting second pass to catch places the grid misses.
