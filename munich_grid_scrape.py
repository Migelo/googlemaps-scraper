#!/usr/bin/env python3
"""
Munich restaurant grid scraper (Google Places API v1, searchNearby).

Builds an N x N Cartesian grid over a square bounding box of side L (meters)
centered on a given lat/lon, then runs one Nearby Search per tile (subdividing
saturated tiles via an adaptive mesh), dedupes by place_id, and writes
restaurants with > MIN_REVIEWS rating counts to CSV.

Provide your API key via the GOOGLE_MAPS_API_KEY environment variable:
    export GOOGLE_MAPS_API_KEY="..."
    python munich_grid_scrape.py

Budget safety:
    --dry-run            estimate calls without spending any quota
    --max-calls N        hard stop the scrape before exceeding N API calls
"""

import os
import csv
import json
import time
import math
import sys
import argparse

import requests


class CallBudgetExceeded(Exception):
    """Raised to abort the recursive scrape once the call cap is reached."""


# ----------------------------- Parameters -----------------------------------
# Default center is Munich Marienplatz; override with --center or --city.
DEFAULT_CENTER_LAT = 48.1370339
DEFAULT_CENTER_LON = 11.5758134
L = 5000.0                   # bounding-box side length in meters (default 5 km)
N = 4                        # tiles per side (default 4 -> 16 tiles)
MIN_REVIEWS = 100            # client-side filter (API has no min-reviews param)
PLACE_TYPE = "restaurant"    # included type for the search
MAX_PER_CALL = 20            # API hard cap per nearby search
SLEEP_BETWEEN = 0.2          # politeness / rate-limit spacing in seconds
MAX_DEPTH = 4                # recursion cap: a tile splits at most this many times
SPLIT_AT = MAX_PER_CALL      # a tile returning >= this many results subdivides

# Meters-per-degree latitude is ~constant (Earth's circumference / 360).
M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat):
    """Meters per degree of longitude at the given latitude."""
    return 111_320.0 * math.cos(math.radians(lat))


# Preset city centers — values are (lat, lon) of a central square / landmark.
CITIES = {
    "munich":  (48.1370339, 11.5758134),    # Marienplatz
    "berlin":  (52.5200,    13.4050),       # Alexanderplatz
    "vienna":  (48.2082,    16.3738),       # Stephansdom
    "hamburg": (53.5511,     9.9937),       # Rathausmarkt
}


def cell_radius(edge_m):
    """Search radius that covers a square cell of the given edge: its half-diagonal."""
    return (edge_m * math.sqrt(2)) / 2.0


def build_grid(center_lat, center_lon, side_m, n):
    """Return a list of (lat, lon, edge_m) square-cell descriptors.

    The box spans [-side/2, +side/2] around the center on each axis, cut into
    n equal columns and rows. Each cell is described by its center and edge
    length; the search radius is derived from the edge via cell_radius(). A cell
    described this way can be subdivided uniformly into four children, which is
    what the adaptive mesh relies on.
    """
    edge_m = side_m / n
    half = side_m / 2.0
    m_per_deg_lon_center = m_per_deg_lon(center_lat)

    cells = []
    for row in range(n):
        for col in range(n):
            # offset of this cell center from the box center, in meters
            dx = -half + edge_m * (col + 0.5)   # east-west
            dy = -half + edge_m * (row + 0.5)   # north-south
            lat = center_lat + dy / M_PER_DEG_LAT
            lon = center_lon + dx / m_per_deg_lon_center
            cells.append((lat, lon, edge_m))
    return cells


def subdivide(lat, lon, edge_m):
    """Split one square cell into four children, each half the edge length.

    Children centers sit at +/- quarter-edge from the parent center on each
    axis, converting meters to degrees with the appropriate per-axis scale.
    """
    child_edge = edge_m / 2.0
    q = edge_m / 4.0  # quarter-edge offset to each child center
    m_lon = m_per_deg_lon(lat)
    children = []
    for sy in (-1, 1):
        for sx in (-1, 1):
            clat = lat + (sy * q) / M_PER_DEG_LAT
            clon = lon + (sx * q) / m_lon
            children.append((clat, clon, child_edge))
    return children


def search_tile(lat, lon, radius, api_key):
    """One Nearby Search call. Returns a list of raw place dicts."""
    url = "https://places.googleapis.com/v1/places:searchNearby"
    # Field mask keeps us in the cheaper billing tier: ids, names, ratings,
    # review counts, location, types, price level. No reviews/contact fields.
    field_mask = ",".join([
        "places.id",
        "places.displayName",
        "places.rating",
        "places.userRatingCount",
        "places.location",
        "places.types",
        "places.priceLevel",
    ])
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    body = {
        "includedTypes": [PLACE_TYPE],
        "maxResultCount": MAX_PER_CALL,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": radius,
            }
        },
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json().get("places", [])


def tile_hash(lat, lon, edge_m):
    """Stable identifier for a tile, rounded to dodge float drift across runs."""
    return f"{lat:.6f}:{lon:.6f}:{edge_m:.1f}"


def load_scanned(path):
    """Return the set of scanned-tile hashes from path, or empty if absent/corrupt."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object at top level, got {type(data).__name__}")
        return set(data.get("tiles", []))
    except (json.JSONDecodeError, ValueError, OSError, AttributeError) as e:
        print(f"  (could not read {path}: {e}; starting fresh)")
        return set()


def save_scanned(path, scanned):
    """Atomically persist the scanned-tile set."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"tiles": sorted(scanned)}, f, indent=2)
    os.replace(tmp, path)


CSV_FIELDS = ["place_id", "name", "rating", "user_rating_count",
              "lat", "lon", "types", "price_level"]


def save_csv(path, seen):
    """Atomically write the dedup dict to CSV. No-op if seen is empty."""
    if not seen:
        return
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(seen.values())
    os.replace(tmp, path)


def load_csv(path):
    """Load an existing scraper CSV into a {place_id: row} dict for resume."""
    if not os.path.exists(path):
        return {}
    seen = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("place_id")
            if pid:
                seen[pid] = row
    return seen


def ingest(places, seen):
    """Merge raw API places into the dedup dict keyed by place_id."""
    for p in places:
        pid = p.get("id")
        if pid and pid not in seen:
            seen[pid] = {
                "place_id": pid,
                "name": p.get("displayName", {}).get("text", ""),
                "rating": p.get("rating"),
                "user_rating_count": p.get("userRatingCount"),
                "lat": p.get("location", {}).get("latitude"),
                "lon": p.get("location", {}).get("longitude"),
                "types": "|".join(p.get("types", [])),
                "price_level": p.get("priceLevel"),
            }


def harvest(lat, lon, edge_m, api_key, seen, stats, scanned,
            depth=0, label="root", max_calls=None, max_depth=MAX_DEPTH):
    """Query one cell; if it saturates the result cap, subdivide and recurse.

    Returns True iff this tile is "fully scanned" — i.e., we have all places
    that this geometry can yield. That is true for a non-saturated leaf and for
    a saturated node whose four children are all fully scanned. False otherwise
    (HTTP error, saturated-at-max-depth, or any descendant failed).

    The scanned set is consulted before each call: if the tile's hash is in it,
    the call is skipped (its data is assumed already in `seen` from a prior run).

    max_calls and max_depth behave as before.
    """
    h = tile_hash(lat, lon, edge_m)
    if h in scanned:
        stats["skipped"] += 1
        return True

    if max_calls is not None and stats["calls"] >= max_calls:
        raise CallBudgetExceeded(label)

    radius = cell_radius(edge_m)
    try:
        places = search_tile(lat, lon, radius, api_key)
    except requests.HTTPError as e:
        print(f"  [{label}] HTTP error {e}; skipping")
        return False

    stats["calls"] += 1
    stats["raw"] += len(places)
    ingest(places, seen)
    saturated = len(places) >= SPLIT_AT
    time.sleep(SLEEP_BETWEEN)

    if saturated and depth < max_depth:
        print(f"  [{label}] {len(places)} results (saturated, edge {edge_m:.0f} m) "
              f"-> subdividing")
        children_ok = True
        for k, (clat, clon, cedge) in enumerate(subdivide(lat, lon, edge_m)):
            ok = harvest(clat, clon, cedge, api_key, seen, stats, scanned,
                         depth + 1, label=f"{label}.{k}",
                         max_calls=max_calls, max_depth=max_depth)
            children_ok = children_ok and ok
        if children_ok:
            scanned.add(h)
            return True
        return False

    note = ""
    if saturated:  # at max-depth: cannot recurse further, undercount likely
        note = "  <-- still saturated at max depth, may be undercounted"
        stats["capped_leaves"] += 1
    print(f"  [{label}] {len(places)} results, {len(seen)} unique so far{note}")
    if saturated:
        return False
    scanned.add(h)
    return True


def estimate_calls(n_seed, max_depth):
    """Bracket the call count for a dry run without spending any quota.

    A real count depends on local density, which we can't know without calling.
    So we report the meaningful bounds:
      floor   = no tile saturates -> exactly the seed-grid count
      ceiling = every tile saturates to full depth -> a 4-ary tree per seed,
                summing 4^0 + 4^1 + ... + 4^max_depth calls per seed tile.
    Real runs land between these, usually much closer to the floor.
    """
    per_seed_ceiling = sum(4 ** d for d in range(max_depth + 1))
    return n_seed, n_seed * per_seed_ceiling


def parse_args():
    p = argparse.ArgumentParser(description="Places (New) grid scraper")
    p.add_argument("--dry-run", action="store_true",
                   help="estimate calls and preview the grid without spending quota")
    p.add_argument("--max-calls", type=int, default=None,
                   help="hard stop: abort before exceeding this many API calls")
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                   help=f"adaptive-mesh subdivision depth (default {MAX_DEPTH}; "
                        f"0 disables refinement)")
    p.add_argument("--city", choices=sorted(CITIES.keys()), default=None,
                   help="preset city center; also defaults --csv and --scanned-db "
                        "to {city}_restaurants.csv / {city}_scanned_tiles.json")
    p.add_argument("--center", default=None,
                   help='manual center as "LAT,LON" (overrides --city)')
    p.add_argument("--side", type=float, default=L,
                   help=f"bounding-box side length in meters (default {int(L)})")
    p.add_argument("--grid", type=int, default=N,
                   help=f"tiles per side (default {N})")
    p.add_argument("--csv", default=None,
                   help="CSV output file (default depends on --city)")
    p.add_argument("--scanned-db", default=None,
                   help="JSON file tracking scanned tile hashes (default depends on --city)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore existing CSV/scanned-db and start fresh")
    args = p.parse_args()

    # Resolve center: --center > --city > default Munich.
    if args.center:
        try:
            lat, lon = (float(x) for x in args.center.split(","))
        except ValueError:
            p.error("--center must be 'LAT,LON'")
        args.center_lat, args.center_lon = lat, lon
        slug = "custom"
    elif args.city:
        args.center_lat, args.center_lon = CITIES[args.city]
        slug = args.city
    else:
        args.center_lat, args.center_lon = DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON
        slug = "munich"

    # Default output paths from the slug. munich keeps its legacy filenames so
    # the existing CSV is picked up automatically.
    if args.csv is None:
        args.csv = "munich_restaurants.csv" if slug == "munich" else f"{slug}_restaurants.csv"
    if args.scanned_db is None:
        args.scanned_db = (
            "scanned_tiles.json" if slug == "munich" else f"{slug}_scanned_tiles.json"
        )
    return args


def main():
    args = parse_args()
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    tiles = build_grid(args.center_lat, args.center_lon, args.side, args.grid)
    seed_radius = cell_radius(args.side / args.grid)
    L_arg, N_arg = args.side, args.grid

    # Derive grid-preview JSON path from the CSV name for per-city outputs.
    if args.csv.endswith("_restaurants.csv"):
        grid_json_path = args.csv.replace("_restaurants.csv", "_grid.json")
    else:
        grid_json_path = "scrape_grid.json"

    print(f"Grid: {N_arg}x{N_arg} = {len(tiles)} seed tiles over a {L_arg:.0f} m box "
          f"centered at ({args.center_lat:.5f}, {args.center_lon:.5f})")
    print(f"Each seed tile: {L_arg/N_arg:.0f} m square, search radius {seed_radius:.0f} m")
    print(f"Adaptive mesh: saturated tiles split into 4, up to depth {args.max_depth} "
          f"(min edge {L_arg/N_arg/(2**args.max_depth):.0f} m)"
          + ("  [refinement OFF]" if args.max_depth == 0 else "") + "\n")

    # Preview the seed grid so it can be inspected before spending any quota.
    for i, (lat, lon, edge) in enumerate(tiles):
        print(f"  tile {i:2d}: ({lat:.5f}, {lon:.5f})  edge={edge:.0f} m  "
              f"r={cell_radius(edge):.0f} m")

    floor, ceiling = estimate_calls(len(tiles), args.max_depth)
    print(f"\nEstimated API calls: floor {floor} (no saturation), "
          f"ceiling {ceiling} (full saturation to depth {args.max_depth}).")
    print("Real runs fall between these, usually near the floor.")
    if args.max_calls is not None:
        print(f"Budget cap: --max-calls {args.max_calls} "
              f"({'binds' if args.max_calls < ceiling else 'above ceiling, will not bind'}).")

    if args.dry_run:
        print("\n[dry-run] No API calls made. Remove --dry-run to execute.")
        with open(grid_json_path, "w") as f:
            json.dump([{"lat": t[0], "lon": t[1], "edge": t[2],
                        "radius": cell_radius(t[2])} for t in tiles], f, indent=2)
        print(f"Wrote {grid_json_path}")
        return

    if not api_key:
        print("\nGOOGLE_MAPS_API_KEY not set. Grid generated but no calls made.")
        print("Set the key and re-run to execute the scrape, or use --dry-run.")
        with open(grid_json_path, "w") as f:
            json.dump([{"lat": t[0], "lon": t[1], "edge": t[2],
                        "radius": cell_radius(t[2])} for t in tiles], f, indent=2)
        print(f"Wrote {grid_json_path}")
        return

    out_csv = args.csv
    scanned_db = args.scanned_db
    seen = {} if args.no_resume else load_csv(out_csv)
    scanned = set() if args.no_resume else load_scanned(scanned_db)
    if seen or scanned:
        print(f"Resume: {len(seen)} known places in {out_csv}, "
              f"{len(scanned)} fully-scanned tiles in {scanned_db}\n")
    # Consistency check: scanned-db without a matching CSV means the next
    # scrape will skip those tiles and produce empty/incomplete output.
    if scanned and not seen:
        print(f"WARNING: {scanned_db} marks {len(scanned)} tiles fully scanned, "
              f"but {out_csv} has no places. Those tiles will be SKIPPED with no "
              f"data. Pass --no-resume to ignore the scanned-db, or restore the CSV.\n")

    stats = {"calls": 0, "raw": 0, "capped_leaves": 0, "skipped": 0}
    aborted = False
    try:
        for i, (lat, lon, edge) in enumerate(tiles):
            harvest(lat, lon, edge, api_key, seen, stats, scanned,
                    depth=0, label=f"t{i}", max_calls=args.max_calls,
                    max_depth=args.max_depth)
    except CallBudgetExceeded as e:
        aborted = True
        print(f"\nBudget cap of {args.max_calls} calls reached at [{e}]. "
              f"Stopping early and saving partial results.")
    finally:
        # Order matters: persist places BEFORE marking tiles scanned. If we
        # saved the scanned marker first and then crashed during the CSV
        # write, a future run would skip those tiles and lose the places.
        save_csv(out_csv, seen)
        save_scanned(scanned_db, scanned)

    filtered_count = sum(
        1 for r in seen.values()
        if int(float(r.get("user_rating_count") or 0)) > MIN_REVIEWS
    )
    print(f"\nTotal API calls this run: {stats['calls']}"
          + (f" (capped at {args.max_calls})" if aborted else ""))
    if stats["skipped"]:
        print(f"Tiles skipped via {scanned_db}: {stats['skipped']}")
    print(f"Raw results across new calls: {stats['raw']}")
    print(f"Unique places after dedup: {len(seen)}")
    print(f"With > {MIN_REVIEWS} reviews: {filtered_count}")
    if aborted:
        print("NOTE: partial scrape; coverage is incomplete. Re-run to resume — "
              "scanned tiles will be skipped automatically.")
    if stats["capped_leaves"]:
        print(f"WARNING: {stats['capped_leaves']} cell(s) still saturated at "
              f"max depth; raise --max-depth or --n for full coverage.")
    if seen:
        print(f"Wrote {out_csv} ({len(seen)} rows)")
    print(f"Wrote {scanned_db} ({len(scanned)} tiles marked fully scanned)")


if __name__ == "__main__":
    main()
