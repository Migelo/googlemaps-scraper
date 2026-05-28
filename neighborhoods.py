#!/usr/bin/env python3
"""
DBSCAN-discovered restaurant neighborhoods.

Projects (lat, lon) to UTM zone 32N meters, clusters with DBSCAN at
eps=150 m, min_samples=8. For each cluster, reports count, dominant
cuisines, mean rating, median price tier, and centroid (lat, lon).

Pass --geocode to look up a human-readable name via the Google Geocoding
API (~$5/1k calls, so ~$0.10 for typical Munich runs). Off by default —
just prints lat/lon centroids so no API spend on a vanilla run.

Usage:
    python neighborhoods.py [input_csv]
    python neighborhoods.py --geocode    (requires GOOGLE_MAPS_API_KEY)
"""

import sys
import os
import csv
import argparse
import collections

import numpy as np
import requests
from prettytable import PrettyTable
from pyproj import Transformer
from sklearn.cluster import DBSCAN

from cuisine import classify, MIN_REVIEWS

# UTM zone 32N covers Munich; using a projected CRS gives DBSCAN real meters.
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

EPS_M = 80         # neighborhood scale: ~one short city block in dense Munich
MIN_SAMPLES = 10


def load(csv_path):
    rows = []
    for r in csv.DictReader(open(csv_path, newline="")):
        try:
            rating = float(r["rating"])
            reviews = int(float(r["user_rating_count"] or 0))
            lat = float(r["lat"])
            lon = float(r["lon"])
        except (ValueError, TypeError, KeyError):
            continue
        if reviews <= MIN_REVIEWS or not (0 < rating <= 5):
            continue
        rows.append({
            "name": (r.get("name") or "").strip() or "(unnamed)",
            "rating": rating, "reviews": reviews, "lat": lat, "lon": lon,
            "cuisine": classify(r.get("types", "")) or "Other",
            "price": (r.get("price_level") or "").replace("PRICE_LEVEL_", "") or "-",
        })
    return rows


def cluster(rows):
    """Add a 'cluster' int to each row (-1 = noise)."""
    coords = np.array([[r["lon"], r["lat"]] for r in rows])
    xs, ys = to_utm.transform(coords[:, 0], coords[:, 1])
    XY = np.column_stack([xs, ys])
    labels = DBSCAN(eps=EPS_M, min_samples=MIN_SAMPLES).fit_predict(XY)
    for r, lab, x, y in zip(rows, labels, xs, ys):
        r["cluster"] = int(lab)
        r["_utm"] = (x, y)
    return labels


def summarize(rows):
    by_cluster = collections.defaultdict(list)
    for r in rows:
        by_cluster[r["cluster"]].append(r)
    summaries = []
    for c, members in by_cluster.items():
        if c == -1:
            continue  # noise — not a cluster
        xs = np.array([m["_utm"][0] for m in members])
        ys = np.array([m["_utm"][1] for m in members])
        cx, cy = xs.mean(), ys.mean()
        clon, clat = to_wgs.transform(cx, cy)
        cuisines = collections.Counter(m["cuisine"] for m in members).most_common(3)
        prices = collections.Counter(m["price"] for m in members
                                     if m["price"] != "-").most_common(1)
        ratings = np.array([m["rating"] for m in members])
        summaries.append({
            "cluster": c,
            "n": len(members),
            "lat": clat, "lon": clon,
            "top_cuisines": ", ".join(f"{name}({n})" for name, n in cuisines),
            "modal_price": prices[0][0] if prices else "—",
            "mean_rating": float(ratings.mean()),
        })
    summaries.sort(key=lambda s: -s["n"])
    return summaries


def reverse_geocode(lat, lon, api_key):
    """Return the most-locality-y component for (lat, lon) via Google Geocoding."""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "key": api_key,
              "result_type": "sublocality|neighborhood|sublocality_level_1"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return f"(geocode error: {e})"
    if data.get("status") != "OK" or not data.get("results"):
        # Fall back: any result, prefer short formatted name.
        params.pop("result_type")
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            return "(no result)"
    formatted = data["results"][0].get("formatted_address", "")
    # Strip the trailing ", Germany" and the postcode to keep the label tight.
    parts = [p.strip() for p in formatted.split(",")]
    return ", ".join(parts[:2]) if parts else formatted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="munich_restaurants.csv")
    p.add_argument("--geocode", action="store_true",
                   help="reverse-geocode each cluster centroid via the Google Geocoding API")
    args = p.parse_args()

    rows = load(args.csv)
    if not rows:
        print(f"No usable rows in {args.csv}.")
        sys.exit(1)

    labels = cluster(rows)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    print(f"Loaded {len(rows)} restaurants from {args.csv}")
    print(f"DBSCAN eps={EPS_M} m, min_samples={MIN_SAMPLES} → "
          f"{n_clusters} clusters, {n_noise} noise points")

    summaries = summarize(rows)
    if args.geocode:
        key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not key:
            print("ERROR: --geocode requires GOOGLE_MAPS_API_KEY in env.")
            sys.exit(1)
        print(f"Reverse-geocoding {len(summaries)} centroids...")
        for s in summaries:
            s["label"] = reverse_geocode(s["lat"], s["lon"], key)

    t = PrettyTable()
    cols = ["Cluster", "n", "Centroid (lat, lon)", "Top cuisines",
            "Modal price", "Mean rating"]
    if args.geocode:
        cols.insert(2, "Label")
    t.field_names = cols
    t.align["Top cuisines"] = "l"
    if args.geocode:
        t.align["Label"] = "l"
    for s in summaries:
        row = [s["cluster"], s["n"], f"({s['lat']:.4f}, {s['lon']:.4f})",
               s["top_cuisines"], s["modal_price"], f"{s['mean_rating']:.2f}"]
        if args.geocode:
            row.insert(2, s.get("label", ""))
        t.add_row(row)
    print(t)


if __name__ == "__main__":
    main()
