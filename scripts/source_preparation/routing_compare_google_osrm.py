"""OSRM vs Google Routing API distance cross-check for YTRA-D-26-01251 (Comment 3.2).

Reviewer 3 noted that route circuity is especially important for short-range
vehicles and asked for more explicit treatment of uncertainty in the synthetic
OD inputs. The OD matrices record flows between zone centroids; before routing,
each flow is spatially disaggregated with the jittering method of Lovelace,
Felix and Carlino (2022), splitting every zone-to-zone flow into multiple
origin-destination point pairs in proportion to trip counts so that routing
produces a realistic diversity of road paths rather than one centroid-to-
centroid path per zone pair. All production routing uses the OSRM driving
profile on OpenStreetMap. As an independent ground-truth check, a sample of
commute routes in the SESTRAN (Edinburgh) region was re-routed with the Google
Routing API (`routing_compare/google_routes.gpkg`, 166 driving routes over 33
Data Zone OD pairs).

This script joins each Google route to its OSRM counterpart in
`data/OD_routes/all_routes_sestran.gpkg` and reports agreement statistics.
The two files share no route identifier (`geo_code2` in the OSRM table is a
numeric destination id, not a Data Zone code), so the join is geometric: a
Google route is matched to the OSRM jittered route with the same origin Data
Zone (`geo_code1`) whose stored request points (`origin_lat/lng`,
`dest_lat/lng`) lie nearest to the Google route geometry's first and last
vertices, accepted when both endpoints agree within `--tolerance` metres
(default 300 m, allowing for Google's snap-to-road displacement of the
requested points).

Reported statistics (printed and written to
`output/revision_validation/routing_compare_google_osrm.csv` per matched
route, with a JSON summary alongside):

* matched route count and match-distance distribution;
* Pearson correlation between OSRM and Google driving distances;
* mean and median OSRM/Google distance ratio;
* mean absolute percentage difference;
* Google routed-vs-Euclidean circuity on all 166 routes (computed from the
  route geometry endpoints), as context for the manuscript's 20-40% circuity
  statement.

The 34 GB OSRM GeoPackage is scanned once (attributes only, no geometry) and
the candidate rows are cached under
`output/revision_validation/cache/osrm_sestran_google_origins.csv`, so reruns
are fast. Geometry blobs are parsed with the standard library only; neither
GeoPandas nor Fiona is required.

Run:

    python code/routing_compare_google_osrm.py \
        [--google routing_compare/google_routes.gpkg] \
        [--osrm data/OD_routes/all_routes_sestran.gpkg] \
        [--results output/revision_validation] \
        [--tolerance 300]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOOGLE = REPO_ROOT / "routing_compare" / "google_routes.gpkg"
DEFAULT_OSRM = REPO_ROOT / "data" / "OD_routes" / "all_routes_sestran.gpkg"
DEFAULT_RESULTS = REPO_ROOT / "output" / "revision_validation"
DEFAULT_TOLERANCE_M = 300.0

CANDIDATE_FIELDS = (
    "od_id",
    "geo_code1",
    "geo_code2",
    "origin_lat",
    "origin_lng",
    "dest_lat",
    "dest_lng",
    "purpose",
    "original_car_trips",
    "osrm_driving_distance_km",
    "osrm_driving_duration_minutes",
)


def _gpkg_linestring_endpoints(blob: bytes) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (first, last) (lon, lat) vertices of a GeoPackage LINESTRING blob."""
    if blob[0:2] != b"GP":
        raise ValueError("not a GeoPackage geometry blob")
    flags = blob[3]
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0x07]
    off = 8 + envelope_bytes
    order = "<" if blob[off] == 1 else ">"
    gtype = struct.unpack_from(order + "I", blob, off + 1)[0]
    if gtype % 1000 != 2:
        raise ValueError(f"expected LINESTRING, got WKB type {gtype}")
    npts = struct.unpack_from(order + "I", blob, off + 5)[0]
    dim = {0: 2, 1: 3, 2: 3, 3: 4}[gtype // 1000]
    first = struct.unpack_from(order + "dd", blob, off + 9)
    last = struct.unpack_from(order + "dd", blob, off + 9 + (npts - 1) * dim * 8)
    return first, last


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    h = (
        math.sin((phi2 - phi1) / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(h))


def load_google_routes(path: Path) -> List[Dict]:
    """Read the 166 Google commute routes with their geometry endpoints."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT fid, geo_code1, geo_code2, driving_distance_km, geom "
            "FROM commute_routes WHERE driving_distance_km IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    routes = []
    for fid, geo1, geo2, google_km, geom in rows:
        start, end = _gpkg_linestring_endpoints(geom)
        routes.append(
            {
                "fid": fid,
                "geo_code1": geo1,
                "geo_code2": geo2,
                "google_km": float(google_km),
                "start": start,
                "end": end,
                "euclid_km": _haversine_m(start[0], start[1], end[0], end[1]) / 1000.0,
            }
        )
    return routes


def load_osrm_candidates(path: Path, origins: Sequence[str], cache_path: Path) -> List[Dict]:
    """Scan the OSRM GeoPackage (attributes only) for rows whose origin Data
    Zone appears in the Google sample; cache the result as CSV."""
    if cache_path.exists():
        with cache_path.open() as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.execute("PRAGMA mmap_size=36000000000")
    placeholders = ",".join("?" * len(origins))
    query = (
        "SELECT od_id, geo_code1, CAST(geo_code2 AS TEXT), origin_lat, origin_lng, "
        "dest_lat, dest_lng, purpose, original_car_trips, driving_distance_km, "
        f"driving_duration_minutes FROM routes WHERE geo_code1 IN ({placeholders})"
    )
    t0 = time.time()
    rows = []
    try:
        for row in con.execute(query, list(origins)):
            rows.append(dict(zip(CANDIDATE_FIELDS, row)))
    finally:
        con.close()
    with cache_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"scanned {path.name} in {time.time() - t0:.0f}s; cached {len(rows)} candidate rows")
    return rows


def match_routes(
    google_routes: List[Dict], candidates: List[Dict], tolerance_m: float
) -> List[Dict]:
    """Match each Google route to the OSRM jittered route with the nearest
    request points (same origin Data Zone, both endpoints within tolerance)."""
    by_origin: Dict[str, List[Dict]] = {}
    for cand in candidates:
        by_origin.setdefault(cand["geo_code1"], []).append(cand)
    matches = []
    for route in google_routes:
        best = None
        for cand in by_origin.get(route["geo_code1"], ()):
            d_start = _haversine_m(
                route["start"][0], route["start"][1],
                float(cand["origin_lng"]), float(cand["origin_lat"]),
            )
            d_end = _haversine_m(
                route["end"][0], route["end"][1],
                float(cand["dest_lng"]), float(cand["dest_lat"]),
            )
            cost = max(d_start, d_end)
            if best is None or cost < best[0]:
                best = (cost, cand)
        if best is not None and best[0] <= tolerance_m:
            cand = best[1]
            osrm_km = float(cand["osrm_driving_distance_km"])
            matches.append(
                {
                    "google_fid": route["fid"],
                    "geo_code1": route["geo_code1"],
                    "geo_code2_google": route["geo_code2"],
                    "osrm_od_id": cand["od_id"],
                    "osrm_purpose": cand["purpose"],
                    "match_cost_m": round(best[0], 1),
                    "euclid_km": round(route["euclid_km"], 3),
                    "google_km": route["google_km"],
                    "osrm_km": osrm_km,
                    "ratio_osrm_over_google": round(osrm_km / route["google_km"], 4),
                    "abs_pct_diff": round(
                        abs(osrm_km - route["google_km"]) / route["google_km"] * 100.0, 2
                    ),
                }
            )
    return matches


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarise(google_routes: List[Dict], matches: List[Dict]) -> Dict:
    google_kms = [m["google_km"] for m in matches]
    osrm_kms = [m["osrm_km"] for m in matches]
    ratios = [m["ratio_osrm_over_google"] for m in matches]
    pct_diffs = [m["abs_pct_diff"] for m in matches]
    circuity = [r["google_km"] / r["euclid_km"] for r in google_routes if r["euclid_km"] > 0.1]
    return {
        "google_routes_total": len(google_routes),
        "google_od_pairs": len({(r["geo_code1"], r["geo_code2"]) for r in google_routes}),
        "matched_routes": len(matches),
        "matched_osrm_rows_unique": len({m["osrm_od_id"] for m in matches}),
        "match_cost_m_median": _median([m["match_cost_m"] for m in matches]) if matches else None,
        "pearson_r_osrm_vs_google": round(_pearson(osrm_kms, google_kms), 4) if len(matches) > 2 else None,
        "mean_ratio_osrm_over_google": round(sum(ratios) / len(ratios), 4) if matches else None,
        "median_ratio_osrm_over_google": round(_median(ratios), 4) if matches else None,
        "mean_abs_pct_diff": round(sum(pct_diffs) / len(pct_diffs), 2) if matches else None,
        "median_abs_pct_diff": round(_median(pct_diffs), 2) if matches else None,
        "google_circuity_mean": round(sum(circuity) / len(circuity), 3),
        "google_circuity_median": round(_median(circuity), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--google", type=Path, default=DEFAULT_GOOGLE)
    parser.add_argument("--osrm", type=Path, default=DEFAULT_OSRM)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_M)
    args = parser.parse_args(argv)

    google_routes = load_google_routes(args.google)
    origins = sorted({r["geo_code1"] for r in google_routes})
    cache_path = args.results / "cache" / "osrm_sestran_google_origins.csv"
    candidates = load_osrm_candidates(args.osrm, origins, cache_path)
    matches = match_routes(google_routes, candidates, args.tolerance)
    summary = summarise(google_routes, matches)

    args.results.mkdir(parents=True, exist_ok=True)
    out_csv = args.results / "routing_compare_google_osrm.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(matches[0].keys()) if matches else
                                ["google_fid"])
        writer.writeheader()
        writer.writerows(matches)
    out_json = args.results / "routing_compare_google_osrm.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"per-route comparison written to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
