"""Render the PR42 charging map from canonical OA diagnostics and cached vectors.

The original seven-panel composition, city centres and 8 km buffers are retained.
This full-domain areal choropleth uses reduced, uniform fill opacity beneath
roads (the GIS skill's areal-map exception); low access remains visibly red.
No model calculations, historical archives or active manuscript figures are
modified. Supply a new output directory for each reviewed candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from shapely.geometry import box


REGIONS = ("HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans")
CITIES = {
    "Inverness": (-4.2247, 57.4778), "Aberdeen": (-2.0954, 57.1497),
    "Glasgow": (-4.2518, 55.8642), "Edinburgh": (-3.1883, 55.9533),
    "Dumfries": (-3.6110, 55.0699), "Dundee": (-2.9707, 56.4620),
}
COLOURS = ("#D73027", "#F46D43", "#FDAE61", "#FEE090", "#FFFFBF",
           "#E0F3F8", "#ABD9E9", "#74ADD1", "#4575B4", "#313695")
TARGET_CRS = "EPSG:27700"
OA_ALPHA = 0.70
LAND = "#eeeDE8"
WATER = "#edf4f8"


def sha256(path: Path) -> str:
    """Return the SHA-256 of complete file bytes."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def diagnostics(frame: gpd.GeoDataFrame, value: str | None = None) -> dict[str, Any]:
    """Describe spatial and numeric inputs before display-only processing."""
    result: dict[str, Any] = {"rows": len(frame), "crs": str(frame.crs),
        "bounds": frame.total_bounds.tolist(), "geometry_types": frame.geom_type.value_counts().to_dict(),
        "empty_geometries": int(frame.geometry.is_empty.sum()),
        "missing_geometries": int(frame.geometry.isna().sum()),
        "invalid_geometries": int((~frame.geometry.is_valid).sum())}
    if value is not None:
        numbers = pd.to_numeric(frame[value], errors="coerce")
        finite = numbers[np.isfinite(numbers)]
        result["value"] = {"column": value, "finite": len(finite),
            "missing_or_nonfinite": int((~np.isfinite(numbers)).sum()),
            "zero": int(finite.eq(0).sum()), "min": float(finite.min()),
            "max": float(finite.max()), "quantiles": {str(key): float(val) for key, val in finite.quantile([0, .25, .5, .75, 1]).items()}}
    return result


def marker_area(count: np.ndarray | float) -> np.ndarray | float:
    """Map reported charging-point count monotonically to marker area in pt²."""
    return 9.0 + 2.0 * count


def subset(frame: gpd.GeoDataFrame, extent: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Select intersecting vector features without changing source geometry."""
    return frame.iloc[frame.sindex.query(box(*extent), predicate="intersects")]


def scale_bar(ax: plt.Axes, extent: tuple[float, float, float, float], length_m: float) -> None:
    """Draw a metric scale bar in British National Grid coordinates."""
    xmin, ymin, xmax, ymax = extent
    x, y = xmin + .06 * (xmax - xmin), ymin + .06 * (ymax - ymin)
    height = .014 * (ymax - ymin)
    ax.plot([x, x + length_m], [y, y], color="#222222", linewidth=2, zorder=20)
    for tick in [x, x + length_m]:
        ax.plot([tick, tick], [y - height / 2, y + height / 2], color="#222222", linewidth=1, zorder=20)
    ax.text(x + length_m / 2, y + height * .65, f"{length_m / 1000:g} km",
        ha="center", va="bottom", fontsize=11, color="#222222", zorder=21,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 1.5})


def main() -> None:
    """Verify input identities and render an isolated seven-panel figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-root", type=Path, required=True)
    parser.add_argument("--area-manifest", type=Path, required=True)
    parser.add_argument("--basemap-dir", type=Path, required=True)
    parser.add_argument("--chargers", type=Path, required=True)
    parser.add_argument("--region-boundaries", type=Path, required=True)
    parser.add_argument("--expected-oa-count", type=int, required=True)
    parser.add_argument("--expected-charger-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    basemap_manifest_path = args.basemap_dir / "vector_basemap_manifest.json"
    basemap_manifest = json.loads(basemap_manifest_path.read_text())
    area_manifest = json.loads(args.area_manifest.read_text())
    paths = [args.area_root / f"{region}.gpkg" for region in REGIONS]
    paths += [args.basemap_dir / name for name in ("admin_0_countries.gpkg", "major_roads.gpkg", "populated_places.gpkg")]
    paths += [args.chargers, args.region_boundaries, args.area_manifest, basemap_manifest_path, Path(__file__).resolve()]
    before_hashes = {str(path.resolve()): sha256(path) for path in paths}
    for name, expected in basemap_manifest["output_sha256"].items():
        if sha256(args.basemap_dir / name) != expected:
            raise ValueError(f"Cached vector basemap hash mismatch: {name}")
    if sha256(args.chargers) != "27b03392ff413a35be78194ea0967d24610fe8b862a4fc57f90cf5d8faf767e3":
        raise ValueError("Charger input differs from frozen v4 OCM source")
    if sha256(args.region_boundaries) != "2fd9b3653007250f13e0394f1879090dc393fda3322986b5f59ae63b899f5059":
        raise ValueError("Regional boundary input differs from frozen v4 source")
    input_diagnostics: dict[str, Any] = {}
    areas = []
    for region in REGIONS:
        path = args.area_root / f"{region}.gpkg"
        key = f"{region}/2022/enhanced_charging_infrastructure.gpkg"
        if sha256(path) != area_manifest["outputs_sha256"][key]:
            raise ValueError(f"Corrected area layer differs from its manifest: {region}")
        area = gpd.read_file(path, columns=["geo_code", "accessibility_score", "geometry"])
        input_diagnostics[region] = diagnostics(area, "accessibility_score")
        if area.crs is None or area.geo_code.isna().any() or not area.geo_code.is_unique:
            raise ValueError(f"Missing CRS/OA identity or duplicate OA identity: {region}")
        if area.geometry.isna().any() or area.geometry.is_empty.any():
            raise ValueError(f"Missing or empty canonical OA geometry: {region}")
        values = pd.to_numeric(area.accessibility_score, errors="coerce")
        if not np.isfinite(values).all() or not values.between(.2 - 1e-12, 1 + 1e-12).all():
            raise ValueError(f"Accessibility score outside published scale: {region}")
        area = area.to_crs(TARGET_CRS)
        # Repairs are restricted to the in-memory display geometry; source files
        # and the area diagnostics are left unchanged and hashed before/after.
        invalid = ~area.geometry.is_valid
        area.loc[invalid, "geometry"] = area.loc[invalid, "geometry"].make_valid()
        if not area.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError(f"Display geometry repair produced non-areal features: {region}")
        area["region"] = region
        areas.append(area)
    oa = gpd.GeoDataFrame(pd.concat(areas, ignore_index=True), crs=TARGET_CRS)
    if len(oa) != args.expected_oa_count or not oa.geo_code.is_unique:
        raise ValueError("Incomplete or overlapping regional OA population")
    display_wgs = oa.to_crs(4326).total_bounds
    if not (-9 < display_wgs[0] < display_wgs[2] < 1 and 54 < display_wgs[1] < display_wgs[3] < 62):
        raise ValueError("Canonical Scottish OA extent is implausible")
    countries = gpd.read_file(args.basemap_dir / "admin_0_countries.gpkg").to_crs(TARGET_CRS)
    roads = gpd.read_file(args.basemap_dir / "major_roads.gpkg").to_crs(TARGET_CRS)
    places = gpd.read_file(args.basemap_dir / "populated_places.gpkg").to_crs(TARGET_CRS)
    for name, frame in [("countries", countries), ("roads", roads), ("places", places)]:
        input_diagnostics[name] = diagnostics(frame)
    chargers = gpd.read_file(args.chargers)
    input_diagnostics["chargers"] = diagnostics(chargers, "num_points")
    if chargers.crs is None or not chargers.geom_type.eq("Point").all() or not chargers.geometry.is_valid.all():
        raise ValueError("Invalid charger point geometry")
    if chargers.ocm_id.isna().any() or not chargers.ocm_id.is_unique:
        raise ValueError("Missing or duplicate OCM location identity")
    chargers = chargers.to_crs(TARGET_CRS)
    retained_regions = gpd.read_file(args.region_boundaries).to_crs(TARGET_CRS)
    input_diagnostics["region_boundaries"] = diagnostics(retained_regions)
    if not set(REGIONS).issubset(set(retained_regions.Region)):
        raise ValueError("Regional boundary population is incomplete")
    retained_by_region = []
    retained_charger_counts = {}
    for region in REGIONS:
        # Match the exact regional clip used by the v4 physical-site ledger and
        # corrected area-diagnostic producer, rather than substituting OA coastlines.
        points = gpd.clip(chargers, retained_regions[retained_regions.Region.eq(region)]).copy()
        retained_charger_counts[region] = len(points)
        retained_by_region.append(points)
    retained = gpd.GeoDataFrame(pd.concat(retained_by_region, ignore_index=True), crs=TARGET_CRS)
    if not retained.ocm_id.is_unique or len(retained) != args.expected_charger_count:
        raise ValueError("Map physical-site population differs from regional ledger")
    numbers = pd.to_numeric(retained.num_points, errors="coerce")
    known_mask = np.isfinite(numbers) & numbers.gt(0)
    known = retained.loc[known_mask].copy()
    known["points_reported"] = numbers[known_mask]
    unknown = retained.loc[~known_mask].copy()
    region_boundaries = oa[["region", "geometry"]].dissolve(by="region").boundary
    city = gpd.GeoDataFrame({"name": list(CITIES)}, geometry=gpd.points_from_xy(
        [p[0] for p in CITIES.values()], [p[1] for p in CITIES.values()]), crs=4326).to_crs(TARGET_CRS).set_index("name")
    xmin, ymin, xmax, ymax = oa.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    national_extent = (xmin - .055 * dx, ymin - .035 * dy, xmax + .055 * dx, ymax + .035 * dy)
    cmap = mcolors.LinearSegmentedColormap.from_list("paper1_accessibility", COLOURS, N=256)
    norm = mcolors.Normalize(.2, 1.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": .65})
    fig = plt.figure(figsize=(14, 11.8), facecolor="white")
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 2.15, 1],
        left=.025, right=.975, bottom=.145, top=.96, wspace=.035, hspace=.12)
    national_ax = fig.add_subplot(grid[:, 1])
    panels: list[dict[str, Any]] = []

    def draw_panel(ax: plt.Axes, extent: tuple[float, float, float, float], title: str, national: bool = False) -> None:
        ax.set_facecolor(WATER)
        visible_oa, visible_roads = subset(oa, extent), subset(roads, extent)
        if visible_oa.empty:
            raise ValueError(f"Empty map panel: {title}")
        subset(countries, extent).plot(ax=ax, color=LAND, edgecolor="#8b9195", linewidth=.30, zorder=1)
        visible_oa.plot(ax=ax, column="accessibility_score", cmap=cmap, norm=norm,
            alpha=OA_ALPHA, linewidth=0, edgecolor="none", antialiased=False,
            rasterized=True, zorder=2)
        if len(visible_roads):
            visible_roads.plot(ax=ax, color="#ffffff", linewidth=.35 if national else 1.0, alpha=.68, rasterized=True, zorder=3)
            visible_roads.plot(ax=ax, color="#72777c", linewidth=.13 if national else .38, alpha=.78, rasterized=True, zorder=4)
        region_boundaries.plot(ax=ax, color="#44484c", linewidth=.44 if national else .40, rasterized=True, zorder=5)
        local_known, local_unknown = subset(known, extent), subset(unknown, extent)
        if len(local_known):
            ax.scatter(local_known.geometry.x, local_known.geometry.y,
                s=marker_area(local_known.points_reported.to_numpy()), c="#fff03b", edgecolor="#45452e",
                linewidth=.38, alpha=.85, zorder=7)
        if len(local_unknown):
            ax.scatter(local_unknown.geometry.x, local_unknown.geometry.y,
                s=13, c="#fff03b", marker="^", edgecolor="#45452e", linewidth=.35, alpha=.85, zorder=7)
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#bdc4ca"); spine.set_linewidth(.55)
        ax.set_title(title, fontsize=17, weight="bold", loc="left", pad=8)
        scale_bar(ax, extent, 100000 if national else 4000)
        panels.append({"panel": title, "extent_bng": list(extent), "visible_oa_features": len(visible_oa),
            "visible_road_features": len(visible_roads), "known_count_chargers": len(local_known),
            "unknown_count_chargers": len(local_unknown), "accessibility_min": float(visible_oa.accessibility_score.min()),
            "accessibility_max": float(visible_oa.accessibility_score.max())})

    draw_panel(national_ax, national_extent, "Scotland", True)
    label_offsets = {
        "Inverness": (-35, 35), "Aberdeen": (35, 20), "Glasgow": (-35, -25),
        "Edinburgh": (35, -22), "Dumfries": (12, -32), "Dundee": (32, 30)
    }
    for name, point in city.geometry.items():
        ox, oy = label_offsets[name]
        # Prominent high-contrast city marker (white circle with thick black outline + inner center dot)
        national_ax.scatter([point.x], [point.y], s=80, facecolor="white", edgecolor="#111111", linewidth=2.2, zorder=15)
        national_ax.scatter([point.x], [point.y], s=18, facecolor="#111111", edgecolor="none", zorder=16)
        # Prominent leader line and label box connecting city location to name
        national_ax.annotate(name, xy=(point.x, point.y), xytext=(ox, oy), textcoords="offset points",
            fontsize=13, weight="bold", ha="left" if ox >= 0 else "right", va="center", zorder=18,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#111111", "alpha": 0.96, "linewidth": 1.2},
            arrowprops={"arrowstyle": "-", "color": "#111111", "linewidth": 2.0, "shrinkA": 2, "shrinkB": 5})
    island_places = places[places.NAME.isin(["Lerwick", "Kirkwall", "Stornoway"])]
    for place in island_places.itertuples():
        national_ax.text(place.geometry.x + 5500, place.geometry.y + 5500, place.NAME,
            fontsize=9.5, color="#39454e", zorder=10,
            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
    national_ax.annotate("N", xy=(.08,.94), xytext=(.08,.88), xycoords="axes fraction",
        ha="center", va="center", fontsize=13, weight="bold",
        arrowprops={"arrowstyle":"-|>", "color":"#30373c", "lw":1.1}, zorder=20)
    for name, location in [("Inverness", (0,0)), ("Aberdeen", (0,2)), ("Glasgow", (1,0)),
                           ("Edinburgh", (1,2)), ("Dumfries", (2,0)), ("Dundee", (2,2))]:
        point = city.loc[name].geometry
        draw_panel(fig.add_subplot(grid[location]), (point.x - 8000, point.y - 8000, point.x + 8000, point.y + 8000), name)
    colour_ax = fig.add_axes([.07,.074,.39,.019])
    # Match the displayed polygon colour at the same opacity over neutral land.
    colour_rgba = cmap(np.linspace(0,1,256))
    colour_rgba[:,:3] = OA_ALPHA * colour_rgba[:,:3] + (1-OA_ALPHA) * np.array(mcolors.to_rgb(LAND))
    legend_cmap = mcolors.ListedColormap(colour_rgba)
    colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=legend_cmap), cax=colour_ax, orientation="horizontal", ticks=[.2,.4,.6,.8,1])
    colourbar.ax.tick_params(labelsize=13, length=3)
    colourbar.outline.set_linewidth(.5)
    colourbar.set_label("Charging accessibility score (poor → good)", fontsize=13, labelpad=5)
    legend_handles = [Line2D([], [], linestyle="none", marker="o", markersize=float(np.sqrt(marker_area(n))),
        markerfacecolor="#fff03b", markeredgecolor="#45452e", markeredgewidth=.6, label=str(n)) for n in [1,5,20]]
    legend_handles.append(Line2D([], [], linestyle="none", marker="^", markersize=np.sqrt(13),
        markerfacecolor="#fff03b", markeredgecolor="#45452e", markeredgewidth=.6, label="Unreported"))
    fig.legend(handles=legend_handles, title="Reported charging points at each location", loc="lower center",
        bbox_to_anchor=(.728,.051), ncol=4, frameon=False, fontsize=12, title_fontsize=13,
        handletextpad=.45, columnspacing=1.1, borderaxespad=0)
    fig.text(.5,.014,"8 km city buffers  •  OA fill opacity 70%; roads drawn above  •  Context: OpenStreetMap contributors; Natural Earth",
        ha="center", va="bottom", fontsize=10, color="#60666b")
    png = out / "charging_map.png"
    pdf = out / "charging_map.pdf"
    fig.savefig(png, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    after_hashes = {str(path.resolve()): sha256(path) for path in paths}
    if before_hashes != after_hashes:
        raise ValueError("A map input changed during rendering")
    report = {"command": sys.argv, "status": "generated; visual inspection required separately",
        "crs": TARGET_CRS, "expected_oa_count": args.expected_oa_count, "displayed_oa_count": len(oa),
        "displayed_oa_keys_unique": bool(oa.geo_code.is_unique), "score_range": [float(oa.accessibility_score.min()),float(oa.accessibility_score.max())],
        "area_manifest": area_manifest, "input_diagnostics": input_diagnostics,
        "display_geometry_policy": "make_valid only for invalid in-memory polygons; original files unchanged",
        "opacity": OA_ALPHA, "opacity_policy": "uniform reduced opacity for complete areal zone map; roads above overlay; no score-based visual filtering",
        "charger_input_locations": len(chargers), "retained_charger_locations": len(retained),
        "excluded_charger_locations_outside_frozen_regional_boundaries": len(chargers)-len(retained),
        "retained_charger_locations_by_region": retained_charger_counts,
        "charger_geography_rule": "Exact gpd.clip of archived OCM points by frozen regional boundaries, matching v4 physical-site and corrected ancillary-area producers",
        "retained_known_point_count_locations": len(known), "retained_unreported_point_count_locations": len(unknown),
        "known_point_count_range": [float(known.points_reported.min()),float(known.points_reported.max())],
        "marker_area_rule_pt_squared": "9 + 2 * reported_num_points; missing/nonpositive/nonfinite count shown as fixed-size yellow triangle, no imputation",
        "panels": panels, "input_sha256_before": before_hashes, "input_sha256_after": after_hashes,
        "output_sha256": {png.name:sha256(png),pdf.name:sha256(pdf)},
        "suggested_caption": "Charging accessibility across all retained Scottish output areas (score 0.2–1.0; red denotes poor access and blue good access), with six city windows extending 8 km from their centres. OA fills use uniform 70% opacity beneath the road context. Yellow circles indicate geographically retained charging locations, with size showing reported charging-point count; yellow triangles indicate locations with unreported point count. Cached OpenStreetMap roads and Natural Earth land and place features provide geographic context."}
    (out / "map_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    pd.DataFrame(panels).to_csv(out / "panel_diagnostics.csv", index=False)
    print(json.dumps({"output":str(png),"oa_count":len(oa),"retained_chargers":len(retained),"known_counts":len(known),"unknown_counts":len(unknown),"panels":panels},indent=2),flush=True)


if __name__ == "__main__":
    main()
