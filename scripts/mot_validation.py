# code/mot_validation.py
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCOTLAND_PC_AREAS: frozenset[str] = frozenset(
    ["AB", "DD", "DG", "EH", "FK", "G", "HS", "IV", "KA", "KW", "KY", "ML", "PA", "PH", "TD", "ZE"]
)

REGIONS: tuple[str, ...] = (
    "HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans"
)

TARGET_CRS = "EPSG:27700"
SCORE_COL = "final_adoption_propensity"


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def _collapse_postcode_areas(
    gdf: gpd.GeoDataFrame,
    *,
    value_cols: list[str],
) -> gpd.GeoDataFrame:
    """Collapse multi-part postcode geometries to one row per ``pc_area``.

    The postcode polygon layer contains many disjoint geometry parts for the same
    postcode area (for example island groups). MOT counts are already aggregated
    to postcode-area level, so keeping one row per geometry fragment duplicates
    the same counts and later creates many-to-many merges.
    """
    required = {"pc_area", "geometry", *value_cols}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Postcode GeoDataFrame missing required columns: {sorted(missing)}")

    validators = gdf[["pc_area", *value_cols]].drop_duplicates()
    duplicated_pc = validators["pc_area"].duplicated(keep=False)
    conflicting = validators.loc[duplicated_pc].groupby("pc_area")[value_cols].nunique()
    conflicting = conflicting[(conflicting > 1).any(axis=1)]
    if not conflicting.empty:
        raise ValueError(
            "Postcode areas have conflicting aggregated values across geometry parts: "
            f"{conflicting.head(10).to_dict(orient='index')}"
        )

    collapsed = gdf.dissolve(by="pc_area", aggfunc="first").reset_index()
    return gpd.GeoDataFrame(collapsed, crs=gdf.crs)

def load_2011_census(output_dir: Path) -> gpd.GeoDataFrame:
    """Load 2011 OA adoption scores by concatenating 7 regional gpkg files.

    Args:
        output_dir: Directory containing one subdirectory per region, each
                    with a ``conversion_potential.gpkg`` file.

    Returns:
        GeoDataFrame in EPSG:27700 with columns including
        ``geo_code``, ``final_adoption_propensity``, and ``geometry``.

    Raises:
        FileNotFoundError: If any regional gpkg is absent.
    """
    frames: list[gpd.GeoDataFrame] = []
    for region in REGIONS:
        gpkg_path = output_dir / region / "conversion_potential.gpkg"
        if not gpkg_path.exists():
            raise FileNotFoundError(
                f"conversion_potential.gpkg not found for region '{region}': {gpkg_path}"
            )
        gdf = gpd.read_file(gpkg_path)
        if gdf.crs is None or gdf.crs.to_epsg() != 27700:
            gdf = gdf.to_crs(TARGET_CRS)
        frames.append(gdf)
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=TARGET_CRS)
    return combined


def load_mot(
    mot_gpkg: Path,
    postcode_polygons: Path,
    scotland_only: bool = True,
) -> gpd.GeoDataFrame:
    """Load MOT EV counts joined to postcode area polygon geometries.

    Args:
        mot_gpkg: Path to ``ev_distribution.gpkg`` (counts per postcode area).
        postcode_polygons: Path to ``postcode_polygons.gpkg``
                           (layer ``postcode_area`` with polygon geometries).
        scotland_only: If True, filter result to ``SCOTLAND_PC_AREAS``.

    Returns:
        GeoDataFrame in EPSG:27700 with columns: ``pc_area``, ``bev_count``,
        ``plugin_count``, ``geometry``.
    """
    counts = gpd.read_file(mot_gpkg)
    required_cols = {"pc_area", "bev_count", "plugin_count"}
    missing = required_cols - set(counts.columns)
    if missing:
        raise ValueError(f"MOT counts file missing required columns: {missing}")

    polygons = gpd.read_file(postcode_polygons, layer="postcode_area")
    if polygons.crs is None or polygons.crs.to_epsg() != 27700:
        polygons = polygons.to_crs(TARGET_CRS)

    result = polygons[["pc_area", "geometry"]].merge(
        counts[["pc_area", "bev_count", "plugin_count"]],
        on="pc_area",
        how="inner",
    )
    result = gpd.GeoDataFrame(result, crs=TARGET_CRS)

    if scotland_only:
        result = result[result["pc_area"].isin(SCOTLAND_PC_AREAS)].copy()

    result = _collapse_postcode_areas(
        result.reset_index(drop=True),
        value_cols=["bev_count", "plugin_count"],
    )
    return result.reset_index(drop=True)


def load_cumulative_bev_counts(
    yearly_csv: Path,
    postcode_polygons: Path,
    up_to_year: int,
    scotland_only: bool = True,
) -> gpd.GeoDataFrame:
    """Compute cumulative BEV counts up to *up_to_year* per postcode area.

    Reads the pre-extracted yearly CSV produced by
    ``extract_yearly_bev_counts.py`` and sums BEV registrations from the
    earliest year through *up_to_year* (inclusive) for each postcode area.

    Args:
        yearly_csv: Path to ``yearly_bev_counts_scotland.csv`` with columns
                    ``pc_area``, ``first_use_year``, ``bev_count``.
        postcode_polygons: Path to ``postcode_polygons.gpkg``
                           (layer ``postcode_area`` with polygon geometries).
        up_to_year: Include all registrations with ``first_use_year <= up_to_year``.
        scotland_only: If True, filter to ``SCOTLAND_PC_AREAS``.

    Returns:
        GeoDataFrame in EPSG:27700 with columns: ``pc_area``, ``bev_count``,
        ``geometry``.
    """
    df = pd.read_csv(yearly_csv)
    required = {"pc_area", "first_use_year", "bev_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Yearly CSV missing required columns: {missing}")

    # Cumulative: sum all years up to the cutoff
    df = df[df["first_use_year"] <= up_to_year]
    counts = df.groupby("pc_area")["bev_count"].sum().reset_index()

    if scotland_only:
        counts = counts[counts["pc_area"].isin(SCOTLAND_PC_AREAS)].copy()

    # Join with polygon geometries
    polygons = gpd.read_file(postcode_polygons, layer="postcode_area")
    if polygons.crs is None or polygons.crs.to_epsg() != 27700:
        polygons = polygons.to_crs(TARGET_CRS)

    result = polygons[["pc_area", "geometry"]].merge(
        counts, on="pc_area", how="inner",
    )
    result = gpd.GeoDataFrame(result, crs=TARGET_CRS)
    result = _collapse_postcode_areas(
        result.reset_index(drop=True),
        value_cols=["bev_count"],
    )
    return result.reset_index(drop=True)


def load_yearly_bev_counts(
    yearly_csv: Path,
    postcode_polygons: Path,
    year_from: int,
    year_to: int,
    scotland_only: bool = True,
) -> gpd.GeoDataFrame:
    """Compute BEV counts for a specific *first_use_year* window per postcode area.

    Unlike :func:`load_cumulative_bev_counts`, this function sums only registrations
    whose ``first_use_year`` falls within ``[year_from, year_to]`` (inclusive).
    This produces a **flow** measure — new BEVs entering the fleet in the given
    window — rather than a cumulative stock figure.

    .. note::
        UK vehicles are exempt from MOT testing for the first 3 years.  The
        underlying ``yearly_bev_counts_scotland.csv`` is derived from MOT test
        records, so recent years (typically the last 2–3 before the dataset
        cut-off) are severely under-counted.  Use a window that ends at least
        3 years before the data cut-off for reliable counts.

    Args:
        yearly_csv: Path to ``yearly_bev_counts_scotland.csv`` with columns
                    ``pc_area``, ``first_use_year``, ``bev_count``.
        postcode_polygons: Path to ``postcode_polygons.gpkg``
                           (layer ``postcode_area`` with polygon geometries).
        year_from: Lower bound of the year window (inclusive).
        year_to: Upper bound of the year window (inclusive).
        scotland_only: If True, filter to ``SCOTLAND_PC_AREAS``.

    Returns:
        GeoDataFrame in EPSG:27700 with columns: ``pc_area``, ``bev_count``,
        ``geometry``.

    Raises:
        ValueError: If the CSV is missing required columns or if no rows fall
                    within the requested year window.
    """
    df = pd.read_csv(yearly_csv)
    required = {"pc_area", "first_use_year", "bev_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Yearly CSV missing required columns: {missing}")

    window = df[(df["first_use_year"] >= year_from) & (df["first_use_year"] <= year_to)]
    if window.empty:
        raise ValueError(
            f"No rows in yearly CSV with first_use_year in [{year_from}, {year_to}]"
        )

    counts = window.groupby("pc_area")["bev_count"].sum().reset_index()

    if scotland_only:
        counts = counts[counts["pc_area"].isin(SCOTLAND_PC_AREAS)].copy()

    polygons = gpd.read_file(postcode_polygons, layer="postcode_area")
    if polygons.crs is None or polygons.crs.to_epsg() != 27700:
        polygons = polygons.to_crs(TARGET_CRS)

    result = polygons[["pc_area", "geometry"]].merge(
        counts, on="pc_area", how="inner",
    )
    result = gpd.GeoDataFrame(result, crs=TARGET_CRS)
    result = _collapse_postcode_areas(
        result.reset_index(drop=True),
        value_cols=["bev_count"],
    )
    return result.reset_index(drop=True)


def aggregate_census_to_postcode(
    census_gdf: gpd.GeoDataFrame,
    postcode_gdf: gpd.GeoDataFrame,
    score_col: str,
) -> pd.DataFrame:
    """Sum OA-level adoption scores within each postcode area polygon.

    Args:
        census_gdf: OA-level GeoDataFrame with ``score_col`` and geometry.
                    Reprojected to EPSG:27700 if needed.
        postcode_gdf: Postcode area GeoDataFrame (output of ``load_mot()``)
                      with ``pc_area`` and polygon geometry in EPSG:27700.
        score_col: Column name to aggregate (e.g. ``final_adoption_propensity``).

    Returns:
        DataFrame with columns: ``pc_area``, ``score_sum``, ``oa_count``.
    """
    if score_col not in census_gdf.columns:
        raise ValueError(f"Score column '{score_col}' not found in census_gdf")

    if census_gdf.crs is None or census_gdf.crs.to_epsg() != 27700:
        census_gdf = census_gdf.to_crs(TARGET_CRS)

    census_small = census_gdf[[score_col, "geometry"]].copy()
    postcode_small = postcode_gdf[["pc_area", "geometry"]].copy()

    overlaid = gpd.overlay(census_small, postcode_small, how="intersection")
    if len(overlaid) == 0:
        raise ValueError("No intersection found between census OAs and postcode polygons — check CRS alignment")

    result = (
        overlaid.groupby("pc_area")[score_col]
        .agg(score_sum="sum", oa_count="count")
        .reset_index()
    )
    return result


def build_validation_table(
    census_by_pc: pd.DataFrame,
    mot_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Merge census postcode aggregates with MOT BEV counts.

    Args:
        census_by_pc: DataFrame with ``pc_area``, ``score_sum``, ``oa_count``
                      (output of ``aggregate_census_to_postcode()``).
        mot_gdf: GeoDataFrame with ``pc_area``, ``bev_count``, ``geometry``
                 (output of ``load_mot()``).

    Returns:
        GeoDataFrame with columns: ``pc_area``, ``score_sum``, ``bev_count``,
        ``oa_count``, ``geometry``.
    """
    if census_by_pc["pc_area"].duplicated().any():
        dupes = census_by_pc.loc[census_by_pc["pc_area"].duplicated(), "pc_area"].tolist()[:10]
        raise ValueError(f"census_by_pc must be unique by pc_area; duplicates include {dupes}")

    if mot_gdf["pc_area"].duplicated().any():
        dupes = mot_gdf.loc[mot_gdf["pc_area"].duplicated(), "pc_area"].tolist()[:10]
        raise ValueError(f"mot_gdf must be unique by pc_area; duplicates include {dupes}")

    merged = mot_gdf[["pc_area", "bev_count", "geometry"]].merge(
        census_by_pc[["pc_area", "score_sum", "oa_count"]],
        on="pc_area",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise ValueError(
            "No common pc_area values found between census_by_pc and mot_gdf — check inputs"
        )
    return gpd.GeoDataFrame(merged, crs=mot_gdf.crs)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str = "pc_area",
    title: str | None = None,
    save_path: Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    font_scale: float = 1.0,
) -> tuple[Figure, Axes]:
    """Scatter plot with per-area labels and a linear fit with R² annotation.

    Args:
        df: DataFrame containing ``x_col``, ``y_col``, and ``label_col``.
        x_col: Column for x-axis (e.g. ``score_sum``).
        y_col: Column for y-axis (e.g. ``bev_count``).
        label_col: Column used to label each point (default ``pc_area``).
        title: Optional plot title.
        save_path: If provided, save figure to this path at 300 dpi.

    Returns:
        ``(fig, ax)`` tuple for inline display in Quarto.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if df.empty or df[[x_col, y_col]].dropna().empty:
        ax.text(
            0.5, 0.5, "No data available for this combination",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=int(round(13 * font_scale)), color="gray",
        )
        ax.set_title(
            title or f"{x_col} vs {y_col}",
            fontsize=int(round(14 * font_scale)), fontweight="bold",
        )
        return fig, ax

    unique_labels = df[label_col].astype(str).unique()
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))

    x_vals = df[x_col].to_numpy(dtype=float)
    y_vals = df[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_range = np.ptp(x_vals[mask]) if mask.any() and np.ptp(x_vals[mask]) != 0 else 1.0
    y_range = np.ptp(y_vals[mask]) if mask.any() and np.ptp(y_vals[mask]) != 0 else 1.0
    dx, dy = 0.02 * x_range, 0.02 * y_range

    for i, label in enumerate(unique_labels):
        subset = df[df[label_col] == label]
        ax.scatter(
            subset[x_col], subset[y_col],
            color=cmap(i), s=80, edgecolor="k", alpha=0.85,
        )
        for _, row in subset.iterrows():
            ax.text(
                row[x_col] + dx, row[y_col] + dy, str(row[label_col]),
                fontsize=int(round(11 * font_scale)), color=cmap(i), weight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

    if mask.sum() >= 2 and np.ptp(x_vals[mask]) > 0:
        m, c = np.polyfit(x_vals[mask], y_vals[mask], 1)
        xs = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
        ax.plot(xs, m * xs + c, color="black", lw=2, linestyle="--", label="Linear fit")
        y_pred = m * x_vals[mask] + c
        ss_res = np.sum((y_vals[mask] - y_pred) ** 2)
        ss_tot = np.sum((y_vals[mask] - np.mean(y_vals[mask])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fit_text = f"y = {m:.3f}x + {c:.2f}\n$R^2$ = {r2:.3f}"
    else:
        fit_text = "Linear fit not available"

    ax.text(
        0.95, 0.05, fit_text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=int(round(11 * font_scale)),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    x_label = xlabel if xlabel is not None else x_col.replace("_", " ").title()
    y_label = ylabel if ylabel is not None else y_col.replace("_", " ").title()
    ax.set_xlabel(x_label, fontsize=int(round(15 * font_scale)))
    ax.set_ylabel(y_label, fontsize=int(round(15 * font_scale)))
    if title is None:
        ax.set_title(
            f"{x_label} vs {y_label}",
            fontsize=int(round(18 * font_scale)), fontweight="bold",
        )
    elif title:
        ax.set_title(title, fontsize=int(round(18 * font_scale)), fontweight="bold")
    ax.tick_params(axis="both", labelsize=int(round(13 * font_scale)))
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def compute_score_deltas(
    table_2011: pd.DataFrame,
    table_2022: pd.DataFrame,
) -> pd.DataFrame:
    """Compute adoption score change between census years per postcode area.

    Args:
        table_2011: Output of ``build_validation_table()`` for 2011 data.
        table_2022: Output of ``build_validation_table()`` for 2022 data.

    Returns:
        DataFrame with columns: ``pc_area``, ``score_sum_delta``
        (2022 − 2011), ``bev_count`` (from 2022 table as validation target).
    """
    if table_2011["pc_area"].duplicated().any():
        dupes = table_2011.loc[table_2011["pc_area"].duplicated(), "pc_area"].tolist()[:10]
        raise ValueError(f"table_2011 must be unique by pc_area; duplicates include {dupes}")

    if table_2022["pc_area"].duplicated().any():
        dupes = table_2022.loc[table_2022["pc_area"].duplicated(), "pc_area"].tolist()[:10]
        raise ValueError(f"table_2022 must be unique by pc_area; duplicates include {dupes}")

    merged = table_2011[["pc_area", "score_sum"]].merge(
        table_2022[["pc_area", "score_sum", "bev_count"]],
        on="pc_area",
        how="inner",
        suffixes=("_2011", "_2022"),
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise ValueError(
            "No common pc_area values found between table_2011 and table_2022 — check inputs"
        )
    merged["score_sum_delta"] = merged["score_sum_2022"] - merged["score_sum_2011"]
    return merged[["pc_area", "score_sum_delta", "bev_count"]].reset_index(drop=True)
