from __future__ import annotations

import re
import shutil
from pathlib import Path
import subprocess
from typing import Any

import geopandas as gpd
import pandas as pd

OUTPUT_AREA_CODE_PATTERN = re.compile(r"^S\d{8}$")


def sanitize_layer_name(name: str) -> str:
    """Create a stable GeoPackage layer name from a file stem."""
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return sanitized[:60] or "layer"


def validate_unique_codes(frame: pd.DataFrame, frame_name: str) -> None:
    """Ensure the code column exists and contains no duplicates or blanks."""
    if "code" not in frame.columns:
        raise ValueError(f"{frame_name} is missing a 'code' column.")

    code_series = frame["code"].astype("string").str.strip()
    if code_series.isna().any() or (code_series == "").any():
        raise ValueError(f"{frame_name} contains blank code values.")

    duplicate_codes = code_series[code_series.duplicated()].unique().tolist()
    if duplicate_codes:
        preview = ", ".join(duplicate_codes[:10])
        raise ValueError(f"{frame_name} contains duplicate code values: {preview}")


def validate_code_sets(boundaries: gpd.GeoDataFrame, table: pd.DataFrame, table_name: str) -> None:
    """Require exact code agreement between boundaries and a census table."""
    boundary_codes = set(boundaries["code"].astype("string").str.strip())
    table_codes = set(table["code"].astype("string").str.strip())

    missing_in_table = sorted(boundary_codes - table_codes)
    missing_in_boundaries = sorted(table_codes - boundary_codes)

    if missing_in_table or missing_in_boundaries:
        message_parts = [f"Code mismatch for {table_name}."]
        if missing_in_table:
            preview = ", ".join(missing_in_table[:10])
            message_parts.append(
                f"Missing in CSV: {len(missing_in_table)} code(s) such as {preview}"
            )
        if missing_in_boundaries:
            preview = ", ".join(missing_in_boundaries[:10])
            message_parts.append(
                f"Missing in boundaries: {len(missing_in_boundaries)} code(s) such as {preview}"
            )
        raise ValueError(" ".join(message_parts))


def load_boundaries(boundary_path: Path) -> gpd.GeoDataFrame:
    """Load OA boundaries and validate the code column."""
    boundaries = gpd.read_file(boundary_path)
    boundaries["code"] = boundaries["code"].astype("string").str.strip()
    validate_unique_codes(boundaries, str(boundary_path))
    return boundaries


def load_census_csv(csv_path: Path) -> pd.DataFrame:
    """Load a cleaned census CSV and normalize the code column."""
    table = pd.read_csv(csv_path, dtype={"code": "string"}, na_values=["-"], low_memory=False)
    table["code"] = table["code"].astype("string").str.strip()
    valid_code_mask = table["code"].str.match(OUTPUT_AREA_CODE_PATTERN, na=False)
    if valid_code_mask.any():
        table = table.loc[valid_code_mask].copy()
    validate_unique_codes(table, str(csv_path))
    return table


def find_ogr2ogr() -> Path:
    """Locate an ogr2ogr executable on common Windows GIS installs."""
    which_path = shutil.which("ogr2ogr")
    if which_path:
        return Path(which_path)

    candidate_paths = [
        Path(r"C:\Program Files\QGIS 3.42.1\bin\ogr2ogr.exe"),
        Path(r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\Library\bin\ogr2ogr.exe"),
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not locate ogr2ogr.exe")


def write_gpkg_with_ogr(joined: gpd.GeoDataFrame, output_path: Path, layer_name: str) -> None:
    """Write a GeoPackage via temporary GeoJSON and ogr2ogr."""
    ogr2ogr_path = find_ogr2ogr()
    stage_dir = Path.home() / ".codex" / "memories" / "gpkg_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    temp_geojson = stage_dir / f"{layer_name}.geojson"
    temp_gpkg = stage_dir / f"{layer_name}.gpkg"

    for stage_path in (temp_geojson, temp_gpkg):
        if stage_path.exists():
            stage_path.unlink()

    joined.to_file(temp_geojson, driver="GeoJSON")

    command = [
        str(ogr2ogr_path),
        "-f",
        "GPKG",
        str(temp_gpkg),
        str(temp_geojson),
        "-nln",
        layer_name,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ogr2ogr failed for {output_path.name}: {result.stderr or result.stdout}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(temp_gpkg, output_path)

    for stage_path in (temp_geojson, temp_gpkg):
        if stage_path.exists():
            stage_path.unlink()


def join_csv_to_boundaries(
    boundaries: gpd.GeoDataFrame,
    csv_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Strictly join one cleaned census CSV to OA boundaries and write a GeoPackage."""
    table = load_census_csv(csv_path)
    validate_code_sets(boundaries, table, csv_path.name)

    joined = boundaries.merge(table, on="code", how="left", validate="one_to_one")
    if joined["code"].isna().any() or len(joined) != len(boundaries):
        raise ValueError(f"Join failed for {csv_path.name}: row count or codes changed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    layer_name = sanitize_layer_name(csv_path.stem)
    write_gpkg_with_ogr(joined, output_path, layer_name)

    return {
        "csv_name": csv_path.name,
        "rows": len(joined),
        "columns": len(joined.columns),
        "layer_name": layer_name,
        "output_path": str(output_path),
    }


def build_gpkgs_for_directory(
    cleaned_dir: Path,
    boundary_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Create a GeoPackage for every cleaned census CSV in a directory."""
    boundaries = load_boundaries(boundary_path)
    summaries: list[dict[str, Any]] = []

    for csv_path in sorted(cleaned_dir.glob("*.csv")):
        output_path = output_dir / f"{csv_path.stem}.gpkg"
        summaries.append(join_csv_to_boundaries(boundaries, csv_path, output_path))

    return summaries
