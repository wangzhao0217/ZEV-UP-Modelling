from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

OUTPUT_AREA_CODE_PATTERN = re.compile(r"^S\d{8}$")


def normalize_header_cell(value: str) -> str:
    """Normalize a header cell to a stable single-line label."""
    cleaned = value.replace("\ufeff", "").replace("\n", " ").strip().strip('"')
    return re.sub(r"\s+", " ", cleaned)


def split_census_sections(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Split raw census CSV rows into metadata, header rows, and data rows."""
    non_empty_rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not non_empty_rows:
        raise ValueError("CSV is empty")

    header_start = next(
        (index for index, row in enumerate(non_empty_rows) if not normalize_header_cell(row[0] if row else "")),
        None,
    )

    if header_start is None:
        return [], [non_empty_rows[0]], non_empty_rows[1:]

    data_start = next(
        (
            index
            for index in range(header_start, len(non_empty_rows))
            if normalize_header_cell(non_empty_rows[index][0] if non_empty_rows[index] else "")
        ),
        None,
    )

    if data_start is None or data_start == header_start:
        raise ValueError("Could not identify data rows after header rows")

    return (
        non_empty_rows[:header_start],
        non_empty_rows[header_start:data_start],
        non_empty_rows[data_start:],
    )


def flatten_header_rows(header_rows: list[list[str]]) -> list[str]:
    """Flatten one to three census header rows into unique single-line column names."""
    if not header_rows:
        raise ValueError("At least one header row is required")

    width = max(len(row) for row in header_rows)
    flattened: list[str] = []

    for column_index in range(width):
        if column_index == 0:
            flattened.append("code")
            continue

        parts: list[str] = []
        for row in header_rows:
            raw_value = row[column_index] if column_index < len(row) else ""
            cleaned_value = normalize_header_cell(raw_value)
            if not cleaned_value:
                continue
            if not parts or parts[-1] != cleaned_value:
                parts.append(cleaned_value)

        flattened.append("__".join(parts) if parts else f"col_{column_index}")

    return make_unique(flattened)


def make_unique(headers: list[str]) -> list[str]:
    """Ensure header names stay unique after flattening."""
    counts: dict[str, int] = {}
    unique_headers: list[str] = []

    for header in headers:
        counts[header] = counts.get(header, 0) + 1
        if counts[header] == 1:
            unique_headers.append(header)
        else:
            unique_headers.append(f"{header}__{counts[header]}")

    return unique_headers


def pad_row(row: list[str], width: int) -> list[str]:
    """Pad or trim a row to the expected output width."""
    if len(row) >= width:
        return row[:width]
    return row + [""] * (width - len(row))


def filter_output_area_rows(data_rows: list[list[str]]) -> list[list[str]]:
    """Keep only valid OA rows when the file is clearly OA-coded."""
    valid_rows = [
        row
        for row in data_rows
        if row and OUTPUT_AREA_CODE_PATTERN.match(normalize_header_cell(row[0]))
    ]
    return valid_rows if valid_rows else data_rows


def flatten_census_csv(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Flatten a raw census CSV with multi-row headers into a one-header-line CSV."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    metadata_rows, header_rows, data_rows = split_census_sections(rows)
    header = flatten_header_rows(header_rows)
    filtered_data_rows = filter_output_area_rows(data_rows)
    width = len(header)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in filtered_data_rows:
            writer.writerow(pad_row(row, width))

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "metadata_rows": len(metadata_rows),
        "header_rows": len(header_rows),
        "data_rows": len(filtered_data_rows),
        "dropped_rows": len(data_rows) - len(filtered_data_rows),
        "columns": width,
    }


def flatten_directory(input_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Flatten all CSV files under a directory and preserve relative paths."""
    summaries: list[dict[str, Any]] = []

    for input_path in sorted(input_dir.rglob("*.csv")):
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        summaries.append(flatten_census_csv(input_path, output_path))

    return summaries
