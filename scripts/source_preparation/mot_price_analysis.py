from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


SCOTLAND_PC_AREAS: frozenset[str] = frozenset(
    ["AB", "DD", "DG", "EH", "FK", "G", "HS", "IV", "KA", "KW", "KY", "ML", "PA", "PH", "TD", "ZE"]
)


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def normalize_postcode_area(value: str | None) -> str | None:
    """Normalize postcode area values to 1-3 uppercase letters."""
    if value is None or pd.isna(value):
        return None
    cleaned = re.sub(r"[^A-Za-z]", "", str(value)).upper()
    return cleaned if 0 < len(cleaned) <= 3 else None


def infer_model_family(make: str | None, model: str | None) -> str | None:
    """Map noisy MOT make/model strings to a report-scoped EV family key.

    The goal is pragmatic reproducibility for this report, not a complete
    vehicle taxonomy. Unmatched families are excluded later with coverage notes.
    """
    make_norm = _normalize_text(make)
    model_norm = _normalize_text(model)

    if not make_norm or not model_norm:
        return None

    if make_norm == "NISSAN":
        if "LEAF" in model_norm:
            return "LEAF"
        if "E-NV200" in model_norm or "NV200" in model_norm:
            return "E-NV200"

    if make_norm == "RENAULT":
        if "ZOE" in model_norm:
            return "ZOE"
        if "KANGOO" in model_norm:
            return "KANGOO ZE"

    if make_norm == "TESLA":
        if "MODEL 3" in model_norm:
            if "PERFORMANCE" in model_norm:
                return "MODEL 3 PERFORMANCE"
            if "LONG RANGE" in model_norm:
                return "MODEL 3 LONG RANGE"
            return "MODEL 3 STANDARD"
        if "MODEL S" in model_norm:
            if "P100D" in model_norm or "PERFORMANCE" in model_norm or "PLAID" in model_norm:
                return "MODEL S PERFORMANCE"
            return "MODEL S STANDARD"
        if "MODEL X" in model_norm:
            if "P100D" in model_norm or "PERFORMANCE" in model_norm or "PLAID" in model_norm:
                return "MODEL X PERFORMANCE"
            return "MODEL X STANDARD"

    if make_norm == "BMW":
        if model_norm.startswith("I3S"):
            return "I3"
        if model_norm.startswith("I3"):
            return "I3"

    if make_norm == "JAGUAR" and "I-PACE" in model_norm:
        return "I-PACE"

    if make_norm == "VOLKSWAGEN":
        if "ID3" in model_norm or "ID.3" in model_norm:
            return "ID.3"
        if "E-GOLF" in model_norm or model_norm == "GOLF":
            return "E-GOLF"
        if "E-UP" in model_norm or model_norm.startswith("UP"):
            return "E-UP"

    if make_norm == "KIA":
        if "NIRO" in model_norm:
            return "NIRO EV"
        if "SOUL" in model_norm:
            return "SOUL EV"

    if make_norm == "MG":
        if model_norm.startswith("ZS"):
            return "ZS EV"
        if model_norm.startswith("5"):
            return "MG5 EV"

    if make_norm == "HYUNDAI":
        if "KONA" in model_norm:
            return "KONA ELECTRIC"
        if "IONIQ" in model_norm:
            return "IONIQ ELECTRIC"

    if make_norm == "VAUXHALL" and "CORSA" in model_norm:
        return "CORSA ELECTRIC"

    if make_norm == "AUDI" and "E-TRON" in model_norm:
        return "E-TRON"

    if make_norm == "PEUGEOT":
        if model_norm.startswith("208"):
            return "E-208"
        if model_norm.startswith("2008"):
            return "E-2008"
        if "PARTNER" in model_norm:
            return None

    if make_norm == "MERCEDES-BENZ" and "EQC" in model_norm:
        return "EQC"

    if make_norm == "POLESTAR" and "POLESTAR 2" in model_norm:
        return "POLESTAR 2"

    if make_norm == "PORSCHE" and "TAYCAN" in model_norm:
        return "TAYCAN"

    if make_norm == "MINI" and "COOPER" in model_norm:
        return "COOPER ELECTRIC"

    if make_norm == "HONDA" and model_norm.startswith("E"):
        return "HONDA E"

    if make_norm == "SEAT" and "MII" in model_norm:
        return "MII ELECTRIC"

    if make_norm == "SKODA" and "CITIGO" in model_norm:
        return "CITIGOE IV"

    if make_norm in {"SMART", "SMART (MCC)"} and ("EQ" in model_norm or "FORTWO" in model_norm or "FORFOUR" in model_norm):
        return "SMART EQ"

    return None


def model_similarity_key(model_family: str | None) -> str | None:
    """Collapse trim-level families to a broader similarity key."""
    if not isinstance(model_family, str) or not model_family:
        return None
    if model_family.startswith("MODEL 3"):
        return "MODEL 3"
    if model_family.startswith("MODEL S"):
        return "MODEL S"
    if model_family.startswith("MODEL X"):
        return "MODEL X"
    return model_family


def load_salary_lookup(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"year", "annualized_median_salary_gbp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Salary CSV missing required columns: {sorted(missing)}")
    return df.copy()


def validate_salary_references(
    salary_lookup: pd.DataFrame, required_years: Iterable[int]
) -> dict[int, float]:
    """Require one finite, positive reference for each year being classified.

    The legacy column name does not establish the reference's statistical
    definition. This validates usable numbers, not their source provenance.
    """
    required = {"year", "annualized_median_salary_gbp"}
    missing = required - set(salary_lookup.columns)
    if missing:
        raise ValueError(f"Salary CSV missing required columns: {sorted(missing)}")
    years = pd.to_numeric(salary_lookup["year"], errors="coerce")
    values: dict[int, float] = {}
    for year in sorted(set(required_years)):
        if not math.isfinite(float(year)) or int(year) != year or year <= 0:
            raise ValueError(f"Invalid required first-use year: {year}")
        rows = salary_lookup.loc[years.eq(year), "annualized_median_salary_gbp"]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one price reference for {year}; found {len(rows)}"
            )
        value = pd.to_numeric(rows, errors="coerce").iloc[0]
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"Price reference for {year} must be finite and positive")
        values[int(year)] = float(value)
    return values


def load_price_lookup(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"make", "model_family", "year_start", "year_end", "rrp_gbp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price lookup CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["make"] = df["make"].map(_normalize_text)
    df["model_family"] = df["model_family"].map(_normalize_text)
    df["similarity_key"] = df["model_family"].map(model_similarity_key)
    return df


def extract_scottish_electric_registrations(
    mot_csv: Path,
    *,
    start_year: int,
    end_year: int,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Extract deduplicated Scottish electric-coded registrations from MOT data."""
    usecols = [
        "vehicle_id",
        "fuel_type",
        "postcode_area",
        "test_result",
        "first_use_date",
        "test_date",
        "make",
        "model",
    ]
    frames: list[pd.DataFrame] = []

    reader = pd.read_csv(mot_csv, usecols=usecols, dtype=str, chunksize=chunk_size)
    for chunk in reader:
        chunk["pc_area"] = chunk["postcode_area"].map(normalize_postcode_area)
        chunk = chunk[chunk["pc_area"].isin(SCOTLAND_PC_AREAS)]
        if chunk.empty:
            continue

        chunk["fuel_norm"] = chunk["fuel_type"].map(_normalize_text)
        chunk = chunk[chunk["fuel_norm"] == "EL"]
        if chunk.empty:
            continue

        chunk["test_result"] = chunk["test_result"].map(_normalize_text)
        chunk = chunk[chunk["test_result"] == "P"]
        if chunk.empty:
            continue

        chunk["first_use_date"] = pd.to_datetime(chunk["first_use_date"], errors="coerce")
        chunk["first_use_year"] = chunk["first_use_date"].dt.year
        chunk = chunk[(chunk["first_use_year"] >= start_year) & (chunk["first_use_year"] <= end_year)]
        if chunk.empty:
            continue

        chunk["make"] = chunk["make"].map(_normalize_text)
        chunk["model"] = chunk["model"].map(_normalize_text)
        chunk["model_family"] = [
            infer_model_family(make, model)
            for make, model in zip(chunk["make"], chunk["model"], strict=False)
        ]
        chunk["test_date"] = pd.to_datetime(chunk["test_date"], errors="coerce")
        frames.append(
            chunk[["vehicle_id", "pc_area", "make", "model", "model_family", "first_use_year", "test_date"]]
        )

    if not frames:
        return pd.DataFrame(
            columns=["vehicle_id", "pc_area", "make", "model", "model_family", "first_use_year", "test_date"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["vehicle_id", "test_date"], na_position="first")
    combined = combined.drop_duplicates(subset=["vehicle_id"], keep="last").reset_index(drop=True)
    combined["similarity_key"] = combined["model_family"].map(model_similarity_key)
    return combined


def _year_distance(year: int, year_start: int, year_end: int) -> int:
    if year < year_start:
        return year_start - year
    if year > year_end:
        return year - year_end
    return 0


def _select_best_candidate(candidates: pd.DataFrame, year: int, *, conservative: bool) -> pd.Series | None:
    if candidates.empty:
        return None

    ranked = candidates.copy()
    ranked["year_distance"] = ranked.apply(
        lambda row: _year_distance(int(year), int(row["year_start"]), int(row["year_end"])),
        axis=1,
    )
    ranked["range_span"] = ranked["year_end"] - ranked["year_start"]
    ranked = ranked.sort_values(
        ["year_distance", "range_span", "rrp_gbp", "model_family"],
        ascending=[True, True, True if conservative else False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def _pick_best_price_match(
    make: str,
    model_family: str | None,
    similarity_key: str | None,
    year: int,
    price_lookup: pd.DataFrame,
) -> pd.Series | None:
    if not model_family:
        return None

    exact_candidates = price_lookup[
        (price_lookup["make"] == make)
        & (price_lookup["model_family"] == model_family)
        & (price_lookup["year_start"] <= year)
        & (price_lookup["year_end"] >= year)
    ]
    exact_match = _select_best_candidate(exact_candidates, year, conservative=False)
    if exact_match is not None:
        return exact_match

    adjacent_candidates = price_lookup[
        (price_lookup["make"] == make)
        & (price_lookup["model_family"] == model_family)
        & (price_lookup["year_start"] - 1 <= year)
        & (price_lookup["year_end"] + 1 >= year)
    ]
    adjacent_match = _select_best_candidate(adjacent_candidates, year, conservative=False)
    if adjacent_match is not None:
        return adjacent_match

    if similarity_key:
        similar_candidates = price_lookup[
            (price_lookup["make"] == make)
            & (price_lookup["similarity_key"] == similarity_key)
            & (price_lookup["year_start"] - 1 <= year)
            & (price_lookup["year_end"] + 1 >= year)
        ]
        similar_match = _select_best_candidate(similar_candidates, year, conservative=True)
        if similar_match is not None:
            return similar_match

    return None


def attach_price_segments(
    registrations: pd.DataFrame,
    price_lookup: pd.DataFrame,
    salary_lookup: pd.DataFrame,
    *,
    premium_multiple: float = 1.5,
) -> pd.DataFrame:
    """Join price and salary data, then classify each matched vehicle."""
    records: list[dict[str, object]] = []
    if not math.isfinite(premium_multiple) or premium_multiple <= 0:
        raise ValueError("premium_multiple must be finite and positive")
    validate_salary_references(salary_lookup, [])
    salary_map = salary_lookup.set_index("year")["annualized_median_salary_gbp"].to_dict()
    checked_references: dict[int, float] = {}

    for row in registrations.itertuples(index=False):
        match = _pick_best_price_match(
            make=row.make,
            model_family=row.model_family,
            similarity_key=row.similarity_key,
            year=int(row.first_use_year),
            price_lookup=price_lookup,
        )
        year = int(row.first_use_year)
        if match is not None and year not in checked_references:
            checked_references.update(validate_salary_references(salary_lookup, [year]))
        salary = checked_references.get(year, salary_map.get(year))
        if match is None and salary is not None:
            salary = pd.to_numeric(pd.Series([salary]), errors="coerce").iloc[0]
            if not math.isfinite(float(salary)) or salary <= 0:
                salary = None
        threshold = salary * premium_multiple if salary is not None else None
        if match is not None and (not math.isfinite(threshold) or threshold <= 0):
            raise ValueError(f"Premium threshold for {year} must be finite and positive")

        if match is None:
            match_rule = "unmatched"
            rrp_gbp = None
            source_url = None
            source_note = None
            matched_family = None
            price_segment = None
        else:
            matched_family = match["model_family"]
            rrp_gbp = float(match["rrp_gbp"])
            source_url = match.get("source_url")
            source_note = match.get("source_note")
            if matched_family == row.model_family and match["year_start"] <= row.first_use_year <= match["year_end"]:
                match_rule = "family_year_range_proxy"
            elif matched_family == row.model_family:
                match_rule = "adjacent_year_proxy"
            else:
                match_rule = "similar_model_family_proxy"
            price_segment = "premium" if threshold is not None and rrp_gbp > threshold else "lower_price"

        records.append(
            {
                "vehicle_id": row.vehicle_id,
                "pc_area": row.pc_area,
                "make": row.make,
                "model": row.model,
                "model_family": row.model_family,
                "matched_model_family": matched_family,
                "first_use_year": int(row.first_use_year),
                "annualized_median_salary_gbp": salary,
                "premium_threshold_gbp": threshold,
                "rrp_gbp": rrp_gbp,
                "price_segment": price_segment,
                "match_rule": match_rule,
                "source_url": source_url,
                "source_note": source_note,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_price_match_coverage(segmented: pd.DataFrame) -> pd.DataFrame:
    """Summarize match coverage and fallback use for reporting."""
    total = len(segmented)
    matched = int(segmented["price_segment"].notna().sum())
    unmatched = total - matched
    premium = int((segmented["price_segment"] == "premium").sum())
    lower = int((segmented["price_segment"] == "lower_price").sum())
    rows = [
        {"metric": "total_registrations", "value": total},
        {"metric": "matched_registrations", "value": matched},
        {"metric": "unmatched_registrations", "value": unmatched},
        {"metric": "matched_share", "value": matched / total if total else float("nan")},
        {"metric": "premium_registrations", "value": premium},
        {"metric": "lower_price_registrations", "value": lower},
    ]
    for rule, count in segmented["match_rule"].value_counts(dropna=False).items():
        rows.append({"metric": f"match_rule::{rule}", "value": int(count)})
    return pd.DataFrame(rows)


def build_segmented_postcode_counts(
    segmented: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Aggregate segmented registrations to postcode area for the report window."""
    window = segmented[(segmented["first_use_year"] >= start_year) & (segmented["first_use_year"] <= end_year)].copy()
    if window.empty:
        return pd.DataFrame(
            columns=[
                "pc_area",
                "total_registrations",
                "matched_registrations",
                "unmatched_registrations",
                "premium_registrations",
                "lower_price_registrations",
                "premium_share_matched",
            ]
        )

    grouped = window.groupby("pc_area")
    summary = grouped["vehicle_id"].nunique().rename("total_registrations").reset_index()
    summary["matched_registrations"] = grouped["price_segment"].apply(lambda s: s.notna().sum()).to_numpy()
    summary["unmatched_registrations"] = summary["total_registrations"] - summary["matched_registrations"]
    summary["premium_registrations"] = grouped["price_segment"].apply(lambda s: (s == "premium").sum()).to_numpy()
    summary["lower_price_registrations"] = grouped["price_segment"].apply(
        lambda s: (s == "lower_price").sum()
    ).to_numpy()
    import numpy as np
    summary["premium_share_matched"] = np.where(
        summary["matched_registrations"] > 0,
        summary["premium_registrations"] / summary["matched_registrations"],
        float("nan"),
    ).astype(float)
    return summary


def build_segmented_analysis_table(
    score_delta_table: pd.DataFrame,
    postcode_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Merge postcode registration summary onto the existing score-delta table."""
    merged = score_delta_table.merge(postcode_counts, on="pc_area", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No common pc_area values found between score delta table and postcode counts")
    return merged
