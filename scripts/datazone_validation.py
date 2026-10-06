"""Data-Zone-scale validation of Study 3 feasibility against DfT plug-in uptake.

Validates feasible_share (aggregated OA -> Data Zone) against DfT VEH0145
battery-electric vehicles per household at Data-Zone scale, where Scotland's
roughly equal-population Data Zones remove the area-size confound that inflates
the postcode-area count correlations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_plugin_counts_by_datazone(
    csv_path: str | Path,
    *,
    fuel: str = "BATTERY ELECTRIC",
    keepership: str = "PRIVATE",
    quarter: str = "2025 Q4",
) -> pd.DataFrame:
    """Read DfT VEH0145 and return Scotland Data-Zone vehicle counts."""
    df = pd.read_csv(csv_path, dtype=str)
    required = {"LSOA21CD", "Fuel", "Keepership", quarter}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"VEH0145 CSV missing required columns: {sorted(missing)}")
    scot = df[df["LSOA21CD"].str.startswith("S", na=False)]
    sel = scot[(scot["Fuel"] == fuel) & (scot["Keepership"] == keepership)].copy()
    sel["bev"] = pd.to_numeric(sel[quarter].str.replace(",", "", regex=False), errors="coerce")
    out = (
        sel.dropna(subset=["bev"])
        .groupby("LSOA21CD", as_index=False)["bev"]
        .sum()
        .rename(columns={"LSOA21CD": "DZ22"})
    )
    return out.sort_values("DZ22").reset_index(drop=True)


def aggregate_feasibility_to_datazone(
    panel: pd.DataFrame,
    oa_dz_lookup: pd.DataFrame,
    *,
    score_column: str = "feasible_share",
    weight_column: str = "households",
    oa_column: str = "geo_code",
) -> pd.DataFrame:
    """Household-weight the OA feasibility score up to Data Zone (DZ22)."""
    required = {oa_column, score_column, weight_column}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    if not {"OA22", "DZ22"}.issubset(oa_dz_lookup.columns):
        raise ValueError("oa_dz_lookup must contain OA22 and DZ22")
    merged = panel.merge(
        oa_dz_lookup[["OA22", "DZ22"]].drop_duplicates("OA22"),
        left_on=oa_column, right_on="OA22", how="inner",
    )
    if merged.empty:
        raise ValueError("No OA rows joined to Data Zones")
    merged = merged.copy()
    merged["_fxh"] = merged[score_column].astype(float) * merged[weight_column].astype(float)
    grouped = merged.groupby("DZ22", as_index=False).agg(
        feas_sum=("_fxh", "sum"), households=(weight_column, "sum")
    )
    grouped["feas"] = grouped["feas_sum"] / grouped["households"]
    return grouped[["DZ22", "feas", "households", "feas_sum"]].sort_values("DZ22").reset_index(drop=True)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho/p that returns NaN for zero-variance input instead of warning."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r that returns NaN for zero-variance input instead of warning."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r)


def correlate_datazone(
    model_dz: pd.DataFrame,
    observed_dz: pd.DataFrame,
    *,
    key: str = "DZ22",
) -> dict[str, Any]:
    """Join model feasibility and observed BEV by Data Zone and correlate.

    Note: loglog_r2 is a log1p-stabilised rate correlation (on small rates log1p(x) ~ x), not a true count log-log elasticity.
    """
    joined = model_dz.merge(observed_dz.drop_duplicates(key), on=key, how="inner").copy()
    if joined.empty:
        raise ValueError("No Data Zones joined between model and observed")
    joined["bev_rate"] = joined["bev"].astype(float) / joined["households"].astype(float)
    valid = joined.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["feas", "bev_rate", "feas_sum", "bev", "households"]
    )
    feas = valid["feas"].to_numpy(float)
    rate = valid["bev_rate"].to_numpy(float)
    feas_sum = valid["feas_sum"].to_numpy(float)
    bev = valid["bev"].to_numpy(float)
    hh = valid["households"].to_numpy(float)
    rho, p = _safe_spearman(feas, rate)
    r = _safe_pearson_r(feas, rate)
    r_ll = _safe_pearson_r(np.log1p(feas), np.log1p(rate))
    r_count = _safe_pearson_r(feas_sum, bev)
    size_rho, _ = _safe_spearman(hh, bev)
    size_r = _safe_pearson_r(hh, bev)
    return {
        "n": int(len(valid)),
        "spearman_rho": rho,
        "spearman_p": p,
        "pearson_r": r,
        "pearson_r2": r**2 if r == r else float("nan"),
        "loglog_r2": r_ll**2 if r_ll == r_ll else float("nan"),
        "count_r2": r_count**2 if r_count == r_count else float("nan"),
        "size_baseline_rho": size_rho,
        "size_baseline_r2": size_r**2 if size_r == size_r else float("nan"),
    }


def impute_suppressed(
    observed_dz: pd.DataFrame,
    all_dz_codes: Any,
    *,
    fill: float = 2.5,
) -> pd.DataFrame:
    """Add rows for Data Zones absent from observed_dz (DfT <5-vehicle suppression)."""
    present = set(observed_dz["DZ22"])
    missing = [c for c in all_dz_codes if c not in present]
    if not missing:
        return observed_dz[["DZ22", "bev"]].sort_values("DZ22").reset_index(drop=True)
    extra = pd.DataFrame({"DZ22": missing, "bev": float(fill)})
    return (
        pd.concat([observed_dz[["DZ22", "bev"]], extra], ignore_index=True)
        .sort_values("DZ22")
        .reset_index(drop=True)
    )


def rollup_to_intermediate_zone(
    model_dz: pd.DataFrame,
    observed_dz: pd.DataFrame,
    dz_iz_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Data-Zone model and observed frames to Intermediate Zone (IZ22)."""
    if not {"DZ22", "IZ22"}.issubset(dz_iz_lookup.columns):
        raise ValueError("dz_iz_lookup must contain DZ22 and IZ22")
    lk = dz_iz_lookup[["DZ22", "IZ22"]].drop_duplicates("DZ22")
    m = model_dz.merge(lk, on="DZ22", how="inner")
    model_iz = m.groupby("IZ22", as_index=False).agg(
        feas_sum=("feas_sum", "sum"), households=("households", "sum")
    )
    model_iz["feas"] = model_iz["feas_sum"] / model_iz["households"]
    o = observed_dz.merge(lk, on="DZ22", how="inner")
    observed_iz = o.groupby("IZ22", as_index=False)["bev"].sum()
    return (
        model_iz[["IZ22", "feas", "households", "feas_sum"]].sort_values("IZ22").reset_index(drop=True),
        observed_iz.sort_values("IZ22").reset_index(drop=True),
    )


def rollup_observed_to_intermediate_zone(
    model_dz: pd.DataFrame,
    observed_dz: pd.DataFrame,
    dz_iz_lookup: pd.DataFrame,
    *,
    complete_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate AP, households and BEVs over identical observed Data Zones.

    Missing counts are excluded; recorded zeros remain observations. Completeness
    uses the full lookup, including zones missing model data. Return model totals,
    observed totals and all-zone coverage. Raise ValueError for ambiguous keys,
    invalid model values or negative/infinite available counts. Existing Study 3
    callers retain the separate legacy helper; Paper 1 uses this explicit policy.
    """
    for frame, columns, name in (
        (model_dz, {"DZ22", "feas", "households", "feas_sum"}, "model"),
        (observed_dz, {"DZ22", "bev"}, "observed"),
        (dz_iz_lookup, {"DZ22", "IZ22"}, "lookup"),
    ):
        if not columns.issubset(frame.columns):
            raise ValueError(f"{name} is missing columns: {columns - set(frame.columns)}")
        if frame["DZ22"].isna().any():
            raise ValueError(f"{name} contains a missing Data-Zone identifier")
    lookup = dz_iz_lookup[["DZ22", "IZ22"]].drop_duplicates()
    if lookup["DZ22"].duplicated().any() or lookup["IZ22"].isna().any():
        raise ValueError("Each Data Zone must map to exactly one Intermediate Zone")
    for frame in (model_dz, observed_dz):
        if frame["DZ22"].duplicated().any():
            raise ValueError("Model and observed Data-Zone keys must be unique")
        if not set(frame["DZ22"]).issubset(set(lookup["DZ22"])):
            raise ValueError("A Data Zone is absent from the geographic lookup")
    values = model_dz[["feas", "households", "feas_sum"]].to_numpy(float)
    if not np.isfinite(values).all() or (model_dz["households"] <= 0).any():
        raise ValueError("Model scores must be finite and household totals positive")
    if not np.allclose(model_dz["feas_sum"], model_dz["feas"] * model_dz["households"]):
        raise ValueError("The weighted-score sum disagrees with score times households")
    observed = observed_dz[["DZ22", "bev"]].copy()
    observed["bev"] = pd.to_numeric(observed["bev"], errors="raise")
    available = observed["bev"].dropna().to_numpy(float)
    if not np.isfinite(available).all() or (available < 0).any():
        raise ValueError("Available vehicle counts must be finite and nonnegative")
    full = lookup.merge(
        model_dz[["DZ22", "feas_sum", "households"]],
        on="DZ22", how="left", validate="one_to_one",
    ).merge(observed, on="DZ22", how="left", validate="one_to_one")
    full["model_available"] = full["households"].notna()
    full["count_available"] = full["bev"].notna()
    full["matched"] = full["model_available"] & full["count_available"]
    full["observed_households"] = full["households"].where(full["matched"], 0.0)
    coverage = full.groupby("IZ22", as_index=False).agg(
        total_dz=("DZ22", "size"), model_dz=("model_available", "sum"),
        observed_dz=("count_available", "sum"), matched_dz=("matched", "sum"),
        households=("households", "sum"), observed_households=("observed_households", "sum"),
    )
    coverage["complete"] = coverage["matched_dz"] == coverage["total_dz"]
    coverage["household_coverage_pct"] = np.where(
        (coverage["model_dz"] == coverage["total_dz"]) & (coverage["households"] > 0),
        100 * coverage["observed_households"] / coverage["households"], np.nan,
    )
    matched = full.loc[full["matched"]].copy()
    if complete_only:
        matched = matched[matched["IZ22"].isin(coverage.loc[coverage["complete"], "IZ22"])]
    totals = matched.groupby("IZ22", as_index=False).agg(
        feas_sum=("feas_sum", "sum"), households=("households", "sum"), bev=("bev", "sum"),
    )
    totals["feas"] = totals["feas_sum"] / totals["households"]
    return (
        totals[["IZ22", "feas", "households", "feas_sum"]],
        totals[["IZ22", "bev"]],
        coverage,
    )
