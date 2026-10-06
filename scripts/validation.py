"""Recalculate geographic BEV comparisons from the released area inputs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts import datazone_validation as dzv


def comparison_row(label: str, scale: str, stats: dict, policy: str) -> dict:
    """Retain the manuscript's approximate Fisher-z uncertainty convention."""
    rho, n = float(stats["spearman_rho"]), int(stats["n"])
    limits = np.tanh(np.arctanh(rho) + np.array([-1, 1]) * 1.96 / np.sqrt(n - 3))
    return {"comparison": label, "scale": scale, "n": n, "spearman_rho": rho,
        "ci95_low": float(limits[0]), "ci95_high": float(limits[1]),
        "pearson_r2_rate": stats["pearson_r2"], "size_baseline_rho": stats["size_baseline_rho"],
        "sample_policy": policy}


def run_validation(root: Path, out: Path) -> dict:
    """Rebuild matched DZ/IZ samples, including suppression and coverage checks."""
    from reproduce import compare_csv

    out.mkdir(parents=True, exist_ok=True)
    inputs = root / "data/validation"
    # Round-trip parsing preserves tied affordability ranks after CSV export.
    oa = pd.read_csv(inputs / "oa_inputs.csv", float_precision="round_trip")
    lookup = pd.read_csv(inputs / "geography_lookup.csv", dtype=str)
    observed = pd.read_csv(inputs / "observed_bev.csv")
    if len(oa) != 46359 or not oa.geo_code.is_unique:
        raise ValueError("Expected 46,359 unique study OAs")
    model = dzv.aggregate_feasibility_to_datazone(oa, lookup, score_column="final_adoption_propensity")
    aff = dzv.aggregate_feasibility_to_datazone(oa, lookup, score_column="A_it")
    aff = aff.rename(columns={"feas": "A_it"})[["DZ22", "A_it"]]
    joined = model.merge(aff, on="DZ22", validate="one_to_one").merge(observed, on="DZ22", validate="one_to_one")
    joined["bev_per_1000_households"] = 1000 * joined.bev / joined.households
    rho = float(spearmanr(joined.feas, joined.bev_per_1000_households).statistic)
    income = float(spearmanr(joined.A_it, joined.bev_per_1000_households).statistic)
    ap_income = float(spearmanr(joined.feas, joined.A_it).statistic)
    partial = (rho - ap_income * income) / np.sqrt((1 - ap_income**2) * (1 - income**2))
    mapping = lookup[["DZ22", "IZ22"]].drop_duplicates()
    m, o, coverage = dzv.rollup_observed_to_intermediate_zone(model, observed, mapping)
    mc, oc, _ = dzv.rollup_observed_to_intermediate_zone(model, observed, mapping, complete_only=True)
    included = coverage[coverage.matched_dz > 0]
    covered = 100 * included.observed_households.sum() / included.households.sum()
    rows = [
        comparison_row("AP propensity vs BEV rate (Data Zone, headline)", "Data Zone", dzv.correlate_datazone(model, observed), "observed Data Zones"),
        {"comparison": "AP propensity vs rate | income (partial)", "scale": "Data Zone", "n": len(joined), "spearman_rho": float(partial), "sample_policy": "same observed Data Zones; affordability-adjusted ranks"},
        {"comparison": "income (A_it) vs rate (income-only baseline)", "scale": "Data Zone", "n": len(joined), "spearman_rho": income, "sample_policy": "same observed Data Zones"},
        comparison_row("AP propensity vs BEV rate (Intermediate Zone, observed support)", "Intermediate Zone", dzv.correlate_datazone(m, o, key="IZ22"), "matching observed constituent Data Zones"),
        comparison_row("AP propensity vs BEV rate (Intermediate Zone, complete coverage)", "Intermediate Zone", dzv.correlate_datazone(mc, oc, key="IZ22"), "all constituent Data Zones observed"),
    ]
    rows[3]["household_coverage_pct"], rows[4]["household_coverage_pct"] = float(covered), 100.0
    pd.DataFrame(rows).to_csv(out / "datazone_ap_validation.csv", index=False)
    coverage.to_csv(out / "datazone_ap_iz_coverage.csv", index=False)
    for name in ("datazone_ap_validation.csv", "datazone_ap_iz_coverage.csv"):
        compare_csv(out / name, root / "data/reference/validation" / name)
    return {"study_oas": len(oa), "observed_datazones": len(joined), "observed_intermediate_zones": len(m), "complete_intermediate_zones": len(mc), "tables_verified": 2}
