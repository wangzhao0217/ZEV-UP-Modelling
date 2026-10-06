"""Final vehicle-assignment and trip-purpose component experiments."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from functions.paper1_grid4 import assign_grid4, summarize_grid4
from scripts.purpose_robustness import calculate


def auxiliary_claims(root: Path, out: Path) -> dict:
    """Check membership and matched-route claims; summarise archived calibration."""
    import pyogrio
    from scipy.stats import pearsonr
    from scripts.band_membership import assign_bands, membership_stability

    ap = np.concatenate([pyogrio.read_dataframe(path,
        columns=["final_adoption_propensity"], read_geometry=False).final_adoption_propensity.to_numpy()
        for path in sorted((root / "data/areas").glob("*.gpkg"))])
    baseline = assign_bands(ap, .3, .8)
    grid = pd.DataFrame([{"lower": lo, "upper": hi,
        "unchanged_share": membership_stability(baseline, assign_bands(ap, lo, hi))}
        for lo in (.2, .25, .3, .35, .4) for hi in (.7, .75, .8, .85, .9)])
    near = grid[(grid.lower.isin([.25, .3, .35]) & grid.upper.eq(.8)) |
                (grid.upper.isin([.75, .8, .85]) & grid.lower.eq(.3))]
    stability = float(near.unchanged_share.min())
    if len(ap) != 46359 or round(100 * stability) != 93:
        raise ValueError("The printed 93% membership claim no longer holds")
    grid.to_csv(out / "band_membership.csv", index=False)

    routes = pd.read_csv(root / "data/assessments/routing_compare_google_osrm.csv")
    expected = json.loads((root / "data/assessments/routing_compare_google_osrm.json").read_text())
    ratios = np.round(routes.osrm_km / routes.google_km, 4)
    differences = np.round(abs(routes.osrm_km - routes.google_km) / routes.google_km * 100, 2)
    metrics = {"matched_routes": len(routes),
        "pearson_r_osrm_vs_google": round(float(pearsonr(routes.osrm_km, routes.google_km).statistic), 4),
        "median_ratio_osrm_over_google": round(float(np.median(ratios)), 4),
        "median_abs_pct_diff": round(float(np.median(differences)), 2)}
    for key, value in metrics.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12)

    directory = root / "data/assessments/cutoff_calibration"
    calibration = json.loads((directory / "recommended_thresholds.json").read_text())
    draws = pd.read_csv(directory / "bootstrap_thresholds.csv")
    for name, column in (("moderate", "low_threshold"), ("high", "high_threshold")):
        interval = np.round(np.quantile(draws[column], [.025, .975]), 4)
        np.testing.assert_allclose(interval, calibration["bootstrap_95ci"][name], atol=1e-12)
    return {"membership_grid_rows": len(grid), "minimum_unchanged_share_pm05": stability,
        "matched_route_statistics": metrics, "archived_cutoff_bootstrap_rows": len(draws),
        "cutoff_scope": "Intervals summarised from retained draws; original 84-row fit input is unavailable."}


def summarize_archived_grids(directory: Path) -> dict:
    """Reproduce reported summaries from retained sweeps, without claiming refits."""
    g1 = pd.read_csv(directory / "grid1_results_comprehensive.csv")
    g2 = pd.read_csv(directory / "grid2_yougov_validation.csv")
    g3 = pd.read_csv(directory / "grid3_results.csv")
    g5 = pd.read_csv(directory / "grid5_results.csv")
    normalized = pd.read_csv(directory / "grid5_results_normalized.csv")
    sample = pd.read_csv(directory / "grid5_bev_validation.csv")
    g6 = pd.read_csv(directory / "grid6_results.csv")
    g7 = pd.read_csv(directory / "grid7_results.csv")
    scheme = g6.groupby("scheme_name")[["commute_score", "mean_suitability"]].mean()
    cv = 100 * scheme.std(ddof=1) / scheme.mean()
    g1_reference = g1[g1.scheme_name.eq("A (Current)")]
    np.testing.assert_allclose(g1_reference.r2_bev_sum, g1_reference.r2_bev_sum.iloc[0], atol=1e-12)
    np.testing.assert_allclose(normalized.mean_adjustment_norm, 1, atol=1e-12)
    np.testing.assert_allclose(sample.r2_bev_hpi, sample.r2_bev_hpi.iloc[0], atol=1e-12)
    by_pair = g7.groupby(["compound_penalty", "peak_age_bonus"]).mean_adjusted_ap
    if (by_pair.max() - by_pair.min()).max() > 1e-12:
        raise ValueError("Archived interaction-grid invariance changed")
    return {
        "scope": "Summaries of retained sweep outputs; original regressions are not refitted.",
        "combinations": {"1": len(g1), "2": len(g2), "3": len(g3), "5": len(g5), "6": len(g6), "7": len(g7)},
        "grid1_reference_score_sum_r2": float(g1_reference.r2_bev_sum.iloc[0]),
        "grid2_deviation_range": [float(g2.deviation.min()), float(g2.deviation.max())],
        "grid3_ordinary_n": sorted(g3.n_pairs.unique().astype(int).tolist()),
        "grid3_ordinary_r_range": [float(g3.r_bev.min()), float(g3.r_bev.max())],
        "grid5_retained_draws": len(sample), "grid5_stored_r2": float(sample.r2_bev_hpi.iloc[0]),
        "grid6_reference_commuting_score": float(g6.loc[g6.scheme_name.eq("A (Current)"), "commute_score"].iloc[0]),
        "grid6_commuting_cv_pct": float(cv.commute_score), "grid6_scheme_mean_cv_pct": float(cv.mean_suitability),
        "grid7_max_within_pair_spread": float((by_pair.max() - by_pair.min()).max()),
    }


def run_assessments(root: Path, out: Path) -> dict:
    """Recompute all 280 vehicle settings and the purpose perturbation experiments."""
    from reproduce import compare_csv

    out.mkdir(parents=True, exist_ok=True)
    factors = pd.read_csv(root / "data/assessments/component_factors.csv.gz", float_precision="round_trip")
    schemes = {"A": (.6, .3, .1), "B": (.5, .3, .2), "C": (.7, .2, .1), "D": (.6, .2, .2), "E": (.4, .3, .3), "F": (.5, .2, .3), "G": (.4, .4, .2), "H": (.8, .1, .1)}
    if len(factors) != 46359:
        raise ValueError("Vehicle diagnostic requires all study OAs")
    rows = []
    for scheme, weights in schemes.items():
        for bonus in (0., .1, .2, .3, .4, .5, .6):
            for threshold in (.1, .2, .3, .4, .5):
                assigned = assign_grid4(factors, weights, bonus, threshold)
                rows.append({"scheme": scheme, "household_size_weight": weights[0], "age_weight": weights[1], "children_weight": weights[2],
                    "children_bonus": bonus, "children_threshold": threshold, "assignment_margin": .3, "two_seater_threshold": .7, "four_seater_threshold": .3,
                    **summarize_grid4(assigned.assignment)})
    pd.DataFrame(rows).to_csv(out / "grid4_results.csv", index=False)
    compare_csv(out / "grid4_results.csv", root / "data/reference/assessments/grid4_results.csv")
    config = json.loads((root / "data/parameters/scoring_weights.json").read_text())
    summary, baseline, _ = calculate(config)
    for frame, name in ((summary, "purpose_summary.csv"), (baseline, "purpose_baseline.csv")):
        frame.to_csv(out / name, index=False)
        compare_csv(out / name, root / "data/reference/assessments" / name)
    archived = summarize_archived_grids(root / "data/assessments/archived")
    (out / "component_grid_summary.json").write_text(json.dumps(archived, indent=2) + "\n")
    return {"vehicle_grid_rows_verified": len(rows), "purpose_result_tables_verified": 2,
            "archived_grid_summaries": archived, "auxiliary_claims": auxiliary_claims(root, out)}
