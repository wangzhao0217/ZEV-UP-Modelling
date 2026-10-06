"""Recompute manuscript regressions and figures from aggregate postcode data."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from scripts.mot_validation import compute_score_deltas, plot_scatter


def r_squared(frame: pd.DataFrame, x: str, y: str) -> float:
    """Return ordinary linear R-squared on finite matched observations."""
    xv, yv = frame[x].to_numpy(float), frame[y].to_numpy(float)
    valid = np.isfinite(xv) & np.isfinite(yv)
    if valid.sum() < 2 or np.ptp(xv[valid]) == 0:
        return float("nan")
    predicted = np.polyval(np.polyfit(xv[valid], yv[valid], 1), xv[valid])
    return float(1 - np.sum((yv[valid] - predicted)**2) / np.sum((yv[valid] - yv[valid].mean())**2))


def run_postcode(root: Path, out: Path, figures: Path, *, return_namespace: bool = False) -> dict:
    """Rebuild the original matched samples without distributing vehicle records."""
    from reproduce import compare_csv

    inputs = root / "data/postcode"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(inputs / "table_fy_2011.csv")
    current = pd.read_csv(inputs / "table_fy_1920_v_2022.csv")
    delta = compute_score_deltas(old, current)
    hpi = pd.read_csv(inputs / "val_hpi_2022.csv")
    segments = pd.read_csv(inputs / "segment_analysis.csv")
    edges = pd.read_parquet(inputs / "edge_scores.parquet")
    premium = segments.dropna(subset=["top20_score_sum_2022", "premium_registrations"])
    lower = segments.dropna(subset=["band3080_score_sum_2022", "lower_price_registrations"])
    r_primary, p_primary = pearsonr(delta.score_sum_delta, delta.bev_count)
    highest = current.sort_values("bev_count", ascending=False).iloc[0]
    highland = current[current.pc_area.isin(["IV", "PH", "KW"])]
    area_names = {"G": "Glasgow", "EH": "Edinburgh", "AB": "Aberdeen", "IV": "Inverness", "DD": "Dundee", "PA": "Paisley", "FK": "Falkirk", "KY": "Kirkcaldy", "PH": "Perth"}
    values = {
        "r_primary": float(r_primary), "p_primary": float(p_primary),
        "r2_primary": round(r_squared(delta, "score_sum_delta", "bev_count"), 3),
        "r_hpi_adoption": float(pearsonr(hpi.tx_total_gbp_million, hpi.score_sum).statistic),
        "r_hpi_bev": float(pearsonr(hpi.tx_total_gbp_million, hpi.bev_count).statistic),
        "bev_fy_1920_total": int(current.bev_count.sum()),
        "r2_segment_top20_premium": round(r_squared(premium, "top20_score_sum_2022", "premium_registrations"), 3),
        "r2_segment_band3080_lower": round(r_squared(lower, "band3080_score_sum_2022", "lower_price_registrations"), 3),
        "highest_area_name": str(highest.pc_area), "highest_area_fullname": area_names.get(highest.pc_area, highest.pc_area),
        "highest_bev_count": int(highest.bev_count), "highland_bev_count": int(highland.bev_count.sum()),
        "highland_adoption_sum": float(round(highland.score_sum.sum(), 0)),
    }
    rows = []
    for lo, hi in ((.30, .80), (.25, .80), (.40, .70), (.20, .90), (.30, .85)):
        row = {"lower_edge": lo, "upper_edge": hi}
        ap = edges.final_adoption_propensity
        for segment, mask, outcome in (("lower", ap.between(lo, hi), "lower_price_registrations"), ("premium", ap > hi, "premium_registrations")):
            scores = edges.loc[mask].groupby("pc_area").final_adoption_propensity.sum().rename("score_sum")
            matched = segments[["pc_area", outcome]].merge(scores, on="pc_area", how="left").dropna(subset=["score_sum", outcome])
            row[f"{segment}_r2"], row[f"{segment}_n"] = round(r_squared(matched, "score_sum", outcome), 3), len(matched)
        rows.append(row)
    band_edge_fits = pd.DataFrame(rows)
    band_edge_fits.to_csv(out / "band_edge_fits.csv", index=False)
    compare_csv(out / "band_edge_fits.csv", root / "data/reference/validation/band_edge_fits.csv")
    pd.DataFrame([{"metric": k, "value": values[k]} for k in ("r2_segment_band3080_lower", "r2_segment_top20_premium")]).to_csv(out / "segment_band_r2.csv", index=False)
    # The band-edge summary is an unused historical read retained by the manuscript.
    source = root / "data/reference/validation/band_edge_evidence_summary.csv"
    (out / source.name).write_bytes(source.read_bytes())
    plots = [
        (delta, "score_sum_delta", "bev_count", "Δ Adoption propensity score (2022 − 2011)", "Passing MOT BEV count (2019–2020 cohort)", "mot_temporal.png"),
        (premium, "top20_score_sum_2022", "premium_registrations", "Total AP score, AP > 0.8", "Premium MOT BEV count", "premium_segment.png"),
        (lower, "band3080_score_sum_2022", "lower_price_registrations", r"Total AP score, $0.3 \leq AP \leq 0.8$", "Lower-price MOT BEV count", "lower_price_segment.png"),
        (hpi, "tx_total_gbp_million", "score_sum", "Property transaction value (£ million, ≤2022)", "2022 Adoption propensity score", "house_price.png"),
    ]
    for frame, x, y, xlabel, ylabel, name in plots:
        fig, ax = plot_scatter(frame, x, y, title="", xlabel=xlabel, ylabel=ylabel, font_scale=1.35, save_path=None)
        if name == "mot_temporal.png":
            ax.axvline(0, color="grey", linewidth=.8, linestyle=":")
        fig.savefig(figures / name, dpi=300, bbox_inches="tight")
        plt.close(fig)
    if return_namespace:
        return {**values, "band_edge_fits": band_edge_fits}
    return {**values, "band_edge_rows_verified": len(rows), "figures_generated": len(plots)}
