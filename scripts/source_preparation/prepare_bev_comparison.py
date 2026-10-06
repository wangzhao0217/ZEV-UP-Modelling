"""Compare frozen Paper 1 AP with BEV rates on matching observed geography.

Run ``python paper/scripts/datazone_ap_validation.py``. Outputs retain the
correlations, analysis inputs, geographic coverage and source hashes.
This comparison does not rebuild AP or estimate frugal-EV substitution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper/scripts"))
import datazone_validation as dzv

SCORE_COL = "final_adoption_propensity"
REGIONS = ("HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans")
EXPECTED_OAS = 46359
QUARTER = "2025 Q4"
FROZEN_INPUTS = ROOT / "output/paper1_r2_v4/2026-09-12/frozen_baseline"
OUT_DIR = ROOT / "output/revision_validation"
PANEL = ROOT / "output/study3/feasible_market_long.parquet"
VEH = ROOT / "data/mot/dft_veh0145_lsoa.csv"
LOOKUP = ROOT / "data/demographic_data/2022_downloads/oa_dz_iz_lookup/OA22_DZ22_IZ22.csv"


def digest(path: Path) -> str:
    """Return SHA-256 without loading a large file into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_ap_by_oa(frozen_inputs: Path = FROZEN_INPUTS) -> pd.DataFrame:
    """Load seven frozen regions; reject missing, duplicate or invalid OAs."""
    frames = []
    for region in REGIONS:
        path = frozen_inputs / region / "2022/adoption_propensity_categorized.gpkg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen AP input: {path}")
        frames.append(pyogrio.read_dataframe(
            path, columns=["geo_code", SCORE_COL], read_geometry=False))
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != EXPECTED_OAS or frame["geo_code"].isna().any() or not frame["geo_code"].is_unique:
        raise ValueError(f"Expected exactly {EXPECTED_OAS:,} unique study OAs")
    if not np.isfinite(frame[SCORE_COL]).all() or not frame[SCORE_COL].between(0, 1).all():
        raise ValueError("Frozen AP scores must be finite and within [0, 1]")
    return frame.sort_values("geo_code").reset_index(drop=True)


def _sp(frame: pd.DataFrame, a: str, b: str) -> float:
    """Return Spearman correlation on an explicitly matched input frame."""
    return float(spearmanr(frame[a], frame[b]).statistic)


def comparison_row(
    label: str, scale: str, stats: dict[str, object], policy: str,
) -> dict[str, object]:
    """Format a correlation and its existing approximate Fisher-z interval."""
    rho, n = float(stats["spearman_rho"]), int(stats["n"])
    z = np.arctanh(rho)
    limits = np.tanh(z + np.array([-1, 1]) * 1.96 / np.sqrt(n - 3))
    return {
        "comparison": label, "scale": scale, "n": n, "spearman_rho": rho,
        "ci95_low": float(limits[0]), "ci95_high": float(limits[1]),
        "pearson_r2_rate": stats["pearson_r2"],
        "size_baseline_rho": stats["size_baseline_rho"], "sample_policy": policy,
    }


def main(argv: list[str] | None = None) -> None:
    """Reproduce the observed-support comparison and retain its input trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-inputs", type=Path, default=FROZEN_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    lookup = pd.read_csv(LOOKUP, dtype=str)[["OA22", "DZ22", "IZ22"]].drop_duplicates()
    if not lookup["OA22"].is_unique or lookup.isna().any().any():
        raise ValueError("OA geography must have one complete mapping per OA")
    ap = load_ap_by_oa(args.frozen_inputs)
    panel = pd.read_parquet(
        PANEL, columns=["geo_code", "A_it", "households"],
        filters=[("scenario", "=", "neutral_lr_central"), ("year", "=", 2025)],
    )
    oa = ap.merge(panel, on="geo_code", how="left", validate="one_to_one")
    if oa[["households", "A_it"]].isna().any().any() or not np.isfinite(oa[["households", "A_it"]]).all().all():
        raise ValueError("Every frozen OA requires finite household weights and affordability")
    if (oa["households"] <= 0).any() or not set(oa["geo_code"]).issubset(set(lookup["OA22"])):
        raise ValueError("Every frozen OA requires positive household weight and a geographic mapping")
    observed = dzv.load_plugin_counts_by_datazone(VEH, quarter=QUARTER)
    model = dzv.aggregate_feasibility_to_datazone(oa, lookup, score_column=SCORE_COL)
    affordability = dzv.aggregate_feasibility_to_datazone(oa, lookup, score_column="A_it")
    affordability = affordability.rename(columns={"feas": "A_it"})[["DZ22", "A_it"]]
    joined = model.merge(affordability, on="DZ22", validate="one_to_one").merge(
        observed, on="DZ22", how="inner", validate="one_to_one")
    joined["bev_per_1000_households"] = 1000 * joined["bev"] / joined["households"]
    dz_stats = dzv.correlate_datazone(model, observed)
    rho = _sp(joined, "feas", "bev_per_1000_households")
    income_rho = _sp(joined, "A_it", "bev_per_1000_households")
    ap_income_rho = _sp(joined, "feas", "A_it")
    partial = (rho - ap_income_rho * income_rho) / np.sqrt(
        (1 - ap_income_rho**2) * (1 - income_rho**2))
    mapping = lookup[["DZ22", "IZ22"]].drop_duplicates()
    model_iz, obs_iz, coverage = dzv.rollup_observed_to_intermediate_zone(model, observed, mapping)
    model_complete, obs_complete, _ = dzv.rollup_observed_to_intermediate_zone(
        model, observed, mapping, complete_only=True)
    iz_stats = dzv.correlate_datazone(model_iz, obs_iz, key="IZ22")
    complete_stats = dzv.correlate_datazone(model_complete, obs_complete, key="IZ22")
    included = coverage[coverage["matched_dz"] > 0]
    if not (coverage["model_dz"] == coverage["total_dz"]).all():
        raise ValueError("Frozen OA inputs do not cover every mapped Data Zone")
    household_coverage = 100 * included["observed_households"].sum() / included["households"].sum()
    rows = [
        comparison_row("AP propensity vs BEV rate (Data Zone, headline)", "Data Zone", dz_stats, "observed Data Zones"),
        {"comparison": "AP propensity vs rate | income (partial)", "scale": "Data Zone", "n": len(joined),
         "spearman_rho": float(partial), "sample_policy": "same observed Data Zones; affordability-adjusted ranks"},
        {"comparison": "income (A_it) vs rate (income-only baseline)", "scale": "Data Zone", "n": len(joined),
         "spearman_rho": income_rho, "sample_policy": "same observed Data Zones"},
        comparison_row("AP propensity vs BEV rate (Intermediate Zone, observed support)", "Intermediate Zone", iz_stats, "matching observed constituent Data Zones"),
        comparison_row("AP propensity vs BEV rate (Intermediate Zone, complete coverage)", "Intermediate Zone", complete_stats, "all constituent Data Zones observed"),
    ]
    rows[3]["household_coverage_pct"] = float(household_coverage)
    rows[4]["household_coverage_pct"] = 100.0
    sources = [Path(__file__), Path(dzv.__file__), PANEL, VEH, LOOKUP,
               args.frozen_inputs / "snapshot_manifest.json"]
    sources.extend(args.frozen_inputs / r / "2022/adoption_propensity_categorized.gpkg" for r in REGIONS)
    identities = {}
    for path in sources:
        resolved = path.resolve()
        name = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
        identities[name] = digest(path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "study_oas": len(oa),
        "quarter": QUARTER, "fuel": "BATTERY ELECTRIC", "keepership": "PRIVATE",
        "panel_selection": {"scenario": "neutral_lr_central", "year": 2025},
        "missing_counts": "excluded, never zero-filled; recorded zeros retained",
        "ci_method": "Fisher-z approximation; spatial dependence is not modelled",
        "all_intermediate_zones": len(coverage), "included_intermediate_zones": len(included),
        "complete_intermediate_zones": int(included["complete"].sum()),
        "partial_intermediate_zones": int((~included["complete"]).sum()),
        "household_coverage_pct": float(household_coverage), "source_sha256": identities,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "datazone_ap_validation.csv", index=False)
    coverage.to_csv(args.output_dir / "datazone_ap_iz_coverage.csv", index=False)
    joined.to_csv(args.output_dir / "datazone_ap_observed_inputs.csv", index=False)
    for suffix, m, o in [("observed", model_iz, obs_iz), ("complete", model_complete, obs_complete)]:
        m.merge(o, on="IZ22", validate="one_to_one").to_csv(
            args.output_dir / f"datazone_ap_iz_{suffix}_inputs.csv", index=False)
    oa.to_csv(args.output_dir / "datazone_ap_oa_inputs.csv", index=False)
    (args.output_dir / "datazone_ap_validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(rows)[["comparison", "n", "spearman_rho"]].to_string(index=False))
    print(f"Observed household coverage: {household_coverage:.6f}%")


if __name__ == "__main__":
    main()
