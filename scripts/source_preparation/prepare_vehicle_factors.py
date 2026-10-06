"""Recalculate Grid 4 on the exact 46,359-OA Paper 1 study inventory.

Run from the repository root. Raw inputs and existing grid files are read only.
The existing EVTypeSelector supplies component factors; a vectorised selector
is checked against its public assignment method on three full-population runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from functions.ev_type_selection import EVTypeSelector
from functions.paper1_grid4 import assign_grid4, summarize_grid4

OUT = ROOT / "output/paper1_r2_f05_2026-10-04/grid4"
INPUT = ROOT / "data/demographic_data/2022_downloads/2022 match 2011/merged_demographics_2022_adoption_compat.gpkg"
FROZEN = ROOT / "output/paper1_r2_v4/2026-09-12/frozen_baseline"
SCHEMES = {"A": (.6, .3, .1), "B": (.5, .3, .2), "C": (.7, .2, .1),
           "D": (.6, .2, .2), "E": (.4, .3, .3), "F": (.5, .2, .3),
           "G": (.4, .4, .2), "H": (.8, .1, .1)}


def digest(path: Path) -> str:
    """Return a streaming SHA-256 for an input or output file."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Read identified inputs, execute and verify the grid, and write evidence."""
    logging.getLogger("functions.ev_type_selection").setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "scoring_weights.json"
    target_path = ROOT / "target_columns_2022.json"
    config = json.loads(config_path.read_text())
    targets = json.loads(target_path.read_text())["columns"]
    input_paths = [INPUT, config_path, target_path,
                   ROOT / "functions/ev_type_selection.py",
                   ROOT / "functions/paper1_grid4.py", Path(__file__)]
    inventory_parts = []
    for p in sorted(FROZEN.glob("*/2022/adoption_propensity_categorized.gpkg")):
        ids = pyogrio.read_dataframe(p, columns=["geo_code"], read_geometry=False)
        ids["region"] = p.parts[-3]
        inventory_parts.append(ids)
        input_paths.append(p)
    inventory = pd.concat(inventory_parts, ignore_index=True)
    assert len(inventory) == 46359 and inventory.geo_code.is_unique
    columns = list(dict.fromkeys(["geo_code"] + sum(
        [targets[k] for k in ["Housing", "Age", "Household Composition"]], [])))
    raw = pyogrio.read_dataframe(INPUT, columns=columns, read_geometry=False)
    assert raw.geo_code.is_unique
    extra = sorted(set(raw.geo_code) - set(inventory.geo_code))
    missing = sorted(set(inventory.geo_code) - set(raw.geo_code))
    assert not missing, missing
    data = inventory.merge(raw, on="geo_code", how="left", validate="one_to_one")
    numeric = data[columns[1:]].apply(pd.to_numeric, errors="raise")
    assert np.isfinite(numeric.to_numpy()).all() and (numeric >= 0).all().all()
    data.loc[:, columns[1:]] = numeric
    # Pass only the explicitly mapped non-overlapping census categories. In
    # particular, an additional 65+ roll-up must not duplicate detailed ages.
    demographic = data[columns].copy()
    selector = EVTypeSelector(weights=config)
    selector._validate_demographic_data(demographic, targets)
    factors = pd.DataFrame({
        "household_size_factor": selector._calculate_household_size_factor(demographic, targets),
        "age_factor": selector._calculate_age_adjustments(demographic, targets),
        "children_factor": selector._calculate_children_indicators(demographic, targets),
    })
    verification = []
    rows = []
    selected = None
    for scheme, weights in SCHEMES.items():
        for bonus in [0., .1, .2, .3, .4, .5, .6]:
            for threshold in [.1, .2, .3, .4, .5]:
                assigned = assign_grid4(factors, weights, bonus, threshold)
                result = {"scheme": scheme, "household_size_weight": weights[0],
                          "age_weight": weights[1], "children_weight": weights[2],
                          "children_bonus": bonus, "children_threshold": threshold,
                          "assignment_margin": .3, "two_seater_threshold": .7,
                          "four_seater_threshold": .3,
                          **summarize_grid4(assigned.assignment)}
                rows.append(result)
                if (scheme, bonus, threshold) in [("A", 0., .1), ("D", .3, .2), ("H", .6, .5)]:
                    cfg = copy.deepcopy(config)
                    cfg["assignment_weights"] = {
                        "household_size": weights[0], "age": weights[1],
                        "children_present": weights[2], "children_bonus": bonus,
                        "children_threshold": threshold, "assignment_threshold": .3,
                        "two_seater_threshold": .7, "four_seater_threshold": .3,
                    }
                    cfg["stage_7_parameters"] = {"enable_not_applicable": False,
                        "require_stage_6_filtering": False, "minimum_conversion_threshold": 0.}
                    check = EVTypeSelector(weights=cfg).assign_ev_type(demographic, target_columns=targets,
                        conversion_potential=pd.Series(1., index=demographic.index))
                    assert np.array_equal(check.ev_type_assignment, assigned.assignment)
                    np.testing.assert_allclose(check.two_seater_score, assigned.two_seater_score, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(check.four_seater_score, assigned.four_seater_score, rtol=0, atol=1e-15)
                    verification.append({"scheme": scheme, "bonus": bonus, "threshold": threshold,
                                         "OAs": len(check), "assignment_mismatches": 0})
                if (scheme, bonus, threshold) == ("A", 0., .1):
                    selected = result
                    evidence = pd.concat([inventory, factors, assigned], axis=1)
                    evidence.to_csv(OUT / "selected_oa_assignments.csv.gz", index=False)
    grid = pd.DataFrame(rows)
    assert len(grid) == 280 and grid.total_areas.eq(46359).all()
    grid.to_csv(OUT / "grid4_results_46359.csv", index=False)
    inventory.to_csv(OUT / "study_oa_inventory.csv", index=False)
    factors.to_csv(OUT / "component_factors.csv.gz", index=False)
    summary = {"selected": selected, "rows": len(grid), "oa_count": len(inventory),
        "all_oa_share_min_pct": float(100 * grid.two_seater_share_all.min()),
        "all_oa_share_max_pct": float(100 * grid.two_seater_share_all.max()),
        "all_oa_rows_above_10pct": int((grid.two_seater_share_all > .1).sum()),
        "binary_share_min_pct": float(100 * grid.two_seater_share_binary.min()),
        "binary_share_max_pct": float(100 * grid.two_seater_share_binary.max()),
        "binary_rows_above_10pct": int((grid.two_seater_share_binary > .1).sum()),
        "closest_binary_target": grid.loc[(grid.two_seater_share_binary - .1).abs().idxmin()].to_dict(),
        "public_selector_verification": verification,
        "source_oa_count": len(raw), "excluded_source_oa_ids": extra,
        "missing_study_oa_ids": missing,
        "component_columns": {k: targets[k] for k in ["Housing", "Age", "Household Composition"]},
        "configuration_scope": "Grid 4 component diagnostic; all study OAs; Stage 6 filter disabled; final trip allocations unchanged",
        "denominators": {"two_seater_share_all": "all 46359 OAs including mixed",
                         "two_seater_share_binary": "two-seater plus four-seater OAs, excluding mixed",
                         "chi2_binary": "same binary subset as two_seater_share_binary; descriptive only"},
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in input_paths}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["component_columns", "inputs_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
