"""Reproduce the final Scotland screening analysis from the released inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
REGIONS = ("HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans")


def digest(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(root: Path = ROOT) -> dict[str, int]:
    """Check every listed input and source; reject missing or changed files."""
    manifest = json.loads((root / "manifest.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            failures.append(f"changed: {entry['path']}")
    if failures:
        raise ValueError("Package integrity check failed:\n" + "\n".join(failures))
    return {"files_checked": len(manifest["files"])}


def compare_csv(actual: Path, expected: Path, *, rtol: float = 1e-8, atol: float = 1e-8) -> None:
    """Compare table shape, labels, missingness and numeric values in row order."""
    import numpy as np
    import pandas as pd

    left, right = pd.read_csv(actual), pd.read_csv(expected)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise ValueError(f"Table shape/columns differ: {expected.name}")
    for col in right:
        a, b = left[col], right[col]
        if pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True,
                                       err_msg=f"{expected.name}: {col}")
        elif not a.fillna("<missing>").equals(b.fillna("<missing>")):
            raise ValueError(f"Table labels differ: {expected.name}: {col}")


def model_results(out: Path) -> dict[str, object]:
    """Recompute screening, shared capacity, scenarios, chaining and circuity."""
    import numpy as np
    import pandas as pd
    import pyogrio

    from functions.paper1_chaining import two_stop_geometry
    from functions.paper1_feasibility import origin_constrained_weights
    from functions.paper1_scenarios import charging_screen_scenario

    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "data/parameters/scoring_weights.json").read_text())
    beta = pd.read_csv(ROOT / "data/parameters/distance_frequency_wide.csv").set_index("NPT purpose").beta
    profiles = {
        "commuting": "Commuting", "business": "Commuting", "education": "Commuting", "escort": "Commuting",
        "shopping": "Shopping", "visit hospital or other health": "Shopping",
        "other personal business": "Shopping", "other journey": "Shopping",
        "visiting friends or relatives": "Social", "eating/drinking": "Leisure",
        "sport/entertainment": "Leisure", "holiday/daytrip": "Leisure",
    }
    collected, comparisons, sensitivity, chains, circuity, oa_counts = [], [], [], [], [], []
    checks = 0
    for region in REGIONS:
        print(f"Model: {region}", flush=True)
        flows = pd.read_parquet(ROOT / f"data/flows/{region}.parquet")
        sites = pd.read_csv(ROOT / f"data/stations/{region}.csv")
        if not flows.flow_id.is_unique or not sites.station_position.equals(pd.Series(range(len(sites)), name="station_position")):
            raise ValueError(f"Nonunique flows or unordered station positions: {region}")
        points = sites.N_points.to_numpy()

        def evaluate(frame: pd.DataFrame, band: tuple[float, float] = (.3, .8),
                     weighted: bool = True, fraction: float = .5) -> tuple:
            return charging_screen_scenario(frame, points, band=band,
                purpose_weighted=weighted, require_charging_access=True,
                charging_config=config, medium_origin_fraction=fraction)

        totals, trips, capacity = evaluate(flows)
        f, w, a, d = (flows[c].to_numpy() for c in ("vol", "w", "ap", "d"))
        access = np.where(d <= 40, (flows.home_score >= .5) | flows.P_i.eq(1),
            ((flows.home_score >= .5) | flows.P_i.eq(1)) & flows.P_j.eq(1)) & (d <= 80)
        eligible = (a >= .3) & (a <= .8) & (d <= 80) & access
        np.testing.assert_allclose(trips.potential_volume, f * eligible * w, rtol=1e-12)
        np.testing.assert_allclose(totals["potential_volume"], totals["immediate_volume"] + totals["conditional_volume"], rtol=1e-12)
        totals.update(region=region, rows=len(flows), short_volume=float(f[d <= 3].sum()))
        collected.append(totals)
        directory = out / "regions" / region
        directory.mkdir(parents=True, exist_ok=True)
        reference = ROOT / "data/reference/model/regions" / region
        capacity["ocm_id"] = sites.ocm_id.to_numpy()
        capacity["point_count_missing"] = sites.point_count_missing.to_numpy()
        capacity.to_csv(directory / "destination_charging_capacity.csv", index=False)
        compare_csv(directory / "destination_charging_capacity.csv", reference / "destination_charging_capacity.csv")
        computed = flows[["purpose", "vol", "w"]].copy()
        for col in ("potential_volume", "immediate_volume", "conditional_volume"):
            computed[col] = trips[col].to_numpy()
        purpose = computed.groupby("purpose", as_index=False).agg(car_trips=("vol", "sum"),
            immediately_feasible=("immediate_volume", "sum"), constrained=("conditional_volume", "sum"),
            replaceable_trips=("potential_volume", "sum"), purpose_weight=("w", "first"))
        purpose["replacement_rate_pct"] = 100 * purpose.replaceable_trips / purpose.car_trips
        purpose.to_csv(directory / "replaceable_trips_by_purpose.csv", index=False)
        compare_csv(directory / "replaceable_trips_by_purpose.csv", reference / "replaceable_trips_by_purpose.csv")
        for weighted in (True, False):
            for band, label in (((.3, .8), "AP 0.3-0.8"), ((0., 1.), "No AP screen")):
                result, _, _ = evaluate(flows, band, weighted)
                comparisons.append({"region": region, "screen": label, "purpose_weighted": weighted, **result})
        for lo, hi in ((.3, .8), (.2, .8), (.4, .8), (.3, .7), (.3, .9), (.2, .9), (.4, .7)):
            result, _, _ = evaluate(flows, (lo, hi))
            sensitivity.append({"region": region, "family": "AP band", "scenario": f"{lo:.1f}-{hi:.1f}", **result})
        signed_beta = flows.purpose.str.casefold().map(profiles).map(beta).to_numpy()
        if not np.isfinite(signed_beta).all() or not (signed_beta < 0).all():
            raise ValueError("Every purpose needs a signed negative decay coefficient")
        groups = flows.origin.astype(str) + ":" + flows.purpose.str.casefold()
        for scale in (.5, .75, 1., 1.25, 1.5):
            alternative = flows.copy()
            multiplier = np.exp((scale - 1) * signed_beta * flows.gravity_distance_km)
            alternative["vol"] = origin_constrained_weights(f, multiplier, groups)
            result, _, _ = evaluate(alternative)
            sensitivity.append({"region": region, "family": "Decay multiplier", "scenario": str(scale), **result})
        for scale in (.75, 1., 1.25):
            alternative = flows.copy()
            alternative["w"] = np.minimum(1., w * scale)
            result, _, _ = evaluate(alternative)
            sensitivity.append({"region": region, "family": "Purpose-weight multiplier", "scenario": str(scale), **result})
        for fraction in (.25, .5, .75):
            result, _, _ = evaluate(flows, fraction=fraction)
            sensitivity.append({"region": region, "family": "Medium public-origin energy share", "scenario": str(fraction), **result})
        home = eligible & (d <= 40) & flows.home_score.ge(.5).to_numpy()
        subset = float((f * w)[home].sum())
        geometry = two_stop_geometry(d[home], (f * w)[home])
        for exposure in (.23, .29):
            for failure in (.25, .5, .75, 1.):
                loss = exposure * failure * subset
                chains.append({"region": region, "exposure": exposure, "failure": failure,
                    "baseline_volume": float(f.sum()), "potential_volume": totals["potential_volume"],
                    "immediate_volume": totals["immediate_volume"], "home_subset_volume": subset,
                    "loss_volume": loss, "adjusted_potential_volume": totals["potential_volume"] - loss,
                    "adjusted_immediate_volume": totals["immediate_volume"] - loss, **geometry})
        e = flows.euclidean_km.to_numpy()
        qs = np.quantile(d[e > 0] / e[e > 0], [0, .25, .5, .75, 1])
        circuity.append(dict(region=region, n_routes=len(flows), **dict(zip(
            ("min_circuity", "q25_circuity", "median_circuity", "q75_circuity", "max_circuity"), qs))))
        rc = np.where(d <= 40, "feasible", np.where(d <= 80, "constrained", "infeasible"))
        ec = np.where(e <= 40, "feasible", np.where(e <= 80, "constrained", "infeasible"))
        name = f"{region}_circuity_conversion_comparison.csv"
        pd.DataFrame([{"category": c, "euclidean_count": int((ec == c).sum()),
            "routed_count": int((rc == c).sum()), "euclidean_pct": 100 * float((ec == c).mean()),
            "routed_pct": 100 * float((rc == c).mean()), "shift": int((ec == c).sum() - (rc == c).sum()),
            "region": region, "total_routes": len(flows), "median_circuity": qs[2]}
            for c in ("feasible", "constrained", "infeasible")]).to_csv(directory / name, index=False)
        compare_csv(directory / name, reference / name)
        areas = pyogrio.read_dataframe(ROOT / f"data/areas/{region}.gpkg", read_geometry=False)
        oa_counts.append({"region": region, "total_areas": len(areas), "target_areas": int(areas.final_adoption_propensity.between(.3, .8).sum())})
        scores = areas.set_index("geo_code")
        np.testing.assert_allclose(flows.ap, flows.origin_oa.map(scores.final_adoption_propensity), rtol=0, atol=1e-12)
        np.testing.assert_allclose(flows.home_score, flows.origin_oa.map(scores.home_charging_feasibility), rtol=0, atol=1e-12)
        summary = json.loads((reference / f"{region}_comprehensive_summary.json").read_text())
        # Regional vehicle allocations and non-trip diagnostic metadata are fixed inputs.
        for key, value in (("total_baseline_trips", totals["baseline_volume"]), ("final_conversion_potential", totals["potential_volume"])):
            np.testing.assert_allclose(summary["conversion_potential"][key], value, rtol=1e-10)
        (directory / f"{region}_comprehensive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        checks += 3
    for name, rows in (("regional_metrics", collected), ("technical_comparison", comparisons),
                       ("sensitivity", sensitivity), ("chaining", chains), ("circuity_summary", circuity), ("oa_counts", oa_counts)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        compare_csv(out / f"{name}.csv", ROOT / f"data/reference/model/{name}.csv")
        checks += 1
    national = pd.DataFrame(collected).drop(columns="region").sum().to_dict()
    expected = json.loads((ROOT / "data/reference/model/national_metrics.json").read_text())
    if national.keys() != expected.keys():
        raise ValueError("National metric names changed")
    for key in national:
        np.testing.assert_allclose(national[key], expected[key], rtol=1e-10, atol=1e-8, err_msg=key)
    (out / "national_metrics.json").write_text(json.dumps(national, indent=2) + "\n")
    return {"tables_verified": checks, "national_metrics_verified": len(national), "national": national}


def reproduce() -> None:
    """Execute every supported final-result analysis and record actual checks."""
    import logging
    from threadpoolctl import threadpool_limits
    from scripts.assessments import run_assessments
    from scripts.postcode import run_postcode
    from scripts.thresholds import run_thresholds
    from scripts.validation import run_validation

    logging.getLogger("functions.charging_infrastructure").setLevel(logging.WARNING)

    report = {"integrity": verify_files()}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    # Delete an old success marker before work: interrupted runs must not look successful.
    (out / "verification.json").unlink(missing_ok=True)
    with threadpool_limits(limits=1):
        report["model"] = model_results(out / "model")
        report["validation"] = run_validation(ROOT, out / "validation")
        report["postcode"] = run_postcode(ROOT, out / "validation", ROOT / "manuscript/figures")
        report["thresholds"] = run_thresholds(ROOT, out / "thresholds", ROOT / "manuscript/figures")
        report["assessments"] = run_assessments(ROOT, out / "assessments")
    report["status"] = "passed"
    report["scope"] = "Downstream reproduction from frozen model and aggregated statistical inputs; see DATA.md for upstream and archived-only boundaries."
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    """Provide check, run and render entry points from any working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "run", "render"), default="run")
    parser.add_argument("--format", choices=("html", "pdf"), default="html")
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(verify_files(), indent=2))
    elif args.command == "run":
        reproduce()
    else:
        verify_files()
        verification = ROOT / "results/verification.json"
        if not verification.is_file() or json.loads(verification.read_text()).get("status") != "passed":
            raise SystemExit("Run python reproduce.py first.")
        environment = dict(os.environ, RETICULATE_PYTHON=sys.executable)
        subprocess.run(["quarto", "render", "manuscript/manuscript.qmd", "--to", args.format,
                        "--execute", "--no-cache"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
