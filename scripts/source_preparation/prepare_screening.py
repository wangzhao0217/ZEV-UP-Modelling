"""Recalculate Reviewer 3 diagnostics without changing the v4 scoring baseline.

Reuses the verified corrected-gravity routing release, including routes <=3 km.
The v4 OA scores, home scores and purpose parameters are frozen. Flow-dependent
totals and the explicit station-capacity assessment are recalculated. Ancillary
area diagnostics remain legacy frozen inputs: five layers use older OA identities
and require the separately verified canonical-2022 correction before reporting.
The correction is published by materialize_paper1_pr42_reporting.py.
An explicit new --output-dir is required; the historical archive is read-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functions.paper1_chaining import two_stop_geometry
from functions.charging_infrastructure import ChargingInfrastructureAnalyzer
from functions.paper1_feasibility import origin_constrained_weights
from functions.paper1_gravity_distance import recover_gravity_distance
from functions.paper1_scenarios import charging_screen_scenario
from functions.paper1_purpose import canonical_purpose
from functions.trip_purpose import TripPurposeAnalyzer

HISTORICAL_ARCHIVE = ROOT / "output/paper1_r2_v4/2026-09-12"
REGIONS = ("HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans")


def sha256(path: Path) -> str:
    """Hash a source file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_area(points: gpd.GeoDataFrame, areas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Match generated endpoints to frozen v4 OAs, breaking boundary ties by code."""
    result = gpd.sjoin_nearest(points, areas, how="left", distance_col="join_distance_m")
    result = result.sort_values(["row", "join_distance_m", "geo_code"])
    result = result.drop_duplicates("row").set_index("row").reindex(points.index)
    if result.geo_code.isna().any():
        raise ValueError("An endpoint lacks an OA assignment")
    return result


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Require a new explicit destination and identify checkpoint/frozen inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "output/paper1_r2_rebuild/2026-09-07")
    parser.add_argument("--frozen-inputs", type=Path,
                        default=HISTORICAL_ARCHIVE / "frozen_baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("checkpoint", "frozen_inputs", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for source in (HISTORICAL_ARCHIVE.resolve(), args.checkpoint, args.frozen_inputs):
        if (args.output_dir == source or source in args.output_dir.parents
                or args.output_dir in source.parents):
            parser.error("--output-dir must be separate from the preserved archives and inputs")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    return args


def main(argv: list[str] | None = None) -> None:
    """Publish compatible regional totals and independently check all accounting."""
    args = parse_arguments(argv)
    SOURCE, BASE_INPUTS, OUT = args.checkpoint, args.frozen_inputs, args.output_dir
    analysis_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=False)
    source_hashes: dict[str, str] = {}

    def path_label(path: Path) -> str:
        """Use repository-relative provenance where possible, otherwise an explicit path."""
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def record(path: Path) -> Path:
        source_hashes[path_label(path)] = sha256(path)
        return path

    beta = pd.read_csv(record(ROOT / "EV_od/inputdata/distance_frequency_wide.csv"))
    if not (beta.beta < 0).all():
        raise ValueError("This release expects signed negative coefficients; inspect the source CSV")
    decay = -beta.beta.to_numpy()
    distances = np.array([0., 1., 5., 40., 80.])
    np.testing.assert_allclose(np.exp(beta.beta.to_numpy()[:, None] * distances),
                               np.exp(-decay[:, None] * distances))
    beta["positive_decay_magnitude"] = decay
    beta.to_csv(OUT / "gravity_sign_audit.csv", index=False)
    for name in ("R/gravity_model.R", "R/utility_trips_ev.R"):
        record(ROOT / "EV_od" / name)
    record(Path(__file__))
    record(ROOT / "functions/paper1_scenarios.py")
    record(ROOT / "functions/paper1_gravity_distance.py")
    record(ROOT / "functions/charging_infrastructure.py")
    record(BASE_INPUTS / "snapshot_manifest.json")
    boundaries = gpd.read_file(record(ROOT / "data/la_regions_scotland_bfe_simplified_2023.geojson")).to_crs(27700)
    chargers = gpd.read_file(record(ROOT / "data/charger_location_data/ocm_scotland_points_local.geojson")).to_crs(27700)
    occupancy = json.loads(record(ROOT / "paper/Paper_1_revision_R2/charging_occupancy_R2.json").read_text())
    occupancy_map = {k.casefold(): v for k, v in occupancy["occupancy_by_purpose"].items()}
    config = json.loads(record(ROOT / "scoring_weights.json").read_text())
    purpose_analyzer = TripPurposeAnalyzer(scoring_weights=config)
    infrastructure = ChargingInfrastructureAnalyzer()
    collected, comparisons, sensitivities, chains, populations, circuities, oa_rows = [], [], [], [], [], [], []
    for region in REGIONS:
        print(f"{region}: reconstructing all routed trips with frozen v4 scores", flush=True)
        old = BASE_INPUTS / region / "2022"
        directory = OUT / "regional_output" / region / "2022"
        directory.mkdir(parents=True, exist_ok=True)
        routes = pd.read_parquet(record(SOURCE / "routing" / region / "routed_attributes.parquet"))
        route_meta = json.loads(record(SOURCE / "routing" / region / "routing_manifest.json").read_text())
        if not routes.flow_key.is_unique or len(routes) != route_meta["rows_routed"]:
            raise ValueError("Routing ledger and unique flow population disagree")
        np.testing.assert_allclose(routes.car.sum(), route_meta["routed_car_person_volume"], rtol=1e-12)
        if (routes[["origin_snap_m", "destination_snap_m"]] > route_meta["max_snap_m"]).any().any():
            raise ValueError("The routing snap bound was exceeded")
        # odjitter scales every numeric ancillary column by child multiplicity.
        # Recover the parent distance before fixed-support decay reweighting.
        gravity_distance = recover_gravity_distance(routes)
        areas = gpd.read_file(record(old / "adoption_propensity_categorized.gpkg"), columns=[
            "geo_code", "final_adoption_propensity", "home_charging_feasibility",
        ]).to_crs(27700)
        if not areas.geo_code.is_unique:
            raise ValueError("Duplicate baseline OAs")
        oa_rows.append({"region": region, "total_areas": len(areas), "target_areas": int(
            areas.final_adoption_propensity.between(.3, .8).sum())})
        points = gpd.GeoDataFrame({"row": routes.index}, geometry=gpd.points_from_xy(
            routes.origin_lng, routes.origin_lat), crs=4326).to_crs(27700)
        origins = nearest_area(points, areas)
        # Apply v4's printed W equation, without the undocumented temporal blend
        # or missing-join defaults in the older flow cache.
        old_flows = pd.read_parquet(record(SOURCE / "current/attributes" / f"{region}.parquet"))
        weight_groups = old_flows.groupby(old_flows.purpose.str.casefold()).w
        weights = pd.Series({p.casefold(): purpose_analyzer.get_suitability_weight(canonical_purpose(p))
                             for p in routes.purpose.unique()})
        pd.DataFrame({"purpose": weights.index, "v4_equation_weight": weights.to_numpy(),
            "old_minimum": weight_groups.min().reindex(weights.index).to_numpy(),
            "old_maximum": weight_groups.max().reindex(weights.index).to_numpy()}
            ).to_csv(directory / "purpose_weight_audit.csv", index=False)
        sites = gpd.clip(chargers, boundaries[boundaries.Region.eq(region)])
        sites = sites.sort_values("ocm_id").reset_index(drop=True)
        if not sites.ocm_id.is_unique:
            raise ValueError("A station would be counted more than once")
        xy = np.column_stack((sites.geometry.x, sites.geometry.y))
        # The manuscript's origin-access test uses the OA centroid.
        centroid = areas.geometry.centroid
        origin_access = infrastructure.assign_accessible_chargers(
            np.column_stack((centroid.x, centroid.y)), xy)
        origin_positions = pd.Series(origin_access.charging_station_position.to_numpy(), index=areas.geo_code)
        destinations = gpd.GeoSeries(gpd.points_from_xy(routes.dest_lng, routes.dest_lat), crs=4326).to_crs(27700)
        destination_access = infrastructure.assign_accessible_chargers(
            np.column_stack((destinations.x, destinations.y)), xy)
        station_position = destination_access.charging_station_position.to_numpy()
        origin_station_position = origins.geo_code.map(origin_positions).to_numpy(dtype=int)
        flows = pd.DataFrame({
            "flow_id": routes.flow_key, "origin": routes.geo_code1, "purpose": routes.purpose,
            "origin_oa": origins.geo_code.to_numpy(), "vol": routes.car,
            "d": routes.distance_km, "euclidean_km": routes.length_euclidean_jittered,
            "gravity_distance_km": gravity_distance.gravity_distance_km,
            "gravity_disaggregation_multiplicity": gravity_distance.gravity_disaggregation_multiplicity,
            "archived_divided_gravity_distance_km": gravity_distance.archived_divided_gravity_distance_km,
            "ap": origins.final_adoption_propensity.to_numpy(),
            "home_score": origins.home_charging_feasibility.to_numpy(),
            "w": routes.purpose.str.casefold().map(weights),
            "occupancy": routes.purpose.str.casefold().map(occupancy_map),
            "P_i": (origin_station_position >= 0).astype(int),
            "P_j": (station_position >= 0).astype(int),
            "origin_charging_station_position": origin_station_position,
            "charging_station_position": station_position,
        })
        if flows.isna().any().any():
            raise ValueError("Missing flow input; no implicit defaults are permitted")
        station_points = pd.to_numeric(sites.num_points, errors="raise").fillna(0).to_numpy()

        def evaluate(frame: pd.DataFrame, band: tuple[float, float] = (.3, .8),
                     weighted: bool = True, origin_fraction: float = .5) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
            return charging_screen_scenario(frame, station_points, band=band,
                purpose_weighted=weighted, require_charging_access=True,
                charging_config=config, medium_origin_fraction=origin_fraction)

        totals, trips, capacity = evaluate(flows)
        f, w, a, d = (flows[c].to_numpy() for c in ("vol", "w", "ap", "d"))
        access = np.where(d <= 40, (flows.home_score >= .5) | flows.P_i.eq(1),
                          ((flows.home_score >= .5) | flows.P_i.eq(1)) & flows.P_j.eq(1)) & (d <= 80)
        r = (a >= .3) & (a <= .8) & (d <= 80) & access
        np.testing.assert_allclose(trips.potential_volume, f * r * w, rtol=1e-12)
        np.testing.assert_allclose(totals["potential_volume"],
            totals["immediate_volume"] + totals["conditional_volume"], rtol=1e-12)
        np.testing.assert_allclose(totals["screened_pre_access_volume"],
            totals["potential_volume"] + totals["access_excluded_volume"], rtol=1e-12)
        for column in trips:
            flows[column] = trips[column].to_numpy()
        flows.to_parquet(directory / "paper1_flow_attributes.parquet", index=False)
        capacity["ocm_id"] = sites.ocm_id.to_numpy()
        capacity["point_count_missing"] = sites.num_points.isna().to_numpy()
        capacity.to_csv(directory / "destination_charging_capacity.csv", index=False)
        totals.update(region=region, rows=len(flows), short_volume=float(f[d <= 3].sum()))
        collected.append(totals)
        populations.extend([
            {"region": region, "stage": "generated_post_jitter_filter", "volume": route_meta["generated_car_person_volume"]},
            {"region": region, "stage": "routed_and_analysed_no_3km_cutoff", "volume": float(f.sum())},
            {"region": region, "stage": "retained_at_or_below_3km", "volume": float(f[d <= 3].sum())},
        ])
        for weighted in (True, False):
            for band, label in (((.3, .8), "AP 0.3-0.8"), ((0., 1.), "No AP screen")):
                result, _, _ = evaluate(flows, band, weighted)
                comparisons.append({"region": region, "screen": label,
                                    "purpose_weighted": weighted, **result})
        for lo, hi in ((.3, .8), (.2, .8), (.4, .8), (.3, .7), (.3, .9), (.2, .9), (.4, .7)):
            result, _, _ = evaluate(flows, (lo, hi))
            sensitivities.append({"region": region, "family": "AP band", "scenario": f"{lo:.1f}-{hi:.1f}", **result})
        purpose_pattern = flows.purpose.str.casefold().map({
            "commuting": "Commuting", "business": "Commuting", "education": "Commuting", "escort": "Commuting",
            "shopping": "Shopping", "visit hospital or other health": "Shopping",
            "other personal business": "Shopping", "other journey": "Shopping",
            "visiting friends or relatives": "Social", "eating/drinking": "Leisure",
            "sport/entertainment": "Leisure", "holiday/daytrip": "Leisure",
        })
        beta_map = beta.set_index("NPT purpose").beta
        signed_beta = purpose_pattern.map(beta_map).to_numpy()
        if not np.isfinite(signed_beta).all():
            raise ValueError("A purpose lacks a verified decay coefficient")
        groups = flows.origin.astype(str) + ":" + flows.purpose.str.casefold()
        for scale in (.5, .75, 1., 1.25, 1.5):
            alternative = flows.copy()
            multiplier = np.exp((scale - 1) * signed_beta * flows.gravity_distance_km)
            alternative["vol"] = origin_constrained_weights(f, multiplier, groups)
            result, _, _ = evaluate(alternative)
            sensitivities.append({"region": region, "family": "Decay multiplier", "scenario": str(scale), **result})
        for scale in (.75, 1., 1.25):
            alternative = flows.copy()
            alternative["w"] = np.minimum(1., w * scale)
            result, _, _ = evaluate(alternative)
            sensitivities.append({"region": region, "family": "Purpose-weight multiplier", "scenario": str(scale), **result})
        for fraction in (.25, .5, .75):
            result, _, _ = evaluate(flows, origin_fraction=fraction)
            sensitivities.append({"region": region, "family": "Medium public-origin energy share",
                                  "scenario": str(fraction), **result})
        home_mask = r & (d <= 40) & flows.home_score.ge(.5).to_numpy()
        subset = float((f * w)[home_mask].sum())
        geometry = two_stop_geometry(d[home_mask], (f * w)[home_mask])
        for exposure in (.23, .29):
            for failure in (.25, .5, .75, 1.):
                loss = exposure * failure * subset
                chains.append({"region": region, "exposure": exposure, "failure": failure,
                    "baseline_volume": float(f.sum()), "potential_volume": totals["potential_volume"],
                    "immediate_volume": totals["immediate_volume"], "home_subset_volume": subset,
                    "loss_volume": loss, "adjusted_potential_volume": totals["potential_volume"] - loss,
                    "adjusted_immediate_volume": totals["immediate_volume"] - loss, **geometry})
        summary = json.loads(record(old / f"{region}_comprehensive_summary.json").read_text())
        b, n, immediate, conditional = (totals[k] for k in (
            "baseline_volume", "potential_volume", "immediate_volume", "conditional_volume"))
        summary["analysis_timestamp"] = analysis_timestamp
        summary["recalculation"] = "v4 scores and weights fixed; corrected gravity; no 3km cutoff; N=fRW; capacity-only tiers"
        summary["area_reporting_status"] = (
            "Legacy frozen ancillary diagnostics retained; canonical-2022 OA "
            "correction and verified reporting materialization required")
        summary["data_coverage"].update(od_pairs=len(flows), routes=len(flows), charging_locations=len(sites))
        summary["adoption_analysis"]["high_adoption_areas"] = oa_rows[-1]["target_areas"]
        summary["conversion_potential"].update(total_baseline_trips=b, final_conversion_potential=n, conversion_rate_pct=100*n/b)
        summary.update(total_daily_trips=b, total_daily_trips_millions=b/1e6,
            theoretical_replaceable_trips=n, theoretical_replacement_rate_pct=100*n/b,
            immediately_feasible_trips=immediate, constrained_unlockable_trips=conditional,
            currently_infeasible_trips=b-n, additional_trips_from_charging=conditional,
            charging_constraint_trip_pct=100*conditional/b, charging_constraint_trip_trips=conditional)
        summary["data_sources"] = {"recalculated_flow_file": path_label(directory / "paper1_flow_attributes.parquet"),
            "fixed_area_diagnostics": path_label(old), "station_snapshot": "existing v4 OCM input, no new date filter"}
        summary["range_analysis"] = {"feasible_routes_pct": 100*float((d <= 40).mean()),
            "constrained_routes_pct": 100*float(((d > 40) & (d <= 80)).mean()),
            "infeasible_routes_pct": 100*float((d > 80).mean())}
        executive = summary["executive_summary"]
        executive.update(theoretical_replacement_rate=100*n/b, immediately_feasible_pct=100*immediate/n,
            constrained_unlockable_pct=100*conditional/n, range_constraint_pct=100*float(f[d > 80].sum())/b,
            charging_constraint_pct=100*totals["access_excluded_volume"]/b,
            demographic_constraint_pct=100*float(f[(a < .3) | (a > .8)].sum())/b,
            no_home_charging_pct=100*float(f[flows.home_score.lt(.5)].sum())/b,
            insufficient_capacity_pct=100*conditional/b,
            low_purpose_pct=100*float(f[w < .5].sum())/b,
            high_adoption_areas=oa_rows[-1]["target_areas"])
        # Vehicle choice is not changed by Reviewer 3's revisions. Retain its
        # regional v4 mix as an explicit allocation assumption, not as a new fit.
        summary["vehicle_mix_role"] = "fixed v4 regional allocation assumption applied to revised volumes"
        np.testing.assert_allclose(sum(executive["ev_type_breakdown"].values()), 100., atol=1e-8)
        purpose = flows.groupby("purpose", as_index=False).agg(car_trips=("vol", "sum"),
            immediately_feasible=("immediate_volume", "sum"), constrained=("conditional_volume", "sum"),
            replaceable_trips=("potential_volume", "sum"), purpose_weight=("w", "first"))
        purpose["replacement_rate_pct"] = 100*purpose.replaceable_trips/purpose.car_trips
        purpose.to_csv(directory / "replaceable_trips_by_purpose.csv", index=False)
        summary["purpose_replacement_analysis"] = {row.purpose: {"baseline_trips": row.car_trips,
            "replaceable_trips": row.replaceable_trips, "replacement_percentage": row.replacement_rate_pct,
            "purpose_weight": row.purpose_weight} for row in purpose.itertuples()}
        (directory / f"{region}_comprehensive_summary.json").write_text(json.dumps(summary, indent=2))
        e = flows.euclidean_km.to_numpy()
        ratio = d[e > 0] / e[e > 0]
        qs = np.quantile(ratio, [0, .25, .5, .75, 1])
        circuities.append(dict(region=region, n_routes=len(flows), **dict(zip(
            ("min_circuity", "q25_circuity", "median_circuity", "q75_circuity", "max_circuity"), qs))))
        categories = np.array(["feasible", "constrained", "infeasible"])
        rc = np.where(d <= 40, "feasible", np.where(d <= 80, "constrained", "infeasible"))
        ec = np.where(e <= 40, "feasible", np.where(e <= 80, "constrained", "infeasible"))
        pd.DataFrame([{"category": c, "euclidean_count": int((ec == c).sum()),
            "routed_count": int((rc == c).sum()), "euclidean_pct": 100*float((ec == c).mean()),
            "routed_pct": 100*float((rc == c).mean()), "shift": int((ec == c).sum()-(rc == c).sum()),
            "region": region, "total_routes": len(flows), "median_circuity": qs[2]} for c in categories]
            ).to_csv(directory / f"{region}_circuity_conversion_comparison.csv", index=False)
        print(f"{region}: baseline={b:,.3f}; N={n:,.3f}; capacity-sufficient={immediate:,.3f}", flush=True)
    for name, rows in (("regional_metrics", collected), ("technical_comparison", comparisons),
                       ("sensitivity", sensitivities), ("chaining", chains), ("population_ledger", populations),
                       ("circuity_summary", circuities), ("oa_counts", oa_rows)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    numeric = pd.DataFrame(collected).drop(columns="region").sum().to_dict()
    (OUT / "national_metrics.json").write_text(json.dumps(numeric, indent=2))
    manifest = {"baseline": "Revised_manuscript--with-changes-marked_R2_v4.qmd",
        "analysis_timestamp": analysis_timestamp,
        "source_sha256": source_hashes, "status": "calculated; independent verification pending", "no_3km_filter": True,
        "frozen_inputs": "v4 OA scores, home scores, purpose parameters, OCM input and vehicle assignments",
        "purpose_weight": "v4 printed W=0.35R+0.35T+0.30CF; removes legacy 0.66 fallback and undocumented temporal reblend",
        "routing": "verified 2026-09-07 corrected-gravity checkpoint; routes not regenerated in this audit",
        "gravity_distance": "pre-jitter geodist cheap-ruler metres converted to km; numeric child scaling reversed using all/(origin_trips*proportion)",
        "ancillary_area_diagnostics": "legacy frozen layers retained; five have older OA identities; canonical-2022 correction and materialize_paper1_pr42_reporting.py required before reporting",
        "outcome": "N=fRW; R=AP band AND routed range AND charging access; capacity tested separately",
        "charging_access": "d<=40: H_i OR P_i; 40<d<=80: (H_i OR P_i) AND P_j; H_i=1{HCF_i>=0.5}",
        "charging_capacity": "ChargingCapacityAnalyzer configured supply; pooled required origin and destination loads; no home-capacity check",
        "medium_public_origin_energy_fraction": 0.5,
        "charging_energy": "One-way trip energy counted once: non-home short trips at origin; home medium trips at destination; non-home medium trips split equally",
        "uncertainty": "scenario envelopes, not confidence intervals or joint uncertainty bounds"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(numeric, indent=2), flush=True)


if __name__ == "__main__":
    main()
