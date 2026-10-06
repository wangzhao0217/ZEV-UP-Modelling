"""Charging-screen comparisons with an explicit candidate load for Paper 1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from functions.charging_infrastructure import (
    ChargingCapacityAnalyzer,
    calculate_charging_condition,
)
from functions.paper1_feasibility import potential_components


def charging_screen_scenario(
    flows: pd.DataFrame,
    station_points: Sequence[float],
    *,
    band: tuple[float, float] = (0.30, 0.80),
    purpose_weighted: bool = True,
    production_factors: Sequence[float] | None = None,
    require_charging_access: bool = False,
    charging_config: dict | None = None,
    medium_origin_fraction: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Recalculate charging demand and tiers for one regional scenario.

    ``flows`` supplies vol, w, ap, d, home_score, P_i, P_j, occupancy and
    origin_charging_station_position and charging_station_position (destination).
    Point counts are ordered by station position. Both required public endpoints
    compete for shared site capacity; home capacity is not separately tested.
    Purpose-weighted scenarios use W for both loading and reported volumes;
    raw scenarios use W=1. Production factors rescale volumes before computing
    the denominator and demand. All input frames remain unchanged. Returns
    totals, trip diagnostics and the station ledger. Invalid inputs raise
    ValueError, including inconsistent access and station assignments.
    With require_charging_access=True, potential is N=fRW with charging
    access included in R; conditional volume then means capacity-only loss.
    The default retains the historical pre-access diagnostic for older runs.
    """
    required = {
        "vol", "w", "ap", "d", "home_score", "P_i", "P_j",
        "occupancy", "origin_charging_station_position", "charging_station_position",
    }
    if required - set(flows.columns):
        raise ValueError(f"Missing scenario columns: {sorted(required - set(flows.columns))}")
    if not isinstance(purpose_weighted, bool):
        raise ValueError("purpose_weighted must be boolean")
    if not isinstance(require_charging_access, bool):
        raise ValueError("require_charging_access must be boolean")
    f = flows.vol.to_numpy(dtype=float).copy()
    if production_factors is not None:
        factors = np.asarray(production_factors, dtype=float)
        if (factors.shape != f.shape or not np.isfinite(factors).all()
                or (factors < 0).any()):
            raise ValueError("Production factors must be finite, nonnegative and match flows")
        f *= factors
    w = flows.w.to_numpy(dtype=float) if purpose_weighted else np.ones(len(flows))
    parts = potential_components(f, w, flows.ap, flows.d, band=band)
    denominator = float(parts.baseline.sum())
    if denominator <= 0:
        raise ValueError("Scenario denominator must be positive")
    home_score = flows.home_score.to_numpy(dtype=float)
    if (not np.isfinite(home_score).all()
            or ((home_score < 0) | (home_score > 1)).any()):
        raise ValueError("Home scores must be finite and lie in [0,1]")
    origin = flows.P_i.to_numpy()
    dest = flows.P_j.to_numpy()
    if not np.isin(origin, [0, 1]).all() or not np.isin(dest, [0, 1]).all():
        raise ValueError("Public-access indicators must be binary")
    station = flows.charging_station_position.to_numpy()
    if not np.array_equal(dest == 1, station >= 0):
        raise ValueError("Destination access and station assignments disagree")
    origin_station = flows.origin_charging_station_position.to_numpy()
    if not np.array_equal(origin == 1, origin_station >= 0):
        raise ValueError("Origin access and station assignments disagree")
    occupancy = flows.occupancy.to_numpy(dtype=float)
    if not np.isfinite(occupancy).all() or (occupancy < 1).any():
        raise ValueError("Occupancy must be finite and at least one person per vehicle")
    d = flows.d.to_numpy(dtype=float)
    home = home_score >= 0.5
    eligible = (parts.in_band & parts.range_eligible).to_numpy()
    charging = calculate_charging_condition(
        pd.Series(home), pd.Series(origin == 1), pd.Series(dest == 1), pd.Series(d),
        medium_origin_rule="home_or_public",
    ).to_numpy(dtype=bool)
    pre_capacity = eligible & charging
    trips, sites = ChargingCapacityAnalyzer(config=charging_config).assess_trip_charging_capacity(
        pd.DataFrame({
            "vehicle_volume": f * w / occupancy, "distance_km": d,
            "eligible": pre_capacity, "home_charging": home,
            "origin_station_position": origin_station,
            "destination_station_position": station,
        }),
        pd.DataFrame({"number_of_points": station_points}),
        medium_origin_fraction=medium_origin_fraction,
    )
    immediate = trips.capacity_sufficient.to_numpy()
    screened = parts.screened_volume.to_numpy()
    potential = screened * charging if require_charging_access else screened
    trips["charging_access"] = charging & (d <= 80.0)
    trips["R"] = pre_capacity
    trips["access_excluded_volume"] = screened * ~charging
    trips["immediate"] = immediate
    trips["potential_volume"] = potential
    trips["immediate_volume"] = f * w * immediate
    trips["conditional_volume"] = potential - trips.immediate_volume.to_numpy()
    totals = {
        "baseline_volume": denominator,
        "range_volume": float(parts.raw_range_volume.sum()),
        "weighted_range_volume": float(parts.weighted_range_volume.sum()),
        "screened_pre_access_volume": float(screened.sum()),
        "access_excluded_volume": float(trips.access_excluded_volume.sum()),
        "excluded_low_volume": float(parts.excluded_low_volume.sum()),
        "excluded_high_volume": float(parts.excluded_high_volume.sum()),
        "potential_volume": float(potential.sum()),
        "immediate_volume": float(trips.immediate_volume.sum()),
        "conditional_volume": float(trips.conditional_volume.sum()),
        "required_vehicle_charger_hours": float(sites.D_j.sum()),
        "required_origin_hours": float(sites.D_origin.sum()),
        "required_destination_hours": float(sites.D_destination.sum()),
        "available_point_hours": float(sites.C_avail_hours.sum()),
        "loaded_sites": float(sites.D_j.gt(0).sum()),
        "overloaded_sites": float(sites.D_j.gt(sites.C_avail_hours).sum()),
    }
    return totals, trips, sites
