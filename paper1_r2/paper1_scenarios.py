"""Charging-screen comparisons with an explicit candidate load for Paper 1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paper1_charging import destination_capacity
from .paper1_feasibility import potential_components


def charging_screen_scenario(
    flows: pd.DataFrame,
    station_points: Sequence[float],
    *,
    band: tuple[float, float] = (0.30, 0.80),
    purpose_weighted: bool = True,
    production_factors: Sequence[float] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Recalculate charging demand and tiers for one regional scenario.

    ``flows`` supplies vol, w, ap, d, home_score, P_i, P_j, occupancy and
    charging_station_position. Point counts are ordered by station position.
    Purpose-weighted scenarios use W for both loading and reported volumes;
    raw scenarios use W=1. Production factors rescale volumes before computing
    the denominator and demand. All input frames remain unchanged. Returns
    totals, trip diagnostics and the station ledger. Invalid inputs raise
    ValueError, including inconsistent access and station assignments.
    """
    required = {
        "vol", "w", "ap", "d", "home_score", "P_i", "P_j",
        "occupancy", "charging_station_position",
    }
    if required - set(flows.columns):
        raise ValueError(f"Missing scenario columns: {sorted(required - set(flows.columns))}")
    if not isinstance(purpose_weighted, bool):
        raise ValueError("purpose_weighted must be boolean")
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
    d = flows.d.to_numpy(dtype=float)
    home = home_score >= 0.5
    short = d <= 40.0
    eligible = (parts.in_band & parts.range_eligible).to_numpy()
    charging = np.where(short, home | (origin == 1), home & (dest == 1))
    pre_capacity = eligible & charging
    trips, sites = destination_capacity(
        f, w, flows.occupancy, d, pre_capacity, station, station_points,
    )
    immediate = pre_capacity & (short | trips.Cap_j.eq(1).to_numpy())
    potential = parts.screened_volume.to_numpy()
    trips["immediate"] = immediate
    trips["potential_volume"] = potential
    trips["immediate_volume"] = f * w * immediate
    trips["conditional_volume"] = potential - trips.immediate_volume.to_numpy()
    totals = {
        "baseline_volume": denominator,
        "range_volume": float(parts.raw_range_volume.sum()),
        "potential_volume": float(potential.sum()),
        "immediate_volume": float(trips.immediate_volume.sum()),
        "conditional_volume": float(trips.conditional_volume.sum()),
        "required_vehicle_charger_hours": float(sites.D_j.sum()),
        "available_point_hours": float(sites.C_avail_hours.sum()),
        "loaded_sites": float(sites.D_j.gt(0).sum()),
        "overloaded_sites": float(sites.D_j.gt(sites.C_avail_hours).sum()),
    }
    return totals, trips, sites
