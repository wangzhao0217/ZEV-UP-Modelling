"""Explicit vehicle-equivalent destination charging for the Paper 1 revision."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def nearest_charger(
    endpoint_xy: np.ndarray, charger_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return metre distances and station positions from projected coordinates.

    Both arrays must use the same metre-based CRS (EPSG:27700 in Paper 1).
    Empty networks return infinite distances and -1 positions. Invalid
    coordinates raise ValueError; no degree-to-distance approximation is used.
    """
    endpoints = np.asarray(endpoint_xy, dtype=float)
    stations = np.asarray(charger_xy, dtype=float)
    for values in (endpoints, stations):
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("Finite N-by-2 projected coordinates required")
    if len(stations) == 0:
        return np.full(len(endpoints), np.inf), np.full(len(endpoints), -1, dtype=int)
    distance, position = cKDTree(stations).query(endpoints)
    return distance, position


def destination_capacity(
    person_volume: Sequence[float], suitability: Sequence[float],
    occupancy: Sequence[float], distance_km: Sequence[float],
    pre_capacity_eligible: Sequence[bool], station_position: Sequence[int],
    station_points: Sequence[float], *, operating_hours: float = 16.0,
    background_utilisation: float = 0.45, hub_premium: float = 0.2,
    hub_scale: float = 5.0, kwh_per_km: float = 0.065,
    charger_kw: float = 7.0, short_km: float = 40.0,
    range_km: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool weighted vehicle demand once per nearest accessible charging site.

    Inputs are same-length trip arrays and a unique-site vector of point counts.
    Eligibility is evaluated before capacity; the caller assigns -1 when the
    destination is outside the access radius. Demand includes every eligible
    medium trip, without discounting required destination charging for home
    access. Point-hours reserve an assumed background utilisation. Returns
    trip diagnostics and a station ledger; malformed inputs raise ValueError.
    """
    arrays = [np.asarray(x, dtype=float) for x in (
        person_volume, suitability, occupancy, distance_km,
        pre_capacity_eligible, station_position)]
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Finite one-dimensional trip inputs required")
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Trip inputs must have equal length")
    f, w, occ, d, gate, station = arrays
    points = np.asarray(station_points, dtype=float)
    if points.ndim != 1 or not np.isfinite(points).all() or (points < 0).any():
        raise ValueError("Finite nonnegative station point counts required")
    if (points != np.floor(points)).any():
        raise ValueError("Station point counts must be integers")
    if ((f < 0).any() or (w < 0).any() or (w > 1).any()
            or (occ < 1).any() or (d < 0).any() or not np.isin(gate, [0, 1]).all()):
        raise ValueError("Invalid demand, weight, occupancy, distance or gate")
    if ((station != np.floor(station)).any() or (station < -1).any()
            or (station >= len(points)).any()):
        raise ValueError("Invalid station position")
    settings = [operating_hours, background_utilisation, hub_premium,
                hub_scale, kwh_per_km, charger_kw, short_km, range_km]
    if (not np.isfinite(settings).all() or min(operating_hours, hub_scale,
            kwh_per_km, charger_kw, short_km) <= 0 or range_km < short_km
            or not 0 <= background_utilisation < 1 or hub_premium < 0
            or background_utilisation * (1 + hub_premium) >= 1):
        raise ValueError("Invalid charging scenario settings")
    station = station.astype(int)
    medium = (d > short_km) & (d <= range_km) & (gate == 1)
    if (medium & (station < 0)).any():
        raise ValueError("Eligible medium trip lacks an accessible station")
    vehicle_volume = f * w / occ
    charge_hours = d * kwh_per_km / charger_kw
    demand = vehicle_volume * charge_hours * medium
    site_demand = np.bincount(station[medium], weights=demand[medium], minlength=len(points))
    extra = np.maximum(points - 1, 0)
    utilisation = background_utilisation * (1 + hub_premium * extra / (extra + hub_scale))
    available = operating_hours * points * (1 - utilisation)
    sufficient = (site_demand <= available) & (points > 0)
    capacity = np.zeros(len(f), dtype=int)
    assigned = station >= 0
    capacity[assigned] = sufficient[station[assigned]].astype(int)
    trips = pd.DataFrame({"vehicle_equivalent_volume": vehicle_volume,
                          "t_ij": charge_hours, "demand_contrib": demand,
                          "Cap_j": capacity})
    sites = pd.DataFrame({"station_position": np.arange(len(points)),
                         "N_points": points, "D_j": site_demand,
                         "background_utilisation": utilisation,
                         "C_avail_hours": available, "Cap_j": sufficient.astype(int)})
    return trips, sites
