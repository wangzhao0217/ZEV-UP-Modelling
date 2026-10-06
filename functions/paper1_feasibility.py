"""Pure calculations for Paper 1's screened, purpose-weighted potential.

Charging and destination capacity partition range-eligible screened volume.
Suitability weights are scenario weights, not estimated adoption probabilities.
All inputs refer to the same flows; no file loading or fallback data are used.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    """Return a finite one-dimensional numeric vector or raise ValueError."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return result


def potential_components(
    volume: Sequence[float],
    suitability: Sequence[float],
    ap: Sequence[float],
    distance_km: Sequence[float],
    band: tuple[float, float] = (0.30, 0.80),
    range_km: float = 80.0,
) -> pd.DataFrame:
    """Return per-flow volumes before and after the inclusive demographic gate.

    Inputs must have equal length, nonnegative demand/distance, and scores in
    [0, 1]. Raises ValueError for invalid inputs or thresholds. The denominator
    remains the complete supplied volume, including out-of-band/range flows.
    """
    vectors = [_vector(v, n) for v, n in zip(
        (volume, suitability, ap, distance_km),
        ("volume", "suitability", "ap", "distance_km"),
    )]
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("flow inputs must have equal length")
    f, w, a, d = vectors
    if (f < 0).any() or (d < 0).any():
        raise ValueError("volume and distance must be nonnegative")
    if ((w < 0) | (w > 1)).any() or ((a < 0) | (a > 1)).any():
        raise ValueError("suitability and ap must lie in [0, 1]")
    lo, hi = band
    if not 0 <= lo <= hi <= 1 or not np.isfinite(range_km) or range_km <= 0:
        raise ValueError("invalid band or effective range")
    eligible = d <= range_km
    screened = (a >= lo) & (a <= hi)
    weighted_range = f * w * eligible
    return pd.DataFrame({
        "baseline": f,
        "raw_range_volume": f * eligible,
        "weighted_range_volume": weighted_range,
        "screened_volume": weighted_range * screened,
        "excluded_low_volume": weighted_range * (a < lo),
        "excluded_high_volume": weighted_range * (a > hi),
        "range_eligible": eligible,
        "in_band": screened,
    })


def classify_feasibility(
    ap: Sequence[float],
    distance_km: Sequence[float],
    charging_condition: Sequence[float],
    destination_capacity: Sequence[float],
    band: tuple[float, float] = (0.30, 0.80),
    short_km: float = 40.0,
    range_km: float = 80.0,
) -> np.ndarray:
    """Partition screened range-eligible flows by charging and capacity.

    Charging condition must encode H_i OR P_i for short trips and H_i AND
    P_j for medium trips. Capacity is additionally required for medium trips.
    Returns legacy-compatible class labels; raises ValueError on invalid data.
    """
    a = _vector(ap, "ap")
    parts = potential_components(np.ones(len(a)), np.ones(len(a)), a,
                                 distance_km, band, range_km)
    c = _vector(charging_condition, "charging_condition")
    cap = _vector(destination_capacity, "destination_capacity")
    if len(c) != len(a) or len(cap) != len(a):
        raise ValueError("flow inputs must have equal length")
    if not np.isin(c, [0, 1]).all() or not np.isin(cap, [0, 1]).all():
        raise ValueError("charging and capacity indicators must be binary")
    if not np.isfinite(short_km) or not 0 < short_km <= range_km:
        raise ValueError("invalid short-trip distance")
    eligible = (parts["range_eligible"] & parts["in_band"]).to_numpy()
    immediate = eligible & (c == 1) & (
        (np.asarray(distance_km) <= short_km) | (cap == 1)
    )
    return np.where(immediate, "immediately_feasible",
                    np.where(eligible, "constrained", "infeasible"))


def summarise_potential(
    components: pd.DataFrame,
    categories: Sequence[str] | None = None,
) -> dict[str, float]:
    """Aggregate component shares, optionally checking a saved tier partition.

    Categories must represent the same band/range scenario as components.
    Raises ValueError for zero demand, unknown categories, or disagreement.
    """
    denominator = float(components["baseline"].sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("baseline demand must be positive")
    result = {"baseline_volume": denominator, "rows": len(components)}
    for col in ("raw_range_volume", "weighted_range_volume", "screened_volume",
                "excluded_low_volume", "excluded_high_volume"):
        result[col] = float(components[col].sum())
        result[col.replace("_volume", "_share")] = result[col] / denominator
    if categories is not None:
        cat = np.asarray(categories)
        allowed = ["fully_feasible", "immediately_feasible", "constrained", "infeasible"]
        if len(cat) != len(components) or not np.isin(cat, allowed).all():
            raise ValueError("invalid saved categories")
        eligible = (components["in_band"] & components["range_eligible"]).to_numpy()
        if not np.array_equal(cat != "infeasible", eligible):
            raise ValueError("saved category rule disagrees with the specified outcome")
        weighted = components["screened_volume"].to_numpy()
        for label, mask in (
            ("immediate", np.isin(cat, ["fully_feasible", "immediately_feasible"])),
            ("conditional", cat == "constrained"),
        ):
            result[f"{label}_volume"] = float(weighted[mask].sum())
            result[f"{label}_share"] = result[f"{label}_volume"] / denominator
    return result


def origin_constrained_weights(
    volume: Sequence[float],
    multiplier: Sequence[float],
    groups: Sequence[str],
) -> np.ndarray:
    """Reweight fixed retained flows while preserving each supplied group's mass.

    This is a fixed-support diagnostic, not regeneration of sampled OD pairs.
    Groups should identify region, original production zone, and purpose.
    Raises ValueError for missing groups or invalid/zero multiplier mass.
    """
    v, m = _vector(volume, "volume"), _vector(multiplier, "multiplier")
    g = pd.Series(groups)
    if len(v) != len(m) or len(g) != len(v) or g.isna().any():
        raise ValueError("invalid reweighting groups or lengths")
    if (v < 0).any() or (m <= 0).any():
        raise ValueError("volume must be nonnegative and multipliers positive")
    frame = pd.DataFrame({"v": v, "vm": v * m, "group": g.to_numpy()})
    totals = frame.groupby("group", sort=False)[["v", "vm"]].transform("sum")
    scales = np.divide(totals["v"].to_numpy(), totals["vm"].to_numpy(),
                       out=np.zeros(len(v)), where=totals["vm"].to_numpy() > 0)
    return v * m * scales
