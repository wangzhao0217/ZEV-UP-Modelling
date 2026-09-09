"""Conditional chaining diagnostics for the screened home-charging subset.

These are assumption scenarios, not estimates of observed tours. The rest of
the headline numerator is held fixed, so the resulting interval is not a bound
on all national tour behaviour.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from paper1_r2.feasibility import _vector, potential_components


def chaining_loss_budget(
    volume: Sequence[float], suitability: Sequence[float], ap: Sequence[float],
    distance_km: Sequence[float], home_score: Sequence[float],
    exposure: float, failure: float, band: tuple[float, float] = (.3, .8),
    range_km: float = 80., home_threshold: float = .5,
) -> dict[str, float]:
    """Subtract exposure × failure only from weighted home round-trip volume.

    A trip is in the affected subset when its AP is in the specified band,
    its one-way distance is at most half the effective range, and its home
    charging score meets the threshold. All supplied volume remains in the
    denominator. Raises ValueError for invalid scores or scenario parameters.
    """
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (exposure, failure, home_threshold)):
        raise ValueError("exposure, failure and home threshold must lie in [0, 1]")
    parts = potential_components(volume, suitability, ap, distance_km, band, range_km)
    home = _vector(home_score, "home_score")
    if len(home) != len(parts) or ((home < 0) | (home > 1)).any():
        raise ValueError("invalid home score")
    denominator = float(parts.baseline.sum())
    if denominator <= 0:
        raise ValueError("baseline demand must be positive")
    affected = (home >= home_threshold) & (np.asarray(distance_km) <= range_km / 2)
    base = float(parts.screened_volume.sum())
    subset = float(parts.loc[affected, "screened_volume"].sum())
    loss = subset * exposure * failure
    return {"baseline_volume": denominator, "screened_volume": base,
            "home_subset_volume": subset, "exposure": exposure, "failure": failure,
            "loss_volume": loss, "screened_share": base / denominator,
            "home_subset_share": subset / denominator,
            "adjusted_share": (base - loss) / denominator,
            "conditional_subset_floor": (base - subset) / denominator}


def two_stop_geometry(
    distance_km: Sequence[float], weight: Sequence[float], range_km: float = 80.,
) -> dict[str, float]:
    """Return two-stop failure fractions for idealised symmetric radial routes.

    Partners are independent weighted draws from the supplied home-feasible
    distances. Collinear tours have length 2*max(d1,d2); opposed tours have
    length 2*(d1+d2). These radial geometries do not identify real road tours.
    """
    d, w = _vector(distance_km, "distance_km"), _vector(weight, "weight")
    if not np.isfinite(range_km) or range_km <= 0 or len(d) != len(w):
        raise ValueError("invalid range or input lengths")
    if (d < 0).any() or (d > range_km / 2).any() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("distances must be home-feasible and weights nonnegative with positive total")
    order = np.argsort(d)
    d, w = d[order], w[order] / w.sum()
    cumulative = np.concatenate(([0.], np.cumsum(w)))
    # Exactly 80 km remains feasible; side='right' includes equality.
    partner_index = np.searchsorted(d, range_km / 2 - d, side="right")
    opposed = float(np.dot(w, 1 - cumulative[partner_index]))
    return {"collinear_failure": 0., "opposed_failure": float(np.clip(opposed, 0, 1))}
