"""Explicit mapping from generated Paper 1 purposes to scoring profiles."""

from __future__ import annotations

import numpy as np

from paper1_r2.purpose_types import TripPurpose


def canonical_purpose(purpose: str) -> str:
    """Resolve production labels without silently assigning a fallback profile.

    Case differences and the SHS health label are accepted. Unrecognised
    purposes raise ValueError so that a scoring mismatch cannot remain hidden.
    """
    if not isinstance(purpose, str):
        raise ValueError("Purpose must be a recognised string")
    names = {item.value.casefold(): item.value for item in TripPurpose}
    names["visit hospital or other health"] = TripPurpose.HEALTH_VISITS.value
    key = purpose.strip().casefold()
    if key not in names:
        raise ValueError(f"Unrecognised generated purpose: {purpose}")
    return names[key]


def purpose_weight(
    ratings: np.ndarray, base_weights: tuple[float, float, float] = (.35, .35, .30),
    blend: float = .60,
) -> np.ndarray:
    """Evaluate the complete configured score for arrays ending in ten ratings.

    Rating order: R, T, D, S, destination compatibility, home compatibility,
    departure flexibility, arrival flexibility, capped minimum dwell/120,
    charging urgency. The last four produce the separate temporal score.
    Arrays may include a draw dimension. Invalid inputs raise ValueError.
    """
    values = np.asarray(ratings, dtype=float)
    weights = np.asarray(base_weights, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 10 or not np.isfinite(values).all():
        raise ValueError("Ten finite ratings per purpose required")
    if ((values < 0).any() or (values > 1).any() or weights.shape != (3,)
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1) or not 0 <= blend <= 1):
        raise ValueError("Ratings and blend must be in [0,1]; weights must sum to one")
    cf = .4 * values[..., 2] + .3 * (1-values[..., 3]) + .2 * values[..., 4] + .1 * values[..., 5]
    base = weights[0] * values[..., 0] + weights[1] * values[..., 1] + weights[2] * cf
    temporal = .25 * values[..., 6] + .25 * values[..., 7] + .30 * values[..., 8] + .20 * (1-values[..., 9])
    return blend * base + (1-blend) * temporal
