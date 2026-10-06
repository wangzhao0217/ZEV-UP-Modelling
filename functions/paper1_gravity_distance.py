"""Recover pre-jitter distances from the archived odjitter numeric attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def recover_gravity_distance(routes: pd.DataFrame) -> pd.DataFrame:
    """Recover exact child multiplicity and parent distance from retained rows.

    The archived odjitter binary divides every numeric column by the number
    of children k. Before jittering, all = origin_trips * proportion; after
    division, all / (origin_trips * proportion) = k. This identifies k for
    each surviving child even if sibling rows were filtered or failed routing.
    The pre-jitter distance must therefore be multiplied by k. No geographic
    nearest join or inference from the observed sibling count is needed.

    Return multiplicity, archived divided distance and recovered kilometres.
    Raise ValueError if inputs are absent, nonfinite, nonpositive where needed,
    or fail the integer/multiplicative invariants of this archived format.
    """
    columns = ("all", "origin_trips", "proportion", "length_euclidean_unjittered")
    missing = set(columns) - set(routes.columns)
    if missing:
        raise ValueError(f"Missing odjitter recovery attributes: {sorted(missing)}")
    values = routes.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :3] <= 0).any() or (values[:, 3] < 0).any():
        raise ValueError("Invalid odjitter recovery attributes")
    all_modes, origin, proportion, stored_distance = values.T
    implied = all_modes / (origin * proportion)
    integer = np.rint(implied)
    if ((integer < 1).any() or (integer > np.iinfo(np.int32).max).any()
            or not np.isclose(implied, integer, rtol=1e-10, atol=1e-9).all()):
        raise ValueError("Archived attributes do not identify an integer odjitter multiplicity")
    if not np.allclose(all_modes, origin * proportion * integer, rtol=1e-10, atol=1e-9):
        raise ValueError("Archived attributes fail the odjitter multiplicative invariant")
    return pd.DataFrame({
        "gravity_disaggregation_multiplicity": integer.astype(np.int64),
        "archived_divided_gravity_distance_km": stored_distance,
        "gravity_distance_km": stored_distance * integer,
    }, index=routes.index)
